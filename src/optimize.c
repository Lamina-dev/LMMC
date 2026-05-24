/**
 * @file optimize.c
 * @brief 优化与非线性方程组求解实现。
 *
 * 包含 Newton、Broyden、L-BFGS、Levenberg-Marquardt、梯度下降算法。
 */
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/linear_algebra.h"
#include "lmmc/optimize.h"

/* ===================== 内部辅助函数 ===================== */

/**
 * @brief 计算向量的 L2 范数。
 */
static lmmc_real_t vec_norm2(const lmmc_vec_t* v) {
    lmmc_real_t sum = 0.0;
    size_t i;
    for (i = 0; i < v->size; ++i) {
        sum += v->data[i] * v->data[i];
    }
    return sqrt(sum);
}

/**
 * @brief 使用前向有限差分计算 Jacobian。
 *
 * J[:,j] ≈ (F(x + h*e_j) - F(x)) / h
 */
static lmmc_status_t finite_difference_jacobian(
    lmmc_opt_func_t F, void* user_data,
    const lmmc_vec_t* x, const lmmc_vec_t* Fx,
    lmmc_mat_t* J)
{
    size_t n = x->size;
    size_t j, i;
    lmmc_vec_t x_pert;
    lmmc_vec_t F_pert;
    lmmc_status_t status;
    lmmc_real_t h;

    status = lmmc_vec_create(n, &x_pert);
    if (status != LMMC_STATUS_OK) return status;

    status = lmmc_vec_create(n, &F_pert);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x_pert);
        return status;
    }

    for (j = 0; j < n; ++j) {
        /* Copy x into x_pert */
        memcpy(x_pert.data, x->data, n * sizeof(lmmc_real_t));

        /* Perturbation step: h = sqrt(eps) * max(|x_j|, 1) */
        h = sqrt(1.4901161193847656e-08) * fmax(fabs(x->data[j]), 1.0);

        x_pert.data[j] += h;

        status = F(&x_pert, &F_pert, user_data);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(&x_pert);
            lmmc_vec_destroy(&F_pert);
            return status;
        }

        /* J[:,j] = (F_pert - Fx) / h */
        for (i = 0; i < n; ++i) {
            J->data[i * J->stride + j] = (F_pert.data[i] - Fx->data[i]) / h;
        }
    }

    lmmc_vec_destroy(&x_pert);
    lmmc_vec_destroy(&F_pert);
    return LMMC_STATUS_OK;
}

/* ===================== 公共函数实现 ===================== */

lmmc_status_t lmmc_optimize_default_config(lmmc_optimize_config_t* cfg) {
    if (cfg == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    cfg->abs_tol = 1e-12;
    cfg->rel_tol = 1e-10;
    cfg->max_iter = 1000;
    cfg->lbfgs_memory = 10;
    cfg->lm_damping = 1e-3;
    cfg->verbose = 0;
    return LMMC_STATUS_OK;
}

/* ===================== Newton 法 ===================== */

lmmc_status_t lmmc_nleq_newton(
    lmmc_opt_func_t F,
    lmmc_opt_jac_t J,
    void* user_data,
    lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out)
{
    size_t n;
    size_t iter;
    lmmc_vec_t Fx, delta;
    lmmc_mat_t Jmat, Jlu;
    size_t* pivots = NULL;
    lmmc_status_t status;
    lmmc_real_t res_norm;

    if (F == NULL || x == NULL || cfg == NULL || out == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = x->size;
    if (n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    out->converged = 0;
    out->num_iter = 0;
    out->final_residual = 0.0;
    out->failure_reason = LMMC_OPT_FAILURE_NONE;

    /* Allocate working vectors and matrices */
    status = lmmc_vec_create(n, &Fx);
    if (status != LMMC_STATUS_OK) return status;

    status = lmmc_vec_create(n, &delta);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&Fx);
        return status;
    }

    status = lmmc_mat_create(n, n, &Jmat);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&Fx);
        lmmc_vec_destroy(&delta);
        return status;
    }

    status = lmmc_mat_create(n, n, &Jlu);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&Fx);
        lmmc_vec_destroy(&delta);
        lmmc_mat_destroy(&Jmat);
        return status;
    }

    pivots = (size_t*)lmmc_alloc(n * sizeof(size_t));
    if (pivots == NULL) {
        lmmc_vec_destroy(&Fx);
        lmmc_vec_destroy(&delta);
        lmmc_mat_destroy(&Jmat);
        lmmc_mat_destroy(&Jlu);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (iter = 0; iter < cfg->max_iter; ++iter) {
        /* Evaluate F(x) */
        status = F(x, &Fx, user_data);
        if (status != LMMC_STATUS_OK) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto cleanup;
        }

        /* Check convergence */
        res_norm = vec_norm2(&Fx);
        out->final_residual = res_norm;
        out->num_iter = iter + 1;

        if (!isfinite(res_norm)) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto cleanup;
        }

        if (res_norm <= cfg->abs_tol) {
            out->converged = 1;
            goto cleanup;
        }

        /* Compute Jacobian */
        if (J != NULL) {
            status = J(x, &Jmat, user_data);
        } else {
            status = finite_difference_jacobian(F, user_data, x, &Fx, &Jmat);
        }
        if (status != LMMC_STATUS_OK) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto cleanup;
        }

        /* Copy Jmat to Jlu for LU decomposition (in-place) */
        memcpy(Jlu.data, Jmat.data, n * n * sizeof(lmmc_real_t));

        /* LU decompose */
        status = lmmc_lu_decompose_inplace(&Jlu, pivots, NULL);
        if (status == LMMC_STATUS_SINGULAR_MATRIX) {
            out->failure_reason = LMMC_OPT_FAILURE_SINGULAR_JACOBIAN;
            goto cleanup;
        }
        if (status != LMMC_STATUS_OK) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto cleanup;
        }

        /* Solve J * delta = -F(x) → set rhs = -Fx, solve for delta */
        {
            size_t i;
            lmmc_vec_t neg_Fx;
            lmmc_vec_create(n, &neg_Fx);
            for (i = 0; i < n; ++i) {
                neg_Fx.data[i] = -Fx.data[i];
            }
            status = lmmc_lu_solve(&Jlu, pivots, &neg_Fx, &delta);
            lmmc_vec_destroy(&neg_Fx);
        }
        if (status != LMMC_STATUS_OK) {
            out->failure_reason = LMMC_OPT_FAILURE_SINGULAR_JACOBIAN;
            goto cleanup;
        }

        /* Update x = x + delta */
        {
            size_t i;
            for (i = 0; i < n; ++i) {
                x->data[i] += delta.data[i];
            }
        }

        if (cfg->verbose) {
            fprintf(stderr, "Newton iter %zu: ||F|| = %.6e\n", iter + 1, res_norm);
        }
    }

    /* Max iterations reached */
    out->failure_reason = LMMC_OPT_FAILURE_MAX_ITER;

cleanup:
    lmmc_vec_destroy(&Fx);
    lmmc_vec_destroy(&delta);
    lmmc_mat_destroy(&Jmat);
    lmmc_mat_destroy(&Jlu);
    lmmc_free(pivots);
    return LMMC_STATUS_OK;
}

/* ===================== Broyden 法 ===================== */

lmmc_status_t lmmc_nleq_broyden(
    lmmc_opt_func_t F,
    void* user_data,
    lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out)
{
    size_t n;
    size_t iter;
    lmmc_vec_t Fx, Fx_new, delta_x, delta_F, Bdx, temp_vec;
    lmmc_mat_t B, Blu;
    size_t* pivots = NULL;
    lmmc_status_t status;
    lmmc_real_t res_norm;

    if (F == NULL || x == NULL || cfg == NULL || out == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = x->size;
    if (n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    out->converged = 0;
    out->num_iter = 0;
    out->final_residual = 0.0;
    out->failure_reason = LMMC_OPT_FAILURE_NONE;

    /* Allocate working storage */
    status = lmmc_vec_create(n, &Fx);
    if (status != LMMC_STATUS_OK) return status;

    status = lmmc_vec_create(n, &Fx_new);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&Fx); return status; }

    status = lmmc_vec_create(n, &delta_x);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&Fx); lmmc_vec_destroy(&Fx_new); return status; }

    status = lmmc_vec_create(n, &delta_F);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&Fx); lmmc_vec_destroy(&Fx_new); lmmc_vec_destroy(&delta_x); return status; }

    status = lmmc_vec_create(n, &Bdx);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&Fx); lmmc_vec_destroy(&Fx_new); lmmc_vec_destroy(&delta_x); lmmc_vec_destroy(&delta_F); return status; }

    status = lmmc_vec_create(n, &temp_vec);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&Fx); lmmc_vec_destroy(&Fx_new); lmmc_vec_destroy(&delta_x); lmmc_vec_destroy(&delta_F); lmmc_vec_destroy(&Bdx); return status; }

    status = lmmc_mat_create(n, n, &B);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&Fx); lmmc_vec_destroy(&Fx_new); lmmc_vec_destroy(&delta_x); lmmc_vec_destroy(&delta_F); lmmc_vec_destroy(&Bdx); lmmc_vec_destroy(&temp_vec); return status; }

    status = lmmc_mat_create(n, n, &Blu);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&Fx); lmmc_vec_destroy(&Fx_new); lmmc_vec_destroy(&delta_x); lmmc_vec_destroy(&delta_F); lmmc_vec_destroy(&Bdx); lmmc_vec_destroy(&temp_vec); lmmc_mat_destroy(&B); return status; }

    pivots = (size_t*)lmmc_alloc(n * sizeof(size_t));
    if (pivots == NULL) {
        lmmc_vec_destroy(&Fx); lmmc_vec_destroy(&Fx_new); lmmc_vec_destroy(&delta_x);
        lmmc_vec_destroy(&delta_F); lmmc_vec_destroy(&Bdx); lmmc_vec_destroy(&temp_vec);
        lmmc_mat_destroy(&B); lmmc_mat_destroy(&Blu);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* Evaluate initial F(x) */
    status = F(x, &Fx, user_data);
    if (status != LMMC_STATUS_OK) {
        out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
        goto broyden_cleanup;
    }

    /* Initialize B with finite-difference Jacobian */
    status = finite_difference_jacobian(F, user_data, x, &Fx, &B);
    if (status != LMMC_STATUS_OK) {
        out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
        goto broyden_cleanup;
    }

    for (iter = 0; iter < cfg->max_iter; ++iter) {
        res_norm = vec_norm2(&Fx);
        out->final_residual = res_norm;
        out->num_iter = iter + 1;

        if (!isfinite(res_norm)) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto broyden_cleanup;
        }

        if (res_norm <= cfg->abs_tol) {
            out->converged = 1;
            goto broyden_cleanup;
        }

        /* Solve B * delta_x = -Fx via LU */
        memcpy(Blu.data, B.data, n * n * sizeof(lmmc_real_t));
        status = lmmc_lu_decompose_inplace(&Blu, pivots, NULL);
        if (status == LMMC_STATUS_SINGULAR_MATRIX) {
            out->failure_reason = LMMC_OPT_FAILURE_SINGULAR_JACOBIAN;
            goto broyden_cleanup;
        }
        if (status != LMMC_STATUS_OK) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto broyden_cleanup;
        }

        {
            size_t i;
            for (i = 0; i < n; ++i) {
                temp_vec.data[i] = -Fx.data[i];
            }
        }
        status = lmmc_lu_solve(&Blu, pivots, &temp_vec, &delta_x);
        if (status != LMMC_STATUS_OK) {
            out->failure_reason = LMMC_OPT_FAILURE_SINGULAR_JACOBIAN;
            goto broyden_cleanup;
        }

        /* Update x */
        {
            size_t i;
            for (i = 0; i < n; ++i) {
                x->data[i] += delta_x.data[i];
            }
        }

        /* Evaluate F at new x */
        status = F(x, &Fx_new, user_data);
        if (status != LMMC_STATUS_OK) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto broyden_cleanup;
        }

        /* Compute delta_F = Fx_new - Fx */
        {
            size_t i;
            for (i = 0; i < n; ++i) {
                delta_F.data[i] = Fx_new.data[i] - Fx.data[i];
            }
        }

        /* Broyden rank-one update: B += (delta_F - B*delta_x) * delta_x^T / (delta_x^T * delta_x) */
        {
            size_t i, j;
            lmmc_real_t dxTdx = 0.0;

            /* Compute B * delta_x */
            for (i = 0; i < n; ++i) {
                Bdx.data[i] = 0.0;
                for (j = 0; j < n; ++j) {
                    Bdx.data[i] += B.data[i * B.stride + j] * delta_x.data[j];
                }
            }

            /* Compute delta_x^T * delta_x */
            for (i = 0; i < n; ++i) {
                dxTdx += delta_x.data[i] * delta_x.data[i];
            }

            if (dxTdx > 1e-300) {
                /* Update B: B += (delta_F - Bdx) * delta_x^T / dxTdx */
                for (i = 0; i < n; ++i) {
                    lmmc_real_t num_i = delta_F.data[i] - Bdx.data[i];
                    for (j = 0; j < n; ++j) {
                        B.data[i * B.stride + j] += num_i * delta_x.data[j] / dxTdx;
                    }
                }
            }
        }

        /* Update Fx for next iteration */
        memcpy(Fx.data, Fx_new.data, n * sizeof(lmmc_real_t));

        if (cfg->verbose) {
            fprintf(stderr, "Broyden iter %zu: ||F|| = %.6e\n", iter + 1, res_norm);
        }
    }

    /* Max iterations reached */
    out->failure_reason = LMMC_OPT_FAILURE_MAX_ITER;

broyden_cleanup:
    lmmc_vec_destroy(&Fx);
    lmmc_vec_destroy(&Fx_new);
    lmmc_vec_destroy(&delta_x);
    lmmc_vec_destroy(&delta_F);
    lmmc_vec_destroy(&Bdx);
    lmmc_vec_destroy(&temp_vec);
    lmmc_mat_destroy(&B);
    lmmc_mat_destroy(&Blu);
    lmmc_free(pivots);
    return LMMC_STATUS_OK;
}


/* ===================== L-BFGS ===================== */

lmmc_status_t lmmc_minimize_lbfgs(
    lmmc_opt_obj_t obj,
    lmmc_opt_grad_t grad,
    void* user_data,
    lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out)
{
    size_t n, m;
    size_t iter;
    size_t k, bound, i, j;
    lmmc_real_t* s_store = NULL;  /* m*n: s_i = x_{i+1} - x_i */
    lmmc_real_t* y_store = NULL;  /* m*n: y_i = g_{i+1} - g_i */
    lmmc_real_t* alpha = NULL;    /* m */
    lmmc_real_t* rho = NULL;      /* m */
    lmmc_vec_t g, g_prev, q, x_new, g_new;
    lmmc_status_t status;
    lmmc_real_t f_val, f_new;
    lmmc_real_t grad_norm;
    int history_count = 0;
    int oldest = 0;

    if (obj == NULL || grad == NULL || x == NULL || cfg == NULL || out == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = x->size;
    m = cfg->lbfgs_memory;
    if (n == 0 || m == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    out->converged = 0;
    out->num_iter = 0;
    out->final_residual = 0.0;
    out->failure_reason = LMMC_OPT_FAILURE_NONE;

    /* Allocate storage */
    s_store = (lmmc_real_t*)lmmc_alloc(m * n * sizeof(lmmc_real_t));
    y_store = (lmmc_real_t*)lmmc_alloc(m * n * sizeof(lmmc_real_t));
    alpha = (lmmc_real_t*)lmmc_alloc(m * sizeof(lmmc_real_t));
    rho = (lmmc_real_t*)lmmc_alloc(m * sizeof(lmmc_real_t));

    if (!s_store || !y_store || !alpha || !rho) {
        if (s_store) lmmc_free(s_store);
        if (y_store) lmmc_free(y_store);
        if (alpha) lmmc_free(alpha);
        if (rho) lmmc_free(rho);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    status = lmmc_vec_create(n, &g);
    if (status != LMMC_STATUS_OK) goto lbfgs_alloc_fail;

    status = lmmc_vec_create(n, &g_prev);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&g); goto lbfgs_alloc_fail; }

    status = lmmc_vec_create(n, &q);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&g); lmmc_vec_destroy(&g_prev); goto lbfgs_alloc_fail; }

    status = lmmc_vec_create(n, &x_new);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&g); lmmc_vec_destroy(&g_prev); lmmc_vec_destroy(&q); goto lbfgs_alloc_fail; }

    status = lmmc_vec_create(n, &g_new);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&g); lmmc_vec_destroy(&g_prev); lmmc_vec_destroy(&q); lmmc_vec_destroy(&x_new); goto lbfgs_alloc_fail; }

    /* Compute initial gradient */
    status = grad(x, &g, user_data);
    if (status != LMMC_STATUS_OK) {
        out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
        goto lbfgs_cleanup;
    }

    f_val = obj(x, user_data);

    for (iter = 0; iter < cfg->max_iter; ++iter) {
        grad_norm = vec_norm2(&g);
        out->final_residual = grad_norm;
        out->num_iter = iter + 1;

        if (!isfinite(grad_norm) || !isfinite(f_val)) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto lbfgs_cleanup;
        }

        if (grad_norm <= cfg->abs_tol) {
            out->converged = 1;
            goto lbfgs_cleanup;
        }

        /* Two-loop recursion to compute search direction q = -H*g */
        memcpy(q.data, g.data, n * sizeof(lmmc_real_t));

        bound = (size_t)history_count;

        /* First loop (backward) */
        for (k = 0; k < bound; ++k) {
            size_t idx = (oldest + (int)history_count - 1 - (int)k) % m;
            lmmc_real_t dot = 0.0;
            for (i = 0; i < n; ++i) {
                dot += s_store[idx * n + i] * q.data[i];
            }
            alpha[k] = rho[idx] * dot;
            for (i = 0; i < n; ++i) {
                q.data[i] -= alpha[k] * y_store[idx * n + i];
            }
        }

        /* Scale q by gamma = s^T y / y^T y (most recent pair) */
        if (history_count > 0) {
            size_t newest_idx = (oldest + history_count - 1) % m;
            lmmc_real_t sy = 0.0, yy = 0.0;
            for (i = 0; i < n; ++i) {
                sy += s_store[newest_idx * n + i] * y_store[newest_idx * n + i];
                yy += y_store[newest_idx * n + i] * y_store[newest_idx * n + i];
            }
            if (yy > 1e-300) {
                lmmc_real_t gamma = sy / yy;
                for (i = 0; i < n; ++i) {
                    q.data[i] *= gamma;
                }
            }
        }

        /* Second loop (forward) */
        for (k = bound; k-- > 0;) {
            size_t idx = (oldest + (int)history_count - 1 - (int)k) % m;
            lmmc_real_t dot = 0.0;
            for (i = 0; i < n; ++i) {
                dot += y_store[idx * n + i] * q.data[i];
            }
            lmmc_real_t beta = rho[idx] * dot;
            for (i = 0; i < n; ++i) {
                q.data[i] += (alpha[k] - beta) * s_store[idx * n + i];
            }
        }

        /* q is now H*g; negate to get search direction d = -H*g */
        for (i = 0; i < n; ++i) {
            q.data[i] = -q.data[i];
        }

        /* Backtracking Armijo line search */
        {
            lmmc_real_t step = 1.0;
            lmmc_real_t c1 = 1e-4;
            lmmc_real_t dg = 0.0; /* directional derivative g^T * d */
            int ls_iter;

            for (i = 0; i < n; ++i) {
                dg += g.data[i] * q.data[i];
            }

            /* If not a descent direction, reset to steepest descent */
            if (dg >= 0.0) {
                for (i = 0; i < n; ++i) {
                    q.data[i] = -g.data[i];
                }
                dg = -grad_norm * grad_norm;
            }

            for (ls_iter = 0; ls_iter < 40; ++ls_iter) {
                for (i = 0; i < n; ++i) {
                    x_new.data[i] = x->data[i] + step * q.data[i];
                }
                f_new = obj(&x_new, user_data);
                if (f_new <= f_val + c1 * step * dg) {
                    break;
                }
                step *= 0.5;
            }

            if (ls_iter >= 40) {
                out->failure_reason = LMMC_OPT_FAILURE_LINE_SEARCH_FAILED;
                goto lbfgs_cleanup;
            }

            /* Compute new gradient */
            status = grad(&x_new, &g_new, user_data);
            if (status != LMMC_STATUS_OK) {
                out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
                goto lbfgs_cleanup;
            }

            /* Store s and y */
            {
                size_t store_idx;
                lmmc_real_t sy_val = 0.0;

                if (history_count < (int)m) {
                    store_idx = (oldest + history_count) % m;
                    history_count++;
                } else {
                    store_idx = oldest;
                    oldest = (oldest + 1) % (int)m;
                }

                for (i = 0; i < n; ++i) {
                    s_store[store_idx * n + i] = x_new.data[i] - x->data[i];
                    y_store[store_idx * n + i] = g_new.data[i] - g.data[i];
                    sy_val += s_store[store_idx * n + i] * y_store[store_idx * n + i];
                }

                if (fabs(sy_val) > 1e-300) {
                    rho[store_idx] = 1.0 / sy_val;
                } else {
                    rho[store_idx] = 0.0;
                }
            }

            /* Update x, f, g */
            memcpy(x->data, x_new.data, n * sizeof(lmmc_real_t));
            f_val = f_new;
            memcpy(g.data, g_new.data, n * sizeof(lmmc_real_t));
        }

        if (cfg->verbose) {
            fprintf(stderr, "L-BFGS iter %zu: f = %.6e, ||g|| = %.6e\n", iter + 1, f_val, grad_norm);
        }
    }

    out->failure_reason = LMMC_OPT_FAILURE_MAX_ITER;

lbfgs_cleanup:
    lmmc_vec_destroy(&g);
    lmmc_vec_destroy(&g_prev);
    lmmc_vec_destroy(&q);
    lmmc_vec_destroy(&x_new);
    lmmc_vec_destroy(&g_new);
    lmmc_free(s_store);
    lmmc_free(y_store);
    lmmc_free(alpha);
    lmmc_free(rho);
    return LMMC_STATUS_OK;

lbfgs_alloc_fail:
    lmmc_free(s_store);
    lmmc_free(y_store);
    lmmc_free(alpha);
    lmmc_free(rho);
    return LMMC_STATUS_ALLOCATION_FAILED;
}


/* ===================== Levenberg-Marquardt ===================== */

lmmc_status_t lmmc_minimize_levenberg_marquardt(
    lmmc_opt_func_t residual,
    lmmc_opt_jac_t J,
    void* user_data,
    lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out)
{
    size_t n;
    size_t iter;
    lmmc_vec_t r, r_new, delta, JtR, x_new;
    lmmc_mat_t Jmat, JtJ;
    size_t* pivots = NULL;
    lmmc_status_t status;
    lmmc_real_t lambda;
    lmmc_real_t res_norm, res_norm_new;

    if (residual == NULL || x == NULL || cfg == NULL || out == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = x->size;
    if (n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    out->converged = 0;
    out->num_iter = 0;
    out->final_residual = 0.0;
    out->failure_reason = LMMC_OPT_FAILURE_NONE;

    lambda = cfg->lm_damping;

    /* Allocate working storage */
    status = lmmc_vec_create(n, &r);
    if (status != LMMC_STATUS_OK) return status;

    status = lmmc_vec_create(n, &r_new);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&r); return status; }

    status = lmmc_vec_create(n, &delta);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&r); lmmc_vec_destroy(&r_new); return status; }

    status = lmmc_vec_create(n, &JtR);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&r); lmmc_vec_destroy(&r_new); lmmc_vec_destroy(&delta); return status; }

    status = lmmc_vec_create(n, &x_new);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&r); lmmc_vec_destroy(&r_new); lmmc_vec_destroy(&delta); lmmc_vec_destroy(&JtR); return status; }

    status = lmmc_mat_create(n, n, &Jmat);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&r); lmmc_vec_destroy(&r_new); lmmc_vec_destroy(&delta); lmmc_vec_destroy(&JtR); lmmc_vec_destroy(&x_new); return status; }

    status = lmmc_mat_create(n, n, &JtJ);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&r); lmmc_vec_destroy(&r_new); lmmc_vec_destroy(&delta); lmmc_vec_destroy(&JtR); lmmc_vec_destroy(&x_new); lmmc_mat_destroy(&Jmat); return status; }

    pivots = (size_t*)lmmc_alloc(n * sizeof(size_t));
    if (pivots == NULL) {
        lmmc_vec_destroy(&r); lmmc_vec_destroy(&r_new); lmmc_vec_destroy(&delta);
        lmmc_vec_destroy(&JtR); lmmc_vec_destroy(&x_new);
        lmmc_mat_destroy(&Jmat); lmmc_mat_destroy(&JtJ);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* Evaluate initial residual */
    status = residual(x, &r, user_data);
    if (status != LMMC_STATUS_OK) {
        out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
        goto lm_cleanup;
    }
    res_norm = vec_norm2(&r);

    for (iter = 0; iter < cfg->max_iter; ++iter) {
        out->final_residual = res_norm;
        out->num_iter = iter + 1;

        if (!isfinite(res_norm)) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto lm_cleanup;
        }

        if (res_norm <= cfg->abs_tol) {
            out->converged = 1;
            goto lm_cleanup;
        }

        /* Compute Jacobian */
        if (J != NULL) {
            status = J(x, &Jmat, user_data);
        } else {
            status = finite_difference_jacobian(residual, user_data, x, &r, &Jmat);
        }
        if (status != LMMC_STATUS_OK) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto lm_cleanup;
        }

        /* Compute J^T * J and J^T * r */
        {
            size_t i, j2, k2;
            for (i = 0; i < n; ++i) {
                JtR.data[i] = 0.0;
                for (k2 = 0; k2 < n; ++k2) {
                    JtR.data[i] += Jmat.data[k2 * Jmat.stride + i] * r.data[k2];
                }
            }
            for (i = 0; i < n; ++i) {
                for (j2 = 0; j2 < n; ++j2) {
                    lmmc_real_t sum = 0.0;
                    for (k2 = 0; k2 < n; ++k2) {
                        sum += Jmat.data[k2 * Jmat.stride + i] * Jmat.data[k2 * Jmat.stride + j2];
                    }
                    JtJ.data[i * JtJ.stride + j2] = sum;
                }
            }
        }

        /* Try solving (J^T J + lambda*I) delta = -J^T r with adaptive lambda */
        {
            int accepted = 0;
            int lm_tries;

            for (lm_tries = 0; lm_tries < 20 && !accepted; ++lm_tries) {
                size_t i, j2;
                lmmc_mat_t A_aug;

                /* Copy JtJ and add lambda*I */
                status = lmmc_mat_create(n, n, &A_aug);
                if (status != LMMC_STATUS_OK) {
                    out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
                    goto lm_cleanup;
                }
                memcpy(A_aug.data, JtJ.data, n * n * sizeof(lmmc_real_t));
                for (i = 0; i < n; ++i) {
                    A_aug.data[i * A_aug.stride + i] += lambda;
                }

                /* Solve for delta */
                status = lmmc_lu_decompose_inplace(&A_aug, pivots, NULL);
                if (status == LMMC_STATUS_SINGULAR_MATRIX) {
                    lmmc_mat_destroy(&A_aug);
                    lambda *= 10.0;
                    continue;
                }
                if (status != LMMC_STATUS_OK) {
                    lmmc_mat_destroy(&A_aug);
                    out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
                    goto lm_cleanup;
                }

                /* rhs = -J^T r */
                {
                    lmmc_vec_t neg_JtR;
                    lmmc_vec_create(n, &neg_JtR);
                    for (i = 0; i < n; ++i) {
                        neg_JtR.data[i] = -JtR.data[i];
                    }
                    status = lmmc_lu_solve(&A_aug, pivots, &neg_JtR, &delta);
                    lmmc_vec_destroy(&neg_JtR);
                }
                lmmc_mat_destroy(&A_aug);

                if (status != LMMC_STATUS_OK) {
                    lambda *= 10.0;
                    continue;
                }

                /* Trial step: x_new = x + delta */
                for (i = 0; i < n; ++i) {
                    x_new.data[i] = x->data[i] + delta.data[i];
                }

                /* Evaluate residual at x_new */
                status = residual(&x_new, &r_new, user_data);
                if (status != LMMC_STATUS_OK) {
                    lambda *= 10.0;
                    continue;
                }

                res_norm_new = vec_norm2(&r_new);

                if (res_norm_new < res_norm) {
                    /* Accept step */
                    memcpy(x->data, x_new.data, n * sizeof(lmmc_real_t));
                    memcpy(r.data, r_new.data, n * sizeof(lmmc_real_t));
                    res_norm = res_norm_new;
                    lambda *= 0.1;
                    if (lambda < 1e-15) lambda = 1e-15;
                    accepted = 1;
                } else {
                    lambda *= 10.0;
                    if (lambda > 1e15) {
                        out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
                        goto lm_cleanup;
                    }
                }
            }

            if (!accepted) {
                out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
                goto lm_cleanup;
            }
        }

        if (cfg->verbose) {
            fprintf(stderr, "LM iter %zu: ||r|| = %.6e, lambda = %.6e\n", iter + 1, res_norm, lambda);
        }
    }

    out->failure_reason = LMMC_OPT_FAILURE_MAX_ITER;

lm_cleanup:
    lmmc_vec_destroy(&r);
    lmmc_vec_destroy(&r_new);
    lmmc_vec_destroy(&delta);
    lmmc_vec_destroy(&JtR);
    lmmc_vec_destroy(&x_new);
    lmmc_mat_destroy(&Jmat);
    lmmc_mat_destroy(&JtJ);
    lmmc_free(pivots);
    return LMMC_STATUS_OK;
}

/* ===================== 梯度下降 ===================== */

lmmc_status_t lmmc_minimize_gradient_descent(
    lmmc_opt_obj_t obj,
    lmmc_opt_grad_t grad,
    void* user_data,
    lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out)
{
    size_t n;
    size_t iter;
    lmmc_vec_t g, x_new;
    lmmc_status_t status;
    lmmc_real_t f_val, f_new;
    lmmc_real_t grad_norm;

    if (obj == NULL || grad == NULL || x == NULL || cfg == NULL || out == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = x->size;
    if (n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    out->converged = 0;
    out->num_iter = 0;
    out->final_residual = 0.0;
    out->failure_reason = LMMC_OPT_FAILURE_NONE;

    status = lmmc_vec_create(n, &g);
    if (status != LMMC_STATUS_OK) return status;

    status = lmmc_vec_create(n, &x_new);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&g); return status; }

    f_val = obj(x, user_data);

    for (iter = 0; iter < cfg->max_iter; ++iter) {
        /* Compute gradient */
        status = grad(x, &g, user_data);
        if (status != LMMC_STATUS_OK) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto gd_cleanup;
        }

        grad_norm = vec_norm2(&g);
        out->final_residual = grad_norm;
        out->num_iter = iter + 1;

        if (!isfinite(grad_norm) || !isfinite(f_val)) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            goto gd_cleanup;
        }

        if (grad_norm <= cfg->abs_tol) {
            out->converged = 1;
            goto gd_cleanup;
        }

        /* Backtracking Armijo line search */
        {
            lmmc_real_t step = 1.0;
            lmmc_real_t c1 = 1e-4;
            lmmc_real_t dg = -grad_norm * grad_norm; /* g^T * (-g) = -||g||^2 */
            int ls_iter;
            size_t i;

            for (ls_iter = 0; ls_iter < 50; ++ls_iter) {
                for (i = 0; i < n; ++i) {
                    x_new.data[i] = x->data[i] - step * g.data[i];
                }
                f_new = obj(&x_new, user_data);
                if (f_new <= f_val + c1 * step * dg) {
                    break;
                }
                step *= 0.5;
            }

            if (ls_iter >= 50) {
                out->failure_reason = LMMC_OPT_FAILURE_LINE_SEARCH_FAILED;
                goto gd_cleanup;
            }

            /* Accept step */
            memcpy(x->data, x_new.data, n * sizeof(lmmc_real_t));
            f_val = f_new;
        }

        if (cfg->verbose) {
            fprintf(stderr, "GD iter %zu: f = %.6e, ||g|| = %.6e\n", iter + 1, f_val, grad_norm);
        }
    }

    out->failure_reason = LMMC_OPT_FAILURE_MAX_ITER;

gd_cleanup:
    lmmc_vec_destroy(&g);
    lmmc_vec_destroy(&x_new);
    return LMMC_STATUS_OK;
}
