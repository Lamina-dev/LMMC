/**
 * @file itersolve_minres_lsqr.c
 * @brief MINRES and LSQR iterative solver implementations with matrix-free support.
 */
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/itersolve.h"

/* Forward declarations of helpers from itersolve.c */
static int lmmc_is_finite_number_ml(lmmc_real_t v) {
    return isfinite(v) ? 1 : 0;
}

static lmmc_status_t vec_norm2_checked_ml(const lmmc_vec_t* v, lmmc_real_t* out_norm) {
    if (v == NULL || out_norm == NULL || v->data == NULL || v->size == 0)
        return LMMC_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < v->size; ++i) {
        if (!lmmc_is_finite_number_ml(v->data[i]))
            return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    lmmc_status_t st = lmmc_vec_norm2(v, out_norm);
    if (st != LMMC_STATUS_OK) return st;
    if (!lmmc_is_finite_number_ml(*out_norm))
        return LMMC_STATUS_NUMERICAL_FAILURE;
    return LMMC_STATUS_OK;
}

static lmmc_status_t vec_dot_checked_ml(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t* out_dot) {
    lmmc_status_t st = lmmc_vec_dot(a, b, out_dot);
    if (st != LMMC_STATUS_OK) return st;
    if (!lmmc_is_finite_number_ml(*out_dot))
        return LMMC_STATUS_NUMERICAL_FAILURE;
    return LMMC_STATUS_OK;
}

static lmmc_status_t apply_precond_ml(
    const lmmc_precond_t* precond, const lmmc_vec_t* rhs, lmmc_vec_t* out
) {
    if (precond == NULL) return lmmc_vec_copy(rhs, out);
    return lmmc_precond_apply(precond, rhs, out);
}

static void do_log_ml(const lmmc_itersolve_config_t* cfg, size_t iter, lmmc_real_t rn) {
    if (cfg->log_cb != NULL)
        cfg->log_cb(iter, rn, cfg->log_user_data);
    else if (cfg->verbose)
        printf("Iteration %zu: residual norm = %.10e\n", iter, rn);
}

/* Matrix-free dispatch: compute y = A*x */
static lmmc_status_t matvec_dispatch(
    const lmmc_sparse_mat_t* a,
    const lmmc_itersolve_config_t* cfg,
    const lmmc_vec_t* x_in,
    lmmc_vec_t* y_out
) {
    if (cfg->apply_op != NULL)
        return cfg->apply_op(x_in, y_out, cfg->op_user_data);
    return lmmc_sparse_mat_vec_mul(a, x_in, y_out);
}

/* Compute y = A^T * x for CSR sparse matrix */
static lmmc_status_t matvec_transpose_csr(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* x_in,
    lmmc_vec_t* y_out
) {
    size_t i, j;
    for (i = 0; i < y_out->size; ++i)
        y_out->data[i] = 0.0;
    for (i = 0; i < a->rows; ++i) {
        size_t start = a->row_ptr[i];
        size_t end = a->row_ptr[i + 1];
        for (j = start; j < end; ++j) {
            size_t col = a->col_idx[j];
            y_out->data[col] += a->values[j] * x_in->data[i];
        }
    }
    return LMMC_STATUS_OK;
}

/* Compute y = A^T * x for CSC sparse matrix */
static lmmc_status_t matvec_transpose_csc(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* x_in,
    lmmc_vec_t* y_out
) {
    size_t i, j;
    for (j = 0; j < y_out->size; ++j)
        y_out->data[j] = 0.0;
    /* CSC: row_ptr is col_ptr, col_idx is row_idx */
    for (j = 0; j < a->cols; ++j) {
        size_t start = a->row_ptr[j];
        size_t end = a->row_ptr[j + 1];
        double acc = 0.0;
        for (i = start; i < end; ++i) {
            size_t row = a->col_idx[i];
            acc += a->values[i] * x_in->data[row];
        }
        y_out->data[j] = acc;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t matvec_transpose_dispatch(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* x_in,
    lmmc_vec_t* y_out
) {
    if (a == NULL)
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->format == LMMC_SPARSE_CSR)
        return matvec_transpose_csr(a, x_in, y_out);
    return matvec_transpose_csc(a, x_in, y_out);
}

/* ===================== MINRES ===================== */

lmmc_status_t lmmc_minres_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
) {
    lmmc_itersolve_config_t local_cfg;
    lmmc_status_t st = LMMC_STATUS_OK;
    size_t n, iter_count = 0;
    int converged = 0;
    double norm_r = 0.0, norm_b_val = 0.0, threshold = 0.0;

    /* Lanczos vectors and work vectors */
    lmmc_vec_t v_prev = {0}, v_curr = {0}, v_next = {0};
    lmmc_vec_t w_prev = {0}, w_curr = {0};
    lmmc_vec_t av = {0}, z_vec = {0};

    /* MINRES scalars */
    double alpha_k, beta_k, beta_kp1;
    double cs_prev, sn_prev, cs_curr, sn_curr;
    double phi_bar, epsilon_curr, delta_bar, gamma_val, phi_val;

    /* Validate inputs */
    if (b == NULL || x == NULL || b->data == NULL || x->data == NULL ||
        b->size == 0 || x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (cfg != NULL)
        local_cfg = *cfg;
    else {
        st = lmmc_itersolve_default_config(b->size, &local_cfg);
        if (st != LMMC_STATUS_OK) return st;
    }

    /* Dispatch validation */
    if (local_cfg.apply_op != NULL && a != NULL)
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (local_cfg.apply_op == NULL && a == NULL)
        return LMMC_STATUS_INVALID_ARGUMENT;

    /* Dimension checks */
    if (a != NULL) {
        if (a->rows != a->cols || b->size != a->rows || x->size != a->cols)
            return LMMC_STATUS_DIMENSION_MISMATCH;
    } else {
        if (b->size != x->size)
            return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    if (precond != NULL && precond->size != b->size)
        return LMMC_STATUS_DIMENSION_MISMATCH;
    if (local_cfg.max_iter == 0)
        return LMMC_STATUS_INVALID_ARGUMENT;

    n = b->size;

    if (out_result != NULL) {
        out_result->converged = 0;
        out_result->num_iter = 0;
        out_result->initial_residual_norm = 0.0;
        out_result->final_residual_norm = 0.0;
    }

    /* Allocate work vectors */
    st = lmmc_vec_create(n, &v_prev); if (st != LMMC_STATUS_OK) goto minres_end;
    st = lmmc_vec_create(n, &v_curr); if (st != LMMC_STATUS_OK) goto minres_end;
    st = lmmc_vec_create(n, &v_next); if (st != LMMC_STATUS_OK) goto minres_end;
    st = lmmc_vec_create(n, &w_prev); if (st != LMMC_STATUS_OK) goto minres_end;
    st = lmmc_vec_create(n, &w_curr); if (st != LMMC_STATUS_OK) goto minres_end;
    st = lmmc_vec_create(n, &av); if (st != LMMC_STATUS_OK) goto minres_end;
    st = lmmc_vec_create(n, &z_vec); if (st != LMMC_STATUS_OK) goto minres_end;

    /* Compute initial residual: r = b - A*x */
    st = matvec_dispatch(a, &local_cfg, x, &av);
    if (st != LMMC_STATUS_OK) goto minres_end;
    {
        size_t i;
        for (i = 0; i < n; ++i)
            v_curr.data[i] = b->data[i] - av.data[i];
    }

    /* Apply preconditioner */
    st = apply_precond_ml(precond, &v_curr, &z_vec);
    if (st != LMMC_STATUS_OK) goto minres_end;

    /* beta_1 = sqrt(r^T * z) */
    {
        double dot_rz;
        st = vec_dot_checked_ml(&v_curr, &z_vec, &dot_rz);
        if (st != LMMC_STATUS_OK) goto minres_end;
        if (dot_rz < 0.0) dot_rz = -dot_rz;
        beta_k = sqrt(dot_rz);
    }

    st = vec_norm2_checked_ml(b, &norm_b_val);
    if (st != LMMC_STATUS_OK) goto minres_end;

    norm_r = beta_k;
    phi_bar = beta_k;
    threshold = local_cfg.abs_tol + local_cfg.rel_tol * norm_b_val;

    if (out_result != NULL) {
        out_result->initial_residual_norm = norm_r;
        out_result->final_residual_norm = norm_r;
    }
    do_log_ml(&local_cfg, 0, norm_r);

    if (norm_r <= threshold) {
        converged = 1;
        goto minres_end;
    }
    if (beta_k <= 1e-30) {
        converged = 1;
        goto minres_end;
    }

    /* Normalize v_curr = z / beta_k, zero out others */
    {
        size_t i;
        for (i = 0; i < n; ++i) {
            v_curr.data[i] = z_vec.data[i] / beta_k;
            v_prev.data[i] = 0.0;
            w_prev.data[i] = 0.0;
            w_curr.data[i] = 0.0;
        }
    }

    /* Initialize Givens rotation state */
    cs_prev = 1.0; sn_prev = 0.0;
    cs_curr = 1.0; sn_curr = 0.0;
    epsilon_curr = 0.0;

    /* Main MINRES iteration */
    {
        size_t iter;
        for (iter = 0; iter < local_cfg.max_iter; ++iter) {
            size_t i;
            double cs_new, sn_new;

            /* Lanczos step: av = A * v_curr */
            st = matvec_dispatch(a, &local_cfg, &v_curr, &av);
            if (st != LMMC_STATUS_OK) goto minres_end;

            /* Apply preconditioner */
            st = apply_precond_ml(precond, &av, &z_vec);
            if (st != LMMC_STATUS_OK) goto minres_end;

            /* alpha_k = v_curr^T * z_vec */
            st = vec_dot_checked_ml(&v_curr, &z_vec, &alpha_k);
            if (st != LMMC_STATUS_OK) goto minres_end;

            /* v_next = z_vec - alpha_k * v_curr - beta_k * v_prev */
            for (i = 0; i < n; ++i)
                v_next.data[i] = z_vec.data[i] - alpha_k * v_curr.data[i] - beta_k * v_prev.data[i];

            /* beta_{k+1} = sqrt(v_next^T * v_next) */
            {
                double dot_vv;
                st = vec_dot_checked_ml(&v_next, &v_next, &dot_vv);
                if (st != LMMC_STATUS_OK) goto minres_end;
                if (dot_vv < 0.0) dot_vv = -dot_vv;
                beta_kp1 = sqrt(dot_vv);
            }

            /* Apply previous Givens rotation to get delta_bar */
            epsilon_curr = sn_prev * beta_k;
            delta_bar = cs_curr * alpha_k - sn_curr * epsilon_curr;

            /* Compute new Givens rotation to eliminate beta_{k+1} */
            gamma_val = sqrt(delta_bar * delta_bar + beta_kp1 * beta_kp1);

            if (gamma_val <= 1e-30) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto minres_end;
            }

            cs_new = delta_bar / gamma_val;
            sn_new = beta_kp1 / gamma_val;

            /* phi = cs_new * phi_bar */
            phi_val = cs_new * phi_bar;
            /* phi_bar = -sn_new * phi_bar */
            phi_bar = -sn_new * phi_bar;

            /* Update w: w_new = (v_curr - epsilon_curr*w_prev - delta_bar*w_curr) / gamma */
            for (i = 0; i < n; ++i)
                z_vec.data[i] = (v_curr.data[i] - epsilon_curr * w_prev.data[i] - delta_bar * w_curr.data[i]) / gamma_val;

            /* Update solution: x = x + phi * w_new */
            for (i = 0; i < n; ++i) {
                x->data[i] += phi_val * z_vec.data[i];
                if (!lmmc_is_finite_number_ml(x->data[i])) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto minres_end;
                }
            }

            /* Shift w vectors */
            for (i = 0; i < n; ++i) {
                w_prev.data[i] = w_curr.data[i];
                w_curr.data[i] = z_vec.data[i];
            }

            /* Update rotations */
            sn_prev = sn_curr;
            cs_prev = cs_curr;
            sn_curr = sn_new;
            cs_curr = cs_new;

            /* Residual norm estimate = |phi_bar| */
            norm_r = fabs(phi_bar);
            iter_count = iter + 1;
            if (out_result != NULL)
                out_result->final_residual_norm = norm_r;
            do_log_ml(&local_cfg, iter_count, norm_r);

            if (norm_r <= threshold) { converged = 1; break; }

            /* Normalize v_next and shift Lanczos vectors */
            if (beta_kp1 <= 1e-30) break; /* Lucky breakdown */

            for (i = 0; i < n; ++i) {
                v_prev.data[i] = v_curr.data[i];
                v_curr.data[i] = v_next.data[i] / beta_kp1;
            }
            beta_k = beta_kp1;
        }
    }

minres_end:
    if (out_result != NULL) {
        out_result->converged = converged;
        out_result->num_iter = iter_count;
        out_result->final_residual_norm = norm_r;
    }
    lmmc_vec_destroy(&z_vec);
    lmmc_vec_destroy(&av);
    lmmc_vec_destroy(&w_curr);
    lmmc_vec_destroy(&w_prev);
    lmmc_vec_destroy(&v_next);
    lmmc_vec_destroy(&v_curr);
    lmmc_vec_destroy(&v_prev);
    return st;
}

/* ===================== LSQR ===================== */

lmmc_status_t lmmc_lsqr_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
) {
    lmmc_itersolve_config_t local_cfg;
    lmmc_status_t st = LMMC_STATUS_OK;
    size_t m, n_cols, iter_count = 0;
    int converged = 0;
    double norm_r = 0.0, norm_b_val = 0.0, threshold = 0.0;
    double alpha_l, beta_l, rho_bar, phi_bar_l;
    double rho_val, cs_l, sn_l, theta_l, phi_l;

    lmmc_vec_t u = {0}, v = {0}, w = {0}, tmp_vec = {0}, av_tmp = {0};

    /* Validate inputs */
    if (b == NULL || x == NULL || b->data == NULL || x->data == NULL ||
        b->size == 0 || x->size == 0)
        return LMMC_STATUS_INVALID_ARGUMENT;

    if (cfg != NULL)
        local_cfg = *cfg;
    else {
        st = lmmc_itersolve_default_config(b->size, &local_cfg);
        if (st != LMMC_STATUS_OK) return st;
    }

    /* Dispatch validation */
    if (local_cfg.apply_op != NULL && a != NULL)
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (local_cfg.apply_op == NULL && a == NULL)
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (local_cfg.max_iter == 0)
        return LMMC_STATUS_INVALID_ARGUMENT;

    /* Determine dimensions */
    if (a != NULL) {
        m = a->rows;
        n_cols = a->cols;
        if (b->size != m || x->size != n_cols)
            return LMMC_STATUS_DIMENSION_MISMATCH;
    } else {
        m = b->size;
        n_cols = x->size;
    }

    if (out_result != NULL) {
        out_result->converged = 0;
        out_result->num_iter = 0;
        out_result->initial_residual_norm = 0.0;
        out_result->final_residual_norm = 0.0;
    }

    /* Allocate work vectors */
    st = lmmc_vec_create(m, &u); if (st != LMMC_STATUS_OK) goto lsqr_end;
    st = lmmc_vec_create(n_cols, &v); if (st != LMMC_STATUS_OK) goto lsqr_end;
    st = lmmc_vec_create(n_cols, &w); if (st != LMMC_STATUS_OK) goto lsqr_end;
    st = lmmc_vec_create(n_cols, &tmp_vec); if (st != LMMC_STATUS_OK) goto lsqr_end;
    st = lmmc_vec_create(m, &av_tmp); if (st != LMMC_STATUS_OK) goto lsqr_end;

    /* Initialize: u = b - A*x */
    st = matvec_dispatch(a, &local_cfg, x, &av_tmp);
    if (st != LMMC_STATUS_OK) goto lsqr_end;
    {
        size_t i;
        for (i = 0; i < m; ++i)
            u.data[i] = b->data[i] - av_tmp.data[i];
    }

    /* beta = ||u|| */
    st = vec_norm2_checked_ml(&u, &beta_l);
    if (st != LMMC_STATUS_OK) goto lsqr_end;

    st = vec_norm2_checked_ml(b, &norm_b_val);
    if (st != LMMC_STATUS_OK) goto lsqr_end;

    norm_r = beta_l;
    threshold = local_cfg.abs_tol + local_cfg.rel_tol * norm_b_val;

    if (out_result != NULL) {
        out_result->initial_residual_norm = norm_r;
        out_result->final_residual_norm = norm_r;
    }
    do_log_ml(&local_cfg, 0, norm_r);

    if (norm_r <= threshold) { converged = 1; goto lsqr_end; }
    if (beta_l <= 1e-30) { converged = 1; goto lsqr_end; }

    /* Normalize u */
    {
        size_t i;
        for (i = 0; i < m; ++i)
            u.data[i] /= beta_l;
    }

    /* v = A^T * u */
    st = matvec_transpose_dispatch(a, &u, &v);
    if (st != LMMC_STATUS_OK) goto lsqr_end;

    /* alpha = ||v|| */
    st = vec_norm2_checked_ml(&v, &alpha_l);
    if (st != LMMC_STATUS_OK) goto lsqr_end;

    if (alpha_l <= 1e-30) { converged = 1; goto lsqr_end; }

    /* Normalize v */
    {
        size_t i;
        for (i = 0; i < n_cols; ++i)
            v.data[i] /= alpha_l;
    }

    /* w = v */
    st = lmmc_vec_copy(&v, &w);
    if (st != LMMC_STATUS_OK) goto lsqr_end;

    /* Initialize */
    rho_bar = alpha_l;
    phi_bar_l = beta_l;

    /* Main LSQR iteration */
    {
        size_t iter;
        for (iter = 0; iter < local_cfg.max_iter; ++iter) {
            size_t i;
            double scale;

            /* Bidiagonalization: u = A*v - alpha*u */
            st = matvec_dispatch(a, &local_cfg, &v, &av_tmp);
            if (st != LMMC_STATUS_OK) goto lsqr_end;
            for (i = 0; i < m; ++i)
                u.data[i] = av_tmp.data[i] - alpha_l * u.data[i];

            /* beta = ||u|| */
            st = vec_norm2_checked_ml(&u, &beta_l);
            if (st != LMMC_STATUS_OK) goto lsqr_end;

            if (beta_l <= 1e-30) {
                converged = 1;
                iter_count = iter + 1;
                break;
            }

            /* Normalize u */
            for (i = 0; i < m; ++i)
                u.data[i] /= beta_l;

            /* v = A^T*u - beta*v */
            st = matvec_transpose_dispatch(a, &u, &tmp_vec);
            if (st != LMMC_STATUS_OK) goto lsqr_end;
            for (i = 0; i < n_cols; ++i)
                v.data[i] = tmp_vec.data[i] - beta_l * v.data[i];

            /* alpha = ||v|| */
            st = vec_norm2_checked_ml(&v, &alpha_l);
            if (st != LMMC_STATUS_OK) goto lsqr_end;

            if (alpha_l <= 1e-30) {
                converged = 1;
                iter_count = iter + 1;
                break;
            }

            /* Normalize v */
            for (i = 0; i < n_cols; ++i)
                v.data[i] /= alpha_l;

            /* Givens rotation */
            rho_val = sqrt(rho_bar * rho_bar + beta_l * beta_l);
            if (rho_val <= 1e-30) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto lsqr_end;
            }

            cs_l = rho_bar / rho_val;
            sn_l = beta_l / rho_val;
            theta_l = sn_l * alpha_l;
            rho_bar = -cs_l * alpha_l;
            phi_l = cs_l * phi_bar_l;
            phi_bar_l = sn_l * phi_bar_l;

            /* Update x: x += (phi/rho) * w */
            scale = phi_l / rho_val;
            for (i = 0; i < n_cols; ++i) {
                x->data[i] += scale * w.data[i];
                if (!lmmc_is_finite_number_ml(x->data[i])) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto lsqr_end;
                }
            }

            /* Update w: w = v - (theta/rho) * w */
            scale = theta_l / rho_val;
            for (i = 0; i < n_cols; ++i)
                w.data[i] = v.data[i] - scale * w.data[i];

            /* Residual norm estimate */
            norm_r = fabs(phi_bar_l);
            iter_count = iter + 1;
            if (out_result != NULL)
                out_result->final_residual_norm = norm_r;
            do_log_ml(&local_cfg, iter_count, norm_r);

            if (norm_r <= threshold) { converged = 1; break; }
        }
    }

lsqr_end:
    if (out_result != NULL) {
        out_result->converged = converged;
        out_result->num_iter = iter_count;
        out_result->final_residual_norm = norm_r;
    }
    lmmc_vec_destroy(&av_tmp);
    lmmc_vec_destroy(&tmp_vec);
    lmmc_vec_destroy(&w);
    lmmc_vec_destroy(&v);
    lmmc_vec_destroy(&u);
    return st;
}
