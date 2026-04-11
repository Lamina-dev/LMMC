#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "lmmc/itersolve.h"

static int lmmc_is_finite_number(double v) {
    return isfinite(v) ? 1 : 0;
}

static lmmc_status_t lmmc_vec_copy(const lmmc_vec_t* src, lmmc_vec_t* dst) {
    if (src == NULL || dst == NULL || src->data == NULL || dst->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (src->size != dst->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    memcpy(dst->data, src->data, src->size * sizeof(double));
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_norm2_checked(const lmmc_vec_t* v, double* out_norm) {
    size_t i = 0;
    double sum = 0.0;

    if (v == NULL || out_norm == NULL || v->data == NULL || v->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < v->size; ++i) {
        double x = v->data[i];
        if (!lmmc_is_finite_number(x)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        sum += x * x;
        if (!lmmc_is_finite_number(sum)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    *out_norm = sqrt(sum);
    if (!lmmc_is_finite_number(*out_norm)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_dot_checked(const lmmc_vec_t* a, const lmmc_vec_t* b, double* out_dot) {
    lmmc_status_t st = lmmc_vec_dot(a, b, out_dot);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (!lmmc_is_finite_number(*out_dot)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_apply_precond_or_identity(
    const lmmc_precond_t* precond,
    const lmmc_vec_t* rhs,
    lmmc_vec_t* out
) {
    if (precond == NULL) {
        return lmmc_vec_copy(rhs, out);
    }
    return lmmc_precond_apply(precond, rhs, out);
}

static lmmc_status_t lmmc_gmres_back_substitute(
    const double* h,
    size_t ld,
    const double* rhs,
    size_t dim,
    double* out_y
) {
    size_t ii = 0;

    if (h == NULL || rhs == NULL || out_y == NULL || ld == 0 || dim == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (ii = dim; ii > 0; --ii) {
        size_t i = ii - 1;
        size_t j = 0;
        double sum = rhs[i];
        double diag = 0.0;

        for (j = i + 1; j < dim; ++j) {
            sum -= h[i * ld + j] * out_y[j];
        }

        diag = h[i * ld + i];
        if (!lmmc_is_finite_number(sum) || !lmmc_is_finite_number(diag) || fabs(diag) <= 1e-30) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        out_y[i] = sum / diag;
        if (!lmmc_is_finite_number(out_y[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_itersolve_default_config(size_t problem_size, lmmc_itersolve_config_t* out_cfg) {
    size_t max_iter = 0;

    if (out_cfg == NULL || problem_size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (problem_size > ((size_t)-1) / 20) {
        max_iter = 1000;
    } else {
        max_iter = problem_size * 20;
        if (max_iter < 100) {
            max_iter = 100;
        }
    }

    out_cfg->abs_tol = LMMC_DEFAULT_ABS_TOL;
    out_cfg->rel_tol = 1e-8;
    out_cfg->max_iter = max_iter;
    out_cfg->restart = (problem_size < 30) ? problem_size : 30;
    out_cfg->verbose = 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_cg_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
) {
    lmmc_itersolve_config_t local_cfg = {0};
    lmmc_vec_t r = {0};
    lmmc_vec_t z = {0};
    lmmc_vec_t p = {0};
    lmmc_vec_t ap = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    double norm_b = 0.0;
    double norm_r = 0.0;
    double threshold = 0.0;
    double rho = 0.0;
    size_t iter_count = 0;
    int converged = 0;

    if (a == NULL || b == NULL || x == NULL || a->row_ptr == NULL || b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows == 0 || a->cols == 0 || b->size == 0 || x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols || b->size != a->rows || x->size != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    if (cfg != NULL) {
        local_cfg = *cfg;
    } else {
        st = lmmc_itersolve_default_config(b->size, &local_cfg);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }

    if (!lmmc_is_finite_number(local_cfg.abs_tol) || !lmmc_is_finite_number(local_cfg.rel_tol) ||
        local_cfg.abs_tol < 0.0 || local_cfg.rel_tol < 0.0 || local_cfg.max_iter == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (precond != NULL && precond->size != b->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    if (out_result != NULL) {
        out_result->converged = 0;
        out_result->num_iter = 0;
        out_result->initial_residual_norm = 0.0;
        out_result->final_residual_norm = 0.0;
    }

    st = lmmc_vec_create(b->size, &r);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(b->size, &z);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(b->size, &p);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(b->size, &ap);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    st = lmmc_sparse_mat_vec_mul(a, x, &ap);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    {
        size_t i = 0;
        for (i = 0; i < b->size; ++i) {
            r.data[i] = b->data[i] - ap.data[i];
        }
    }

    st = lmmc_vec_norm2_checked(b, &norm_b);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    st = lmmc_vec_norm2_checked(&r, &norm_r);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    threshold = local_cfg.abs_tol + local_cfg.rel_tol * norm_b;
    if (!lmmc_is_finite_number(threshold)) {
        st = LMMC_STATUS_NUMERICAL_FAILURE;
        goto cleanup;
    }

    if (out_result != NULL) {
        out_result->initial_residual_norm = norm_r;
        out_result->final_residual_norm = norm_r;
    }

    if (norm_r <= threshold) {
        converged = 1;
        iter_count = 0;
        goto cleanup;
    }

    st = lmmc_apply_precond_or_identity(precond, &r, &z);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    st = lmmc_vec_dot_checked(&r, &z, &rho);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    if (fabs(rho) <= 1e-30) {
        st = LMMC_STATUS_NUMERICAL_FAILURE;
        goto cleanup;
    }

    st = lmmc_vec_copy(&z, &p);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    {
        size_t iter = 0;
        for (iter = 0; iter < local_cfg.max_iter; ++iter) {
            double denom = 0.0;
            double alpha = 0.0;
            double rho_new = 0.0;
            double beta = 0.0;
            size_t i = 0;

            st = lmmc_sparse_mat_vec_mul(a, &p, &ap);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            st = lmmc_vec_dot_checked(&p, &ap, &denom);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }
            if (fabs(denom) <= 1e-30) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            alpha = rho / denom;
            if (!lmmc_is_finite_number(alpha)) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            for (i = 0; i < x->size; ++i) {
                x->data[i] += alpha * p.data[i];
                if (!lmmc_is_finite_number(x->data[i])) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }
            }

            for (i = 0; i < r.size; ++i) {
                r.data[i] -= alpha * ap.data[i];
            }

            st = lmmc_vec_norm2_checked(&r, &norm_r);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            iter_count = iter + 1;
            if (out_result != NULL) {
                out_result->final_residual_norm = norm_r;
            }

            if (norm_r <= threshold) {
                converged = 1;
                break;
            }

            st = lmmc_apply_precond_or_identity(precond, &r, &z);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            st = lmmc_vec_dot_checked(&r, &z, &rho_new);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }
            if (fabs(rho) <= 1e-30) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            beta = rho_new / rho;
            if (!lmmc_is_finite_number(beta)) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            for (i = 0; i < p.size; ++i) {
                p.data[i] = z.data[i] + beta * p.data[i];
                if (!lmmc_is_finite_number(p.data[i])) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }
            }

            rho = rho_new;
            if (fabs(rho) <= 1e-30) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }
        }
    }

cleanup:
    if (out_result != NULL) {
        out_result->converged = converged;
        out_result->num_iter = iter_count;
        if (lmmc_is_finite_number(norm_r)) {
            out_result->final_residual_norm = norm_r;
        }
    }

    lmmc_vec_destroy(&ap);
    lmmc_vec_destroy(&p);
    lmmc_vec_destroy(&z);
    lmmc_vec_destroy(&r);

    return st;
}

lmmc_status_t lmmc_bicgstab_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
) {
    lmmc_itersolve_config_t local_cfg = {0};
    lmmc_vec_t r = {0};
    lmmc_vec_t r_hat = {0};
    lmmc_vec_t p = {0};
    lmmc_vec_t v = {0};
    lmmc_vec_t s = {0};
    lmmc_vec_t t = {0};
    lmmc_vec_t y = {0};
    lmmc_vec_t z = {0};
    lmmc_vec_t ax = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    double norm_b = 0.0;
    double norm_r = 0.0;
    double threshold = 0.0;
    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    size_t iter_count = 0;
    int converged = 0;

    if (a == NULL || b == NULL || x == NULL || a->row_ptr == NULL || b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows == 0 || a->cols == 0 || b->size == 0 || x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols || b->size != a->rows || x->size != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    if (cfg != NULL) {
        local_cfg = *cfg;
    } else {
        st = lmmc_itersolve_default_config(b->size, &local_cfg);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }

    if (!lmmc_is_finite_number(local_cfg.abs_tol) || !lmmc_is_finite_number(local_cfg.rel_tol) ||
        local_cfg.abs_tol < 0.0 || local_cfg.rel_tol < 0.0 || local_cfg.max_iter == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (precond != NULL && precond->size != b->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    if (out_result != NULL) {
        out_result->converged = 0;
        out_result->num_iter = 0;
        out_result->initial_residual_norm = 0.0;
        out_result->final_residual_norm = 0.0;
    }

    st = lmmc_vec_create(b->size, &r);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(b->size, &r_hat);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(b->size, &p);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(b->size, &v);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(b->size, &s);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(b->size, &t);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(b->size, &y);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(b->size, &z);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(b->size, &ax);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    st = lmmc_sparse_mat_vec_mul(a, x, &ax);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    {
        size_t i = 0;
        for (i = 0; i < b->size; ++i) {
            r.data[i] = b->data[i] - ax.data[i];
            p.data[i] = 0.0;
            v.data[i] = 0.0;
        }
    }

    st = lmmc_vec_copy(&r, &r_hat);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    st = lmmc_vec_norm2_checked(b, &norm_b);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_norm2_checked(&r, &norm_r);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    threshold = local_cfg.abs_tol + local_cfg.rel_tol * norm_b;
    if (!lmmc_is_finite_number(threshold)) {
        st = LMMC_STATUS_NUMERICAL_FAILURE;
        goto cleanup;
    }

    if (out_result != NULL) {
        out_result->initial_residual_norm = norm_r;
        out_result->final_residual_norm = norm_r;
    }

    if (norm_r <= threshold) {
        converged = 1;
        iter_count = 0;
        goto cleanup;
    }

    {
        size_t iter = 0;
        for (iter = 0; iter < local_cfg.max_iter; ++iter) {
            double rho_hat = 0.0;
            double beta = 0.0;
            double denom = 0.0;
            double ts = 0.0;
            double tt = 0.0;
            double norm_s = 0.0;
            size_t i = 0;

            st = lmmc_vec_dot_checked(&r_hat, &r, &rho_hat);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }
            if (fabs(rho_hat) <= 1e-30) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            if (iter == 0) {
                st = lmmc_vec_copy(&r, &p);
                if (st != LMMC_STATUS_OK) {
                    goto cleanup;
                }
            } else {
                if (fabs(omega) <= 1e-30) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }
                beta = (rho_hat / rho) * (alpha / omega);
                if (!lmmc_is_finite_number(beta)) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }

                for (i = 0; i < p.size; ++i) {
                    p.data[i] = r.data[i] + beta * (p.data[i] - omega * v.data[i]);
                    if (!lmmc_is_finite_number(p.data[i])) {
                        st = LMMC_STATUS_NUMERICAL_FAILURE;
                        goto cleanup;
                    }
                }
            }

            st = lmmc_apply_precond_or_identity(precond, &p, &y);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            st = lmmc_sparse_mat_vec_mul(a, &y, &v);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            st = lmmc_vec_dot_checked(&r_hat, &v, &denom);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }
            if (fabs(denom) <= 1e-30) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            alpha = rho_hat / denom;
            if (!lmmc_is_finite_number(alpha)) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            for (i = 0; i < s.size; ++i) {
                s.data[i] = r.data[i] - alpha * v.data[i];
            }

            st = lmmc_vec_norm2_checked(&s, &norm_s);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            if (norm_s <= threshold) {
                for (i = 0; i < x->size; ++i) {
                    x->data[i] += alpha * y.data[i];
                    if (!lmmc_is_finite_number(x->data[i])) {
                        st = LMMC_STATUS_NUMERICAL_FAILURE;
                        goto cleanup;
                    }
                }
                norm_r = norm_s;
                converged = 1;
                iter_count = iter + 1;
                if (out_result != NULL) {
                    out_result->final_residual_norm = norm_r;
                }
                break;
            }

            st = lmmc_apply_precond_or_identity(precond, &s, &z);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            st = lmmc_sparse_mat_vec_mul(a, &z, &t);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            st = lmmc_vec_dot_checked(&t, &s, &ts);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }
            st = lmmc_vec_dot_checked(&t, &t, &tt);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }
            if (fabs(tt) <= 1e-30) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            omega = ts / tt;
            if (!lmmc_is_finite_number(omega)) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            for (i = 0; i < x->size; ++i) {
                x->data[i] += alpha * y.data[i] + omega * z.data[i];
                if (!lmmc_is_finite_number(x->data[i])) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }
            }

            for (i = 0; i < r.size; ++i) {
                r.data[i] = s.data[i] - omega * t.data[i];
            }

            st = lmmc_vec_norm2_checked(&r, &norm_r);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            iter_count = iter + 1;
            if (out_result != NULL) {
                out_result->final_residual_norm = norm_r;
            }

            if (norm_r <= threshold) {
                converged = 1;
                break;
            }

            if (fabs(omega) <= 1e-30) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            rho = rho_hat;
        }
    }

cleanup:
    if (out_result != NULL) {
        out_result->converged = converged;
        out_result->num_iter = iter_count;
        if (lmmc_is_finite_number(norm_r)) {
            out_result->final_residual_norm = norm_r;
        }
    }

    lmmc_vec_destroy(&ax);
    lmmc_vec_destroy(&z);
    lmmc_vec_destroy(&y);
    lmmc_vec_destroy(&t);
    lmmc_vec_destroy(&s);
    lmmc_vec_destroy(&v);
    lmmc_vec_destroy(&p);
    lmmc_vec_destroy(&r_hat);
    lmmc_vec_destroy(&r);

    return st;
}

lmmc_status_t lmmc_gmres_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
) {
    lmmc_itersolve_config_t local_cfg = {0};
    lmmc_vec_t r = {0};
    lmmc_vec_t z = {0};
    lmmc_vec_t w = {0};
    lmmc_vec_t ax = {0};
    lmmc_vec_t x_base = {0};
    lmmc_vec_t x_trial = {0};
    lmmc_vec_t* basis = NULL;
    double* h = NULL;
    double* cs = NULL;
    double* sn = NULL;
    double* g = NULL;
    double* y = NULL;
    size_t basis_count = 0;
    size_t restart = 0;
    size_t iter_count = 0;
    size_t n = 0;
    size_t i = 0;
    lmmc_status_t st = LMMC_STATUS_OK;
    double norm_b = 0.0;
    double norm_r = 0.0;
    double threshold = 0.0;
    int converged = 0;

    if (a == NULL || b == NULL || x == NULL || a->row_ptr == NULL || b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows == 0 || a->cols == 0 || b->size == 0 || x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols || b->size != a->rows || x->size != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    if (cfg != NULL) {
        local_cfg = *cfg;
    } else {
        st = lmmc_itersolve_default_config(b->size, &local_cfg);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }

    if (!lmmc_is_finite_number(local_cfg.abs_tol) || !lmmc_is_finite_number(local_cfg.rel_tol) ||
        local_cfg.abs_tol < 0.0 || local_cfg.rel_tol < 0.0 || local_cfg.max_iter == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (precond != NULL && precond->size != b->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    n = b->size;
    restart = local_cfg.restart;
    if (restart == 0) {
        restart = (n < 30) ? n : 30;
    }
    if (restart > n) {
        restart = n;
    }
    if (restart == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (out_result != NULL) {
        out_result->converged = 0;
        out_result->num_iter = 0;
        out_result->initial_residual_norm = 0.0;
        out_result->final_residual_norm = 0.0;
    }

    st = lmmc_vec_create(n, &r);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(n, &z);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(n, &w);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(n, &ax);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(n, &x_base);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(n, &x_trial);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    basis = (lmmc_vec_t*)lmmc_alloc((restart + 1) * sizeof(lmmc_vec_t));
    if (basis == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    memset(basis, 0, (restart + 1) * sizeof(lmmc_vec_t));

    for (i = 0; i < restart + 1; ++i) {
        st = lmmc_vec_create(n, &basis[i]);
        if (st != LMMC_STATUS_OK) {
            goto cleanup;
        }
        ++basis_count;
    }

    h = (double*)lmmc_alloc((restart + 1) * restart * sizeof(double));
    cs = (double*)lmmc_alloc(restart * sizeof(double));
    sn = (double*)lmmc_alloc(restart * sizeof(double));
    g = (double*)lmmc_alloc((restart + 1) * sizeof(double));
    y = (double*)lmmc_alloc(restart * sizeof(double));

    if (h == NULL || cs == NULL || sn == NULL || g == NULL || y == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }

    st = lmmc_vec_norm2_checked(b, &norm_b);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    st = lmmc_sparse_mat_vec_mul(a, x, &ax);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    for (i = 0; i < n; ++i) {
        r.data[i] = b->data[i] - ax.data[i];
    }

    st = lmmc_vec_norm2_checked(&r, &norm_r);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    threshold = local_cfg.abs_tol + local_cfg.rel_tol * norm_b;
    if (!lmmc_is_finite_number(threshold)) {
        st = LMMC_STATUS_NUMERICAL_FAILURE;
        goto cleanup;
    }

    if (out_result != NULL) {
        out_result->initial_residual_norm = norm_r;
        out_result->final_residual_norm = norm_r;
    }

    if (norm_r <= threshold) {
        converged = 1;
        iter_count = 0;
        goto cleanup;
    }

    while (iter_count < local_cfg.max_iter && !converged) {
        size_t inner_limit = restart;
        size_t j = 0;
        int breakdown = 0;
        double beta = 0.0;

        if (inner_limit > local_cfg.max_iter - iter_count) {
            inner_limit = local_cfg.max_iter - iter_count;
        }

        st = lmmc_vec_copy(x, &x_base);
        if (st != LMMC_STATUS_OK) {
            goto cleanup;
        }

        st = lmmc_sparse_mat_vec_mul(a, &x_base, &ax);
        if (st != LMMC_STATUS_OK) {
            goto cleanup;
        }
        for (i = 0; i < n; ++i) {
            r.data[i] = b->data[i] - ax.data[i];
        }

        st = lmmc_apply_precond_or_identity(precond, &r, &z);
        if (st != LMMC_STATUS_OK) {
            goto cleanup;
        }
        st = lmmc_vec_norm2_checked(&z, &beta);
        if (st != LMMC_STATUS_OK) {
            goto cleanup;
        }

        if (beta <= 1e-30) {
            st = LMMC_STATUS_NUMERICAL_FAILURE;
            goto cleanup;
        }

        memset(h, 0, (restart + 1) * restart * sizeof(double));
        memset(cs, 0, restart * sizeof(double));
        memset(sn, 0, restart * sizeof(double));
        memset(g, 0, (restart + 1) * sizeof(double));
        memset(y, 0, restart * sizeof(double));

        for (i = 0; i < n; ++i) {
            basis[0].data[i] = z.data[i] / beta;
            if (!lmmc_is_finite_number(basis[0].data[i])) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }
        }
        g[0] = beta;

        for (j = 0; j < inner_limit; ++j) {
            size_t ii = 0;
            double h_next = 0.0;

            st = lmmc_sparse_mat_vec_mul(a, &basis[j], &ax);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }
            st = lmmc_apply_precond_or_identity(precond, &ax, &w);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            for (ii = 0; ii <= j; ++ii) {
                double hij = 0.0;
                st = lmmc_vec_dot_checked(&basis[ii], &w, &hij);
                if (st != LMMC_STATUS_OK) {
                    goto cleanup;
                }
                h[ii * restart + j] = hij;

                for (i = 0; i < n; ++i) {
                    w.data[i] -= hij * basis[ii].data[i];
                }
            }

            st = lmmc_vec_norm2_checked(&w, &h_next);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }
            h[(j + 1) * restart + j] = h_next;

            for (ii = 0; ii < j; ++ii) {
                double h0 = h[ii * restart + j];
                double h1 = h[(ii + 1) * restart + j];
                h[ii * restart + j] = cs[ii] * h0 + sn[ii] * h1;
                h[(ii + 1) * restart + j] = -sn[ii] * h0 + cs[ii] * h1;
            }

            {
                double hj = h[j * restart + j];
                double hsub = h[(j + 1) * restart + j];
                double denom = sqrt(hj * hj + hsub * hsub);
                double gtmp = 0.0;

                if (!lmmc_is_finite_number(denom)) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }
                if (denom <= 1e-30) {
                    breakdown = 1;
                    break;
                }

                cs[j] = hj / denom;
                sn[j] = hsub / denom;
                if (!lmmc_is_finite_number(cs[j]) || !lmmc_is_finite_number(sn[j])) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }

                h[j * restart + j] = cs[j] * hj + sn[j] * hsub;
                h[(j + 1) * restart + j] = 0.0;

                gtmp = cs[j] * g[j];
                g[j + 1] = -sn[j] * g[j];
                g[j] = gtmp;
            }

            st = lmmc_gmres_back_substitute(h, restart, g, j + 1, y);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            for (i = 0; i < n; ++i) {
                double val = x_base.data[i];
                size_t k = 0;
                for (k = 0; k < j + 1; ++k) {
                    val += y[k] * basis[k].data[i];
                }
                if (!lmmc_is_finite_number(val)) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }
                x_trial.data[i] = val;
            }

            st = lmmc_sparse_mat_vec_mul(a, &x_trial, &ax);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }
            for (i = 0; i < n; ++i) {
                r.data[i] = b->data[i] - ax.data[i];
            }
            st = lmmc_vec_norm2_checked(&r, &norm_r);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            iter_count += 1;
            if (out_result != NULL) {
                out_result->final_residual_norm = norm_r;
            }

            st = lmmc_vec_copy(&x_trial, x);
            if (st != LMMC_STATUS_OK) {
                goto cleanup;
            }

            if (norm_r <= threshold) {
                converged = 1;
                break;
            }

            if (h_next <= 1e-30) {
                breakdown = 1;
                break;
            }

            for (i = 0; i < n; ++i) {
                basis[j + 1].data[i] = w.data[i] / h_next;
                if (!lmmc_is_finite_number(basis[j + 1].data[i])) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }
            }
        }

        if (converged) {
            break;
        }

        if (breakdown) {
            if (norm_r <= threshold) {
                converged = 1;
                break;
            }
            st = LMMC_STATUS_NUMERICAL_FAILURE;
            goto cleanup;
        }
    }

cleanup:
    if (out_result != NULL) {
        out_result->converged = converged;
        out_result->num_iter = iter_count;
        if (lmmc_is_finite_number(norm_r)) {
            out_result->final_residual_norm = norm_r;
        }
    }

    if (basis != NULL) {
        for (i = 0; i < basis_count; ++i) {
            lmmc_vec_destroy(&basis[i]);
        }
        lmmc_free(basis);
    }

    lmmc_free(y);
    lmmc_free(g);
    lmmc_free(sn);
    lmmc_free(cs);
    lmmc_free(h);

    lmmc_vec_destroy(&x_trial);
    lmmc_vec_destroy(&x_base);
    lmmc_vec_destroy(&ax);
    lmmc_vec_destroy(&w);
    lmmc_vec_destroy(&z);
    lmmc_vec_destroy(&r);

    return st;
}
