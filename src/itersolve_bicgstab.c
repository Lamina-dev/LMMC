#include "itersolve_internal.h"

typedef struct {
    lmmc_vec_t r, r_hat, p, v, s, t, y, z, ax;
    lmmc_real_t rho, rho_hat, alpha, omega, norm, norm_s, threshold;
} lmmc_bicgstab_state_t;

static lmmc_status_t lmmc_bicgstab_start(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x, lmmc_bicgstab_state_t* s) {
    lmmc_status_t st = lmmc_itersolve_residual(a, b, x, &s->ax, &s->r);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_copy(&s->r, &s->r_hat);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_itersolve_threshold(cfg, b, &s->r, &s->norm, &s->threshold);
}

static lmmc_status_t lmmc_bicgstab_direction(lmmc_bicgstab_state_t* s, size_t iter) {
    lmmc_status_t st = lmmc_vec_dot_checked(&s->r_hat, &s->r, &s->rho_hat);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (fabs(s->rho_hat) <= 1e-30) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (iter == 0) {
        return lmmc_vec_copy(&s->r, &s->p);
    }
    if (fabs(s->omega) <= 1e-30) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    lmmc_real_t beta = (s->rho_hat / s->rho) * (s->alpha / s->omega);
    if (!isfinite(beta)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    for (size_t i = 0; i < s->p.size; ++i) {
        lmmc_real_t product = s->omega * s->v.data[i];
        lmmc_real_t difference = s->p.data[i] - product;
        product = beta * difference;
        s->p.data[i] = s->r.data[i] + product;
        if (!isfinite(s->p.data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_bicgstab_alpha(const lmmc_sparse_mat_t* a,
    const lmmc_precond_t* precond, lmmc_bicgstab_state_t* s) {
    lmmc_real_t denom;
    lmmc_status_t st = lmmc_apply_precond_or_identity(precond, &s->p, &s->y);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_sparse_mat_vec_mul(a, &s->y, &s->v);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_dot_checked(&s->r_hat, &s->v, &denom);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (fabs(denom) <= 1e-30) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    s->alpha = s->rho_hat / denom;
    if (!isfinite(s->alpha)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    st = lmmc_vec_copy(&s->r, &s->s);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_axpy(-s->alpha, &s->v, &s->s);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_vec_norm2_checked(&s->s, &s->norm_s);
}

static lmmc_status_t lmmc_bicgstab_omega(const lmmc_sparse_mat_t* a,
    const lmmc_precond_t* precond, lmmc_bicgstab_state_t* s) {
    lmmc_real_t ts, tt;
    lmmc_status_t st = lmmc_apply_precond_or_identity(precond, &s->s, &s->z);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_sparse_mat_vec_mul(a, &s->z, &s->t);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_dot_checked(&s->t, &s->s, &ts);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_dot_checked(&s->t, &s->t, &tt);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (fabs(tt) <= 1e-30) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    s->omega = ts / tt;
    return isfinite(s->omega) ? LMMC_STATUS_OK : LMMC_STATUS_NUMERICAL_FAILURE;
}

static lmmc_status_t lmmc_bicgstab_update(lmmc_vec_t* x,
    lmmc_bicgstab_state_t* s, int early) {
    for (size_t i = 0; i < x->size; ++i) {
        lmmc_real_t delta = s->alpha * s->y.data[i];
        if (!early) {
            lmmc_real_t second = s->omega * s->z.data[i];
            delta = delta + second;
        }
        x->data[i] = x->data[i] + delta;
        if (!isfinite(x->data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    if (early) {
        s->norm = s->norm_s;
        return LMMC_STATUS_OK;
    }
    lmmc_status_t st = lmmc_vec_copy(&s->s, &s->r);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_axpy(-s->omega, &s->t, &s->r);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_vec_norm2_checked(&s->r, &s->norm);
}

static lmmc_status_t lmmc_bicgstab_step(const lmmc_sparse_mat_t* a,
    const lmmc_precond_t* precond, lmmc_vec_t* x,
    lmmc_bicgstab_state_t* s, size_t iter) {
    lmmc_status_t st = lmmc_bicgstab_direction(s, iter);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_bicgstab_alpha(a, precond, s);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (s->norm_s <= s->threshold) {
        return lmmc_bicgstab_update(x, s, 1);
    }
    st = lmmc_bicgstab_omega(a, precond, s);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_bicgstab_update(x, s, 0);
}

static lmmc_status_t lmmc_bicgstab_iterate(const lmmc_sparse_mat_t* a,
    const lmmc_precond_t* precond, const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x, lmmc_bicgstab_state_t* s, lmmc_itersolve_result_t* result,
    size_t* iterations, int* converged) {
    for (size_t iter = 0; iter < cfg->max_iter; ++iter) {
        lmmc_status_t st = lmmc_bicgstab_step(a, precond, x, s, iter);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        *iterations = iter + 1;
        if (result != NULL) {
            result->final_residual_norm = s->norm;
        }
        lmmc_itersolve_do_log(cfg, *iterations, s->norm);
        if (s->norm <= s->threshold) {
            *converged = 1;
            break;
        }
        if (fabs(s->omega) <= 1e-30) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        s->rho = s->rho_hat;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_bicgstab_solve(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg, lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result) {
    lmmc_itersolve_config_t local = {0};
    lmmc_bicgstab_state_t s = {0};
    lmmc_real_t* storage = NULL;
    size_t iterations = 0;
    int converged = 0;
    s.rho = s.alpha = s.omega = 1.0;
    lmmc_status_t st = lmmc_itersolve_validate(a, b, precond, cfg, x, &local, out_result);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    lmmc_vec_t* vectors[] = {&s.r, &s.r_hat, &s.p, &s.v, &s.s, &s.t, &s.y, &s.z, &s.ax};
    const size_t n = b->size;
    const size_t sizes[] = {n, n, n, n, n, n, n, n, n};
    st = lmmc_itersolve_workspace(vectors, sizes, 9, &storage);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_bicgstab_start(a, b, &local, x, &s);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    lmmc_itersolve_initial(out_result, s.norm);
    lmmc_itersolve_do_log(&local, 0, s.norm);
    if (s.norm <= s.threshold) {
        converged = 1;
    }
    else {
        st = lmmc_bicgstab_iterate(a, precond, &local, x, &s, out_result, &iterations, &converged);
    }
cleanup:
    lmmc_itersolve_finish(out_result, converged, iterations, s.norm);
    lmmc_memory_free(storage);
    return st;
}
