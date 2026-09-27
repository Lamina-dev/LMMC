#include "itersolve_internal.h"

typedef struct {
    lmmc_vec_t r, z, p, ap;
    lmmc_real_t rho, norm, threshold;
} lmmc_cg_state_t;

static lmmc_status_t lmmc_cg_start(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x, lmmc_cg_state_t* s) {
    lmmc_status_t st = lmmc_itersolve_residual(a, b, x, &s->ap, &s->r);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_itersolve_threshold(cfg, b, &s->r, &s->norm, &s->threshold);
}

static lmmc_status_t lmmc_cg_direction(const lmmc_precond_t* precond,
    lmmc_cg_state_t* s, int initial) {
    lmmc_real_t rho_new;
    lmmc_status_t st = lmmc_apply_precond_or_identity(precond, &s->r, &s->z);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_dot_checked(&s->r, &s->z, &rho_new);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (fabs(rho_new) <= 1e-30) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (initial) {
        s->rho = rho_new;
        return lmmc_vec_copy(&s->z, &s->p);
    }
    lmmc_real_t beta = rho_new / s->rho;
    if (!isfinite(beta)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    for (size_t i = 0; i < s->p.size; ++i) {
        lmmc_real_t product = beta * s->p.data[i];
        s->p.data[i] = s->z.data[i] + product;
        if (!isfinite(s->p.data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    s->rho = rho_new;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_cg_step(const lmmc_sparse_mat_t* a,
    lmmc_vec_t* x, lmmc_cg_state_t* s) {
    lmmc_real_t denom;
    lmmc_status_t st = lmmc_sparse_mat_vec_mul(a, &s->p, &s->ap);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_dot_checked(&s->p, &s->ap, &denom);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (fabs(denom) <= 1e-30) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    lmmc_real_t alpha = s->rho / denom;
    if (!isfinite(alpha)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    for (size_t i = 0; i < x->size; ++i) {
        lmmc_real_t product = alpha * s->p.data[i];
        x->data[i] = x->data[i] + product;
        if (!isfinite(x->data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    st = lmmc_vec_axpy(-alpha, &s->ap, &s->r);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_vec_norm2_checked(&s->r, &s->norm);
}

static lmmc_status_t lmmc_cg_iterate(const lmmc_sparse_mat_t* a,
    const lmmc_precond_t* precond, const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x, lmmc_cg_state_t* s, lmmc_itersolve_result_t* result,
    size_t* iterations, int* converged) {
    lmmc_status_t st = lmmc_cg_direction(precond, s, 1);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    for (size_t iter = 0; iter < cfg->max_iter; ++iter) {
        st = lmmc_cg_step(a, x, s);
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
        st = lmmc_cg_direction(precond, s, 0);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_cg_solve(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg, lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result) {
    lmmc_itersolve_config_t local = {0};
    lmmc_cg_state_t s = {0};
    lmmc_real_t* storage = NULL;
    size_t iterations = 0;
    int converged = 0;
    lmmc_status_t st = lmmc_itersolve_validate(a, b, precond, cfg, x, &local, out_result);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    lmmc_vec_t* vectors[] = {&s.r, &s.z, &s.p, &s.ap};
    const size_t sizes[] = {b->size, b->size, b->size, b->size};
    st = lmmc_itersolve_workspace(vectors, sizes, 4, &storage);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_cg_start(a, b, &local, x, &s);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    lmmc_itersolve_initial(out_result, s.norm);
    lmmc_itersolve_do_log(&local, 0, s.norm);
    if (s.norm <= s.threshold) {
        converged = 1;
    }
    else {
        st = lmmc_cg_iterate(a, precond, &local, x, &s, out_result, &iterations, &converged);
    }
cleanup:
    lmmc_itersolve_finish(out_result, converged, iterations, s.norm);
    lmmc_memory_free(storage);
    return st;
}
