#include "itersolve_gmres_internal.h"

static lmmc_status_t lmmc_gmres_start(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_itersolve_config_t* cfg,
    const lmmc_vec_t* x, lmmc_gmres_state_t* s) {
    lmmc_real_t norm_b;
    lmmc_status_t st = lmmc_vec_norm2_checked(b, &norm_b);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_itersolve_residual(a, b, x, &s->ax, &s->r);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_norm2_checked(&s->r, &s->norm);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    s->threshold = cfg->abs_tol + cfg->rel_tol * norm_b;
    return isfinite(s->threshold) ? LMMC_STATUS_OK : LMMC_STATUS_NUMERICAL_FAILURE;
}

static lmmc_status_t lmmc_gmres_restart(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_precond_t* precond,
    lmmc_vec_t* x, lmmc_gmres_state_t* s) {
    lmmc_real_t beta;
    lmmc_status_t st = lmmc_vec_copy(x, &s->x_base);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_itersolve_residual(a, b, &s->x_base, &s->ax, &s->r);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_apply_precond_or_identity(precond, &s->r, &s->z);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_norm2_checked(&s->z, &beta);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (beta <= 1e-30) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    memset(s->h, 0, s->h_count * sizeof(lmmc_real_t));
    memset(s->cs, 0, s->restart * sizeof(lmmc_real_t));
    memset(s->sn, 0, s->restart * sizeof(lmmc_real_t));
    memset(s->g, 0, (s->restart + 1) * sizeof(lmmc_real_t));
    memset(s->y, 0, s->restart * sizeof(lmmc_real_t));
    st = lmmc_gmres_normalize(&s->basis[0], &s->z, beta);
    if (st == LMMC_STATUS_OK) {
        s->g[0] = beta;
    }
    return st;
}

static lmmc_status_t lmmc_gmres_step(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_precond_t* precond,
    lmmc_gmres_state_t* s, size_t j, lmmc_real_t* h_next, int* breakdown) {
    lmmc_status_t st = lmmc_gmres_arnoldi(a, precond, s, j, h_next);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_gmres_rotate(s, j, breakdown);
    if (st != LMMC_STATUS_OK || *breakdown) {
        return st;
    }
    st = lmmc_gmres_back_substitute(s, j + 1);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_gmres_trial(s, j);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_itersolve_residual(a, b, &s->x_trial, &s->ax, &s->r);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_vec_norm2_checked(&s->r, &s->norm);
}

static lmmc_status_t lmmc_gmres_cycle(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg, lmmc_vec_t* x,
    lmmc_gmres_state_t* s, lmmc_itersolve_result_t* result,
    size_t* iterations, int* converged) {
    size_t limit = s->restart;
    if (limit > cfg->max_iter - *iterations) {
        limit = cfg->max_iter - *iterations;
    }
    lmmc_status_t st = lmmc_gmres_restart(a, b, precond, x, s);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    for (size_t j = 0; j < limit; ++j) {
        lmmc_real_t h_next;
        int breakdown;
        st = lmmc_gmres_step(a, b, precond, s, j, &h_next, &breakdown);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        if (breakdown) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        *iterations += 1;
        if (result != NULL) {
            result->final_residual_norm = s->norm;
        }
        lmmc_itersolve_do_log(cfg, *iterations, s->norm);
        st = lmmc_vec_copy(&s->x_trial, x);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        if (s->norm <= s->threshold) {
            *converged = 1;
            return LMMC_STATUS_OK;
        }
        if (h_next <= 1e-30) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        st = lmmc_gmres_normalize(&s->basis[j + 1], &s->w, h_next);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_gmres_solve(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg, lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result) {
    lmmc_itersolve_config_t local = {0};
    lmmc_gmres_state_t s = {0};
    size_t iterations = 0;
    int converged = 0;
    lmmc_status_t st = lmmc_itersolve_validate(a, b, precond, cfg, x, &local, out_result);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_gmres_workspace(b->size, local.restart, &s);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_gmres_start(a, b, &local, x, &s);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    lmmc_itersolve_initial(out_result, s.norm);
    lmmc_itersolve_do_log(&local, 0, s.norm);
    converged = s.norm <= s.threshold;
    while (iterations < local.max_iter && !converged) {
        st = lmmc_gmres_cycle(a, b, precond, &local, x, &s, out_result, &iterations, &converged);
        if (st != LMMC_STATUS_OK) {
            break;
        }
    }
cleanup:
    lmmc_itersolve_finish(out_result, converged, iterations, s.norm);
    lmmc_memory_free(s.basis);
    return st;
}
