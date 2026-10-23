#include "itersolve_internal.h"
#include "precond_internal.h"

typedef struct {
    lmmc_vec_t r_prev, r_curr, y, v, lanczos_tmp;
    lmmc_vec_t w_prev2, w_prev, w_next, true_r;
    double beta, old_beta, beta_next, alpha;
    double dbar, epsln, cs, sn, phibar;
    double norm_r, threshold;
} lmmc_minres_state_t;

static lmmc_status_t lmmc_minres_validate(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg, const lmmc_vec_t* x,
    lmmc_itersolve_config_t* local) {
    lmmc_status_t st = lmmc_itersolve_dispatch_config(a, b, cfg, x, local);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (a != NULL) {
        if (a->rows != a->cols || b->size != a->rows || x->size != a->cols) {
            return LMMC_STATUS_DIMENSION_MISMATCH;
        }
    } else if (b->size != x->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    return lmmc_precond_validate_spd(precond, b->size);
}

static lmmc_status_t lmmc_minres_beta(const lmmc_vec_t* r,
    const lmmc_vec_t* y, double* beta) {
    int nonzero = 0;
    for (size_t i = 0; i < r->size; ++i) {
        if (!isfinite(r->data[i]) || !isfinite(y->data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        nonzero |= r->data[i] != 0.0;
    }
    double dot;
    lmmc_status_t st = lmmc_vec_dot_checked(r, y, &dot);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (dot < 0.0) {
        return LMMC_STATUS_NOT_POSITIVE_DEFINITE;
    }
    if (dot == 0.0 && nonzero) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *beta = sqrt(dot);
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_minres_start(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_itersolve_config_t* cfg,
    const lmmc_vec_t* x, lmmc_minres_state_t* s) {
    double norm_b;
    lmmc_status_t st = lmmc_vec_norm2_checked(b, &norm_b);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    s->threshold = cfg->abs_tol + cfg->rel_tol * norm_b;
    if (!isfinite(s->threshold)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    st = lmmc_itersolve_true_residual(a, cfg, b, x, &s->true_r, &s->norm_r);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_vec_copy(&s->true_r, &s->r_curr);
}

static lmmc_status_t lmmc_minres_lanczos(const lmmc_sparse_mat_t* a,
    const lmmc_precond_t* precond, const lmmc_itersolve_config_t* cfg,
    lmmc_minres_state_t* s) {
    for (size_t i = 0; i < s->v.size; ++i) {
        s->v.data[i] = s->y.data[i] / s->beta;
        if (!isfinite(s->v.data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    lmmc_status_t st = lmmc_itersolve_matvec(a, cfg, &s->v, &s->lanczos_tmp);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (s->old_beta != 0.0) {
        const double scale = s->beta / s->old_beta;
        if (!isfinite(scale)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        for (size_t i = 0; i < s->v.size; ++i) {
            s->lanczos_tmp.data[i] -= scale * s->r_prev.data[i];
        }
    }
    for (size_t i = 0; i < s->v.size; ++i) {
        if (!isfinite(s->lanczos_tmp.data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    st = lmmc_vec_dot_checked(&s->v, &s->lanczos_tmp, &s->alpha);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    const double scale = s->alpha / s->beta;
    if (!isfinite(scale)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    for (size_t i = 0; i < s->v.size; ++i) {
        s->lanczos_tmp.data[i] -= scale * s->r_curr.data[i];
        if (!isfinite(s->lanczos_tmp.data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    lmmc_vec_t tmp = s->r_prev;
    s->r_prev = s->r_curr;
    s->r_curr = s->lanczos_tmp;
    s->lanczos_tmp = tmp;
    st = lmmc_apply_precond_or_identity(precond, &s->r_curr, &s->y);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_minres_beta(&s->r_curr, &s->y, &s->beta_next);
}

static lmmc_status_t lmmc_minres_rotate(lmmc_vec_t* x,
    lmmc_minres_state_t* s, int* updated) {
    const double oldeps = s->epsln;
    const double delta = s->cs * s->dbar + s->sn * s->alpha;
    const double gbar = s->sn * s->dbar - s->cs * s->alpha;
    s->epsln = s->sn * s->beta_next;
    s->dbar = -s->cs * s->beta_next;
    const double gamma = hypot(gbar, s->beta_next);
    if (!isfinite(delta) || !isfinite(gbar) || !isfinite(s->epsln) ||
        !isfinite(s->dbar) || !isfinite(gamma)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *updated = 0;
    if (gamma == 0.0) {
        return LMMC_STATUS_OK;
    }
    s->cs = gbar / gamma;
    s->sn = s->beta_next / gamma;
    const double phi = s->cs * s->phibar;
    s->phibar = s->sn * s->phibar;
    if (!isfinite(s->cs) || !isfinite(s->sn) || !isfinite(phi) || !isfinite(s->phibar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    for (size_t i = 0; i < x->size; ++i) {
        s->w_next.data[i] = (s->v.data[i] - oldeps * s->w_prev2.data[i] -
            delta * s->w_prev.data[i]) / gamma;
        if (!isfinite(s->w_next.data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    for (size_t i = 0; i < x->size; ++i) {
        x->data[i] += phi * s->w_next.data[i];
        if (!isfinite(x->data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    lmmc_vec_t tmp = s->w_prev2;
    s->w_prev2 = s->w_prev;
    s->w_prev = s->w_next;
    s->w_next = tmp;
    *updated = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_minres_iterate(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg, lmmc_vec_t* x,
    lmmc_minres_state_t* s, size_t* iterations, int* converged) {
    lmmc_status_t st = lmmc_apply_precond_or_identity(precond, &s->r_curr, &s->y);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_minres_beta(&s->r_curr, &s->y, &s->beta);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    s->cs = -1.0;
    s->phibar = s->beta;
    for (size_t iter = 0; iter < cfg->max_iter; ++iter) {
        st = lmmc_minres_lanczos(a, precond, cfg, s);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        int updated;
        st = lmmc_minres_rotate(x, s, &updated);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        if (!updated) {
            break;
        }
        *iterations = iter + 1;
        st = lmmc_itersolve_true_residual(a, cfg, b, x, &s->true_r, &s->norm_r);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        lmmc_itersolve_do_log(cfg, *iterations, s->norm_r);
        if (s->norm_r <= s->threshold) {
            *converged = 1;
            break;
        }
        if (s->beta_next == 0.0) {
            break;
        }
        s->old_beta = s->beta;
        s->beta = s->beta_next;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_minres_solve(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg, lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result) {
    lmmc_itersolve_config_t local;
    lmmc_status_t st = lmmc_minres_validate(a, b, precond, cfg, x, &local);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    lmmc_minres_state_t s = {.norm_r = NAN};
    lmmc_real_t* storage = NULL;
    size_t iterations = 0;
    int converged = 0;
    lmmc_itersolve_result_init(out_result);
    lmmc_vec_t* vectors[] = {&s.r_prev, &s.r_curr, &s.y, &s.v, &s.lanczos_tmp,
        &s.w_prev2, &s.w_prev, &s.w_next, &s.true_r};
    const size_t n = b->size;
    const size_t sizes[] = {n, n, n, n, n, n, n, n, n};
    st = lmmc_itersolve_workspace(vectors, sizes, 9, &storage);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_minres_start(a, b, &local, x, &s);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    lmmc_itersolve_initial(out_result, s.norm_r);
    lmmc_itersolve_do_log(&local, 0, s.norm_r);
    if (s.norm_r <= s.threshold) {
        converged = 1;
    } else {
        st = lmmc_minres_iterate(a, b, precond, &local, x, &s, &iterations, &converged);
    }
cleanup:
    lmmc_itersolve_finish(out_result, converged, iterations, s.norm_r);
    lmmc_memory_free(storage);
    return st;
}
