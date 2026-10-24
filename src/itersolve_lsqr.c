#include "itersolve_internal.h"

typedef struct {
    lmmc_vec_t u, v, w, tmp_vec, av_tmp;
    double alpha, beta, rho_bar, phi_bar, anorm;
    double rho, theta, norm_r, threshold;
} lmmc_lsqr_state_t;

static lmmc_status_t lmmc_lsqr_validate(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_itersolve_config_t* cfg,
    const lmmc_vec_t* x, lmmc_itersolve_config_t* local) {
    lmmc_status_t st = lmmc_itersolve_dispatch_config(a, b, cfg, x, local);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if ((a == NULL && local->apply_transpose_op == NULL) ||
        (a != NULL && local->apply_transpose_op != NULL)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a != NULL && (b->size != a->rows || x->size != a->cols)) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsqr_start(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_itersolve_config_t* cfg,
    const lmmc_vec_t* x, lmmc_lsqr_state_t* s) {
    double norm_b;
    lmmc_status_t st = lmmc_vec_norm2_checked(b, &norm_b);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    s->threshold = cfg->abs_tol + cfg->rel_tol * norm_b;
    if (!isfinite(s->threshold)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    st = lmmc_itersolve_true_residual(a, cfg, b, x, &s->av_tmp, &s->norm_r);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    s->beta = s->norm_r;
    s->phi_bar = s->beta;
    return lmmc_vec_copy(&s->av_tmp, &s->u);
}

static lmmc_status_t lmmc_lsqr_stationary(const lmmc_sparse_mat_t* a,
    const lmmc_itersolve_config_t* cfg, lmmc_lsqr_state_t* s,
    double anorm, int* converged) {
    /* Normalize before A^T so tiny residuals cannot underflow the test. */
    for (size_t i = 0; i < s->av_tmp.size; ++i) {
        s->av_tmp.data[i] /= s->norm_r;
    }
    lmmc_status_t st = lmmc_itersolve_transpose(a, cfg, &s->av_tmp, &s->tmp_vec);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    double norm_atr;
    st = lmmc_vec_norm2_checked(&s->tmp_vec, &norm_atr);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (anorm == 0.0) {
        if (norm_atr != 0.0) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        *converged = 1;
    } else {
        double ratio = norm_atr / anorm;
        if (!isfinite(ratio)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        *converged = ratio <= cfg->rel_tol;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsqr_basis(const lmmc_sparse_mat_t* a,
    const lmmc_itersolve_config_t* cfg, lmmc_lsqr_state_t* s) {
    for (size_t i = 0; i < s->u.size; ++i) {
        s->u.data[i] /= s->beta;
    }
    lmmc_status_t st = lmmc_itersolve_transpose(a, cfg, &s->u, &s->v);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_norm2_checked(&s->v, &s->alpha);
    if (st != LMMC_STATUS_OK || s->alpha == 0.0) {
        return st;
    }
    for (size_t i = 0; i < s->v.size; ++i) {
        s->v.data[i] /= s->alpha;
    }
    s->rho_bar = s->alpha;
    return lmmc_vec_copy(&s->v, &s->w);
}

static lmmc_status_t lmmc_lsqr_bidiagonalize(const lmmc_sparse_mat_t* a,
    const lmmc_itersolve_config_t* cfg, lmmc_lsqr_state_t* s,
    int* basis_exhausted) {
    const double old_alpha = s->alpha;
    lmmc_status_t st = lmmc_itersolve_matvec(a, cfg, &s->v, &s->av_tmp);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    for (size_t i = 0; i < s->u.size; ++i) {
        s->u.data[i] = s->av_tmp.data[i] - old_alpha * s->u.data[i];
    }
    st = lmmc_vec_norm2_checked(&s->u, &s->beta);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    s->anorm = hypot(hypot(s->anorm, old_alpha), s->beta);
    if (!isfinite(s->anorm)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    s->alpha = 0.0;
    if (s->beta != 0.0) {
        for (size_t i = 0; i < s->u.size; ++i) {
            s->u.data[i] /= s->beta;
        }
        st = lmmc_itersolve_transpose(a, cfg, &s->u, &s->tmp_vec);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        for (size_t i = 0; i < s->v.size; ++i) {
            s->v.data[i] = s->tmp_vec.data[i] - s->beta * s->v.data[i];
        }
        st = lmmc_vec_norm2_checked(&s->v, &s->alpha);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        if (s->alpha != 0.0) {
            for (size_t i = 0; i < s->v.size; ++i) {
                s->v.data[i] /= s->alpha;
            }
        }
    }
    *basis_exhausted = s->beta == 0.0 || s->alpha == 0.0;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsqr_rotate(lmmc_vec_t* x,
    lmmc_lsqr_state_t* s, int* updated) {
    *updated = 0;
    s->rho = hypot(s->rho_bar, s->beta);
    if (!isfinite(s->rho)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (s->rho == 0.0) {
        return LMMC_STATUS_OK;
    }
    const double cs = s->rho_bar / s->rho;
    const double sn = s->beta / s->rho;
    s->theta = sn * s->alpha;
    s->rho_bar = -cs * s->alpha;
    const double phi = cs * s->phi_bar;
    s->phi_bar = sn * s->phi_bar;
    const double scale = phi / s->rho;
    if (!isfinite(cs) || !isfinite(sn) || !isfinite(s->theta) ||
        !isfinite(s->rho_bar) || !isfinite(phi) || !isfinite(s->phi_bar) ||
        !isfinite(scale)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    for (size_t i = 0; i < x->size; ++i) {
        x->data[i] += scale * s->w.data[i];
        if (!isfinite(x->data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    *updated = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsqr_iterate(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_itersolve_config_t* cfg, lmmc_vec_t* x,
    lmmc_lsqr_state_t* s, size_t* iterations, int* converged) {
    lmmc_status_t st = lmmc_lsqr_basis(a, cfg, s);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (s->alpha == 0.0) {
        return lmmc_lsqr_stationary(a, cfg, s, 0.0, converged);
    }
    for (size_t iter = 0; iter < cfg->max_iter; ++iter) {
        int basis_exhausted;
        st = lmmc_lsqr_bidiagonalize(a, cfg, s, &basis_exhausted);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        /* Even an exhausted basis still supplies its final solution update. */
        int updated;
        st = lmmc_lsqr_rotate(x, s, &updated);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        if (updated) {
            *iterations = iter + 1;
        }
        st = lmmc_itersolve_true_residual(a, cfg, b, x, &s->av_tmp, &s->norm_r);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        if (updated) {
            lmmc_itersolve_do_log(cfg, *iterations, s->norm_r);
        }
        if (s->norm_r <= s->threshold) {
            *converged = 1;
        } else {
            st = lmmc_lsqr_stationary(a, cfg, s, s->anorm, converged);
            if (st != LMMC_STATUS_OK) {
                return st;
            }
        }
        if (!updated || *converged || basis_exhausted || *iterations == cfg->max_iter) {
            break;
        }
        const double scale = s->theta / s->rho;
        if (!isfinite(scale)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        for (size_t i = 0; i < s->w.size; ++i) {
            s->w.data[i] = s->v.data[i] - scale * s->w.data[i];
            if (!isfinite(s->w.data[i])) {
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsqr_solve(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x, lmmc_itersolve_result_t* out_result) {
    lmmc_itersolve_config_t local;
    lmmc_status_t st = lmmc_lsqr_validate(a, b, cfg, x, &local);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    lmmc_lsqr_state_t s = {.norm_r = NAN};
    lmmc_real_t* storage = NULL;
    size_t iterations = 0;
    int converged = 0;
    lmmc_itersolve_result_init(out_result);
    lmmc_vec_t* vectors[] = {&s.u, &s.v, &s.w, &s.tmp_vec, &s.av_tmp};
    const size_t sizes[] = {b->size, x->size, x->size, x->size, b->size};
    st = lmmc_itersolve_workspace(vectors, sizes, 5, &storage);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_lsqr_start(a, b, &local, x, &s);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    lmmc_itersolve_initial(out_result, s.norm_r);
    lmmc_itersolve_do_log(&local, 0, s.norm_r);
    if (s.norm_r <= s.threshold) {
        converged = 1;
    } else {
        st = lmmc_lsqr_iterate(a, b, &local, x, &s, &iterations, &converged);
    }
cleanup:
    lmmc_itersolve_finish(out_result, converged, iterations, s.norm_r);
    lmmc_memory_free(storage);
    return st;
}
