#include "itersolve_internal.h"

typedef struct {
    lmmc_vec_t u, v, w, tmp_vec, av_tmp;
    double alpha_l, beta_l, rho_bar, phi_bar_l;
    double rho_val, cs_l, sn_l, theta_l, phi_l;
    double norm_r, norm_b_val, threshold;
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
    if (local->max_iter == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a != NULL && (b->size != a->rows || x->size != a->cols)) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsqr_start(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x, lmmc_lsqr_state_t* s) {
    const size_t m = b->size;
    lmmc_status_t st;
    /** @brief 初始残差：u = b - A*x。 */
    st = lmmc_itersolve_matvec(a, cfg, x, &s->av_tmp);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    {
        size_t i;
        for (i = 0; i < m; ++i) {
            s->u.data[i] = b->data[i] - s->av_tmp.data[i];
        }
    }

    /** @brief 双对角化系数：beta = ||u||。 */
    st = lmmc_vec_norm2_checked(&s->u, &s->beta_l);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    st = lmmc_vec_norm2_checked(b, &s->norm_b_val);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    s->norm_r = s->beta_l;
    s->threshold = cfg->abs_tol + cfg->rel_tol * s->norm_b_val;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsqr_basis(const lmmc_sparse_mat_t* a,
    const lmmc_itersolve_config_t* cfg, lmmc_lsqr_state_t* s, int* converged) {
    const size_t m = s->u.size;
    const size_t n_cols = s->v.size;
    lmmc_status_t st;
    {
        size_t i;
        for (i = 0; i < m; ++i) {
            s->u.data[i] /= s->beta_l;
        }
    }

    /** @brief 转置乘积：v = A^T * u。 */
    st = lmmc_itersolve_transpose(a, cfg, &s->u, &s->v);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    /** @brief 双对角化系数：alpha = ||v||。 */
    st = lmmc_vec_norm2_checked(&s->v, &s->alpha_l);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    if (s->alpha_l <= 1e-30) {
        *converged = 1;
        return st;
    }

    {
        size_t i;
        for (i = 0; i < n_cols; ++i) {
            s->v.data[i] /= s->alpha_l;
        }
    }

    st = lmmc_vec_copy(&s->v, &s->w);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    s->rho_bar = s->alpha_l;
    s->phi_bar_l = s->beta_l;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsqr_bidiagonalize(const lmmc_sparse_mat_t* a,
    const lmmc_itersolve_config_t* cfg, lmmc_lsqr_state_t* s,
    int* converged) {
    const size_t m = s->u.size;
    const size_t n_cols = s->v.size;
    size_t i;
    lmmc_status_t st;
    /** @brief 双对角化递推：u = A*v - alpha*u。 */
    st = lmmc_itersolve_matvec(a, cfg, &s->v, &s->av_tmp);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    for (i = 0; i < m; ++i) {
        s->u.data[i] = s->av_tmp.data[i] - s->alpha_l * s->u.data[i];
    }

    /** @brief 双对角化系数：beta = ||u||。 */
    st = lmmc_vec_norm2_checked(&s->u, &s->beta_l);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    if (s->beta_l <= 1e-30) {
        *converged = 1;
        return LMMC_STATUS_OK;
    }

    for (i = 0; i < m; ++i) {
        s->u.data[i] /= s->beta_l;
    }

    /** @brief 双对角化递推：v = A^T*u - beta*v。 */
    st = lmmc_itersolve_transpose(a, cfg, &s->u, &s->tmp_vec);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    for (i = 0; i < n_cols; ++i) {
        s->v.data[i] = s->tmp_vec.data[i] - s->beta_l * s->v.data[i];
    }

    /** @brief 双对角化系数：alpha = ||v||。 */
    st = lmmc_vec_norm2_checked(&s->v, &s->alpha_l);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    if (s->alpha_l <= 1e-30) {
        *converged = 1;
        return LMMC_STATUS_OK;
    }

    for (i = 0; i < n_cols; ++i) {
        s->v.data[i] /= s->alpha_l;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsqr_rotate(lmmc_vec_t* x, lmmc_lsqr_state_t* s) {
    const size_t n_cols = x->size;
    size_t i;
    double scale;
    lmmc_status_t st;
    /** @brief 构造 Givens 旋转。 */
    s->rho_val = sqrt(s->rho_bar * s->rho_bar + s->beta_l * s->beta_l);
    if (s->rho_val <= 1e-30) {
        st = LMMC_STATUS_NUMERICAL_FAILURE;
        return st;
    }

    s->cs_l = s->rho_bar / s->rho_val;
    s->sn_l = s->beta_l / s->rho_val;
    s->theta_l = s->sn_l * s->alpha_l;
    s->rho_bar = -s->cs_l * s->alpha_l;
    s->phi_l = s->cs_l * s->phi_bar_l;
    s->phi_bar_l = s->sn_l * s->phi_bar_l;

    /** @brief 解递推：x += (phi/rho) * w。 */
    scale = s->phi_l / s->rho_val;
    for (i = 0; i < n_cols; ++i) {
        x->data[i] += scale * s->w.data[i];
        if (!isfinite(x->data[i])) {
            st = LMMC_STATUS_NUMERICAL_FAILURE;
            return st;
        }
    }

    /** @brief 方向递推：w = v - (theta/rho) * w。 */
    scale = s->theta_l / s->rho_val;
    for (i = 0; i < n_cols; ++i) {
        s->w.data[i] = s->v.data[i] - scale * s->w.data[i];
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsqr_iterate(const lmmc_sparse_mat_t* a,
    const lmmc_itersolve_config_t* cfg, lmmc_vec_t* x,
    lmmc_lsqr_state_t* s, lmmc_itersolve_result_t* result,
    size_t* iterations, int* converged) {
    lmmc_status_t st = lmmc_lsqr_basis(a, cfg, s, converged);
    if (st != LMMC_STATUS_OK || *converged) {
        return st;
    }
    for (size_t iter = 0; iter < cfg->max_iter; ++iter) {
        st = lmmc_lsqr_bidiagonalize(a, cfg, s, converged);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        if (*converged) {
            *iterations = iter + 1;
            break;
        }
        st = lmmc_lsqr_rotate(x, s);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        s->norm_r = fabs(s->phi_bar_l);
        *iterations = iter + 1;
        if (result != NULL) {
            result->final_residual_norm = s->norm_r;
        }
        lmmc_itersolve_do_log(cfg, *iterations, s->norm_r);
        if (s->norm_r <= s->threshold) {
            *converged = 1;
            break;
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
    lmmc_lsqr_state_t s = {0};
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
    if (s.norm_r <= s.threshold || s.beta_l <= 1e-30) {
        converged = 1;
    }
    else {
        st = lmmc_lsqr_iterate(a, &local, x, &s, out_result, &iterations, &converged);
    }
cleanup:
    lmmc_itersolve_finish(out_result, converged, iterations, s.norm_r);
    if (out_result != NULL) {
        out_result->final_residual_norm = s.norm_r;
    }
    lmmc_memory_free(storage);
    return st;
}
