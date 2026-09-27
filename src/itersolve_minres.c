#include "itersolve_internal.h"

typedef struct {
    lmmc_vec_t v_prev, v_curr, v_next, w_prev, w_curr, av, z_vec;
    double alpha_k, beta_k, beta_kp1;
    double sn_prev, cs_curr, sn_curr;
    double phi_bar, epsilon_curr, delta_bar, gamma_val, phi_val;
    double norm_r, norm_b_val, threshold;
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
    } else {
        if (b->size != x->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    }
    if (precond != NULL && precond->size != b->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    if (local->max_iter == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_minres_start(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg, lmmc_vec_t* x,
    lmmc_minres_state_t* s) {
    size_t n = b->size;
    lmmc_status_t st;
    /** @brief 初始残差：r = b - A*x。 */
    st = lmmc_itersolve_matvec(a, cfg, x, &s->av);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    {
        size_t i;
        for (i = 0; i < n; ++i) {
            s->v_curr.data[i] = b->data[i] - s->av.data[i];
        }
    }

    st = lmmc_apply_precond_or_identity(precond, &s->v_curr, &s->z_vec);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    /** @brief 初始 Lanczos 系数：beta_1 = sqrt(r^T * z)。 */
    {
        double dot_rz;
        st = lmmc_vec_dot_checked(&s->v_curr, &s->z_vec, &dot_rz);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        if (dot_rz < 0.0) {
            dot_rz = -dot_rz;
        }
        s->beta_k = sqrt(dot_rz);
    }

    st = lmmc_vec_norm2_checked(b, &s->norm_b_val);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    s->norm_r = s->beta_k;
    s->phi_bar = s->beta_k;
    s->threshold = cfg->abs_tol + cfg->rel_tol * s->norm_b_val;
    return LMMC_STATUS_OK;
}

static void lmmc_minres_normalize(lmmc_minres_state_t* s) {
    size_t n = s->v_curr.size;
    /** @brief 归一化 v_curr = z / beta_k，其余向量置零。 */
    {
        size_t i;
        for (i = 0; i < n; ++i) {
            s->v_curr.data[i] = s->z_vec.data[i] / s->beta_k;
            s->v_prev.data[i] = 0.0;
            s->w_prev.data[i] = 0.0;
            s->w_curr.data[i] = 0.0;
        }
    }

    /** @brief 初始化 Givens 旋转状态。 */
    s->sn_prev = 0.0;
    s->cs_curr = 1.0;
    s->sn_curr = 0.0;
}

static lmmc_status_t lmmc_minres_lanczos(const lmmc_sparse_mat_t* a,
    const lmmc_precond_t* precond, const lmmc_itersolve_config_t* cfg,
    lmmc_minres_state_t* s) {
    size_t n = s->v_curr.size;
    size_t i;
    lmmc_status_t st;
    /** @brief Lanczos 乘积：av = A * v_curr。 */
    st = lmmc_itersolve_matvec(a, cfg, &s->v_curr, &s->av);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    st = lmmc_apply_precond_or_identity(precond, &s->av, &s->z_vec);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    /** @brief Lanczos 系数：alpha_k = v_curr^T * z_vec。 */
    st = lmmc_vec_dot_checked(&s->v_curr, &s->z_vec, &s->alpha_k);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    /** @brief Lanczos 递推：v_next = z_vec - alpha_k * v_curr - beta_k * v_prev。 */
    for (i = 0; i < n; ++i) {
        s->v_next.data[i] = s->z_vec.data[i] - s->alpha_k * s->v_curr.data[i] - s->beta_k * s->v_prev.data[i];
    }

    /** @brief 下一 Lanczos 系数：beta_{k+1} = sqrt(v_next^T * v_next)。 */
    {
        double dot_vv;
        st = lmmc_vec_dot_checked(&s->v_next, &s->v_next, &dot_vv);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        if (dot_vv < 0.0) {
            dot_vv = -dot_vv;
        }
        s->beta_kp1 = sqrt(dot_vv);
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_minres_rotate(lmmc_vec_t* x, lmmc_minres_state_t* s) {
    size_t n = x->size;
    size_t i;
    double cs_new, sn_new;
    lmmc_status_t st;
    /** @brief 应用前一 Givens 旋转求 delta_bar。 */
    s->epsilon_curr = s->sn_prev * s->beta_k;
    s->delta_bar = s->cs_curr * s->alpha_k - s->sn_curr * s->epsilon_curr;

    /** @brief 构造 Givens 旋转以消去 beta_{k+1}。 */
    s->gamma_val = sqrt(s->delta_bar * s->delta_bar + s->beta_kp1 * s->beta_kp1);

    if (s->gamma_val <= 1e-30) {
        st = LMMC_STATUS_NUMERICAL_FAILURE;
        return st;
    }

    cs_new = s->delta_bar / s->gamma_val;
    sn_new = s->beta_kp1 / s->gamma_val;

    /** @brief 旋转系数：phi = cs_new * phi_bar。 */
    s->phi_val = cs_new * s->phi_bar;
    /** @brief 残差递推：phi_bar = -sn_new * phi_bar。 */
    s->phi_bar = -sn_new * s->phi_bar;

    /** @brief 方向递推：w_new = (v_curr - epsilon_curr*w_prev - delta_bar*w_curr) / gamma。 */
    for (i = 0; i < n; ++i) {
        s->z_vec.data[i] = (s->v_curr.data[i] - s->epsilon_curr * s->w_prev.data[i] - s->delta_bar * s->w_curr.data[i]) / s->gamma_val;
    }

    /** @brief 解递推：x = x + phi * w_new。 */
    for (i = 0; i < n; ++i) {
        x->data[i] += s->phi_val * s->z_vec.data[i];
        if (!isfinite(x->data[i])) {
            st = LMMC_STATUS_NUMERICAL_FAILURE;
            return st;
        }
    }

    for (i = 0; i < n; ++i) {
        s->w_prev.data[i] = s->w_curr.data[i];
        s->w_curr.data[i] = s->z_vec.data[i];
    }

    s->sn_prev = s->sn_curr;
    s->sn_curr = sn_new;
    s->cs_curr = cs_new;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_minres_iterate(const lmmc_sparse_mat_t* a,
    const lmmc_precond_t* precond, const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x, lmmc_minres_state_t* s, lmmc_itersolve_result_t* result,
    size_t* iterations, int* converged) {
    lmmc_minres_normalize(s);
    for (size_t iter = 0; iter < cfg->max_iter; ++iter) {
        lmmc_status_t st = lmmc_minres_lanczos(a, precond, cfg, s);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        st = lmmc_minres_rotate(x, s);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        s->norm_r = fabs(s->phi_bar);
        *iterations = iter + 1;
        if (result != NULL) {
            result->final_residual_norm = s->norm_r;
        }
        lmmc_itersolve_do_log(cfg, *iterations, s->norm_r);
        if (s->norm_r <= s->threshold) {
            *converged = 1;
            break;
        }
        if (s->beta_kp1 <= 1e-30) {
            break;
        }
        for (size_t i = 0; i < x->size; ++i) {
            s->v_prev.data[i] = s->v_curr.data[i];
            s->v_curr.data[i] = s->v_next.data[i] / s->beta_kp1;
        }
        s->beta_k = s->beta_kp1;
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
    lmmc_minres_state_t s = {0};
    lmmc_real_t* storage = NULL;
    size_t iterations = 0;
    int converged = 0;
    lmmc_itersolve_result_init(out_result);
    lmmc_vec_t* vectors[] = {&s.v_prev, &s.v_curr, &s.v_next, &s.w_prev,
        &s.w_curr, &s.av, &s.z_vec};
    const size_t n = b->size;
    const size_t sizes[] = {n, n, n, n, n, n, n};
    st = lmmc_itersolve_workspace(vectors, sizes, 7, &storage);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_minres_start(a, b, precond, &local, x, &s);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    lmmc_itersolve_initial(out_result, s.norm_r);
    lmmc_itersolve_do_log(&local, 0, s.norm_r);
    if (s.norm_r <= s.threshold || s.beta_k <= 1e-30) {
        converged = 1;
    }
    else {
        st = lmmc_minres_iterate(a, precond, &local, x, &s, out_result, &iterations, &converged);
    }
cleanup:
    lmmc_itersolve_finish(out_result, converged, iterations, s.norm_r);
    if (out_result != NULL) {
        out_result->final_residual_norm = s.norm_r;
    }
    lmmc_memory_free(storage);
    return st;
}
