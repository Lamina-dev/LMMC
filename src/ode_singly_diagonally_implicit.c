/**
 * @file ode_singly_diagonally_implicit.c
 * @brief Hairer-Wanner SDIRK4(3) 五级嵌入式求解器.
 *
 * 系数和缺陷平滑对应 https://www.unige.ch/~hairer/prog/stiff/Oldies/sdirk4.f
 * 的 METH=1 路径. 阶段向量采用导数归一化.
 */
#include <math.h>
#include <string.h>

#include "memory_bridge.h"
#include "internal.h"
#include "ode_internal.h"
#include "lmmc/ode.h"
#include "lmmc/dense.h"
#include "lmmc/linear_algebra.h"

#define SDIRK_STAGES 5
#define SDIRK_VECTORS 16
#define SDIRK_NEWTON_MAX 12

static const lmmc_real_t sdirk_gamma = 0.25;
static const lmmc_real_t sdirk_c[SDIRK_STAGES] = {0.25, 0.75, 0.55, 0.5, 1.0};
static const lmmc_real_t sdirk_a[SDIRK_STAGES][SDIRK_STAGES] = {
    {0.25, 0.0, 0.0, 0.0, 0.0},
    {0.5, 0.25, 0.0, 0.0, 0.0},
    {0.34, -0.04, 0.25, 0.0, 0.0},
    {371.0 / 1360.0, -137.0 / 2720.0, 15.0 / 544.0, 0.25, 0.0},
    {25.0 / 24.0, -49.0 / 48.0, 125.0 / 16.0, -85.0 / 12.0, 0.25}
};
static const lmmc_real_t sdirk_b[SDIRK_STAGES] = {
    25.0 / 24.0, -49.0 / 48.0, 125.0 / 16.0, -85.0 / 12.0, 0.25
};
static const lmmc_real_t sdirk_bhat[SDIRK_STAGES] = {
    59.0 / 48.0, -17.0 / 96.0, 225.0 / 32.0, -85.0 / 12.0, 0.0
};


static lmmc_status_t sdirk_linear_solve(
    lmmc_mat_t* matrix,
    size_t* pivots,
    lmmc_real_t* rhs,
    lmmc_real_t* solution,
    size_t dim
) {
    size_t swaps = 0;
    const lmmc_vec_t rhs_vec = {dim, rhs, 0};
    lmmc_vec_t solution_vec = {dim, solution, 0};
    lmmc_status_t st = lmmc_lu_decompose_inplace(matrix, pivots, &swaps);
    if (st != LMMC_STATUS_OK) { return st; }
    st = lmmc_lu_solve(matrix, pivots, &rhs_vec, &solution_vec);
    if (st != LMMC_STATUS_OK) { return st; }
    return lmmc_ode_values_are_finite(solution, dim);
}

typedef struct {
    lmmc_ode_rhs_t rhs;
    void* user_data;
    size_t dim, work_bytes;
    lmmc_real_t* y;
    lmmc_ode_config_t local_cfg;
    lmmc_ode_result_t* out_result;
    lmmc_real_t t, h;
    lmmc_real_t* work;
    lmmc_real_t* matrix_data;
    size_t* pivots;
    lmmc_real_t* k[SDIRK_STAGES];
    lmmc_real_t *rhs_sum, *y_stage, *f_stage, *y_pert, *f_pert;
    lmmc_real_t *residual, *delta, *y_new, *y_hat, *error, *smoothed;
    lmmc_mat_t matrix;
} ode_sdirk_state_t;

static lmmc_status_t sdirk_newton_correction(ode_sdirk_state_t* s, size_t stage) {
    size_t i, j;
    int callback_failed = 0;
    lmmc_status_t st;
                st = lmmc_ode_jacobian_eval(&(lmmc_ode_jacobian_request_t){
                    s->rhs, s->local_cfg.jacobian, s->user_data,
                    s->t + sdirk_c[stage] * s->h, s->y_stage, s->f_stage, s->dim,
                    s->matrix_data, s->rhs_sum, s->y_pert, s->f_pert,
                    &s->out_result->num_rhs_evals, &callback_failed});
                if (st != LMMC_STATUS_OK) {
                    s->out_result->failure_reason =
                        callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                        : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                    return st;
                }
                for (i = 0; i < s->dim; ++i) {
                    for (j = 0; j < s->dim; ++j) {
                        s->matrix.data[i * s->dim + j] = -s->h * sdirk_gamma * s->matrix_data[i * s->dim + j];
                    }
                    s->matrix.data[i * s->dim + i] += 1.0;
                    s->residual[i] = -s->residual[i];
                }
                st = sdirk_linear_solve(&s->matrix, s->pivots, s->residual, s->delta, s->dim);
                if (st != LMMC_STATUS_OK) { return st; }
                for (i = 0; i < s->dim; ++i) s->k[stage][i] += s->delta[i];
    return LMMC_STATUS_OK;
}

static lmmc_status_t sdirk_stage_norm(
    ode_sdirk_state_t* s, const lmmc_real_t* derivative, lmmc_real_t* out_norm)
{
    size_t i;
    int zero_scale = 0;
    lmmc_status_t st;
    for (i = 0; i < s->dim; ++i) s->error[i] = s->h * derivative[i];
    st = lmmc_ode_weighted_rms(s->error, s->y, s->y_stage, s->dim,
                             s->local_cfg.abs_tol, s->local_cfg.rel_tol, out_norm);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) return st;
    for (i = 0; i < s->dim; ++i) {
        const lmmc_real_t scale = s->local_cfg.abs_tol +
            s->local_cfg.rel_tol * fmax(fabs(s->y[i]), fabs(s->y_stage[i]));
        if (!isfinite(scale) || !isfinite(s->error[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (scale == 0.0) zero_scale = 1;
    }
    /** @brief 中间修正的相对尺度为零时继续 Newton 修正，待尺度有效后判断收敛。 */
    if (zero_scale) {
        *out_norm = INFINITY;
        return LMMC_STATUS_OK;
    }
    return st;
}

static lmmc_status_t sdirk_stage_residual(
    ode_sdirk_state_t* s, size_t stage, int* residual_zero)
{
    size_t i;
    int callback_failed = 0;
    lmmc_status_t st;
    for (i = 0; i < s->dim; ++i) {
        s->y_stage[i] = s->y[i] + s->h * (s->rhs_sum[i] + sdirk_gamma * s->k[stage][i]);
        if (!isfinite(s->y_stage[i])) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    }
    st = lmmc_ode_rhs_eval(s->rhs, s->t + sdirk_c[stage] * s->h, s->y_stage, s->f_stage,
                         s->dim, s->user_data, &s->out_result->num_rhs_evals, &callback_failed);
    if (st != LMMC_STATUS_OK) {
        s->out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                                       : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return st;
    }
    *residual_zero = 1;
    for (i = 0; i < s->dim; ++i) {
        s->residual[i] = s->k[stage][i] - s->f_stage[i];
        if (!isfinite(s->residual[i])) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        if (s->residual[i] != 0.0) { *residual_zero = 0; }
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t sdirk_solve_stage(ode_sdirk_state_t* s, size_t stage) {
    const lmmc_real_t eta = 0.001;
    size_t i, j, corrections = 0;
    lmmc_status_t st;
    memset(s->rhs_sum, 0, s->work_bytes);
    for (j = 0; j < stage; ++j) {
        for (i = 0; i < s->dim; ++i) {
            s->rhs_sum[i] += sdirk_a[stage][j] * s->k[j][i];
        }
    }
    if (stage == 0) { memset(s->k[stage], 0, s->work_bytes); }
    else { memcpy(s->k[stage], s->k[stage - 1], s->work_bytes); }

    for (;;) {
        int residual_zero;
        st = sdirk_stage_residual(s, stage, &residual_zero);
        if (st != LMMC_STATUS_OK) { return st; }
        if (residual_zero) { return LMMC_STATUS_OK; }
        /** @brief 每次修正后重查残差和修正量范数，包括最后一次修正。 */
        if (corrections != 0) {
            lmmc_real_t residual_norm, correction_norm;
            st = sdirk_stage_norm(s, s->residual, &residual_norm);
            if (st != LMMC_STATUS_OK) { return st; }
            st = sdirk_stage_norm(s, s->delta, &correction_norm);
            if (st != LMMC_STATUS_OK) { return st; }
            if (residual_norm <= eta && correction_norm <= eta) { return LMMC_STATUS_OK; }
        }
        if (corrections == SDIRK_NEWTON_MAX) {
            s->out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_CONVERGENCE_FAILED;
        }
        st = sdirk_newton_correction(s, stage);
        if (st != LMMC_STATUS_OK) { return st; }
        ++corrections;
    }
}

static lmmc_status_t sdirk_estimate_error(ode_sdirk_state_t* s, lmmc_real_t* error_norm) {
    size_t i, j, stage;
    int callback_failed = 0;
    lmmc_status_t st;
        for (i = 0; i < s->dim; ++i) {
            s->y_new[i] = s->y[i];
            s->y_hat[i] = s->y[i];
            for (stage = 0; stage < SDIRK_STAGES; ++stage) {
                s->y_new[i] += s->h * sdirk_b[stage] * s->k[stage][i];
                s->y_hat[i] += s->h * sdirk_bhat[stage] * s->k[stage][i];
            }
            s->error[i] = s->y_new[i] - s->y_hat[i];
        }
        if (lmmc_ode_values_are_finite(s->y_new, s->dim) != LMMC_STATUS_OK) {
            st = LMMC_STATUS_NUMERICAL_FAILURE;
            return st;
        }

        st = lmmc_ode_rhs_eval(s->rhs, s->t + s->h, s->y_new, s->f_stage, s->dim, s->user_data,
                               &s->out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            s->out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                                         : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            return st;
        }
        st = lmmc_ode_jacobian_eval(&(lmmc_ode_jacobian_request_t){
            s->rhs, s->local_cfg.jacobian, s->user_data, s->t + s->h, s->y_new, s->f_stage, s->dim,
            s->matrix_data, s->rhs_sum, s->y_pert, s->f_pert,
            &s->out_result->num_rhs_evals, &callback_failed});
        if (st != LMMC_STATUS_OK) {
            s->out_result->failure_reason =
                callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            return st;
        }
        for (i = 0; i < s->dim; ++i) {
            for (j = 0; j < s->dim; ++j) {
                s->matrix.data[i * s->dim + j] = -s->h * sdirk_gamma * s->matrix_data[i * s->dim + j];
            }
            s->matrix.data[i * s->dim + i] += 1.0;
        }
        st = sdirk_linear_solve(&s->matrix, s->pivots, s->error, s->smoothed, s->dim);
        if (st != LMMC_STATUS_OK) { return st; }
        st = lmmc_ode_weighted_rms(s->smoothed, s->y, s->y_new, s->dim, s->local_cfg.abs_tol,
                                    s->local_cfg.rel_tol, error_norm);
        if (st != LMMC_STATUS_OK) { return st; }
    return LMMC_STATUS_OK;
}

static lmmc_status_t sdirk_integrate(ode_sdirk_state_t* s, lmmc_real_t t_end) {
    size_t attempts = 0;
    lmmc_status_t st;
    while (s->t < t_end && attempts < s->local_cfg.max_steps) {
        lmmc_real_t error_norm = 0.0;
        ++attempts;
        if (s->h > t_end - s->t) s->h = t_end - s->t;
        for (size_t stage = 0; stage < SDIRK_STAGES; ++stage) {
            st = sdirk_solve_stage(s, stage);
            if (st != LMMC_STATUS_OK) { return st; }
        }
        st = sdirk_estimate_error(s, &error_norm);
        if (st != LMMC_STATUS_OK) { return st; }
        if (error_norm <= 1.0) {
            memcpy(s->y, s->y_new, s->work_bytes);
            s->t += s->h;
            s->out_result->num_steps += 1;
            s->out_result->final_t = s->t;
            lmmc_ode_do_log(&s->local_cfg, s->out_result->num_steps, s->t, s->y, s->dim);
        } else if (s->h <= s->local_cfg.min_step) {
            st = LMMC_STATUS_CONVERGENCE_FAILED;
            s->out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
            return st;
        }
        s->h = lmmc_ode_next_step(s->h, error_norm, &s->local_cfg);
    }

    if (s->t >= t_end) {
        s->out_result->converged = 1;
        s->out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
        st = LMMC_STATUS_OK;
    } else {
        s->out_result->failure_reason = LMMC_ODE_FAILURE_MAX_STEPS;
        st = LMMC_STATUS_CONVERGENCE_FAILED;
    }
    return st;
}

static lmmc_status_t sdirk_cleanup(ode_sdirk_state_t* s, lmmc_status_t st) {
    if (st != LMMC_STATUS_OK && s->out_result != NULL &&
        s->out_result->failure_reason == LMMC_ODE_FAILURE_NONE) {
        s->out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
    }
    lmmc_memory_free(s->pivots);
    lmmc_memory_free(s->matrix_data);
    lmmc_memory_free(s->work);
    return st;
}

lmmc_status_t lmmc_ode_sdirk4_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    lmmc_real_t* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
) {
    ode_sdirk_state_t state = {0};
    ode_sdirk_state_t* s = &state;
    size_t vectors_bytes = 0, matrix_count = 0, matrix_bytes = 0, pivot_bytes = 0;
    size_t stage;
    lmmc_status_t st;
    s->rhs = rhs;
    s->user_data = user_data;
    s->dim = dim;
    s->y = y;
    s->out_result = out_result;
    s->t = t_start;
    st = validate_and_init_ode_config(s->rhs, s->dim, t_start, t_end, s->y, cfg,
                                      &s->local_cfg, s->out_result, &s->work_bytes);
    if (st != LMMC_STATUS_OK) { return st; }
    if (!lmmc_safe_mul_size(s->work_bytes, SDIRK_VECTORS, &vectors_bytes) ||
        !lmmc_safe_mul_size(s->dim, s->dim, &matrix_count) ||
        !lmmc_safe_mul_size(matrix_count, sizeof(lmmc_real_t), &matrix_bytes) ||
        !lmmc_safe_mul_size(s->dim, sizeof(size_t), &pivot_bytes)) {
        s->out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    s->work = (lmmc_real_t*)lmmc_memory_alloc(vectors_bytes);
    s->matrix_data = (lmmc_real_t*)lmmc_memory_alloc(matrix_bytes);
    s->pivots = (size_t*)lmmc_memory_alloc(pivot_bytes);
    if (s->work == NULL || s->matrix_data == NULL || s->pivots == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        return sdirk_cleanup(s, st);
    }
    for (stage = 0; stage < SDIRK_STAGES; ++stage) s->k[stage] = s->work + stage * s->dim;
    s->rhs_sum = s->work + 5 * s->dim;
    s->y_stage = s->work + 6 * s->dim;
    s->f_stage = s->work + 7 * s->dim;
    s->y_pert = s->work + 8 * s->dim;
    s->f_pert = s->work + 9 * s->dim;
    s->residual = s->work + 10 * s->dim;
    s->delta = s->work + 11 * s->dim;
    s->y_new = s->work + 12 * s->dim;
    s->y_hat = s->work + 13 * s->dim;
    s->error = s->work + 14 * s->dim;
    s->smoothed = s->work + 15 * s->dim;
    s->matrix = (lmmc_mat_t){s->dim, s->dim, s->dim, s->matrix_data, 0};

    s->h = lmmc_clamp(s->local_cfg.initial_step, s->local_cfg.min_step, s->local_cfg.max_step);
    if (s->h > t_end - t_start) s->h = t_end - t_start;
    lmmc_ode_do_log(&s->local_cfg, 0, s->t, s->y, s->dim);
    st = sdirk_integrate(s, t_end);
    return sdirk_cleanup(s, st);
}
