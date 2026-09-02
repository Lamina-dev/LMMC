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
    if (st != LMMC_STATUS_OK) return st;
    st = lmmc_lu_solve(matrix, pivots, &rhs_vec, &solution_vec);
    if (st != LMMC_STATUS_OK) return st;
    return lmmc_ode_values_are_finite(solution, dim);
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
    lmmc_ode_config_t local_cfg = {0};
    size_t work_bytes = 0, vectors_bytes = 0, matrix_count = 0, matrix_bytes = 0, pivot_bytes = 0;
    lmmc_real_t* work = NULL;
    lmmc_real_t* matrix_data = NULL;
    size_t* pivots = NULL;
    lmmc_real_t* k[SDIRK_STAGES];
    lmmc_real_t *rhs_sum, *y_stage, *f_stage, *y_pert, *f_pert;
    lmmc_real_t *residual, *delta, *y_new, *y_hat, *error, *smoothed;
    lmmc_mat_t matrix = {0};
    lmmc_real_t t = t_start, h;
    size_t attempts = 0, i, j, stage;
    lmmc_status_t st;

    st = validate_and_init_ode_config(rhs, dim, t_start, t_end, y, cfg,
                                      &local_cfg, out_result, &work_bytes);
    if (st != LMMC_STATUS_OK) return st;
    if (!lmmc_safe_mul_size(work_bytes, SDIRK_VECTORS, &vectors_bytes) ||
        !lmmc_safe_mul_size(dim, dim, &matrix_count) ||
        !lmmc_safe_mul_size(matrix_count, sizeof(lmmc_real_t), &matrix_bytes) ||
        !lmmc_safe_mul_size(dim, sizeof(size_t), &pivot_bytes)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    work = (lmmc_real_t*)lmmc_alloc(vectors_bytes);
    matrix_data = (lmmc_real_t*)lmmc_alloc(matrix_bytes);
    pivots = (size_t*)lmmc_alloc(pivot_bytes);
    if (work == NULL || matrix_data == NULL || pivots == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    for (stage = 0; stage < SDIRK_STAGES; ++stage) k[stage] = work + stage * dim;
    rhs_sum = work + 5 * dim;
    y_stage = work + 6 * dim;
    f_stage = work + 7 * dim;
    y_pert = work + 8 * dim;
    f_pert = work + 9 * dim;
    residual = work + 10 * dim;
    delta = work + 11 * dim;
    y_new = work + 12 * dim;
    y_hat = work + 13 * dim;
    error = work + 14 * dim;
    smoothed = work + 15 * dim;
    matrix = (lmmc_mat_t){dim, dim, dim, matrix_data, 0};

    h = lmmc_clamp(local_cfg.initial_step, local_cfg.min_step, local_cfg.max_step);
    if (h > t_end - t_start) h = t_end - t_start;
    lmmc_ode_do_log(&local_cfg, 0, t, y, dim);

    while (t < t_end && attempts < local_cfg.max_steps) {
        lmmc_real_t error_norm = 0.0;
        int callback_failed = 0;
        ++attempts;
        if (h > t_end - t) h = t_end - t;

        for (stage = 0; stage < SDIRK_STAGES; ++stage) {
            size_t iteration;
            int converged = 0;
            memset(rhs_sum, 0, work_bytes);
            for (j = 0; j < stage; ++j) {
                for (i = 0; i < dim; ++i) {
                    rhs_sum[i] += sdirk_a[stage][j] * k[j][i];
                }
            }
            if (stage == 0) memset(k[stage], 0, work_bytes);
            else memcpy(k[stage], k[stage - 1], work_bytes);

            for (iteration = 0; iteration < SDIRK_NEWTON_MAX; ++iteration) {
                lmmc_real_t residual_max = 0.0;
                lmmc_real_t state_scale = 0.0;
                for (i = 0; i < dim; ++i) {
                    y_stage[i] = y[i] + h * (rhs_sum[i] + sdirk_gamma * k[stage][i]);
                    state_scale = fmax(state_scale, fabs(k[stage][i]));
                }
                st = lmmc_ode_rhs_eval(rhs, t + sdirk_c[stage] * h, y_stage, f_stage,
                                       dim, user_data, &out_result->num_rhs_evals,
                                       &callback_failed);
                if (st != LMMC_STATUS_OK) {
                    out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                                                 : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                    goto cleanup;
                }
                for (i = 0; i < dim; ++i) {
                    residual[i] = k[stage][i] - f_stage[i];
                    residual_max = fmax(residual_max, fabs(residual[i]));
                }
                if (residual_max <= 1.0e-12 * fmax(1.0, state_scale)) {
                    converged = 1;
                    break;
                }

                st = lmmc_ode_jacobian_eval(
                    rhs, local_cfg.jacobian, user_data,
                    t + sdirk_c[stage] * h, y_stage, f_stage, dim,
                    matrix_data, rhs_sum, y_pert, f_pert,
                    &out_result->num_rhs_evals, &callback_failed);
                if (st != LMMC_STATUS_OK) {
                    out_result->failure_reason =
                        callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                        : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                    goto cleanup;
                }
                for (i = 0; i < dim; ++i) {
                    for (j = 0; j < dim; ++j) {
                        matrix.data[i * dim + j] = -h * sdirk_gamma * matrix_data[i * dim + j];
                    }
                    matrix.data[i * dim + i] += 1.0;
                    residual[i] = -residual[i];
                }
                st = sdirk_linear_solve(&matrix, pivots, residual, delta, dim);
                if (st != LMMC_STATUS_OK) goto cleanup;
                for (i = 0; i < dim; ++i) k[stage][i] += delta[i];
            }
            if (!converged) {
                st = LMMC_STATUS_CONVERGENCE_FAILED;
                out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto cleanup;
            }
        }

        for (i = 0; i < dim; ++i) {
            y_new[i] = y[i];
            y_hat[i] = y[i];
            for (stage = 0; stage < SDIRK_STAGES; ++stage) {
                y_new[i] += h * sdirk_b[stage] * k[stage][i];
                y_hat[i] += h * sdirk_bhat[stage] * k[stage][i];
            }
            error[i] = y_new[i] - y_hat[i];
        }
        if (lmmc_ode_values_are_finite(y_new, dim) != LMMC_STATUS_OK) {
            st = LMMC_STATUS_NUMERICAL_FAILURE;
            goto cleanup;
        }

        st = lmmc_ode_rhs_eval(rhs, t + h, y_new, f_stage, dim, user_data,
                               &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                                         : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto cleanup;
        }
        st = lmmc_ode_jacobian_eval(
            rhs, local_cfg.jacobian, user_data, t + h, y_new, f_stage, dim,
            matrix_data, rhs_sum, y_pert, f_pert,
            &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason =
                callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto cleanup;
        }
        for (i = 0; i < dim; ++i) {
            for (j = 0; j < dim; ++j) {
                matrix.data[i * dim + j] = -h * sdirk_gamma * matrix_data[i * dim + j];
            }
            matrix.data[i * dim + i] += 1.0;
        }
        st = sdirk_linear_solve(&matrix, pivots, error, smoothed, dim);
        if (st != LMMC_STATUS_OK) goto cleanup;
        st = lmmc_ode_weighted_rms(smoothed, y, y_new, dim, local_cfg.abs_tol,
                                    local_cfg.rel_tol, &error_norm);
        if (st != LMMC_STATUS_OK) goto cleanup;

        if (error_norm <= 1.0) {
            memcpy(y, y_new, work_bytes);
            t += h;
            out_result->num_steps += 1;
            out_result->final_t = t;
            lmmc_ode_do_log(&local_cfg, out_result->num_steps, t, y, dim);
        } else if (h <= local_cfg.min_step) {
            st = LMMC_STATUS_CONVERGENCE_FAILED;
            out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
            goto cleanup;
        }
        h = lmmc_ode_next_step(h, error_norm, &local_cfg);
    }

    if (t >= t_end) {
        out_result->converged = 1;
        out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
        st = LMMC_STATUS_OK;
    } else {
        out_result->failure_reason = LMMC_ODE_FAILURE_MAX_STEPS;
        st = LMMC_STATUS_CONVERGENCE_FAILED;
    }

cleanup:
    if (st != LMMC_STATUS_OK && out_result != NULL &&
        out_result->failure_reason == LMMC_ODE_FAILURE_NONE) {
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
    }
    lmmc_free(pivots);
    lmmc_free(matrix_data);
    lmmc_free(work);
    return st;
}
