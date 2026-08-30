/**
 * @file ode_rosenbrock.c
 * @brief Hairer ROS4 中 Kaps-Rentrop GRK4T 四阶 Rosenbrock-Wanner 求解器.
 *
 * 系数和阶段归一化对应 https://www.unige.ch/~hairer/prog/stiff/Oldies/ros4.f
 * 的 GRK4T 路径. 每次尝试只分解一次 W=I/(h*gamma)-J.
 */
#include <math.h>
#include <string.h>

#include "memory_bridge.h"
#include "internal.h"
#include "ode_internal.h"
#include "lmmc/ode.h"
#include "lmmc/dense.h"
#include "lmmc/linear_algebra.h"

#define ROS_STAGES 4
#define ROS_VECTORS 13

static const lmmc_real_t ros_gamma = 0.231;
static const lmmc_real_t ros_a21 = 2.0;
static const lmmc_real_t ros_a31 = 4.524708207373116;
static const lmmc_real_t ros_a32 = 4.163528788597648;
static const lmmc_real_t ros_c21 = -5.071675338776316;
static const lmmc_real_t ros_c31 = 6.020152728650786;
static const lmmc_real_t ros_c32 = 0.1597506846727117;
static const lmmc_real_t ros_c41 = -1.856343618686113;
static const lmmc_real_t ros_c42 = -8.505380858179826;
static const lmmc_real_t ros_c43 = -2.084075136023187;
static const lmmc_real_t ros_b[ROS_STAGES] = {
    3.957503746640777, 4.624892388363313,
    0.6174772638750108, 1.282612945269037
};
static const lmmc_real_t ros_e[ROS_STAGES] = {
    2.302155402932996, 3.073634485392623,
    -0.8732808018045032, -1.282612945269037
};
static const lmmc_real_t ros_d[ROS_STAGES] = {
    0.231, -0.03962966775244303,
    0.5507789395789127, -0.05535098457052764
};
static const lmmc_real_t ros_c2 = 0.462;
static const lmmc_real_t ros_c3 = 0.8802083333333334;

static lmmc_status_t ros_check_finite(const lmmc_real_t* values, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t ros_jacobian(
    lmmc_ode_rhs_t rhs,
    lmmc_ode_jac_t jacobian,
    void* user_data,
    lmmc_real_t t,
    const lmmc_real_t* y,
    const lmmc_real_t* f0,
    size_t dim,
    lmmc_real_t* jac,
    lmmc_real_t* y_pert,
    lmmc_real_t* f_pert,
    lmmc_ode_result_t* result
) {
    size_t i, j;
    lmmc_status_t st;
    int callback_failed = 0;

    if (jacobian != NULL) {
        st = jacobian(t, y, jac, dim, user_data);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        return ros_check_finite(jac, dim * dim);
    }

    for (j = 0; j < dim; ++j) {
        lmmc_real_t delta = sqrt(2.2204460492503131e-16) * fmax(1.0, fabs(y[j]));
        memcpy(y_pert, y, dim * sizeof(*y));
        y_pert[j] += delta;
        st = lmmc_ode_rhs_eval(rhs, t, y_pert, f_pert, dim, user_data,
                               &result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        for (i = 0; i < dim; ++i) {
            jac[i * dim + j] = (f_pert[i] - f0[i]) / delta;
        }
    }
    return ros_check_finite(jac, dim * dim);
}

static lmmc_status_t ros_time_derivative(
    lmmc_ode_rhs_t rhs,
    lmmc_ode_dfdt_t time_derivative,
    void* user_data,
    lmmc_real_t t,
    const lmmc_real_t* y,
    const lmmc_real_t* f0,
    size_t dim,
    lmmc_real_t* dfdt,
    lmmc_real_t* f_pert,
    lmmc_ode_result_t* result
) {
    size_t i;
    lmmc_status_t st;
    int callback_failed = 0;

    if (time_derivative != NULL) {
        st = time_derivative(t, y, dfdt, dim, user_data);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        return ros_check_finite(dfdt, dim);
    }

    {
        lmmc_real_t delta = sqrt(2.2204460492503131e-16) * fmax(1.0, fabs(t));
        st = lmmc_ode_rhs_eval(rhs, t + delta, y, f_pert, dim, user_data,
                               &result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        for (i = 0; i < dim; ++i) {
            dfdt[i] = (f_pert[i] - f0[i]) / delta;
        }
    }
    return ros_check_finite(dfdt, dim);
}

static lmmc_status_t ros_solve_stage(
    const lmmc_mat_t* lu,
    const size_t* pivots,
    const lmmc_real_t* rhs,
    lmmc_real_t* stage,
    size_t dim
) {
    const lmmc_vec_t rhs_vec = {dim, (lmmc_real_t*)rhs, 0};
    lmmc_vec_t stage_vec = {dim, stage, 0};
    lmmc_status_t st = lmmc_lu_solve(lu, pivots, &rhs_vec, &stage_vec);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return ros_check_finite(stage, dim);
}

lmmc_status_t lmmc_ode_rosenbrock_grk4t_solve(
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
    size_t work_bytes = 0, vectors_bytes = 0, matrix_count = 0, matrix_bytes = 0;
    lmmc_real_t* work = NULL;
    lmmc_real_t* matrix_data = NULL;
    size_t* pivots = NULL;
    lmmc_real_t* k[ROS_STAGES];
    lmmc_real_t *f0, *f_stage, *y_stage, *y_pert, *f_pert;
    lmmc_real_t *rhs_vec, *dfdt, *y_new, *error;
    lmmc_mat_t W = {0};
    lmmc_real_t t = t_start, h;
    size_t attempts = 0, i, j;
    lmmc_status_t st;

    st = validate_and_init_ode_config(rhs, dim, t_start, t_end, y, cfg,
                                      &local_cfg, out_result, &work_bytes);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (!lmmc_safe_mul_size(work_bytes, ROS_VECTORS, &vectors_bytes) ||
        !lmmc_safe_mul_size(dim, dim, &matrix_count) ||
        !lmmc_safe_mul_size(matrix_count, sizeof(lmmc_real_t), &matrix_bytes) ||
        !lmmc_safe_mul_size(dim, sizeof(size_t), &matrix_count)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    work = (lmmc_real_t*)lmmc_alloc(vectors_bytes);
    matrix_data = (lmmc_real_t*)lmmc_alloc(matrix_bytes);
    pivots = (size_t*)lmmc_alloc(matrix_count);
    if (work == NULL || matrix_data == NULL || pivots == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }

    for (i = 0; i < ROS_STAGES; ++i) {
        k[i] = work + i * dim;
    }
    f0 = work + 4 * dim;
    f_stage = work + 5 * dim;
    y_stage = work + 6 * dim;
    y_pert = work + 7 * dim;
    f_pert = work + 8 * dim;
    rhs_vec = work + 9 * dim;
    dfdt = work + 10 * dim;
    y_new = work + 11 * dim;
    error = work + 12 * dim;
    W = (lmmc_mat_t){dim, dim, dim, matrix_data, 0};

    h = lmmc_clamp(local_cfg.initial_step, local_cfg.min_step, local_cfg.max_step);
    if (h > t_end - t_start) {
        h = t_end - t_start;
    }
    lmmc_ode_do_log(&local_cfg, 0, t, y, dim);

    while (t < t_end && attempts < local_cfg.max_steps) {
        lmmc_real_t error_norm = 0.0;
        size_t swap_count = 0;
        int callback_failed = 0;
        ++attempts;
        if (h > t_end - t) {
            h = t_end - t;
        }

        st = lmmc_ode_rhs_eval(rhs, t, y, f0, dim, user_data,
                               &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                                         : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto cleanup;
        }
        st = ros_jacobian(rhs, local_cfg.jacobian, user_data, t, y, f0, dim,
                          matrix_data, y_pert, f_pert, out_result);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto cleanup;
        }
        st = ros_time_derivative(rhs, local_cfg.time_derivative, user_data, t, y,
                                 f0, dim, dfdt, f_pert, out_result);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto cleanup;
        }

        for (i = 0; i < dim; ++i) {
            for (j = 0; j < dim; ++j) {
                W.data[i * dim + j] = -matrix_data[i * dim + j];
            }
            W.data[i * dim + i] += 1.0 / (h * ros_gamma);
        }
        st = lmmc_lu_decompose_inplace(&W, pivots, &swap_count);
        if (st != LMMC_STATUS_OK) {
            goto cleanup;
        }

        for (i = 0; i < dim; ++i) {
            rhs_vec[i] = f0[i] + h * ros_d[0] * dfdt[i];
        }
        st = ros_solve_stage(&W, pivots, rhs_vec, k[0], dim);
        if (st != LMMC_STATUS_OK) goto cleanup;

        for (i = 0; i < dim; ++i) {
            y_stage[i] = y[i] + ros_a21 * k[0][i];
        }
        st = lmmc_ode_rhs_eval(rhs, t + ros_c2 * h, y_stage, f_stage, dim, user_data,
                               &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                                         : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto cleanup;
        }
        for (i = 0; i < dim; ++i) {
            rhs_vec[i] = f_stage[i] + h * ros_d[1] * dfdt[i] + (ros_c21 / h) * k[0][i];
        }
        st = ros_solve_stage(&W, pivots, rhs_vec, k[1], dim);
        if (st != LMMC_STATUS_OK) goto cleanup;

        for (i = 0; i < dim; ++i) {
            y_stage[i] = y[i] + ros_a31 * k[0][i] + ros_a32 * k[1][i];
        }
        st = lmmc_ode_rhs_eval(rhs, t + ros_c3 * h, y_stage, f_stage, dim, user_data,
                               &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                                         : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto cleanup;
        }
        for (i = 0; i < dim; ++i) {
            rhs_vec[i] = f_stage[i] + h * ros_d[2] * dfdt[i]
                       + (ros_c31 / h) * k[0][i] + (ros_c32 / h) * k[1][i];
        }
        st = ros_solve_stage(&W, pivots, rhs_vec, k[2], dim);
        if (st != LMMC_STATUS_OK) goto cleanup;

        for (i = 0; i < dim; ++i) {
            rhs_vec[i] = f_stage[i] + h * ros_d[3] * dfdt[i]
                       + (ros_c41 / h) * k[0][i] + (ros_c42 / h) * k[1][i]
                       + (ros_c43 / h) * k[2][i];
        }
        st = ros_solve_stage(&W, pivots, rhs_vec, k[3], dim);
        if (st != LMMC_STATUS_OK) goto cleanup;

        for (i = 0; i < dim; ++i) {
            y_new[i] = y[i] + ros_b[0] * k[0][i] + ros_b[1] * k[1][i]
                     + ros_b[2] * k[2][i] + ros_b[3] * k[3][i];
            error[i] = ros_e[0] * k[0][i] + ros_e[1] * k[1][i]
                     + ros_e[2] * k[2][i] + ros_e[3] * k[3][i];
        }
        st = ros_check_finite(y_new, dim);
        if (st != LMMC_STATUS_OK) goto cleanup;
        st = lmmc_ode_weighted_rms(error, y, y_new, dim, local_cfg.abs_tol,
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
