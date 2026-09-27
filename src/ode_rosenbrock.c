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
        return lmmc_ode_values_are_finite(dfdt, dim);
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
    return lmmc_ode_values_are_finite(dfdt, dim);
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
    return lmmc_ode_values_are_finite(stage, dim);
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
    lmmc_real_t* k[ROS_STAGES];
    lmmc_real_t *f0, *f_stage, *y_stage, *y_pert, *f_pert;
    lmmc_real_t *rhs_vec, *dfdt, *y_new, *error;
    lmmc_mat_t W;
} ode_rosenbrock_state_t;

static lmmc_status_t ros_prepare_system(ode_rosenbrock_state_t* s) {
    size_t i, j, swap_count = 0;
    int callback_failed = 0;
    lmmc_status_t st;
        st = lmmc_ode_rhs_eval(s->rhs, s->t, s->y, s->f0, s->dim, s->user_data,
                               &s->out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            s->out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                                         : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            return st;
        }
        st = lmmc_ode_jacobian_eval(&(lmmc_ode_jacobian_request_t){
            s->rhs, s->local_cfg.jacobian, s->user_data, s->t, s->y, s->f0, s->dim, s->matrix_data,
            s->f_stage, s->y_pert, s->f_pert, &s->out_result->num_rhs_evals,
            &callback_failed});
        if (st != LMMC_STATUS_OK) {
            s->out_result->failure_reason =
                callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            return st;
        }
        st = ros_time_derivative(s->rhs, s->local_cfg.time_derivative, s->user_data, s->t, s->y,
                                 s->f0, s->dim, s->dfdt, s->f_pert, s->out_result);
        if (st != LMMC_STATUS_OK) {
            s->out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            return st;
        }

        for (i = 0; i < s->dim; ++i) {
            for (j = 0; j < s->dim; ++j) {
                s->W.data[i * s->dim + j] = -s->matrix_data[i * s->dim + j];
            }
            s->W.data[i * s->dim + i] += 1.0 / (s->h * ros_gamma);
        }
        st = lmmc_lu_decompose_inplace(&s->W, s->pivots, &swap_count);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    return LMMC_STATUS_OK;
}

static lmmc_status_t ros_evaluate_stages(ode_rosenbrock_state_t* s) {
    size_t i;
    int callback_failed = 0;
    lmmc_status_t st;
        for (i = 0; i < s->dim; ++i) {
            s->rhs_vec[i] = s->f0[i] + s->h * ros_d[0] * s->dfdt[i];
        }
        st = ros_solve_stage(&s->W, s->pivots, s->rhs_vec, s->k[0], s->dim);
        if (st != LMMC_STATUS_OK) { return st; }

        for (i = 0; i < s->dim; ++i) {
            s->y_stage[i] = s->y[i] + ros_a21 * s->k[0][i];
        }
        st = lmmc_ode_rhs_eval(s->rhs, s->t + ros_c2 * s->h, s->y_stage, s->f_stage, s->dim, s->user_data,
                               &s->out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            s->out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                                         : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            return st;
        }
        for (i = 0; i < s->dim; ++i) {
            s->rhs_vec[i] = s->f_stage[i] + s->h * ros_d[1] * s->dfdt[i] + (ros_c21 / s->h) * s->k[0][i];
        }
        st = ros_solve_stage(&s->W, s->pivots, s->rhs_vec, s->k[1], s->dim);
        if (st != LMMC_STATUS_OK) { return st; }

        for (i = 0; i < s->dim; ++i) {
            s->y_stage[i] = s->y[i] + ros_a31 * s->k[0][i] + ros_a32 * s->k[1][i];
        }
        st = lmmc_ode_rhs_eval(s->rhs, s->t + ros_c3 * s->h, s->y_stage, s->f_stage, s->dim, s->user_data,
                               &s->out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            s->out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                                         : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            return st;
        }
        for (i = 0; i < s->dim; ++i) {
            s->rhs_vec[i] = s->f_stage[i] + s->h * ros_d[2] * s->dfdt[i]
                       + (ros_c31 / s->h) * s->k[0][i] + (ros_c32 / s->h) * s->k[1][i];
        }
        st = ros_solve_stage(&s->W, s->pivots, s->rhs_vec, s->k[2], s->dim);
        if (st != LMMC_STATUS_OK) { return st; }

        for (i = 0; i < s->dim; ++i) {
            s->rhs_vec[i] = s->f_stage[i] + s->h * ros_d[3] * s->dfdt[i]
                       + (ros_c41 / s->h) * s->k[0][i] + (ros_c42 / s->h) * s->k[1][i]
                       + (ros_c43 / s->h) * s->k[2][i];
        }
        st = ros_solve_stage(&s->W, s->pivots, s->rhs_vec, s->k[3], s->dim);
        if (st != LMMC_STATUS_OK) { return st; }
    return LMMC_STATUS_OK;
}

static lmmc_status_t ros_estimate_error(ode_rosenbrock_state_t* s, lmmc_real_t* error_norm) {
    size_t i;
    lmmc_status_t st;
        for (i = 0; i < s->dim; ++i) {
            s->y_new[i] = s->y[i] + ros_b[0] * s->k[0][i] + ros_b[1] * s->k[1][i]
                     + ros_b[2] * s->k[2][i] + ros_b[3] * s->k[3][i];
            s->error[i] = ros_e[0] * s->k[0][i] + ros_e[1] * s->k[1][i]
                     + ros_e[2] * s->k[2][i] + ros_e[3] * s->k[3][i];
        }
        st = lmmc_ode_values_are_finite(s->y_new, s->dim);
        if (st != LMMC_STATUS_OK) { return st; }
        st = lmmc_ode_weighted_rms(s->error, s->y, s->y_new, s->dim, s->local_cfg.abs_tol,
                                    s->local_cfg.rel_tol, error_norm);
        if (st != LMMC_STATUS_OK) { return st; }
    return LMMC_STATUS_OK;
}

static lmmc_status_t ros_integrate(ode_rosenbrock_state_t* s, lmmc_real_t t_end) {
    size_t attempts = 0;
    lmmc_status_t st;
    while (s->t < t_end && attempts < s->local_cfg.max_steps) {
        lmmc_real_t error_norm = 0.0;
        ++attempts;
        if (s->h > t_end - s->t) {
            s->h = t_end - s->t;
        }
        st = ros_prepare_system(s);
        if (st != LMMC_STATUS_OK) { return st; }
        st = ros_evaluate_stages(s);
        if (st != LMMC_STATUS_OK) { return st; }
        st = ros_estimate_error(s, &error_norm);
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

static lmmc_status_t ros_cleanup(ode_rosenbrock_state_t* s, lmmc_status_t st) {
    if (st != LMMC_STATUS_OK && s->out_result != NULL &&
        s->out_result->failure_reason == LMMC_ODE_FAILURE_NONE) {
        s->out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
    }
    lmmc_memory_free(s->pivots);
    lmmc_memory_free(s->matrix_data);
    lmmc_memory_free(s->work);
    return st;
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
    ode_rosenbrock_state_t state = {0};
    ode_rosenbrock_state_t* s = &state;
    size_t vectors_bytes = 0, matrix_count = 0, matrix_bytes = 0;
    size_t i;
    lmmc_status_t st;
    s->rhs = rhs;
    s->user_data = user_data;
    s->dim = dim;
    s->y = y;
    s->out_result = out_result;
    s->t = t_start;
    st = validate_and_init_ode_config(s->rhs, s->dim, t_start, t_end, s->y, cfg,
                                      &s->local_cfg, s->out_result, &s->work_bytes);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (!lmmc_safe_mul_size(s->work_bytes, ROS_VECTORS, &vectors_bytes) ||
        !lmmc_safe_mul_size(s->dim, s->dim, &matrix_count) ||
        !lmmc_safe_mul_size(matrix_count, sizeof(lmmc_real_t), &matrix_bytes) ||
        !lmmc_safe_mul_size(s->dim, sizeof(size_t), &matrix_count)) {
        s->out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    s->work = (lmmc_real_t*)lmmc_memory_alloc(vectors_bytes);
    s->matrix_data = (lmmc_real_t*)lmmc_memory_alloc(matrix_bytes);
    s->pivots = (size_t*)lmmc_memory_alloc(matrix_count);
    if (s->work == NULL || s->matrix_data == NULL || s->pivots == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        return ros_cleanup(s, st);
    }

    for (i = 0; i < ROS_STAGES; ++i) {
        s->k[i] = s->work + i * s->dim;
    }
    s->f0 = s->work + 4 * s->dim;
    s->f_stage = s->work + 5 * s->dim;
    s->y_stage = s->work + 6 * s->dim;
    s->y_pert = s->work + 7 * s->dim;
    s->f_pert = s->work + 8 * s->dim;
    s->rhs_vec = s->work + 9 * s->dim;
    s->dfdt = s->work + 10 * s->dim;
    s->y_new = s->work + 11 * s->dim;
    s->error = s->work + 12 * s->dim;
    s->W = (lmmc_mat_t){s->dim, s->dim, s->dim, s->matrix_data, 0};

    s->h = lmmc_clamp(s->local_cfg.initial_step, s->local_cfg.min_step, s->local_cfg.max_step);
    if (s->h > t_end - t_start) {
        s->h = t_end - t_start;
    }
    lmmc_ode_do_log(&s->local_cfg, 0, s->t, s->y, s->dim);
    st = ros_integrate(s, t_end);
    return ros_cleanup(s, st);
}
