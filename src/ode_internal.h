/**
 * @file ode_internal.h
 * @brief ODE 求解器内部共享辅助函数（仅源文件可见）。
 *
 * 提供日志记录、结果记账、配置加载校验、RHS 求值与 Jacobian
 * 计算等内联辅助，供各 ODE 求解器实现文件共用。
 *
 * @internal
 */
#ifndef LMMC_ODE_INTERNAL_H
#define LMMC_ODE_INTERNAL_H

#include <math.h>
#include <string.h>

#include "internal.h"
#include "lmmc/ode.h"
#include "lmmc/diagnostic.h"

static inline void lmmc_ode_do_log(const lmmc_ode_config_t* cfg, size_t step, lmmc_real_t t, const lmmc_real_t* y, size_t dim) {
    const lmmc_real_t values[] = {t, dim > 0 ? y[0] : 0.0};
    const lmmc_diagnostic_t diagnostic = {
        LMMC_DIAGNOSTIC_TRACE, "ode", "accepted step", step, values, 2
    };
    lmmc_diagnostic_emit(&cfg->diagnostics, &diagnostic);
}

static inline void lmmc_ode_reset_result(lmmc_ode_result_t* out_result, lmmc_real_t t_start) {
    if (out_result == NULL) {
        return;
    }

    out_result->converged = 0;
    out_result->num_steps = 0;
    out_result->num_rhs_evals = 0;
    out_result->final_t = t_start;
    out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
}

static inline lmmc_status_t lmmc_ode_load_and_validate_config(
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_config_t* out_cfg,
    lmmc_ode_result_t* out_result
) {
    if (dim == 0) {
        if (out_result != NULL) {
            out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_is_finite(&t_start) || !lmmc_is_finite(&t_end) || !(t_end > t_start)) {
        if (out_result != NULL) {
            out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
        }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (out_cfg == NULL) {
        if (out_result != NULL) {
            out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
        }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (cfg != NULL) {
        *out_cfg = *cfg;
    } else {
        lmmc_status_t st = lmmc_ode_default_config(t_start, t_end, dim, out_cfg);
        if (st != LMMC_STATUS_OK) {
            if (out_result != NULL) {
                out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
            }
            return st;
        }
    }

    if (!lmmc_is_finite(&out_cfg->abs_tol) || !lmmc_is_finite(&out_cfg->rel_tol) ||
        out_cfg->abs_tol < 0.0 || out_cfg->rel_tol < 0.0 ||
        (out_cfg->abs_tol == 0.0 && out_cfg->rel_tol == 0.0)) {
        if (out_result != NULL) {
            out_result->failure_reason = LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT;
        }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_is_finite(&out_cfg->initial_step) || !lmmc_is_finite(&out_cfg->min_step) ||
        !lmmc_is_finite(&out_cfg->max_step) || out_cfg->initial_step <= 0.0 ||
        out_cfg->min_step <= 0.0 || out_cfg->max_step <= 0.0 || out_cfg->min_step > out_cfg->max_step ||
        out_cfg->max_steps == 0 || !lmmc_is_finite(&out_cfg->adaptive_step_beta) ||
        out_cfg->adaptive_step_beta <= 0.0 || out_cfg->adaptive_step_beta > 1.0) {
        if (out_result != NULL) {
            out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
        }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_ode_rhs_eval(
    lmmc_ode_rhs_t rhs,
    lmmc_real_t t,
    const lmmc_real_t* y,
    lmmc_real_t* y_prime,
    size_t dim,
    void* user_data,
    size_t* io_eval_count,
    int* out_callback_failed
) {
    size_t i = 0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (rhs == NULL || y == NULL || y_prime == NULL || io_eval_count == NULL || out_callback_failed == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    *out_callback_failed = 0;
    *io_eval_count += 1;
    st = rhs(t, y, y_prime, dim, user_data);
    if (st != LMMC_STATUS_OK) {
        *out_callback_failed = 1;
        return st;
    }
    for (i = 0; i < dim; ++i) {
        if (!lmmc_is_finite(&y_prime[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    return LMMC_STATUS_OK;
}

static inline int lmmc_ode_state_is_finite(const lmmc_real_t* y, size_t dim) {
    size_t i = 0;
    if (y == NULL || dim == 0) {
        return 0;
    }

    for (i = 0; i < dim; ++i) {
        if (!lmmc_is_finite(&y[i])) {
            return 0;
        }
    }

    return 1;
}

static inline lmmc_status_t validate_and_init_ode_config(
    lmmc_ode_rhs_t rhs,
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    const lmmc_real_t* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_config_t* out_cfg,
    lmmc_ode_result_t* out_result,
    size_t* out_work_bytes
) {
    lmmc_ode_reset_result(out_result, t_start);

    if (rhs == NULL || y == NULL || out_result == NULL || out_cfg == NULL || out_work_bytes == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_ode_load_and_validate_config(dim, t_start, t_end, cfg, out_cfg, out_result) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_ode_state_is_finite(y, dim)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (!lmmc_safe_mul_size(dim, sizeof(lmmc_real_t), out_work_bytes)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_ode_weighted_rms(
    const lmmc_real_t* error,
    const lmmc_real_t* y_old,
    const lmmc_real_t* y_new,
    size_t dim,
    lmmc_real_t abs_tol,
    lmmc_real_t rel_tol,
    lmmc_real_t* out_norm
) {
    lmmc_real_t sum = 0.0;
    size_t i;
    if (error == NULL || y_old == NULL || y_new == NULL || out_norm == NULL || dim == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0; i < dim; ++i) {
        lmmc_real_t scale = abs_tol + rel_tol * fmax(fabs(y_old[i]), fabs(y_new[i]));
        lmmc_real_t ratio;
        if (!isfinite(scale) || scale <= 0.0 || !isfinite(error[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        ratio = error[i] / scale;
        sum += ratio * ratio;
        if (!isfinite(sum)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    *out_norm = sqrt(sum / (lmmc_real_t)dim);
    return isfinite(*out_norm) ? LMMC_STATUS_OK : LMMC_STATUS_NUMERICAL_FAILURE;
}

static inline lmmc_real_t lmmc_ode_next_step(
    lmmc_real_t h,
    lmmc_real_t error_norm,
    const lmmc_ode_config_t* cfg
) {
    lmmc_real_t factor = 5.0;
    if (error_norm > 0.0 && isfinite(error_norm)) {
        factor = cfg->adaptive_step_beta * pow(error_norm, -0.25);
    }
    factor = lmmc_clamp(factor, 0.2, 5.0);
    return lmmc_clamp(h * factor, cfg->min_step, cfg->max_step);
}

/**
 * @brief 有限差分近似 Jacobian df/dy。
 */
static inline lmmc_status_t ode_fd_jacobian(
    lmmc_ode_rhs_t rhs, void* user_data,
    lmmc_real_t t, const lmmc_real_t* y, size_t dim,
    lmmc_real_t* J, lmmc_real_t* f0, lmmc_real_t* y_pert, lmmc_real_t* f_pert
) {
    size_t i, j;
    lmmc_real_t eps_fd = 1.0e-7;
    lmmc_status_t st;

    st = rhs(t, y, f0, dim, user_data);
    if (st != LMMC_STATUS_OK) return st;

    for (j = 0; j < dim; ++j) {
        lmmc_real_t delta = eps_fd * (1.0 + fabs(y[j]));
        memcpy(y_pert, y, dim * sizeof(lmmc_real_t));
        y_pert[j] += delta;
        st = rhs(t, y_pert, f_pert, dim, user_data);
        if (st != LMMC_STATUS_OK) return st;
        for (i = 0; i < dim; ++i) {
            J[i * dim + j] = (f_pert[i] - f0[i]) / delta;
        }
    }
    return LMMC_STATUS_OK;
}

/**
 * @brief 计算或获取 Jacobian（用户提供或有限差分）。
 */
static inline lmmc_status_t ode_get_jacobian(
    lmmc_ode_rhs_t rhs, void* user_data, lmmc_ode_jac_t jac_cb,
    lmmc_real_t t, const lmmc_real_t* y, size_t dim,
    lmmc_real_t* J, lmmc_real_t* f0, lmmc_real_t* y_pert, lmmc_real_t* f_pert
) {
    if (jac_cb != NULL) {
        return jac_cb(t, y, J, dim, user_data);
    }
    return ode_fd_jacobian(rhs, user_data, t, y, dim, J, f0, y_pert, f_pert);
}

#endif
