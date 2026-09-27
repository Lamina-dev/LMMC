/**
 * @file nonlinear.c
 * @brief 标量非线性方程求根算法实现。
 */
#include <math.h>
#include <float.h>
#include "internal.h"
#include "internal/nonlinear_internal.h"

void lmmc_nonlinear_do_log(const lmmc_nonlinear_config_t* cfg, size_t iter, lmmc_real_t x, lmmc_real_t f_x) {
    const lmmc_real_t values[] = {x, f_x};
    const lmmc_diagnostic_t diagnostic = {
        LMMC_DIAGNOSTIC_TRACE, "nonlinear", "iteration", iter, values, 2
    };
    lmmc_diagnostic_emit(&cfg->diagnostics, &diagnostic);
}


lmmc_real_t lmmc_nonlinear_midpoint(
    lmmc_real_t left, lmmc_real_t right) {
    return lmmc_interval_midpoint(left, right);
}

lmmc_real_t lmmc_nonlinear_half_width(
    lmmc_real_t left, lmmc_real_t right) {
    lmmc_real_t center;
    lmmc_real_t half_width;
    lmmc_interval_center_half_width(
        left, right, &center, &half_width);
    return half_width;
}
lmmc_real_t lmmc_nonlinear_distance(
    lmmc_real_t left, lmmc_real_t right) {
    const lmmc_real_t difference = left - right;
    return lmmc_is_finite(&difference) ? lmmc_abs(difference) : DBL_MAX;
}


lmmc_status_t lmmc_nonlinear_load_and_validate_config(
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_config_t* out_cfg
) {
    if (out_cfg == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (cfg != NULL) {
        *out_cfg = *cfg;
    } else {
        lmmc_status_t st = lmmc_nonlinear_default_config(out_cfg);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }

    if (!lmmc_is_finite(&out_cfg->abs_tol) || !lmmc_is_finite(&out_cfg->rel_tol) ||
        !lmmc_is_finite(&out_cfg->derivative_step) || !lmmc_is_finite(&out_cfg->min_derivative) ||
        !lmmc_is_finite(&out_cfg->min_step)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (out_cfg->abs_tol < 0.0 || out_cfg->rel_tol < 0.0 ||
        out_cfg->max_iter == 0 || out_cfg->derivative_step <= 0.0 || out_cfg->min_derivative <= 0.0 ||
        out_cfg->min_step < 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_nonlinear_x_tolerance(
    lmmc_real_t x,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_real_t* out_x_tol
) {
    lmmc_real_t x_scale = 0.0;
    lmmc_real_t x_tol = 0.0;
    lmmc_real_t abs_x = 0.0;
    lmmc_real_t one = 1.0;
    lmmc_real_t tmp = 0.0;

    if (cfg == NULL || out_x_tol == NULL || !lmmc_is_finite(&x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    abs_x = lmmc_abs(x);
    x_scale = lmmc_max(abs_x, one);
    LMMC_REAL_MUL(&tmp, &cfg->rel_tol, &x_scale);
    LMMC_REAL_ADD(&x_tol, &cfg->abs_tol, &tmp);

    if (!lmmc_is_finite(&x_tol)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    *out_x_tol = x_tol;
    return LMMC_STATUS_OK;
}

void lmmc_nonlinear_reset_result(lmmc_nonlinear_result_t* out_result) {
    if (out_result == NULL) {
        return;
    }

    out_result->converged = 0;
    out_result->num_iter = 0;
    out_result->root = 0.0;
    out_result->function_value = 0.0;
    out_result->residual_norm = 0.0;
    out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
}

const char* lmmc_nonlinear_failure_string(lmmc_nonlinear_failure_t reason) {
    switch (reason) {
        case LMMC_NONLINEAR_FAILURE_NONE:
            return "none";
        case LMMC_NONLINEAR_FAILURE_INVALID_BRACKET:
            return "invalid bracket";
        case LMMC_NONLINEAR_FAILURE_MAX_ITER:
            return "max iterations reached";
        case LMMC_NONLINEAR_FAILURE_ZERO_DERIVATIVE:
            return "zero derivative";
        case LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE:
            return "numerical issue";
        case LMMC_NONLINEAR_FAILURE_SINGULAR_STEP:
            return "singular step";
        default:
            return "unknown";
    }
}

lmmc_status_t lmmc_nonlinear_default_config(lmmc_nonlinear_config_t* out_cfg) {
    if (out_cfg == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    out_cfg->abs_tol = LMMC_DEFAULT_ABS_TOL;
    out_cfg->rel_tol = LMMC_DEFAULT_REL_TOL;
    out_cfg->max_iter = 100;
    out_cfg->derivative_step = 1e-6;
    out_cfg->min_derivative = 1e-14;
    out_cfg->min_step = 1e-14;
    out_cfg->diagnostics = (lmmc_diagnostic_sink_t){0};
    return LMMC_STATUS_OK;
}
