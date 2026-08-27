/**
 * @file ode_common.c
 * @brief ODE 公共实现：失败原因描述与默认求解配置。
 */
#include "internal.h"
#include "lmmc/ode.h"

const char* lmmc_ode_failure_string(lmmc_ode_failure_t reason) {
    switch (reason) {
        case LMMC_ODE_FAILURE_NONE:
            return "none";
        case LMMC_ODE_FAILURE_INVALID_DIMENSION:
            return "invalid dimension";
        case LMMC_ODE_FAILURE_INVALID_STEP:
            return "invalid step";
        case LMMC_ODE_FAILURE_MAX_STEPS:
            return "max steps reached";
        case LMMC_ODE_FAILURE_NUMERICAL_ISSUE:
            return "numerical issue";
        case LMMC_ODE_FAILURE_RHS_EVAL_FAILED:
            return "rhs evaluation failed";
        case LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT:
            return "tolerance inconsistent";
        default:
            return "unknown";
    }
}

lmmc_status_t lmmc_ode_default_config(
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    size_t problem_dim,
    lmmc_ode_config_t* out_cfg
) {
    lmmc_real_t span = 0.0;
    lmmc_real_t initial_step = 0.0;
    lmmc_real_t min_step = 0.0;
    lmmc_real_t max_step = 0.0;
    size_t max_steps = 0;
    lmmc_real_t hundred = 100.0;
    lmmc_real_t ten = 10.0;
    lmmc_real_t span_factor = 1e-10;
    lmmc_real_t step_factor = 1e-6;
    lmmc_real_t tenth = 0.1;

    if (out_cfg == NULL || problem_dim == 0 || !lmmc_is_finite(&t_start) ||
        !lmmc_is_finite(&t_end) || !(t_end > t_start)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_SUB(&span, &t_end, &t_start);
    LMMC_REAL_DIV(&initial_step, &span, &hundred);
    if (!lmmc_is_finite(&initial_step) || initial_step <= 0.0) {
        LMMC_REAL_SET_D(&initial_step, 1e-3);
    }

    LMMC_REAL_MUL(&min_step, &span, &span_factor);
    if (!lmmc_is_finite(&min_step) || min_step <= 0.0) {
        LMMC_REAL_MUL(&min_step, &initial_step, &step_factor);
    }
    if (min_step > initial_step) {
        LMMC_REAL_MUL(&min_step, &initial_step, &tenth);
    }

    LMMC_REAL_DIV(&max_step, &span, &ten);
    if (!lmmc_is_finite(&max_step) || max_step <= 0.0) {
        LMMC_REAL_SET(&max_step, &initial_step);
    }
    if (max_step < initial_step) {
        LMMC_REAL_SET(&max_step, &initial_step);
    }

    if (problem_dim > ((size_t)-1) / 10000) {
        max_steps = 100000;
    } else {
        max_steps = problem_dim * 10000;
        if (max_steps < 1000) {
            max_steps = 1000;
        }
    }

    out_cfg->initial_step = initial_step;
    out_cfg->min_step = min_step;
    out_cfg->max_step = max_step;
    out_cfg->abs_tol = LMMC_DEFAULT_ABS_TOL;
    out_cfg->rel_tol = LMMC_DEFAULT_REL_TOL;
    out_cfg->max_steps = max_steps;
    out_cfg->adaptive_step_beta = 0.9;
    out_cfg->diagnostics = (lmmc_diagnostic_sink_t){0};
    out_cfg->jacobian = NULL;
    return LMMC_STATUS_OK;
}
