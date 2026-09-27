#include "ode_internal.h"

void lmmc_ode_do_log(const lmmc_ode_config_t* cfg, size_t step, lmmc_real_t t, const lmmc_real_t* y, size_t dim) {
    const lmmc_real_t values[] = {t, dim > 0 ? y[0] : 0.0};
    const lmmc_diagnostic_t diagnostic = {
        LMMC_DIAGNOSTIC_TRACE, "ode", "accepted step", step, values, 2
    };
    lmmc_diagnostic_emit(&cfg->diagnostics, &diagnostic);
}

void lmmc_ode_reset_result(lmmc_ode_result_t* out_result, lmmc_real_t t_start) {
    if (out_result == NULL) {
        return;
    }

    out_result->converged = 0;
    out_result->num_steps = 0;
    out_result->num_rhs_evals = 0;
    out_result->final_t = t_start;
    out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
}

static lmmc_status_t ode_invalid_config(
    lmmc_ode_result_t* result, lmmc_ode_failure_t reason
) {
    if (result != NULL) result->failure_reason = reason;
    return LMMC_STATUS_INVALID_ARGUMENT;
}

static int ode_tolerances_valid(const lmmc_ode_config_t* cfg) {
    if (!lmmc_is_finite(&cfg->abs_tol) || !lmmc_is_finite(&cfg->rel_tol)) { return 0; }
    if (cfg->abs_tol < 0.0 || cfg->rel_tol < 0.0) { return 0; }
    return cfg->abs_tol != 0.0 || cfg->rel_tol != 0.0;
}

static int ode_steps_valid(const lmmc_ode_config_t* cfg) {
    if (!lmmc_is_finite(&cfg->initial_step) || !lmmc_is_finite(&cfg->min_step)) {
        return 0;
    }
    if (!lmmc_is_finite(&cfg->max_step)) {
        return 0;
    }
    if (cfg->initial_step <= 0.0 || cfg->min_step <= 0.0 || cfg->max_step <= 0.0) {
        return 0;
    }
    if (cfg->min_step > cfg->max_step || cfg->max_steps == 0) {
        return 0;
    }
    return lmmc_is_finite(&cfg->adaptive_step_beta) &&
           cfg->adaptive_step_beta > 0.0 && cfg->adaptive_step_beta <= 1.0;
}

lmmc_status_t lmmc_ode_load_and_validate_config(
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_config_t* out_cfg,
    lmmc_ode_result_t* out_result
) {
    if (dim == 0) { return ode_invalid_config(out_result, LMMC_ODE_FAILURE_INVALID_DIMENSION); }
    if (!lmmc_is_finite(&t_start) || !lmmc_is_finite(&t_end) || !(t_end > t_start)) { return ode_invalid_config(out_result, LMMC_ODE_FAILURE_INVALID_STEP); }
    if (out_cfg == NULL) { return ode_invalid_config(out_result, LMMC_ODE_FAILURE_INVALID_STEP); }
    if (cfg != NULL) {
        *out_cfg = *cfg;
    } else {
        lmmc_status_t st = lmmc_ode_default_config(t_start, t_end, dim, out_cfg);
        if (st != LMMC_STATUS_OK) {
            ode_invalid_config(out_result, LMMC_ODE_FAILURE_INVALID_STEP);
            return st;
        }
    }
    if (!ode_tolerances_valid(out_cfg)) { return ode_invalid_config(out_result, LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT); }
    if (!ode_steps_valid(out_cfg)) { return ode_invalid_config(out_result, LMMC_ODE_FAILURE_INVALID_STEP); }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ode_rhs_eval(
    lmmc_ode_rhs_t rhs,
    lmmc_real_t t,
    const lmmc_real_t* y,
    lmmc_real_t* y_prime,
    size_t dim,
    void* user_data,
    size_t* io_eval_count,
    int* out_callback_failed
) {
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
    return lmmc_ode_values_are_finite(y_prime, dim);
}

int lmmc_ode_state_is_finite(const lmmc_real_t* y, size_t dim) {
    if (y == NULL || dim == 0) {
        return 0;
    }

    return lmmc_ode_values_are_finite(y, dim) == LMMC_STATUS_OK;
}

lmmc_status_t validate_and_init_ode_config(
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

lmmc_status_t lmmc_ode_weighted_rms(
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

lmmc_real_t lmmc_ode_next_step(
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

lmmc_status_t lmmc_ode_values_are_finite(
    const lmmc_real_t* values,
    size_t count
) {
    if (values == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return lmmc_real_range_is_finite(values, count)
        ? LMMC_STATUS_OK
        : LMMC_STATUS_NUMERICAL_FAILURE;
}
