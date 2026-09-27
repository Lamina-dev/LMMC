#include "internal/nonlinear_internal.h"

typedef struct {
    lmmc_scalar_func_t func;
    void* user_data;
    lmmc_nonlinear_config_t local_cfg;
    lmmc_nonlinear_result_t* out_result;
    lmmc_real_t x0;
    lmmc_real_t x1;
    lmmc_real_t f0;
    lmmc_real_t f1;
    size_t iter;
} secant_state_t;

static lmmc_status_t secant_prediction(secant_state_t* s, lmmc_real_t* candidate, lmmc_real_t* distance) {
        lmmc_real_t scaled_f0;
        lmmc_real_t scaled_f1;
        lmmc_real_t scaled_denom;
        lmmc_real_t function_scale;
        lmmc_real_t step_magnitude;
        lmmc_real_t x2 = 0.0;
        function_scale = lmmc_max(lmmc_abs(s->f0), lmmc_abs(s->f1));
        scaled_f0 = s->f0 / function_scale;
        scaled_f1 = s->f1 / function_scale;
        scaled_denom = scaled_f1 - scaled_f0;
        if (!lmmc_is_finite(&scaled_denom)) {
            s->out_result->num_iter = s->iter;
            s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (lmmc_abs(scaled_denom) <= DBL_EPSILON * 64.0) {
            s->out_result->num_iter = s->iter;
            s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        if ((s->f0 < 0.0) != (s->f1 < 0.0)) {
            const lmmc_real_t weight_sum =
                lmmc_abs(scaled_f0) + lmmc_abs(scaled_f1);
            const lmmc_real_t weight0 = lmmc_abs(scaled_f1) / weight_sum;
            const lmmc_real_t weight1 = lmmc_abs(scaled_f0) / weight_sum;
            x2 = fma(weight0, s->x0, weight1 * s->x1);
        } else {
            const lmmc_real_t numerator =
                fma(s->x0, scaled_f1, -s->x1 * scaled_f0);
            x2 = numerator / scaled_denom;
        }
        step_magnitude = lmmc_nonlinear_distance(x2, s->x1);

        if (!lmmc_is_finite(&x2)) {
            s->out_result->num_iter = s->iter;
            s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    *candidate = x2;
    *distance = step_magnitude;
    return LMMC_STATUS_OK;
}

static lmmc_status_t secant_iteration(secant_state_t* s) {
    lmmc_real_t x2, step_magnitude;
    lmmc_real_t f2 = 0.0, x_tol = 0.0;
    lmmc_status_t st = secant_prediction(s, &x2, &step_magnitude);
    if (st != LMMC_STATUS_OK) { return st; }
        f2 = s->func(x2, s->user_data);
        if (!lmmc_is_finite(&f2)) {
            s->out_result->num_iter = s->iter;
            s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        if (lmmc_nonlinear_x_tolerance(x2, &s->local_cfg, &x_tol) != LMMC_STATUS_OK) {
            s->out_result->num_iter = s->iter;
            s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        s->out_result->num_iter = s->iter;
        s->out_result->root = x2;
        s->out_result->function_value = f2;
        s->out_result->residual_norm = lmmc_abs(f2);

        lmmc_nonlinear_do_log(&s->local_cfg, s->iter, s->out_result->root, s->out_result->function_value);

        {
            if (s->out_result->residual_norm <= s->local_cfg.abs_tol ||
                step_magnitude <= x_tol) {
                s->out_result->converged = 1;
                s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
                return LMMC_STATUS_OK;
            }

            if (step_magnitude < s->local_cfg.min_step) {
                s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }

        s->x0 = s->x1;
        s->f0 = s->f1;
        s->x1 = x2;
        s->f1 = f2;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_secant_solve(
    lmmc_scalar_func_t func,
    void* user_data,
    lmmc_real_t x0,
    lmmc_real_t x1,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
) {
    secant_state_t state = {0};
    secant_state_t* s = &state;
    s->func = func;
    s->user_data = user_data;
    s->out_result = out_result;
    s->x0 = x0;
    s->x1 = x1;
    if (s->func == NULL || s->out_result == NULL || !lmmc_is_finite(&s->x0) || !lmmc_is_finite(&s->x1) || s->x0 == s->x1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_nonlinear_reset_result(s->out_result);

    {
        lmmc_status_t st_cfg = lmmc_nonlinear_load_and_validate_config(cfg, &s->local_cfg);
        if (st_cfg != LMMC_STATUS_OK) {
            return st_cfg;
        }
    }

    s->f0 = s->func(s->x0, s->user_data);
    s->f1 = s->func(s->x1, s->user_data);

    if (!lmmc_is_finite(&s->f0) || !lmmc_is_finite(&s->f1)) {
        s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (lmmc_abs(s->f0) <= s->local_cfg.abs_tol) {
        s->out_result->converged = 1;
        s->out_result->root = s->x0;
        s->out_result->function_value = s->f0;
        s->out_result->residual_norm = lmmc_abs(s->f0);
        s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
        lmmc_nonlinear_do_log(&s->local_cfg, 0, s->out_result->root, s->out_result->function_value);
        return LMMC_STATUS_OK;
    }

    s->out_result->root = s->x1;
    s->out_result->function_value = s->f1;
    s->out_result->residual_norm = lmmc_abs(s->f1);

    lmmc_nonlinear_do_log(&s->local_cfg, 0, s->out_result->root, s->out_result->function_value);

    if (lmmc_abs(s->f1) <= s->local_cfg.abs_tol) {
        s->out_result->converged = 1;
        s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
        return LMMC_STATUS_OK;
    }
    for (s->iter = 1; s->iter <= s->local_cfg.max_iter; ++s->iter) {
        lmmc_status_t st = secant_iteration(s);
        if (st != LMMC_STATUS_OK || out_result->converged) { return st; }
    }
    s->out_result->converged = 0;
    s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_MAX_ITER;
    return LMMC_STATUS_NUMERICAL_FAILURE;
}
