#include "internal/nonlinear_internal.h"

typedef struct {
    lmmc_scalar_func_t func;
    lmmc_scalar_dfunc_t dfunc;
    void* user_data;
    lmmc_nonlinear_config_t local_cfg;
    lmmc_nonlinear_result_t* out_result;
    lmmc_real_t x;
    lmmc_real_t fx;
    size_t iter;
} newton_state_t;

static lmmc_status_t newton_derivative(newton_state_t* s, lmmc_real_t* derivative) {
        if (s->dfunc != NULL) {
            *derivative = s->dfunc(s->x, s->user_data);
        } else {
            lmmc_real_t abs_x = lmmc_abs(s->x);
            lmmc_real_t one = 1.0;
            lmmc_real_t max_val = lmmc_max(abs_x, one);
            lmmc_real_t h = 0.0;
            lmmc_real_t f_plus = 0.0;
            lmmc_real_t f_minus = 0.0;
            lmmc_real_t x_plus_h = 0.0;
            lmmc_real_t x_minus_h = 0.0;
            lmmc_real_t f_diff = 0.0;
            lmmc_real_t two_h = 0.0;
            lmmc_real_t two = 2.0;

            LMMC_REAL_MUL(&h, &s->local_cfg.derivative_step, &max_val);

            if (!lmmc_is_finite(&h) || h <= 0.0) {
                s->out_result->num_iter = s->iter;
                s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            LMMC_REAL_ADD(&x_plus_h, &s->x, &h);
            LMMC_REAL_SUB(&x_minus_h, &s->x, &h);
            f_plus = s->func(x_plus_h, s->user_data);
            f_minus = s->func(x_minus_h, s->user_data);

            if (!lmmc_is_finite(&f_plus) || !lmmc_is_finite(&f_minus)) {
                s->out_result->num_iter = s->iter;
                s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            LMMC_REAL_SUB(&f_diff, &f_plus, &f_minus);
            LMMC_REAL_MUL(&two_h, &two, &h);
            LMMC_REAL_DIV(derivative, &f_diff, &two_h);
        }

        if (!lmmc_is_finite(derivative)) {
            s->out_result->num_iter = s->iter;
            s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        if (lmmc_abs(*derivative) < s->local_cfg.min_derivative) {
            s->out_result->num_iter = s->iter;
            s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_ZERO_DERIVATIVE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    return LMMC_STATUS_OK;
}

static lmmc_status_t newton_iteration(newton_state_t* s) {
        lmmc_real_t derivative = 0.0;
        lmmc_real_t step = 0.0;
        lmmc_real_t x_next = 0.0;
        lmmc_real_t f_next = 0.0;
        lmmc_real_t x_tol = 0.0;

        lmmc_status_t st = newton_derivative(s, &derivative);
        if (st != LMMC_STATUS_OK) { return st; }
        LMMC_REAL_DIV(&step, &s->fx, &derivative);
        LMMC_REAL_SUB(&x_next, &s->x, &step);

        if (!lmmc_is_finite(&step) || !lmmc_is_finite(&x_next)) {
            s->out_result->num_iter = s->iter;
            s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        f_next = s->func(x_next, s->user_data);
        if (!lmmc_is_finite(&f_next)) {
            s->out_result->num_iter = s->iter;
            s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        if (lmmc_nonlinear_x_tolerance(x_next, &s->local_cfg, &x_tol) != LMMC_STATUS_OK) {
            s->out_result->num_iter = s->iter;
            s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        s->out_result->num_iter = s->iter;
        s->out_result->root = x_next;
        s->out_result->function_value = f_next;
        s->out_result->residual_norm = lmmc_abs(f_next);

        lmmc_nonlinear_do_log(&s->local_cfg, s->iter, s->out_result->root, s->out_result->function_value);

        {
            lmmc_real_t x_diff = 0.0;
            LMMC_REAL_SUB(&x_diff, &x_next, &s->x);
            if (s->out_result->residual_norm <= s->local_cfg.abs_tol || lmmc_abs(x_diff) <= x_tol) {
                s->out_result->converged = 1;
                s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
                return LMMC_STATUS_OK;
            }
        }

        if (lmmc_abs(step) < s->local_cfg.min_step) {
            s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        s->x = x_next;
        s->fx = f_next;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_newton_solve(
    lmmc_scalar_func_t func,
    lmmc_scalar_dfunc_t dfunc,
    void* user_data,
    lmmc_real_t x0,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
) {
    newton_state_t state = {0};
    newton_state_t* s = &state;
    s->func = func;
    s->dfunc = dfunc;
    s->user_data = user_data;
    s->out_result = out_result;
    s->x = x0;
    if (s->func == NULL || s->out_result == NULL || !lmmc_is_finite(&x0)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_nonlinear_reset_result(s->out_result);

    {
        lmmc_status_t st_cfg = lmmc_nonlinear_load_and_validate_config(cfg, &s->local_cfg);
        if (st_cfg != LMMC_STATUS_OK) {
            return st_cfg;
        }
    }

    s->fx = s->func(s->x, s->user_data);
    if (!lmmc_is_finite(&s->fx)) {
        s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    s->out_result->root = s->x;
    s->out_result->function_value = s->fx;
    s->out_result->residual_norm = lmmc_abs(s->fx);

    lmmc_nonlinear_do_log(&s->local_cfg, 0, s->out_result->root, s->out_result->function_value);

    if (s->out_result->residual_norm <= s->local_cfg.abs_tol) {
        s->out_result->converged = 1;
        s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
        return LMMC_STATUS_OK;
    }
    for (s->iter = 1; s->iter <= s->local_cfg.max_iter; ++s->iter) {
        lmmc_status_t st = newton_iteration(s);
        if (st != LMMC_STATUS_OK || out_result->converged) { return st; }
    }
    s->out_result->converged = 0;
    s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_MAX_ITER;
    return LMMC_STATUS_NUMERICAL_FAILURE;
}
