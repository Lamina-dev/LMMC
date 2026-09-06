/**
 * @file nonlinear.c
 * @brief 标量非线性方程求根算法实现。
 */
#include <math.h>
#include <float.h>
#include "internal.h"
#include "lmmc/nonlinear.h"

static void lmmc_nonlinear_do_log(const lmmc_nonlinear_config_t* cfg, size_t iter, lmmc_real_t x, lmmc_real_t f_x) {
    const lmmc_real_t values[] = {x, f_x};
    const lmmc_diagnostic_t diagnostic = {
        LMMC_DIAGNOSTIC_TRACE, "nonlinear", "iteration", iter, values, 2
    };
    lmmc_diagnostic_emit(&cfg->diagnostics, &diagnostic);
}


static lmmc_real_t lmmc_nonlinear_midpoint(
    lmmc_real_t left, lmmc_real_t right) {
    if (left < 0.0 && right > 0.0)
        return left * 0.5 + right * 0.5;
    return left + (right - left) * 0.5;
}

static lmmc_real_t lmmc_nonlinear_half_width(
    lmmc_real_t left, lmmc_real_t right) {
    if (left < 0.0 && right > 0.0)
        return right * 0.5 - left * 0.5;
    return (right - left) * 0.5;
}
static lmmc_real_t lmmc_nonlinear_distance(
    lmmc_real_t left, lmmc_real_t right) {
    const lmmc_real_t difference = left - right;
    return lmmc_is_finite(&difference) ? lmmc_abs(difference) : DBL_MAX;
}


static lmmc_status_t lmmc_nonlinear_load_and_validate_config(
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
        !lmmc_is_finite(&out_cfg->min_step) || out_cfg->abs_tol < 0.0 || out_cfg->rel_tol < 0.0 ||
        out_cfg->max_iter == 0 || out_cfg->derivative_step <= 0.0 || out_cfg->min_derivative <= 0.0 ||
        out_cfg->min_step < 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_nonlinear_x_tolerance(
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

static void lmmc_nonlinear_reset_result(lmmc_nonlinear_result_t* out_result) {
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

lmmc_status_t lmmc_bisection_solve(
    lmmc_scalar_func_t func,
    void* user_data,
    lmmc_real_t left,
    lmmc_real_t right,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
) {
    lmmc_nonlinear_config_t local_cfg = {0};
    lmmc_real_t f_left = 0.0;
    lmmc_real_t f_right = 0.0;
    size_t iter = 0;

    if (func == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_nonlinear_reset_result(out_result);

    if (!lmmc_is_finite(&left) || !lmmc_is_finite(&right) || !(left < right)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (cfg != NULL) {
        local_cfg = *cfg;
    } else {
        lmmc_status_t st_cfg = lmmc_nonlinear_default_config(&local_cfg);
        if (st_cfg != LMMC_STATUS_OK) {
            return st_cfg;
        }
    }

    if (!lmmc_is_finite(&local_cfg.abs_tol) || !lmmc_is_finite(&local_cfg.rel_tol) ||
        local_cfg.abs_tol < 0.0 || local_cfg.rel_tol < 0.0 || local_cfg.max_iter == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    f_left = func(left, user_data);
    f_right = func(right, user_data);

    if (!lmmc_is_finite(&f_left) || !lmmc_is_finite(&f_right)) {
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    out_result->root = left;
    out_result->function_value = f_left;
    out_result->residual_norm = lmmc_abs(f_left);

    lmmc_nonlinear_do_log(&local_cfg, 0, out_result->root, out_result->function_value);

    if (lmmc_abs(f_left) <= local_cfg.abs_tol) {
        out_result->converged = 1;
        return LMMC_STATUS_OK;
    }

    if (lmmc_abs(f_right) <= local_cfg.abs_tol) {
        out_result->converged = 1;
        out_result->root = right;
        out_result->function_value = f_right;
        out_result->residual_norm = lmmc_abs(f_right);
        lmmc_nonlinear_do_log(&local_cfg, 0, out_result->root, out_result->function_value);
        return LMMC_STATUS_OK;
    }

    if ((f_left > 0.0 && f_right > 0.0) || (f_left < 0.0 && f_right < 0.0)) {
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_INVALID_BRACKET;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (iter = 1; iter <= local_cfg.max_iter; ++iter) {
        lmmc_real_t mid;
        lmmc_real_t f_mid = 0.0;
        lmmc_real_t x_scale = 0.0;
        lmmc_real_t x_tol = 0.0;
        lmmc_real_t abs_mid = 0.0;
        lmmc_real_t one = 1.0;
        lmmc_real_t half_width;
        lmmc_real_t tmp = 0.0;

        mid = lmmc_nonlinear_midpoint(left, right);
        half_width = lmmc_nonlinear_half_width(left, right);
        f_mid = func(mid, user_data);
        abs_mid = lmmc_abs(mid);
        x_scale = lmmc_max(abs_mid, one);
        LMMC_REAL_MUL(&tmp, &local_cfg.rel_tol, &x_scale);
        LMMC_REAL_ADD(&x_tol, &local_cfg.abs_tol, &tmp);

        if (!lmmc_is_finite(&mid) || !lmmc_is_finite(&f_mid) ||
            !lmmc_is_finite(&half_width) || !lmmc_is_finite(&x_tol)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        out_result->num_iter = iter;
        out_result->root = mid;
        out_result->function_value = f_mid;
        out_result->residual_norm = lmmc_abs(f_mid);

        lmmc_nonlinear_do_log(&local_cfg, iter, out_result->root, out_result->function_value);

        if (lmmc_abs(f_mid) <= local_cfg.abs_tol || half_width <= x_tol) {
            out_result->converged = 1;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
            return LMMC_STATUS_OK;
        }

        if ((f_left > 0.0 && f_mid < 0.0) || (f_left < 0.0 && f_mid > 0.0)) {
            right = mid;
            f_right = f_mid;
        } else {
            left = mid;
            f_left = f_mid;
        }
    }

    out_result->converged = 0;
    out_result->failure_reason = LMMC_NONLINEAR_FAILURE_MAX_ITER;
    return LMMC_STATUS_NUMERICAL_FAILURE;
}

lmmc_status_t lmmc_newton_solve(
    lmmc_scalar_func_t func,
    lmmc_scalar_dfunc_t dfunc,
    void* user_data,
    lmmc_real_t x0,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
) {
    lmmc_nonlinear_config_t local_cfg = {0};
    lmmc_real_t x = x0;
    lmmc_real_t fx = 0.0;
    size_t iter = 0;

    if (func == NULL || out_result == NULL || !lmmc_is_finite(&x0)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_nonlinear_reset_result(out_result);

    {
        lmmc_status_t st_cfg = lmmc_nonlinear_load_and_validate_config(cfg, &local_cfg);
        if (st_cfg != LMMC_STATUS_OK) {
            return st_cfg;
        }
    }

    fx = func(x, user_data);
    if (!lmmc_is_finite(&fx)) {
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    out_result->root = x;
    out_result->function_value = fx;
    out_result->residual_norm = lmmc_abs(fx);

    lmmc_nonlinear_do_log(&local_cfg, 0, out_result->root, out_result->function_value);

    if (out_result->residual_norm <= local_cfg.abs_tol) {
        out_result->converged = 1;
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
        return LMMC_STATUS_OK;
    }

    for (iter = 1; iter <= local_cfg.max_iter; ++iter) {
        lmmc_real_t derivative = 0.0;
        lmmc_real_t step = 0.0;
        lmmc_real_t x_next = 0.0;
        lmmc_real_t f_next = 0.0;
        lmmc_real_t x_tol = 0.0;

        if (dfunc != NULL) {
            derivative = dfunc(x, user_data);
        } else {
            lmmc_real_t abs_x = lmmc_abs(x);
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

            LMMC_REAL_MUL(&h, &local_cfg.derivative_step, &max_val);

            if (!lmmc_is_finite(&h) || h <= 0.0) {
                out_result->num_iter = iter;
                out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            LMMC_REAL_ADD(&x_plus_h, &x, &h);
            LMMC_REAL_SUB(&x_minus_h, &x, &h);
            f_plus = func(x_plus_h, user_data);
            f_minus = func(x_minus_h, user_data);

            if (!lmmc_is_finite(&f_plus) || !lmmc_is_finite(&f_minus)) {
                out_result->num_iter = iter;
                out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            LMMC_REAL_SUB(&f_diff, &f_plus, &f_minus);
            LMMC_REAL_MUL(&two_h, &two, &h);
            LMMC_REAL_DIV(&derivative, &f_diff, &two_h);
        }

        if (!lmmc_is_finite(&derivative)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        if (lmmc_abs(derivative) < local_cfg.min_derivative) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_ZERO_DERIVATIVE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        LMMC_REAL_DIV(&step, &fx, &derivative);
        LMMC_REAL_SUB(&x_next, &x, &step);

        if (!lmmc_is_finite(&step) || !lmmc_is_finite(&x_next)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        f_next = func(x_next, user_data);
        if (!lmmc_is_finite(&f_next)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        if (lmmc_nonlinear_x_tolerance(x_next, &local_cfg, &x_tol) != LMMC_STATUS_OK) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        out_result->num_iter = iter;
        out_result->root = x_next;
        out_result->function_value = f_next;
        out_result->residual_norm = lmmc_abs(f_next);

        lmmc_nonlinear_do_log(&local_cfg, iter, out_result->root, out_result->function_value);

        {
            lmmc_real_t x_diff = 0.0;
            LMMC_REAL_SUB(&x_diff, &x_next, &x);
            if (out_result->residual_norm <= local_cfg.abs_tol || lmmc_abs(x_diff) <= x_tol) {
                out_result->converged = 1;
                out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
                return LMMC_STATUS_OK;
            }
        }

        if (lmmc_abs(step) < local_cfg.min_step) {
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        x = x_next;
        fx = f_next;
    }

    out_result->converged = 0;
    out_result->failure_reason = LMMC_NONLINEAR_FAILURE_MAX_ITER;
    return LMMC_STATUS_NUMERICAL_FAILURE;
}

lmmc_status_t lmmc_secant_solve(
    lmmc_scalar_func_t func,
    void* user_data,
    lmmc_real_t x0,
    lmmc_real_t x1,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
) {
    lmmc_nonlinear_config_t local_cfg = {0};
    lmmc_real_t f0 = 0.0;
    lmmc_real_t f1 = 0.0;
    size_t iter = 0;

    if (func == NULL || out_result == NULL || !lmmc_is_finite(&x0) || !lmmc_is_finite(&x1) || x0 == x1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_nonlinear_reset_result(out_result);

    {
        lmmc_status_t st_cfg = lmmc_nonlinear_load_and_validate_config(cfg, &local_cfg);
        if (st_cfg != LMMC_STATUS_OK) {
            return st_cfg;
        }
    }

    f0 = func(x0, user_data);
    f1 = func(x1, user_data);

    if (!lmmc_is_finite(&f0) || !lmmc_is_finite(&f1)) {
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (lmmc_abs(f0) <= local_cfg.abs_tol) {
        out_result->converged = 1;
        out_result->root = x0;
        out_result->function_value = f0;
        out_result->residual_norm = lmmc_abs(f0);
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
        lmmc_nonlinear_do_log(&local_cfg, 0, out_result->root, out_result->function_value);
        return LMMC_STATUS_OK;
    }

    out_result->root = x1;
    out_result->function_value = f1;
    out_result->residual_norm = lmmc_abs(f1);

    lmmc_nonlinear_do_log(&local_cfg, 0, out_result->root, out_result->function_value);

    if (lmmc_abs(f1) <= local_cfg.abs_tol) {
        out_result->converged = 1;
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
        return LMMC_STATUS_OK;
    }

    for (iter = 1; iter <= local_cfg.max_iter; ++iter) {
        lmmc_real_t scaled_f0;
        lmmc_real_t scaled_f1;
        lmmc_real_t scaled_denom;
        lmmc_real_t function_scale;
        lmmc_real_t step_magnitude;
        lmmc_real_t x2 = 0.0;
        lmmc_real_t f2 = 0.0;
        lmmc_real_t x_tol = 0.0;

        function_scale = lmmc_max(lmmc_abs(f0), lmmc_abs(f1));
        scaled_f0 = f0 / function_scale;
        scaled_f1 = f1 / function_scale;
        scaled_denom = scaled_f1 - scaled_f0;
        if (!lmmc_is_finite(&scaled_denom)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (lmmc_abs(scaled_denom) <= DBL_EPSILON * 64.0) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        if ((f0 < 0.0) != (f1 < 0.0)) {
            const lmmc_real_t weight_sum =
                lmmc_abs(scaled_f0) + lmmc_abs(scaled_f1);
            const lmmc_real_t weight0 = lmmc_abs(scaled_f1) / weight_sum;
            const lmmc_real_t weight1 = lmmc_abs(scaled_f0) / weight_sum;
            x2 = fma(weight0, x0, weight1 * x1);
        } else {
            const lmmc_real_t numerator =
                fma(x0, scaled_f1, -x1 * scaled_f0);
            x2 = numerator / scaled_denom;
        }
        step_magnitude = lmmc_nonlinear_distance(x2, x1);

        if (!lmmc_is_finite(&x2)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        f2 = func(x2, user_data);
        if (!lmmc_is_finite(&f2)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        if (lmmc_nonlinear_x_tolerance(x2, &local_cfg, &x_tol) != LMMC_STATUS_OK) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        out_result->num_iter = iter;
        out_result->root = x2;
        out_result->function_value = f2;
        out_result->residual_norm = lmmc_abs(f2);

        lmmc_nonlinear_do_log(&local_cfg, iter, out_result->root, out_result->function_value);

        {
            if (out_result->residual_norm <= local_cfg.abs_tol ||
                step_magnitude <= x_tol) {
                out_result->converged = 1;
                out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
                return LMMC_STATUS_OK;
            }

            if (step_magnitude < local_cfg.min_step) {
                out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }

        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f2;
    }

    out_result->converged = 0;
    out_result->failure_reason = LMMC_NONLINEAR_FAILURE_MAX_ITER;
    return LMMC_STATUS_NUMERICAL_FAILURE;
}
