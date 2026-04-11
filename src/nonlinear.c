#include <math.h>
#include "lmmc/nonlinear.h"

static int lmmc_is_finite_number(double v) {
    return isfinite(v) ? 1 : 0;
}

static double lmmc_absd(double x) {
    return x < 0.0 ? -x : x;
}

static double lmmc_maxd(double a, double b) {
    return a > b ? a : b;
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

    if (!lmmc_is_finite_number(out_cfg->abs_tol) || !lmmc_is_finite_number(out_cfg->rel_tol) ||
        !lmmc_is_finite_number(out_cfg->derivative_step) || !lmmc_is_finite_number(out_cfg->min_derivative) ||
        !lmmc_is_finite_number(out_cfg->min_step) || out_cfg->abs_tol < 0.0 || out_cfg->rel_tol < 0.0 ||
        out_cfg->max_iter == 0 || out_cfg->derivative_step <= 0.0 || out_cfg->min_derivative <= 0.0 ||
        out_cfg->min_step < 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_nonlinear_x_tolerance(
    double x,
    const lmmc_nonlinear_config_t* cfg,
    double* out_x_tol
) {
    double x_scale = 0.0;
    double x_tol = 0.0;

    if (cfg == NULL || out_x_tol == NULL || !lmmc_is_finite_number(x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    x_scale = lmmc_maxd(lmmc_absd(x), 1.0);
    x_tol = cfg->abs_tol + cfg->rel_tol * x_scale;

    if (!lmmc_is_finite_number(x_tol)) {
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
    out_cfg->verbose = 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_bisection_solve(
    lmmc_scalar_func_t func,
    void* user_data,
    double left,
    double right,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
) {
    lmmc_nonlinear_config_t local_cfg = {0};
    double f_left = 0.0;
    double f_right = 0.0;
    size_t iter = 0;

    if (func == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_nonlinear_reset_result(out_result);

    if (!lmmc_is_finite_number(left) || !lmmc_is_finite_number(right) || !(left < right)) {
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

    if (!lmmc_is_finite_number(local_cfg.abs_tol) || !lmmc_is_finite_number(local_cfg.rel_tol) ||
        local_cfg.abs_tol < 0.0 || local_cfg.rel_tol < 0.0 || local_cfg.max_iter == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    f_left = func(left, user_data);
    f_right = func(right, user_data);

    if (!lmmc_is_finite_number(f_left) || !lmmc_is_finite_number(f_right)) {
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    out_result->root = left;
    out_result->function_value = f_left;
    out_result->residual_norm = lmmc_absd(f_left);

    if (lmmc_absd(f_left) <= local_cfg.abs_tol) {
        out_result->converged = 1;
        return LMMC_STATUS_OK;
    }

    if (lmmc_absd(f_right) <= local_cfg.abs_tol) {
        out_result->converged = 1;
        out_result->root = right;
        out_result->function_value = f_right;
        out_result->residual_norm = lmmc_absd(f_right);
        return LMMC_STATUS_OK;
    }

    if ((f_left > 0.0 && f_right > 0.0) || (f_left < 0.0 && f_right < 0.0)) {
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_INVALID_BRACKET;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (iter = 1; iter <= local_cfg.max_iter; ++iter) {
        double mid = 0.5 * (left + right);
        double f_mid = func(mid, user_data);
        double interval_width = right - left;
        double x_scale = lmmc_maxd(lmmc_absd(mid), 1.0);
        double x_tol = local_cfg.abs_tol + local_cfg.rel_tol * x_scale;

        if (!lmmc_is_finite_number(mid) || !lmmc_is_finite_number(f_mid) ||
            !lmmc_is_finite_number(interval_width) || !lmmc_is_finite_number(x_tol)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        out_result->num_iter = iter;
        out_result->root = mid;
        out_result->function_value = f_mid;
        out_result->residual_norm = lmmc_absd(f_mid);

        if (lmmc_absd(f_mid) <= local_cfg.abs_tol || 0.5 * interval_width <= x_tol) {
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
    double x0,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
) {
    lmmc_nonlinear_config_t local_cfg = {0};
    double x = x0;
    double fx = 0.0;
    size_t iter = 0;

    if (func == NULL || out_result == NULL || !lmmc_is_finite_number(x0)) {
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
    if (!lmmc_is_finite_number(fx)) {
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    out_result->root = x;
    out_result->function_value = fx;
    out_result->residual_norm = lmmc_absd(fx);

    if (out_result->residual_norm <= local_cfg.abs_tol) {
        out_result->converged = 1;
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
        return LMMC_STATUS_OK;
    }

    for (iter = 1; iter <= local_cfg.max_iter; ++iter) {
        double derivative = 0.0;
        double step = 0.0;
        double x_next = 0.0;
        double f_next = 0.0;
        double x_tol = 0.0;

        if (dfunc != NULL) {
            derivative = dfunc(x, user_data);
        } else {
            double h = local_cfg.derivative_step * lmmc_maxd(lmmc_absd(x), 1.0);
            double f_plus = 0.0;
            double f_minus = 0.0;

            if (!lmmc_is_finite_number(h) || h <= 0.0) {
                out_result->num_iter = iter;
                out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            f_plus = func(x + h, user_data);
            f_minus = func(x - h, user_data);

            if (!lmmc_is_finite_number(f_plus) || !lmmc_is_finite_number(f_minus)) {
                out_result->num_iter = iter;
                out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            derivative = (f_plus - f_minus) / (2.0 * h);
        }

        if (!lmmc_is_finite_number(derivative)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        if (lmmc_absd(derivative) < local_cfg.min_derivative) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_ZERO_DERIVATIVE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        step = fx / derivative;
        x_next = x - step;

        if (!lmmc_is_finite_number(step) || !lmmc_is_finite_number(x_next)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        f_next = func(x_next, user_data);
        if (!lmmc_is_finite_number(f_next)) {
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
        out_result->residual_norm = lmmc_absd(f_next);

        if (out_result->residual_norm <= local_cfg.abs_tol || lmmc_absd(x_next - x) <= x_tol) {
            out_result->converged = 1;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
            return LMMC_STATUS_OK;
        }

        if (lmmc_absd(step) < local_cfg.min_step) {
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
    double x0,
    double x1,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
) {
    lmmc_nonlinear_config_t local_cfg = {0};
    double f0 = 0.0;
    double f1 = 0.0;
    size_t iter = 0;

    if (func == NULL || out_result == NULL || !lmmc_is_finite_number(x0) || !lmmc_is_finite_number(x1) || x0 == x1) {
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

    if (!lmmc_is_finite_number(f0) || !lmmc_is_finite_number(f1)) {
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (lmmc_absd(f0) <= local_cfg.abs_tol) {
        out_result->converged = 1;
        out_result->root = x0;
        out_result->function_value = f0;
        out_result->residual_norm = lmmc_absd(f0);
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
        return LMMC_STATUS_OK;
    }

    out_result->root = x1;
    out_result->function_value = f1;
    out_result->residual_norm = lmmc_absd(f1);

    if (lmmc_absd(f1) <= local_cfg.abs_tol) {
        out_result->converged = 1;
        out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
        return LMMC_STATUS_OK;
    }

    for (iter = 1; iter <= local_cfg.max_iter; ++iter) {
        double denom = f1 - f0;
        double step = 0.0;
        double x2 = 0.0;
        double f2 = 0.0;
        double x_tol = 0.0;

        if (!lmmc_is_finite_number(denom)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        if (lmmc_absd(denom) < local_cfg.min_derivative) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        step = f1 * (x1 - x0) / denom;
        x2 = x1 - step;

        if (!lmmc_is_finite_number(step) || !lmmc_is_finite_number(x2)) {
            out_result->num_iter = iter;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        f2 = func(x2, user_data);
        if (!lmmc_is_finite_number(f2)) {
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
        out_result->residual_norm = lmmc_absd(f2);

        if (out_result->residual_norm <= local_cfg.abs_tol || lmmc_absd(x2 - x1) <= x_tol) {
            out_result->converged = 1;
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
            return LMMC_STATUS_OK;
        }

        if (lmmc_absd(x2 - x1) < local_cfg.min_step) {
            out_result->failure_reason = LMMC_NONLINEAR_FAILURE_SINGULAR_STEP;
            return LMMC_STATUS_NUMERICAL_FAILURE;
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
