#include "internal/nonlinear_internal.h"

typedef struct {
    lmmc_scalar_func_t func;
    void* user_data;
    lmmc_nonlinear_config_t local_cfg;
    lmmc_nonlinear_result_t* out_result;
    lmmc_real_t left;
    lmmc_real_t right;
    lmmc_real_t f_left;
    lmmc_real_t f_right;
    size_t iter;
} bisection_state_t;

static lmmc_status_t bisection_endpoints(bisection_state_t* s) {
    s->f_left = s->func(s->left, s->user_data);
    s->f_right = s->func(s->right, s->user_data);

    if (!lmmc_is_finite(&s->f_left) || !lmmc_is_finite(&s->f_right)) {
        s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    s->out_result->root = s->left;
    s->out_result->function_value = s->f_left;
    s->out_result->residual_norm = lmmc_abs(s->f_left);

    lmmc_nonlinear_do_log(&s->local_cfg, 0, s->out_result->root, s->out_result->function_value);

    if (lmmc_abs(s->f_left) <= s->local_cfg.abs_tol) {
        s->out_result->converged = 1;
        return LMMC_STATUS_OK;
    }

    if (lmmc_abs(s->f_right) <= s->local_cfg.abs_tol) {
        s->out_result->converged = 1;
        s->out_result->root = s->right;
        s->out_result->function_value = s->f_right;
        s->out_result->residual_norm = lmmc_abs(s->f_right);
        lmmc_nonlinear_do_log(&s->local_cfg, 0, s->out_result->root, s->out_result->function_value);
        return LMMC_STATUS_OK;
    }

    if ((s->f_left > 0.0 && s->f_right > 0.0) || (s->f_left < 0.0 && s->f_right < 0.0)) {
        s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_INVALID_BRACKET;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t bisection_iteration(bisection_state_t* s) {
    lmmc_real_t mid;
    lmmc_real_t f_mid;
    lmmc_real_t x_tol;
    lmmc_real_t half_width;
    lmmc_status_t tolerance_status;

    mid = lmmc_nonlinear_midpoint(s->left, s->right);
    half_width = lmmc_nonlinear_half_width(s->left, s->right);
    f_mid = s->func(mid, s->user_data);
    tolerance_status = lmmc_nonlinear_x_tolerance(
        mid, &s->local_cfg, &x_tol);

    if (!lmmc_is_finite(&mid) || !lmmc_is_finite(&f_mid) ||
        !lmmc_is_finite(&half_width) || tolerance_status != LMMC_STATUS_OK) {
        s->out_result->num_iter = s->iter;
        s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    s->out_result->num_iter = s->iter;
    s->out_result->root = mid;
    s->out_result->function_value = f_mid;
    s->out_result->residual_norm = lmmc_abs(f_mid);

    lmmc_nonlinear_do_log(
        &s->local_cfg, s->iter,
        s->out_result->root, s->out_result->function_value);

    if (lmmc_abs(f_mid) <= s->local_cfg.abs_tol || half_width <= x_tol) {
        s->out_result->converged = 1;
        s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_NONE;
        return LMMC_STATUS_OK;
    }

    if ((s->f_left < 0.0) != (f_mid < 0.0)) {
        s->right = mid;
        s->f_right = f_mid;
    } else {
        s->left = mid;
        s->f_left = f_mid;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t bisection_initialize(
    bisection_state_t* s, const lmmc_nonlinear_config_t* cfg
) {
    if (s->func == NULL || s->out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_nonlinear_reset_result(s->out_result);
    if (!lmmc_is_finite(&s->left) || !lmmc_is_finite(&s->right)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!(s->left < s->right)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (cfg != NULL) {
        s->local_cfg = *cfg;
    } else {
        lmmc_status_t st = lmmc_nonlinear_default_config(&s->local_cfg);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }
    if (!lmmc_is_finite(&s->local_cfg.abs_tol) || !lmmc_is_finite(&s->local_cfg.rel_tol)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (s->local_cfg.abs_tol < 0.0 || s->local_cfg.rel_tol < 0.0 || s->local_cfg.max_iter == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
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
    bisection_state_t state = {0};
    bisection_state_t* s = &state;
    s->func = func;
    s->user_data = user_data;
    s->out_result = out_result;
    s->left = left;
    s->right = right;
    lmmc_status_t st = bisection_initialize(s, cfg);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = bisection_endpoints(s);
    if (st != LMMC_STATUS_OK || out_result->converged) { return st; }
    for (s->iter = 1; s->iter <= s->local_cfg.max_iter; ++s->iter) {
        st = bisection_iteration(s);
        if (st != LMMC_STATUS_OK || out_result->converged) { return st; }
    }
    s->out_result->converged = 0;
    s->out_result->failure_reason = LMMC_NONLINEAR_FAILURE_MAX_ITER;
    return LMMC_STATUS_NUMERICAL_FAILURE;
}
