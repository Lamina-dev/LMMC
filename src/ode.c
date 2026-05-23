/**
 * @file ode.c
 * @brief ODE 初值问题求解器实现：Euler / RK4 / RK45。
 */
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/ode.h"

static void lmmc_ode_do_log(const lmmc_ode_config_t* cfg, size_t step, lmmc_real_t t, const lmmc_real_t* y, size_t dim) {
    if (cfg->log_cb != NULL) {
        cfg->log_cb(step, t, y, dim, cfg->log_user_data);
    } else if (cfg->verbose) {
        printf("Step %zu: t = %.10e, y[0] = %.10e\n", step, t, dim > 0 ? y[0] : 0.0);
    }
}

static void lmmc_ode_reset_result(lmmc_ode_result_t* out_result, lmmc_real_t t_start) {
    if (out_result == NULL) {
        return;
    }

    out_result->converged = 0;
    out_result->num_steps = 0;
    out_result->num_rhs_evals = 0;
    out_result->final_t = t_start;
    out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
}

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
    out_cfg->verbose = 0;
    out_cfg->log_cb = NULL;
    out_cfg->log_user_data = NULL;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_ode_load_and_validate_config(
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
        out_cfg->abs_tol < 0.0 || out_cfg->rel_tol < 0.0) {
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

static lmmc_status_t lmmc_ode_rhs_eval(
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
    st = rhs(t, y, y_prime, dim, user_data);
    if (st != LMMC_STATUS_OK) {
        *out_callback_failed = 1;
        return st;
    }

    *io_eval_count += 1;
    for (i = 0; i < dim; ++i) {
        if (!lmmc_is_finite(&y_prime[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    return LMMC_STATUS_OK;
}

static int lmmc_ode_state_is_finite(const lmmc_real_t* y, size_t dim) {
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

static lmmc_status_t validate_and_init_ode_config(
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
    if (rhs == NULL || y == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_ode_reset_result(out_result, t_start);

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


lmmc_status_t lmmc_ode_euler_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    lmmc_real_t* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
) {
    lmmc_ode_config_t local_cfg = {0};
    size_t work_bytes = 0;
    lmmc_real_t* y_prime = NULL;
    lmmc_real_t t = t_start;
    lmmc_real_t h = 0.0;
    lmmc_status_t init_st;

    init_st = validate_and_init_ode_config(rhs, dim, t_start, t_end, y, cfg, &local_cfg, out_result, &work_bytes);
    if (init_st != LMMC_STATUS_OK) {
        return init_st;
    }

    y_prime = (lmmc_real_t*)lmmc_alloc(work_bytes);
    if (y_prime == NULL) {
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    h = lmmc_clamp(local_cfg.initial_step, local_cfg.min_step, local_cfg.max_step);
    {
        lmmc_real_t span;
        LMMC_REAL_SUB(&span, &t_end, &t_start);
        if (h > span) {
            h = span;
        }
    }

    lmmc_ode_do_log(&local_cfg, 0, t, y, dim);

    while (t < t_end && out_result->num_steps < local_cfg.max_steps) {
        size_t i = 0;
        lmmc_real_t rem;
        int callback_failed = 0;
        lmmc_status_t st = LMMC_STATUS_OK;

        LMMC_REAL_SUB(&rem, &t_end, &t);
        if (rem <= 0.0 || !lmmc_is_finite(&rem)) {
            break;
        }

        if (h > rem) {
            h = rem;
        }

        st = lmmc_ode_rhs_eval(rhs, t, y, y_prime, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            lmmc_free(y_prime);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        for (i = 0; i < dim; ++i) {
            lmmc_real_t tmp;
            LMMC_REAL_MUL(&tmp, &h, &y_prime[i]);
            LMMC_REAL_ADD(&y[i], &y[i], &tmp);
        }

        if (!lmmc_ode_state_is_finite(y, dim)) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            lmmc_free(y_prime);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        LMMC_REAL_ADD(&t, &t, &h);
        if (!lmmc_is_finite(&t)) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            lmmc_free(y_prime);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        out_result->num_steps += 1;
        out_result->final_t = t;
        lmmc_ode_do_log(&local_cfg, out_result->num_steps, t, y, dim);
    }

    if (t >= t_end) {
        out_result->converged = 1;
        out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
    } else {
        out_result->converged = 0;
        out_result->failure_reason = LMMC_ODE_FAILURE_MAX_STEPS;
    }

    lmmc_free(y_prime);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ode_rk4_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    lmmc_real_t* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
) {
    lmmc_ode_config_t local_cfg = {0};
    size_t work_bytes = 0;
    lmmc_real_t* k1 = NULL;
    lmmc_real_t* k2 = NULL;
    lmmc_real_t* k3 = NULL;
    lmmc_real_t* k4 = NULL;
    lmmc_real_t* y_tmp = NULL;
    lmmc_real_t t = t_start;
    lmmc_real_t h = 0.0;
    lmmc_status_t init_st;

    init_st = validate_and_init_ode_config(rhs, dim, t_start, t_end, y, cfg, &local_cfg, out_result, &work_bytes);
    if (init_st != LMMC_STATUS_OK) {
        return init_st;
    }

    k1 = (lmmc_real_t*)lmmc_alloc(work_bytes);
    k2 = (lmmc_real_t*)lmmc_alloc(work_bytes);
    k3 = (lmmc_real_t*)lmmc_alloc(work_bytes);
    k4 = (lmmc_real_t*)lmmc_alloc(work_bytes);
    y_tmp = (lmmc_real_t*)lmmc_alloc(work_bytes);

    if (k1 == NULL || k2 == NULL || k3 == NULL || k4 == NULL || y_tmp == NULL) {
        lmmc_free(y_tmp);
        lmmc_free(k4);
        lmmc_free(k3);
        lmmc_free(k2);
        lmmc_free(k1);
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    h = lmmc_clamp(local_cfg.initial_step, local_cfg.min_step, local_cfg.max_step);
    {
        lmmc_real_t span;
        LMMC_REAL_SUB(&span, &t_end, &t_start);
        if (h > span) {
            h = span;
        }
    }

    lmmc_ode_do_log(&local_cfg, 0, t, y, dim);

    while (t < t_end && out_result->num_steps < local_cfg.max_steps) {
        size_t i = 0;
        lmmc_real_t rem;
        int callback_failed = 0;
        lmmc_status_t st = LMMC_STATUS_OK;
        lmmc_real_t half_h;
        lmmc_real_t t_mid;
        lmmc_real_t t_next;
        lmmc_real_t half = 0.5;
        lmmc_real_t sixth;
        lmmc_real_t two = 2.0;
        lmmc_real_t six = 6.0;
        lmmc_real_t h_over_6;

        LMMC_REAL_SUB(&rem, &t_end, &t);
        if (rem <= 0.0 || !lmmc_is_finite(&rem)) {
            break;
        }

        if (h > rem) {
            h = rem;
        }


        st = lmmc_ode_rhs_eval(rhs, t, y, k1, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto rk4_fail;
        }


        LMMC_REAL_MUL(&half_h, &half, &h);
        for (i = 0; i < dim; ++i) {
            lmmc_real_t tmp;
            LMMC_REAL_MUL(&tmp, &half_h, &k1[i]);
            LMMC_REAL_ADD(&y_tmp[i], &y[i], &tmp);
        }


        LMMC_REAL_ADD(&t_mid, &t, &half_h);
        st = lmmc_ode_rhs_eval(rhs, t_mid, y_tmp, k2, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto rk4_fail;
        }


        for (i = 0; i < dim; ++i) {
            lmmc_real_t tmp;
            LMMC_REAL_MUL(&tmp, &half_h, &k2[i]);
            LMMC_REAL_ADD(&y_tmp[i], &y[i], &tmp);
        }


        st = lmmc_ode_rhs_eval(rhs, t_mid, y_tmp, k3, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto rk4_fail;
        }


        for (i = 0; i < dim; ++i) {
            lmmc_real_t tmp;
            LMMC_REAL_MUL(&tmp, &h, &k3[i]);
            LMMC_REAL_ADD(&y_tmp[i], &y[i], &tmp);
        }


        LMMC_REAL_ADD(&t_next, &t, &h);
        st = lmmc_ode_rhs_eval(rhs, t_next, y_tmp, k4, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto rk4_fail;
        }


        LMMC_REAL_DIV(&h_over_6, &h, &six);
        for (i = 0; i < dim; ++i) {
            lmmc_real_t t2k2, t2k3, sum1, sum2, sum3, weighted;
            LMMC_REAL_MUL(&t2k2, &two, &k2[i]);
            LMMC_REAL_MUL(&t2k3, &two, &k3[i]);
            LMMC_REAL_ADD(&sum1, &k1[i], &t2k2);
            LMMC_REAL_ADD(&sum2, &sum1, &t2k3);
            LMMC_REAL_ADD(&sum3, &sum2, &k4[i]);
            LMMC_REAL_MUL(&weighted, &h_over_6, &sum3);
            LMMC_REAL_ADD(&y[i], &y[i], &weighted);
        }

        if (!lmmc_ode_state_is_finite(y, dim)) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto rk4_fail;
        }

        LMMC_REAL_ADD(&t, &t, &h);
        if (!lmmc_is_finite(&t)) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto rk4_fail;
        }

        out_result->num_steps += 1;
        out_result->final_t = t;
        lmmc_ode_do_log(&local_cfg, out_result->num_steps, t, y, dim);
    }

    if (t >= t_end) {
        out_result->converged = 1;
        out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
    } else {
        out_result->converged = 0;
        out_result->failure_reason = LMMC_ODE_FAILURE_MAX_STEPS;
    }

    lmmc_free(y_tmp);
    lmmc_free(k4);
    lmmc_free(k3);
    lmmc_free(k2);
    lmmc_free(k1);
    return LMMC_STATUS_OK;

rk4_fail:
    lmmc_free(y_tmp);
    lmmc_free(k4);
    lmmc_free(k3);
    lmmc_free(k2);
    lmmc_free(k1);
    return LMMC_STATUS_NUMERICAL_FAILURE;
}
