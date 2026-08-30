/**
 * @file ode_explicit.c
 * @brief 显式 ODE 求解器:Euler,RK45(Cash-Karp 嵌入对)与 RK4.
 */
#include <math.h>

#include "memory_bridge.h"
#include "internal.h"
#include "ode_internal.h"
#include "lmmc/ode.h"

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
            return st;
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
    return out_result->converged ? LMMC_STATUS_OK : LMMC_STATUS_CONVERGENCE_FAILED;
}

/*
 * Cash-Karp embedded Runge-Kutta 4(5) pair.
 * 6 stages, 4th-order solution for stepping, 5th-order for error estimation.
 *
 * Butcher tableau nodes (c_i):
 */
static const lmmc_real_t ck_c[6] = {
    0.0,
    1.0 / 5.0,
    3.0 / 10.0,
    3.0 / 5.0,
    1.0,
    7.0 / 8.0
};

/*
 * Butcher tableau matrix (a_ij), stored row-major.
 * a[i][j] for j < i. Only non-zero entries listed per row.
 */
static const lmmc_real_t ck_a21 = 1.0 / 5.0;

static const lmmc_real_t ck_a31 = 3.0 / 40.0;
static const lmmc_real_t ck_a32 = 9.0 / 40.0;

static const lmmc_real_t ck_a41 = 3.0 / 10.0;
static const lmmc_real_t ck_a42 = -9.0 / 10.0;
static const lmmc_real_t ck_a43 = 6.0 / 5.0;

static const lmmc_real_t ck_a51 = -11.0 / 54.0;
static const lmmc_real_t ck_a52 = 5.0 / 2.0;
static const lmmc_real_t ck_a53 = -70.0 / 27.0;
static const lmmc_real_t ck_a54 = 35.0 / 27.0;

static const lmmc_real_t ck_a61 = 1631.0 / 55296.0;
static const lmmc_real_t ck_a62 = 175.0 / 512.0;
static const lmmc_real_t ck_a63 = 575.0 / 13824.0;
static const lmmc_real_t ck_a64 = 44275.0 / 110592.0;
static const lmmc_real_t ck_a65 = 253.0 / 4096.0;


/* 5th-order weights (for the error estimation solution) */
static const lmmc_real_t ck_b5[6] = {
    37.0 / 378.0,
    0.0,
    250.0 / 621.0,
    125.0 / 594.0,
    0.0,
    512.0 / 1771.0
};

/* Error coefficients: e_i = b5_i - b4_i */
static const lmmc_real_t ck_e[6] = {
    37.0 / 378.0 - 2825.0 / 27648.0,
    0.0,
    250.0 / 621.0 - 18575.0 / 48384.0,
    125.0 / 594.0 - 13525.0 / 55296.0,
    0.0 - 277.0 / 14336.0,
    512.0 / 1771.0 - 1.0 / 4.0
};

lmmc_status_t lmmc_ode_rk45_solve(
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
    lmmc_real_t* k5 = NULL;
    lmmc_real_t* k6 = NULL;
    lmmc_real_t* y_tmp = NULL;
    lmmc_real_t t = t_start;
    lmmc_real_t h = 0.0;
    lmmc_status_t init_st;
    size_t attempts = 0;
    lmmc_status_t failure_status = LMMC_STATUS_NUMERICAL_FAILURE;

    /* Pre-flight validation */
    init_st = validate_and_init_ode_config(rhs, dim, t_start, t_end, y, cfg, &local_cfg, out_result, &work_bytes);
    if (init_st != LMMC_STATUS_OK) {
        return init_st;
    }

    k1 = (lmmc_real_t*)lmmc_alloc(work_bytes);
    k2 = (lmmc_real_t*)lmmc_alloc(work_bytes);
    k3 = (lmmc_real_t*)lmmc_alloc(work_bytes);
    k4 = (lmmc_real_t*)lmmc_alloc(work_bytes);
    k5 = (lmmc_real_t*)lmmc_alloc(work_bytes);
    k6 = (lmmc_real_t*)lmmc_alloc(work_bytes);
    y_tmp = (lmmc_real_t*)lmmc_alloc(work_bytes);

    if (k1 == NULL || k2 == NULL || k3 == NULL || k4 == NULL ||
        k5 == NULL || k6 == NULL || y_tmp == NULL) {
        lmmc_free(y_tmp);
        lmmc_free(k6);
        lmmc_free(k5);
        lmmc_free(k4);
        lmmc_free(k3);
        lmmc_free(k2);
        lmmc_free(k1);
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* Initial step size */
    h = lmmc_clamp(local_cfg.initial_step, local_cfg.min_step, local_cfg.max_step);
    {
        lmmc_real_t span;
        LMMC_REAL_SUB(&span, &t_end, &t_start);
        if (h > span) {
            h = span;
        }
    }

    /* Log initial state */
    lmmc_ode_do_log(&local_cfg, 0, t, y, dim);

    /* Main integration loop */
    while (t < t_end && attempts < local_cfg.max_steps) {
        size_t i = 0;
        lmmc_real_t rem;
        int callback_failed = 0;
        lmmc_status_t st = LMMC_STATUS_OK;
        lmmc_real_t err_norm = 0.0;
        lmmc_real_t h_new = 0.0;
        int step_accepted = 0;
        ++attempts;

        LMMC_REAL_SUB(&rem, &t_end, &t);
        if (rem <= 0.0 || !lmmc_is_finite(&rem)) {
            break;
        }

        if (h > rem) {
            h = rem;
        }

        /* --- Stage 1: k1 = f(t, y) --- */
        st = lmmc_ode_rhs_eval(rhs, t, y, k1, dim, user_data,
                                &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ?
                LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            failure_status = st;
            goto rk45_fail;
        }

        /* --- Stage 2: k2 = f(t + c2*h, y + h*a21*k1) --- */
        for (i = 0; i < dim; ++i) {
            y_tmp[i] = y[i] + h * ck_a21 * k1[i];
        }
        st = lmmc_ode_rhs_eval(rhs, t + ck_c[1] * h, y_tmp, k2, dim, user_data,
                                &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ?
                LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            failure_status = st;
            goto rk45_fail;
        }

        /* --- Stage 3: k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2)) --- */
        for (i = 0; i < dim; ++i) {
            y_tmp[i] = y[i] + h * (ck_a31 * k1[i] + ck_a32 * k2[i]);
        }
        st = lmmc_ode_rhs_eval(rhs, t + ck_c[2] * h, y_tmp, k3, dim, user_data,
                                &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ?
                LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            failure_status = st;
            goto rk45_fail;
        }

        /* --- Stage 4: k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3)) --- */
        for (i = 0; i < dim; ++i) {
            y_tmp[i] = y[i] + h * (ck_a41 * k1[i] + ck_a42 * k2[i] + ck_a43 * k3[i]);
        }
        st = lmmc_ode_rhs_eval(rhs, t + ck_c[3] * h, y_tmp, k4, dim, user_data,
                                &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ?
                LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            failure_status = st;
            goto rk45_fail;
        }

        /* --- Stage 5: k5 = f(t + c5*h, y + h*(a51*k1 + ... + a54*k4)) --- */
        for (i = 0; i < dim; ++i) {
            y_tmp[i] = y[i] + h * (ck_a51 * k1[i] + ck_a52 * k2[i] +
                                    ck_a53 * k3[i] + ck_a54 * k4[i]);
        }
        st = lmmc_ode_rhs_eval(rhs, t + ck_c[4] * h, y_tmp, k5, dim, user_data,
                                &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ?
                LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            failure_status = st;
            goto rk45_fail;
        }

        /* --- Stage 6: k6 = f(t + c6*h, y + h*(a61*k1 + ... + a65*k5)) --- */
        for (i = 0; i < dim; ++i) {
            y_tmp[i] = y[i] + h * (ck_a61 * k1[i] + ck_a62 * k2[i] +
                                    ck_a63 * k3[i] + ck_a64 * k4[i] +
                                    ck_a65 * k5[i]);
        }
        st = lmmc_ode_rhs_eval(rhs, t + ck_c[5] * h, y_tmp, k6, dim, user_data,
                                &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ?
                LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            failure_status = st;
            goto rk45_fail;
        }

        /* 先构造候选解, 再用新旧状态共同缩放嵌入误差. */
        err_norm = 0.0;
        for (i = 0; i < dim; ++i) {
            lmmc_real_t err_i = h * (ck_e[0] * k1[i] + ck_e[2] * k3[i] +
                                     ck_e[3] * k4[i] + ck_e[4] * k5[i] +
                                     ck_e[5] * k6[i]);
            lmmc_real_t scale;
            lmmc_real_t ratio;
            y_tmp[i] = y[i] + h * (ck_b5[0] * k1[i] + ck_b5[2] * k3[i] +
                                   ck_b5[3] * k4[i] + ck_b5[5] * k6[i]);
            scale = local_cfg.abs_tol +
                    local_cfg.rel_tol * fmax(fabs(y[i]), fabs(y_tmp[i]));
            ratio = err_i / scale;
            err_norm += ratio * ratio;
        }
        err_norm = sqrt(err_norm / (lmmc_real_t)dim);

        /* --- Step acceptance / rejection --- */
        if (err_norm <= 1.0) {
            step_accepted = 1;
            memcpy(y, y_tmp, work_bytes);

            /* Check for non-finite state */
            if (!lmmc_ode_state_is_finite(y, dim)) {
                out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto rk45_fail;
            }

            /* Advance time */
            t += h;
            if (!lmmc_is_finite(&t)) {
                out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto rk45_fail;
            }

            out_result->num_steps += 1;
            out_result->final_t = t;
            lmmc_ode_do_log(&local_cfg, out_result->num_steps, t, y, dim);
        }

        if (err_norm > 0.0 && isfinite(err_norm)) {
            lmmc_real_t factor = local_cfg.adaptive_step_beta * pow(err_norm, -0.2);
            factor = lmmc_clamp(factor, 0.2, 5.0);
            h_new = lmmc_clamp(h * factor, local_cfg.min_step, local_cfg.max_step);
        } else {
            h_new = lmmc_clamp(h * 5.0, local_cfg.min_step, local_cfg.max_step);
        }

        if (!step_accepted) {
            /** min_step 生效后,持续超出容差即报告步长失败. */
            if (h_new <= local_cfg.min_step && err_norm > 1.0) {
                if (h <= local_cfg.min_step) {
                    out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
                    failure_status = LMMC_STATUS_CONVERGENCE_FAILED;
                    goto rk45_fail;
                }
            }
        }

        h = h_new;
    }

    if (t >= t_end) {
        out_result->converged = 1;
        out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
    } else {
        out_result->converged = 0;
        out_result->failure_reason = LMMC_ODE_FAILURE_MAX_STEPS;
        lmmc_free(y_tmp);
        lmmc_free(k6);
        lmmc_free(k5);
        lmmc_free(k4);
        lmmc_free(k3);
        lmmc_free(k2);
        lmmc_free(k1);
        return LMMC_STATUS_CONVERGENCE_FAILED;
    }

    lmmc_free(y_tmp);
    lmmc_free(k6);
    lmmc_free(k5);
    lmmc_free(k4);
    lmmc_free(k3);
    lmmc_free(k2);
    lmmc_free(k1);
    return LMMC_STATUS_OK;

rk45_fail:
    lmmc_free(y_tmp);
    lmmc_free(k6);
    lmmc_free(k5);
    lmmc_free(k4);
    lmmc_free(k3);
    lmmc_free(k2);
    lmmc_free(k1);
    return failure_status;
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
    return out_result->converged ? LMMC_STATUS_OK : LMMC_STATUS_CONVERGENCE_FAILED;

rk4_fail:
    lmmc_free(y_tmp);
    lmmc_free(k4);
    lmmc_free(k3);
    lmmc_free(k2);
    lmmc_free(k1);
    return LMMC_STATUS_NUMERICAL_FAILURE;
}
