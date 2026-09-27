/**
 * @file ode_explicit.c
 * @brief 显式 ODE 求解器:Euler,RK45(Cash-Karp 嵌入对)与 RK4.
 */
#include <math.h>

#include "memory_bridge.h"
#include "internal.h"
#include "ode_internal.h"
#include "lmmc/ode.h"

typedef struct {
    lmmc_ode_rhs_t rhs;
    void* user_data;
    size_t dim, work_bytes;
    lmmc_real_t* y;
    lmmc_ode_config_t local_cfg;
    lmmc_ode_result_t* out_result;
    lmmc_real_t t, h;
    lmmc_real_t* k[6];
    lmmc_real_t* y_tmp;
} ode_explicit_state_t;

static void explicit_free(ode_explicit_state_t* s, size_t stages) {
    lmmc_memory_free(s->y_tmp);
    while (stages > 0) {
        --stages;
        lmmc_memory_free(s->k[stages]);
    }
}

static lmmc_status_t explicit_allocate(ode_explicit_state_t* s, size_t stages) {
    int allocated = 1;
    for (size_t stage = 0; stage < stages; ++stage) {
        s->k[stage] = (lmmc_real_t*)lmmc_memory_alloc(s->work_bytes);
        if (s->k[stage] == NULL) { allocated = 0; }
    }
    s->y_tmp = (lmmc_real_t*)lmmc_memory_alloc(s->work_bytes);
    if (s->y_tmp == NULL) { allocated = 0; }
    if (!allocated) {
        explicit_free(s, stages);
        s->out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t explicit_rhs(
    ode_explicit_state_t* s, lmmc_real_t t, const lmmc_real_t* y, lmmc_real_t* derivative
) {
    int callback_failed = 0;
    lmmc_status_t st = lmmc_ode_rhs_eval(s->rhs, t, y, derivative, s->dim,
        s->user_data, &s->out_result->num_rhs_evals, &callback_failed);
    if (st != LMMC_STATUS_OK) {
        s->out_result->failure_reason = callback_failed ?
            LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
    }
    return st;
}

static lmmc_status_t explicit_advance(ode_explicit_state_t* s) {
    if (!lmmc_ode_state_is_finite(s->y, s->dim)) {
        s->out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_ADD(&s->t, &s->t, &s->h);
    if (!lmmc_is_finite(&s->t)) {
        s->out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    s->out_result->num_steps += 1;
    s->out_result->final_t = s->t;
    lmmc_ode_do_log(&s->local_cfg, s->out_result->num_steps, s->t, s->y, s->dim);
    return LMMC_STATUS_OK;
}

static lmmc_status_t explicit_finish(ode_explicit_state_t* s, lmmc_real_t t_end) {
    if (s->t >= t_end) {
        s->out_result->converged = 1;
        s->out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
    } else {
        s->out_result->converged = 0;
        s->out_result->failure_reason = LMMC_ODE_FAILURE_MAX_STEPS;
    }
    return s->out_result->converged ? LMMC_STATUS_OK : LMMC_STATUS_CONVERGENCE_FAILED;
}

static lmmc_status_t euler_integrate(ode_explicit_state_t* s, lmmc_real_t t_end) {
    while (s->t < t_end && s->out_result->num_steps < s->local_cfg.max_steps) {
        size_t i = 0;
        lmmc_real_t rem;
        lmmc_status_t st = LMMC_STATUS_OK;

        LMMC_REAL_SUB(&rem, &t_end, &s->t);
        if (rem <= 0.0 || !lmmc_is_finite(&rem)) {
            break;
        }

        if (s->h > rem) {
            s->h = rem;
        }

        st = explicit_rhs(s, s->t, s->y, s->k[0]);
        if (st != LMMC_STATUS_OK) { return st; }
        for (i = 0; i < s->dim; ++i) {
            lmmc_real_t tmp;
            LMMC_REAL_MUL(&tmp, &s->h, &s->k[0][i]);
            LMMC_REAL_ADD(&s->y[i], &s->y[i], &tmp);
        }
        st = explicit_advance(s);
        if (st != LMMC_STATUS_OK) { return st; }
    }
    return explicit_finish(s, t_end);
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
    ode_explicit_state_t state = {0};
    ode_explicit_state_t* s = &state;
    s->rhs = rhs;
    s->user_data = user_data;
    s->dim = dim;
    s->y = y;
    s->out_result = out_result;
    s->t = t_start;
    lmmc_status_t st = validate_and_init_ode_config(rhs, dim, t_start, t_end, y,
        cfg, &s->local_cfg, out_result, &s->work_bytes);
    if (st != LMMC_STATUS_OK) { return st; }
    s->k[0] = (lmmc_real_t*)lmmc_memory_alloc(s->work_bytes);
    if (s->k[0] == NULL) {
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    s->h = lmmc_clamp(s->local_cfg.initial_step, s->local_cfg.min_step, s->local_cfg.max_step);
    {
        lmmc_real_t span;
        LMMC_REAL_SUB(&span, &t_end, &t_start);
        if (s->h > span) {
            s->h = span;
        }
    }

    lmmc_ode_do_log(&s->local_cfg, 0, s->t, s->y, s->dim);
    st = euler_integrate(s, t_end);
    lmmc_memory_free(s->k[0]);
    return st;
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

static lmmc_status_t cash_karp_stages(ode_explicit_state_t* s) {
    size_t i;
    lmmc_status_t st;
        st = explicit_rhs(s, s->t, s->y, s->k[0]);
        if (st != LMMC_STATUS_OK) { return st; }

        for (i = 0; i < s->dim; ++i) {
            s->y_tmp[i] = s->y[i] + s->h * ck_a21 * s->k[0][i];
        }
        st = explicit_rhs(s, s->t + ck_c[1] * s->h, s->y_tmp, s->k[1]);
        if (st != LMMC_STATUS_OK) { return st; }

        for (i = 0; i < s->dim; ++i) {
            s->y_tmp[i] = s->y[i] + s->h * (ck_a31 * s->k[0][i] + ck_a32 * s->k[1][i]);
        }
        st = explicit_rhs(s, s->t + ck_c[2] * s->h, s->y_tmp, s->k[2]);
        if (st != LMMC_STATUS_OK) { return st; }

        for (i = 0; i < s->dim; ++i) {
            s->y_tmp[i] = s->y[i] + s->h * (ck_a41 * s->k[0][i] + ck_a42 * s->k[1][i] + ck_a43 * s->k[2][i]);
        }
        st = explicit_rhs(s, s->t + ck_c[3] * s->h, s->y_tmp, s->k[3]);
        if (st != LMMC_STATUS_OK) { return st; }

        for (i = 0; i < s->dim; ++i) {
            s->y_tmp[i] = s->y[i] + s->h * (ck_a51 * s->k[0][i] + ck_a52 * s->k[1][i] +
                                    ck_a53 * s->k[2][i] + ck_a54 * s->k[3][i]);
        }
        st = explicit_rhs(s, s->t + ck_c[4] * s->h, s->y_tmp, s->k[4]);
        if (st != LMMC_STATUS_OK) { return st; }

        for (i = 0; i < s->dim; ++i) {
            s->y_tmp[i] = s->y[i] + s->h * (ck_a61 * s->k[0][i] + ck_a62 * s->k[1][i] +
                                    ck_a63 * s->k[2][i] + ck_a64 * s->k[3][i] +
                                    ck_a65 * s->k[4][i]);
        }
        st = explicit_rhs(s, s->t + ck_c[5] * s->h, s->y_tmp, s->k[5]);
        if (st != LMMC_STATUS_OK) { return st; }
    return LMMC_STATUS_OK;
}

static lmmc_real_t cash_karp_error(ode_explicit_state_t* s) {
    size_t i;
    lmmc_real_t err_norm;
        /* 先构造候选解, 再用新旧状态共同缩放嵌入误差. */
        err_norm = 0.0;
        for (i = 0; i < s->dim; ++i) {
            lmmc_real_t err_i = s->h * (ck_e[0] * s->k[0][i] + ck_e[2] * s->k[2][i] +
                                     ck_e[3] * s->k[3][i] + ck_e[4] * s->k[4][i] +
                                     ck_e[5] * s->k[5][i]);
            lmmc_real_t scale;
            lmmc_real_t ratio;
            s->y_tmp[i] = s->y[i] + s->h * (ck_b5[0] * s->k[0][i] + ck_b5[2] * s->k[2][i] +
                                   ck_b5[3] * s->k[3][i] + ck_b5[5] * s->k[5][i]);
            scale = s->local_cfg.abs_tol +
                    s->local_cfg.rel_tol * fmax(fabs(s->y[i]), fabs(s->y_tmp[i]));
            ratio = err_i / scale;
            err_norm += ratio * ratio;
        }
        err_norm = sqrt(err_norm / (lmmc_real_t)s->dim);
    return err_norm;
}

static lmmc_status_t cash_karp_control(ode_explicit_state_t* s, lmmc_real_t err_norm, int step_accepted) {
    lmmc_real_t h_new;
        if (err_norm > 0.0 && isfinite(err_norm)) {
            lmmc_real_t factor = s->local_cfg.adaptive_step_beta * pow(err_norm, -0.2);
            factor = lmmc_clamp(factor, 0.2, 5.0);
            h_new = lmmc_clamp(s->h * factor, s->local_cfg.min_step, s->local_cfg.max_step);
        } else {
            h_new = lmmc_clamp(s->h * 5.0, s->local_cfg.min_step, s->local_cfg.max_step);
        }

        if (!step_accepted) {
            /** min_step 生效后,持续超出容差即报告步长失败. */
            if (h_new <= s->local_cfg.min_step && err_norm > 1.0) {
                if (s->h <= s->local_cfg.min_step) {
                    s->out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
                    return LMMC_STATUS_CONVERGENCE_FAILED;
                }
            }
        }

        s->h = h_new;
    return LMMC_STATUS_OK;
}

static lmmc_status_t cash_karp_integrate(ode_explicit_state_t* s, lmmc_real_t t_end) {
    size_t attempts = 0;
    while (s->t < t_end && attempts < s->local_cfg.max_steps) {
        lmmc_real_t rem;
        lmmc_status_t st = LMMC_STATUS_OK;
        lmmc_real_t err_norm = 0.0;
        int step_accepted = 0;
        ++attempts;

        LMMC_REAL_SUB(&rem, &t_end, &s->t);
        if (rem <= 0.0 || !lmmc_is_finite(&rem)) {
            break;
        }

        if (s->h > rem) {
            s->h = rem;
        }
        st = cash_karp_stages(s);
        if (st != LMMC_STATUS_OK) { return st; }
        err_norm = cash_karp_error(s);
        if (err_norm <= 1.0) {
            step_accepted = 1;
            memcpy(s->y, s->y_tmp, s->work_bytes);
            st = explicit_advance(s);
            if (st != LMMC_STATUS_OK) { return st; }
        }
        st = cash_karp_control(s, err_norm, step_accepted);
        if (st != LMMC_STATUS_OK) { return st; }
    }
    return explicit_finish(s, t_end);
}

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
    ode_explicit_state_t state = {0};
    ode_explicit_state_t* s = &state;
    s->rhs = rhs;
    s->user_data = user_data;
    s->dim = dim;
    s->y = y;
    s->out_result = out_result;
    s->t = t_start;
    lmmc_status_t st = validate_and_init_ode_config(rhs, dim, t_start, t_end, y,
        cfg, &s->local_cfg, out_result, &s->work_bytes);
    if (st != LMMC_STATUS_OK) { return st; }
    st = explicit_allocate(s, 6);
    if (st != LMMC_STATUS_OK) { return st; }
    s->h = lmmc_clamp(s->local_cfg.initial_step, s->local_cfg.min_step, s->local_cfg.max_step);
    {
        lmmc_real_t span;
        LMMC_REAL_SUB(&span, &t_end, &t_start);
        if (s->h > span) {
            s->h = span;
        }
    }

    lmmc_ode_do_log(&s->local_cfg, 0, s->t, s->y, s->dim);
    st = cash_karp_integrate(s, t_end);
    explicit_free(s, 6);
    return st;
}

static lmmc_status_t rk4_step(ode_explicit_state_t* s) {
    size_t i;
    lmmc_status_t st;
        lmmc_real_t half_h;
        lmmc_real_t t_mid;
        lmmc_real_t t_next;
        lmmc_real_t half = 0.5;
        lmmc_real_t two = 2.0;
        lmmc_real_t six = 6.0;
        lmmc_real_t h_over_6;
        st = explicit_rhs(s, s->t, s->y, s->k[0]);
        if (st != LMMC_STATUS_OK) { return st; }

        LMMC_REAL_MUL(&half_h, &half, &s->h);
        for (i = 0; i < s->dim; ++i) {
            lmmc_real_t tmp;
            LMMC_REAL_MUL(&tmp, &half_h, &s->k[0][i]);
            LMMC_REAL_ADD(&s->y_tmp[i], &s->y[i], &tmp);
        }

        LMMC_REAL_ADD(&t_mid, &s->t, &half_h);
        st = explicit_rhs(s, t_mid, s->y_tmp, s->k[1]);
        if (st != LMMC_STATUS_OK) { return st; }

        for (i = 0; i < s->dim; ++i) {
            lmmc_real_t tmp;
            LMMC_REAL_MUL(&tmp, &half_h, &s->k[1][i]);
            LMMC_REAL_ADD(&s->y_tmp[i], &s->y[i], &tmp);
        }

        st = explicit_rhs(s, t_mid, s->y_tmp, s->k[2]);
        if (st != LMMC_STATUS_OK) { return st; }

        for (i = 0; i < s->dim; ++i) {
            lmmc_real_t tmp;
            LMMC_REAL_MUL(&tmp, &s->h, &s->k[2][i]);
            LMMC_REAL_ADD(&s->y_tmp[i], &s->y[i], &tmp);
        }

        LMMC_REAL_ADD(&t_next, &s->t, &s->h);
        st = explicit_rhs(s, t_next, s->y_tmp, s->k[3]);
        if (st != LMMC_STATUS_OK) { return st; }

        LMMC_REAL_DIV(&h_over_6, &s->h, &six);
        for (i = 0; i < s->dim; ++i) {
            lmmc_real_t t2k2, t2k3, sum1, sum2, sum3, weighted;
            LMMC_REAL_MUL(&t2k2, &two, &s->k[1][i]);
            LMMC_REAL_MUL(&t2k3, &two, &s->k[2][i]);
            LMMC_REAL_ADD(&sum1, &s->k[0][i], &t2k2);
            LMMC_REAL_ADD(&sum2, &sum1, &t2k3);
            LMMC_REAL_ADD(&sum3, &sum2, &s->k[3][i]);
            LMMC_REAL_MUL(&weighted, &h_over_6, &sum3);
            LMMC_REAL_ADD(&s->y[i], &s->y[i], &weighted);
        }
    return explicit_advance(s);
}

static lmmc_status_t rk4_integrate(ode_explicit_state_t* s, lmmc_real_t t_end) {
    while (s->t < t_end && s->out_result->num_steps < s->local_cfg.max_steps) {
        lmmc_real_t rem;
        lmmc_status_t st = LMMC_STATUS_OK;
        LMMC_REAL_SUB(&rem, &t_end, &s->t);
        if (rem <= 0.0 || !lmmc_is_finite(&rem)) {
            break;
        }

        if (s->h > rem) {
            s->h = rem;
        }
        st = rk4_step(s);
        if (st != LMMC_STATUS_OK) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    }
    return explicit_finish(s, t_end);
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
    ode_explicit_state_t state = {0};
    ode_explicit_state_t* s = &state;
    s->rhs = rhs;
    s->user_data = user_data;
    s->dim = dim;
    s->y = y;
    s->out_result = out_result;
    s->t = t_start;
    lmmc_status_t st = validate_and_init_ode_config(rhs, dim, t_start, t_end, y,
        cfg, &s->local_cfg, out_result, &s->work_bytes);
    if (st != LMMC_STATUS_OK) { return st; }
    st = explicit_allocate(s, 4);
    if (st != LMMC_STATUS_OK) { return st; }
    s->h = lmmc_clamp(s->local_cfg.initial_step, s->local_cfg.min_step, s->local_cfg.max_step);
    {
        lmmc_real_t span;
        LMMC_REAL_SUB(&span, &t_end, &t_start);
        if (s->h > span) {
            s->h = span;
        }
    }

    lmmc_ode_do_log(&s->local_cfg, 0, s->t, s->y, s->dim);
    st = rk4_integrate(s, t_end);
    explicit_free(s, 4);
    return st;
}
