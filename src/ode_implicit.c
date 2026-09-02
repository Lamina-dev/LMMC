/**
 * @file ode_implicit.c
 * @brief 隐式 ODE 求解器：隐式 Euler 与梯形法（Newton 迭代）。
 */
#include <math.h>
#include <string.h>

#include "memory_bridge.h"
#include "internal.h"
#include "ode_internal.h"
#include "lmmc/ode.h"
#include "lmmc/optimize.h"
#include "lmmc/dense.h"
#include "lmmc/linear_algebra.h"

typedef struct {
    lmmc_ode_rhs_t rhs;
    void* user_data;
    lmmc_ode_jac_t jac_cb;
    lmmc_real_t t_next;
    lmmc_real_t h;
    const lmmc_real_t* y_n;
    size_t dim;
    lmmc_real_t* jac_base_rhs;
    lmmc_real_t* jac_y_perturbed;
    lmmc_real_t* jac_rhs_perturbed;
    size_t* rhs_eval_count;
    int callback_failed;
} ode_implicit_euler_ctx_t;

static lmmc_status_t implicit_euler_F(
    const lmmc_vec_t* x, lmmc_vec_t* F, void* ud
) {
    ode_implicit_euler_ctx_t* ctx = (ode_implicit_euler_ctx_t*)ud;
    size_t i;
    lmmc_status_t st = lmmc_ode_rhs_eval(
        ctx->rhs, ctx->t_next, x->data, F->data, ctx->dim, ctx->user_data,
        ctx->rhs_eval_count, &ctx->callback_failed);
    if (st != LMMC_STATUS_OK) return st;
    /* G(z) = z - y_n - h*f(t_{n+1}, z) */
    for (i = 0; i < ctx->dim; ++i) {
        F->data[i] = x->data[i] - ctx->y_n[i] - ctx->h * F->data[i];
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t implicit_euler_J(
    const lmmc_vec_t* x, lmmc_mat_t* J, void* ud
) {
    ode_implicit_euler_ctx_t* ctx = (ode_implicit_euler_ctx_t*)ud;
    size_t i, j, n = ctx->dim;
    lmmc_real_t* jac_data = J->data;

    {
        lmmc_status_t st = lmmc_ode_jacobian_eval(
            ctx->rhs, ctx->jac_cb, ctx->user_data, ctx->t_next, x->data,
            NULL, n, jac_data, ctx->jac_base_rhs, ctx->jac_y_perturbed,
            ctx->jac_rhs_perturbed, ctx->rhs_eval_count,
            &ctx->callback_failed);
        if (st != LMMC_STATUS_OK) return st;
    }
    /* J_G = I - h * J_f */
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            jac_data[i * J->stride + j] = -ctx->h * jac_data[i * J->stride + j];
        }
        jac_data[i * J->stride + i] += 1.0;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ode_implicit_euler_solve(
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
    size_t total_work_bytes = 0;
    lmmc_real_t t = t_start;
    lmmc_real_t h = 0.0;
    lmmc_status_t init_st;
    lmmc_vec_t x_vec = {0};
    lmmc_optimize_config_t opt_cfg;
    lmmc_optimize_result_t opt_res;
    lmmc_real_t* work = NULL;
    ode_implicit_euler_ctx_t ctx;

    init_st = validate_and_init_ode_config(rhs, dim, t_start, t_end, y,
                                            cfg, &local_cfg, out_result, &work_bytes);
    if (init_st != LMMC_STATUS_OK) return init_st;

    h = lmmc_clamp(local_cfg.initial_step, local_cfg.min_step, local_cfg.max_step);
    { lmmc_real_t span = t_end - t_start; if (h > span) h = span; }

    /* Setup Newton config */
    lmmc_optimize_default_config(&opt_cfg);
    opt_cfg.abs_tol = local_cfg.abs_tol;
    opt_cfg.rel_tol = local_cfg.rel_tol;
    opt_cfg.max_iter = 50;

    if (!lmmc_safe_mul_size(work_bytes, 4, &total_work_bytes)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    work = (lmmc_real_t*)lmmc_alloc(total_work_bytes);
    if (work == NULL) {
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    x_vec.size = dim;
    x_vec.data = work;
    x_vec.owns_data = 0;

    ctx.rhs = rhs;
    ctx.user_data = user_data;
    ctx.jac_cb = local_cfg.jacobian;
    ctx.dim = dim;

    ctx.jac_base_rhs = work + dim;
    ctx.jac_y_perturbed = work + 2 * dim;
    ctx.jac_rhs_perturbed = work + 3 * dim;
    ctx.rhs_eval_count = &out_result->num_rhs_evals;
    ctx.callback_failed = 0;
    lmmc_ode_do_log(&local_cfg, 0, t, y, dim);

    while (t < t_end && out_result->num_steps < local_cfg.max_steps) {
        lmmc_real_t rem = t_end - t;
        lmmc_status_t st;
        if (rem <= 0.0) break;
        if (h > rem) h = rem;

        ctx.t_next = t + h;
        ctx.h = h;
        ctx.y_n = y;

        /* Initial guess: explicit Euler prediction */
        memcpy(x_vec.data, y, dim * sizeof(lmmc_real_t));

        st = lmmc_nleq_newton(implicit_euler_F, implicit_euler_J,
                               &ctx, &x_vec, &opt_cfg, &opt_res);
        if (st != LMMC_STATUS_OK || !opt_res.converged) {
            out_result->failure_reason =
                ctx.callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                    : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            lmmc_free(work);
            return st != LMMC_STATUS_OK ? st : LMMC_STATUS_NUMERICAL_FAILURE;
        }

        memcpy(y, x_vec.data, dim * sizeof(lmmc_real_t));

        if (!lmmc_ode_state_is_finite(y, dim)) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            lmmc_free(work);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        t += h;
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

    lmmc_free(work);
    return out_result->converged ? LMMC_STATUS_OK : LMMC_STATUS_CONVERGENCE_FAILED;
}

typedef struct {
    lmmc_ode_rhs_t rhs;
    void* user_data;
    lmmc_ode_jac_t jac_cb;
    lmmc_real_t t_next;
    lmmc_real_t h;
    const lmmc_real_t* y_n;
    lmmc_real_t* f_n;
    size_t dim;
    lmmc_real_t* jac_base_rhs;
    lmmc_real_t* jac_y_perturbed;
    lmmc_real_t* jac_rhs_perturbed;
    size_t* rhs_eval_count;
    int callback_failed;
} ode_trapezoidal_ctx_t;

static lmmc_status_t trapezoidal_F(
    const lmmc_vec_t* x, lmmc_vec_t* F, void* ud
) {
    ode_trapezoidal_ctx_t* ctx = (ode_trapezoidal_ctx_t*)ud;
    size_t i;
    lmmc_status_t st = lmmc_ode_rhs_eval(
        ctx->rhs, ctx->t_next, x->data, F->data, ctx->dim, ctx->user_data,
        ctx->rhs_eval_count, &ctx->callback_failed);
    if (st != LMMC_STATUS_OK) return st;
    /* G(z) = z - y_n - h/2*(f_n + f(t_{n+1}, z)) */
    for (i = 0; i < ctx->dim; ++i) {
        F->data[i] = x->data[i] - ctx->y_n[i]
                     - 0.5 * ctx->h * (ctx->f_n[i] + F->data[i]);
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t trapezoidal_J(
    const lmmc_vec_t* x, lmmc_mat_t* J, void* ud
) {
    ode_trapezoidal_ctx_t* ctx = (ode_trapezoidal_ctx_t*)ud;
    size_t i, j, n = ctx->dim;
    lmmc_real_t* jac_data = J->data;

    {
        lmmc_status_t st = lmmc_ode_jacobian_eval(
            ctx->rhs, ctx->jac_cb, ctx->user_data, ctx->t_next, x->data,
            NULL, n, jac_data, ctx->jac_base_rhs, ctx->jac_y_perturbed,
            ctx->jac_rhs_perturbed, ctx->rhs_eval_count,
            &ctx->callback_failed);
        if (st != LMMC_STATUS_OK) return st;
    }
    /* J_G = I - (h/2) * J_f */
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            jac_data[i * J->stride + j] *= -0.5 * ctx->h;
        }
        jac_data[i * J->stride + i] += 1.0;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ode_trapezoidal_solve(
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
    size_t total_work_bytes = 0;
    lmmc_real_t t = t_start;
    lmmc_real_t h = 0.0;
    lmmc_status_t init_st;
    lmmc_vec_t x_vec = {0};
    lmmc_real_t* f_n = NULL;
    lmmc_real_t* work = NULL;
    lmmc_optimize_config_t opt_cfg;
    lmmc_optimize_result_t opt_res;
    ode_trapezoidal_ctx_t ctx;

    init_st = validate_and_init_ode_config(rhs, dim, t_start, t_end, y,
                                            cfg, &local_cfg, out_result, &work_bytes);
    if (init_st != LMMC_STATUS_OK) return init_st;

    h = lmmc_clamp(local_cfg.initial_step, local_cfg.min_step, local_cfg.max_step);
    { lmmc_real_t span = t_end - t_start; if (h > span) h = span; }

    lmmc_optimize_default_config(&opt_cfg);
    opt_cfg.abs_tol = local_cfg.abs_tol;
    opt_cfg.rel_tol = local_cfg.rel_tol;
    opt_cfg.max_iter = 50;

    if (!lmmc_safe_mul_size(work_bytes, 5, &total_work_bytes)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    work = (lmmc_real_t*)lmmc_alloc(total_work_bytes);
    if (work == NULL) {
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    x_vec.size = dim;
    x_vec.data = work;
    x_vec.owns_data = 0;
    f_n = work + dim;

    ctx.rhs = rhs;
    ctx.user_data = user_data;
    ctx.jac_cb = local_cfg.jacobian;
    ctx.dim = dim;
    ctx.f_n = f_n;
    ctx.jac_base_rhs = work + 2 * dim;
    ctx.jac_y_perturbed = work + 3 * dim;
    ctx.jac_rhs_perturbed = work + 4 * dim;
    ctx.rhs_eval_count = &out_result->num_rhs_evals;
    ctx.callback_failed = 0;

    lmmc_ode_do_log(&local_cfg, 0, t, y, dim);

    while (t < t_end && out_result->num_steps < local_cfg.max_steps) {
        lmmc_real_t rem = t_end - t;
        lmmc_status_t st;
        if (rem <= 0.0) break;
        if (h > rem) h = rem;

        /* Evaluate f(t_n, y_n). */
        st = lmmc_ode_rhs_eval(
            rhs, t, y, f_n, dim, user_data, &out_result->num_rhs_evals,
            &ctx.callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason =
                ctx.callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                    : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            lmmc_free(work);
            return st;
        }

        ctx.t_next = t + h;
        ctx.h = h;
        ctx.y_n = y;

        /* Initial guess: explicit Euler */
        memcpy(x_vec.data, y, dim * sizeof(lmmc_real_t));

        st = lmmc_nleq_newton(trapezoidal_F, trapezoidal_J,
                               &ctx, &x_vec, &opt_cfg, &opt_res);
        if (st != LMMC_STATUS_OK || !opt_res.converged) {
            out_result->failure_reason =
                ctx.callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED
                                    : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            lmmc_free(work);
            return st != LMMC_STATUS_OK ? st : LMMC_STATUS_NUMERICAL_FAILURE;
        }

        memcpy(y, x_vec.data, dim * sizeof(lmmc_real_t));

        if (!lmmc_ode_state_is_finite(y, dim)) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            lmmc_free(work);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        t += h;
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

    lmmc_free(work);
    return out_result->converged ? LMMC_STATUS_OK : LMMC_STATUS_CONVERGENCE_FAILED;
}
