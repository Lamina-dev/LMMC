#include "optimize_internal.h"

typedef struct {
    lmmc_opt_workspace_t arena;
    lmmc_vec_t g, q, x_new, g_new, alpha, rho;
    lmmc_mat_t s, y;
    size_t history_count, oldest;
} lmmc_lbfgs_workspace_t;

static lmmc_status_t lbfgs_workspace_create(
    size_t n, size_t m, lmmc_lbfgs_workspace_t* work)
{
    size_t mn, pairs, scalars, extra;
    lmmc_status_t status;
    if (!lmmc_safe_mul_size(m, n, &mn) ||
        !lmmc_safe_mul_size(mn, 2, &pairs) ||
        !lmmc_safe_mul_size(m, 2, &scalars) ||
        !lmmc_safe_add_size(pairs, scalars, &extra)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    status = lmmc_opt_workspace_create(n, 4, 0, extra, 0, &work->arena);
    if (status != LMMC_STATUS_OK) { return LMMC_STATUS_ALLOCATION_FAILED; }
    work->g = lmmc_opt_workspace_vector(&work->arena, n);
    work->q = lmmc_opt_workspace_vector(&work->arena, n);
    work->x_new = lmmc_opt_workspace_vector(&work->arena, n);
    work->g_new = lmmc_opt_workspace_vector(&work->arena, n);
    work->alpha = lmmc_opt_workspace_vector(&work->arena, m);
    work->rho = lmmc_opt_workspace_vector(&work->arena, m);
    work->s = (lmmc_mat_t){m, n, n, work->arena.next, 0};
    work->arena.next += mn;
    work->y = (lmmc_mat_t){m, n, n, work->arena.next, 0};
    work->history_count = 0;
    work->oldest = 0;
    return LMMC_STATUS_OK;
}

static void lbfgs_scale_direction(lmmc_lbfgs_workspace_t* work)
{
    const size_t n = work->g.size;
    size_t i, newest;
    lmmc_real_t sy = 0.0, yy = 0.0;
    if (work->history_count == 0) return;
    newest = (work->oldest + work->history_count - 1) % work->s.rows;
    for (i = 0; i < n; ++i) {
        sy += work->s.data[newest * n + i] * work->y.data[newest * n + i];
        yy += work->y.data[newest * n + i] * work->y.data[newest * n + i];
    }
    if (yy > 1e-300) {
        const lmmc_real_t gamma = sy / yy;
        for (i = 0; i < n; ++i) {
            work->q.data[i] *= gamma;
        }
    }
}

/** @brief 双循环递推先从新到旧，再从旧到新遍历历史。 */
static void lbfgs_direction(lmmc_lbfgs_workspace_t* work)
{
    const size_t n = work->g.size;
    const size_t m = work->s.rows;
    const size_t bound = work->history_count;
    size_t k, i;
    memcpy(work->q.data, work->g.data, n * sizeof(lmmc_real_t));
    for (k = 0; k < bound; ++k) {
        const size_t idx = (work->oldest + work->history_count - 1 - k) % m;
        lmmc_real_t dot = 0.0;
        for (i = 0; i < n; ++i) {
            dot += work->s.data[idx * n + i] * work->q.data[i];
        }
        work->alpha.data[k] = work->rho.data[idx] * dot;
        for (i = 0; i < n; ++i) {
            work->q.data[i] -= work->alpha.data[k] * work->y.data[idx * n + i];
        }
    }
    lbfgs_scale_direction(work);
    for (k = bound; k-- > 0;) {
        const size_t idx = (work->oldest + work->history_count - 1 - k) % m;
        lmmc_real_t dot = 0.0;
        lmmc_real_t beta;
        for (i = 0; i < n; ++i) {
            dot += work->y.data[idx * n + i] * work->q.data[i];
        }
        beta = work->rho.data[idx] * dot;
        for (i = 0; i < n; ++i) {
            work->q.data[i] += (work->alpha.data[k] - beta) * work->s.data[idx * n + i];
        }
    }
    for (i = 0; i < n; ++i) {
        work->q.data[i] = -work->q.data[i];
    }
}

static void lbfgs_store_history(
    lmmc_lbfgs_workspace_t* work, const lmmc_vec_t* x)
{
    const size_t n = x->size;
    const size_t m = work->s.rows;
    size_t store_idx, i;
    lmmc_real_t sy = 0.0;
    if (work->history_count < m) {
        store_idx = (work->oldest + work->history_count) % m;
        work->history_count++;
    } else {
        store_idx = work->oldest;
        work->oldest = (work->oldest + 1) % m;
    }
    for (i = 0; i < n; ++i) {
        work->s.data[store_idx * n + i] = work->x_new.data[i] - x->data[i];
        work->y.data[store_idx * n + i] = work->g_new.data[i] - work->g.data[i];
        sy += work->s.data[store_idx * n + i] * work->y.data[store_idx * n + i];
    }
    if (fabs(sy) > 1e-300) {
        work->rho.data[store_idx] = 1.0 / sy;
    } else {
        work->rho.data[store_idx] = 0.0;
    }
}

static lmmc_real_t lbfgs_descent_derivative(
    lmmc_lbfgs_workspace_t* work, lmmc_real_t norm)
{
    size_t i;
    lmmc_real_t dg = 0.0;
    for (i = 0; i < work->g.size; ++i) {
        dg += work->g.data[i] * work->q.data[i];
    }
    if (dg >= 0.0) {
        for (i = 0; i < work->g.size; ++i) {
            work->q.data[i] = -work->g.data[i];
        }
        dg = -norm * norm;
    }
    return dg;
}

static lmmc_optimize_failure_t lbfgs_step(
    lmmc_lbfgs_workspace_t* work, lmmc_opt_obj_t obj,
    lmmc_opt_grad_t grad, void* user_data, lmmc_vec_t* x,
    lmmc_real_t* value, lmmc_real_t norm)
{
    lmmc_real_t dg;
    lbfgs_direction(work);
    dg = lbfgs_descent_derivative(work, norm);
    if (!lmmc_optimize_armijo(obj, user_data, x, &work->q,
            &work->x_new, value, dg, 40, 0)) {
        return LMMC_OPT_FAILURE_LINE_SEARCH_FAILED;
    }
    if (grad(&work->x_new, &work->g_new, user_data) != LMMC_STATUS_OK) {
        return LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
    }
    lbfgs_store_history(work, x);
    memcpy(x->data, work->x_new.data, x->size * sizeof(lmmc_real_t));
    memcpy(work->g.data, work->g_new.data, x->size * sizeof(lmmc_real_t));
    return LMMC_OPT_FAILURE_NONE;
}

lmmc_status_t lmmc_minimize_lbfgs(
    lmmc_opt_obj_t obj, lmmc_opt_grad_t grad, void* user_data,
    lmmc_vec_t* x, const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out)
{
    lmmc_lbfgs_workspace_t work;
    lmmc_status_t status;
    lmmc_real_t value, initial_residual = 0.0;
    size_t iter;
    if (obj == NULL || grad == NULL ||
        !lmmc_optimize_arguments_valid(x, cfg, out)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_optimize_result_init(out);
    status = lbfgs_workspace_create(x->size, cfg->lbfgs_memory, &work);
    if (status != LMMC_STATUS_OK) { return status; }
    if (grad(x, &work.g, user_data) != LMMC_STATUS_OK) {
        out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
        goto cleanup;
    }
    value = obj(x, user_data);
    for (iter = 0; ; ++iter) {
        const lmmc_real_t norm = lmmc_optimize_norm(&work.g);
        out->final_residual = norm;
        if (!isfinite(value)) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            break;
        }
        if (lmmc_optimize_stop(norm, &initial_residual, iter, cfg, out)) { break; }
        out->failure_reason = lbfgs_step(&work, obj, grad, user_data, x, &value, norm);
        if (out->failure_reason != LMMC_OPT_FAILURE_NONE) { break; }
        out->num_iter = iter + 1;
        {
            const lmmc_real_t values[] = {value, norm};
            lmmc_optimize_emit(cfg, "lbfgs", iter + 1, values, 2);
        }
    }
cleanup:
    lmmc_memory_free(work.arena.allocation);
    return LMMC_STATUS_OK;
}
