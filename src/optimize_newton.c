#include "optimize_internal.h"

typedef struct {
    lmmc_opt_workspace_t arena;
    lmmc_vec_t value, delta, rhs;
    lmmc_mat_t jacobian, lu;
} lmmc_newton_workspace_t;

static lmmc_status_t newton_workspace_create(
    size_t n, lmmc_newton_workspace_t* work)
{
    lmmc_status_t status = lmmc_opt_workspace_create(
        n, 3, 2, 0, 1, &work->arena);
    if (status != LMMC_STATUS_OK) { return status; }
    work->value = lmmc_opt_workspace_vector(&work->arena, n);
    work->delta = lmmc_opt_workspace_vector(&work->arena, n);
    work->rhs = lmmc_opt_workspace_vector(&work->arena, n);
    work->jacobian = lmmc_opt_workspace_matrix(&work->arena, n);
    work->lu = lmmc_opt_workspace_matrix(&work->arena, n);
    return LMMC_STATUS_OK;
}

static lmmc_optimize_failure_t newton_step(
    lmmc_newton_workspace_t* work, lmmc_vec_t* x)
{
    const size_t n = x->size;
    lmmc_status_t status;
    size_t i;
    memcpy(work->lu.data, work->jacobian.data, n * n * sizeof(lmmc_real_t));
    status = lmmc_lu_decompose_inplace(&work->lu, work->arena.pivots, NULL);
    if (status == LMMC_STATUS_SINGULAR_MATRIX) {
        return LMMC_OPT_FAILURE_SINGULAR_JACOBIAN;
    }
    if (status != LMMC_STATUS_OK) { return LMMC_OPT_FAILURE_NUMERICAL_ISSUE; }
    for (i = 0; i < n; ++i) {
        work->rhs.data[i] = -work->value.data[i];
    }
    status = lmmc_lu_solve(&work->lu, work->arena.pivots,
                          &work->rhs, &work->delta);
    if (status != LMMC_STATUS_OK) { return LMMC_OPT_FAILURE_SINGULAR_JACOBIAN; }
    for (i = 0; i < n; ++i) {
        x->data[i] += work->delta.data[i];
    }
    return LMMC_OPT_FAILURE_NONE;
}

lmmc_status_t lmmc_nleq_newton(
    lmmc_opt_func_t F, lmmc_opt_jac_t J, void* user_data,
    lmmc_vec_t* x, const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out)
{
    lmmc_newton_workspace_t work;
    lmmc_status_t status;
    lmmc_real_t initial_residual = 0.0;
    size_t iter;
    if (F == NULL || !lmmc_optimize_arguments_valid(x, cfg, out)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_optimize_result_init(out);
    status = newton_workspace_create(x->size, &work);
    if (status != LMMC_STATUS_OK) { return status; }

    /** @brief 检查每个迭代结果，包括最后一次允许的更新。 */
    for (iter = 0; ; ++iter) {
        lmmc_real_t norm;
        status = F(x, &work.value, user_data);
        if (status != LMMC_STATUS_OK) {
            out->final_residual = NAN;
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            break;
        }
        norm = lmmc_optimize_norm(&work.value);
        if (lmmc_optimize_stop(norm, &initial_residual, iter, cfg, out)) { break; }
        /** @brief 差分计算复用求解阶段暂时空闲的 delta 与 rhs 缓冲区。 */
        status = lmmc_optimize_jacobian(F, J, user_data, x, &work.value,
            &work.jacobian, &work.delta, &work.rhs);
        if (status != LMMC_STATUS_OK) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            break;
        }
        out->failure_reason = newton_step(&work, x);
        if (out->failure_reason != LMMC_OPT_FAILURE_NONE) { break; }
        out->num_iter = iter + 1;
        lmmc_optimize_emit(cfg, "newton", iter + 1, &norm, 1);
    }
    lmmc_memory_free(work.arena.allocation);
    return LMMC_STATUS_OK;
}
