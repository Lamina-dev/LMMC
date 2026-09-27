#include "optimize_internal.h"

typedef struct {
    lmmc_opt_workspace_t arena;
    lmmc_vec_t value, next_value, delta_x, delta_F, Bdx, rhs;
    lmmc_mat_t B, lu;
} lmmc_broyden_workspace_t;

static lmmc_status_t broyden_workspace_create(
    size_t n, lmmc_broyden_workspace_t* work)
{
    lmmc_status_t status = lmmc_opt_workspace_create(n, 6, 2, 0, 1, &work->arena);
    if (status != LMMC_STATUS_OK) { return status; }
    work->value = lmmc_opt_workspace_vector(&work->arena, n);
    work->next_value = lmmc_opt_workspace_vector(&work->arena, n);
    work->delta_x = lmmc_opt_workspace_vector(&work->arena, n);
    work->delta_F = lmmc_opt_workspace_vector(&work->arena, n);
    work->Bdx = lmmc_opt_workspace_vector(&work->arena, n);
    work->rhs = lmmc_opt_workspace_vector(&work->arena, n);
    work->B = lmmc_opt_workspace_matrix(&work->arena, n);
    work->lu = lmmc_opt_workspace_matrix(&work->arena, n);
    return LMMC_STATUS_OK;
}

static lmmc_optimize_failure_t broyden_step(
    lmmc_broyden_workspace_t* work, lmmc_vec_t* x)
{
    const size_t n = x->size;
    lmmc_status_t status;
    size_t i;
    memcpy(work->lu.data, work->B.data, n * n * sizeof(lmmc_real_t));
    status = lmmc_lu_decompose_inplace(&work->lu, work->arena.pivots, NULL);
    if (status == LMMC_STATUS_SINGULAR_MATRIX) {
        return LMMC_OPT_FAILURE_SINGULAR_JACOBIAN;
    }
    if (status != LMMC_STATUS_OK) { return LMMC_OPT_FAILURE_NUMERICAL_ISSUE; }
    for (i = 0; i < n; ++i) {
        work->rhs.data[i] = -work->value.data[i];
    }
    status = lmmc_lu_solve(&work->lu, work->arena.pivots,
                          &work->rhs, &work->delta_x);
    if (status != LMMC_STATUS_OK) { return LMMC_OPT_FAILURE_SINGULAR_JACOBIAN; }
    for (i = 0; i < n; ++i) {
        x->data[i] += work->delta_x.data[i];
    }
    return LMMC_OPT_FAILURE_NONE;
}

static void broyden_rank_one_update(lmmc_broyden_workspace_t* work)
{
    const size_t n = work->value.size;
    lmmc_real_t dxTdx = 0.0;
    size_t i, j;
    for (i = 0; i < n; ++i) {
        work->delta_F.data[i] = work->next_value.data[i] - work->value.data[i];
    }
    for (i = 0; i < n; ++i) {
        work->Bdx.data[i] = 0.0;
        for (j = 0; j < n; ++j) {
            work->Bdx.data[i] += work->B.data[i * work->B.stride + j] *
                                 work->delta_x.data[j];
        }
    }
    for (i = 0; i < n; ++i) {
        dxTdx += work->delta_x.data[i] * work->delta_x.data[i];
    }
    if (dxTdx > 1e-300) {
        for (i = 0; i < n; ++i) {
            const lmmc_real_t num_i = work->delta_F.data[i] - work->Bdx.data[i];
            for (j = 0; j < n; ++j) {
                work->B.data[i * work->B.stride + j] +=
                    num_i * work->delta_x.data[j] / dxTdx;
            }
        }
    }
    memcpy(work->value.data, work->next_value.data, n * sizeof(lmmc_real_t));
}

lmmc_status_t lmmc_nleq_broyden(
    lmmc_opt_func_t F, void* user_data, lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg, lmmc_optimize_result_t* out)
{
    lmmc_broyden_workspace_t work;
    lmmc_status_t status;
    lmmc_real_t initial_residual = 0.0;
    size_t iter;
    if (F == NULL || !lmmc_optimize_arguments_valid(x, cfg, out)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_optimize_result_init(out);
    status = broyden_workspace_create(x->size, &work);
    if (status != LMMC_STATUS_OK) { return status; }
    status = F(x, &work.value, user_data);
    if (status != LMMC_STATUS_OK) {
        out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
        goto cleanup;
    }
    for (iter = 0; ; ++iter) {
        const lmmc_real_t norm = lmmc_optimize_norm(&work.value);
        if (lmmc_optimize_stop(norm, &initial_residual, iter, cfg, out)) { break; }
        if (iter == 0) {
            /** @brief 首次更新前尚无步长差分或下一步残差。 */
            status = lmmc_optimize_jacobian(F, NULL, user_data, x, &work.value,
                &work.B, &work.delta_x, &work.next_value);
            if (status != LMMC_STATUS_OK) {
                out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
                break;
            }
        }
        out->failure_reason = broyden_step(&work, x);
        if (out->failure_reason != LMMC_OPT_FAILURE_NONE) { break; }
        out->num_iter = iter + 1;
        status = F(x, &work.next_value, user_data);
        if (status != LMMC_STATUS_OK) {
            out->final_residual = NAN;
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            break;
        }
        broyden_rank_one_update(&work);
        lmmc_optimize_emit(cfg, "broyden", iter + 1, &norm, 1);
    }
cleanup:
    lmmc_memory_free(work.arena.allocation);
    return LMMC_STATUS_OK;
}
