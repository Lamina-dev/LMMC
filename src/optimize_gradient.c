#include "optimize_internal.h"

lmmc_status_t lmmc_minimize_gradient_descent(
    lmmc_opt_obj_t obj, lmmc_opt_grad_t grad, void* user_data,
    lmmc_vec_t* x, const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out)
{
    lmmc_opt_workspace_t arena;
    lmmc_vec_t g, x_new;
    lmmc_status_t status;
    lmmc_real_t f_val, initial_residual = 0.0;
    size_t iter;
    if (obj == NULL || grad == NULL ||
        !lmmc_optimize_arguments_valid(x, cfg, out)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_optimize_result_init(out);
    status = lmmc_opt_workspace_create(x->size, 2, 0, 0, 0, &arena);
    if (status != LMMC_STATUS_OK) { return status; }
    g = lmmc_opt_workspace_vector(&arena, x->size);
    x_new = lmmc_opt_workspace_vector(&arena, x->size);
    f_val = obj(x, user_data);

    for (iter = 0; ; ++iter) {
        lmmc_real_t grad_norm;
        status = grad(x, &g, user_data);
        if (status != LMMC_STATUS_OK) {
            out->final_residual = NAN;
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            break;
        }
        grad_norm = lmmc_optimize_norm(&g);
        out->final_residual = grad_norm;
        if (!isfinite(f_val)) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            break;
        }
        if (lmmc_optimize_stop(grad_norm, &initial_residual, iter, cfg, out)) {
            break;
        }
        if (!lmmc_optimize_armijo(obj, user_data, x, &g, &x_new,
                &f_val, -grad_norm * grad_norm, 50, 1)) {
            out->failure_reason = LMMC_OPT_FAILURE_LINE_SEARCH_FAILED;
            break;
        }
        memcpy(x->data, x_new.data, x->size * sizeof(lmmc_real_t));
        out->num_iter = iter + 1;
        {
            const lmmc_real_t values[] = {f_val, grad_norm};
            lmmc_optimize_emit(cfg, "gradient_descent", iter + 1, values, 2);
        }
    }
    lmmc_memory_free(arena.allocation);
    return LMMC_STATUS_OK;
}
