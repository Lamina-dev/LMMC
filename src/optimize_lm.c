#include "optimize_internal.h"

typedef struct {
    lmmc_opt_workspace_t arena;
    lmmc_vec_t residual, next_residual, delta, gradient, trial, rhs;
    lmmc_mat_t jacobian, gram, augmented;
} lmmc_lm_workspace_t;

typedef enum {
    LMMC_LM_TRIAL_READY,
    LMMC_LM_TRIAL_RETRY,
    LMMC_LM_TRIAL_FAILURE
} lmmc_lm_trial_t;

static lmmc_status_t lm_workspace_create(size_t n, lmmc_lm_workspace_t* work)
{
    lmmc_status_t status = lmmc_opt_workspace_create(n, 6, 3, 0, 1, &work->arena);
    if (status != LMMC_STATUS_OK) { return status; }
    work->residual = lmmc_opt_workspace_vector(&work->arena, n);
    work->next_residual = lmmc_opt_workspace_vector(&work->arena, n);
    work->delta = lmmc_opt_workspace_vector(&work->arena, n);
    work->gradient = lmmc_opt_workspace_vector(&work->arena, n);
    work->trial = lmmc_opt_workspace_vector(&work->arena, n);
    work->rhs = lmmc_opt_workspace_vector(&work->arena, n);
    work->jacobian = lmmc_opt_workspace_matrix(&work->arena, n);
    work->gram = lmmc_opt_workspace_matrix(&work->arena, n);
    work->augmented = lmmc_opt_workspace_matrix(&work->arena, n);
    return LMMC_STATUS_OK;
}

static lmmc_real_t lm_gradient(lmmc_lm_workspace_t* work)
{
    const size_t n = work->gradient.size;
    size_t i, k;
    for (i = 0; i < n; ++i) {
        work->gradient.data[i] = 0.0;
        for (k = 0; k < n; ++k) {
            work->gradient.data[i] += work->jacobian.data[k * work->jacobian.stride + i] *
                                      work->residual.data[k];
        }
    }
    return lmmc_optimize_norm(&work->gradient);
}

static void lm_gram(lmmc_lm_workspace_t* work)
{
    const size_t n = work->gradient.size;
    size_t i, j, k;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            lmmc_real_t sum = 0.0;
            for (k = 0; k < n; ++k) {
                sum += work->jacobian.data[k * work->jacobian.stride + i] *
                       work->jacobian.data[k * work->jacobian.stride + j];
            }
            work->gram.data[i * work->gram.stride + j] = sum;
        }
    }
}

static lmmc_lm_trial_t lm_trial_direction(
    lmmc_lm_workspace_t* work, lmmc_real_t lambda)
{
    const size_t n = work->gradient.size;
    lmmc_status_t status;
    size_t i;
    memcpy(work->augmented.data, work->gram.data, n * n * sizeof(lmmc_real_t));
    for (i = 0; i < n; ++i) {
        work->augmented.data[i * work->augmented.stride + i] += lambda;
    }
    status = lmmc_lu_decompose_inplace(&work->augmented, work->arena.pivots, NULL);
    if (status == LMMC_STATUS_SINGULAR_MATRIX) { return LMMC_LM_TRIAL_RETRY; }
    if (status != LMMC_STATUS_OK) { return LMMC_LM_TRIAL_FAILURE; }
    for (i = 0; i < n; ++i) {
        work->rhs.data[i] = -work->gradient.data[i];
    }
    status = lmmc_lu_solve(&work->augmented, work->arena.pivots,
                          &work->rhs, &work->delta);
    return status == LMMC_STATUS_OK ? LMMC_LM_TRIAL_READY : LMMC_LM_TRIAL_RETRY;
}

static int lm_step(
    lmmc_lm_workspace_t* work, lmmc_opt_func_t residual, void* user_data,
    lmmc_vec_t* x, lmmc_real_t* lambda, lmmc_real_t* norm)
{
    int attempt;
    size_t i;
    lm_gram(work);
    for (attempt = 0; attempt < 20; ++attempt) {
        lmmc_real_t next_norm;
        lmmc_lm_trial_t trial = lm_trial_direction(work, *lambda);
        if (trial == LMMC_LM_TRIAL_FAILURE) { return 0; }
        if (trial == LMMC_LM_TRIAL_RETRY) {
            *lambda *= 10.0;
            continue;
        }
        for (i = 0; i < x->size; ++i) {
            work->trial.data[i] = x->data[i] + work->delta.data[i];
        }
        if (residual(&work->trial, &work->next_residual, user_data) != LMMC_STATUS_OK) {
            *lambda *= 10.0;
            continue;
        }
        next_norm = lmmc_optimize_norm(&work->next_residual);
        if (next_norm < *norm) {
            memcpy(x->data, work->trial.data, x->size * sizeof(lmmc_real_t));
            memcpy(work->residual.data, work->next_residual.data,
                   x->size * sizeof(lmmc_real_t));
            *norm = next_norm;
            *lambda *= 0.1;
            if (*lambda < 1e-15) *lambda = 1e-15;
            return 1;
        }
        *lambda *= 10.0;
        if (*lambda > 1e15) { return 0; }
    }
    return 0;
}

static int lm_stop(lmmc_real_t norm, lmmc_real_t grad_norm,
    lmmc_real_t initial_residual, lmmc_real_t* initial_gradient,
    size_t iter, const lmmc_optimize_config_t* cfg, lmmc_optimize_result_t* out)
{
    if (!isfinite(grad_norm)) {
        out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
        return 1;
    }
    if (iter == 0) *initial_gradient = grad_norm;
    if (lmmc_optimize_has_converged(norm, initial_residual, cfg) ||
        lmmc_optimize_has_converged(grad_norm, *initial_gradient, cfg)) {
        out->converged = 1;
        return 1;
    }
    if (iter == cfg->max_iter) {
        out->failure_reason = LMMC_OPT_FAILURE_MAX_ITER;
        return 1;
    }
    return 0;
}

lmmc_status_t lmmc_minimize_levenberg_marquardt(
    lmmc_opt_func_t residual, lmmc_opt_jac_t J, void* user_data,
    lmmc_vec_t* x, const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out)
{
    lmmc_lm_workspace_t work;
    lmmc_status_t status;
    lmmc_real_t lambda, norm, initial_residual, initial_gradient = 0.0;
    size_t iter;
    if (residual == NULL || !lmmc_optimize_arguments_valid(x, cfg, out)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_optimize_result_init(out);
    lambda = cfg->lm_damping;
    status = lm_workspace_create(x->size, &work);
    if (status != LMMC_STATUS_OK) { return status; }
    if (residual(x, &work.residual, user_data) != LMMC_STATUS_OK) {
        out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
        goto cleanup;
    }
    norm = lmmc_optimize_norm(&work.residual);
    initial_residual = norm;
    for (iter = 0; ; ++iter) {
        lmmc_real_t grad_norm;
        out->final_residual = norm;
        if (!isfinite(norm)) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            break;
        }
        /** @brief 尝试阻尼步之前，Jacobian 计算可复用试探缓冲区。 */
        status = lmmc_optimize_jacobian(residual, J, user_data, x,
            &work.residual, &work.jacobian, &work.trial, &work.next_residual);
        if (status != LMMC_STATUS_OK) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            break;
        }
        grad_norm = lm_gradient(&work);
        if (lm_stop(norm, grad_norm, initial_residual, &initial_gradient,
                    iter, cfg, out)) { break; }
        if (!lm_step(&work, residual, user_data, x, &lambda, &norm)) {
            out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
            break;
        }
        out->num_iter = iter + 1;
        {
            const lmmc_real_t values[] = {norm, lambda};
            lmmc_optimize_emit(cfg, "levenberg_marquardt", iter + 1, values, 2);
        }
    }
cleanup:
    lmmc_memory_free(work.arena.allocation);
    return LMMC_STATUS_OK;
}
