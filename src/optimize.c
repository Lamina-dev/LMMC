#include "optimize_internal.h"

void lmmc_optimize_emit(
    const lmmc_optimize_config_t* cfg,
    const char* operation,
    size_t iteration,
    const lmmc_real_t* values,
    size_t value_count
) {
    const lmmc_diagnostic_t diagnostic = {
        LMMC_DIAGNOSTIC_TRACE, operation, "iteration", iteration,
        values, value_count
    };
    lmmc_diagnostic_emit(&cfg->diagnostics, &diagnostic);
}

int lmmc_optimize_config_is_valid(
    const lmmc_optimize_config_t* cfg)
{
    if (cfg == NULL) {
        return 0;
    }
    if (!isfinite(cfg->abs_tol) || cfg->abs_tol < 0.0) {
        return 0;
    }
    if (!isfinite(cfg->rel_tol) || cfg->rel_tol < 0.0 || cfg->rel_tol >= 1.0) {
        return 0;
    }
    if (cfg->max_iter == 0 || cfg->lbfgs_memory == 0) {
        return 0;
    }
    return isfinite(cfg->lm_damping) && cfg->lm_damping > 0.0;
}

int lmmc_optimize_has_converged(
    lmmc_real_t residual,
    lmmc_real_t initial_residual,
    const lmmc_optimize_config_t* cfg)
{
    return residual <= cfg->abs_tol ||
           (initial_residual > 0.0 &&
            residual <= cfg->rel_tol * initial_residual);
}

/**
 * @brief 计算向量的 L2 范数。
 */
lmmc_real_t lmmc_optimize_norm(const lmmc_vec_t* v) {
    lmmc_scaled_sumsq_t acc;
    size_t i;
    lmmc_scaled_sumsq_init(&acc);
    for (i = 0; i < v->size; ++i) {
        const lmmc_real_t value_abs = fabs(v->data[i]);
        if (!isfinite(value_abs)) {
            return value_abs;
        }
        lmmc_scaled_sumsq_add(&acc, value_abs);
    }
    return lmmc_scaled_sumsq_norm(&acc);
}

lmmc_status_t lmmc_optimize_default_config(lmmc_optimize_config_t* cfg) {
    if (cfg == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    cfg->abs_tol = 1e-12;
    cfg->rel_tol = 1e-10;
    cfg->max_iter = 1000;
    cfg->lbfgs_memory = 10;
    cfg->lm_damping = 1e-3;
    cfg->diagnostics = (lmmc_diagnostic_sink_t){0};
    return LMMC_STATUS_OK;
}

int lmmc_optimize_arguments_valid(const lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg, const lmmc_optimize_result_t* out)
{
    return x != NULL && out != NULL && x->data != NULL &&
           lmmc_optimize_config_is_valid(cfg) && x->size != 0;
}

void lmmc_optimize_result_init(lmmc_optimize_result_t* out)
{
    out->converged = 0;
    out->num_iter = 0;
    out->final_residual = NAN;
    out->failure_reason = LMMC_OPT_FAILURE_NONE;
}

int lmmc_optimize_stop(lmmc_real_t norm, lmmc_real_t* initial,
    size_t iteration, const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out)
{
    out->final_residual = norm;
    if (!isfinite(norm)) {
        out->failure_reason = LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
        return 1;
    }
    if (iteration == 0) {
        *initial = norm;
    }
    if (lmmc_optimize_has_converged(norm, *initial, cfg)) {
        out->converged = 1;
        return 1;
    }
    if (iteration == cfg->max_iter) {
        out->failure_reason = LMMC_OPT_FAILURE_MAX_ITER;
        return 1;
    }
    return 0;
}

static int optimize_workspace_count(
    size_t n, size_t vectors, size_t matrices, size_t extra, size_t* count)
{
    size_t vector_count, matrix_count = 0;
    if (!lmmc_safe_mul_size(n, vectors, &vector_count)) {
        return 0;
    }
    if (matrices != 0) {
        if (!lmmc_safe_mul_size(n, n, &matrix_count) ||
            !lmmc_safe_mul_size(matrix_count, matrices, &matrix_count)) {
            return 0;
        }
    }
    return lmmc_safe_add_size(vector_count, matrix_count, count) &&
           lmmc_safe_add_size(*count, extra, count);
}

lmmc_status_t lmmc_opt_workspace_create(
    size_t n, size_t vectors, size_t matrices, size_t extra,
    int need_pivots, lmmc_opt_workspace_t* work)
{
    size_t count, scalar_bytes, pivot_bytes = 0, bytes;
    const size_t alignment = _Alignof(lmmc_real_t);
    if (!optimize_workspace_count(n, vectors, matrices, extra, &count) ||
        !lmmc_safe_mul_size(count, sizeof(lmmc_real_t), &scalar_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (need_pivots) {
        if (!lmmc_safe_mul_size(n, sizeof(size_t), &pivot_bytes) ||
            !lmmc_safe_add_size(pivot_bytes, alignment - 1, &pivot_bytes)) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        pivot_bytes -= pivot_bytes % alignment;
    }
    if (!lmmc_safe_add_size(pivot_bytes, scalar_bytes, &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    work->allocation = lmmc_memory_alloc(bytes);
    if (work->allocation == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(work->allocation, 0, bytes);
    work->pivots = need_pivots ? (size_t*)work->allocation : NULL;
    work->next = (lmmc_real_t*)((unsigned char*)work->allocation + pivot_bytes);
    return LMMC_STATUS_OK;
}

lmmc_vec_t lmmc_opt_workspace_vector(lmmc_opt_workspace_t* work, size_t n)
{
    const lmmc_vec_t result = {n, work->next, 0};
    work->next += n;
    return result;
}

lmmc_mat_t lmmc_opt_workspace_matrix(lmmc_opt_workspace_t* work, size_t n)
{
    const lmmc_mat_t result = {n, n, n, work->next, 0};
    work->next += n * n;
    return result;
}

static lmmc_status_t optimize_perturb(
    const lmmc_vec_t* x, size_t j, lmmc_vec_t* x_pert, lmmc_real_t* step)
{
    lmmc_real_t h = sqrt(LMMC_REAL_EPSILON) * fmax(fabs(x->data[j]), 1.0);
    lmmc_real_t perturbed_x = x->data[j] + h;
    memcpy(x_pert->data, x->data, x->size * sizeof(lmmc_real_t));
    if (!isfinite(perturbed_x) || perturbed_x == x->data[j]) {
        perturbed_x = x->data[j] - h;
    }
    if (!isfinite(perturbed_x) || perturbed_x == x->data[j]) {
        perturbed_x = nextafter(x->data[j], 0.0);
    }
    h = perturbed_x - x->data[j];
    if (!isfinite(perturbed_x) || !isfinite(h) || h == 0.0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    x_pert->data[j] = perturbed_x;
    *step = h;
    return LMMC_STATUS_OK;
}

static lmmc_status_t optimize_difference_column(
    const lmmc_vec_t* Fx, const lmmc_vec_t* F_pert,
    lmmc_real_t h, size_t j, lmmc_mat_t* matrix)
{
    size_t i;
    for (i = 0; i < Fx->size; ++i) {
        lmmc_real_t derivative;
        if (!isfinite(F_pert->data[i]) || !isfinite(Fx->data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        derivative = (F_pert->data[i] - Fx->data[i]) / h;
        if (!isfinite(derivative)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        matrix->data[i * matrix->stride + j] = derivative;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_optimize_jacobian(
    lmmc_opt_func_t F, lmmc_opt_jac_t J, void* user_data,
    const lmmc_vec_t* x, const lmmc_vec_t* Fx, lmmc_mat_t* matrix,
    lmmc_vec_t* x_pert, lmmc_vec_t* F_pert)
{
    size_t j;
    if (J != NULL) {
        return J(x, matrix, user_data);
    }
    for (j = 0; j < x->size; ++j) {
        lmmc_real_t h;
        lmmc_status_t status = optimize_perturb(x, j, x_pert, &h);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
        status = F(x_pert, F_pert, user_data);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
        status = optimize_difference_column(Fx, F_pert, h, j, matrix);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
    }
    return LMMC_STATUS_OK;
}

int lmmc_optimize_armijo(
    lmmc_opt_obj_t obj, void* user_data, const lmmc_vec_t* x,
    const lmmc_vec_t* direction, lmmc_vec_t* trial,
    lmmc_real_t* value, lmmc_real_t derivative, int attempts, int subtract)
{
    lmmc_real_t step = 1.0;
    const lmmc_real_t c1 = 1e-4;
    int attempt;
    size_t i;
    for (attempt = 0; attempt < attempts; ++attempt) {
        for (i = 0; i < x->size; ++i) {
            if (subtract) {
                trial->data[i] = x->data[i] - step * direction->data[i];
            } else {
                trial->data[i] = x->data[i] + step * direction->data[i];
            }
        }
        const lmmc_real_t next = obj(trial, user_data);
        if (next <= *value + c1 * step * derivative) {
            *value = next;
            return 1;
        }
        step *= 0.5;
    }
    return 0;
}
