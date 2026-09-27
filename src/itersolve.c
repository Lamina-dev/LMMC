#include "itersolve_internal.h"
#include "sparse_internal.h"

lmmc_status_t lmmc_vec_norm2_checked(const lmmc_vec_t* v, lmmc_real_t* out_norm) {
    if (v == NULL || out_norm == NULL || v->data == NULL || v->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    for (size_t i = 0; i < v->size; ++i) {
        if (!lmmc_is_finite(&v->data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    lmmc_status_t st = lmmc_vec_norm2(v, out_norm);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    if (!lmmc_is_finite(&*out_norm)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_dot_checked(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t* out_dot) {
    lmmc_status_t st = lmmc_vec_dot(a, b, out_dot);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (!lmmc_is_finite(&*out_dot)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_apply_precond_or_identity(
    const lmmc_precond_t* precond,
    const lmmc_vec_t* rhs,
    lmmc_vec_t* out
) {
    if (precond == NULL) {
        return lmmc_vec_copy(rhs, out);
    }
    return lmmc_precond_apply(precond, rhs, out);
}

void lmmc_itersolve_do_log(const lmmc_itersolve_config_t* cfg, size_t iter, lmmc_real_t residual_norm) {
    const lmmc_diagnostic_t diagnostic = {
        LMMC_DIAGNOSTIC_TRACE, "itersolve", "iteration", iter,
        &residual_norm, 1
    };
    lmmc_diagnostic_emit(&cfg->diagnostics, &diagnostic);
}

lmmc_status_t lmmc_itersolve_default_config(size_t problem_size, lmmc_itersolve_config_t* out_cfg) {
    size_t max_iter = 0;

    if (out_cfg == NULL || problem_size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (problem_size > ((size_t)-1) / 20) {
        max_iter = 1000;
    } else {
        max_iter = problem_size * 20;
        if (max_iter < 100) {
            max_iter = 100;
        }
    }


    LMMC_REAL_INIT(&out_cfg->abs_tol);
    LMMC_REAL_INIT(&out_cfg->rel_tol);
    LMMC_REAL_SET_D(&out_cfg->abs_tol, LMMC_DEFAULT_ABS_TOL);
    LMMC_REAL_SET_D(&out_cfg->rel_tol, 1e-8);

    out_cfg->max_iter = max_iter;
    out_cfg->restart = (problem_size < 30) ? problem_size : 30;
    out_cfg->diagnostics = (lmmc_diagnostic_sink_t){0};
    out_cfg->apply_op = NULL;
    out_cfg->op_user_data = NULL;
    out_cfg->apply_transpose_op = NULL;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_itersolve_square_dimensions(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_vec_t* x) {
    if (lmmc_sparse_validate(a) != LMMC_STATUS_OK || b == NULL || x == NULL ||
        b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows == 0 || a->cols == 0 || b->size == 0 || x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols || b->size != a->rows || x->size != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_itersolve_validate(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_config_t* out_cfg,
    lmmc_itersolve_result_t* out_result
) {
    lmmc_status_t st = lmmc_itersolve_square_dimensions(a, b, x);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    if (cfg != NULL) {
        *out_cfg = *cfg;
    } else {
        st = lmmc_itersolve_default_config(b->size, out_cfg);
        if (st != LMMC_STATUS_OK) {
            goto cleanup;
        }
    }

    if (!isfinite(out_cfg->abs_tol) || !isfinite(out_cfg->rel_tol) ||
        out_cfg->abs_tol < 0.0 || out_cfg->rel_tol < 0.0 || out_cfg->max_iter == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (precond != NULL && precond->size != b->size) {
        st = LMMC_STATUS_DIMENSION_MISMATCH;
        goto cleanup;
    }

    if (out_result != NULL) {
        out_result->converged = 0;
        out_result->num_iter = 0;
        LMMC_REAL_INIT(&out_result->initial_residual_norm);
        LMMC_REAL_INIT(&out_result->final_residual_norm);
        LMMC_REAL_SET_D(&out_result->initial_residual_norm, 0.0);
        LMMC_REAL_SET_D(&out_result->final_residual_norm, 0.0);
    }

cleanup:
    return st;
}

lmmc_status_t lmmc_itersolve_workspace(lmmc_vec_t* const* vectors,
    const size_t* sizes, size_t count, lmmc_real_t** storage) {
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!lmmc_safe_add_size(total, sizes[i], &total)) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    size_t bytes;
    if (!lmmc_safe_mul_size(total, sizeof(lmmc_real_t), &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *storage = lmmc_memory_alloc(bytes);
    if (*storage == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(*storage, 0, bytes);
    lmmc_real_t* next = *storage;
    for (size_t i = 0; i < count; ++i) {
        *vectors[i] = (lmmc_vec_t){sizes[i], next, 0};
        next += sizes[i];
    }
    return LMMC_STATUS_OK;
}

void lmmc_itersolve_result_init(lmmc_itersolve_result_t* result) {
    if (result != NULL) {
        result->converged = 0;
        result->num_iter = 0;
        result->initial_residual_norm = 0.0;
        result->final_residual_norm = 0.0;
    }
}

void lmmc_itersolve_initial(lmmc_itersolve_result_t* result, lmmc_real_t norm) {
    if (result != NULL) {
        result->initial_residual_norm = norm;
        result->final_residual_norm = norm;
    }
}

void lmmc_itersolve_finish(lmmc_itersolve_result_t* result,
    int converged, size_t iterations, lmmc_real_t norm) {
    if (result != NULL) {
        result->converged = converged;
        result->num_iter = iterations;
        if (isfinite(norm)) {
            result->final_residual_norm = norm;
        }
    }
}

lmmc_status_t lmmc_itersolve_residual(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_vec_t* x, lmmc_vec_t* ax, lmmc_vec_t* r) {
    lmmc_status_t st = lmmc_sparse_mat_vec_mul(a, x, ax);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    for (size_t i = 0; i < b->size; ++i) {
        r->data[i] = b->data[i] - ax->data[i];
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_itersolve_threshold(const lmmc_itersolve_config_t* cfg,
    const lmmc_vec_t* b, const lmmc_vec_t* r, lmmc_real_t* norm,
    lmmc_real_t* threshold) {
    lmmc_real_t norm_b;
    lmmc_status_t st = lmmc_vec_norm2_checked(b, &norm_b);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_norm2_checked(r, norm);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    *threshold = cfg->abs_tol + cfg->rel_tol * norm_b;
    return isfinite(*threshold) ? LMMC_STATUS_OK : LMMC_STATUS_NUMERICAL_FAILURE;
}

lmmc_status_t lmmc_itersolve_dispatch_config(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_itersolve_config_t* cfg,
    const lmmc_vec_t* x, lmmc_itersolve_config_t* local) {
    if (b == NULL || x == NULL || b->data == NULL || x->data == NULL ||
        b->size == 0 || x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a != NULL && lmmc_sparse_validate(a) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (cfg != NULL) {
        *local = *cfg;
    }
    else {
        lmmc_status_t st = lmmc_itersolve_default_config(b->size, local);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }
    if ((local->apply_op != NULL) == (a != NULL)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}
/** @brief 通过无矩阵算子或稀疏矩阵计算 y = A*x。 */
lmmc_status_t lmmc_itersolve_matvec(
    const lmmc_sparse_mat_t* a,
    const lmmc_itersolve_config_t* cfg,
    const lmmc_vec_t* x_in,
    lmmc_vec_t* y_out
) {
    if (cfg->apply_op != NULL) {
        return cfg->apply_op(x_in, y_out, cfg->op_user_data);
    }
    return lmmc_sparse_mat_vec_mul(a, x_in, y_out);
}

/** @brief 对 CSR 稀疏矩阵计算 y = A^T * x。 */
static lmmc_status_t lmmc_matvec_transpose_csr(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* x_in,
    lmmc_vec_t* y_out
) {
    size_t i, j;
    for (i = 0; i < y_out->size; ++i) {
        y_out->data[i] = 0.0;
    }
    for (i = 0; i < a->rows; ++i) {
        size_t start = a->row_ptr[i];
        size_t end = a->row_ptr[i + 1];
        for (j = start; j < end; ++j) {
            size_t col = a->col_idx[j];
            y_out->data[col] += a->values[j] * x_in->data[i];
        }
    }
    return LMMC_STATUS_OK;
}

/** @brief 对 CSC 稀疏矩阵计算 y = A^T * x。 */
static lmmc_status_t lmmc_matvec_transpose_csc(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* x_in,
    lmmc_vec_t* y_out
) {
    size_t i, j;
    for (j = 0; j < y_out->size; ++j) {
        y_out->data[j] = 0.0;
    }
    /** @brief CSC 的 row_ptr 存列指针，col_idx 存行索引。 */
    for (j = 0; j < a->cols; ++j) {
        size_t start = a->row_ptr[j];
        size_t end = a->row_ptr[j + 1];
        double acc = 0.0;
        for (i = start; i < end; ++i) {
            size_t row = a->col_idx[i];
            acc += a->values[i] * x_in->data[row];
        }
        y_out->data[j] = acc;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_itersolve_transpose(
    const lmmc_sparse_mat_t* a,
    const lmmc_itersolve_config_t* cfg,
    const lmmc_vec_t* x_in,
    lmmc_vec_t* y_out
) {
    if (cfg->apply_transpose_op != NULL) {
        return cfg->apply_transpose_op(x_in, y_out, cfg->op_user_data);
    }
    if (lmmc_sparse_validate(a) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->format == LMMC_SPARSE_CSR) {
        return lmmc_matvec_transpose_csr(a, x_in, y_out);
    }
    return lmmc_matvec_transpose_csc(a, x_in, y_out);
}
