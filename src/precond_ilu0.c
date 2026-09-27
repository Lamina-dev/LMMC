#include "precond_internal.h"

static lmmc_status_t lmmc_ilu0_clone(const lmmc_sparse_mat_t* a,
    lmmc_precond_ilu_impl_t** out) {
    size_t idx_bytes, val_bytes;
    if (!lmmc_safe_mul_size(a->nnz, sizeof(size_t), &idx_bytes) ||
        !lmmc_safe_mul_size(a->nnz, sizeof(lmmc_real_t), &val_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_precond_ilu_impl_t* impl = NULL;
    lmmc_status_t st = lmmc_ilu_impl_create(a->rows, &impl);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    impl->nnz = impl->capacity = a->nnz;
    if (a->nnz > 0) {
        impl->col_idx = lmmc_memory_alloc(idx_bytes);
        impl->lu_values = lmmc_memory_alloc(val_bytes);
        if (impl->col_idx == NULL || impl->lu_values == NULL) {
            lmmc_ilu_impl_destroy(impl);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        memcpy(impl->col_idx, a->col_idx, idx_bytes);
        memcpy(impl->lu_values, a->values, val_bytes);
    }
    memcpy(impl->row_ptr, a->row_ptr, (a->rows + 1) * sizeof(size_t));
    *out = impl;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_ilu0_diagonals(lmmc_precond_ilu_impl_t* impl) {
    for (size_t i = 0; i < impl->size; ++i) {
        impl->diag_pos[i] = (size_t)-1;
        if (!lmmc_find_col_pos(impl->col_idx, impl->row_ptr[i],
            impl->row_ptr[i + 1], i, &impl->diag_pos[i])) {
            return LMMC_STATUS_SINGULAR_MATRIX;
        }
    }
    return LMMC_STATUS_OK;
}

/** @brief 按存储列顺序遍历，包括未排序行。 */
static lmmc_real_t lmmc_ilu0_entry_sum(const lmmc_precond_ilu_impl_t* impl,
    size_t i, size_t p, size_t limit) {
    const size_t j = impl->col_idx[p];
    lmmc_real_t sum = impl->lu_values[p];
    for (size_t q = impl->row_ptr[i]; q < impl->row_ptr[i + 1]; ++q) {
        size_t k = impl->col_idx[q];
        size_t pos;
        if (k >= limit) {
            continue;
        }
        if (lmmc_find_col_pos(impl->col_idx, impl->row_ptr[k],
            impl->row_ptr[k + 1], j, &pos)) {
            lmmc_real_t product = impl->lu_values[q] * impl->lu_values[pos];
            sum = sum - product;
        }
    }
    return sum;
}

static lmmc_status_t lmmc_ilu0_lower(lmmc_precond_ilu_impl_t* impl, size_t i) {
    for (size_t p = impl->row_ptr[i]; p < impl->row_ptr[i + 1]; ++p) {
        size_t j = impl->col_idx[p];
        if (j >= i) {
            continue;
        }
        lmmc_real_t sum = lmmc_ilu0_entry_sum(impl, i, p, j);
        lmmc_real_t diag = impl->lu_values[impl->diag_pos[j]];
        if (!isfinite(sum) || !isfinite(diag) || fabs(diag) <= 1e-15) {
            return LMMC_STATUS_SINGULAR_MATRIX;
        }
        impl->lu_values[p] = sum / diag;
        if (!isfinite(impl->lu_values[p])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_ilu0_upper(lmmc_precond_ilu_impl_t* impl, size_t i) {
    for (size_t p = impl->row_ptr[i]; p < impl->row_ptr[i + 1]; ++p) {
        if (impl->col_idx[p] < i) {
            continue;
        }
        lmmc_real_t sum = lmmc_ilu0_entry_sum(impl, i, p, i);
        if (!isfinite(sum)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        impl->lu_values[p] = sum;
    }
    lmmc_real_t diag = impl->lu_values[impl->diag_pos[i]];
    if (!isfinite(diag) || fabs(diag) <= 1e-15) {
        return LMMC_STATUS_SINGULAR_MATRIX;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_precond_create_ilu0(const lmmc_sparse_mat_t* a,
    lmmc_precond_t* out_precond) {
    lmmc_status_t st = lmmc_sparse_validate(a);
    if (out_precond == NULL || st != LMMC_STATUS_OK || a->format != LMMC_SPARSE_CSR) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    lmmc_precond_ilu_impl_t* impl = NULL;
    st = lmmc_ilu0_clone(a, &impl);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_ilu0_diagonals(impl);
    if (st != LMMC_STATUS_OK) {
        goto fail;
    }
    for (size_t i = 0; i < a->rows; ++i) {
        st = lmmc_ilu0_lower(impl, i);
        if (st != LMMC_STATUS_OK) {
            goto fail;
        }
        st = lmmc_ilu0_upper(impl, i);
        if (st != LMMC_STATUS_OK) {
            goto fail;
        }
    }
    out_precond->type = LMMC_PRECOND_ILU0;
    out_precond->size = a->rows;
    out_precond->impl = impl;
    out_precond->owns_data = 1;
    return LMMC_STATUS_OK;
fail:
    lmmc_ilu_impl_destroy(impl);
    return st;
}
