#include "precond_internal.h"

int lmmc_find_col_pos(
    const size_t* col_idx,
    size_t start,
    size_t end,
    size_t target_col,
    size_t* out_pos
) {
    size_t p = 0;

    if (col_idx == NULL || out_pos == NULL || start > end) {
        return 0;
    }

    for (p = start; p < end; ++p) {
        if (col_idx[p] == target_col) {
            *out_pos = p;
            return 1;
        }
    }
    return 0;
}

void lmmc_ilu_impl_destroy(lmmc_precond_ilu_impl_t* impl) {
    if (impl == NULL) {
        return;
    }

    if (impl->lu_values != NULL) {
        size_t k;
        for (k = 0; k < impl->capacity; ++k) {
            LMMC_REAL_CLEAR(&impl->lu_values[k]);
        }
        lmmc_memory_free(impl->lu_values);
    }
    if (impl->y_arr != NULL) {
        size_t k;
        for (k = 0; k < impl->size; ++k) {
            LMMC_REAL_CLEAR(&impl->y_arr[k]);
        }
        lmmc_memory_free(impl->y_arr);
    }
    lmmc_memory_free(impl->diag_pos);
    lmmc_memory_free(impl->col_idx);
    lmmc_memory_free(impl->row_ptr);
    lmmc_memory_free(impl);
}

lmmc_status_t lmmc_ilu_impl_reserve(lmmc_precond_ilu_impl_t* impl, size_t required_capacity) {
    size_t new_capacity = 0;
    size_t idx_bytes = 0;
    size_t val_bytes = 0;
    size_t* new_col_idx = NULL;
    lmmc_real_t* new_lu_values = NULL;

    if (impl == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (required_capacity <= impl->capacity) {
        return LMMC_STATUS_OK;
    }

    new_capacity = (impl->capacity == 0) ? 16 : impl->capacity;
    while (new_capacity < required_capacity) {
        size_t doubled = 0;
        if (!lmmc_safe_mul_size(new_capacity, 2, &doubled)) {
            new_capacity = required_capacity;
            break;
        }
        new_capacity = doubled;
    }

    if (!lmmc_safe_mul_size(new_capacity, sizeof(size_t), &idx_bytes) ||
        !lmmc_safe_mul_size(new_capacity, sizeof(lmmc_real_t), &val_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    new_col_idx = (size_t*)lmmc_memory_alloc(idx_bytes);
    new_lu_values = (lmmc_real_t*)lmmc_memory_alloc(val_bytes);
    if (new_col_idx == NULL || new_lu_values == NULL) {
        lmmc_memory_free(new_lu_values);
        lmmc_memory_free(new_col_idx);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    size_t k;
    for (k = 0; k < new_capacity; ++k) {
        LMMC_REAL_INIT(&new_lu_values[k]);
        LMMC_REAL_SET_D(&new_lu_values[k], 0.0);
    }

    if (impl->nnz > 0) {
        memcpy(new_col_idx, impl->col_idx, impl->nnz * sizeof(size_t));
        for (k = 0; k < impl->nnz; ++k) {
            LMMC_REAL_SET(&new_lu_values[k], &impl->lu_values[k]);
        }
    }

    if (impl->lu_values != NULL) {
        for (k = 0; k < impl->capacity; ++k) {
            LMMC_REAL_CLEAR(&impl->lu_values[k]);
        }
        lmmc_memory_free(impl->lu_values);
    }

    lmmc_memory_free(impl->col_idx);
    impl->col_idx = new_col_idx;
    impl->lu_values = new_lu_values;
    impl->capacity = new_capacity;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ilu_impl_create(size_t size, lmmc_precond_ilu_impl_t** out) {
    size_t rows, row_bytes, diag_bytes, y_bytes;
    if (!lmmc_safe_add_size(size, 1, &rows) ||
        !lmmc_safe_mul_size(rows, sizeof(size_t), &row_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_safe_mul_size(size, sizeof(size_t), &diag_bytes) ||
        !lmmc_safe_mul_size(size, sizeof(lmmc_real_t), &y_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_precond_ilu_impl_t* impl = lmmc_memory_alloc(sizeof(*impl));
    if (impl == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(impl, 0, sizeof(*impl));
    impl->size = size;
    impl->row_ptr = lmmc_memory_alloc(row_bytes);
    impl->diag_pos = lmmc_memory_alloc(diag_bytes);
    impl->y_arr = lmmc_memory_alloc(y_bytes);
    if (impl->row_ptr == NULL || impl->diag_pos == NULL || impl->y_arr == NULL) {
        lmmc_ilu_impl_destroy(impl);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(impl->y_arr, 0, y_bytes);
    *out = impl;
    return LMMC_STATUS_OK;
}
