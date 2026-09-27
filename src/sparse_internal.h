/**
 * @file sparse_internal.h
 * @brief 稀疏矩阵模块内部共享辅助函数（仅源文件可见）。
 *
 * 提供稀疏矩阵结构校验与存储创建等内联辅助,
 * 通用安全算术由 internal.h 提供.
 *
 * @internal
 */
#ifndef LMMC_SPARSE_INTERNAL_H
#define LMMC_SPARSE_INTERNAL_H

#include <stddef.h>
#include <string.h>

#include "internal.h"
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/sparse.h"


static inline lmmc_status_t lmmc_sparse_validate_outer(
    const lmmc_sparse_mat_t* sparse, size_t outer_size) {
    if (sparse->row_ptr[0] != 0 || sparse->row_ptr[outer_size] != sparse->nnz) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < outer_size; ++i) {
        if (sparse->row_ptr[i] > sparse->row_ptr[i + 1] ||
            sparse->row_ptr[i + 1] > sparse->nnz) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_sparse_validate_inner(
    const lmmc_sparse_mat_t* sparse, size_t outer_size, size_t inner_limit) {
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
            if (sparse->col_idx[p] >= inner_limit ||
                (p > sparse->row_ptr[i] && sparse->col_idx[p - 1] >= sparse->col_idx[p])) {
                return LMMC_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_sparse_validate(const lmmc_sparse_mat_t* sparse) {
    size_t outer_size = 0;
    size_t inner_limit = 0;

    if (sparse == NULL || sparse->row_ptr == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if ((sparse->format != LMMC_SPARSE_CSR && sparse->format != LMMC_SPARSE_CSC) ||
        sparse->rows == 0 || sparse->cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (sparse->nnz > 0 && (sparse->col_idx == NULL || sparse->values == NULL)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (sparse->format == LMMC_SPARSE_CSR) {
        outer_size = sparse->rows;
        inner_limit = sparse->cols;
    } else {
        outer_size = sparse->cols;
        inner_limit = sparse->rows;
    }

    const lmmc_status_t status = lmmc_sparse_validate_outer(sparse, outer_size);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    return lmmc_sparse_validate_inner(sparse, outer_size, inner_limit);
}



static inline lmmc_status_t lmmc_sparse_storage_sizes(
    size_t rows, size_t cols, size_t nnz, lmmc_sparse_format_t format,
    size_t* outer_ptr_bytes, size_t* idx_bytes, size_t* val_bytes) {
    const size_t outer_size = (format == LMMC_SPARSE_CSR) ? rows : cols;
    if (rows == 0 || cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (format != LMMC_SPARSE_CSR && format != LMMC_SPARSE_CSC) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (outer_size == SIZE_MAX) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_safe_mul_size(outer_size + 1, sizeof(size_t), outer_ptr_bytes) ||
        !lmmc_safe_mul_size(nnz, sizeof(size_t), idx_bytes) ||
        !lmmc_safe_mul_size(nnz, sizeof(lmmc_real_t), val_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

/**
 * @internal
 * @brief 分配并初始化 CSR/CSC 稀疏矩阵存储。
 */
static inline lmmc_status_t lmmc_sparse_create(size_t rows, size_t cols, size_t nnz, lmmc_sparse_format_t format, lmmc_sparse_mat_t* out_sparse) {
    size_t outer_ptr_bytes = 0;
    size_t idx_bytes = 0;
    size_t val_bytes = 0;
    size_t* outer_ptr = NULL;
    size_t* inner_idx = NULL;
    lmmc_real_t* values = NULL;

    if (out_sparse == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    const lmmc_status_t status = lmmc_sparse_storage_sizes(
        rows, cols, nnz, format, &outer_ptr_bytes, &idx_bytes, &val_bytes);
    if (status != LMMC_STATUS_OK) {
        return status;
    }

    outer_ptr = (size_t*)lmmc_memory_alloc(outer_ptr_bytes);
    if (outer_ptr == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(outer_ptr, 0, outer_ptr_bytes);

    if (nnz > 0) {
        inner_idx = (size_t*)lmmc_memory_alloc(idx_bytes);
        values = (lmmc_real_t*)lmmc_memory_alloc(val_bytes);
        if (inner_idx == NULL || values == NULL) {
            if (inner_idx != NULL) {
                lmmc_memory_free(inner_idx);
            }
            if (values != NULL) {
                lmmc_memory_free(values);
            }
            lmmc_memory_free(outer_ptr);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        memset(inner_idx, 0, idx_bytes);
        memset(values, 0, val_bytes);
        for (size_t i = 0; i < nnz; ++i) {
            LMMC_REAL_INIT(&values[i]);
        }
    }

    out_sparse->rows = rows;
    out_sparse->cols = cols;
    out_sparse->nnz = nnz;
    out_sparse->row_ptr = outer_ptr;
    out_sparse->col_idx = inner_idx;
    out_sparse->values = values;
    out_sparse->format = format;
    out_sparse->owns_data = 1;
    return LMMC_STATUS_OK;
}
#endif
