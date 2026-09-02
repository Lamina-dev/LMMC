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


static inline lmmc_status_t lmmc_sparse_validate(const lmmc_sparse_mat_t* sparse) {
    size_t i = 0;
    size_t outer_size = 0;
    size_t inner_limit = 0;

    if (sparse == NULL || sparse->row_ptr == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (sparse->rows == 0 || sparse->cols == 0) {
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

    if (sparse->row_ptr[0] != 0 || sparse->row_ptr[outer_size] != sparse->nnz) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < outer_size; ++i) {
        if (sparse->row_ptr[i] > sparse->row_ptr[i + 1]) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }

    for (i = 0; i < sparse->nnz; ++i) {
        if (sparse->col_idx[i] >= inner_limit) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    return LMMC_STATUS_OK;
}



/**
 * @internal
 * @brief 分配并初始化 CSR/CSC 稀疏矩阵存储。
 */
static inline lmmc_status_t lmmc_sparse_create(size_t rows, size_t cols, size_t nnz, lmmc_sparse_format_t format, lmmc_sparse_mat_t* out_sparse) {
    size_t outer_size = (format == LMMC_SPARSE_CSR) ? rows : cols;
    size_t outer_ptr_bytes = 0;
    size_t idx_bytes = 0;
    size_t val_bytes = 0;
    size_t* outer_ptr = NULL;
    size_t* inner_idx = NULL;
    lmmc_real_t* values = NULL;

    if (out_sparse == NULL || rows == 0 || cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_safe_mul_size(outer_size + 1, sizeof(size_t), &outer_ptr_bytes) ||
        !lmmc_safe_mul_size(nnz, sizeof(size_t), &idx_bytes) ||
        !lmmc_safe_mul_size(nnz, sizeof(lmmc_real_t), &val_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    outer_ptr = (size_t*)lmmc_alloc(outer_ptr_bytes);
    if (outer_ptr == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(outer_ptr, 0, outer_ptr_bytes);

    if (nnz > 0) {
        inner_idx = (size_t*)lmmc_alloc(idx_bytes);
        values = (lmmc_real_t*)lmmc_alloc(val_bytes);
        if (inner_idx == NULL || values == NULL) {
            if (inner_idx != NULL) lmmc_free(inner_idx);
            if (values != NULL) lmmc_free(values);
            lmmc_free(outer_ptr);
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
