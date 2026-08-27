/**
 * @file sparse_storage.c
 * @brief 稀疏矩阵存储生命周期：创建、外部数据包装与销毁。
 */
#include <string.h>

#include "memory_bridge.h"
#include "internal.h"
#include "sparse_internal.h"
#include "lmmc/config.h"
#include "lmmc/sparse.h"


lmmc_status_t lmmc_sparse_create_csr(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse) {
    return lmmc_sparse_create(rows, cols, nnz, LMMC_SPARSE_CSR, out_sparse);
}

lmmc_status_t lmmc_sparse_create_csc(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse) {
    return lmmc_sparse_create(rows, cols, nnz, LMMC_SPARSE_CSC, out_sparse);
}

lmmc_status_t lmmc_sparse_wrap_csr(
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t* row_ptr,
    size_t* col_idx,
    lmmc_real_t* values,
    lmmc_sparse_mat_t* out_sparse
) {
    lmmc_sparse_mat_t candidate = {0};
    if (out_sparse == NULL || rows == 0 || cols == 0 || row_ptr == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (nnz > 0 && (col_idx == NULL || values == NULL)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    candidate.rows = rows;
    candidate.cols = cols;
    candidate.nnz = nnz;
    candidate.row_ptr = row_ptr;
    candidate.col_idx = col_idx;
    candidate.values = values;
    candidate.format = LMMC_SPARSE_CSR;
    candidate.owns_data = 0;

    if (lmmc_sparse_validate(&candidate) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    *out_sparse = candidate;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_wrap_csc(
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t* col_ptr,
    size_t* row_idx,
    lmmc_real_t* values,
    lmmc_sparse_mat_t* out_sparse
) {
    lmmc_sparse_mat_t candidate = {0};
    if (out_sparse == NULL || rows == 0 || cols == 0 || col_ptr == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (nnz > 0 && (row_idx == NULL || values == NULL)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    candidate.rows = rows;
    candidate.cols = cols;
    candidate.nnz = nnz;
    candidate.row_ptr = col_ptr;
    candidate.col_idx = row_idx;
    candidate.values = values;
    candidate.format = LMMC_SPARSE_CSC;
    candidate.owns_data = 0;

    if (lmmc_sparse_validate(&candidate) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    *out_sparse = candidate;
    return LMMC_STATUS_OK;
}

void lmmc_sparse_destroy(lmmc_sparse_mat_t* sparse) {
    if (sparse == NULL) {
        return;
    }
    if (sparse->owns_data) {
        if (sparse->row_ptr != NULL) lmmc_free(sparse->row_ptr);
        if (sparse->col_idx != NULL) lmmc_free(sparse->col_idx);
        if (sparse->values != NULL) {
            for (size_t i = 0; i < sparse->nnz; ++i) {
                LMMC_REAL_CLEAR(&sparse->values[i]);
            }
            lmmc_free(sparse->values);
        }
    }
    sparse->rows = 0;
    sparse->cols = 0;
    sparse->nnz = 0;
    sparse->row_ptr = NULL;
    sparse->col_idx = NULL;
    sparse->values = NULL;
    sparse->owns_data = 0;
}
