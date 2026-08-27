/**
 * @file sparse_block.c
 * @brief 块稀疏（BSR）存储：创建、稠密互转与销毁。
 */
#include <string.h>

#include "memory_bridge.h"
#include "internal.h"
#include "sparse_internal.h"
#include "lmmc/sparse.h"

lmmc_status_t lmmc_sparse_bsr_create(size_t rows, size_t cols, size_t block_size,
    size_t nnz_blocks, lmmc_sparse_bsr_t* out) {
    size_t row_ptr_bytes = 0;
    size_t col_idx_bytes = 0;
    size_t val_count = 0;
    size_t val_bytes = 0;

    if (out == NULL || rows == 0 || cols == 0 || block_size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* 计算所需内存大小，检查溢出 */
    if (lmmc_mul_overflow_size(rows + 1, sizeof(size_t), &row_ptr_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_mul_overflow_size(nnz_blocks, sizeof(size_t), &col_idx_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_mul_overflow_size(block_size, block_size, &val_count)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_mul_overflow_size(val_count, nnz_blocks, &val_count)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_mul_overflow_size(val_count, sizeof(lmmc_real_t), &val_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    out->row_ptr = (size_t*)lmmc_alloc(row_ptr_bytes);
    if (out->row_ptr == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(out->row_ptr, 0, row_ptr_bytes);

    out->col_idx = NULL;
    out->values = NULL;

    if (nnz_blocks > 0) {
        out->col_idx = (size_t*)lmmc_alloc(col_idx_bytes);
        if (out->col_idx == NULL) {
            lmmc_free(out->row_ptr);
            out->row_ptr = NULL;
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        memset(out->col_idx, 0, col_idx_bytes);

        out->values = (lmmc_real_t*)lmmc_alloc(val_bytes);
        if (out->values == NULL) {
            lmmc_free(out->col_idx);
            lmmc_free(out->row_ptr);
            out->col_idx = NULL;
            out->row_ptr = NULL;
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        memset(out->values, 0, val_bytes);
    }

    out->rows = rows;
    out->cols = cols;
    out->block_size = block_size;
    out->nnz_blocks = nnz_blocks;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_bsr_to_dense(const lmmc_sparse_bsr_t* bsr, lmmc_mat_t* out) {
    size_t bi, p, li, lj;
    size_t dense_rows, dense_cols;
    lmmc_real_t zero;
    lmmc_status_t st;

    if (bsr == NULL || out == NULL || out->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (bsr->rows == 0 || bsr->cols == 0 || bsr->block_size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (bsr->row_ptr == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (bsr->nnz_blocks > 0 && (bsr->col_idx == NULL || bsr->values == NULL)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    dense_rows = bsr->rows * bsr->block_size;
    dense_cols = bsr->cols * bsr->block_size;

    if (out->rows != dense_rows || out->cols != dense_cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    /* 先清零输出矩阵 */
    LMMC_REAL_INIT(&zero);
    LMMC_REAL_SET_D(&zero, 0.0);
    st = lmmc_mat_fill(out, zero);
    LMMC_REAL_CLEAR(&zero);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    /* 遍历每个块行 */
    for (bi = 0; bi < bsr->rows; ++bi) {
        for (p = bsr->row_ptr[bi]; p < bsr->row_ptr[bi + 1]; ++p) {
            size_t bj = bsr->col_idx[p];
            size_t block_offset = p * bsr->block_size * bsr->block_size;
            size_t row_base = bi * bsr->block_size;
            size_t col_base = bj * bsr->block_size;

            /* 将块内元素写入稠密矩阵 */
            for (li = 0; li < bsr->block_size; ++li) {
                for (lj = 0; lj < bsr->block_size; ++lj) {
                    size_t dense_idx = (row_base + li) * out->stride + (col_base + lj);
                    size_t block_idx = block_offset + li * bsr->block_size + lj;
                    LMMC_REAL_SET(&out->data[dense_idx], &bsr->values[block_idx]);
                }
            }
        }
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_dense_to_bsr(const lmmc_mat_t* dense, size_t block_size,
    lmmc_real_t eps, lmmc_sparse_bsr_t* out) {
    size_t block_rows, block_cols;
    size_t bi, bj, li, lj;
    size_t nnz_blocks = 0;
    size_t nz_idx = 0;
    lmmc_status_t st;
    lmmc_real_t abs_v;
    lmmc_real_t zero;

    if (dense == NULL || out == NULL || dense->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (block_size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (dense->rows % block_size != 0 || dense->cols % block_size != 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_INIT(&abs_v);
    LMMC_REAL_INIT(&zero);
    LMMC_REAL_SET_D(&zero, 0.0);

    if (LMMC_REAL_CMP(&eps, &zero) < 0) {
        LMMC_REAL_CLEAR(&abs_v);
        LMMC_REAL_CLEAR(&zero);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    block_rows = dense->rows / block_size;
    block_cols = dense->cols / block_size;

    /* 第一遍：计算非零块数量 */
    for (bi = 0; bi < block_rows; ++bi) {
        for (bj = 0; bj < block_cols; ++bj) {
            int block_nonzero = 0;
            for (li = 0; li < block_size && !block_nonzero; ++li) {
                for (lj = 0; lj < block_size && !block_nonzero; ++lj) {
                    size_t row = bi * block_size + li;
                    size_t col = bj * block_size + lj;
                    LMMC_REAL_ABS(&abs_v, &dense->data[row * dense->stride + col]);
                    if (LMMC_REAL_CMP(&abs_v, &eps) > 0) {
                        block_nonzero = 1;
                    }
                }
            }
            if (block_nonzero) {
                ++nnz_blocks;
            }
        }
    }

    /* 分配 BSR 结构 */
    st = lmmc_sparse_bsr_create(block_rows, block_cols, block_size, nnz_blocks, out);
    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&abs_v);
        LMMC_REAL_CLEAR(&zero);
        return st;
    }

    /* 第二遍：填充数据 */
    nz_idx = 0;
    out->row_ptr[0] = 0;
    for (bi = 0; bi < block_rows; ++bi) {
        for (bj = 0; bj < block_cols; ++bj) {
            int block_nonzero = 0;
            for (li = 0; li < block_size && !block_nonzero; ++li) {
                for (lj = 0; lj < block_size && !block_nonzero; ++lj) {
                    size_t row = bi * block_size + li;
                    size_t col = bj * block_size + lj;
                    LMMC_REAL_ABS(&abs_v, &dense->data[row * dense->stride + col]);
                    if (LMMC_REAL_CMP(&abs_v, &eps) > 0) {
                        block_nonzero = 1;
                    }
                }
            }
            if (block_nonzero) {
                size_t block_offset = nz_idx * block_size * block_size;
                out->col_idx[nz_idx] = bj;

                /* 复制块内所有元素 */
                for (li = 0; li < block_size; ++li) {
                    for (lj = 0; lj < block_size; ++lj) {
                        size_t row = bi * block_size + li;
                        size_t col = bj * block_size + lj;
                        size_t blk_idx = block_offset + li * block_size + lj;
                        LMMC_REAL_SET(&out->values[blk_idx],
                                      &dense->data[row * dense->stride + col]);
                    }
                }
                ++nz_idx;
            }
        }
        out->row_ptr[bi + 1] = nz_idx;
    }

    LMMC_REAL_CLEAR(&abs_v);
    LMMC_REAL_CLEAR(&zero);
    return LMMC_STATUS_OK;
}

void lmmc_sparse_bsr_destroy(lmmc_sparse_bsr_t* bsr) {
    if (bsr == NULL) return;
    if (bsr->owns_data) {
        if (bsr->row_ptr != NULL) lmmc_free(bsr->row_ptr);
        if (bsr->col_idx != NULL) lmmc_free(bsr->col_idx);
        if (bsr->values != NULL) lmmc_free(bsr->values);
    }
    bsr->row_ptr = NULL;
    bsr->col_idx = NULL;
    bsr->values = NULL;
    bsr->rows = 0;
    bsr->cols = 0;
    bsr->block_size = 0;
    bsr->nnz_blocks = 0;
    bsr->owns_data = 0;
}
