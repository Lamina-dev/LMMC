/**
 * @file sparse_symmetric.c
 * @brief 对称稀疏（SymCSR）格式：由 CSR 构建、对称 SpMV 与销毁。
 */
#include "memory_bridge.h"
#include "internal.h"
#include "sparse_internal.h"
#include "lmmc/sparse.h"

lmmc_status_t lmmc_sparse_sym_csr_from_csr(const lmmc_sparse_mat_t* full,
    lmmc_sparse_sym_half_t half, lmmc_sparse_sym_csr_t* out) {
    size_t i, p;
    size_t n;
    size_t nnz_half = 0;
    size_t nz_idx = 0;
    size_t row_ptr_bytes, col_idx_bytes, val_bytes;
    lmmc_status_t st;

    if (full == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_sparse_validate(full);
    if (st != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (full->rows != full->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (full->format != LMMC_SPARSE_CSR) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = full->rows;

    /* 第一遍：计算半三角中的非零元数量 */
    for (i = 0; i < n; ++i) {
        for (p = full->row_ptr[i]; p < full->row_ptr[i + 1]; ++p) {
            size_t j = full->col_idx[p];
            if (half == LMMC_SPARSE_SYM_UPPER) {
                if (j >= i) ++nnz_half;
            } else {
                if (j <= i) ++nnz_half;
            }
        }
    }

    /* 分配内存 */
    if (!lmmc_safe_mul_size(n + 1, sizeof(size_t), &row_ptr_bytes) ||
        !lmmc_safe_mul_size(nnz_half, sizeof(size_t), &col_idx_bytes) ||
        !lmmc_safe_mul_size(nnz_half, sizeof(lmmc_real_t), &val_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    out->row_ptr = (size_t*)lmmc_alloc(row_ptr_bytes);
    if (out->row_ptr == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    out->col_idx = NULL;
    out->values = NULL;

    if (nnz_half > 0) {
        out->col_idx = (size_t*)lmmc_alloc(col_idx_bytes);
        if (out->col_idx == NULL) {
            lmmc_free(out->row_ptr);
            out->row_ptr = NULL;
            return LMMC_STATUS_ALLOCATION_FAILED;
        }

        out->values = (lmmc_real_t*)lmmc_alloc(val_bytes);
        if (out->values == NULL) {
            lmmc_free(out->col_idx);
            lmmc_free(out->row_ptr);
            out->col_idx = NULL;
            out->row_ptr = NULL;
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
    }
    if (nnz_half == 0) {
        for (i = 0; i < n; ++i) {
            out->row_ptr[i] = 0;
        }
        out->row_ptr[n] = 0;
        out->n = n;
        out->nnz = 0;
        out->half = half;
        out->owns_data = 1;
        return LMMC_STATUS_OK;
    }

    /* 第二遍：填充数据 */
    nz_idx = 0;
    out->row_ptr[0] = 0;
    for (i = 0; i < n; ++i) {
        for (p = full->row_ptr[i]; p < full->row_ptr[i + 1]; ++p) {
            size_t j = full->col_idx[p];
            int keep = 0;
            if (half == LMMC_SPARSE_SYM_UPPER) {
                keep = (j >= i);
            } else {
                keep = (j <= i);
            }
            if (keep) {
                out->col_idx[nz_idx] = j;
                LMMC_REAL_SET(&out->values[nz_idx], &full->values[p]);
                ++nz_idx;
            }
        }
        out->row_ptr[i + 1] = nz_idx;
    }

    out->n = n;
    out->nnz = nnz_half;
    out->half = half;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_sym_spmv(const lmmc_sparse_sym_csr_t* A,
    const lmmc_vec_t* x, lmmc_vec_t* y) {
    size_t i, p;
    lmmc_real_t zero;
    lmmc_real_t prod;

    if (A == NULL || x == NULL || y == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->data == NULL || y->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (A->row_ptr == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (A->nnz > 0 && (A->col_idx == NULL || A->values == NULL)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size != A->n || y->size != A->n) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    /* 清零输出向量 */
    LMMC_REAL_INIT(&zero);
    LMMC_REAL_INIT(&prod);
    LMMC_REAL_SET_D(&zero, 0.0);

    for (i = 0; i < A->n; ++i) {
        LMMC_REAL_SET_D(&y->data[i], 0.0);
    }

    /* 对称 SpMV：利用对称性，每个非对角元贡献两次 */
    for (i = 0; i < A->n; ++i) {
        for (p = A->row_ptr[i]; p < A->row_ptr[i + 1]; ++p) {
            size_t j = A->col_idx[p];
            LMMC_REAL_MUL(&prod, &A->values[p], &x->data[j]);

            /* y[i] += A[i,j] * x[j] */
            {
                lmmc_real_t tmp;
                LMMC_REAL_INIT(&tmp);
                LMMC_REAL_ADD(&tmp, &y->data[i], &prod);
                LMMC_REAL_SET(&y->data[i], &tmp);
                LMMC_REAL_CLEAR(&tmp);
            }

            /* 对称贡献：y[j] += A[i,j] * x[i]（仅非对角元） */
            if (i != j) {
                lmmc_real_t prod2;
                lmmc_real_t tmp2;
                LMMC_REAL_INIT(&prod2);
                LMMC_REAL_INIT(&tmp2);
                LMMC_REAL_MUL(&prod2, &A->values[p], &x->data[i]);
                LMMC_REAL_ADD(&tmp2, &y->data[j], &prod2);
                LMMC_REAL_SET(&y->data[j], &tmp2);
                LMMC_REAL_CLEAR(&prod2);
                LMMC_REAL_CLEAR(&tmp2);
            }
        }
    }

    LMMC_REAL_CLEAR(&zero);
    LMMC_REAL_CLEAR(&prod);
    return LMMC_STATUS_OK;
}

void lmmc_sparse_sym_csr_destroy(lmmc_sparse_sym_csr_t* s) {
    if (s == NULL) return;
    if (s->owns_data) {
        if (s->row_ptr != NULL) lmmc_free(s->row_ptr);
        if (s->col_idx != NULL) lmmc_free(s->col_idx);
        if (s->values != NULL) lmmc_free(s->values);
    }
    s->row_ptr = NULL;
    s->col_idx = NULL;
    s->values = NULL;
    s->n = 0;
    s->nnz = 0;
    s->owns_data = 0;
}
