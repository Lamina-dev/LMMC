/**
 * @file sparse_products.c
 * @brief 稀疏矩阵乘法：矩阵-向量、矩阵-稠密矩阵与稀疏×稀疏。
 */
#include <string.h>

#include "memory_bridge.h"
#include "internal.h"
#include "sparse_internal.h"
#include "lmmc/sparse.h"

lmmc_status_t lmmc_sparse_mat_vec_mul(const lmmc_sparse_mat_t* sparse, const lmmc_vec_t* x, lmmc_vec_t* y) {
    size_t i = 0, p = 0;
    lmmc_status_t st = lmmc_sparse_validate(sparse);
    if (st != LMMC_STATUS_OK || x == NULL || y == NULL || x->data == NULL || y->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size != sparse->cols || y->size != sparse->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    const lmmc_real_t* restrict vals = sparse->values;
    const size_t* restrict col_idx = sparse->col_idx;
    const size_t* restrict row_ptr = sparse->row_ptr;
    const lmmc_real_t* restrict x_data = x->data;
    lmmc_real_t* restrict y_data = y->data;

    lmmc_real_t sum; LMMC_REAL_INIT(&sum);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
    lmmc_real_t zero; LMMC_REAL_INIT(&zero);
    LMMC_REAL_SET_D(&zero, 0.0);

    if (sparse->format == LMMC_SPARSE_CSR) {
        for (i = 0; i < sparse->rows; ++i) {
            LMMC_REAL_SET_D(&sum, 0.0);
            size_t start = row_ptr[i];
            size_t end = row_ptr[i + 1];

            p = start;
            for (; p + 3 < end; p += 4) {
                LMMC_REAL_MUL(&tmp_mul, &vals[p], &x_data[col_idx[p]]);
                LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
                LMMC_REAL_SET(&sum, &tmp_sum);

                LMMC_REAL_MUL(&tmp_mul, &vals[p + 1], &x_data[col_idx[p + 1]]);
                LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
                LMMC_REAL_SET(&sum, &tmp_sum);

                LMMC_REAL_MUL(&tmp_mul, &vals[p + 2], &x_data[col_idx[p + 2]]);
                LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
                LMMC_REAL_SET(&sum, &tmp_sum);

                LMMC_REAL_MUL(&tmp_mul, &vals[p + 3], &x_data[col_idx[p + 3]]);
                LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
                LMMC_REAL_SET(&sum, &tmp_sum);
            }
            for (; p < end; ++p) {
                LMMC_REAL_MUL(&tmp_mul, &vals[p], &x_data[col_idx[p]]);
                LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
                LMMC_REAL_SET(&sum, &tmp_sum);
            }
            LMMC_REAL_SET(&y_data[i], &sum);
        }
    } else {
        st = lmmc_vec_fill(y, zero);
        if (st != LMMC_STATUS_OK) {
            LMMC_REAL_CLEAR(&sum);
            LMMC_REAL_CLEAR(&tmp_mul);
            LMMC_REAL_CLEAR(&tmp_sum);
            LMMC_REAL_CLEAR(&zero);
            return st;
        }
        for (i = 0; i < sparse->cols; ++i) {
            size_t start = row_ptr[i];
            size_t end = row_ptr[i + 1];
            for (p = start; p < end; ++p) {
                LMMC_REAL_MUL(&tmp_mul, &vals[p], &x_data[i]);
                LMMC_REAL_ADD(&tmp_sum, &y_data[col_idx[p]], &tmp_mul);
                LMMC_REAL_SET(&y_data[col_idx[p]], &tmp_sum);
            }
        }
    }

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    LMMC_REAL_CLEAR(&zero);
    return LMMC_STATUS_OK;
}
lmmc_status_t lmmc_sparse_mat_mat_mul_dense(const lmmc_sparse_mat_t* sparse, const lmmc_mat_t* b, lmmc_mat_t* c) {
    size_t i = 0, p = 0, j = 0;
    lmmc_status_t st = lmmc_sparse_validate(sparse);
    if (st != LMMC_STATUS_OK || b == NULL || c == NULL || b->data == NULL || c->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (sparse->cols != b->rows || c->rows != sparse->rows || c->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    const lmmc_real_t* restrict vals = sparse->values;
    const size_t* restrict col_idx = sparse->col_idx;
    const size_t* restrict row_ptr = sparse->row_ptr;
    const lmmc_real_t* restrict b_data = b->data;
    lmmc_real_t* restrict c_data = c->data;

    size_t b_stride = b->stride;
    size_t c_stride = c->stride;
    size_t b_cols = b->cols;

    lmmc_real_t zero; LMMC_REAL_INIT(&zero);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_SET_D(&zero, 0.0);

    st = lmmc_mat_fill(c, zero);
    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&zero);
        LMMC_REAL_CLEAR(&tmp_mul);
        LMMC_REAL_CLEAR(&tmp_sum);
        return st;
    }

    if (sparse->format == LMMC_SPARSE_CSR) {
        for (i = 0; i < sparse->rows; ++i) {
            size_t start = row_ptr[i];
            size_t end = row_ptr[i + 1];
            for (p = start; p < end; ++p) {
                size_t k = col_idx[p];
                lmmc_real_t val_p; LMMC_REAL_INIT(&val_p);
                LMMC_REAL_SET(&val_p, &vals[p]);

                size_t j_limit = b_cols & ~((size_t)3);
                for (j = 0; j < j_limit; j += 4) {
                    LMMC_REAL_MUL(&tmp_mul, &val_p, &b_data[k * b_stride + j]);
                    LMMC_REAL_ADD(&tmp_sum, &c_data[i * c_stride + j], &tmp_mul);
                    LMMC_REAL_SET(&c_data[i * c_stride + j], &tmp_sum);

                    LMMC_REAL_MUL(&tmp_mul, &val_p, &b_data[k * b_stride + j + 1]);
                    LMMC_REAL_ADD(&tmp_sum, &c_data[i * c_stride + j + 1], &tmp_mul);
                    LMMC_REAL_SET(&c_data[i * c_stride + j + 1], &tmp_sum);

                    LMMC_REAL_MUL(&tmp_mul, &val_p, &b_data[k * b_stride + j + 2]);
                    LMMC_REAL_ADD(&tmp_sum, &c_data[i * c_stride + j + 2], &tmp_mul);
                    LMMC_REAL_SET(&c_data[i * c_stride + j + 2], &tmp_sum);

                    LMMC_REAL_MUL(&tmp_mul, &val_p, &b_data[k * b_stride + j + 3]);
                    LMMC_REAL_ADD(&tmp_sum, &c_data[i * c_stride + j + 3], &tmp_mul);
                    LMMC_REAL_SET(&c_data[i * c_stride + j + 3], &tmp_sum);
                }
                for (; j < b_cols; ++j) {
                    LMMC_REAL_MUL(&tmp_mul, &val_p, &b_data[k * b_stride + j]);
                    LMMC_REAL_ADD(&tmp_sum, &c_data[i * c_stride + j], &tmp_mul);
                    LMMC_REAL_SET(&c_data[i * c_stride + j], &tmp_sum);
                }
                LMMC_REAL_CLEAR(&val_p);
            }
        }
    } else {
        for (i = 0; i < sparse->cols; ++i) {
            size_t start = row_ptr[i];
            size_t end = row_ptr[i + 1];
            for (p = start; p < end; ++p) {
                size_t row = col_idx[p];
                lmmc_real_t val_p; LMMC_REAL_INIT(&val_p);
                LMMC_REAL_SET(&val_p, &vals[p]);

                size_t j_limit = b_cols & ~((size_t)3);
                for (j = 0; j < j_limit; j += 4) {
                    LMMC_REAL_MUL(&tmp_mul, &val_p, &b_data[i * b_stride + j]);
                    LMMC_REAL_ADD(&tmp_sum, &c_data[row * c_stride + j], &tmp_mul);
                    LMMC_REAL_SET(&c_data[row * c_stride + j], &tmp_sum);

                    LMMC_REAL_MUL(&tmp_mul, &val_p, &b_data[i * b_stride + j + 1]);
                    LMMC_REAL_ADD(&tmp_sum, &c_data[row * c_stride + j + 1], &tmp_mul);
                    LMMC_REAL_SET(&c_data[row * c_stride + j + 1], &tmp_sum);

                    LMMC_REAL_MUL(&tmp_mul, &val_p, &b_data[i * b_stride + j + 2]);
                    LMMC_REAL_ADD(&tmp_sum, &c_data[row * c_stride + j + 2], &tmp_mul);
                    LMMC_REAL_SET(&c_data[row * c_stride + j + 2], &tmp_sum);

                    LMMC_REAL_MUL(&tmp_mul, &val_p, &b_data[i * b_stride + j + 3]);
                    LMMC_REAL_ADD(&tmp_sum, &c_data[row * c_stride + j + 3], &tmp_mul);
                    LMMC_REAL_SET(&c_data[row * c_stride + j + 3], &tmp_sum);
                }
                for (; j < b_cols; ++j) {
                    LMMC_REAL_MUL(&tmp_mul, &val_p, &b_data[i * b_stride + j]);
                    LMMC_REAL_ADD(&tmp_sum, &c_data[row * c_stride + j], &tmp_mul);
                    LMMC_REAL_SET(&c_data[row * c_stride + j], &tmp_sum);
                }
                LMMC_REAL_CLEAR(&val_p);
            }
        }
    }

    LMMC_REAL_CLEAR(&zero);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_mat_mat_mul_sparse(const lmmc_sparse_mat_t* a, const lmmc_sparse_mat_t* b, lmmc_sparse_mat_t* c) {
    lmmc_sparse_mat_t a_csr = {0};
    lmmc_sparse_mat_t b_csr = {0};
    const lmmc_sparse_mat_t *pa = a;
    const lmmc_sparse_mat_t *pb = b;
    lmmc_status_t st = LMMC_STATUS_OK;
    size_t *marker = NULL;
    lmmc_real_t *accumulator = NULL;
    size_t *c_row_ptr = NULL;
    size_t *c_col_idx = NULL;
    lmmc_real_t *c_values = NULL;
    size_t nnz_est = 0;
    size_t i, j, k, p1, p2;

    if (a == NULL || b == NULL || c == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->cols != b->rows) return LMMC_STATUS_DIMENSION_MISMATCH;

    if (a->format == LMMC_SPARSE_CSC) {
        st = lmmc_sparse_to_csr(a, &a_csr);
        if (st != LMMC_STATUS_OK) return st;
        pa = &a_csr;
    }
    if (b->format == LMMC_SPARSE_CSC) {
        st = lmmc_sparse_to_csr(b, &b_csr);
        if (st != LMMC_STATUS_OK) {
            lmmc_sparse_destroy(&a_csr);
            return st;
        }
        pb = &b_csr;
    }

    marker = (size_t*)lmmc_alloc_array(pb->cols, sizeof(size_t));
    c_row_ptr = (size_t*)lmmc_alloc_array_plus(
        pa->rows, 1, sizeof(size_t));
    if (marker == NULL || c_row_ptr == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    memset(marker, 0xFF, pb->cols * sizeof(size_t));
    memset(c_row_ptr, 0, (pa->rows + 1) * sizeof(size_t));

    nnz_est = 0;
    for (i = 0; i < pa->rows; ++i) {
        for (p1 = pa->row_ptr[i]; p1 < pa->row_ptr[i + 1]; ++p1) {
            k = pa->col_idx[p1];
            for (p2 = pb->row_ptr[k]; p2 < pb->row_ptr[k + 1]; ++p2) {
                j = pb->col_idx[p2];
                if (marker[j] != i) {
                    marker[j] = i;
                    nnz_est++;
                }
            }
        }
        c_row_ptr[i + 1] = nnz_est;
    }

    c_col_idx = (size_t*)lmmc_alloc_array(nnz_est, sizeof(size_t));
    c_values = (lmmc_real_t*)lmmc_alloc_array(
        nnz_est, sizeof(lmmc_real_t));
    accumulator = (lmmc_real_t*)lmmc_alloc_array(
        pb->cols, sizeof(lmmc_real_t));
    if ((nnz_est > 0 && (c_col_idx == NULL || c_values == NULL)) || accumulator == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    memset(marker, 0xFF, pb->cols * sizeof(size_t));
    for (size_t act_i = 0; act_i < pb->cols; act_i++) {
        LMMC_REAL_INIT(&accumulator[act_i]);
        LMMC_REAL_SET_D(&accumulator[act_i], 0.0);
    }
    for (size_t act_i = 0; act_i < nnz_est; act_i++) {
        LMMC_REAL_INIT(&c_values[act_i]);
    }

    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);

    size_t current_nnz = 0;
    for (i = 0; i < pa->rows; ++i) {
        size_t row_start = current_nnz;
        for (p1 = pa->row_ptr[i]; p1 < pa->row_ptr[i + 1]; ++p1) {
            k = pa->col_idx[p1];
            for (p2 = pb->row_ptr[k]; p2 < pb->row_ptr[k + 1]; ++p2) {
                j = pb->col_idx[p2];
                if (marker[j] != i) {
                    marker[j] = i;
                    c_col_idx[current_nnz++] = j;
                }
                LMMC_REAL_MUL(&tmp_mul, &pa->values[p1], &pb->values[p2]);
                LMMC_REAL_ADD(&tmp_sum, &accumulator[j], &tmp_mul);
                LMMC_REAL_SET(&accumulator[j], &tmp_sum);
            }
        }
        for (p1 = row_start; p1 < current_nnz; ++p1) {
            j = c_col_idx[p1];
            LMMC_REAL_SET(&c_values[p1], &accumulator[j]);
            LMMC_REAL_SET_D(&accumulator[j], 0.0);
        }
    }

    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);

    c->rows = pa->rows;
    c->cols = pb->cols;
    c->nnz = nnz_est;
    c->row_ptr = c_row_ptr;
    c->col_idx = c_col_idx;
    c->values = c_values;
    c->format = LMMC_SPARSE_CSR;
    c->owns_data = 1;

    c_row_ptr = NULL;
    c_col_idx = NULL;
    c_values = NULL;

cleanup:
    if (marker) lmmc_free(marker);
    if (accumulator) {
        for (size_t act_i = 0; act_i < pb->cols; act_i++) {
            LMMC_REAL_CLEAR(&accumulator[act_i]);
        }
        lmmc_free(accumulator);
    }
    if (c_row_ptr) lmmc_free(c_row_ptr);
    if (c_col_idx) lmmc_free(c_col_idx);
    if (c_values) {
        for (size_t act_i = 0; act_i < nnz_est; act_i++) {
            LMMC_REAL_CLEAR(&c_values[act_i]);
        }
        lmmc_free(c_values);
    }
    lmmc_sparse_destroy(&a_csr);
    lmmc_sparse_destroy(&b_csr);
    return st;
}
