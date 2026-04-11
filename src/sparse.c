#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "lmmc/sparse.h"

static int lmmc_mul_overflow_size(size_t a, size_t b, size_t* out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }
    if (a > ((size_t)-1) / b) {
        return 1;
    }
    *out = a * b;
    return 0;
}

static lmmc_status_t lmmc_sparse_validate_csr(const lmmc_sparse_mat_t* sparse) {
    size_t i = 0;
    if (sparse == NULL || sparse->row_ptr == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (sparse->rows == 0 || sparse->cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (sparse->nnz > 0 && (sparse->col_idx == NULL || sparse->values == NULL)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (sparse->row_ptr[0] != 0 || sparse->row_ptr[sparse->rows] != sparse->nnz) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0; i < sparse->rows; ++i) {
        if (sparse->row_ptr[i] > sparse->row_ptr[i + 1]) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    for (i = 0; i < sparse->nnz; ++i) {
        if (sparse->col_idx[i] >= sparse->cols) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_create_csr(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse) {
    size_t row_ptr_bytes = 0;
    size_t idx_bytes = 0;
    size_t val_bytes = 0;
    size_t* row_ptr = NULL;
    size_t* col_idx = NULL;
    double* values = NULL;

    if (out_sparse == NULL || rows == 0 || cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_mul_overflow_size(rows + 1, sizeof(size_t), &row_ptr_bytes) ||
        lmmc_mul_overflow_size(nnz, sizeof(size_t), &idx_bytes) ||
        lmmc_mul_overflow_size(nnz, sizeof(double), &val_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    row_ptr = (size_t*)lmmc_alloc(row_ptr_bytes);
    if (row_ptr == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(row_ptr, 0, row_ptr_bytes);

    if (nnz > 0) {
        col_idx = (size_t*)lmmc_alloc(idx_bytes);
        values = (double*)lmmc_alloc(val_bytes);
        if (col_idx == NULL || values == NULL) {
            if (col_idx != NULL) {
                lmmc_free(col_idx);
            }
            if (values != NULL) {
                lmmc_free(values);
            }
            lmmc_free(row_ptr);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        memset(col_idx, 0, idx_bytes);
        memset(values, 0, val_bytes);
    }

    out_sparse->rows = rows;
    out_sparse->cols = cols;
    out_sparse->nnz = nnz;
    out_sparse->row_ptr = row_ptr;
    out_sparse->col_idx = col_idx;
    out_sparse->values = values;
    out_sparse->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_wrap_csr(
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t* row_ptr,
    size_t* col_idx,
    double* values,
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
    candidate.owns_data = 0;

    if (lmmc_sparse_validate_csr(&candidate) != LMMC_STATUS_OK) {
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
        if (sparse->row_ptr != NULL) {
            lmmc_free(sparse->row_ptr);
        }
        if (sparse->col_idx != NULL) {
            lmmc_free(sparse->col_idx);
        }
        if (sparse->values != NULL) {
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

lmmc_status_t lmmc_sparse_from_dense(const lmmc_mat_t* dense, double eps, lmmc_sparse_mat_t* out_sparse) {
    size_t i = 0;
    size_t j = 0;
    size_t nz = 0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (dense == NULL || out_sparse == NULL || dense->data == NULL || eps < 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < dense->rows; ++i) {
        for (j = 0; j < dense->cols; ++j) {
            double v = dense->data[i * dense->stride + j];
            if (fabs(v) > eps) {
                ++nz;
            }
        }
    }

    st = lmmc_sparse_create_csr(dense->rows, dense->cols, nz, out_sparse);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    nz = 0;
    out_sparse->row_ptr[0] = 0;
    for (i = 0; i < dense->rows; ++i) {
        for (j = 0; j < dense->cols; ++j) {
            double v = dense->data[i * dense->stride + j];
            if (fabs(v) > eps) {
                out_sparse->col_idx[nz] = j;
                out_sparse->values[nz] = v;
                ++nz;
            }
        }
        out_sparse->row_ptr[i + 1] = nz;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_to_dense(const lmmc_sparse_mat_t* sparse, lmmc_mat_t* out_dense) {
    size_t i = 0;
    lmmc_status_t st = lmmc_sparse_validate_csr(sparse);
    if (st != LMMC_STATUS_OK || out_dense == NULL || out_dense->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (out_dense->rows != sparse->rows || out_dense->cols != sparse->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    st = lmmc_mat_fill(out_dense, 0.0);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    for (i = 0; i < sparse->rows; ++i) {
        size_t p = 0;
        for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
            size_t col = sparse->col_idx[p];
            out_dense->data[i * out_dense->stride + col] = sparse->values[p];
        }
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_mat_vec_mul(const lmmc_sparse_mat_t* sparse, const lmmc_vec_t* x, lmmc_vec_t* y) {
    size_t i = 0;
    lmmc_status_t st = lmmc_sparse_validate_csr(sparse);
    if (st != LMMC_STATUS_OK || x == NULL || y == NULL || x->data == NULL || y->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size != sparse->cols || y->size != sparse->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    for (i = 0; i < sparse->rows; ++i) {
        size_t p = 0;
        double sum = 0.0;
        for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
            sum += sparse->values[p] * x->data[sparse->col_idx[p]];
        }
        y->data[i] = sum;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_transpose(const lmmc_sparse_mat_t* sparse, lmmc_sparse_mat_t* out_transposed) {
    size_t i = 0;
    size_t p = 0;
    size_t bytes = 0;
    size_t* next = NULL;
    lmmc_status_t st = lmmc_sparse_validate_csr(sparse);

    if (st != LMMC_STATUS_OK || out_transposed == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_sparse_create_csr(sparse->cols, sparse->rows, sparse->nnz, out_transposed);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    for (p = 0; p < sparse->nnz; ++p) {
        size_t col = sparse->col_idx[p];
        ++out_transposed->row_ptr[col + 1];
    }

    for (i = 0; i < out_transposed->rows; ++i) {
        out_transposed->row_ptr[i + 1] += out_transposed->row_ptr[i];
    }

    if (out_transposed->nnz == 0) {
        return LMMC_STATUS_OK;
    }

    if (lmmc_mul_overflow_size(out_transposed->rows, sizeof(size_t), &bytes)) {
        lmmc_sparse_destroy(out_transposed);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    next = (size_t*)lmmc_alloc(bytes);
    if (next == NULL) {
        lmmc_sparse_destroy(out_transposed);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    memcpy(next, out_transposed->row_ptr, bytes);

    for (i = 0; i < sparse->rows; ++i) {
        for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
            size_t col = sparse->col_idx[p];
            size_t dst = next[col]++;
            out_transposed->col_idx[dst] = i;
            out_transposed->values[dst] = sparse->values[p];
        }
    }

    lmmc_free(next);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_mat_mat_mul_dense(const lmmc_sparse_mat_t* sparse, const lmmc_mat_t* b, lmmc_mat_t* c) {
    size_t i = 0;
    lmmc_status_t st = lmmc_sparse_validate_csr(sparse);

    if (st != LMMC_STATUS_OK || b == NULL || c == NULL || b->data == NULL || c->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (b->rows == 0 || b->cols == 0 || c->rows == 0 || c->cols == 0 || b->stride < b->cols || c->stride < c->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (sparse->cols != b->rows || c->rows != sparse->rows || c->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    st = lmmc_mat_fill(c, 0.0);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    for (i = 0; i < sparse->rows; ++i) {
        size_t p = 0;
        for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
            size_t k = sparse->col_idx[p];
            size_t j = 0;
            double av = sparse->values[p];
            for (j = 0; j < b->cols; ++j) {
                c->data[i * c->stride + j] += av * b->data[k * b->stride + j];
            }
        }
    }

    return LMMC_STATUS_OK;
}
