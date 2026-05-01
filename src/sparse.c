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

static lmmc_status_t lmmc_sparse_validate(const lmmc_sparse_mat_t* sparse) {
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

static lmmc_status_t lmmc_sparse_create(size_t rows, size_t cols, size_t nnz, lmmc_sparse_format_t format, lmmc_sparse_mat_t* out_sparse) {
    size_t outer_size = (format == LMMC_SPARSE_CSR) ? rows : cols;
    size_t outer_ptr_bytes = 0;
    size_t idx_bytes = 0;
    size_t val_bytes = 0;
    size_t* outer_ptr = NULL;
    size_t* inner_idx = NULL;
    double* values = NULL;

    if (out_sparse == NULL || rows == 0 || cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (lmmc_mul_overflow_size(outer_size + 1, sizeof(size_t), &outer_ptr_bytes) ||
        lmmc_mul_overflow_size(nnz, sizeof(size_t), &idx_bytes) ||
        lmmc_mul_overflow_size(nnz, sizeof(double), &val_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    outer_ptr = (size_t*)lmmc_alloc(outer_ptr_bytes);
    if (outer_ptr == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(outer_ptr, 0, outer_ptr_bytes);

    if (nnz > 0) {
        inner_idx = (size_t*)lmmc_alloc(idx_bytes);
        values = (double*)lmmc_alloc(val_bytes);
        if (inner_idx == NULL || values == NULL) {
            if (inner_idx != NULL) lmmc_free(inner_idx);
            if (values != NULL) lmmc_free(values);
            lmmc_free(outer_ptr);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        memset(inner_idx, 0, idx_bytes);
        memset(values, 0, val_bytes);
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
    double* values,
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
        if (sparse->values != NULL) lmmc_free(sparse->values);
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
    size_t i = 0, j = 0, nz = 0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (dense == NULL || out_sparse == NULL || dense->data == NULL || eps < 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < dense->rows; ++i) {
        for (j = 0; j < dense->cols; ++j) {
            if (fabs(dense->data[i * dense->stride + j]) > eps) ++nz;
        }
    }

    st = lmmc_sparse_create_csr(dense->rows, dense->cols, nz, out_sparse);
    if (st != LMMC_STATUS_OK) return st;

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
    size_t i = 0, p = 0;
    lmmc_status_t st = lmmc_sparse_validate(sparse);
    if (st != LMMC_STATUS_OK || out_dense == NULL || out_dense->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (out_dense->rows != sparse->rows || out_dense->cols != sparse->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    st = lmmc_mat_fill(out_dense, 0.0);
    if (st != LMMC_STATUS_OK) return st;

    if (sparse->format == LMMC_SPARSE_CSR) {
        for (i = 0; i < sparse->rows; ++i) {
            for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
                out_dense->data[i * out_dense->stride + sparse->col_idx[p]] = sparse->values[p];
            }
        }
    } else {
        for (i = 0; i < sparse->cols; ++i) {
            for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
                out_dense->data[sparse->col_idx[p] * out_dense->stride + i] = sparse->values[p];
            }
        }
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_mat_vec_mul(const lmmc_sparse_mat_t* sparse, const lmmc_vec_t* x, lmmc_vec_t* y) {
    size_t i = 0, p = 0;
    lmmc_status_t st = lmmc_sparse_validate(sparse);
    if (st != LMMC_STATUS_OK || x == NULL || y == NULL || x->data == NULL || y->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size != sparse->cols || y->size != sparse->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    if (sparse->format == LMMC_SPARSE_CSR) {
        for (i = 0; i < sparse->rows; ++i) {
            double sum = 0.0;
            for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
                sum += sparse->values[p] * x->data[sparse->col_idx[p]];
            }
            y->data[i] = sum;
        }
    } else {
        st = lmmc_vec_fill(y, 0.0);
        if (st != LMMC_STATUS_OK) return st;
        for (i = 0; i < sparse->cols; ++i) {
            double xv = x->data[i];
            for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
                y->data[sparse->col_idx[p]] += sparse->values[p] * xv;
            }
        }
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_transpose(const lmmc_sparse_mat_t* sparse, lmmc_sparse_mat_t* out_transposed) {
    size_t i = 0, p = 0;
    size_t* next = NULL;
    lmmc_status_t st = lmmc_sparse_validate(sparse);

    if (st != LMMC_STATUS_OK || out_transposed == NULL) return LMMC_STATUS_INVALID_ARGUMENT;

    st = lmmc_sparse_create(sparse->cols, sparse->rows, sparse->nnz, sparse->format, out_transposed);
    if (st != LMMC_STATUS_OK) return st;

    size_t outer_size_src = (sparse->format == LMMC_SPARSE_CSR) ? sparse->rows : sparse->cols;
    size_t inner_size_src = (sparse->format == LMMC_SPARSE_CSR) ? sparse->cols : sparse->rows;

    for (p = 0; p < sparse->nnz; ++p) {
        ++out_transposed->row_ptr[sparse->col_idx[p] + 1];
    }

    for (i = 0; i < inner_size_src; ++i) {
        out_transposed->row_ptr[i + 1] += out_transposed->row_ptr[i];
    }

    if (out_transposed->nnz == 0) return LMMC_STATUS_OK;

    next = (size_t*)lmmc_alloc(inner_size_src * sizeof(size_t));
    if (next == NULL) {
        lmmc_sparse_destroy(out_transposed);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(next, out_transposed->row_ptr, inner_size_src * sizeof(size_t));

    for (i = 0; i < outer_size_src; ++i) {
        for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
            size_t inner = sparse->col_idx[p];
            size_t dst = next[inner]++;
            out_transposed->col_idx[dst] = i;
            out_transposed->values[dst] = sparse->values[p];
        }
    }

    lmmc_free(next);
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

    st = lmmc_mat_fill(c, 0.0);
    if (st != LMMC_STATUS_OK) return st;

    if (sparse->format == LMMC_SPARSE_CSR) {
        for (i = 0; i < sparse->rows; ++i) {
            for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
                size_t k = sparse->col_idx[p];
                double av = sparse->values[p];
                for (j = 0; j < b->cols; ++j) {
                    c->data[i * c->stride + j] += av * b->data[k * b->stride + j];
                }
            }
        }
    } else {
        for (i = 0; i < sparse->cols; ++i) {
            for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
                size_t row = sparse->col_idx[p];
                double av = sparse->values[p];
                for (j = 0; j < b->cols; ++j) {
                    c->data[row * c->stride + j] += av * b->data[i * b->stride + j];
                }
            }
        }
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_mat_mat_mul_sparse(const lmmc_sparse_mat_t* a, const lmmc_sparse_mat_t* b, lmmc_sparse_mat_t* c) {
    lmmc_sparse_mat_t a_csr = {0};
    lmmc_sparse_mat_t b_csr = {0};
    const lmmc_sparse_mat_t *pa = a;
    const lmmc_sparse_mat_t *pb = b;
    lmmc_status_t st = LMMC_STATUS_OK;
    size_t *marker = NULL;
    double *accumulator = NULL;
    size_t *c_row_ptr = NULL;
    size_t *c_col_idx = NULL;
    double *c_values = NULL;
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

    marker = (size_t*)lmmc_alloc(pb->cols * sizeof(size_t));
    c_row_ptr = (size_t*)lmmc_alloc((pa->rows + 1) * sizeof(size_t));
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

    c_col_idx = (size_t*)lmmc_alloc(nnz_est * sizeof(size_t));
    c_values = (double*)lmmc_alloc(nnz_est * sizeof(double));
    accumulator = (double*)lmmc_alloc(pb->cols * sizeof(double));
    if ((nnz_est > 0 && (c_col_idx == NULL || c_values == NULL)) || accumulator == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    memset(marker, 0xFF, pb->cols * sizeof(size_t));
    memset(accumulator, 0, pb->cols * sizeof(double));

    size_t current_nnz = 0;
    for (i = 0; i < pa->rows; ++i) {
        size_t row_start = current_nnz;
        for (p1 = pa->row_ptr[i]; p1 < pa->row_ptr[i + 1]; ++p1) {
            k = pa->col_idx[p1];
            double val_a = pa->values[p1];
            for (p2 = pb->row_ptr[k]; p2 < pb->row_ptr[k + 1]; ++p2) {
                j = pb->col_idx[p2];
                if (marker[j] != i) {
                    marker[j] = i;
                    c_col_idx[current_nnz++] = j;
                }
                accumulator[j] += val_a * pb->values[p2];
            }
        }
        for (p1 = row_start; p1 < current_nnz; ++p1) {
            j = c_col_idx[p1];
            c_values[p1] = accumulator[j];
            accumulator[j] = 0.0;
        }
    }

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
    if (accumulator) lmmc_free(accumulator);
    if (c_row_ptr) lmmc_free(c_row_ptr);
    if (c_col_idx) lmmc_free(c_col_idx);
    if (c_values) lmmc_free(c_values);
    lmmc_sparse_destroy(&a_csr);
    lmmc_sparse_destroy(&b_csr);
    return st;
}

lmmc_status_t lmmc_sparse_to_csc(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst) {
    if (src == NULL || dst == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (src->format == LMMC_SPARSE_CSC) {
        lmmc_status_t st = lmmc_sparse_create_csc(src->rows, src->cols, src->nnz, dst);
        if (st != LMMC_STATUS_OK) return st;
        memcpy(dst->row_ptr, src->row_ptr, (src->cols + 1) * sizeof(size_t));
        if (src->nnz > 0) {
            memcpy(dst->col_idx, src->col_idx, src->nnz * sizeof(size_t));
            memcpy(dst->values, src->values, src->nnz * sizeof(double));
        }
        return LMMC_STATUS_OK;
    }
    
    lmmc_status_t st = lmmc_sparse_create_csc(src->rows, src->cols, src->nnz, dst);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t p = 0; p < src->nnz; ++p) {
        ++dst->row_ptr[src->col_idx[p] + 1];
    }
    for (size_t i = 0; i < src->cols; ++i) {
        dst->row_ptr[i + 1] += dst->row_ptr[i];
    }
    
    size_t* next = (size_t*)lmmc_alloc(src->cols * sizeof(size_t));
    if (next == NULL) {
        lmmc_sparse_destroy(dst);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(next, dst->row_ptr, src->cols * sizeof(size_t));

    for (size_t i = 0; i < src->rows; ++i) {
        for (size_t p = src->row_ptr[i]; p < src->row_ptr[i + 1]; ++p) {
            size_t col = src->col_idx[p];
            size_t dest_idx = next[col]++;
            dst->col_idx[dest_idx] = i;
            dst->values[dest_idx] = src->values[p];
        }
    }
    lmmc_free(next);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_to_csr(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst) {
    if (src == NULL || dst == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (src->format == LMMC_SPARSE_CSR) {
        lmmc_status_t st = lmmc_sparse_create_csr(src->rows, src->cols, src->nnz, dst);
        if (st != LMMC_STATUS_OK) return st;
        memcpy(dst->row_ptr, src->row_ptr, (src->rows + 1) * sizeof(size_t));
        if (src->nnz > 0) {
            memcpy(dst->col_idx, src->col_idx, src->nnz * sizeof(size_t));
            memcpy(dst->values, src->values, src->nnz * sizeof(double));
        }
        return LMMC_STATUS_OK;
    }
    
    lmmc_status_t st = lmmc_sparse_create_csr(src->rows, src->cols, src->nnz, dst);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t p = 0; p < src->nnz; ++p) {
        ++dst->row_ptr[src->col_idx[p] + 1];
    }
    for (size_t i = 0; i < src->rows; ++i) {
        dst->row_ptr[i + 1] += dst->row_ptr[i];
    }
    
    size_t* next = (size_t*)lmmc_alloc(src->rows * sizeof(size_t));
    if (next == NULL) {
        lmmc_sparse_destroy(dst);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(next, dst->row_ptr, src->rows * sizeof(size_t));

    for (size_t i = 0; i < src->cols; ++i) {
        for (size_t p = src->row_ptr[i]; p < src->row_ptr[i + 1]; ++p) {
            size_t row = src->col_idx[p];
            size_t dest_idx = next[row]++;
            dst->col_idx[dest_idx] = i;
            dst->values[dest_idx] = src->values[p];
        }
    }
    lmmc_free(next);
    return LMMC_STATUS_OK;
}
