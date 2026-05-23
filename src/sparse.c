#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
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
    lmmc_real_t* values = NULL;

    if (out_sparse == NULL || rows == 0 || cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (lmmc_mul_overflow_size(outer_size + 1, sizeof(size_t), &outer_ptr_bytes) ||
        lmmc_mul_overflow_size(nnz, sizeof(size_t), &idx_bytes) ||
        lmmc_mul_overflow_size(nnz, sizeof(lmmc_real_t), &val_bytes)) {
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

lmmc_status_t lmmc_sparse_from_dense(const lmmc_mat_t* dense, lmmc_real_t eps, lmmc_sparse_mat_t* out_sparse) {
    size_t i = 0, j = 0, nz = 0;
    lmmc_status_t st = LMMC_STATUS_OK;
    if (dense == NULL || out_sparse == NULL || dense->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t zero; LMMC_REAL_INIT(&zero);
    lmmc_real_t abs_v; LMMC_REAL_INIT(&abs_v);
    LMMC_REAL_SET_D(&zero, 0.0);

    if (LMMC_REAL_CMP(&eps, &zero) < 0) {
        LMMC_REAL_CLEAR(&zero);
        LMMC_REAL_CLEAR(&abs_v);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < dense->rows; ++i) {
        for (j = 0; j < dense->cols; ++j) {
            LMMC_REAL_ABS(&abs_v, &dense->data[i * dense->stride + j]);
            if (LMMC_REAL_CMP(&abs_v, &eps) > 0) ++nz;
        }
    }

    st = lmmc_sparse_create_csr(dense->rows, dense->cols, nz, out_sparse);
    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&zero);
        LMMC_REAL_CLEAR(&abs_v);
        return st;
    }

    nz = 0;
    out_sparse->row_ptr[0] = 0;
    for (i = 0; i < dense->rows; ++i) {
        for (j = 0; j < dense->cols; ++j) {
            lmmc_real_t* v_ptr = &dense->data[i * dense->stride + j];
            LMMC_REAL_ABS(&abs_v, v_ptr);
            if (LMMC_REAL_CMP(&abs_v, &eps) > 0) {
                out_sparse->col_idx[nz] = j;
                LMMC_REAL_SET(&out_sparse->values[nz], v_ptr);
                ++nz;
            }
        }
        out_sparse->row_ptr[i + 1] = nz;
    }

    LMMC_REAL_CLEAR(&zero);
    LMMC_REAL_CLEAR(&abs_v);
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

    lmmc_real_t zero; LMMC_REAL_INIT(&zero);
    LMMC_REAL_SET_D(&zero, 0.0);
    st = lmmc_mat_fill(out_dense, zero);
    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&zero);
        return st;
    }

    if (sparse->format == LMMC_SPARSE_CSR) {
        for (i = 0; i < sparse->rows; ++i) {
            for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
                LMMC_REAL_SET(&out_dense->data[i * out_dense->stride + sparse->col_idx[p]], &sparse->values[p]);
            }
        }
    } else {
        for (i = 0; i < sparse->cols; ++i) {
            for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
                LMMC_REAL_SET(&out_dense->data[sparse->col_idx[p] * out_dense->stride + i], &sparse->values[p]);
            }
        }
    }

    LMMC_REAL_CLEAR(&zero);
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
            LMMC_REAL_SET(&out_transposed->values[dst], &sparse->values[p]);
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
    c_values = (lmmc_real_t*)lmmc_alloc(nnz_est * sizeof(lmmc_real_t));
    accumulator = (lmmc_real_t*)lmmc_alloc(pb->cols * sizeof(lmmc_real_t));
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

lmmc_status_t lmmc_sparse_to_csc(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst) {
    if (src == NULL || dst == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    lmmc_status_t st = lmmc_sparse_validate(src);
    if (st != LMMC_STATUS_OK) return st;

    if (src->format == LMMC_SPARSE_CSC) {
        st = lmmc_sparse_create_csc(src->rows, src->cols, src->nnz, dst);
        if (st != LMMC_STATUS_OK) return st;
        memcpy(dst->row_ptr, src->row_ptr, (src->cols + 1) * sizeof(size_t));
        if (src->nnz > 0) {
            memcpy(dst->col_idx, src->col_idx, src->nnz * sizeof(size_t));
            for (size_t k = 0; k < src->nnz; ++k) {
                LMMC_REAL_SET(&dst->values[k], &src->values[k]);
            }
        }
        return LMMC_STATUS_OK;
    }

    st = lmmc_sparse_create_csc(src->rows, src->cols, src->nnz, dst);
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
            LMMC_REAL_SET(&dst->values[dest_idx], &src->values[p]);
        }
    }
    lmmc_free(next);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_to_csr(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst) {
    if (src == NULL || dst == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    lmmc_status_t st = lmmc_sparse_validate(src);
    if (st != LMMC_STATUS_OK) return st;

    if (src->format == LMMC_SPARSE_CSR) {
        st = lmmc_sparse_create_csr(src->rows, src->cols, src->nnz, dst);
        if (st != LMMC_STATUS_OK) return st;
        memcpy(dst->row_ptr, src->row_ptr, (src->rows + 1) * sizeof(size_t));
        if (src->nnz > 0) {
            memcpy(dst->col_idx, src->col_idx, src->nnz * sizeof(size_t));
            for (size_t k = 0; k < src->nnz; ++k) {
                LMMC_REAL_SET(&dst->values[k], &src->values[k]);
            }
        }
        return LMMC_STATUS_OK;
    }

    st = lmmc_sparse_create_csr(src->rows, src->cols, src->nnz, dst);
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
            LMMC_REAL_SET(&dst->values[dest_idx], &src->values[p]);
        }
    }
    lmmc_free(next);
    return LMMC_STATUS_OK;
}

/* --- Sparse Builder Implementation --- */

struct lmmc_sparse_builder_t {
    size_t rows;
    size_t cols;
    size_t nnz;
    size_t capacity;
    size_t* r_idx;
    size_t* c_idx;
    lmmc_real_t* vals;
};

lmmc_status_t lmmc_sparse_builder_create(size_t rows, size_t cols, size_t initial_capacity, lmmc_sparse_builder_t** out_builder) {
    lmmc_sparse_builder_t* b = NULL;
    if (out_builder == NULL || rows == 0 || cols == 0) return LMMC_STATUS_INVALID_ARGUMENT;

    b = (lmmc_sparse_builder_t*)lmmc_alloc(sizeof(lmmc_sparse_builder_t));
    if (b == NULL) return LMMC_STATUS_ALLOCATION_FAILED;

    b->rows = rows;
    b->cols = cols;
    b->nnz = 0;
    b->capacity = initial_capacity > 0 ? initial_capacity : 16;
    
    size_t sz_idx = 0;
    size_t sz_vals = 0;
    if (lmmc_mul_overflow_size(b->capacity, sizeof(size_t), &sz_idx) ||
        lmmc_mul_overflow_size(b->capacity, sizeof(lmmc_real_t), &sz_vals)) {
        lmmc_free(b);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    b->r_idx = (size_t*)lmmc_alloc(sz_idx);
    b->c_idx = (size_t*)lmmc_alloc(sz_idx);
    b->vals = (lmmc_real_t*)lmmc_alloc(sz_vals);

    if (b->r_idx == NULL || b->c_idx == NULL || b->vals == NULL) {
        lmmc_sparse_builder_destroy(b);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (size_t k = 0; k < b->capacity; ++k) {
        LMMC_REAL_INIT(&b->vals[k]);
    }

    *out_builder = b;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_builder_add(lmmc_sparse_builder_t* b, size_t row, size_t col, lmmc_real_t val) {
    if (b == NULL || row >= b->rows || col >= b->cols) return LMMC_STATUS_INVALID_ARGUMENT;

    if (b->nnz >= b->capacity) {
        size_t new_cap = 0;
        size_t sz_idx = 0;
        size_t sz_vals = 0;
        size_t* nr = NULL;
        size_t* nc = NULL;
        lmmc_real_t* nv = NULL;

        if (lmmc_mul_overflow_size(b->capacity, 2, &new_cap) ||
            lmmc_mul_overflow_size(new_cap, sizeof(size_t), &sz_idx) ||
            lmmc_mul_overflow_size(new_cap, sizeof(lmmc_real_t), &sz_vals)) {
            return LMMC_STATUS_ALLOCATION_FAILED;
        }

        nr = (size_t*)lmmc_alloc(sz_idx);
        nc = (size_t*)lmmc_alloc(sz_idx);
        nv = (lmmc_real_t*)lmmc_alloc(sz_vals);

        if (nr == NULL || nc == NULL || nv == NULL) {
            if (nr) lmmc_free(nr);
            if (nc) lmmc_free(nc);
            if (nv) lmmc_free(nv);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }

        for (size_t k = 0; k < new_cap; ++k) {
            LMMC_REAL_INIT(&nv[k]);
        }

        if (b->nnz > 0) {
            memcpy(nr, b->r_idx, b->nnz * sizeof(size_t));
            memcpy(nc, b->c_idx, b->nnz * sizeof(size_t));
            for (size_t k = 0; k < b->nnz; ++k) {
                LMMC_REAL_SET(&nv[k], &b->vals[k]);
            }
        }

        for (size_t k = 0; k < b->capacity; ++k) {
            LMMC_REAL_CLEAR(&b->vals[k]);
        }
        lmmc_free(b->r_idx);
        lmmc_free(b->c_idx);
        lmmc_free(b->vals);

        b->r_idx = nr;
        b->c_idx = nc;
        b->vals = nv;
        b->capacity = new_cap;
    }

    b->r_idx[b->nnz] = row;
    b->c_idx[b->nnz] = col;
    LMMC_REAL_SET(&b->vals[b->nnz], &val);
    b->nnz++;
    return LMMC_STATUS_OK;
}

void lmmc_sparse_builder_destroy(lmmc_sparse_builder_t* b) {
    if (b) {
        if (b->r_idx) lmmc_free(b->r_idx);
        if (b->c_idx) lmmc_free(b->c_idx);
        if (b->vals) {
            for (size_t k = 0; k < b->capacity; ++k) {
                LMMC_REAL_CLEAR(&b->vals[k]);
            }
            lmmc_free(b->vals);
        }
        lmmc_free(b);
    }
}

lmmc_status_t lmmc_sparse_builder_build(lmmc_sparse_builder_t* b, lmmc_sparse_format_t format, lmmc_sparse_mat_t* out) {
    lmmc_status_t st;
    size_t i, outer_dim;
    if (b == NULL || out == NULL) return LMMC_STATUS_INVALID_ARGUMENT;

    st = (format == LMMC_SPARSE_CSR) ? lmmc_sparse_create_csr(b->rows, b->cols, b->nnz, out) 
                                    : lmmc_sparse_create_csc(b->rows, b->cols, b->nnz, out);
    if (st != LMMC_STATUS_OK) return st;

    outer_dim = (format == LMMC_SPARSE_CSR) ? b->rows : b->cols;
    size_t* major = (format == LMMC_SPARSE_CSR) ? b->r_idx : b->c_idx;
    size_t* minor = (format == LMMC_SPARSE_CSR) ? b->c_idx : b->r_idx;

    /* Count NNZ per row/column */
    for (i = 0; i < b->nnz; ++i) {
        out->row_ptr[major[i] + 1]++;
    }
    /* Prefix sum */
    for (i = 0; i < outer_dim; ++i) {
        out->row_ptr[i+1] += out->row_ptr[i];
    }

    /* Temp pointer for filling */
    size_t* next = (size_t*)lmmc_alloc(outer_dim * sizeof(size_t));
    if (next == NULL) {
        lmmc_sparse_destroy(out);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(next, out->row_ptr, outer_dim * sizeof(size_t));

    for (i = 0; i < b->nnz; ++i) {
        size_t idx = next[major[i]]++;
        out->col_idx[idx] = minor[i];
        LMMC_REAL_SET(&out->values[idx], &b->vals[i]);
    }

    lmmc_free(next);
    return LMMC_STATUS_OK;
}

/* === COO 格式操作 === */

lmmc_status_t lmmc_sparse_coo_create(
    size_t rows, size_t cols, size_t capacity,
    lmmc_sparse_coo_t* out_coo
) {
    size_t sz_idx = 0;
    size_t sz_vals = 0;
    size_t actual_cap;

    if (out_coo == NULL || rows == 0 || cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    actual_cap = capacity > 0 ? capacity : 16;

    if (lmmc_mul_overflow_size(actual_cap, sizeof(size_t), &sz_idx) ||
        lmmc_mul_overflow_size(actual_cap, sizeof(lmmc_real_t), &sz_vals)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    out_coo->row_idx = (size_t*)lmmc_alloc(sz_idx);
    out_coo->col_idx = (size_t*)lmmc_alloc(sz_idx);
    out_coo->values = (lmmc_real_t*)lmmc_alloc(sz_vals);

    if (out_coo->row_idx == NULL || out_coo->col_idx == NULL || out_coo->values == NULL) {
        if (out_coo->row_idx) lmmc_free(out_coo->row_idx);
        if (out_coo->col_idx) lmmc_free(out_coo->col_idx);
        if (out_coo->values) lmmc_free(out_coo->values);
        out_coo->row_idx = NULL;
        out_coo->col_idx = NULL;
        out_coo->values = NULL;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (size_t k = 0; k < actual_cap; ++k) {
        LMMC_REAL_INIT(&out_coo->values[k]);
    }

    out_coo->rows = rows;
    out_coo->cols = cols;
    out_coo->nnz = 0;
    out_coo->capacity = actual_cap;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_coo_add_entry(
    lmmc_sparse_coo_t* coo,
    size_t row, size_t col, lmmc_real_t value
) {
    if (coo == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (row >= coo->rows || col >= coo->cols) {
        return LMMC_STATUS_INDEX_OUT_OF_BOUNDS;
    }

    /* Auto-expand capacity with 2x strategy */
    if (coo->nnz >= coo->capacity) {
        size_t new_cap = 0;
        size_t sz_idx = 0;
        size_t sz_vals = 0;
        size_t* nr = NULL;
        size_t* nc = NULL;
        lmmc_real_t* nv = NULL;

        if (lmmc_mul_overflow_size(coo->capacity, 2, &new_cap) ||
            lmmc_mul_overflow_size(new_cap, sizeof(size_t), &sz_idx) ||
            lmmc_mul_overflow_size(new_cap, sizeof(lmmc_real_t), &sz_vals)) {
            return LMMC_STATUS_ALLOCATION_FAILED;
        }

        nr = (size_t*)lmmc_alloc(sz_idx);
        nc = (size_t*)lmmc_alloc(sz_idx);
        nv = (lmmc_real_t*)lmmc_alloc(sz_vals);

        if (nr == NULL || nc == NULL || nv == NULL) {
            if (nr) lmmc_free(nr);
            if (nc) lmmc_free(nc);
            if (nv) lmmc_free(nv);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }

        for (size_t k = 0; k < new_cap; ++k) {
            LMMC_REAL_INIT(&nv[k]);
        }

        if (coo->nnz > 0) {
            memcpy(nr, coo->row_idx, coo->nnz * sizeof(size_t));
            memcpy(nc, coo->col_idx, coo->nnz * sizeof(size_t));
            for (size_t k = 0; k < coo->nnz; ++k) {
                LMMC_REAL_SET(&nv[k], &coo->values[k]);
            }
        }

        for (size_t k = 0; k < coo->capacity; ++k) {
            LMMC_REAL_CLEAR(&coo->values[k]);
        }
        lmmc_free(coo->row_idx);
        lmmc_free(coo->col_idx);
        lmmc_free(coo->values);

        coo->row_idx = nr;
        coo->col_idx = nc;
        coo->values = nv;
        coo->capacity = new_cap;
    }

    coo->row_idx[coo->nnz] = row;
    coo->col_idx[coo->nnz] = col;
    LMMC_REAL_SET(&coo->values[coo->nnz], &value);
    coo->nnz++;
    return LMMC_STATUS_OK;
}

/* Helper: comparison function for sorting COO entries by (row, col) for CSR */
typedef struct {
    size_t row;
    size_t col;
    size_t orig_idx;
} lmmc_coo_sort_entry_t;

static int lmmc_coo_cmp_row_col(const void* a, const void* b) {
    const lmmc_coo_sort_entry_t* ea = (const lmmc_coo_sort_entry_t*)a;
    const lmmc_coo_sort_entry_t* eb = (const lmmc_coo_sort_entry_t*)b;
    if (ea->row != eb->row) return (ea->row < eb->row) ? -1 : 1;
    if (ea->col != eb->col) return (ea->col < eb->col) ? -1 : 1;
    return 0;
}

/* Helper: comparison function for sorting COO entries by (col, row) for CSC */
static int lmmc_coo_cmp_col_row(const void* a, const void* b) {
    const lmmc_coo_sort_entry_t* ea = (const lmmc_coo_sort_entry_t*)a;
    const lmmc_coo_sort_entry_t* eb = (const lmmc_coo_sort_entry_t*)b;
    if (ea->col != eb->col) return (ea->col < eb->col) ? -1 : 1;
    if (ea->row != eb->row) return (ea->row < eb->row) ? -1 : 1;
    return 0;
}

lmmc_status_t lmmc_sparse_coo_to_csr(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csr
) {
    lmmc_coo_sort_entry_t* entries = NULL;
    size_t unique_nnz = 0;
    size_t i;
    lmmc_status_t st;

    if (coo == NULL || out_csr == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Handle empty matrix */
    if (coo->nnz == 0) {
        return lmmc_sparse_create_csr(coo->rows, coo->cols, 0, out_csr);
    }

    /* Allocate sort entries */
    entries = (lmmc_coo_sort_entry_t*)lmmc_alloc(coo->nnz * sizeof(lmmc_coo_sort_entry_t));
    if (entries == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (i = 0; i < coo->nnz; ++i) {
        entries[i].row = coo->row_idx[i];
        entries[i].col = coo->col_idx[i];
        entries[i].orig_idx = i;
    }

    /* Sort by (row, col) */
    qsort(entries, coo->nnz, sizeof(lmmc_coo_sort_entry_t), lmmc_coo_cmp_row_col);

    /* Count unique entries (merge duplicates) */
    unique_nnz = 1;
    for (i = 1; i < coo->nnz; ++i) {
        if (entries[i].row != entries[i - 1].row || entries[i].col != entries[i - 1].col) {
            unique_nnz++;
        }
    }

    /* Create CSR matrix */
    st = lmmc_sparse_create_csr(coo->rows, coo->cols, unique_nnz, out_csr);
    if (st != LMMC_STATUS_OK) {
        lmmc_free(entries);
        return st;
    }

    /* Fill CSR data: merge duplicates by summing values, build row_ptr simultaneously */
    {
        size_t csr_idx = 0;
        lmmc_real_t sum; LMMC_REAL_INIT(&sum);
        lmmc_real_t tmp; LMMC_REAL_INIT(&tmp);

        /* Initialize row_ptr to zero */
        memset(out_csr->row_ptr, 0, (coo->rows + 1) * sizeof(size_t));

        /* First pass: count unique entries per row */
        {
            size_t prev_row = entries[0].row;
            size_t prev_col = entries[0].col;
            out_csr->row_ptr[prev_row + 1]++;
            for (i = 1; i < coo->nnz; ++i) {
                if (entries[i].row != prev_row || entries[i].col != prev_col) {
                    out_csr->row_ptr[entries[i].row + 1]++;
                    prev_row = entries[i].row;
                    prev_col = entries[i].col;
                }
            }
        }

        /* Prefix sum to build row_ptr */
        for (i = 0; i < coo->rows; ++i) {
            out_csr->row_ptr[i + 1] += out_csr->row_ptr[i];
        }

        /* Second pass: fill col_idx and values with merged duplicates */
        LMMC_REAL_SET(&sum, &coo->values[entries[0].orig_idx]);
        {
            size_t cur_row = entries[0].row;
            size_t cur_col = entries[0].col;

            for (i = 1; i < coo->nnz; ++i) {
                if (entries[i].row == cur_row && entries[i].col == cur_col) {
                    /* Duplicate: sum values */
                    LMMC_REAL_ADD(&tmp, &sum, &coo->values[entries[i].orig_idx]);
                    LMMC_REAL_SET(&sum, &tmp);
                } else {
                    /* Store previous entry */
                    out_csr->col_idx[csr_idx] = cur_col;
                    LMMC_REAL_SET(&out_csr->values[csr_idx], &sum);
                    csr_idx++;

                    /* Start new entry */
                    cur_row = entries[i].row;
                    cur_col = entries[i].col;
                    LMMC_REAL_SET(&sum, &coo->values[entries[i].orig_idx]);
                }
            }
            /* Store last entry */
            out_csr->col_idx[csr_idx] = cur_col;
            LMMC_REAL_SET(&out_csr->values[csr_idx], &sum);
        }

        LMMC_REAL_CLEAR(&sum);
        LMMC_REAL_CLEAR(&tmp);
    }

    lmmc_free(entries);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_coo_to_csc(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csc
) {
    lmmc_coo_sort_entry_t* entries = NULL;
    size_t unique_nnz = 0;
    size_t i;
    lmmc_status_t st;

    if (coo == NULL || out_csc == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Handle empty matrix */
    if (coo->nnz == 0) {
        return lmmc_sparse_create_csc(coo->rows, coo->cols, 0, out_csc);
    }

    /* Allocate sort entries */
    entries = (lmmc_coo_sort_entry_t*)lmmc_alloc(coo->nnz * sizeof(lmmc_coo_sort_entry_t));
    if (entries == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (i = 0; i < coo->nnz; ++i) {
        entries[i].row = coo->row_idx[i];
        entries[i].col = coo->col_idx[i];
        entries[i].orig_idx = i;
    }

    /* Sort by (col, row) */
    qsort(entries, coo->nnz, sizeof(lmmc_coo_sort_entry_t), lmmc_coo_cmp_col_row);

    /* Count unique entries (merge duplicates) */
    unique_nnz = 1;
    for (i = 1; i < coo->nnz; ++i) {
        if (entries[i].row != entries[i - 1].row || entries[i].col != entries[i - 1].col) {
            unique_nnz++;
        }
    }

    /* Create CSC matrix */
    st = lmmc_sparse_create_csc(coo->rows, coo->cols, unique_nnz, out_csc);
    if (st != LMMC_STATUS_OK) {
        lmmc_free(entries);
        return st;
    }

    /* Fill CSC data: merge duplicates by summing values */
    {
        size_t csc_idx = 0;
        lmmc_real_t sum; LMMC_REAL_INIT(&sum);
        lmmc_real_t tmp; LMMC_REAL_INIT(&tmp);

        LMMC_REAL_SET(&sum, &coo->values[entries[0].orig_idx]);
        size_t cur_row = entries[0].row;
        size_t cur_col = entries[0].col;

        for (i = 1; i < coo->nnz; ++i) {
            if (entries[i].row == cur_row && entries[i].col == cur_col) {
                /* Duplicate: sum values */
                LMMC_REAL_ADD(&tmp, &sum, &coo->values[entries[i].orig_idx]);
                LMMC_REAL_SET(&sum, &tmp);
            } else {
                /* Store previous entry - in CSC, col_idx stores row indices */
                out_csc->col_idx[csc_idx] = cur_row;
                LMMC_REAL_SET(&out_csc->values[csc_idx], &sum);
                csc_idx++;

                /* Start new entry */
                cur_row = entries[i].row;
                cur_col = entries[i].col;
                LMMC_REAL_SET(&sum, &coo->values[entries[i].orig_idx]);
            }
        }
        /* Store last entry */
        out_csc->col_idx[csc_idx] = cur_row;
        LMMC_REAL_SET(&out_csc->values[csc_idx], &sum);

        /* Build col_ptr (stored in row_ptr for CSC format) */
        memset(out_csc->row_ptr, 0, (coo->cols + 1) * sizeof(size_t));

        /* Count unique entries per column */
        {
            size_t prev_row = entries[0].row;
            size_t prev_col = entries[0].col;

            out_csc->row_ptr[entries[0].col + 1]++;
            for (i = 1; i < coo->nnz; ++i) {
                if (entries[i].row != prev_row || entries[i].col != prev_col) {
                    out_csc->row_ptr[entries[i].col + 1]++;
                    prev_row = entries[i].row;
                    prev_col = entries[i].col;
                }
            }

            /* Prefix sum */
            for (i = 0; i < coo->cols; ++i) {
                out_csc->row_ptr[i + 1] += out_csc->row_ptr[i];
            }
        }

        LMMC_REAL_CLEAR(&sum);
        LMMC_REAL_CLEAR(&tmp);
    }

    lmmc_free(entries);
    return LMMC_STATUS_OK;
}

void lmmc_sparse_coo_destroy(lmmc_sparse_coo_t* coo) {
    if (coo == NULL) {
        return;
    }
    if (coo->values != NULL) {
        for (size_t k = 0; k < coo->capacity; ++k) {
            LMMC_REAL_CLEAR(&coo->values[k]);
        }
        lmmc_free(coo->values);
    }
    if (coo->row_idx != NULL) lmmc_free(coo->row_idx);
    if (coo->col_idx != NULL) lmmc_free(coo->col_idx);
    coo->rows = 0;
    coo->cols = 0;
    coo->nnz = 0;
    coo->capacity = 0;
    coo->row_idx = NULL;
    coo->col_idx = NULL;
    coo->values = NULL;
}

/* === 稀疏矩阵工具运算 === */

lmmc_status_t lmmc_sparse_add(
    lmmc_real_t alpha, const lmmc_sparse_mat_t* a,
    lmmc_real_t beta, const lmmc_sparse_mat_t* b,
    lmmc_sparse_mat_t* out_c
) {
    lmmc_sparse_mat_t a_csr = {0};
    lmmc_sparse_mat_t b_csr = {0};
    const lmmc_sparse_mat_t* pa = a;
    const lmmc_sparse_mat_t* pb = b;
    lmmc_status_t st = LMMC_STATUS_OK;
    size_t i;
    size_t nnz_c = 0;
    size_t* c_row_ptr = NULL;
    size_t* c_col_idx = NULL;
    lmmc_real_t* c_values = NULL;

    if (a == NULL || b == NULL || out_c == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    st = lmmc_sparse_validate(a);
    if (st != LMMC_STATUS_OK) return LMMC_STATUS_INVALID_ARGUMENT;
    st = lmmc_sparse_validate(b);
    if (st != LMMC_STATUS_OK) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Convert to CSR if needed */
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

    /* First pass: count nnz in result using dual-pointer merge */
    c_row_ptr = (size_t*)lmmc_alloc((pa->rows + 1) * sizeof(size_t));
    if (c_row_ptr == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    c_row_ptr[0] = 0;

    for (i = 0; i < pa->rows; ++i) {
        size_t a_start = pa->row_ptr[i];
        size_t a_end = pa->row_ptr[i + 1];
        size_t b_start = pb->row_ptr[i];
        size_t b_end = pb->row_ptr[i + 1];
        size_t ai = a_start;
        size_t bi = b_start;
        size_t row_nnz = 0;

        while (ai < a_end && bi < b_end) {
            if (pa->col_idx[ai] < pb->col_idx[bi]) {
                ai++;
                row_nnz++;
            } else if (pa->col_idx[ai] > pb->col_idx[bi]) {
                bi++;
                row_nnz++;
            } else {
                /* Same column: merge */
                ai++;
                bi++;
                row_nnz++;
            }
        }
        row_nnz += (a_end - ai) + (b_end - bi);
        nnz_c += row_nnz;
        c_row_ptr[i + 1] = nnz_c;
    }

    /* Allocate result arrays */
    if (nnz_c > 0) {
        c_col_idx = (size_t*)lmmc_alloc(nnz_c * sizeof(size_t));
        c_values = (lmmc_real_t*)lmmc_alloc(nnz_c * sizeof(lmmc_real_t));
        if (c_col_idx == NULL || c_values == NULL) {
            st = LMMC_STATUS_ALLOCATION_FAILED;
            goto cleanup;
        }
        for (size_t k = 0; k < nnz_c; ++k) {
            LMMC_REAL_INIT(&c_values[k]);
        }
    }

    /* Second pass: fill values using dual-pointer merge */
    {
        lmmc_real_t tmp_a; LMMC_REAL_INIT(&tmp_a);
        lmmc_real_t tmp_b; LMMC_REAL_INIT(&tmp_b);
        lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
        size_t c_idx = 0;

        for (i = 0; i < pa->rows; ++i) {
            size_t a_start = pa->row_ptr[i];
            size_t a_end = pa->row_ptr[i + 1];
            size_t b_start = pb->row_ptr[i];
            size_t b_end = pb->row_ptr[i + 1];
            size_t ai = a_start;
            size_t bi = b_start;

            while (ai < a_end && bi < b_end) {
                if (pa->col_idx[ai] < pb->col_idx[bi]) {
                    /* Only A contributes */
                    c_col_idx[c_idx] = pa->col_idx[ai];
                    LMMC_REAL_MUL(&c_values[c_idx], &alpha, &pa->values[ai]);
                    c_idx++;
                    ai++;
                } else if (pa->col_idx[ai] > pb->col_idx[bi]) {
                    /* Only B contributes */
                    c_col_idx[c_idx] = pb->col_idx[bi];
                    LMMC_REAL_MUL(&c_values[c_idx], &beta, &pb->values[bi]);
                    c_idx++;
                    bi++;
                } else {
                    /* Both contribute: alpha*A[i,j] + beta*B[i,j] */
                    c_col_idx[c_idx] = pa->col_idx[ai];
                    LMMC_REAL_MUL(&tmp_a, &alpha, &pa->values[ai]);
                    LMMC_REAL_MUL(&tmp_b, &beta, &pb->values[bi]);
                    LMMC_REAL_ADD(&c_values[c_idx], &tmp_a, &tmp_b);
                    c_idx++;
                    ai++;
                    bi++;
                }
            }
            /* Remaining A entries */
            while (ai < a_end) {
                c_col_idx[c_idx] = pa->col_idx[ai];
                LMMC_REAL_MUL(&c_values[c_idx], &alpha, &pa->values[ai]);
                c_idx++;
                ai++;
            }
            /* Remaining B entries */
            while (bi < b_end) {
                c_col_idx[c_idx] = pb->col_idx[bi];
                LMMC_REAL_MUL(&c_values[c_idx], &beta, &pb->values[bi]);
                c_idx++;
                bi++;
            }
        }

        LMMC_REAL_CLEAR(&tmp_a);
        LMMC_REAL_CLEAR(&tmp_b);
        LMMC_REAL_CLEAR(&tmp_sum);
    }

    /* Assign result */
    out_c->rows = pa->rows;
    out_c->cols = pa->cols;
    out_c->nnz = nnz_c;
    out_c->row_ptr = c_row_ptr;
    out_c->col_idx = c_col_idx;
    out_c->values = c_values;
    out_c->format = LMMC_SPARSE_CSR;
    out_c->owns_data = 1;

    /* Prevent cleanup from freeing these */
    c_row_ptr = NULL;
    c_col_idx = NULL;
    c_values = NULL;
    st = LMMC_STATUS_OK;

cleanup:
    if (c_row_ptr) lmmc_free(c_row_ptr);
    if (c_col_idx) lmmc_free(c_col_idx);
    if (c_values) {
        for (size_t k = 0; k < nnz_c; ++k) {
            LMMC_REAL_CLEAR(&c_values[k]);
        }
        lmmc_free(c_values);
    }
    lmmc_sparse_destroy(&a_csr);
    lmmc_sparse_destroy(&b_csr);
    return st;
}

lmmc_status_t lmmc_sparse_scale(
    lmmc_sparse_mat_t* a,
    lmmc_real_t alpha
) {
    size_t i;
    lmmc_real_t tmp; LMMC_REAL_INIT(&tmp);

    if (a == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_status_t st = lmmc_sparse_validate(a);
    if (st != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < a->nnz; ++i) {
        LMMC_REAL_MUL(&tmp, &a->values[i], &alpha);
        LMMC_REAL_SET(&a->values[i], &tmp);
    }

    LMMC_REAL_CLEAR(&tmp);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_norm_fro(
    const lmmc_sparse_mat_t* a,
    lmmc_real_t* out_norm
) {
    size_t i;
    lmmc_real_t sum; LMMC_REAL_INIT(&sum);
    lmmc_real_t tmp; LMMC_REAL_INIT(&tmp);
    lmmc_real_t tmp_add; LMMC_REAL_INIT(&tmp_add);

    if (a == NULL || out_norm == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_status_t st = lmmc_sparse_validate(a);
    if (st != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_SET_D(&sum, 0.0);

    for (i = 0; i < a->nnz; ++i) {
        LMMC_REAL_MUL(&tmp, &a->values[i], &a->values[i]);
        LMMC_REAL_ADD(&tmp_add, &sum, &tmp);
        LMMC_REAL_SET(&sum, &tmp_add);
    }

    LMMC_REAL_SQRT(out_norm, &sum);

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp);
    LMMC_REAL_CLEAR(&tmp_add);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_diag(
    const lmmc_sparse_mat_t* a,
    lmmc_vec_t* out_diag
) {
    lmmc_sparse_mat_t a_csr = {0};
    const lmmc_sparse_mat_t* pa = a;
    lmmc_status_t st;
    size_t i, p;
    size_t n;

    if (a == NULL || out_diag == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_sparse_validate(a);
    if (st != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Check square matrix */
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = a->rows;

    /* Convert to CSR if needed */
    if (a->format == LMMC_SPARSE_CSC) {
        st = lmmc_sparse_to_csr(a, &a_csr);
        if (st != LMMC_STATUS_OK) return st;
        pa = &a_csr;
    }

    /* Create output vector if needed */
    if (out_diag->data == NULL || out_diag->size != n) {
        if (out_diag->data != NULL && out_diag->owns_data) {
            lmmc_vec_destroy(out_diag);
        }
        st = lmmc_vec_create(n, out_diag);
        if (st != LMMC_STATUS_OK) {
            lmmc_sparse_destroy(&a_csr);
            return st;
        }
    }

    /* Initialize diagonal to zero */
    {
        lmmc_real_t zero;
        LMMC_REAL_INIT(&zero);
        LMMC_REAL_SET_D(&zero, 0.0);
        st = lmmc_vec_fill(out_diag, zero);
        LMMC_REAL_CLEAR(&zero);
        if (st != LMMC_STATUS_OK) {
            lmmc_sparse_destroy(&a_csr);
            return st;
        }
    }

    /* Extract diagonal elements from CSR */
    for (i = 0; i < n; ++i) {
        for (p = pa->row_ptr[i]; p < pa->row_ptr[i + 1]; ++p) {
            if (pa->col_idx[p] == i) {
                LMMC_REAL_SET(&out_diag->data[i], &pa->values[p]);
                break;  /* CSR has sorted columns within a row (or at most one diagonal per row) */
            }
        }
    }

    lmmc_sparse_destroy(&a_csr);
    return LMMC_STATUS_OK;
}
