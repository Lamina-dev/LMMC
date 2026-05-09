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
    if (src->format == LMMC_SPARSE_CSC) {
        lmmc_status_t st = lmmc_sparse_create_csc(src->rows, src->cols, src->nnz, dst);
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
            LMMC_REAL_SET(&dst->values[dest_idx], &src->values[p]);
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
            for (size_t k = 0; k < src->nnz; ++k) {
                LMMC_REAL_SET(&dst->values[k], &src->values[k]);
            }
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
