/**
 * @file sparse_builder.c
 * @brief 稀疏矩阵增量构造器与 COO 条目累积。
 */
#include <string.h>

#include "memory_bridge.h"
#include "internal.h"
#include "sparse_internal.h"
#include "lmmc/sparse.h"

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
    if (!lmmc_safe_mul_size(b->capacity, sizeof(size_t), &sz_idx) ||
        !lmmc_safe_mul_size(b->capacity, sizeof(lmmc_real_t), &sz_vals)) {
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

        if (!lmmc_safe_mul_size(b->capacity, 2, &new_cap) ||
            !lmmc_safe_mul_size(new_cap, sizeof(size_t), &sz_idx) ||
            !lmmc_safe_mul_size(new_cap, sizeof(lmmc_real_t), &sz_vals)) {
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


    for (i = 0; i < b->nnz; ++i) {
        out->row_ptr[major[i] + 1]++;
    }

    for (i = 0; i < outer_dim; ++i) {
        out->row_ptr[i+1] += out->row_ptr[i];
    }


    size_t* next = (size_t*)lmmc_alloc_array(outer_dim, sizeof(size_t));
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

    if (!lmmc_safe_mul_size(actual_cap, sizeof(size_t), &sz_idx) ||
        !lmmc_safe_mul_size(actual_cap, sizeof(lmmc_real_t), &sz_vals)) {
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


    if (coo->nnz >= coo->capacity) {
        size_t new_cap = 0;
        size_t sz_idx = 0;
        size_t sz_vals = 0;
        size_t* nr = NULL;
        size_t* nc = NULL;
        lmmc_real_t* nv = NULL;

        if (!lmmc_safe_mul_size(coo->capacity, 2, &new_cap) ||
            !lmmc_safe_mul_size(new_cap, sizeof(size_t), &sz_idx) ||
            !lmmc_safe_mul_size(new_cap, sizeof(lmmc_real_t), &sz_vals)) {
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
