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

static lmmc_status_t lmmc_sparse_entries_grow(
    size_t nnz, size_t* capacity, size_t** rows, size_t** cols,
    lmmc_real_t** values)
{
    size_t new_cap = 0;
    size_t sz_idx = 0;
    size_t sz_vals = 0;
    size_t* nr = NULL;
    size_t* nc = NULL;
    lmmc_real_t* nv = NULL;

    if (!lmmc_safe_mul_size(*capacity, 2, &new_cap) ||
        !lmmc_safe_mul_size(new_cap, sizeof(size_t), &sz_idx) ||
        !lmmc_safe_mul_size(new_cap, sizeof(lmmc_real_t), &sz_vals)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    nr = (size_t*)lmmc_memory_alloc(sz_idx);
    nc = (size_t*)lmmc_memory_alloc(sz_idx);
    nv = (lmmc_real_t*)lmmc_memory_alloc(sz_vals);

    if (nr == NULL || nc == NULL || nv == NULL) {
        if (nr) {
            lmmc_memory_free(nr);
        }
        if (nc) {
            lmmc_memory_free(nc);
        }
        if (nv) {
            lmmc_memory_free(nv);
        }
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (size_t k = 0; k < new_cap; ++k) {
        LMMC_REAL_INIT(&nv[k]);
    }

    if (nnz > 0) {
        memcpy(nr, *rows, nnz * sizeof(size_t));
        memcpy(nc, *cols, nnz * sizeof(size_t));
        for (size_t k = 0; k < nnz; ++k) {
            LMMC_REAL_SET(&nv[k], &(*values)[k]);
        }
    }

    for (size_t k = 0; k < *capacity; ++k) {
        LMMC_REAL_CLEAR(&(*values)[k]);
    }
    lmmc_memory_free(*rows);
    lmmc_memory_free(*cols);
    lmmc_memory_free(*values);

    *rows = nr;
    *cols = nc;
    *values = nv;
    *capacity = new_cap;
    return LMMC_STATUS_OK;
}


lmmc_status_t lmmc_sparse_builder_create(size_t rows, size_t cols, size_t initial_capacity, lmmc_sparse_builder_t** out_builder) {
    lmmc_sparse_builder_t* b = NULL;
    if (out_builder == NULL || rows == 0 || cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    b = (lmmc_sparse_builder_t*)lmmc_memory_alloc(sizeof(lmmc_sparse_builder_t));
    if (b == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    b->rows = rows;
    b->cols = cols;
    b->nnz = 0;
    b->capacity = initial_capacity > 0 ? initial_capacity : 16;

    size_t sz_idx = 0;
    size_t sz_vals = 0;
    if (!lmmc_safe_mul_size(b->capacity, sizeof(size_t), &sz_idx) ||
        !lmmc_safe_mul_size(b->capacity, sizeof(lmmc_real_t), &sz_vals)) {
        lmmc_memory_free(b);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    b->r_idx = (size_t*)lmmc_memory_alloc(sz_idx);
    b->c_idx = (size_t*)lmmc_memory_alloc(sz_idx);
    b->vals = (lmmc_real_t*)lmmc_memory_alloc(sz_vals);

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
    if (b == NULL || row >= b->rows || col >= b->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (b->nnz >= b->capacity) {
        lmmc_status_t st = lmmc_sparse_entries_grow(
            b->nnz, &b->capacity, &b->r_idx, &b->c_idx, &b->vals);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }

    b->r_idx[b->nnz] = row;
    b->c_idx[b->nnz] = col;
    LMMC_REAL_SET(&b->vals[b->nnz], &val);
    b->nnz++;
    return LMMC_STATUS_OK;
}

void lmmc_sparse_builder_destroy(lmmc_sparse_builder_t* b) {
    if (b) {
        if (b->r_idx) {
            lmmc_memory_free(b->r_idx);
        }
        if (b->c_idx) {
            lmmc_memory_free(b->c_idx);
        }
        if (b->vals) {
            for (size_t k = 0; k < b->capacity; ++k) {
                LMMC_REAL_CLEAR(&b->vals[k]);
            }
            lmmc_memory_free(b->vals);
        }
        lmmc_memory_free(b);
    }
}

lmmc_status_t lmmc_sparse_builder_build(lmmc_sparse_builder_t* b, lmmc_sparse_format_t format, lmmc_sparse_mat_t* out) {
    if (b == NULL || out == NULL ||
        (format != LMMC_SPARSE_CSR && format != LMMC_SPARSE_CSC)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_sparse_coo_t coo = {0};
    coo.rows = b->rows;
    coo.cols = b->cols;
    coo.nnz = b->nnz;
    coo.capacity = b->capacity;
    coo.row_idx = b->r_idx;
    coo.col_idx = b->c_idx;
    coo.values = b->vals;
    return format == LMMC_SPARSE_CSR
        ? lmmc_sparse_coo_to_csr(&coo, out)
        : lmmc_sparse_coo_to_csc(&coo, out);
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

    out_coo->row_idx = (size_t*)lmmc_memory_alloc(sz_idx);
    out_coo->col_idx = (size_t*)lmmc_memory_alloc(sz_idx);
    out_coo->values = (lmmc_real_t*)lmmc_memory_alloc(sz_vals);

    if (out_coo->row_idx == NULL || out_coo->col_idx == NULL || out_coo->values == NULL) {
        if (out_coo->row_idx) {
            lmmc_memory_free(out_coo->row_idx);
        }
        if (out_coo->col_idx) {
            lmmc_memory_free(out_coo->col_idx);
        }
        if (out_coo->values) {
            lmmc_memory_free(out_coo->values);
        }
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
        lmmc_status_t st = lmmc_sparse_entries_grow(
            coo->nnz, &coo->capacity, &coo->row_idx, &coo->col_idx, &coo->values);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }

    coo->row_idx[coo->nnz] = row;
    coo->col_idx[coo->nnz] = col;
    LMMC_REAL_SET(&coo->values[coo->nnz], &value);
    coo->nnz++;
    return LMMC_STATUS_OK;
}
