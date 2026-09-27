/**
 * @file sparse_arithmetic.c
 * @brief 稀疏矩阵算术：线性组合、缩放、Frobenius 范数与对角线提取。
 */
#include "memory_bridge.h"
#include "internal.h"
#include "sparse_internal.h"
#include "lmmc/sparse.h"

static size_t lmmc_sparse_add_row_counts(
    const lmmc_sparse_mat_t* a, const lmmc_sparse_mat_t* b,
    size_t* row_ptr)
{
    size_t nnz = 0;
    for (size_t i = 0; i < a->rows; ++i) {
        size_t ai = a->row_ptr[i], bi = b->row_ptr[i];
        size_t a_end = a->row_ptr[i + 1], b_end = b->row_ptr[i + 1];
        size_t row_nnz = 0;
        while (ai < a_end && bi < b_end) {
            if (a->col_idx[ai] < b->col_idx[bi]) {
                ai++;
            } else if (a->col_idx[ai] > b->col_idx[bi]) {
                bi++;
            } else {
                ai++;
                bi++;
            }
            row_nnz++;
        }
        row_nnz += (a_end - ai) + (b_end - bi);
        nnz += row_nnz;
        row_ptr[i + 1] = nnz;
    }
    return nnz;
}

static void lmmc_sparse_add_merge(
    const lmmc_sparse_mat_t* pa, const lmmc_sparse_mat_t* pb,
    lmmc_real_t alpha, lmmc_real_t beta,
    size_t* c_col_idx, lmmc_real_t* c_values)
{
    lmmc_real_t tmp_a; LMMC_REAL_INIT(&tmp_a);
    lmmc_real_t tmp_b; LMMC_REAL_INIT(&tmp_b);
    size_t c_idx = 0;
    for (size_t i = 0; i < pa->rows; ++i) {
        size_t ai = pa->row_ptr[i], bi = pb->row_ptr[i];
        size_t a_end = pa->row_ptr[i + 1], b_end = pb->row_ptr[i + 1];
        while (ai < a_end && bi < b_end) {
            if (pa->col_idx[ai] < pb->col_idx[bi]) {
                c_col_idx[c_idx] = pa->col_idx[ai];
                LMMC_REAL_MUL(&c_values[c_idx], &alpha, &pa->values[ai]);
                c_idx++;
                ai++;
            } else if (pa->col_idx[ai] > pb->col_idx[bi]) {
                c_col_idx[c_idx] = pb->col_idx[bi];
                LMMC_REAL_MUL(&c_values[c_idx], &beta, &pb->values[bi]);
                c_idx++;
                bi++;
            } else {
                c_col_idx[c_idx] = pa->col_idx[ai];
                LMMC_REAL_MUL(&tmp_a, &alpha, &pa->values[ai]);
                LMMC_REAL_MUL(&tmp_b, &beta, &pb->values[bi]);
                LMMC_REAL_ADD(&c_values[c_idx], &tmp_a, &tmp_b);
                c_idx++;
                ai++;
                bi++;
            }
        }
        while (ai < a_end) {
            c_col_idx[c_idx] = pa->col_idx[ai];
            LMMC_REAL_MUL(&c_values[c_idx], &alpha, &pa->values[ai]);
            c_idx++;
            ai++;
        }
        while (bi < b_end) {
            c_col_idx[c_idx] = pb->col_idx[bi];
            LMMC_REAL_MUL(&c_values[c_idx], &beta, &pb->values[bi]);
            c_idx++;
            bi++;
        }
    }
    LMMC_REAL_CLEAR(&tmp_a);
    LMMC_REAL_CLEAR(&tmp_b);
}

static void lmmc_sparse_diag_copy(
    const lmmc_sparse_mat_t* pa, lmmc_vec_t* out_diag)
{
    for (size_t i = 0; i < pa->rows; ++i) {
        for (size_t p = pa->row_ptr[i]; p < pa->row_ptr[i + 1]; ++p) {
            if (pa->col_idx[p] == i) {
                LMMC_REAL_SET(&out_diag->data[i], &pa->values[p]);
                break;
            }
        }
    }
}

static lmmc_status_t lmmc_sparse_add_validate(
    const lmmc_sparse_mat_t* a, const lmmc_sparse_mat_t* b,
    const lmmc_sparse_mat_t* out_c)
{
    if (a == NULL || b == NULL || out_c == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_sparse_validate(a) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_sparse_validate(b) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_sparse_add_allocate_values(
    size_t nnz, size_t** indices, lmmc_real_t** values)
{
    *indices = (size_t*)lmmc_memory_alloc_array(nnz, sizeof(size_t));
    *values = (lmmc_real_t*)lmmc_memory_alloc_array(nnz, sizeof(lmmc_real_t));
    if (*indices == NULL || *values == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    for (size_t k = 0; k < nnz; ++k) {
        LMMC_REAL_INIT(&(*values)[k]);
    }
    return LMMC_STATUS_OK;
}

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
    size_t nnz_c = 0;
    size_t* c_row_ptr = NULL;
    size_t* c_col_idx = NULL;
    lmmc_real_t* c_values = NULL;

    st = lmmc_sparse_add_validate(a, b, out_c);
    if (st != LMMC_STATUS_OK) {
        return st;
    }


    if (a->format == LMMC_SPARSE_CSC) {
        st = lmmc_sparse_to_csr(a, &a_csr);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
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


    c_row_ptr = (size_t*)lmmc_memory_alloc_array_plus(pa->rows, 1, sizeof(size_t));
    if (c_row_ptr == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    c_row_ptr[0] = 0;

    nnz_c = lmmc_sparse_add_row_counts(pa, pb, c_row_ptr);


    if (nnz_c > 0) {
        st = lmmc_sparse_add_allocate_values(nnz_c, &c_col_idx, &c_values);
        if (st != LMMC_STATUS_OK) {
            goto cleanup;
        }
    }


    if (nnz_c > 0) {
        lmmc_sparse_add_merge(pa, pb, alpha, beta, c_col_idx, c_values);
    }


    out_c->rows = pa->rows;
    out_c->cols = pa->cols;
    out_c->nnz = nnz_c;
    out_c->row_ptr = c_row_ptr;
    out_c->col_idx = c_col_idx;
    out_c->values = c_values;
    out_c->format = LMMC_SPARSE_CSR;
    out_c->owns_data = 1;


    c_row_ptr = NULL;
    c_col_idx = NULL;
    c_values = NULL;
    st = LMMC_STATUS_OK;

cleanup:
    if (c_row_ptr) {
        lmmc_memory_free(c_row_ptr);
    }
    if (c_col_idx) {
        lmmc_memory_free(c_col_idx);
    }
    if (c_values) {
        for (size_t k = 0; k < nnz_c; ++k) {
            LMMC_REAL_CLEAR(&c_values[k]);
        }
        lmmc_memory_free(c_values);
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
    lmmc_scaled_sumsq_t acc;

    if (a == NULL || out_norm == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_status_t st = lmmc_sparse_validate(a);
    if (st != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_scaled_sumsq_init(&acc);
    for (size_t i = 0; i < a->nnz; ++i) {
        lmmc_scaled_sumsq_add(&acc, a->values[i]);
    }
    *out_norm = lmmc_scaled_sumsq_norm(&acc);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_diag(
    const lmmc_sparse_mat_t* a,
    lmmc_vec_t* out_diag
) {
    lmmc_sparse_mat_t a_csr = {0};
    const lmmc_sparse_mat_t* pa = a;
    lmmc_status_t st;
    size_t n;

    if (a == NULL || out_diag == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_sparse_validate(a);
    if (st != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = a->rows;


    if (a->format == LMMC_SPARSE_CSC) {
        st = lmmc_sparse_to_csr(a, &a_csr);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        pa = &a_csr;
    }


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


    lmmc_sparse_diag_copy(pa, out_diag);

    lmmc_sparse_destroy(&a_csr);
    return LMMC_STATUS_OK;
}
