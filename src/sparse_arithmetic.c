/**
 * @file sparse_arithmetic.c
 * @brief 稀疏矩阵算术：线性组合、缩放、Frobenius 范数与对角线提取。
 */
#include "memory_bridge.h"
#include "internal.h"
#include "sparse_internal.h"
#include "lmmc/sparse.h"

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

                ai++;
                bi++;
                row_nnz++;
            }
        }
        row_nnz += (a_end - ai) + (b_end - bi);
        nnz_c += row_nnz;
        c_row_ptr[i + 1] = nnz_c;
    }


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


    if (nnz_c > 0) {
        lmmc_real_t tmp_a; LMMC_REAL_INIT(&tmp_a);
        lmmc_real_t tmp_b; LMMC_REAL_INIT(&tmp_b);
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


    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = a->rows;


    if (a->format == LMMC_SPARSE_CSC) {
        st = lmmc_sparse_to_csr(a, &a_csr);
        if (st != LMMC_STATUS_OK) return st;
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


    for (i = 0; i < n; ++i) {
        for (p = pa->row_ptr[i]; p < pa->row_ptr[i + 1]; ++p) {
            if (pa->col_idx[p] == i) {
                LMMC_REAL_SET(&out_diag->data[i], &pa->values[p]);
                break;
            }
        }
    }

    lmmc_sparse_destroy(&a_csr);
    return LMMC_STATUS_OK;
}
