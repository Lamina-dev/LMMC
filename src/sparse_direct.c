/**
 * @file sparse_direct.c
 * @brief 稀疏 LU / Cholesky 直接分解实现。
 */
#include <string.h>
#include <math.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/sparse.h"


struct lmmc_sparse_lu_t {
    size_t n;
    size_t* col_perm;
    size_t* row_perm;
    size_t* L_col_ptr;
    size_t* U_col_ptr;
    size_t* L_row_idx;
    lmmc_real_t* L_values;
    size_t* U_row_idx;
    lmmc_real_t* U_values;
    size_t L_nnz;
    size_t U_nnz;
};


static lmmc_status_t ensure_csc(const lmmc_sparse_mat_t* a,
                                lmmc_sparse_mat_t* csc_out,
                                int* needs_free)
{
    *needs_free = 0;
    if (a->format == LMMC_SPARSE_CSC) {

        *csc_out = *a;
        return LMMC_STATUS_OK;
    }

    *needs_free = 1;
    return lmmc_sparse_to_csc(a, csc_out);
}


lmmc_status_t lmmc_sparse_lu_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t** out_lu)
{
    lmmc_sparse_lu_t* lu = NULL;
    size_t n;


    if (a == NULL || out_lu == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows == 0 || a->cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = a->rows;


    lu = (lmmc_sparse_lu_t*)lmmc_alloc(sizeof(lmmc_sparse_lu_t));
    if (lu == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(lu, 0, sizeof(lmmc_sparse_lu_t));
    lu->n = n;


    lu->col_perm = (size_t*)lmmc_alloc(n * sizeof(size_t));
    lu->row_perm = (size_t*)lmmc_alloc(n * sizeof(size_t));
    if (lu->col_perm == NULL || lu->row_perm == NULL) {
        lmmc_sparse_lu_destroy(lu);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }


    for (size_t i = 0; i < n; i++) {
        lu->col_perm[i] = i;
        lu->row_perm[i] = i;
    }


    lu->L_col_ptr = (size_t*)lmmc_alloc((n + 1) * sizeof(size_t));
    lu->U_col_ptr = (size_t*)lmmc_alloc((n + 1) * sizeof(size_t));
    if (lu->L_col_ptr == NULL || lu->U_col_ptr == NULL) {
        lmmc_sparse_lu_destroy(lu);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(lu->L_col_ptr, 0, (n + 1) * sizeof(size_t));
    memset(lu->U_col_ptr, 0, (n + 1) * sizeof(size_t));


    {
        size_t max_nnz;
        size_t n_sq = n * n;
        size_t heuristic = a->nnz * 10;

        if (n <= 128) {

            max_nnz = n_sq;
        } else {

            max_nnz = (heuristic < n_sq) ? heuristic : n_sq;
        }

        lu->L_row_idx = (size_t*)lmmc_alloc(max_nnz * sizeof(size_t));
        lu->L_values = (lmmc_real_t*)lmmc_alloc(max_nnz * sizeof(lmmc_real_t));
        lu->U_row_idx = (size_t*)lmmc_alloc(max_nnz * sizeof(size_t));
        lu->U_values = (lmmc_real_t*)lmmc_alloc(max_nnz * sizeof(lmmc_real_t));

        if (lu->L_row_idx == NULL || lu->L_values == NULL ||
            lu->U_row_idx == NULL || lu->U_values == NULL) {
            lmmc_sparse_lu_destroy(lu);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
    }

    lu->L_nnz = 0;
    lu->U_nnz = 0;

    *out_lu = lu;
    return LMMC_STATUS_OK;
}


lmmc_status_t lmmc_sparse_lu_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t* lu)
{
    lmmc_sparse_mat_t csc;
    int csc_needs_free = 0;
    lmmc_status_t status = LMMC_STATUS_OK;
    size_t n;
    lmmc_real_t* col_dense = NULL;
    int* nonzero_flag = NULL;
    size_t* nonzero_list = NULL;
    size_t* piv_inv = NULL;


    if (a == NULL || lu == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != lu->n) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = lu->n;


    status = ensure_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) {
        return status;
    }


    col_dense = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    nonzero_flag = (int*)lmmc_alloc(n * sizeof(int));
    nonzero_list = (size_t*)lmmc_alloc(n * sizeof(size_t));
    piv_inv = (size_t*)lmmc_alloc(n * sizeof(size_t));

    if (col_dense == NULL || nonzero_flag == NULL ||
        nonzero_list == NULL || piv_inv == NULL) {
        status = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }


    for (size_t i = 0; i < n; i++) {
        piv_inv[i] = i;
    }


    lu->L_nnz = 0;
    lu->U_nnz = 0;


    for (size_t j = 0; j < n; j++) {
        size_t nz_count = 0;


        memset(col_dense, 0, n * sizeof(lmmc_real_t));
        memset(nonzero_flag, 0, n * sizeof(int));


        {
            size_t col_start = csc.row_ptr[j];
            size_t col_end = csc.row_ptr[j + 1];
            for (size_t p = col_start; p < col_end; p++) {
                size_t row = csc.col_idx[p];

                size_t prow = piv_inv[row];
                col_dense[prow] = csc.values[p];
                if (!nonzero_flag[prow]) {
                    nonzero_flag[prow] = 1;
                    nonzero_list[nz_count++] = prow;
                }
            }
        }


        for (size_t k = 0; k < j; k++) {
            if (col_dense[k] == 0.0) continue;

            lmmc_real_t ukj = col_dense[k];


            size_t L_start = lu->L_col_ptr[k];
            size_t L_end = lu->L_col_ptr[k + 1];
            for (size_t p = L_start; p < L_end; p++) {
                size_t row = lu->L_row_idx[p];
                lmmc_real_t old_val = col_dense[row];
                col_dense[row] = old_val - lu->L_values[p] * ukj;
                if (!nonzero_flag[row]) {
                    nonzero_flag[row] = 1;
                    nonzero_list[nz_count++] = row;
                }
            }
        }


        {
            size_t pivot_row = j;
            lmmc_real_t max_val = lmmc_abs(col_dense[j]);

            for (size_t i = j + 1; i < n; i++) {
                lmmc_real_t av = lmmc_abs(col_dense[i]);
                if (av > max_val) {
                    max_val = av;
                    pivot_row = i;
                }
            }


            if (max_val == 0.0) {
                status = LMMC_STATUS_SINGULAR_MATRIX;
                goto cleanup;
            }


            if (pivot_row != j) {
                lmmc_real_t tmp = col_dense[j];
                col_dense[j] = col_dense[pivot_row];
                col_dense[pivot_row] = tmp;


                size_t orig_j = lu->row_perm[j];
                size_t orig_p = lu->row_perm[pivot_row];
                lu->row_perm[j] = orig_p;
                lu->row_perm[pivot_row] = orig_j;
                piv_inv[orig_p] = j;
                piv_inv[orig_j] = pivot_row;


                for (size_t k = 0; k < j; k++) {
                    size_t L_start = lu->L_col_ptr[k];
                    size_t L_end = lu->L_col_ptr[k + 1];

                    size_t idx_j = (size_t)-1;
                    size_t idx_p = (size_t)-1;
                    for (size_t p = L_start; p < L_end; p++) {
                        if (lu->L_row_idx[p] == j) idx_j = p;
                        if (lu->L_row_idx[p] == pivot_row) idx_p = p;
                    }
                    if (idx_j != (size_t)-1 && idx_p != (size_t)-1) {
                        lmmc_real_t tmp_v = lu->L_values[idx_j];
                        lu->L_values[idx_j] = lu->L_values[idx_p];
                        lu->L_values[idx_p] = tmp_v;
                    } else if (idx_j != (size_t)-1) {
                        lu->L_row_idx[idx_j] = pivot_row;
                    } else if (idx_p != (size_t)-1) {
                        lu->L_row_idx[idx_p] = j;
                    }
                }
            }
        }


        lu->U_col_ptr[j] = lu->U_nnz;
        for (size_t i = 0; i <= j; i++) {
            if (col_dense[i] != 0.0) {
                lu->U_row_idx[lu->U_nnz] = i;
                lu->U_values[lu->U_nnz] = col_dense[i];
                lu->U_nnz++;
            }
        }


        lu->L_col_ptr[j] = lu->L_nnz;
        {
            lmmc_real_t diag = col_dense[j];
            for (size_t i = j + 1; i < n; i++) {
                if (col_dense[i] != 0.0) {
                    lu->L_row_idx[lu->L_nnz] = i;
                    lu->L_values[lu->L_nnz] = col_dense[i] / diag;
                    lu->L_nnz++;
                }
            }
        }
    }


    lu->L_col_ptr[n] = lu->L_nnz;
    lu->U_col_ptr[n] = lu->U_nnz;

cleanup:
    if (col_dense) lmmc_free(col_dense);
    if (nonzero_flag) lmmc_free(nonzero_flag);
    if (nonzero_list) lmmc_free(nonzero_list);
    if (piv_inv) lmmc_free(piv_inv);
    if (csc_needs_free) {
        lmmc_sparse_destroy(&csc);
    }

    return status;
}


lmmc_status_t lmmc_sparse_lu_solve(
    const lmmc_sparse_lu_t* lu,
    const lmmc_vec_t* b,
    lmmc_vec_t* x)
{
    size_t n;
    lmmc_real_t* work = NULL;


    if (lu == NULL || b == NULL || x == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = lu->n;

    if (b->size != n || x->size != n) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }


    work = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (work == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }


    for (size_t i = 0; i < n; i++) {
        work[i] = b->data[lu->row_perm[i]];
    }


    for (size_t j = 0; j < n; j++) {

        size_t L_start = lu->L_col_ptr[j];
        size_t L_end = lu->L_col_ptr[j + 1];
        for (size_t p = L_start; p < L_end; p++) {
            size_t i = lu->L_row_idx[p];
            work[i] -= lu->L_values[p] * work[j];
        }
    }


    for (size_t jj = 0; jj < n; jj++) {
        size_t j = n - 1 - jj;
        size_t U_start = lu->U_col_ptr[j];
        size_t U_end = lu->U_col_ptr[j + 1];


        lmmc_real_t diag = 0.0;
        size_t diag_idx = (size_t)-1;
        for (size_t p = U_start; p < U_end; p++) {
            if (lu->U_row_idx[p] == j) {
                diag = lu->U_values[p];
                diag_idx = p;
                break;
            }
        }

        if (diag == 0.0) {
            lmmc_free(work);
            return LMMC_STATUS_SINGULAR_MATRIX;
        }

        work[j] = work[j] / diag;


        for (size_t p = U_start; p < U_end; p++) {
            size_t i = lu->U_row_idx[p];
            if (i != j) {
                work[i] -= lu->U_values[p] * work[j];
            }
        }
    }


    memcpy(x->data, work, n * sizeof(lmmc_real_t));

    lmmc_free(work);
    return LMMC_STATUS_OK;
}


void lmmc_sparse_lu_destroy(lmmc_sparse_lu_t* lu)
{
    if (lu == NULL) return;

    if (lu->col_perm) lmmc_free(lu->col_perm);
    if (lu->row_perm) lmmc_free(lu->row_perm);
    if (lu->L_col_ptr) lmmc_free(lu->L_col_ptr);
    if (lu->U_col_ptr) lmmc_free(lu->U_col_ptr);
    if (lu->L_row_idx) lmmc_free(lu->L_row_idx);
    if (lu->L_values) lmmc_free(lu->L_values);
    if (lu->U_row_idx) lmmc_free(lu->U_row_idx);
    if (lu->U_values) lmmc_free(lu->U_values);

    lmmc_free(lu);
}


struct lmmc_sparse_chol_t {
    size_t n;
    size_t* perm;
    size_t* L_col_ptr;
    size_t* L_row_idx;
    lmmc_real_t* L_values;
    size_t L_nnz;
};


lmmc_status_t lmmc_sparse_chol_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t** out_chol)
{
    lmmc_sparse_chol_t* chol = NULL;
    size_t n;
    size_t max_nnz;


    if (a == NULL || out_chol == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows == 0 || a->cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = a->rows;


    chol = (lmmc_sparse_chol_t*)lmmc_alloc(sizeof(lmmc_sparse_chol_t));
    if (chol == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(chol, 0, sizeof(lmmc_sparse_chol_t));
    chol->n = n;


    chol->perm = (size_t*)lmmc_alloc(n * sizeof(size_t));
    if (chol->perm == NULL) {
        lmmc_sparse_chol_destroy(chol);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    for (size_t i = 0; i < n; i++) {
        chol->perm[i] = i;
    }


    chol->L_col_ptr = (size_t*)lmmc_alloc((n + 1) * sizeof(size_t));
    if (chol->L_col_ptr == NULL) {
        lmmc_sparse_chol_destroy(chol);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(chol->L_col_ptr, 0, (n + 1) * sizeof(size_t));


    max_nnz = n * (n + 1) / 2;
    if (max_nnz == 0) max_nnz = 1;

    chol->L_row_idx = (size_t*)lmmc_alloc(max_nnz * sizeof(size_t));
    chol->L_values = (lmmc_real_t*)lmmc_alloc(max_nnz * sizeof(lmmc_real_t));
    if (chol->L_row_idx == NULL || chol->L_values == NULL) {
        lmmc_sparse_chol_destroy(chol);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    chol->L_nnz = 0;
    *out_chol = chol;
    return LMMC_STATUS_OK;
}


lmmc_status_t lmmc_sparse_chol_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t* chol)
{
    lmmc_sparse_mat_t csc;
    int csc_needs_free = 0;
    lmmc_status_t status = LMMC_STATUS_OK;
    size_t n;
    lmmc_real_t* col_dense = NULL;


    if (a == NULL || chol == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != chol->n) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = chol->n;


    status = ensure_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) {
        return status;
    }


    col_dense = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (col_dense == NULL) {
        if (csc_needs_free) lmmc_sparse_destroy(&csc);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }


    chol->L_nnz = 0;


    for (size_t j = 0; j < n; j++) {

        chol->L_col_ptr[j] = chol->L_nnz;


        memset(col_dense, 0, n * sizeof(lmmc_real_t));


        {
            size_t col_start = csc.row_ptr[j];
            size_t col_end = csc.row_ptr[j + 1];
            for (size_t p = col_start; p < col_end; p++) {
                size_t row = csc.col_idx[p];
                if (row >= j) {
                    col_dense[row] = csc.values[p];
                }
            }
        }


        for (size_t k = 0; k < j; k++) {
            size_t L_start = chol->L_col_ptr[k];
            size_t L_end = chol->L_col_ptr[k + 1];
            lmmc_real_t Ljk = 0.0;


            for (size_t p = L_start; p < L_end; p++) {
                size_t row = chol->L_row_idx[p];
                if (row == j) {
                    Ljk = chol->L_values[p];
                    break;
                }
                if (row > j) {
                    break;
                }
            }

            if (Ljk == 0.0) continue;


            for (size_t p = L_start; p < L_end; p++) {
                size_t i = chol->L_row_idx[p];
                if (i >= j) {
                    col_dense[i] -= chol->L_values[p] * Ljk;
                }
            }
        }


        {
            lmmc_real_t diag = col_dense[j];
            lmmc_real_t sqrt_diag;

            if (diag <= 0.0) {
                status = LMMC_STATUS_NOT_POSITIVE_DEFINITE;
                goto cleanup;
            }

            sqrt_diag = sqrt(diag);


            chol->L_row_idx[chol->L_nnz] = j;
            chol->L_values[chol->L_nnz] = sqrt_diag;
            chol->L_nnz++;


            for (size_t i = j + 1; i < n; i++) {
                if (col_dense[i] != 0.0) {
                    chol->L_row_idx[chol->L_nnz] = i;
                    chol->L_values[chol->L_nnz] = col_dense[i] / sqrt_diag;
                    chol->L_nnz++;
                }
            }
        }
    }


    chol->L_col_ptr[n] = chol->L_nnz;

cleanup:
    if (col_dense) lmmc_free(col_dense);
    if (csc_needs_free) {
        lmmc_sparse_destroy(&csc);
    }
    return status;
}


lmmc_status_t lmmc_sparse_chol_solve(
    const lmmc_sparse_chol_t* chol,
    const lmmc_vec_t* b,
    lmmc_vec_t* x)
{
    size_t n;
    lmmc_real_t* work = NULL;


    if (chol == NULL || b == NULL || x == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = chol->n;

    if (b->size != n || x->size != n) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }


    work = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (work == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }


    for (size_t i = 0; i < n; i++) {
        work[i] = b->data[chol->perm[i]];
    }


    for (size_t j = 0; j < n; j++) {
        size_t L_start = chol->L_col_ptr[j];
        size_t L_end = chol->L_col_ptr[j + 1];
        lmmc_real_t diag = 0.0;
        size_t diag_idx = (size_t)-1;


        for (size_t p = L_start; p < L_end; p++) {
            if (chol->L_row_idx[p] == j) {
                diag = chol->L_values[p];
                diag_idx = p;
                break;
            }
        }

        if (diag == 0.0) {
            lmmc_free(work);
            return LMMC_STATUS_NOT_POSITIVE_DEFINITE;
        }

        work[j] = work[j] / diag;


        for (size_t p = L_start; p < L_end; p++) {
            if (p == diag_idx) continue;
            size_t i = chol->L_row_idx[p];
            work[i] -= chol->L_values[p] * work[j];
        }
    }


    for (size_t jj = 0; jj < n; jj++) {
        size_t j = n - 1 - jj;
        size_t L_start = chol->L_col_ptr[j];
        size_t L_end = chol->L_col_ptr[j + 1];
        lmmc_real_t diag = 0.0;
        size_t diag_idx = (size_t)-1;

        for (size_t p = L_start; p < L_end; p++) {
            if (chol->L_row_idx[p] == j) {
                diag = chol->L_values[p];
                diag_idx = p;
                break;
            }
        }

        if (diag == 0.0) {
            lmmc_free(work);
            return LMMC_STATUS_NOT_POSITIVE_DEFINITE;
        }

        for (size_t p = L_start; p < L_end; p++) {
            if (p == diag_idx) continue;
            size_t i = chol->L_row_idx[p];
            work[j] -= chol->L_values[p] * work[i];
        }
        work[j] = work[j] / diag;
    }


    for (size_t i = 0; i < n; i++) {
        x->data[chol->perm[i]] = work[i];
    }

    lmmc_free(work);
    return LMMC_STATUS_OK;
}


void lmmc_sparse_chol_destroy(lmmc_sparse_chol_t* chol)
{
    if (chol == NULL) return;

    if (chol->perm) lmmc_free(chol->perm);
    if (chol->L_col_ptr) lmmc_free(chol->L_col_ptr);
    if (chol->L_row_idx) lmmc_free(chol->L_row_idx);
    if (chol->L_values) lmmc_free(chol->L_values);

    lmmc_free(chol);
}
