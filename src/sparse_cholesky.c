#include <string.h>
#include <math.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/sparse.h"
#include "sparse_direct_internal.h"

static lmmc_status_t chol_allocate_indices(lmmc_sparse_chol_t* chol)
{
    size_t n = chol->n;
    chol->perm = (size_t*)lmmc_memory_alloc_array(n, sizeof(size_t));
    chol->perm_inv = (size_t*)lmmc_memory_alloc_array(n, sizeof(size_t));
    chol->L_col_ptr = (size_t*)lmmc_memory_alloc_array_plus(n, 1, sizeof(size_t));

if (!chol->perm || !chol->perm_inv || !chol->L_col_ptr) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(chol->L_col_ptr, 0, (n + 1) * sizeof(size_t));
    return LMMC_STATUS_OK;
}

static lmmc_status_t chol_allocate_factors(lmmc_sparse_chol_t* chol, size_t nnz)
{
    size_t n = chol->n;
        size_t initial_capacity = (nnz > 64) ? nnz : 64;
        if (initial_capacity < n) {
            initial_capacity = n;
        }

        chol->L_row_idx = (size_t*)lmmc_memory_alloc_array(initial_capacity, sizeof(size_t));
        chol->L_values = (lmmc_real_t*)lmmc_memory_alloc_array(initial_capacity, sizeof(lmmc_real_t));

if (!chol->L_row_idx || !chol->L_values) {
    return LMMC_STATUS_ALLOCATION_FAILED;
}
    chol->L_capacity = initial_capacity;
    return LMMC_STATUS_OK;
}
lmmc_status_t lmmc_sparse_chol_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t** out_chol)
{
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_sparse_mat_t csc;
    int csc_needs_free = 0;
    size_t n;
    lmmc_status_t status;

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

    status = lmmc_sparse_direct_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) {
        return status;
    }

    chol = (lmmc_sparse_chol_t*)lmmc_memory_alloc(sizeof(lmmc_sparse_chol_t));
    if (chol == NULL) {
        if (csc_needs_free) {
            lmmc_sparse_destroy(&csc);
        }
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(chol, 0, sizeof(lmmc_sparse_chol_t));
    chol->n = n;
status = chol_allocate_indices(chol);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_sparse_residual_reorder(&csc, chol->perm);
    }
    if (status != LMMC_STATUS_OK) {
        goto fail;
    }
    for (size_t i = 0; i < n; ++i) chol->perm_inv[chol->perm[i]] = i;
    status = chol_allocate_factors(chol, a->nnz);
    if (status != LMMC_STATUS_OK) {
        goto fail;
    }
    chol->L_nnz = 0;
    if (csc_needs_free) {
        lmmc_sparse_destroy(&csc);
    }
    *out_chol = chol;
    return LMMC_STATUS_OK;
fail:
    lmmc_sparse_chol_destroy(chol);
    if (csc_needs_free) {
        lmmc_sparse_destroy(&csc);
    }
    return status;
}

static void chol_scatter_column(const lmmc_sparse_mat_t* csc,
    const lmmc_sparse_chol_t* chol, size_t j, lmmc_real_t* col_dense)
{
    size_t orig_col = chol->perm[j];
    memset(col_dense, 0, chol->n * sizeof(lmmc_real_t));
            size_t col_start = csc->row_ptr[orig_col];
            size_t col_end = csc->row_ptr[orig_col + 1];
            for (size_t p = col_start; p < col_end; p++) {
                size_t orig_row = csc->col_idx[p];
                size_t perm_row = chol->perm_inv[orig_row];
                if (perm_row >= j) {
                    col_dense[perm_row] += csc->values[p];
                }
            }
}

static lmmc_real_t chol_column_entry(const lmmc_sparse_chol_t* chol,
                                      size_t column, size_t row)
{
    for (size_t p = chol->L_col_ptr[column]; p < chol->L_col_ptr[column + 1]; ++p) {
        size_t current = chol->L_row_idx[p];
        if (current == row) {
            return chol->L_values[p];
        }
        if (current > row) {
            break;
        }
    }
    return 0.0;
}

static void chol_update_column(const lmmc_sparse_chol_t* chol, size_t j,
                                lmmc_real_t* col_dense)
{
        /** @brief 扣除 L 已分解列的贡献。 */
        for (size_t k = 0; k < j; k++) {
            size_t L_start = chol->L_col_ptr[k];
            size_t L_end = chol->L_col_ptr[k + 1];
            lmmc_real_t Ljk = chol_column_entry(chol, k, j);
            if (Ljk == 0.0) {
                continue;
            }

            /** @brief 对 i >= j 更新 col_dense[i] -= L(i,k) * L(j,k)。 */
            for (size_t p = L_start; p < L_end; p++) {
                size_t i = chol->L_row_idx[p];
                if (i >= j) {
                    col_dense[i] -= chol->L_values[p] * Ljk;
                }
            }
        }
}

static lmmc_status_t chol_store_column(lmmc_sparse_chol_t* chol, size_t j,
                                       const lmmc_real_t* col_dense)
{
    size_t n = chol->n;
    lmmc_status_t status;
            lmmc_real_t diag = col_dense[j];
            lmmc_real_t sqrt_diag;

            if (diag <= 0.0) {
                status = LMMC_STATUS_NOT_POSITIVE_DEFINITE;
                return status;
            }

            sqrt_diag = sqrt(diag);

            status = lmmc_sparse_factor_capacity(&chol->L_row_idx, &chol->L_values,
                                            &chol->L_capacity, chol->L_nnz);
            if (status != LMMC_STATUS_OK) {
                return status;
            }
            chol->L_row_idx[chol->L_nnz] = j;
            chol->L_values[chol->L_nnz] = sqrt_diag;
            chol->L_nnz++;

            for (size_t i = j + 1; i < n; i++) {
                if (col_dense[i] != 0.0) {
                    status = lmmc_sparse_factor_capacity(&chol->L_row_idx, &chol->L_values,
                                                    &chol->L_capacity, chol->L_nnz);
                    if (status != LMMC_STATUS_OK) {
                        return status;
                    }
                    chol->L_row_idx[chol->L_nnz] = i;
                    chol->L_values[chol->L_nnz] = col_dense[i] / sqrt_diag;
                    chol->L_nnz++;
                }
            }
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

    status = lmmc_sparse_direct_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) {
        return status;
    }

    col_dense = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    if (col_dense == NULL) {
        if (csc_needs_free) {
            lmmc_sparse_destroy(&csc);
        }
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    chol->L_nnz = 0;

    /** @brief 使用稠密列工作区执行左看 Cholesky 分解。 */
for (size_t j = 0; j < n; j++) {
    chol->L_col_ptr[j] = chol->L_nnz;
    chol_scatter_column(&csc, chol, j, col_dense);
    chol_update_column(chol, j, col_dense);
    status = chol_store_column(chol, j, col_dense);
    if (status != LMMC_STATUS_OK) {
        goto chol_cleanup;
    }
}
    chol->L_col_ptr[n] = chol->L_nnz;

chol_cleanup:
    if (col_dense) {
        lmmc_memory_free(col_dense);
    }
    if (csc_needs_free) {
        lmmc_sparse_destroy(&csc);
    }
    return status;
}

static lmmc_status_t chol_forward_solve(const lmmc_sparse_chol_t* chol,
                                        lmmc_real_t* work)
{
    size_t n = chol->n;
    /** @brief 前代求解 L * z = P*b。 */
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
            return LMMC_STATUS_NOT_POSITIVE_DEFINITE;
        }

        work[j] = work[j] / diag;

        for (size_t p = L_start; p < L_end; p++) {
            if (p == diag_idx) {
                continue;
            }
            size_t i = chol->L_row_idx[p];
            work[i] -= chol->L_values[p] * work[j];
        }
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t chol_backward_solve(const lmmc_sparse_chol_t* chol,
                                         lmmc_real_t* work)
{
    size_t n = chol->n;
    /** @brief 回代求解 L^T * y = z。 */
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
            return LMMC_STATUS_NOT_POSITIVE_DEFINITE;
        }

        for (size_t p = L_start; p < L_end; p++) {
            if (p == diag_idx) {
                continue;
            }
            size_t i = chol->L_row_idx[p];
            work[j] -= chol->L_values[p] * work[i];
        }
        work[j] = work[j] / diag;
    }
    return LMMC_STATUS_OK;
}

/**
 * @brief 使用残余度排序的 Cholesky 因子求解 A*x = b。
 *
 * P*A*P^T = L*L^T,其中 P 为残余度置换。
 */
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

    work = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    if (work == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /** @brief 应用置换：work = P * b。 */
    for (size_t i = 0; i < n; i++) {
        work[i] = b->data[chol->perm[i]];
    }

lmmc_status_t status = chol_forward_solve(chol, work);
if (status == LMMC_STATUS_OK) {
    status = chol_backward_solve(chol, work);
}
if (status != LMMC_STATUS_OK) {
    lmmc_memory_free(work);
    return status;
}
    /** @brief 还原置换：x = P^T * y。 */
    /** @brief perm[k] 为置换后位置 k 的原索引。 */
    for (size_t k = 0; k < n; k++) {
        x->data[chol->perm[k]] = work[k];
    }

    lmmc_memory_free(work);
    return LMMC_STATUS_OK;
}

void lmmc_sparse_chol_destroy(lmmc_sparse_chol_t* chol)
{
    if (chol == NULL) {
        return;
    }
    if (chol->perm) {
        lmmc_memory_free(chol->perm);
    }
    if (chol->perm_inv) {
        lmmc_memory_free(chol->perm_inv);
    }
    if (chol->L_col_ptr) {
        lmmc_memory_free(chol->L_col_ptr);
    }
    if (chol->L_row_idx) {
        lmmc_memory_free(chol->L_row_idx);
    }
    if (chol->L_values) {
        lmmc_memory_free(chol->L_values);
    }
    lmmc_memory_free(chol);
}
