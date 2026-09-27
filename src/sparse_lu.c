#include <string.h>
#include <math.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/sparse.h"
#include "sparse_direct_internal.h"

static lmmc_status_t lu_allocate_factors(lmmc_sparse_lu_t* lu, size_t nnz)
{
    size_t n = lu->n;
        size_t initial_capacity = (nnz > 64) ? nnz : 64;
        /** @brief 为预期填充预留因子容量。 */
        if (initial_capacity < n) {
            initial_capacity = n;
        }

        lu->L_row_idx = (size_t*)lmmc_memory_alloc_array(initial_capacity, sizeof(size_t));
        lu->L_values = (lmmc_real_t*)lmmc_memory_alloc_array(initial_capacity, sizeof(lmmc_real_t));
        lu->U_row_idx = (size_t*)lmmc_memory_alloc_array(initial_capacity, sizeof(size_t));
        lu->U_values = (lmmc_real_t*)lmmc_memory_alloc_array(initial_capacity, sizeof(lmmc_real_t));

if (!lu->L_row_idx || !lu->L_values || !lu->U_row_idx || !lu->U_values) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    lu->L_capacity = initial_capacity;
    lu->U_capacity = initial_capacity;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lu_allocate_indices(lmmc_sparse_lu_t* lu)
{
    size_t n = lu->n;
    lu->col_perm = (size_t*)lmmc_memory_alloc_array(n, sizeof(size_t));
    lu->row_perm = (size_t*)lmmc_memory_alloc_array(n, sizeof(size_t));
    lu->L_col_ptr = (size_t*)lmmc_memory_alloc_array_plus(n, 1, sizeof(size_t));
    lu->U_col_ptr = (size_t*)lmmc_memory_alloc_array_plus(n, 1, sizeof(size_t));

if (!lu->col_perm || !lu->row_perm || !lu->L_col_ptr || !lu->U_col_ptr) {
    return LMMC_STATUS_ALLOCATION_FAILED;
}
    memset(lu->L_col_ptr, 0, (n + 1) * sizeof(size_t));
    memset(lu->U_col_ptr, 0, (n + 1) * sizeof(size_t));

    /** @brief 行置换从恒等置换开始，由部分主元法更新。 */
    for (size_t i = 0; i < n; i++) lu->row_perm[i] = i;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_lu_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t** out_lu)
{
    lmmc_sparse_lu_t* lu = NULL;
    lmmc_sparse_mat_t csc;
    int csc_needs_free = 0;
    size_t n;
    lmmc_status_t status;

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

    status = lmmc_sparse_direct_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) {
        return status;
    }

    lu = (lmmc_sparse_lu_t*)lmmc_memory_alloc(sizeof(lmmc_sparse_lu_t));
    if (lu == NULL) {
        if (csc_needs_free) {
            lmmc_sparse_destroy(&csc);
        }
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(lu, 0, sizeof(lmmc_sparse_lu_t));
    lu->n = n;
status = lu_allocate_indices(lu);
if (status == LMMC_STATUS_OK) {
    status = lmmc_sparse_residual_reorder(&csc, lu->col_perm);
}
if (status == LMMC_STATUS_OK) {
    status = lu_allocate_factors(lu, a->nnz);
}
if (status != LMMC_STATUS_OK) {
    lmmc_sparse_lu_destroy(lu);
    if (csc_needs_free) {
        lmmc_sparse_destroy(&csc);
    }
    return status;
}
    lu->L_nnz = 0;
    lu->U_nnz = 0;

    if (csc_needs_free) {
        lmmc_sparse_destroy(&csc);
    }
    *out_lu = lu;
    return LMMC_STATUS_OK;
}

static void lu_scatter_column(const lmmc_sparse_mat_t* csc, size_t orig_col,
    const size_t* piv_inv, lmmc_real_t* col_dense)
{
    memset(col_dense, 0, csc->rows * sizeof(lmmc_real_t));
            size_t col_start = csc->row_ptr[orig_col];
            size_t col_end = csc->row_ptr[orig_col + 1];
            for (size_t p = col_start; p < col_end; p++) {
                size_t row = csc->col_idx[p];
                size_t prow = piv_inv[row]; /**< 置换后的行索引。 */
                col_dense[prow] += csc->values[p];
            }
}

static void lu_update_column(const lmmc_sparse_lu_t* lu, size_t j,
                              lmmc_real_t* col_dense)
{
        /** @brief 用已分解且当前主元分量非零的 L 列更新工作区。 */
        for (size_t k = 0; k < j; k++) {
            if (col_dense[k] == 0.0) {
                continue;
            }

            lmmc_real_t ukj = col_dense[k];

            /** @brief 从 col_dense 扣除 L(:,k) * ukj。 */
            size_t L_start = lu->L_col_ptr[k];
            size_t L_end = lu->L_col_ptr[k + 1];
            for (size_t p = L_start; p < L_end; p++) {
                size_t row = lu->L_row_idx[p];
                lmmc_real_t old_val = col_dense[row];
                col_dense[row] = old_val - lu->L_values[p] * ukj;
            }
        }
}

static void lu_swap_previous_rows(lmmc_sparse_lu_t* lu, size_t j,
                                   size_t pivot_row)
{
                /** @brief 同步交换已存储 L 列中的行。 */
                for (size_t k = 0; k < j; k++) {
                    size_t L_start = lu->L_col_ptr[k];
                    size_t L_end = lu->L_col_ptr[k + 1];
                    size_t idx_j = (size_t)-1;
                    size_t idx_p = (size_t)-1;
                    for (size_t p = L_start; p < L_end; p++) {
                        if (lu->L_row_idx[p] == j) {
                            idx_j = p;
                        }
                        if (lu->L_row_idx[p] == pivot_row) {
                            idx_p = p;
                        }
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

static lmmc_status_t lu_pivot_column(lmmc_sparse_lu_t* lu, size_t j,
    size_t* piv_inv, lmmc_real_t* col_dense)
{
    size_t n = lu->n;
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
                return LMMC_STATUS_SINGULAR_MATRIX;
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

lu_swap_previous_rows(lu, j, pivot_row);
    }
    return LMMC_STATUS_OK;
}
static lmmc_status_t lu_store_column(lmmc_sparse_lu_t* lu, size_t j,
                                     const lmmc_real_t* col_dense)
{
    size_t n = lu->n;
    lmmc_status_t status;
        /** @brief 存储 U 的第 0..j 行。 */
        lu->U_col_ptr[j] = lu->U_nnz;
        for (size_t i = 0; i <= j; i++) {
            if (col_dense[i] != 0.0) {
                status = lmmc_sparse_factor_capacity(&lu->U_row_idx, &lu->U_values,
                                                &lu->U_capacity, lu->U_nnz);
                if (status != LMMC_STATUS_OK) {
                    return status;
                }
                lu->U_row_idx[lu->U_nnz] = i;
                lu->U_values[lu->U_nnz] = col_dense[i];
                lu->U_nnz++;
            }
        }

        /** @brief 存储除以对角元的 L 第 j+1..n-1 行；本轮开始时已设置 L_col_ptr[j]。 */
        {
            lmmc_real_t diag = col_dense[j];
            for (size_t i = j + 1; i < n; i++) {
                if (col_dense[i] != 0.0) {
                    status = lmmc_sparse_factor_capacity(&lu->L_row_idx, &lu->L_values,
                                                    &lu->L_capacity, lu->L_nnz);
                    if (status != LMMC_STATUS_OK) {
                        return status;
                    }
                    lu->L_row_idx[lu->L_nnz] = i;
                    lu->L_values[lu->L_nnz] = col_dense[i] / diag;
                    lu->L_nnz++;
                }
            }
        }
    return LMMC_STATUS_OK;
}

/**
 * @brief 使用稠密列工作区执行左看稀疏 LU 分解。
 *
 * 每个置换后的输入列先散布到工作区,再由已存储的 L 列更新、选择主元,
 * 最后压缩为稀疏 L 和 U。
 */
static lmmc_status_t lu_factor_columns(const lmmc_sparse_mat_t* csc,
    lmmc_sparse_lu_t* lu, lmmc_real_t* col_dense, size_t* piv_inv)
{
    size_t n = lu->n;
    for (size_t i = 0; i < n; i++) {
        piv_inv[i] = i;
    }
    lu->L_nnz = 0;
    lu->U_nnz = 0;
    for (size_t i = 0; i < n; i++) {
        lu->row_perm[i] = i;
    }
    for (size_t j = 0; j < n; j++) {
        /** @brief 左看更新前封闭上一列的存储区间。 */
        lu->L_col_ptr[j] = lu->L_nnz;
        lu_scatter_column(csc, lu->col_perm[j], piv_inv, col_dense);
        lu_update_column(lu, j, col_dense);
        lmmc_status_t status = lu_pivot_column(lu, j, piv_inv, col_dense);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
        status = lu_store_column(lu, j, col_dense);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
    }
    lu->L_col_ptr[n] = lu->L_nnz;
    lu->U_col_ptr[n] = lu->U_nnz;
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

    status = lmmc_sparse_direct_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) {
        return status;
    }

    /** @brief 稠密列工作区与逆行置换。 */
    col_dense = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    piv_inv = (size_t*)lmmc_memory_alloc_array(n, sizeof(size_t));

    if (!col_dense || !piv_inv) {
        status = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }

    status = lu_factor_columns(&csc, lu, col_dense, piv_inv);

cleanup:
    if (col_dense) {
        lmmc_memory_free(col_dense);
    }
    if (piv_inv) {
        lmmc_memory_free(piv_inv);
    }
    if (csc_needs_free) {
        lmmc_sparse_destroy(&csc);
    }
    return status;
}

static lmmc_status_t lu_triangular_solve(const lmmc_sparse_lu_t* lu,
                                        lmmc_real_t* work)
{
    size_t n = lu->n;
    /** @brief 前代求解 L * z = P*b。 */
    for (size_t j = 0; j < n; j++) {
        size_t L_start = lu->L_col_ptr[j];
        size_t L_end = lu->L_col_ptr[j + 1];
        for (size_t p = L_start; p < L_end; p++) {
            size_t i = lu->L_row_idx[p];
            work[i] -= lu->L_values[p] * work[j];
        }
    }
    /** @brief 回代求解 U * y = z。 */
    for (size_t jj = 0; jj < n; jj++) {
        size_t j = n - 1 - jj;
        size_t U_start = lu->U_col_ptr[j];
        size_t U_end = lu->U_col_ptr[j + 1];
        lmmc_real_t diag = 0.0;
        for (size_t p = U_start; p < U_end; p++) {
            if (lu->U_row_idx[p] == j) {
                diag = lu->U_values[p];
                break;
            }
        }
        if (diag == 0.0) {
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
    return LMMC_STATUS_OK;
}

/**
 * @brief 使用残余度排序的 LU 因子求解 A*x = b。
 *
 * 分解满足 P*A*Q = L*U,其中 P 为部分主元行置换,Q 为残余度列置换。
 */
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

    work = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    if (work == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /** @brief 应用行置换：work = P * b。 */
    for (size_t i = 0; i < n; i++) {
        work[i] = b->data[lu->row_perm[i]];
    }

    lmmc_status_t status = lu_triangular_solve(lu, work);
    if (status != LMMC_STATUS_OK) {
        lmmc_memory_free(work);
        return status;
    }

    /** @brief 还原列置换：x = Q * y。 */
    /** @brief col_perm[k] 为置换后第 k 列的原列索引。 */
    for (size_t k = 0; k < n; k++) {
        x->data[lu->col_perm[k]] = work[k];
    }

    lmmc_memory_free(work);
    return LMMC_STATUS_OK;
}

void lmmc_sparse_lu_destroy(lmmc_sparse_lu_t* lu)
{
    if (lu == NULL) {
        return;
    }
    if (lu->col_perm) {
        lmmc_memory_free(lu->col_perm);
    }
    if (lu->row_perm) {
        lmmc_memory_free(lu->row_perm);
    }
    if (lu->L_col_ptr) {
        lmmc_memory_free(lu->L_col_ptr);
    }
    if (lu->U_col_ptr) {
        lmmc_memory_free(lu->U_col_ptr);
    }
    if (lu->L_row_idx) {
        lmmc_memory_free(lu->L_row_idx);
    }
    if (lu->L_values) {
        lmmc_memory_free(lu->L_values);
    }
    if (lu->U_row_idx) {
        lmmc_memory_free(lu->U_row_idx);
    }
    if (lu->U_values) {
        lmmc_memory_free(lu->U_values);
    }
    lmmc_memory_free(lu);
}
