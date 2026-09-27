#ifndef LMMC_SPARSE_DIRECT_INTERNAL_H
#define LMMC_SPARSE_DIRECT_INTERNAL_H

#include "lmmc/sparse.h"

struct lmmc_sparse_lu_t {
    size_t n;
    size_t* col_perm;      /**< 残余度列置换：新索引 -> 原索引。 */
    size_t* row_perm;      /**< 部分主元法产生的行置换。 */
    size_t* L_col_ptr;
    size_t* U_col_ptr;
    size_t* L_row_idx;
    lmmc_real_t* L_values;
    size_t* U_row_idx;
    lmmc_real_t* U_values;
    size_t L_nnz;
    size_t U_nnz;
    size_t L_capacity;
    size_t U_capacity;
};


struct lmmc_sparse_chol_t {
    size_t n;
    size_t* perm;          /**< 残余度置换：新索引 -> 原索引。 */
    size_t* perm_inv;      /**< 逆置换：原索引 -> 新索引。 */
    size_t* L_col_ptr;
    size_t* L_row_idx;
    lmmc_real_t* L_values;
    size_t L_nnz;
    size_t L_capacity;
};
lmmc_status_t lmmc_sparse_direct_csc(const lmmc_sparse_mat_t* a,
    lmmc_sparse_mat_t* csc_out, int* needs_free);
lmmc_status_t lmmc_sparse_factor_capacity(size_t** idx, lmmc_real_t** vals,
    size_t* capacity, size_t needed);
lmmc_status_t lmmc_sparse_residual_reorder(const lmmc_sparse_mat_t* csc,
    size_t* perm);

#endif
