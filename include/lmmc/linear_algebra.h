#ifndef LMMC_LINEAR_ALGEBRA_H
#define LMMC_LINEAR_ALGEBRA_H

#include <stddef.h>
#include "lmmc/dense.h"

#ifdef __cplusplus
extern "C" {
#endif

lmmc_status_t lmmc_lu_decompose_inplace(lmmc_mat_t* a, size_t* pivots, size_t* out_swap_count);
lmmc_status_t lmmc_lu_solve(const lmmc_mat_t* lu, const size_t* pivots, const lmmc_vec_t* b, lmmc_vec_t* x);

lmmc_status_t lmmc_cholesky_decompose_inplace(lmmc_mat_t* a);
lmmc_status_t lmmc_cholesky_solve(const lmmc_mat_t* l, const lmmc_vec_t* b, lmmc_vec_t* x);

lmmc_status_t lmmc_qr_decompose_inplace(lmmc_mat_t* a, lmmc_real_t* tau, size_t tau_size);
lmmc_status_t lmmc_qr_solve(const lmmc_mat_t* qr, const lmmc_real_t* tau, const lmmc_vec_t* b, lmmc_vec_t* x);

#ifdef __cplusplus
}
#endif

#endif
