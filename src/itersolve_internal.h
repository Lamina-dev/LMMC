#ifndef LMMC_ITERSOLVE_INTERNAL_H
#define LMMC_ITERSOLVE_INTERNAL_H

#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/itersolve.h"

/** @brief 所有视图借用求解过程统一分配的存储区，最后一次日志回调后释放。 */
lmmc_status_t lmmc_itersolve_workspace(lmmc_vec_t* const* vectors,
    const size_t* sizes, size_t count, lmmc_real_t** storage);
lmmc_status_t lmmc_vec_norm2_checked(const lmmc_vec_t* v, lmmc_real_t* norm);
lmmc_status_t lmmc_vec_dot_checked(const lmmc_vec_t* a, const lmmc_vec_t* b,
    lmmc_real_t* dot);
lmmc_status_t lmmc_apply_precond_or_identity(const lmmc_precond_t* precond,
    const lmmc_vec_t* rhs, lmmc_vec_t* out);
void lmmc_itersolve_do_log(const lmmc_itersolve_config_t* cfg,
    size_t iter, lmmc_real_t residual);
lmmc_status_t lmmc_itersolve_validate(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg, lmmc_vec_t* x,
    lmmc_itersolve_config_t* local, lmmc_itersolve_result_t* result);
lmmc_status_t lmmc_itersolve_dispatch_config(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_itersolve_config_t* cfg,
    const lmmc_vec_t* x, lmmc_itersolve_config_t* local);
lmmc_status_t lmmc_itersolve_matvec(const lmmc_sparse_mat_t* a,
    const lmmc_itersolve_config_t* cfg, const lmmc_vec_t* x, lmmc_vec_t* y);
lmmc_status_t lmmc_itersolve_transpose(const lmmc_sparse_mat_t* a,
    const lmmc_itersolve_config_t* cfg, const lmmc_vec_t* x, lmmc_vec_t* y);
void lmmc_itersolve_result_init(lmmc_itersolve_result_t* result);
void lmmc_itersolve_initial(lmmc_itersolve_result_t* result, lmmc_real_t norm);
void lmmc_itersolve_finish(lmmc_itersolve_result_t* result,
    int converged, size_t iterations, lmmc_real_t norm);
lmmc_status_t lmmc_itersolve_residual(const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b, const lmmc_vec_t* x, lmmc_vec_t* ax,
    lmmc_vec_t* r);
lmmc_status_t lmmc_itersolve_threshold(const lmmc_itersolve_config_t* cfg,
    const lmmc_vec_t* b, const lmmc_vec_t* r, lmmc_real_t* norm,
    lmmc_real_t* threshold);

#endif
