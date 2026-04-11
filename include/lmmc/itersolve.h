#ifndef LMMC_ITERSOLVE_H
#define LMMC_ITERSOLVE_H

#include <stddef.h>
#include "lmmc/numeric.h"
#include "lmmc/precond.h"
#include "lmmc/sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double abs_tol;
    double rel_tol;
    size_t max_iter;
    size_t restart;
    int verbose;
} lmmc_itersolve_config_t;

typedef struct {
    int converged;
    size_t num_iter;
    double initial_residual_norm;
    double final_residual_norm;
} lmmc_itersolve_result_t;

lmmc_status_t lmmc_itersolve_default_config(size_t problem_size, lmmc_itersolve_config_t* out_cfg);

lmmc_status_t lmmc_cg_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
);

lmmc_status_t lmmc_bicgstab_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
);

lmmc_status_t lmmc_gmres_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
);

#ifdef __cplusplus
}
#endif

#endif
