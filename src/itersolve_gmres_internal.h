#ifndef LMMC_ITERSOLVE_GMRES_INTERNAL_H
#define LMMC_ITERSOLVE_GMRES_INTERNAL_H

#include "itersolve_internal.h"

typedef struct {
    lmmc_vec_t r, z, w, ax, x_base, x_trial;
    lmmc_vec_t* basis;
    lmmc_real_t* h;
    lmmc_real_t* cs;
    lmmc_real_t* sn;
    lmmc_real_t* g;
    lmmc_real_t* y;
    size_t restart;
    size_t h_count;
    lmmc_real_t norm;
    lmmc_real_t threshold;
} lmmc_gmres_state_t;

lmmc_status_t lmmc_gmres_workspace(size_t n, size_t restart,
    lmmc_gmres_state_t* s);
lmmc_status_t lmmc_gmres_arnoldi(const lmmc_sparse_mat_t* a,
    const lmmc_precond_t* precond, lmmc_gmres_state_t* s,
    size_t j, lmmc_real_t* h_next);
lmmc_status_t lmmc_gmres_rotate(lmmc_gmres_state_t* s, size_t j, int* breakdown);
lmmc_status_t lmmc_gmres_back_substitute(lmmc_gmres_state_t* s, size_t dim);
lmmc_status_t lmmc_gmres_trial(lmmc_gmres_state_t* s, size_t j);
lmmc_status_t lmmc_gmres_normalize(lmmc_vec_t* out,
    const lmmc_vec_t* in, lmmc_real_t norm);

#endif
