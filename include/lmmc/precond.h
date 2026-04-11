#ifndef LMMC_PRECOND_H
#define LMMC_PRECOND_H

#include <stddef.h>
#include "lmmc/dense.h"
#include "lmmc/sparse.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LMMC_PRECOND_NONE = 0,
    LMMC_PRECOND_JACOBI = 1,
    LMMC_PRECOND_ILU0 = 2,
    LMMC_PRECOND_ILUT = 3
} lmmc_precond_type_t;

typedef struct {
    lmmc_precond_type_t type;
    size_t size;
    void* impl;
    int owns_data;
} lmmc_precond_t;

lmmc_status_t lmmc_precond_create_none(size_t size, lmmc_precond_t* out_precond);
lmmc_status_t lmmc_precond_create_jacobi(const lmmc_sparse_mat_t* a, lmmc_precond_t* out_precond);
lmmc_status_t lmmc_precond_create_ilu0(const lmmc_sparse_mat_t* a, lmmc_precond_t* out_precond);
lmmc_status_t lmmc_precond_create_ilut(
    const lmmc_sparse_mat_t* a,
    double drop_tol,
    size_t max_fill_per_row,
    lmmc_precond_t* out_precond
);
lmmc_status_t lmmc_precond_apply(const lmmc_precond_t* precond, const lmmc_vec_t* rhs, lmmc_vec_t* out);
void lmmc_precond_destroy(lmmc_precond_t* precond);

#ifdef __cplusplus
}
#endif

#endif
