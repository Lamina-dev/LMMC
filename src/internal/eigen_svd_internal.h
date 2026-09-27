#ifndef LMMC_EIGEN_SVD_INTERNAL_H
#define LMMC_EIGEN_SVD_INTERNAL_H

#include "lmmc/eigen.h"

lmmc_status_t lmmc_svd_bidiagonalize(const lmmc_mat_t *a,
                                   lmmc_real_t input_scale,
                                   lmmc_real_t *d, lmmc_real_t *e,
                                   lmmc_mat_t *u, lmmc_mat_t *v);
lmmc_status_t lmmc_svd_diagonalize(lmmc_real_t *d, lmmc_real_t *e,
                                 size_t n, lmmc_mat_t *u, lmmc_mat_t *v);
lmmc_status_t lmmc_svd_tall(const lmmc_mat_t *a, lmmc_real_t input_scale,
                          lmmc_svd_result_t *result);

#endif
