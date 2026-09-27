#ifndef LMMC_EIGEN_GENERAL_INTERNAL_H
#define LMMC_EIGEN_GENERAL_INTERNAL_H

#include "lmmc/eigen.h"

void lmmc_eigen_copy_scaled(const lmmc_mat_t* a, lmmc_real_t scale, lmmc_mat_t* result);
lmmc_status_t lmmc_eigen_schur_reduce(lmmc_mat_t* h, lmmc_mat_t* q);
void lmmc_eigen_values_2x2(lmmc_real_t a00, lmmc_real_t a01,
    lmmc_real_t a10, lmmc_real_t a11, lmmc_real_t* real0, lmmc_real_t* imag0,
    lmmc_real_t* real1, lmmc_real_t* imag1);
lmmc_status_t lmmc_eigen_restore_scale(lmmc_vec_t* real, lmmc_vec_t* imag,
                                       lmmc_real_t scale);
void lmmc_eigen_vectors_2x2(const lmmc_mat_t* a, lmmc_real_t scale,
                           lmmc_eigen_gen_full_result_t* result);
lmmc_status_t lmmc_eigen_inverse_vectors(const lmmc_mat_t* a,
                                         lmmc_eigen_gen_full_result_t* result);

#endif
