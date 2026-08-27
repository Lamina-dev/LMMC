/**
 * @file eigen_internal.h
 * @brief 特征值 / SVD 模块内部共享的 Householder 与 Givens 辅助接口（仅源文件可见）。
 *
 * @internal
 */
#ifndef LMMC_EIGEN_INTERNAL_H
#define LMMC_EIGEN_INTERNAL_H

#include <stddef.h>
#include "lmmc/config.h"
#include "lmmc/dense.h"

#define EIGEN_MAX_ITER 30
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])

/* Apply Householder reflector helpers shared by symmetric eigensolver,
 * general eigensolver and SVD code paths. Defined once in eigen_internal.c;
 * deliberately not part of the public API. */

void householder_make(lmmc_real_t *x, size_t len, lmmc_real_t *tau_out,
                      lmmc_real_t *beta_out);
void householder_apply_left(lmmc_mat_t *M, size_t i0, size_t len,
                            size_t j0, size_t j1,
                            const lmmc_real_t *v, lmmc_real_t tau);
void householder_apply_right(lmmc_mat_t *M, size_t i0, size_t i1,
                             size_t j0, size_t len,
                             const lmmc_real_t *v, lmmc_real_t tau);

/* Givens rotation helpers shared by the SVD code path. */

void givens_right(lmmc_mat_t *M, size_t rows, size_t p, size_t q,
                  lmmc_real_t c, lmmc_real_t s);
void givens_compute(lmmc_real_t a, lmmc_real_t b,
                    lmmc_real_t *c, lmmc_real_t *s, lmmc_real_t *r);

#endif /* LMMC_EIGEN_INTERNAL_H */
