#ifndef LMMC_BLAS_BACKEND_H
#define LMMC_BLAS_BACKEND_H

#include "lmmc/config.h"
#include <stddef.h>

/* 当 LMMC_USE_BLAS 定义时，委托给外部 BLAS */
#ifdef LMMC_USE_BLAS

void lmmc_blas_dgemm(
    size_t m, size_t n, size_t k,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
    const lmmc_real_t* B, size_t ldb,
    lmmc_real_t beta,
    lmmc_real_t* C, size_t ldc
);

void lmmc_blas_daxpy(
    size_t n, lmmc_real_t alpha,
    const lmmc_real_t* x, size_t incx,
    lmmc_real_t* y, size_t incy
);

lmmc_real_t lmmc_blas_dnrm2(
    size_t n, const lmmc_real_t* x, size_t incx
);

#endif /* LMMC_USE_BLAS */

#endif /* LMMC_BLAS_BACKEND_H */
