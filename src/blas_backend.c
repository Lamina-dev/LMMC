/**
 * @file blas_backend.c
 * @brief BLAS dispatch implementation - delegates to external cblas_* functions.
 *
 * This file is only compiled when LMMC_USE_BLAS is defined via CMake.
 * It provides thin wrappers that convert LMMC's size_t-based interface
 * to the int-based CBLAS interface.
 */

#ifdef LMMC_USE_BLAS

#include "blas_backend.h"
#include <cblas.h>

/**
 * @brief Matrix-matrix multiply via CBLAS: C := alpha*A*B + beta*C
 *
 * Maps to cblas_dgemm with row-major layout and no transpose.
 * Converts size_t dimensions to int for the CBLAS interface.
 */
void lmmc_blas_dgemm(
    size_t m, size_t n, size_t k,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
    const lmmc_real_t* B, size_t ldb,
    lmmc_real_t beta,
    lmmc_real_t* C, size_t ldc)
{
    const int im  = (int)m;
    const int in  = (int)n;
    const int ik  = (int)k;
    const int ilda = (int)lda;
    const int ildb = (int)ldb;
    const int ildc = (int)ldc;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                im, in, ik,
                alpha,
                A, ilda,
                B, ildb,
                beta,
                C, ildc);
}

/**
 * @brief Vector axpy via CBLAS: y := alpha*x + y
 *
 * Converts size_t parameters to int for the CBLAS interface.
 */
void lmmc_blas_daxpy(
    size_t n, lmmc_real_t alpha,
    const lmmc_real_t* x, size_t incx,
    lmmc_real_t* y, size_t incy)
{
    const int in    = (int)n;
    const int iincx = (int)incx;
    const int iincy = (int)incy;

    cblas_daxpy(in, alpha, x, iincx, y, iincy);
}

/**
 * @brief Vector 2-norm via CBLAS: returns ||x||_2
 *
 * Converts size_t parameters to int for the CBLAS interface.
 */
lmmc_real_t lmmc_blas_dnrm2(
    size_t n, const lmmc_real_t* x, size_t incx)
{
    const int in    = (int)n;
    const int iincx = (int)incx;

    return cblas_dnrm2(in, x, iincx);
}

#endif /* LMMC_USE_BLAS */
