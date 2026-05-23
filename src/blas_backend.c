/**
 * @file blas_backend.c
 * @brief 外部 BLAS 后端封装实现。
 *
 * 仅当编译时定义 @c LMMC_USE_BLAS 时启用。
 */
#ifdef LMMC_USE_BLAS

#include "blas_backend.h"
#include <cblas.h>


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


lmmc_real_t lmmc_blas_dnrm2(
    size_t n, const lmmc_real_t* x, size_t incx)
{
    const int in    = (int)n;
    const int iincx = (int)incx;

    return cblas_dnrm2(in, x, iincx);
}

#endif
