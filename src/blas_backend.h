/**
 * @file blas_backend.h
 * @brief 可选的外部 BLAS 后端封装（仅在定义 @c LMMC_USE_BLAS 时启用）。
 *
 * 当连接外部 BLAS 时，库内的稠密矩阵核心运算会调用本文件声明的封装函数。
 *
 * @internal
 */
#ifndef LMMC_BLAS_BACKEND_H
#define LMMC_BLAS_BACKEND_H

#include "lmmc/config.h"
#include <stddef.h>

#ifdef LMMC_USE_BLAS

/**
 * @internal
 * @brief DGEMM 封装：@c C := alpha * A * B + beta * C 。
 *
 * 行优先约定， @p lda / @p ldb / @p ldc 为各矩阵的行步距。
 */
void lmmc_blas_dgemm(
    size_t m, size_t n, size_t k,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
    const lmmc_real_t* B, size_t ldb,
    lmmc_real_t beta,
    lmmc_real_t* C, size_t ldc
);

/**
 * @internal
 * @brief DAXPY 封装：@c y := alpha * x + y 。
 */
void lmmc_blas_daxpy(
    size_t n, lmmc_real_t alpha,
    const lmmc_real_t* x, size_t incx,
    lmmc_real_t* y, size_t incy
);

/** @internal @brief DNRM2 封装：返回向量 2-范数。 */
lmmc_real_t lmmc_blas_dnrm2(
    size_t n, const lmmc_real_t* x, size_t incx
);

#endif

#endif
