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
 * @brief DGEMM with transpose support.
 *
 * @p transA / @p transB: 0 = no transpose, non-zero = transpose.
 * Row-major convention.
 */
void lmmc_blas_dgemm_ex(
    int transA, int transB,
    size_t m, size_t n, size_t k,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
    const lmmc_real_t* B, size_t ldb,
    lmmc_real_t beta,
    lmmc_real_t* C, size_t ldc
);

/**
 * @internal
 * @brief DGEMV 封装：@c y := alpha * op(A) * x + beta * y 。
 *
 * @p trans: 'N' = no transpose, 'T' = transpose.
 * Row-major convention.
 */
void lmmc_blas_dgemv(
    char trans, size_t m, size_t n,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
    const lmmc_real_t* x, size_t incx,
    lmmc_real_t beta,
    lmmc_real_t* y, size_t incy
);

/**
 * @internal
 * @brief DDOT 封装：返回向量内积。
 */
lmmc_real_t lmmc_blas_ddot(
    size_t n, const lmmc_real_t* x, size_t incx,
    const lmmc_real_t* y, size_t incy
);

/**
 * @internal
 * @brief DSCAL 封装：@c x := alpha * x 。
 */
void lmmc_blas_dscal(
    size_t n, lmmc_real_t alpha,
    lmmc_real_t* x, size_t incx
);

/**
 * @internal
 * @brief DCOPY 封装：@c y := x 。
 */
void lmmc_blas_dcopy(
    size_t n, const lmmc_real_t* x, size_t incx,
    lmmc_real_t* y, size_t incy
);

/**
 * @internal
 * @brief DTRSM 封装：求解三角矩阵方程。
 *
 * Solves op(A) * X = alpha * B  or  X * op(A) = alpha * B
 * where A is triangular.
 *
 * @p side: 'L' = left, 'R' = right.
 * @p uplo: 'U' = upper, 'L' = lower.
 * @p trans: 'N' = no transpose, 'T' = transpose.
 * @p diag: 'N' = non-unit, 'U' = unit diagonal.
 */
void lmmc_blas_dtrsm(
    char side, char uplo, char trans, char diag,
    size_t m, size_t n,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
    lmmc_real_t* B, size_t ldb
);

/**
 * @internal
 * @brief DSYRK 封装：对称秩-k 更新。
 *
 * C := alpha * A * A^T + beta * C  (trans='N')
 * C := alpha * A^T * A + beta * C  (trans='T')
 * 仅更新 C 的上三角或下三角部分。
 *
 * @p uplo: 'U' = upper, 'L' = lower.
 * @p trans: 'N' = no transpose, 'T' = transpose.
 * @p n: C 的阶数。
 * @p k: A 的列数 (trans='N') 或行数 (trans='T')。
 */
void lmmc_blas_dsyrk(
    char uplo, char trans,
    size_t n, size_t k,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
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

/**
 * LAPACK Bridges
 *
 * 返回值为 LAPACK info 参数：
 *   info = 0  成功
 *   info < 0  第 |info| 个参数非法（维度验证失败）
 *   info > 0  算法特定的失败指示
 */

/**
 * @internal
 * @brief DGETRF 封装：LU 分解（带行主元选取）。
 *
 * A = P * L * U，其中 P 为置换矩阵。
 * @p ipiv 输出长度为 min(m,n) 的主元索引数组。
 * @return LAPACK info 值。
 */
int lmmc_lapack_getrf(int m, int n, double* A, int lda, int* ipiv);

/**
 * @internal
 * @brief DGETRS 封装：基于 LU 分解求解线性方程组。
 *
 * 求解 op(A) * X = B，其中 A 已由 getrf 分解。
 * @p trans: 'N' = A*X=B, 'T' = A^T*X=B.
 * @return LAPACK info 值。
 */
int lmmc_lapack_getrs(char trans, int n, int nrhs, const double* A, int lda,
                      const int* ipiv, double* B, int ldb);

/**
 * @internal
 * @brief DPOTRF 封装：Cholesky 分解。
 *
 * 计算对称正定矩阵 A 的 Cholesky 分解 A = L*L^T 或 A = U^T*U。
 * @p uplo: 'U' = 上三角, 'L' = 下三角。
 * @return LAPACK info 值。
 */
int lmmc_lapack_potrf(char uplo, int n, double* A, int lda);

/**
 * @internal
 * @brief DPOTRS 封装：基于 Cholesky 分解求解线性方程组。
 *
 * 求解 A * X = B，其中 A 已由 potrf 分解。
 * @p uplo: 'U' = 上三角, 'L' = 下三角。
 * @return LAPACK info 值。
 */
int lmmc_lapack_potrs(char uplo, int n, int nrhs, const double* A, int lda,
                      double* B, int ldb);

/**
 * @internal
 * @brief DGEQRF 封装：QR 分解。
 *
 * 计算 m×n 矩阵 A 的 QR 分解 A = Q * R。
 * @p tau 输出长度为 min(m,n) 的 Householder 标量数组。
 * @return LAPACK info 值。
 */
int lmmc_lapack_geqrf(int m, int n, double* A, int lda, double* tau);

/**
 * @internal
 * @brief DGESVD 封装：奇异值分解。
 *
 * 计算 m×n 矩阵 A 的 SVD：A = U * Σ * V^T。
 * @p jobu: 'A' = 全部 U, 'S' = 前 min(m,n) 列, 'N' = 不计算。
 * @p jobvt: 'A' = 全部 V^T, 'S' = 前 min(m,n) 行, 'N' = 不计算。
 * @return LAPACK info 值。
 */
int lmmc_lapack_gesvd(char jobu, char jobvt, int m, int n, double* A, int lda,
                      double* S, double* U, int ldu, double* VT, int ldvt);

#endif /* LMMC_USE_BLAS */

#endif /* LMMC_BLAS_BACKEND_H */
