/** @file dense_solvers.h */
#ifndef LMMC_DENSE_SOLVERS_H
#define LMMC_DENSE_SOLVERS_H

#include "lmmc/dense_types.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 通过 LU 分解计算方阵行列式。
 *
 * 使用临时工作空间，对 1x1 和 2x2 矩阵采用专用快速路径。
 *
 * @param[in]  a       输入方阵，保持不变。
 * @param[out] out_det 输出行列式，奇异矩阵返回 0。
 * @return
 * - ::LMMC_STATUS_OK - 成功，包括奇异矩阵。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或矩阵非方阵。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 临时内存分配失败。
 * @note 临时工作空间在返回前释放。
 */
lmmc_status_t lmmc_mat_det(const lmmc_mat_t* a, lmmc_real_t* out_det);

/**
 * @brief 计算方阵的逆矩阵。
 *
 * LU 分解后逐列求解 @f$A \cdot \text{col}_i = e_i@f$。
 * 奇异矩阵返回 ::LMMC_STATUS_SINGULAR_MATRIX，且 @p A_inv 保持不变。
 *
 * @param[in]  A     输入方阵。
 * @param[out] A_inv 输出逆矩阵，须已创建且与 @p A 同维。
 * @return ::LMMC_STATUS_OK 表示成功。
 */
lmmc_status_t lmmc_mat_inv(const lmmc_mat_t* A, lmmc_mat_t* A_inv);

/**
 * @brief 求解三角系统 @f$T x = b@f$。
 *
 * @param[in]  T         上三角或下三角矩阵。
 * @param[in]  upper     非零表示上三角，零表示下三角。
 * @param[in]  diag_unit 非零时将对角线视为 1。
 * @param[in]  b         右端向量。
 * @param[out] x         解向量，须已创建且与 @p b 同长。
 * @return ::LMMC_STATUS_OK 表示成功；对角线为零返回 ::LMMC_STATUS_SINGULAR_MATRIX。
 */
lmmc_status_t lmmc_solve_triangular(const lmmc_mat_t* T, int upper, int diag_unit,
    const lmmc_vec_t* b, lmmc_vec_t* x);

/**
 * @brief 通过 SVD 计算矩阵的数值秩。
 *
 * 用 ::lmmc_svd 统计大于 @p tol 的奇异值个数。
 * tol <= 0 时采用 max(rows, cols) * LMMC_REAL_EPSILON * sigma_0，
 * 其中 sigma_0 为最大奇异值。
 *
 * @param[in]  a        输入矩阵，保持不变。
 * @param[in]  tol      奇异值截断容差，<= 0 时使用默认值。
 * @param[out] out_rank 输出秩值。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 * - ::LMMC_STATUS_CONVERGENCE_FAILED - SVD 迭代未收敛。
 * @note 临时 SVD 工作空间在返回前释放。
 */
lmmc_status_t lmmc_mat_rank(const lmmc_mat_t* a, lmmc_real_t tol, size_t* out_rank);

/**
 * @brief 计算方阵整数幂 @f$\text{out} = A^n@f$。
 *
 * 二进制快速幂需要 O(log|n|) 次乘法。
 * n=0 生成单位矩阵，n=1 复制输入，n<0 对逆矩阵求 |n| 次幂。
 *
 * @param[in]  a   输入方阵，保持不变。
 * @param[in]  n   整数指数，可为负、零或正。
 * @param[out] out 输出矩阵，须已创建且与 @p a 同维（nxn），内容被覆写。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL、矩阵非方阵或 out 维度不匹配。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 临时内存分配失败。
 * - ::LMMC_STATUS_SINGULAR_MATRIX - n<0 且 A 奇异。
 * @note 迭代使用两个 nxn 临时矩阵，返回前释放。
 */
lmmc_status_t lmmc_mat_pow(const lmmc_mat_t* a, int n, lmmc_mat_t* out);

/**
 * @brief 计算矩阵右除 @f$X = B \cdot A^{-1}@f$。
 *
 * 对 A^T 进行 LU 分解，逐列求解 @f$A^T \cdot X^T = B^T@f$。
 *
 * @param[in]  B 被除矩阵（mxn），列数须等于 A 的行数，保持不变。
 * @param[in]  A 除数方阵（nxn），保持不变。
 * @param[out] X 输出矩阵（mxn），须已创建为 (B->rows x A->cols)，内容被覆写。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL、A 非方阵或维度不匹配。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 临时内存分配失败。
 * - ::LMMC_STATUS_SINGULAR_MATRIX - A 奇异。
 * @note 临时工作矩阵在返回前释放。
 */
lmmc_status_t lmmc_mat_rdiv(const lmmc_mat_t* B, const lmmc_mat_t* A, lmmc_mat_t* X);

#ifdef __cplusplus
}
#endif

#endif
