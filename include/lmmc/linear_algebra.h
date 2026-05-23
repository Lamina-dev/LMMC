/**
 * @file linear_algebra.h
 * @brief 稠密矩阵直接分解接口（LU、Cholesky、QR）。
 *
 * 三组接口均采用 in-place + 辅助数据 的形式，先分解再求解，
 * 与 LAPACK 风格一致；分解结果会覆盖输入矩阵。
 */
#ifndef LMMC_LINEAR_ALGEBRA_H
#define LMMC_LINEAR_ALGEBRA_H

#include <stddef.h>
#include "lmmc/dense.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 就地 LU 分解（带部分主元）：@f$P A = L U@f$ 。
 *
 * 分解后 @p a 同时存储 L（严格下三角，单位对角线隐含）和 U（上三角）。
 *
 * @param[in,out] a              方阵；输入为 A，输出为 LU 因子。
 * @param[out]    pivots         置换数组，长度至少为 @c a->rows 。
 * @param[out]    out_swap_count 主元交换次数（用于行列式符号），可为 NULL 。
 * @return 若发现奇异主元返回 ::LMMC_STATUS_SINGULAR_MATRIX 。
 */
lmmc_status_t lmmc_lu_decompose_inplace(lmmc_mat_t* a, size_t* pivots, size_t* out_swap_count);

/**
 * @brief 利用 ::lmmc_lu_decompose_inplace 的结果求解 @c A x = b 。
 */
lmmc_status_t lmmc_lu_solve(const lmmc_mat_t* lu, const size_t* pivots, const lmmc_vec_t* b, lmmc_vec_t* x);

/**
 * @brief 就地 Cholesky 分解：@f$A = L L^T@f$ ，要求 @p a 对称正定。
 *
 * 分解后 @p a 的下三角存储 @c L ，上三角内容未定义。
 *
 * @return 若 @p a 非正定返回 ::LMMC_STATUS_NOT_POSITIVE_DEFINITE 。
 */
lmmc_status_t lmmc_cholesky_decompose_inplace(lmmc_mat_t* a);

/** @brief 利用 Cholesky 因子求解 @c A x = b 。 */
lmmc_status_t lmmc_cholesky_solve(const lmmc_mat_t* l, const lmmc_vec_t* b, lmmc_vec_t* x);

/**
 * @brief 就地 QR 分解（Householder 变换）。
 *
 * @param[in,out] a        m×n 矩阵，分解后存储 R 与下三角的 Householder 反射子。
 * @param[out]    tau      Householder 标量数组。
 * @param[in]     tau_size @p tau 的容量，至少为 min(m,n)。
 */
lmmc_status_t lmmc_qr_decompose_inplace(lmmc_mat_t* a, lmmc_real_t* tau, size_t tau_size);

/** @brief 利用 QR 分解结果求解最小二乘 / 方阵线性系统。 */
lmmc_status_t lmmc_qr_solve(const lmmc_mat_t* qr, const lmmc_real_t* tau, const lmmc_vec_t* b, lmmc_vec_t* x);

#ifdef __cplusplus
}
#endif

#endif
