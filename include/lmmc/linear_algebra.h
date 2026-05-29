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
 * @brief 就地 LU 分解（带部分主元选取）：P*A = L*U。
 *
 * 分解后 a 同时存储 L（严格下三角，单位对角线隐含）和 U（上三角含对角线）。
 * 行交换记录在 pivots 数组中：pivots[k] 表示第 k 步与第 pivots[k] 行交换。
 *
 * @param[in,out] a              输入方阵 A，输出为 LU 因子（就地覆写）。
 * @param[out]    pivots         置换数组，长度至少为 a->rows。
 * @param[out]    out_swap_count 主元交换次数（用于行列式符号），可为 NULL。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或矩阵非方阵。
 * - ::LMMC_STATUS_SINGULAR_MATRIX — 主元绝对值 <= 1e-15，矩阵奇异。
 *
 * @par 副作用
 * - 就地覆写 a->data（原始矩阵内容丢失）。
 * - 覆写 pivots 数组。无堆内存分配。
 *
 * @par 使用模式
 * @code
 * lmmc_lu_decompose_inplace(&A, pivots, NULL);
 * lmmc_lu_solve(&A, pivots, &b, &x);
 * @endcode
 */
lmmc_status_t lmmc_lu_decompose_inplace(lmmc_mat_t* a, size_t* pivots, size_t* out_swap_count);

/**
 * @brief 利用 LU 分解结果求解线性系统 A*x = b。
 *
 * lu 和 pivots 须来自同一次 ::lmmc_lu_decompose_inplace 调用。
 * 内部执行前代（L）+ 回代（U）+ 行置换应用。
 *
 * @param[in]  lu     LU 因子矩阵（来自 decompose_inplace，不被修改）。
 * @param[in]  pivots 置换数组（来自 decompose_inplace）。
 * @param[in]  b      右端向量，长度须等于 lu->rows。
 * @param[out] x      解向量，长度须等于 lu->rows，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不匹配。
 * - ::LMMC_STATUS_SINGULAR_MATRIX — U 对角线为零。
 *
 * @par 副作用
 * - 覆写 x->data。无内存分配。lu 和 b 不被修改。
 */
lmmc_status_t lmmc_lu_solve(const lmmc_mat_t* lu, const size_t* pivots, const lmmc_vec_t* b, lmmc_vec_t* x);

/**
 * @brief 就地 Cholesky 分解：A = L*L^T，要求 A 对称正定。
 *
 * 分解后 a 的下三角（含对角线）存储 L，上三角内容未定义。
 * 使用相对阈值检测非正定：若某步对角元 <= rel_tol * max_diag 则判定失败。
 *
 * @param[in,out] a 输入对称正定方阵，输出为下三角 L（就地覆写）。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或矩阵非方阵。
 * - ::LMMC_STATUS_NOT_POSITIVE_DEFINITE — 矩阵非正定。
 *
 * @par 副作用
 * - 就地覆写 a->data（原始矩阵内容丢失）。无堆内存分配。
 */
lmmc_status_t lmmc_cholesky_decompose_inplace(lmmc_mat_t* a);

/**
 * @brief 利用 Cholesky 因子求解 A*x = b（A = L*L^T）。
 *
 * 内部执行前代 L*y=b + 回代 L^T*x=y。
 *
 * @param[in]  l Cholesky 下三角因子（来自 decompose_inplace，不被修改）。
 * @param[in]  b 右端向量。
 * @param[out] x 解向量，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不匹配。
 *
 * @par 副作用
 * - 覆写 x->data。无内存分配。
 */
lmmc_status_t lmmc_cholesky_solve(const lmmc_mat_t* l, const lmmc_vec_t* b, lmmc_vec_t* x);

/**
 * @brief 就地 Householder QR 分解。
 *
 * 分解后 a 的上三角存储 R，下三角（对角线以下）存储 Householder 反射子 v_k，
 * tau[k] 存储对应的标量因子。
 *
 * @param[in,out] a        m×n 矩阵，输出为 QR 紧凑表示（就地覆写）。
 * @param[out]    tau      Householder 标量数组，长度至少 min(m,n)。
 * @param[in]     tau_size tau 数组的容量，必须 >= min(m,n)。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或 tau_size 不足。
 *
 * @par 副作用
 * - 就地覆写 a->data 和 tau 数组。无堆内存分配。
 */
lmmc_status_t lmmc_qr_decompose_inplace(lmmc_mat_t* a, lmmc_real_t* tau, size_t tau_size);

/**
 * @brief 利用 QR 分解结果求解线性系统或最小二乘问题。
 *
 * 对 m×n 矩阵（m>=n）：求解 min||Ax-b||_2，即 R*x = Q^T*b 的前 n 行。
 * qr 和 tau 须来自同一次 ::lmmc_qr_decompose_inplace 调用。
 *
 * @param[in]  qr  QR 紧凑表示矩阵（不被修改）。
 * @param[in]  tau Householder 标量数组。
 * @param[in]  b   右端向量，长度须等于 qr->rows。
 * @param[out] x   解向量，长度须等于 qr->cols，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — m < n 或向量长度不匹配。
 * - ::LMMC_STATUS_SINGULAR_MATRIX — R 对角线为零。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 临时工作内存分配失败。
 *
 * @par 副作用
 * - 覆写 x->data。分配并释放长度为 m 的临时数组。
 */
lmmc_status_t lmmc_qr_solve(const lmmc_mat_t* qr, const lmmc_real_t* tau, const lmmc_vec_t* b, lmmc_vec_t* x);

#ifdef __cplusplus
}
#endif

#endif
