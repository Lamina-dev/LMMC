/**
 * @file eigen.h
 * @brief 特征值 / 奇异值分解,伪逆与条件数接口.
 */
#ifndef LMMC_EIGEN_H
#define LMMC_EIGEN_H

#include "lmmc/dense.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 对称矩阵特征分解结果.
 *
 * 第 @c i 个特征值对应 @c eigenvectors 的第 @c i 列.
 */
typedef struct {
    lmmc_vec_t eigenvalues;    /**< 升序特征值向量,长度 n . */
    lmmc_mat_t eigenvectors;   /**< 正交特征向量矩阵,nxn . */
} lmmc_eigen_sym_result_t;

/**
 * @brief 一般实矩阵特征分解结果(仅特征值,使用实部 / 虚部分别存储).
 */
typedef struct {
    lmmc_vec_t real_parts;     /**< 特征值实部数组,长度 n . */
    lmmc_vec_t imag_parts;     /**< 特征值虚部数组,长度 n . */
} lmmc_eigen_gen_result_t;

/**
 * @brief 一般实矩阵完整特征分解结果(特征值 + 特征向量).
 *
 * 对于实特征值 lambda_i,对应特征向量存储在 vectors_real 的第 i 列,
 * vectors_imag 的第 i 列全为零.
 * 对于共轭复数对 lambda_i = a + bi, lambda_{i+1} = a - bi,
 * 特征向量的实部存储在 vectors_real 的第 i 和 i+1 列,
 * 虚部存储在 vectors_imag 的第 i 列(正号)和第 i+1 列(负号).
 */
typedef struct {
    lmmc_vec_t real_parts;      /**< 特征值实部数组,长度 n . */
    lmmc_vec_t imag_parts;      /**< 特征值虚部数组,长度 n . */
    lmmc_mat_t vectors_real;    /**< 特征向量实部矩阵,nxn . */
    lmmc_mat_t vectors_imag;    /**< 特征向量虚部矩阵,nxn . */
} lmmc_eigen_gen_full_result_t;

/**
 * @brief 奇异值分解结果:@f$A = U \Sigma V^T@f$ .
 */
typedef struct {
    lmmc_mat_t U;       /**< 左奇异向量矩阵,mxm . */
    lmmc_vec_t sigma;   /**< 奇异值(降序),长度 min(m,n) . */
    lmmc_mat_t Vt;      /**< 右奇异向量矩阵的转置 V^T ,nxn . */
} lmmc_svd_result_t;

/**
 * @brief 计算实对称矩阵的全部特征值与特征向量.
 *
 * 算法:Householder 三对角化 -> 隐式 QL 迭代.
 * 特征值按升序排列,第 i 个特征值对应 eigenvectors 的第 i 列.
 *
 * @param[in]  a          nxn 对称矩阵(不被修改).
 * @param[out] out_result 输出结果,调用方需配对调用 ::lmmc_eigen_sym_result_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或矩阵非方阵.
 * - ::LMMC_STATUS_CONVERGENCE_FAILED - QL 迭代未收敛(超过 30 次).
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存(eigenvalues 向量 + eigenvectors 矩阵 + 临时工作空间).
 * - a 不被修改.
 */
lmmc_status_t lmmc_eigen_symmetric(
    const lmmc_mat_t* a,
    lmmc_eigen_sym_result_t* out_result
);

/**
 * @brief 计算一般实矩阵的特征值(不输出特征向量).
 *
 * 算法:Householder Hessenberg 约化 -> Francis 双移位 QR 迭代.
 * 复数特征值以共轭对形式出现在 real_parts/imag_parts 中.
 *
 * @param[in]  a          nxn 实矩阵(不被修改).
 * @param[out] out_result 输出结果,调用方需配对调用 ::lmmc_eigen_gen_result_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或矩阵非方阵.
 * - ::LMMC_STATUS_CONVERGENCE_FAILED - QR 迭代未收敛.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存.a 不被修改.
 */
lmmc_status_t lmmc_eigen_general(
    const lmmc_mat_t* a,
    lmmc_eigen_gen_result_t* out_result
);

/**
 * @brief 计算奇异值分解:A = U * Σ * V^T.
 *
 * 算法:Householder 双对角化 -> Golub-Kahan 隐式 QR 迭代.
 * 奇异值按降序排列.
 *
 * @param[in]  a          mxn 矩阵(不被修改).
 * @param[out] out_result 输出 U(mxm),sigma(min(m,n)),Vt(nxn).
 *                        调用方需配对调用 ::lmmc_svd_result_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL.
 * - ::LMMC_STATUS_CONVERGENCE_FAILED - 迭代未收敛.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存.a 不被修改.
 */
lmmc_status_t lmmc_svd(
    const lmmc_mat_t* a,
    lmmc_svd_result_t* out_result
);

/**
 * @brief 计算 Moore-Penrose 伪逆 A^+.
 *
 * 内部通过 SVD 实现:A^+ = V * Σ^{-1}_trunc * U^T,
 * 其中奇异值 < tol * sigma_max 的被截断为零.
 *
 * @param[in]  a        mxn 矩阵(不被修改).
 * @param[in]  tol      奇异值截断阈值;<=0 时使用 max(m,n)*eps*sigma_max.
 * @param[out] out_pinv nxm 输出矩阵,须已创建.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL.
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - out_pinv 维度与 nxm 不匹配.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 覆写 out_pinv->data.内部分配并释放 SVD 工作空间.
 */
lmmc_status_t lmmc_pinv(
    const lmmc_mat_t* a,
    lmmc_real_t tol,
    lmmc_mat_t* out_pinv
);

/**
 * @brief 计算矩阵的 2-范数条件数:κ_2(A) = sigma_max / sigma_min.
 *
 * 内部通过 SVD 计算.若 sigma_min ~= 0 则条件数为 Inf.
 *
 * @param[in]  a        输入矩阵(不被修改).
 * @param[out] out_cond 输出条件数.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 内部分配并释放 SVD 工作空间.a 不被修改.
 */
lmmc_status_t lmmc_cond(
    const lmmc_mat_t* a,
    lmmc_real_t* out_cond
);

/**
 * @brief 计算一般实矩阵的特征值与特征向量.
 *
 * 使用 Hessenberg 约化 + Francis 双移位 QR 迭代求特征值,
 * 再通过逆迭代(Wilkinson 移位)计算特征向量.
 *
 * @param[in]  a          nxn 实矩阵.
 * @param[out] out_result 调用方需用 ::lmmc_eigen_gen_full_result_destroy 释放.
 */
lmmc_status_t lmmc_eigen_general_full(
    const lmmc_mat_t* a,
    lmmc_eigen_gen_full_result_t* out_result
);

/** @brief 释放 ::lmmc_eigen_symmetric 输出结果. */
void lmmc_eigen_sym_result_destroy(lmmc_eigen_sym_result_t* result);
/** @brief 释放 ::lmmc_eigen_general 输出结果. */
void lmmc_eigen_gen_result_destroy(lmmc_eigen_gen_result_t* result);
/** @brief 释放 ::lmmc_eigen_general_full 输出结果. */
void lmmc_eigen_gen_full_result_destroy(lmmc_eigen_gen_full_result_t* result);
/** @brief 释放 ::lmmc_svd 输出结果. */
void lmmc_svd_result_destroy(lmmc_svd_result_t* result);

#ifdef __cplusplus
}
#endif

#endif
