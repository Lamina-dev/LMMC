/**
 * @file eigen.h
 * @brief 特征值 / 奇异值分解、伪逆与条件数接口。
 */
#ifndef LMMC_EIGEN_H
#define LMMC_EIGEN_H

#include "lmmc/dense.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 对称矩阵特征分解结果。
 *
 * 第 @c i 个特征值对应 @c eigenvectors 的第 @c i 列。
 */
typedef struct {
    lmmc_vec_t eigenvalues;    /**< 升序特征值向量，长度 n 。 */
    lmmc_mat_t eigenvectors;   /**< 正交特征向量矩阵，n×n 。 */
} lmmc_eigen_sym_result_t;

/**
 * @brief 一般实矩阵特征分解结果（仅特征值，使用实部 / 虚部分别存储）。
 */
typedef struct {
    lmmc_vec_t real_parts;     /**< 特征值实部数组，长度 n 。 */
    lmmc_vec_t imag_parts;     /**< 特征值虚部数组，长度 n 。 */
} lmmc_eigen_gen_result_t;

/**
 * @brief 奇异值分解结果：@f$A = U \Sigma V^T@f$ 。
 */
typedef struct {
    lmmc_mat_t U;       /**< 左奇异向量矩阵，m×m 。 */
    lmmc_vec_t sigma;   /**< 奇异值（降序），长度 min(m,n) 。 */
    lmmc_mat_t Vt;      /**< 右奇异向量矩阵的转置 V^T ，n×n 。 */
} lmmc_svd_result_t;

/**
 * @brief 计算实对称矩阵的全部特征值与特征向量。
 *
 * @param[in]  a          n×n 对称矩阵。
 * @param[out] out_result 调用方需用 ::lmmc_eigen_sym_result_destroy 释放。
 */
lmmc_status_t lmmc_eigen_symmetric(
    const lmmc_mat_t* a,
    lmmc_eigen_sym_result_t* out_result
);

/**
 * @brief 计算一般实矩阵的特征值（不输出特征向量）。
 *
 * @param[in]  a          n×n 实矩阵。
 * @param[out] out_result 调用方需用 ::lmmc_eigen_gen_result_destroy 释放。
 */
lmmc_status_t lmmc_eigen_general(
    const lmmc_mat_t* a,
    lmmc_eigen_gen_result_t* out_result
);

/**
 * @brief 计算奇异值分解 @f$A = U \Sigma V^T@f$ 。
 *
 * @param[in]  a          m×n 矩阵。
 * @param[out] out_result 调用方需用 ::lmmc_svd_result_destroy 释放。
 */
lmmc_status_t lmmc_svd(
    const lmmc_mat_t* a,
    lmmc_svd_result_t* out_result
);

/**
 * @brief 计算 Moore-Penrose 伪逆 @f$A^+@f$ 。
 *
 * @param[in]  a       m×n 矩阵。
 * @param[in]  tol     奇异值截断阈值；@c <=0 时使用默认机器精度规则。
 * @param[out] out_pinv n×m 输出矩阵，需事先创建。
 */
lmmc_status_t lmmc_pinv(
    const lmmc_mat_t* a,
    lmmc_real_t tol,
    lmmc_mat_t* out_pinv
);

/**
 * @brief 计算矩阵的 2-范数条件数 @f$\sigma_{\max}/\sigma_{\min}@f$ 。
 */
lmmc_status_t lmmc_cond(
    const lmmc_mat_t* a,
    lmmc_real_t* out_cond
);

/** @brief 释放 ::lmmc_eigen_symmetric 输出结果。 */
void lmmc_eigen_sym_result_destroy(lmmc_eigen_sym_result_t* result);
/** @brief 释放 ::lmmc_eigen_general 输出结果。 */
void lmmc_eigen_gen_result_destroy(lmmc_eigen_gen_result_t* result);
/** @brief 释放 ::lmmc_svd 输出结果。 */
void lmmc_svd_result_destroy(lmmc_svd_result_t* result);

#ifdef __cplusplus
}
#endif

#endif
