/**
 * @file stats.h
 * @brief 基础统计量与组合数学接口。
 *
 * 提供阶乘、排列数 nPr、组合数 nCr，以及一维向量与列优先矩阵的
 * 均值、方差、标准差、协方差、相关系数等。
 */
#ifndef LMMC_STATS_H
#define LMMC_STATS_H

#include "lmmc/dense.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 计算 @f$n!@f$ ，溢出时按 IEEE-754 规则返回 inf 。 */
void lmmc_stats_factorial(lmmc_real_t* out_val, uint32_t n);
/** @brief 计算排列数 @f$P(n,r)=n!/(n-r)!@f$ 。 */
void lmmc_stats_nPr(lmmc_real_t* out_val, uint32_t n, uint32_t r);
/** @brief 计算组合数 @f$C(n,r)=\binom{n}{r}@f$ 。 */
void lmmc_stats_nCr(lmmc_real_t* out_val, uint32_t n, uint32_t r);

/** @brief 计算向量样本均值 @f$\bar{x}@f$ 。 */
lmmc_status_t lmmc_vec_mean(const lmmc_vec_t* x, lmmc_real_t* out_mean);

/** @brief 总体方差 @f$\sigma^2 = \frac{1}{n}\sum (x_i-\bar{x})^2@f$ 。 */
lmmc_status_t lmmc_vec_variance_population(const lmmc_vec_t* x, lmmc_real_t* out_variance);
/** @brief 样本方差 @f$s^2 = \frac{1}{n-1}\sum (x_i-\bar{x})^2@f$ ，要求 n>=2 。 */
lmmc_status_t lmmc_vec_variance_sample(const lmmc_vec_t* x, lmmc_real_t* out_variance);

/** @brief 总体标准差。 */
lmmc_status_t lmmc_vec_stddev_population(const lmmc_vec_t* x, lmmc_real_t* out_stddev);
/** @brief 样本标准差，要求 n>=2 。 */
lmmc_status_t lmmc_vec_stddev_sample(const lmmc_vec_t* x, lmmc_real_t* out_stddev);

/** @brief 总体协方差 @f$\mathrm{Cov}(X,Y)@f$ 。 */
lmmc_status_t lmmc_vec_covariance_population(const lmmc_vec_t* x, const lmmc_vec_t* y, lmmc_real_t* out_covariance);
/** @brief 样本协方差，要求 n>=2 。 */
lmmc_status_t lmmc_vec_covariance_sample(const lmmc_vec_t* x, const lmmc_vec_t* y, lmmc_real_t* out_covariance);

/** @brief 总体 Pearson 相关系数。 */
lmmc_status_t lmmc_vec_correlation_population(const lmmc_vec_t* x, const lmmc_vec_t* y, lmmc_real_t* out_correlation);
/** @brief 样本 Pearson 相关系数，要求 n>=2 。 */
lmmc_status_t lmmc_vec_correlation_sample(const lmmc_vec_t* x, const lmmc_vec_t* y, lmmc_real_t* out_correlation);

/**
 * @brief 计算矩阵各列均值，输出长度为 @c x->cols 的向量。
 *
 * 矩阵 @p x 的每行视为一次观测样本。
 */
lmmc_status_t lmmc_mat_column_mean(const lmmc_mat_t* x, lmmc_vec_t* out_means);

/** @brief 计算总体协方差矩阵（cols × cols）。 */
lmmc_status_t lmmc_mat_covariance_population(const lmmc_mat_t* x, lmmc_mat_t* out_covariance);
/** @brief 计算样本协方差矩阵（cols × cols），要求行数 >= 2 。 */
lmmc_status_t lmmc_mat_covariance_sample(const lmmc_mat_t* x, lmmc_mat_t* out_covariance);

/** @brief 计算总体 Pearson 相关矩阵。 */
lmmc_status_t lmmc_mat_correlation_population(const lmmc_mat_t* x, lmmc_mat_t* out_correlation);
/** @brief 计算样本 Pearson 相关矩阵，要求行数 >= 2 。 */
lmmc_status_t lmmc_mat_correlation_sample(const lmmc_mat_t* x, lmmc_mat_t* out_correlation);

#ifdef __cplusplus
}
#endif

#endif
