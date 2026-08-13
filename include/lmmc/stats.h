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

/**
 * @brief 计算向量样本均值 @f$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i@f$ 。
 *
 * @param[in]  x        输入向量，size >= 1。
 * @param[out] out_mean 输出均值。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 x 为空向量（size == 0）或指针为 NULL。
 *
 * @par 副作用
 * - 无。只读访问输入向量，不分配内存。
 */
lmmc_status_t lmmc_vec_mean(const lmmc_vec_t* x, lmmc_real_t* out_mean);

/** @brief 总体方差 @f$\sigma^2 = \frac{1}{n}\sum (x_i-\bar{x})^2@f$ 。 */
lmmc_status_t lmmc_vec_variance_population(const lmmc_vec_t* x, lmmc_real_t* out_variance);

/**
 * @brief 计算样本方差 @f$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2@f$ 。
 *
 * 使用 Bessel 校正（除以 n-1），要求至少 2 个样本。
 *
 * @param[in]  x            输入向量，size >= 2。
 * @param[out] out_variance 输出样本方差。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 x->size < 2 或指针为 NULL。
 *
 * @par 副作用
 * - 无。只读访问输入向量，不分配内存。
 */
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

/**
 * @brief 计算样本协方差矩阵（cols × cols），要求行数 >= 2 。
 *
 * 输出矩阵 @p out_covariance 的 (i,j) 元素为第 i 列与第 j 列的样本协方差。
 * 使用 Bessel 校正（除以 rows-1）。
 *
 * @param[in]  x              输入矩阵，rows >= 2，每行为一次观测。
 * @param[out] out_covariance 输出协方差矩阵，大小 cols × cols，调用方需预分配。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 rows < 2、维度不匹配或指针为 NULL。
 *
 * @par 副作用
 * - 就地写入 @p out_covariance 矩阵的全部元素。
 * - 内部可能分配临时向量用于列均值计算，函数返回前释放。
 */
lmmc_status_t lmmc_mat_covariance_sample(const lmmc_mat_t* x, lmmc_mat_t* out_covariance);

/** @brief 计算总体 Pearson 相关矩阵。 */
lmmc_status_t lmmc_mat_correlation_population(const lmmc_mat_t* x, lmmc_mat_t* out_correlation);
/** @brief 计算样本 Pearson 相关矩阵，要求行数 >= 2 。 */
lmmc_status_t lmmc_mat_correlation_sample(const lmmc_mat_t* x, lmmc_mat_t* out_correlation);

/**
 * @brief 计算向量中位数。
 *
 * 内部会复制并排序数据。对偶数长度取中间两个值的平均。
 *
 * @param[in]  x   输入向量，size >= 1。
 * @param[out] out 输出中位数。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 x->size == 0 或指针为 NULL；
 *         ::LMMC_STATUS_ALLOCATION_FAILED 若临时数组分配失败。
 *
 * @par 副作用
 * - 内部分配长度为 x->size 的临时数组用于排序，函数返回前释放。
 * - 不修改输入向量 @p x 。
 */
lmmc_status_t lmmc_vec_median(const lmmc_vec_t* x, lmmc_real_t* out);

/**
 * @brief 计算向量的 p 分位数（线性插值法）。
 *
 * @param[in]  x   输入向量，size >= 1。
 * @param[in]  p   分位数，范围 [0, 1]。
 * @param[out] out 输出分位数值。
 * @return LMMC_STATUS_OK 成功；LMMC_STATUS_INVALID_ARGUMENT 若 p 不在 [0,1]。
 */
lmmc_status_t lmmc_vec_quantile(const lmmc_vec_t* x, lmmc_real_t p, lmmc_real_t* out);

/**
 * @brief 计算向量的等宽直方图。
 *
 * 将数据范围 [min, max] 等分为 nbins 个区间，统计每个区间的计数。
 *
 * @param[in]  x      输入向量，size >= 1。
 * @param[in]  nbins  区间数，必须 >= 1。
 * @param[out] edges  输出区间边界数组，长度 nbins+1，调用方预分配。
 * @param[out] counts 输出每个区间的计数，长度 nbins，调用方预分配。
 * @return LMMC_STATUS_OK 成功。
 */
lmmc_status_t lmmc_vec_histogram(const lmmc_vec_t* x, size_t nbins, lmmc_real_t* edges, size_t* counts);

/* --- 正态分布 --- */

/** @brief 正态分布概率密度函数。 */
lmmc_status_t lmmc_dist_normal_pdf(lmmc_real_t x, lmmc_real_t mu, lmmc_real_t sigma, lmmc_real_t* out);
/** @brief 正态分布累积分布函数。 */
lmmc_status_t lmmc_dist_normal_cdf(lmmc_real_t x, lmmc_real_t mu, lmmc_real_t sigma, lmmc_real_t* out);
/** @brief 正态分布分位数函数（逆 CDF）。 */
lmmc_status_t lmmc_dist_normal_quantile(lmmc_real_t p, lmmc_real_t mu, lmmc_real_t sigma, lmmc_real_t* out);

/* --- t 分布 --- */

/** @brief t 分布概率密度函数。 */
lmmc_status_t lmmc_dist_t_pdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out);
/** @brief t 分布累积分布函数。 */
lmmc_status_t lmmc_dist_t_cdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out);
/** @brief t 分布分位数函数。 */
lmmc_status_t lmmc_dist_t_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out);

/* --- χ² 分布 --- */

/** @brief χ² 分布概率密度函数。 */
lmmc_status_t lmmc_dist_chi2_pdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out);
/** @brief χ² 分布累积分布函数。 */
lmmc_status_t lmmc_dist_chi2_cdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out);
/** @brief χ² 分布分位数函数。 */
lmmc_status_t lmmc_dist_chi2_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out);

/* --- F 分布 --- */

/** @brief F 分布概率密度函数。 */
lmmc_status_t lmmc_dist_f_pdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2, lmmc_real_t* out);
/** @brief F 分布累积分布函数。 */
lmmc_status_t lmmc_dist_f_cdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2, lmmc_real_t* out);
/** @brief F 分布分位数函数。 */
lmmc_status_t lmmc_dist_f_quantile(lmmc_real_t p, lmmc_real_t df1, lmmc_real_t df2, lmmc_real_t* out);

/* --- 伽马分布 --- */

/** @brief 伽马分布概率密度函数。 */
lmmc_status_t lmmc_dist_gamma_pdf(lmmc_real_t x, lmmc_real_t shape, lmmc_real_t scale, lmmc_real_t* out);
/** @brief 伽马分布累积分布函数。 */
lmmc_status_t lmmc_dist_gamma_cdf(lmmc_real_t x, lmmc_real_t shape, lmmc_real_t scale, lmmc_real_t* out);
/** @brief 伽马分布分位数函数。 */
lmmc_status_t lmmc_dist_gamma_quantile(lmmc_real_t p, lmmc_real_t shape, lmmc_real_t scale, lmmc_real_t* out);

/* --- 贝塔分布 --- */

/** @brief 贝塔分布概率密度函数。 */
lmmc_status_t lmmc_dist_beta_pdf(lmmc_real_t x, lmmc_real_t alpha, lmmc_real_t beta, lmmc_real_t* out);
/** @brief 贝塔分布累积分布函数。 */
lmmc_status_t lmmc_dist_beta_cdf(lmmc_real_t x, lmmc_real_t alpha, lmmc_real_t beta, lmmc_real_t* out);
/** @brief 贝塔分布分位数函数。 */
lmmc_status_t lmmc_dist_beta_quantile(lmmc_real_t p, lmmc_real_t alpha, lmmc_real_t beta, lmmc_real_t* out);

/* --- 二项分布 --- */

/** @brief 二项分布概率质量函数。 */
lmmc_status_t lmmc_dist_binomial_pmf(size_t k, size_t n, lmmc_real_t p, lmmc_real_t* out);
/** @brief 二项分布累积分布函数。 */
lmmc_status_t lmmc_dist_binomial_cdf(size_t k, size_t n, lmmc_real_t p_param, lmmc_real_t* out);

/* --- 泊松分布 --- */

/** @brief 泊松分布概率质量函数。 */
lmmc_status_t lmmc_dist_poisson_pmf(size_t k, lmmc_real_t lambda, lmmc_real_t* out);
/** @brief 泊松分布累积分布函数。 */
lmmc_status_t lmmc_dist_poisson_cdf(size_t k, lmmc_real_t lambda, lmmc_real_t* out);

#ifdef __cplusplus
}
#endif

#endif
