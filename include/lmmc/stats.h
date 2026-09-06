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
 * 使用在线均值更新；当 `x-mean` 溢出时切换到等价的有限凸组合，因此
 * 只要最终均值可表示，就不会因为原始总和、极差或无关的方差溢出而失败。
 *
 * @param[in]  x        输入有限向量，size >= 1。
 * @param[out] out_mean 输出均值。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 x 为空向量或指针为 NULL；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若输入包含非有限值。
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

/**
 * @brief 总体 Pearson 相关系数。
 *
 * 两个输入分别按其最大绝对值缩放后累计中心矩，利用相关系数的尺度
 * 不变性避免原始中心矩和方差乘积发生不必要的溢出。
 */
lmmc_status_t lmmc_vec_correlation_population(const lmmc_vec_t* x, const lmmc_vec_t* y, lmmc_real_t* out_correlation);
/** @brief 样本 Pearson 相关系数；使用同一缩放累计路径，要求 n>=2。 */
lmmc_status_t lmmc_vec_correlation_sample(const lmmc_vec_t* x, const lmmc_vec_t* y, lmmc_real_t* out_correlation);

/**
 * @brief 计算矩阵各列均值，输出长度为 @c x->cols 的向量。
 *
 * 矩阵 @p x 的每行视为一次观测样本；每列复用向量均值的溢出安全在线
 * 更新，不分配临时存储。
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

/**
 * @brief 计算总体 Pearson 相关矩阵。
 *
 * 每一列对分别复用缩放后的中心矩累计，不构造可能溢出的原尺度方差或
 * 标准差，也不分配临时均值和标准差向量。
 */
lmmc_status_t lmmc_mat_correlation_population(const lmmc_mat_t* x, lmmc_mat_t* out_correlation);
/** @brief 使用同一缩放路径计算样本 Pearson 相关矩阵，要求行数 >= 2。 */
lmmc_status_t lmmc_mat_correlation_sample(const lmmc_mat_t* x, lmmc_mat_t* out_correlation);

/**
 * @brief 计算向量中位数。
 *
 * 内部会复制并排序数据。对偶数长度使用无溢出的浮点中点计算。
 * 所有观测值必须有限；非有限观测不会进入排序比较器。
 *
 * @param[in]  x   输入有限向量，size >= 1。
 * @param[out] out 输出中位数。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 x->size == 0 或指针为 NULL；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若任一观测值非有限；
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
 * 非有限概率会在转换为数组索引前拒绝；所有观测值必须有限。
 *
 * @param[in]  x   输入有限向量，size >= 1。
 * @param[in]  p   有限分位数，范围 [0, 1]。
 * @param[out] out 输出分位数值。
 * @return LMMC_STATUS_OK 成功；LMMC_STATUS_INVALID_ARGUMENT 若 p 非有限
 *         或不在 [0,1]；LMMC_STATUS_NUMERICAL_FAILURE 若观测值非有限。
 */
lmmc_status_t lmmc_vec_quantile(const lmmc_vec_t* x, lmmc_real_t p, lmmc_real_t* out);

/**
 * @brief 计算向量的等宽直方图。
 *
 * 将有限数据范围 [min, max] 等分为 nbins 个区间。跨零范围采用
 * 凸组合构造边界，避免 max-min 的中间溢出；分箱使用缩放后的归一化
 * 坐标，保持线性时间复杂度。最后一个区间包含右端点。
 *
 * @param[in]  x      输入有限向量，size >= 1。
 * @param[in]  nbins  区间数，必须 >= 1。
 * @param[out] edges  输出区间边界数组，长度 nbins+1，调用方预分配。
 * @param[out] counts 输出每个区间的计数，长度 nbins，调用方预分配。
 * @return LMMC_STATUS_OK 成功；LMMC_STATUS_NUMERICAL_FAILURE 若任一
 *         观测值非有限，此时输出数组保持不变。
 */
lmmc_status_t lmmc_vec_histogram(const lmmc_vec_t* x, size_t nbins, lmmc_real_t* edges, size_t* counts);

/**
 * @brief 连续分布函数的公共参数约束。
 *
 * 所有实数输入必须有限；尺度、形状和自由度参数必须为正数；分位数函数的
 * 概率参数必须严格位于 (0, 1)。违反约束时返回
 * LMMC_STATUS_INVALID_ARGUMENT。
 * t、χ²、F、Gamma 和 Beta 分位数使用保持根区间的求解器；不能建立
 * 有限根区间时返回 LMMC_STATUS_OUT_OF_RANGE，达到数值分辨率或迭代上限
 * 但未满足概率残差时返回 LMMC_STATUS_CONVERGENCE_FAILED。基于不完全
 * Gamma/Beta 的 CDF 达到内部级数或连分数迭代上限时也返回该状态。
 * 连续分布 PDF 在对数域组合密度项；若有限参数对应的密度不可用有限
 * double 表示，或中间计算无法得到有限密度，则返回
 * LMMC_STATUS_NUMERICAL_FAILURE。CDF 的计算结果必须是 [0, 1] 内的有限
 * 概率，否则返回同一状态。正态分位数产生不可表示的结果时亦如此。
 * 数值失败和收敛失败均不写入输出参数；其他非成功状态不保证写入。
 */

/* --- 正态分布 --- */

/**
 * @brief 正态分布概率密度函数。
 *
 * 在对数域组合指数与尺度项；标准化在原始 `x-mu` 溢出时改用等价的
 * `x/sigma-mu/sigma`，避免丢失仍可表示的标准分数。
 * @return 成功时写入有限密度；密度不可表示时返回
 * LMMC_STATUS_NUMERICAL_FAILURE。
 */
lmmc_status_t lmmc_dist_normal_pdf(lmmc_real_t x, lmmc_real_t mu, lmmc_real_t sigma, lmmc_real_t* out);
/**
 * @brief 正态分布累积分布函数。
 *
 * 使用与 PDF 相同的溢出安全标准化，并通过 erfc 直接保留可表示的下尾概率。
 */
lmmc_status_t lmmc_dist_normal_cdf(lmmc_real_t x, lmmc_real_t mu, lmmc_real_t sigma, lmmc_real_t* out);
/**
 * @brief 正态分布分位数函数（逆 CDF）。
 *
 * 位置尺度变换使用融合乘加，使 `sigma*z+mu` 仅舍入一次，并避免最终
 * 结果可表示时乘积中间值的虚假溢出。
 */
lmmc_status_t lmmc_dist_normal_quantile(lmmc_real_t p, lmmc_real_t mu, lmmc_real_t sigma, lmmc_real_t* out);

/* --- t 分布 --- */

/**
 * @brief t 分布概率密度函数。
 *
 * 对数核按 `|x|/sqrt(df)` 的大小选取等价形式，避免直接计算 `x*x`。
 * 当自由度大到 t 分布与标准正态分布的差异低于双精度分辨率时，直接
 * 使用正态极限，避免两个巨大 `lgamma` 值相减。
 */
lmmc_status_t lmmc_dist_t_pdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out);
/**
 * @brief t 分布累积分布函数。
 *
 * 不完全 Beta 参数通过尺度比构造；参数本身下溢时在对数域计算其
 * 小参数渐近值，使可表示的重尾概率不因 `x*x` 溢出而被截断为零。
 * 超大自由度使用标准正态极限。
 */
lmmc_status_t lmmc_dist_t_cdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out);
/**
 * @brief t 分布分位数函数。
 *
 * 超大自由度使用标准正态分位数，避免退化的不完全 Beta 反演。
 */
lmmc_status_t lmmc_dist_t_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out);

/* --- χ² 分布 --- */

/**
 * @brief χ² 分布概率密度函数。
 *
 * 大自由度通过形状 `df / 2`、尺度 2 的稳定 Gamma PDF 求值，避免
 * 多个巨大对数 Gamma 项抵消。
 */
lmmc_status_t lmmc_dist_chi2_pdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out);
/**
 * @brief χ² 分布累积分布函数。
 *
 * 通过形状 `df / 2`、尺度 2 的 Gamma CDF 求值，共享其极小下尾
 * 对数域路径，避免先计算 `x / 2` 导致下溢。
 */
lmmc_status_t lmmc_dist_chi2_cdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out);
/** @brief χ² 分布分位数函数。 */
lmmc_status_t lmmc_dist_chi2_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out);

/* --- F 分布 --- */

/**
 * @brief F 分布概率密度函数。
 *
 * 大自由度使用 Beta 变量变换、偏差函数和 Stirling 余项计算对数密度，
 * 避免中心区域的 log-gamma 灾难性抵消。
 */
lmmc_status_t lmmc_dist_f_pdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2, lmmc_real_t* out);
/** @brief F 分布累积分布函数。 */
lmmc_status_t lmmc_dist_f_cdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2, lmmc_real_t* out);
/** @brief F 分布分位数函数。 */
lmmc_status_t lmmc_dist_f_quantile(lmmc_real_t p, lmmc_real_t df1, lmmc_real_t df2, lmmc_real_t* out);

/* --- 伽马分布 --- */

/**
 * @brief 伽马分布概率密度函数。
 *
 * 大形状参数通过 Poisson 偏差与 Stirling 余项计算，避免在分布中心
 * 抵消多个巨大对数项。
 */
lmmc_status_t lmmc_dist_gamma_pdf(lmmc_real_t x, lmmc_real_t shape, lmmc_real_t scale, lmmc_real_t* out);
/**
 * @brief 伽马分布累积分布函数。
 *
 * 当 `x / scale` 下溢但最终概率仍可表示时，在对数域计算小参数下
 * 不完全 Gamma 的首项，避免把可表示的极小尾概率错误截断为零。
 */
lmmc_status_t lmmc_dist_gamma_cdf(lmmc_real_t x, lmmc_real_t shape, lmmc_real_t scale, lmmc_real_t* out);
/** @brief 伽马分布分位数函数。 */
lmmc_status_t lmmc_dist_gamma_quantile(lmmc_real_t p, lmmc_real_t shape, lmmc_real_t scale, lmmc_real_t* out);

/* --- 贝塔分布 --- */

/**
 * @brief 贝塔分布概率密度函数。
 *
 * 大形状参数使用 Stirling 余项与偏差形式组合对数密度，避免中心区域
 * 的 log-gamma 灾难性抵消。对称中心在形状和溢出时使用 Gamma 比值
 * 渐近式，不要求显式构造 `alpha + beta`。
 */
lmmc_status_t lmmc_dist_beta_pdf(lmmc_real_t x, lmmc_real_t alpha, lmmc_real_t beta, lmmc_real_t* out);
/** @brief 贝塔分布累积分布函数。 */
lmmc_status_t lmmc_dist_beta_cdf(lmmc_real_t x, lmmc_real_t alpha, lmmc_real_t beta, lmmc_real_t* out);
/** @brief 贝塔分布分位数函数。 */
lmmc_status_t lmmc_dist_beta_quantile(lmmc_real_t p, lmmc_real_t alpha, lmmc_real_t beta, lmmc_real_t* out);

/**
 * @brief 离散分布函数的公共参数约束。
 *
 * 二项分布概率 p 必须为 [0, 1] 内的有限值；泊松分布率 lambda 必须为
 * 非负有限值。违反约束时返回 LMMC_STATUS_INVALID_ARGUMENT。PMF 中间值或
 * 最终值不是有限概率时返回 LMMC_STATUS_NUMERICAL_FAILURE；允许正确结果
 * 下溢为零。CDF 在大参数中心区域使用不完全 Gamma 的一致渐近式以及
 * 对称 Beta 恒等式，其他区域的迭代达到上限时返回
 * LMMC_STATUS_CONVERGENCE_FAILED。数值失败和收敛失败均不写入输出参数。
 */

/* --- 二项分布 --- */

/**
 * @brief 二项分布概率质量函数。
 *
 * 使用 Stirling 余项与二项偏差形式组合对数质量，避免大 n 中心区域
 * 的多个 log-gamma 项发生灾难性抵消。
 */
lmmc_status_t lmmc_dist_binomial_pmf(size_t k, size_t n, lmmc_real_t p, lmmc_real_t* out);
/**
 * @brief 二项分布累积分布函数。
 *
 * 对称奇数试验的中心概率通过 Beta 对称性精确返回 1/2。
 */
lmmc_status_t lmmc_dist_binomial_cdf(size_t k, size_t n, lmmc_real_t p_param, lmmc_real_t* out);

/* --- 泊松分布 --- */

/**
 * @brief 泊松分布概率质量函数。
 *
 * 使用 Stirling 余项与泊松偏差形式组合对数质量，避免大率中心区域
 * 的 count*log(lambda)-lambda-log-gamma 灾难性抵消。
 */
lmmc_status_t lmmc_dist_poisson_pmf(size_t k, lmmc_real_t lambda, lmmc_real_t* out);
/**
 * @brief 泊松分布累积分布函数；直接计算正则化上不完全伽马尾。
 *
 * 大率中心区域采用一致渐近展开，避免级数或连分数工作量随率的平方根增长。
 */
lmmc_status_t lmmc_dist_poisson_cdf(size_t k, lmmc_real_t lambda, lmmc_real_t* out);

#ifdef __cplusplus
}
#endif

#endif
