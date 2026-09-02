/**
 * @file random.h
 * @brief 伪随机数发生器与常用分布。
 *
 * 内部使用线程不安全的 PCG/xoshiro 类生成器（不透明实现），
 * 调用方应为每个线程独立创建 ::lmmc_rng_t 。
 */
#ifndef LMMC_RANDOM_H
#define LMMC_RANDOM_H

#include "lmmc/config.h"
#include "lmmc/status.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 随机数发生器（不透明类型）。 */
typedef struct lmmc_rng_t lmmc_rng_t;

/**
 * @brief 创建随机数发生器实例。
 *
 * 分配并初始化一个 xoshiro256** 发生器。句柄由调用方拥有，默认
 * 线程受限；跨线程传递或并发访问必须由调用方同步。并行工作负载应为
 * 每个线程创建独立句柄，或使用 ::lmmc_rng_clone 和 ::lmmc_rng_jump。
 *
 * @param[out] out_rng 输出新创建的 RNG 句柄。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out_rng 为 NULL；
 *         ::LMMC_STATUS_ALLOCATION_FAILED 若内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存存储 RNG 内部状态，调用方必须调用 ::lmmc_rng_destroy 释放。
 */
lmmc_status_t lmmc_rng_create(lmmc_rng_t** out_rng);

/** @brief 重新设定种子。 */
lmmc_status_t lmmc_rng_seed(lmmc_rng_t* rng, uint64_t seed);
/** @brief 销毁发生器，释放内部分配的所有内存。 */
void lmmc_rng_destroy(lmmc_rng_t* rng);

/**
 * @brief 深拷贝 RNG 状态。
 *
 * 调用方可对副本调用 lmmc_rng_jump 以获得独立的并行流。
 *
 * @param[in]  src     源 RNG（不可为 NULL）。
 * @param[out] out_rng 输出新分配的 RNG 副本。
 * @return LMMC_STATUS_OK 成功；LMMC_STATUS_INVALID_ARGUMENT 若 src 或 out_rng 为 NULL。
 *
 * @par 副作用
 * - 分配堆内存存储副本状态，调用方必须调用 ::lmmc_rng_destroy 释放。
 */
lmmc_status_t lmmc_rng_clone(const lmmc_rng_t* src, lmmc_rng_t** out_rng);

/**
 * @brief 将 RNG 状态前进 2^128 步（xoshiro256** jump 多项式）。
 *
 * 用于将一个 RNG 流拆分为多个不重叠的子流。
 *
 * @param[in,out] rng 已初始化的 RNG（不可为 NULL）。
 * @return LMMC_STATUS_OK 成功；LMMC_STATUS_INVALID_ARGUMENT 若 rng 为 NULL。
 *
 * @par 副作用
 * - 就地修改 @p rng 的内部状态。
 */
lmmc_status_t lmmc_rng_jump(lmmc_rng_t* rng);

/**
 * @brief 将 RNG 状态前进 2^192 步（xoshiro256** long_jump 多项式）。
 *
 * 用于在更大尺度上拆分并行流。
 *
 * @param[in,out] rng 已初始化的 RNG（不可为 NULL）。
 * @return LMMC_STATUS_OK 成功；LMMC_STATUS_INVALID_ARGUMENT 若 rng 为 NULL。
 *
 * @par 副作用
 * - 就地修改 @p rng 的内部状态。
 */
lmmc_status_t lmmc_rng_long_jump(lmmc_rng_t* rng);

/** @brief 直接获取下一个 64 位均匀整数。 */
uint64_t lmmc_rng_next_u64(lmmc_rng_t* rng);

/**
 * @brief 在 @f$[a,b)@f$ 区间内生成均匀分布样本。
 */
lmmc_status_t lmmc_rng_uniform(
    lmmc_rng_t* rng,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t* out_value
);

/**
 * @brief 生成正态分布 @f$\mathcal{N}(\mu, \sigma^2)@f$ 样本。
 *
 * 使用 Box-Muller 或 Ziggurat 方法生成标准正态样本后进行仿射变换。
 *
 * @param[in]     rng       已初始化的 RNG。
 * @param[in]     mean      均值 @f$\mu@f$ 。
 * @param[in]     stddev    标准差 @f$\sigma@f$ ，必须 > 0。
 * @param[out]    out_value 输出样本值。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 stddev <= 0 或指针为 NULL。
 *
 * @par 副作用
 * - 就地修改 @p rng 的内部状态（消耗随机数）。
 * - 不分配堆内存。
 */
lmmc_status_t lmmc_rng_normal(
    lmmc_rng_t* rng,
    lmmc_real_t mean,
    lmmc_real_t stddev,
    lmmc_real_t* out_value
);

/** @brief 生成参数为 @p rate 的指数分布样本。 */
lmmc_status_t lmmc_rng_exponential(
    lmmc_rng_t* rng,
    lmmc_real_t rate,
    lmmc_real_t* out_value
);

/** @brief 批量填充均匀分布样本到 @p array 。 */
lmmc_status_t lmmc_rng_fill_uniform(
    lmmc_rng_t* rng,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t* array,
    size_t count
);

/**
 * @brief Fisher-Yates 洗牌：就地随机重排 @p array 中的 @p count 个元素。
 * 每次 Fisher-Yates 选择均使用无偏的拒绝采样。
 *
 * @param[in]     rng       已初始化的 RNG。
 * @param[in,out] array     待洗牌的数组，就地重排。
 * @param[in]     count     元素个数。
 * @param[in]     elem_size 每个元素的字节大小。
 *
 * @par 副作用
 * - 就地修改 @p array 中元素的顺序。
 * - 就地修改 @p rng 的内部状态。
 */
lmmc_status_t lmmc_rng_shuffle(
    lmmc_rng_t* rng,
    void* array,
    size_t count,
    size_t elem_size
);

/**
 * @brief 生成 Gamma 分布样本 @f$\mathrm{Gamma}(\alpha, \beta)@f$ 。
 *
 * 使用 Marsaglia-Tsang 方法（shape >= 1），shape < 1 时使用
 * @f$X = Y \cdot U^{1/\alpha}@f$ 变换法（Y ~ Gamma(1+α, β)）。
 *
 * @param[in]  rng    已初始化的 RNG。
 * @param[in]  shape  形状参数 @f$\alpha@f$ （> 0）。
 * @param[in]  scale  尺度参数 @f$\beta@f$ （> 0）。
 * @param[out] out    输出样本值。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 shape <= 0、scale <= 0 或指针为 NULL。
 *
 * @par 副作用
 * - 就地修改 @p rng 的内部状态（消耗多个随机数，次数不确定）。
 * - 不分配堆内存。
 */
lmmc_status_t lmmc_rng_gamma(
    lmmc_rng_t* rng,
    lmmc_real_t shape,
    lmmc_real_t scale,
    lmmc_real_t* out
);

/**
 * @brief 生成 Beta 分布样本 Beta(alpha, beta)。
 *
 * @param[in]  rng    已初始化的 RNG。
 * @param[in]  alpha  参数 alpha（> 0）。
 * @param[in]  beta_param   参数 beta（> 0）。
 * @param[out] out    输出样本值。
 * @return LMMC_STATUS_OK 成功。
 */
lmmc_status_t lmmc_rng_beta(
    lmmc_rng_t* rng,
    lmmc_real_t alpha,
    lmmc_real_t beta_param,
    lmmc_real_t* out
);

/**
 * @brief 生成卡方分布样本 Chi-squared(df)。
 *
 * @param[in]  rng  已初始化的 RNG。
 * @param[in]  df   自由度（> 0）。
 * @param[out] out  输出样本值。
 * @return LMMC_STATUS_OK 成功。
 */
lmmc_status_t lmmc_rng_chi_squared(
    lmmc_rng_t* rng,
    lmmc_real_t df,
    lmmc_real_t* out
);

/**
 * @brief 生成 Student-t 分布样本 t(df)。
 *
 * @param[in]  rng  已初始化的 RNG。
 * @param[in]  df   自由度（> 0）。
 * @param[out] out  输出样本值。
 * @return LMMC_STATUS_OK 成功。
 */
lmmc_status_t lmmc_rng_student_t(
    lmmc_rng_t* rng,
    lmmc_real_t df,
    lmmc_real_t* out
);

/**
 * @brief 生成 F 分布样本 F(df1, df2)。
 *
 * @param[in]  rng  已初始化的 RNG。
 * @param[in]  df1  分子自由度（> 0）。
 * @param[in]  df2  分母自由度（> 0）。
 * @param[out] out  输出样本值。
 * @return LMMC_STATUS_OK 成功。
 */
lmmc_status_t lmmc_rng_f(
    lmmc_rng_t* rng,
    lmmc_real_t df1,
    lmmc_real_t df2,
    lmmc_real_t* out
);

/**
 * @brief 生成 Poisson 分布样本 @f$\mathrm{Poisson}(\lambda)@f$ 。
 *
 * lambda >= 10 使用 PTRD（Hörmann）算法（接受-拒绝），
 * lambda < 10 使用逆变换法（Knuth 方法）。
 *
 * @param[in]  rng     已初始化的 RNG。
 * @param[in]  lambda  期望值（> 0）。
 * @param[out] out     输出样本值（非负整数）。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 lambda <= 0 或指针为 NULL。
 *
 * @par 副作用
 * - 就地修改 @p rng 的内部状态（消耗随机数次数不确定，取决于接受-拒绝过程）。
 * - 不分配堆内存。
 */
lmmc_status_t lmmc_rng_poisson(
    lmmc_rng_t* rng,
    lmmc_real_t lambda,
    size_t* out
);

/**
 * @brief 生成二项分布样本 Binomial(n, p)。
 *
 * 小均值参数使用精确逆 CDF，其余参数使用 BTPE 接受-拒绝算法；
 * 超过 IEEE-754 精确整数范围的 n 会拆分为独立精确子问题。
 *
 * @param[in]  rng  已初始化的 RNG。
 * @param[in]  n    试验次数。
 * @param[in]  p    有限成功概率（0 <= p <= 1）。
 * @param[out] out  输出样本值；参数无效时保持不变。
 * @return LMMC_STATUS_OK 成功；参数或指针无效时返回
 *         LMMC_STATUS_INVALID_ARGUMENT。
 */
lmmc_status_t lmmc_rng_binomial(
    lmmc_rng_t* rng,
    size_t n,
    lmmc_real_t p,
    size_t* out
);

/**
 * @brief 在 [lo, hi] 闭区间内生成无偏均匀整数分布样本。
 *
 * 支持完整的 int64_t 定义域且不执行有符号溢出；lo == hi 时不消耗 RNG 状态。
 *
 * @param[in]  rng  已初始化的 RNG。
 * @param[in]  lo   下界（含）。
 * @param[in]  hi   上界（含）。
 * @param[out] out  输出样本值。
 * @return LMMC_STATUS_OK 成功；LMMC_STATUS_INVALID_ARGUMENT 若 lo > hi。
 */
lmmc_status_t lmmc_rng_int_uniform(
    lmmc_rng_t* rng,
    int64_t lo,
    int64_t hi,
    int64_t* out
);

#ifdef __cplusplus
}
#endif

#endif
