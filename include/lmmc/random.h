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

/** @brief 创建发生器，初始种子由库实现确定。 */
lmmc_status_t lmmc_rng_create(lmmc_rng_t** out_rng);
/** @brief 重新设定种子。 */
lmmc_status_t lmmc_rng_seed(lmmc_rng_t* rng, uint64_t seed);
/** @brief 销毁发生器。 */
void lmmc_rng_destroy(lmmc_rng_t* rng);

/**
 * @brief 深拷贝 RNG 状态。
 *
 * 调用方可对副本调用 lmmc_rng_jump 以获得独立的并行流。
 *
 * @param[in]  src     源 RNG（不可为 NULL）。
 * @param[out] out_rng 输出新分配的 RNG 副本。
 * @return LMMC_STATUS_OK 成功；LMMC_STATUS_INVALID_ARGUMENT 若 src 或 out_rng 为 NULL。
 */
lmmc_status_t lmmc_rng_clone(const lmmc_rng_t* src, lmmc_rng_t** out_rng);

/**
 * @brief 将 RNG 状态前进 2^128 步（xoshiro256** jump 多项式）。
 *
 * 用于将一个 RNG 流拆分为多个不重叠的子流。
 *
 * @param[in,out] rng 已初始化的 RNG（不可为 NULL）。
 * @return LMMC_STATUS_OK 成功；LMMC_STATUS_INVALID_ARGUMENT 若 rng 为 NULL。
 */
lmmc_status_t lmmc_rng_jump(lmmc_rng_t* rng);

/**
 * @brief 将 RNG 状态前进 2^192 步（xoshiro256** long_jump 多项式）。
 *
 * 用于在更大尺度上拆分并行流。
 *
 * @param[in,out] rng 已初始化的 RNG（不可为 NULL）。
 * @return LMMC_STATUS_OK 成功；LMMC_STATUS_INVALID_ARGUMENT 若 rng 为 NULL。
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

/** @brief 生成正态分布 @f$\mathcal{N}(\mu, \sigma^2)@f$ 样本。 */
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
 *
 * @param[in]     elem_size 每个元素的字节大小。
 */
lmmc_status_t lmmc_rng_shuffle(
    lmmc_rng_t* rng,
    void* array,
    size_t count,
    size_t elem_size
);

/**
 * @brief 生成 Gamma 分布样本 Gamma(shape, scale)。
 *
 * 使用 Marsaglia-Tsang 方法（shape >= 1），shape < 1 时使用变换法。
 *
 * @param[in]  rng    已初始化的 RNG。
 * @param[in]  shape  形状参数（> 0）。
 * @param[in]  scale  尺度参数（> 0）。
 * @param[out] out    输出样本值。
 * @return LMMC_STATUS_OK 成功。
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
 * @brief 生成 Poisson 分布样本 Poisson(lambda)。
 *
 * lambda >= 10 使用 PTRD（Hörmann）算法，lambda < 10 使用逆变换法。
 *
 * @param[in]  rng     已初始化的 RNG。
 * @param[in]  lambda  期望值（> 0）。
 * @param[out] out     输出样本值。
 * @return LMMC_STATUS_OK 成功。
 */
lmmc_status_t lmmc_rng_poisson(
    lmmc_rng_t* rng,
    lmmc_real_t lambda,
    size_t* out
);

/**
 * @brief 生成二项分布样本 Binomial(n, p)。
 *
 * @param[in]  rng  已初始化的 RNG。
 * @param[in]  n    试验次数。
 * @param[in]  p    成功概率（0 <= p <= 1）。
 * @param[out] out  输出样本值。
 * @return LMMC_STATUS_OK 成功。
 */
lmmc_status_t lmmc_rng_binomial(
    lmmc_rng_t* rng,
    size_t n,
    lmmc_real_t p,
    size_t* out
);

/**
 * @brief 在 [lo, hi] 闭区间内生成均匀整数分布样本。
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
