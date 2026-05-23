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

#ifdef __cplusplus
}
#endif

#endif
