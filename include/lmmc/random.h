#ifndef LMMC_RANDOM_H
#define LMMC_RANDOM_H

#include "lmmc/config.h"
#include "lmmc/status.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 随机数生成器（不透明类型） */
typedef struct lmmc_rng_t lmmc_rng_t;

/* 生命周期管理 */
lmmc_status_t lmmc_rng_create(lmmc_rng_t** out_rng);
lmmc_status_t lmmc_rng_seed(lmmc_rng_t* rng, uint64_t seed);
void lmmc_rng_destroy(lmmc_rng_t* rng);

/* 基础生成 */
uint64_t lmmc_rng_next_u64(lmmc_rng_t* rng);

/* 分布采样 */
lmmc_status_t lmmc_rng_uniform(
    lmmc_rng_t* rng,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t* out_value
);

lmmc_status_t lmmc_rng_normal(
    lmmc_rng_t* rng,
    lmmc_real_t mean,
    lmmc_real_t stddev,
    lmmc_real_t* out_value
);

lmmc_status_t lmmc_rng_exponential(
    lmmc_rng_t* rng,
    lmmc_real_t rate,
    lmmc_real_t* out_value
);

/* 批量操作 */
lmmc_status_t lmmc_rng_fill_uniform(
    lmmc_rng_t* rng,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t* array,
    size_t count
);

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
