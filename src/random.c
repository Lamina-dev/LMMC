/*
 * random.c - 随机数生成模块
 *
 * 实现 xoshiro256** 伪随机数引擎和分布采样。
 * 使用 SplitMix64 将单个 64 位种子扩展为 256 位内部状态。
 */

#include <string.h>
#include <math.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/random.h"
#include "lmmc/status.h"

/* ========================================================================
 * 内部结构定义
 * ======================================================================== */

struct lmmc_rng_t {
    uint64_t state[4];  /* xoshiro256** 的 256 位状态 */
};

/* ========================================================================
 * 内部辅助函数
 * ======================================================================== */

/* 左旋转 */
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/* SplitMix64：用于从单个种子扩展出多个状态字 */
static inline uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

/* 默认种子值（非零，确保初始状态有效） */
#define LMMC_RNG_DEFAULT_SEED UINT64_C(0x12345678DEADBEEF)

/* ========================================================================
 * xoshiro256** 核心算法
 * ======================================================================== */

/*
 * xoshiro256** 1.0 - 生成下一个 64 位伪随机数
 * 参考：https://prng.di.unimi.it/xoshiro256starstar.c
 */
static inline uint64_t xoshiro256ss_next(uint64_t* s) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
}

/* ========================================================================
 * 公共 API：生命周期管理
 * ======================================================================== */

lmmc_status_t lmmc_rng_create(lmmc_rng_t** out_rng) {
    lmmc_rng_t* rng;

    if (out_rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    rng = (lmmc_rng_t*)lmmc_alloc(sizeof(lmmc_rng_t));
    if (rng == NULL) {
        *out_rng = NULL;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* 使用默认种子初始化状态 */
    lmmc_rng_seed(rng, LMMC_RNG_DEFAULT_SEED);

    *out_rng = rng;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_seed(lmmc_rng_t* rng, uint64_t seed) {
    uint64_t sm_state;

    if (rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* 使用 SplitMix64 将单个 64 位种子扩展为 4 个 64 位状态字 */
    sm_state = seed;
    rng->state[0] = splitmix64_next(&sm_state);
    rng->state[1] = splitmix64_next(&sm_state);
    rng->state[2] = splitmix64_next(&sm_state);
    rng->state[3] = splitmix64_next(&sm_state);

    return LMMC_STATUS_OK;
}

void lmmc_rng_destroy(lmmc_rng_t* rng) {
    if (rng != NULL) {
        /* 清零状态以避免信息泄漏 */
        memset(rng->state, 0, sizeof(rng->state));
        lmmc_free(rng);
    }
}

/* ========================================================================
 * 公共 API：基础生成
 * ======================================================================== */

uint64_t lmmc_rng_next_u64(lmmc_rng_t* rng) {
    if (rng == NULL) {
        return 0;
    }
    return xoshiro256ss_next(rng->state);
}

/* ========================================================================
 * 内部辅助：u64 转 [0, 1) 双精度浮点
 * ======================================================================== */

/*
 * 将 64 位无符号整数转换为 [0, 1) 区间的 double。
 * 使用高 53 位除以 2^53，保证均匀性。
 */
static inline double u64_to_double01(uint64_t x) {
    return (double)(x >> 11) * (1.0 / 9007199254740992.0); /* 1.0 / 2^53 */
}

/* ========================================================================
 * 公共 API：分布采样
 * ======================================================================== */

lmmc_status_t lmmc_rng_uniform(
    lmmc_rng_t* rng,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t* out_value)
{
    double u;

    if (rng == NULL || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    u = u64_to_double01(xoshiro256ss_next(rng->state));
    *out_value = a + (b - a) * u;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_normal(
    lmmc_rng_t* rng,
    lmmc_real_t mean,
    lmmc_real_t stddev,
    lmmc_real_t* out_value)
{
    double u1, u2, r, theta, z;

    if (rng == NULL || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (stddev <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Box-Muller 变换：从两个均匀样本生成正态样本 */
    /* 确保 u1 不为零（避免 log(0)） */
    do {
        u1 = u64_to_double01(xoshiro256ss_next(rng->state));
    } while (u1 == 0.0);

    u2 = u64_to_double01(xoshiro256ss_next(rng->state));

    r = sqrt(-2.0 * log(u1));
    theta = 2.0 * LMMC_CONST_PI * u2;
    z = r * cos(theta);

    *out_value = mean + stddev * z;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_exponential(
    lmmc_rng_t* rng,
    lmmc_real_t rate,
    lmmc_real_t* out_value)
{
    double u;

    if (rng == NULL || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (rate <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* 逆变换法：X = -ln(1 - U) / rate */
    /* 确保 1-u 不为零 */
    do {
        u = u64_to_double01(xoshiro256ss_next(rng->state));
    } while (u == 1.0);

    *out_value = -log(1.0 - u) / rate;
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * 公共 API：批量操作
 * ======================================================================== */

lmmc_status_t lmmc_rng_fill_uniform(
    lmmc_rng_t* rng,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t* array,
    size_t count)
{
    size_t i;
    double u;

    if (rng == NULL || array == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < count; i++) {
        u = u64_to_double01(xoshiro256ss_next(rng->state));
        array[i] = a + (b - a) * u;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_shuffle(
    lmmc_rng_t* rng,
    void* array,
    size_t count,
    size_t elem_size)
{
    size_t i, j;
    unsigned char* arr;
    unsigned char* tmp;

    if (rng == NULL || array == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (elem_size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* 0 或 1 个元素无需洗牌 */
    if (count <= 1) {
        return LMMC_STATUS_OK;
    }

    arr = (unsigned char*)array;
    tmp = (unsigned char*)lmmc_alloc(elem_size);
    if (tmp == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* Fisher-Yates 洗牌算法（从后向前） */
    for (i = count - 1; i > 0; i--) {
        /* 生成 [0, i] 范围内的随机索引 */
        uint64_t r = xoshiro256ss_next(rng->state);
        j = (size_t)(r % (i + 1));

        /* 交换 arr[i] 和 arr[j] */
        if (i != j) {
            memcpy(tmp, arr + i * elem_size, elem_size);
            memcpy(arr + i * elem_size, arr + j * elem_size, elem_size);
            memcpy(arr + j * elem_size, tmp, elem_size);
        }
    }

    lmmc_free(tmp);
    return LMMC_STATUS_OK;
}
