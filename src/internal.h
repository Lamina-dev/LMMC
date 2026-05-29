/**
 * @file internal.h
 * @brief LMMC 内部使用的公共工具（仅源文件可见）。
 *
 * 提供 size_t 安全算术、浮点工具与内联辅助函数，避免在多个 .c 文件中重复实现。
 *
 * @internal
 */
#ifndef LMMC_INTERNAL_H
#define LMMC_INTERNAL_H

#include "lmmc/config.h"
#include <stddef.h>
#include <stdint.h>
#include <math.h>

/**
 * @internal
 * @brief 计算 @c a*b 并检测溢出。
 *
 * @param[out] result 乘积，溢出时未定义。
 * @return 1 表示成功，0 表示溢出。
 */
static inline int lmmc_safe_mul_size(size_t a, size_t b, size_t *result)
{
    if (a == 0 || b == 0) {
        *result = 0;
        return 1;
    }
    if (a > SIZE_MAX / b) {
        return 0;
    }
    *result = a * b;
    return 1;
}

/**
 * @internal
 * @brief 计算 @c a+b 并检测溢出。
 * @return 1 成功，0 溢出。
 */
static inline int lmmc_safe_add_size(size_t a, size_t b, size_t *result)
{
    if (a > SIZE_MAX - b) {
        return 0;
    }
    *result = a + b;
    return 1;
}

/** @internal @brief 判断 @c *x 是否为有限数。 */
static inline int lmmc_is_finite(const lmmc_real_t *x)
{
    return isfinite(*x) != 0;
}

/** @internal @brief 绝对值。 */
static inline lmmc_real_t lmmc_abs(lmmc_real_t x)
{
    return fabs(x);
}

/** @internal @brief 取较大值。 */
static inline lmmc_real_t lmmc_max(lmmc_real_t a, lmmc_real_t b)
{
    return (a >= b) ? a : b;
}

/** @internal @brief 取较小值。 */
static inline lmmc_real_t lmmc_min(lmmc_real_t a, lmmc_real_t b)
{
    return (a <= b) ? a : b;
}

/** @internal @brief 将 @c x 截断到 [lo, hi] 区间。 */
static inline lmmc_real_t lmmc_clamp(lmmc_real_t x, lmmc_real_t lo, lmmc_real_t hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/** @internal @brief 交换两个 ::lmmc_real_t 。 */
static inline void lmmc_swap(lmmc_real_t *a, lmmc_real_t *b)
{
    lmmc_real_t tmp = *a;
    *a = *b;
    *b = tmp;
}

#endif
