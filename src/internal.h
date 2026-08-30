/**
 * @file internal.h
 * @brief LMMC 内部使用的公共工具(仅源文件可见).
 *
 * 集中提供 size_t 安全算术,浮点工具与内联辅助函数,供各实现文件共享.
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
 * @brief 计算 @c a*b 并检测溢出.
 *
 * @param[out] result 乘积,溢出时未定义.
 * @return 1 表示成功,0 表示溢出.
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
 * @brief 计算 @c a+b 并检测溢出.
 * @return 1 成功,0 溢出.
 */
static inline int lmmc_safe_add_size(size_t a, size_t b, size_t *result)
{
    if (a > SIZE_MAX - b) {
        return 0;
    }
    *result = a + b;
    return 1;
}

typedef struct lmmc_storage_envelope {
    uintptr_t begin;
    uintptr_t end;
} lmmc_storage_envelope_t;

/**
 * @internal
 * @brief 计算覆盖带步长矩形存储的半开字节包络.
 *
 * 包络包含首尾可达元素之间的步长填充;零尺寸矩形的包络为空.
 */
static inline int lmmc_storage_envelope_checked(
    const void *data,
    size_t rows,
    size_t cols,
    size_t stride,
    size_t element_size,
    lmmc_storage_envelope_t *out)
{
    size_t last_row;
    size_t elements;
    size_t bytes;
    uintptr_t begin;

    if (data == NULL || out == NULL) return 0;
    begin = (uintptr_t)data;
    out->begin = begin;
    out->end = begin;
    if (rows == 0 || cols == 0) return 1;
    if (!lmmc_safe_mul_size(rows - 1, stride, &last_row) ||
        !lmmc_safe_add_size(last_row, cols, &elements) ||
        !lmmc_safe_mul_size(elements, element_size, &bytes) ||
        (uintptr_t)bytes > UINTPTR_MAX - begin) {
        return 0;
    }
    out->end = begin + (uintptr_t)bytes;
    return 1;
}

/** @internal @brief 检查两个已验证半开包络是否重叠. */
static inline int lmmc_storage_envelopes_overlap(
    const lmmc_storage_envelope_t *left,
    const lmmc_storage_envelope_t *right)
{
    return left->begin < right->end && right->begin < left->end;
}

/** @internal @brief 判断 @c *x 是否为有限数. */
static inline int lmmc_is_finite(const lmmc_real_t *x)
{
    return isfinite(*x) != 0;
}

/** @internal @brief 绝对值. */
static inline lmmc_real_t lmmc_abs(lmmc_real_t x)
{
    return fabs(x);
}

/** @internal @brief 取较大值. */
static inline lmmc_real_t lmmc_max(lmmc_real_t a, lmmc_real_t b)
{
    return (a >= b) ? a : b;
}

/** @internal @brief 取较小值. */
static inline lmmc_real_t lmmc_min(lmmc_real_t a, lmmc_real_t b)
{
    return (a <= b) ? a : b;
}

/** @internal @brief 将 @c x 截断到 [lo, hi] 区间. */
static inline lmmc_real_t lmmc_clamp(lmmc_real_t x, lmmc_real_t lo, lmmc_real_t hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/** @internal @brief 交换两个 ::lmmc_real_t . */
static inline void lmmc_swap(lmmc_real_t *a, lmmc_real_t *b)
{
    lmmc_real_t tmp = *a;
    *a = *b;
    *b = tmp;
}

#endif
