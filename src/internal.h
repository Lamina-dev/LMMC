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
#include "lmmc/dense.h"
#include <math.h>

typedef struct lmmc_rng_t lmmc_rng_t;

/** Return the allocation-free default RNG for the current thread. */
lmmc_rng_t* lmmc_rng_default_get(void);

/** Clear only the current thread's default RNG state. */
void lmmc_rng_default_reset(void);
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

/**
 * @internal
 * @brief Validate a public dense-matrix descriptor before indexing it.
 *
 * This validates descriptor shape and all address arithmetic LMMC can check;
 * callers remain responsible for providing storage large enough for the
 * declared envelope.
 */
static inline int lmmc_mat_descriptor_is_valid(const lmmc_mat_t *matrix)
{
    lmmc_storage_envelope_t envelope;
    return matrix != NULL && matrix->data != NULL &&
           matrix->rows != 0 && matrix->cols != 0 &&
           matrix->stride >= matrix->cols &&
           lmmc_storage_envelope_checked(
               matrix->data, matrix->rows, matrix->cols, matrix->stride,
               sizeof(lmmc_real_t), &envelope);
}

/** @internal @brief Validate a public dense-vector descriptor. */
static inline int lmmc_vec_descriptor_is_valid(const lmmc_vec_t *vector)
{
    lmmc_storage_envelope_t envelope;
    return vector != NULL && vector->data != NULL && vector->size != 0 &&
           lmmc_storage_envelope_checked(
               vector->data, 1, vector->size, vector->size,
               sizeof(lmmc_real_t), &envelope);
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
    return LMMC_REAL_IS_FINITE(x) ? 1 : 0;
}

/**
 * @internal
 * @brief LAPACK DLASSQ 风格的缩放平方和累加器.
 *
 * 保持 sum(x_i^2) = scale^2 * sumsq，避免直接平方造成的中间
 * 溢出或下溢。最终范数超出 binary64 表示域时仍按 IEEE-754 溢出。
 */
typedef struct lmmc_scaled_sumsq {
    lmmc_real_t scale;
    lmmc_real_t sumsq;
} lmmc_scaled_sumsq_t;

static inline void lmmc_scaled_sumsq_init(lmmc_scaled_sumsq_t *acc)
{
    acc->scale = 0.0;
    acc->sumsq = 1.0;
}

static inline void lmmc_scaled_sumsq_add(
    lmmc_scaled_sumsq_t *acc, lmmc_real_t value)
{
    lmmc_real_t magnitude = fabs(value);
    if (isnan(acc->scale)) return;
    if (isnan(magnitude)) {
        acc->scale = magnitude;
        acc->sumsq = 1.0;
        return;
    }
    if (magnitude == 0.0 || isinf(acc->scale)) return;
    if (isinf(magnitude)) {
        acc->scale = magnitude;
        acc->sumsq = 1.0;
        return;
    }
    if (acc->scale < magnitude) {
        lmmc_real_t ratio = acc->scale / magnitude;
        acc->sumsq = 1.0 + acc->sumsq * ratio * ratio;
        acc->scale = magnitude;
    } else {
        lmmc_real_t ratio = magnitude / acc->scale;
        acc->sumsq += ratio * ratio;
    }
}

static inline lmmc_real_t lmmc_scaled_sumsq_norm(
    const lmmc_scaled_sumsq_t *acc)
{
    return acc->scale == 0.0 ? 0.0 : acc->scale * sqrt(acc->sumsq);
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
