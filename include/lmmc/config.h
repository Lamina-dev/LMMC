/**
 * @file config.h
 * @brief LMMC binary real type and arithmetic spelling.
 *
 * LMMC's C ABI is fixed to IEEE-754 binary64 (`double`). The operation
 * macros centralize initialization and arithmetic syntax; they are not a
 * pluggable precision backend.
 */
#ifndef LMMC_CONFIG_H
#define LMMC_CONFIG_H

#include <math.h>

/* C++ expansion must not resolve libm calls in the caller's namespace. */
#ifdef __cplusplus
#define LMMC_LIBM(function) ::function
#else
#define LMMC_LIBM(function) function
#endif

/**
 * @brief Public and internal LMMC real type.
 *
 * This is deliberately fixed to @c double. Changing it is an ABI-breaking
 * source migration requiring coordinated implementations of every direct
 * arithmetic and libm call.
 */
typedef double lmmc_real_t;

/** @brief 用于 @c printf 的 ::lmmc_real_t 格式说明符。 */
#define LMMC_PRI_REAL "f"
/** @brief 用于 @c scanf 的 ::lmmc_real_t 格式说明符。 */
#define LMMC_SCN_REAL "lf"

/** @brief 初始化一个 ::lmmc_real_t 变量（POD 类型为空操作）。 */
#define LMMC_REAL_INIT(x)           ((void)0)
/** @brief 释放一个 ::lmmc_real_t 变量（POD 类型为空操作）。 */
#define LMMC_REAL_CLEAR(x)          ((void)0)

/** @brief 将一个 ::lmmc_real_t 赋值到另一个：@c *dst = *src 。 */
#define LMMC_REAL_SET(dst, src)     (*(dst) = *(src))
/** @brief 将一个 @c double 字面量赋值到 ::lmmc_real_t 。 */
#define LMMC_REAL_SET_D(dst, dbl)   (*(dst) = (dbl))

/** @brief 计算 @c *res = *a + *b 。 */
#define LMMC_REAL_ADD(res, a, b)    (*(res) = *(a) + *(b))
/** @brief 计算 @c *res = *a - *b 。 */
#define LMMC_REAL_SUB(res, a, b)    (*(res) = *(a) - *(b))
/** @brief 计算 @c *res = *a * *b 。 */
#define LMMC_REAL_MUL(res, a, b)    (*(res) = *(a) * *(b))
/** @brief 计算 @c *res = *a / *b 。 */
#define LMMC_REAL_DIV(res, a, b)    (*(res) = *(a) / *(b))

/** @brief 取相反数，@c *res = -*a 。 */
#define LMMC_REAL_NEG(res, a)       (*(res) = -*(a))
/** @brief 计算平方根，@c *res = sqrt(*a) 。 */
#define LMMC_REAL_SQRT(res, a)      (*(res) = LMMC_LIBM(sqrt)(*(a)))
/** @brief 计算绝对值，@c *res = |*a| 。 */
#define LMMC_REAL_ABS(res, a)       (*(res) = LMMC_LIBM(fabs)(*(a)))

/** @brief 计算自然指数，@c *res = exp(*a) 。 */
#define LMMC_REAL_EXP(res, a)       (*(res) = LMMC_LIBM(exp)(*(a)))
/** @brief 计算自然对数，@c *res = log(*a) 。 */
#define LMMC_REAL_LOG(res, a)       (*(res) = LMMC_LIBM(log)(*(a)))
/** @brief 计算正弦，@c *res = sin(*a) 。 */
#define LMMC_REAL_SIN(res, a)       (*(res) = LMMC_LIBM(sin)(*(a)))
/** @brief 计算余弦，@c *res = cos(*a) 。 */
#define LMMC_REAL_COS(res, a)       (*(res) = LMMC_LIBM(cos)(*(a)))
/** @brief 计算正切，@c *res = tan(*a) 。 */
#define LMMC_REAL_TAN(res, a)       (*(res) = LMMC_LIBM(tan)(*(a)))

/** @brief 计算双曲余弦，@c *res = cosh(*a) 。 */
#define LMMC_REAL_COSH(res, a)      (*(res) = LMMC_LIBM(cosh)(*(a)))
/** @brief 计算双曲正弦，@c *res = sinh(*a) 。 */
#define LMMC_REAL_SINH(res, a)      (*(res) = LMMC_LIBM(sinh)(*(a)))
/** @brief 计算反正切2，@c *res = atan2(*y, *x) 。 */
#define LMMC_REAL_ATAN2(res, y, x)  (*(res) = LMMC_LIBM(atan2)(*(y), *(x)))

/** @brief 比较 @c *a 与 @c *b ，返回 -1/0/1 。 */
#define LMMC_REAL_CMP(a, b)         ((*(a) > *(b)) ? 1 : ((*(a) < *(b)) ? -1 : 0))
/** @brief 判断 @c *x 是否为有限数。 */
#define LMMC_REAL_IS_FINITE(x)      LMMC_LIBM(isfinite)(*(x))

/** @brief 默认的机器精度阈值，用于近似相等比较。 */
#define LMMC_REAL_EPSILON  ((lmmc_real_t)1e-15)
/** @brief 圆周率常数 @f$\pi@f$ 。 */
#define LMMC_CONST_PI      ((lmmc_real_t)3.14159265358979323846)

#endif
