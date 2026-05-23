/**
 * @file numeric.h
 * @brief 数值常量、特殊值、近似比较与基本数学补充函数。
 *
 * 此模块包含 IEEE-754 特殊值生成 / 检测、扩展三角与对数函数、
 * 浮点拆分、近似相等判定，以及一个轻量的 radix-4 FFT 与
 * Lambert W 函数。
 */
#ifndef LMMC_NUMERIC_H
#define LMMC_NUMERIC_H

#include <stddef.h>
#include "lmmc/config.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LMMC_PI
/** @brief @f$\pi@f$ 常量。 */
#define LMMC_PI (3.14159265358979323846)
#endif
#ifndef LMMC_INV_PI
/** @brief @f$1/\pi@f$ 常量。 */
#define LMMC_INV_PI (0.31830988618379067154)
#endif
#ifndef LMMC_SQRT2
/** @brief @f$\sqrt{2}@f$ 常量。 */
#define LMMC_SQRT2 (1.41421356237309504880)
#endif

#ifndef LMMC_DEFAULT_ABS_TOL
/** @brief 库默认的绝对容差。 */
#define LMMC_DEFAULT_ABS_TOL 1e-12
#endif
#ifndef LMMC_DEFAULT_REL_TOL
/** @brief 库默认的相对容差。 */
#define LMMC_DEFAULT_REL_TOL 1e-10
#endif

/* ===================== 特殊值生成 ===================== */

/** @brief 写入 IEEE-754 正无穷。 */
lmmc_status_t lmmc_inf(lmmc_real_t* out_inf);
/** @brief 写入 IEEE-754 NaN。 */
lmmc_status_t lmmc_nan(lmmc_real_t* out_nan);
/** @brief 写入机器精度（1 与下一个可表示数的差）。 */
lmmc_status_t lmmc_eps(lmmc_real_t* out_eps);

/* ===================== 浮点分类 ===================== */

/** @brief 检测是否为 NaN ，结果写入 @c *out_isnan （0/1）。 */
lmmc_status_t lmmc_isnan(lmmc_real_t x, int* out_isnan);
/** @brief 检测是否为 ±∞ 。 */
lmmc_status_t lmmc_isinf(lmmc_real_t x, int* out_isinf);
/** @brief 检测是否为有限数（非 NaN 且非 ±∞）。 */
lmmc_status_t lmmc_isfinite(lmmc_real_t x, int* out_isfinite);
/** @brief 检测符号位（含 -0 ）。 */
lmmc_status_t lmmc_signbit(lmmc_real_t x, int* out_signbit);

/* ===================== 三角与几何 ===================== */

/** @brief 计算 @f$\mathrm{atan2}(y,x)@f$ 。 */
lmmc_status_t lmmc_atan2(lmmc_real_t y, lmmc_real_t x, lmmc_real_t* out_res);
/** @brief 同时计算 sin/cos （某些平台可加速）。 */
lmmc_status_t lmmc_sincos(lmmc_real_t x, lmmc_real_t* out_sin, lmmc_real_t* out_cos);
/** @brief 计算 @f$\sqrt{x^2+y^2}@f$ ，避免溢出。 */
lmmc_status_t lmmc_hypot(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res);

/* ===================== 指数与对数补充 ===================== */

/** @brief 计算 @f$2^x@f$ 。 */
lmmc_status_t lmmc_exp2(lmmc_real_t x, lmmc_real_t* out_res);
/** @brief 计算 @f$\log_2 x@f$ 。 */
lmmc_status_t lmmc_log2(lmmc_real_t x, lmmc_real_t* out_res);
/** @brief 计算 @f$e^x-1@f$ ，对小 x 更精确。 */
lmmc_status_t lmmc_expm1(lmmc_real_t x, lmmc_real_t* out_res);
/** @brief 计算 @f$\log(1+x)@f$ ，对小 x 更精确。 */
lmmc_status_t lmmc_log1p(lmmc_real_t x, lmmc_real_t* out_res);

/* ===================== 浮点分解 ===================== */

/** @brief 拆分整数部分与小数部分：x = iptr + frac 。 */
lmmc_status_t lmmc_split_int_frac(lmmc_real_t x, lmmc_real_t* out_iptr, lmmc_real_t* out_frac);
/** @brief 计算 @f$x \mod y@f$ （fmod 语义）。 */
lmmc_status_t lmmc_fmod(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res);
/** @brief 计算 @f$x \cdot 2^{exp}@f$ 。 */
lmmc_status_t lmmc_ldexp(lmmc_real_t x, int exp, lmmc_real_t* out_res);
/** @brief 计算 @c x 朝 @c y 方向的下一个可表示浮点数。 */
lmmc_status_t lmmc_nextafter(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res);

/* ===================== 近似相等比较 ===================== */

/**
 * @brief 简单 epsilon 比较：@f$|a-b| \le \epsilon@f$ 。
 */
lmmc_status_t lmmc_approx_eq(lmmc_real_t a, lmmc_real_t b, lmmc_real_t epsilon, int* out_equal);

/**
 * @brief 同时考虑绝对与相对容差的近似相等：
 * @f$|a-b| \le \max(\mathrm{abs\_tol}, \mathrm{rel\_tol}\cdot\max(|a|,|b|))@f$ 。
 */
lmmc_status_t lmmc_double_nearly_equal_tol(
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t abs_tol,
    lmmc_real_t rel_tol,
    int* out_equal
);

/** @brief 使用默认容差的近似相等比较，等价于以 ::LMMC_DEFAULT_ABS_TOL / ::LMMC_DEFAULT_REL_TOL 调用 ::lmmc_double_nearly_equal_tol 。 */
lmmc_status_t lmmc_double_nearly_equal(lmmc_real_t a, lmmc_real_t b, int* out_equal);

/* ===================== 离散傅里叶变换（radix-4） ===================== */

/**
 * @brief 计算不小于 @p n 的最近 4 的幂，用于 FFT 长度对齐。
 */
lmmc_status_t lmmc_fft_radix4_next_size(size_t n, size_t* out_nfft);

/**
 * @brief 通用 radix-4 FFT 入口。
 *
 * @param[in,out] real    实部数组，长度 @p n （要求为 4 的幂）。
 * @param[in,out] imag    虚部数组，长度 @p n 。
 * @param[in]     n       FFT 长度，必须为 4 的幂。
 * @param[in]     inverse 非 0 时执行逆变换并归一化。
 */
lmmc_status_t lmmc_fft_radix4(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse);

/** @brief 等价于 ::lmmc_fft_radix4(real, imag, n, 0) 。 */
lmmc_status_t lmmc_fft_radix4_forward(lmmc_real_t* real, lmmc_real_t* imag, size_t n);

/** @brief 等价于 ::lmmc_fft_radix4(real, imag, n, 1) 。 */
lmmc_status_t lmmc_fft_radix4_inverse(lmmc_real_t* real, lmmc_real_t* imag, size_t n);

/* ===================== 特殊函数 ===================== */

/**
 * @brief 计算 Lambert W 函数主分支 @f$W_0(z)@f$ ，要求 @f$z \ge -1/e@f$ 。
 */
lmmc_status_t lmmc_lambertw(lmmc_real_t z, lmmc_real_t* out_res);

#ifdef __cplusplus
}
#endif

#endif
