#ifndef LMMC_NUMERIC_SCALAR_H
#define LMMC_NUMERIC_SCALAR_H

#include <stddef.h>
#include "lmmc/config.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LMMC_PI
/** @brief @f$\pi@f$ 常量. */
#define LMMC_PI (3.14159265358979323846)
#endif
#ifndef LMMC_INV_PI
/** @brief @f$1/\pi@f$ 常量. */
#define LMMC_INV_PI (0.31830988618379067154)
#endif
#ifndef LMMC_SQRT2
/** @brief @f$\sqrt{2}@f$ 常量. */
#define LMMC_SQRT2 (1.41421356237309504880)
#endif
#ifndef LMMC_INV_E
/** @brief @f$1/e@f$ 常量,用于 Lambert W 函数域检查. */
#define LMMC_INV_E (0.36787944117144232160)
#endif

#ifndef LMMC_DEFAULT_ABS_TOL
/** @brief 库默认的绝对容差. */
#define LMMC_DEFAULT_ABS_TOL 1e-12
#endif
#ifndef LMMC_DEFAULT_REL_TOL
/** @brief 库默认的相对容差. */
#define LMMC_DEFAULT_REL_TOL 1e-10
#endif


/** @brief 写入 IEEE-754 正无穷. */
lmmc_status_t lmmc_inf(lmmc_real_t* out_inf);
/** @brief 写入 IEEE-754 NaN. */
lmmc_status_t lmmc_nan(lmmc_real_t* out_nan);
/** @brief 写入机器精度(1 与下一个可表示数的差). */
lmmc_status_t lmmc_eps(lmmc_real_t* out_eps);


/** @brief 检测是否为 NaN ,结果写入 @c *out_isnan (0/1). */
lmmc_status_t lmmc_isnan(lmmc_real_t x, int* out_isnan);
/** @brief 检测是否为 +/-infinity . */
lmmc_status_t lmmc_isinf(lmmc_real_t x, int* out_isinf);
/** @brief 检测是否为有限数(非 NaN 且非 +/-infinity). */
lmmc_status_t lmmc_isfinite(lmmc_real_t x, int* out_isfinite);
/** @brief 检测符号位(含 -0 ). */
lmmc_status_t lmmc_signbit(lmmc_real_t x, int* out_signbit);


/** @brief 计算 @f$\mathrm{atan2}(y,x)@f$ . */
lmmc_status_t lmmc_atan2(lmmc_real_t y, lmmc_real_t x, lmmc_real_t* out_res);
/**
 * @brief 计算同一有限弧度值的正弦和余弦.
 * @param[in] x 有限输入弧度.
 * @param[out] out_sin @p x 的正弦.
 * @param[out] out_cos @p x 的余弦.
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空、两个输出相同或 @p x 非有限;
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若任一计算结果超出有限值范围.
 * 两个输出仅在参数与结果验证均成功后写入.
 * @see POSIX.1-2024, sin
 * https://pubs.opengroup.org/onlinepubs/9799919799/functions/sin.html
 */
lmmc_status_t lmmc_sincos(lmmc_real_t x, lmmc_real_t* out_sin, lmmc_real_t* out_cos);
/**
 * @brief 缩放计算欧氏距离 @f$\sqrt{x^2+y^2}@f$。
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或任一输入非有限;
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果上溢为非有限值.
 * 可表示的次正规结果正常返回；输出仅在成功时写入.
 */
lmmc_status_t lmmc_hypot(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res);


/**
 * @brief 计算 @f$2^x@f$ .
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或 @p x 非有限;
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果上溢为非有限值.
 * 输出仅在成功时写入。
 */
lmmc_status_t lmmc_exp2(lmmc_real_t x, lmmc_real_t* out_res);
/**
 * @brief 计算 @f$\log_2 x@f$ .
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或 @p x 非有限;
 *         ::LMMC_STATUS_OUT_OF_RANGE 若 @p x 小于或等于零;
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果不可表示为有限值.
 * 输出仅在成功时写入。
 */
lmmc_status_t lmmc_log2(lmmc_real_t x, lmmc_real_t* out_res);
/**
 * @brief 计算 @f$e^x-1@f$ ,对小 x 更精确.
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或 @p x 非有限;
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果上溢为非有限值.
 * 输出仅在成功时写入.
 */
lmmc_status_t lmmc_expm1(lmmc_real_t x, lmmc_real_t* out_res);
/**
 * @brief 计算 @f$\log(1+x)@f$ ,对小 x 更精确.
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或 @p x 非有限;
 *         ::LMMC_STATUS_OUT_OF_RANGE 若 @p x 小于或等于 -1;
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果不可表示为有限值.
 * 输出仅在成功时写入。
 */
lmmc_status_t lmmc_log1p(lmmc_real_t x, lmmc_real_t* out_res);


/**
 * @brief 拆分整数部分与小数部分:@f$x=\mathrm{iptr}+\mathrm{frac}@f$ .
 * @param[in] x 待拆分值.
 * @param[out] out_iptr 整数部分.
 * @param[out] out_frac 带 @p x 符号的小数部分.
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或两个输出相同.
 * 两个输出均在参数验证完成后写入.
 */
lmmc_status_t lmmc_split_int_frac(lmmc_real_t x, lmmc_real_t* out_iptr, lmmc_real_t* out_frac);
/**
 * @brief 计算浮点余数，结果符号与 @p x 相同，幅值小于 @p y 的幅值.
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或任一输入为 NaN;
 *         ::LMMC_STATUS_OUT_OF_RANGE 若 @p x 为无穷或 @p y 为正零或负零;
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果超出有限值范围.
 * 有限 @p x 与无穷 @p y 的余数为 @p x；零结果保留 @p x 的符号.
 * 输出仅在成功时写入.
 * @see POSIX.1-2024, fmod
 * https://pubs.opengroup.org/onlinepubs/9799919799/functions/fmod.html
 */
lmmc_status_t lmmc_fmod(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res);
/**
 * @brief 计算 @f$x \cdot 2^{exp}@f$ .
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或 @p x 非有限;
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果上溢为非有限值，或非零结果下溢为零.
 * 输出仅在成功时写入.
 */
lmmc_status_t lmmc_ldexp(lmmc_real_t x, int exp, lmmc_real_t* out_res);
/** @brief 计算 @c x 朝 @c y 方向的下一个可表示浮点数. */
lmmc_status_t lmmc_nextafter(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res);


/**
 * @brief 按绝对容差比较：@f$|a-b| \le \epsilon@f$。
 * @param[in] a 第一个比较值.
 * @param[in] b 第二个比较值.
 * @param[in] epsilon 有限且非负的绝对容差.
 * @param[out] out_equal 输出比较结果.
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或容差无效.
 */
lmmc_status_t lmmc_approx_eq(lmmc_real_t a, lmmc_real_t b, lmmc_real_t epsilon, int* out_equal);

/**
 * @brief 同时考虑绝对与相对容差的近似相等判定.
 *
 * 判定条件:@f$|a-b| \le \max(\mathrm{abs\_tol},\; \mathrm{rel\_tol}\cdot\max(|a|,|b|))@f$ .
 *
 * @param[in]  a        第一个比较值.
 * @param[in]  b        第二个比较值.
 * @param[in]  abs_tol  绝对容差(>= 0).
 * @param[in]  rel_tol  相对容差(>= 0).
 * @param[out] out_equal 输出比较结果:1 表示近似相等,0 表示不等.
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out_equal 为 NULL 或容差为负.
 *
 * @note 不分配内存，输入保持原值。
 */
lmmc_status_t lmmc_double_nearly_equal_tol(
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t abs_tol,
    lmmc_real_t rel_tol,
    int* out_equal
);

/** @brief 使用默认容差的近似相等比较,等价于以 ::LMMC_DEFAULT_ABS_TOL / ::LMMC_DEFAULT_REL_TOL 调用 ::lmmc_double_nearly_equal_tol . */
lmmc_status_t lmmc_double_nearly_equal(lmmc_real_t a, lmmc_real_t b, int* out_equal);
/**
 * @brief 实标量函数的公共状态约定。
 *
 * 所有输入必须有限，否则返回 LMMC_STATUS_INVALID_ARGUMENT。定义域或极点
 * 错误返回 LMMC_STATUS_OUT_OF_RANGE；有限输入产生不可表示结果时返回
 * LMMC_STATUS_NUMERICAL_FAILURE。输出仅在成功时写入。
 */

/**
 * @brief 计算反正弦 @f$\arcsin(x)@f$ .
 *
 * @param[in]  x   输入值,必须满足 @f$|x| \le 1@f$ .
 * @param[out] out 输出 @f$\arcsin(x)@f$ ,结果在 @f$[-\pi/2, \pi/2]@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL;
 *         ::LMMC_STATUS_OUT_OF_RANGE 若 @f$|x| > 1@f$ .
 */
lmmc_status_t lmmc_asin(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算反余弦 @f$\arccos(x)@f$ .
 *
 * @param[in]  x   输入值,必须满足 @f$|x| \le 1@f$ .
 * @param[out] out 输出 @f$\arccos(x)@f$ ,结果在 @f$[0, \pi]@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL;
 *         ::LMMC_STATUS_OUT_OF_RANGE 若 @f$|x| > 1@f$ .
 */
lmmc_status_t lmmc_acos(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算反正切 @f$\arctan(x)@f$ .
 *
 * @param[in]  x   输入值,无定义域限制.
 * @param[out] out 输出 @f$\arctan(x)@f$ ,结果在 @f$(-\pi/2, \pi/2)@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL.
 */
lmmc_status_t lmmc_atan(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算双曲正弦 @f$\sinh(x)@f$ .
 *
 * @param[in]  x   输入值,无定义域限制.
 * @param[out] out 输出 @f$\sinh(x)@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输入非有限或 out 为 NULL；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果溢出。
 */
lmmc_status_t lmmc_sinh(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算双曲余弦 @f$\cosh(x)@f$ .
 *
 * @param[in]  x   输入值,无定义域限制.
 * @param[out] out 输出 @f$\cosh(x)@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输入非有限或 out 为 NULL；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果溢出。
 */
lmmc_status_t lmmc_cosh(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算双曲正切 @f$\tanh(x)@f$ .
 *
 * @param[in]  x   输入值,无定义域限制.
 * @param[out] out 输出 @f$\tanh(x)@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL.
 */
lmmc_status_t lmmc_tanh(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算反双曲正弦 @f$\mathrm{asinh}(x)@f$ .
 *
 * @param[in]  x   输入值,无定义域限制.
 * @param[out] out 输出 @f$\mathrm{asinh}(x)@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL.
 */
lmmc_status_t lmmc_asinh(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算反双曲余弦 @f$\mathrm{acosh}(x)@f$ ,要求 @f$x \ge 1@f$ .
 *
 * @param[in]  x   输入值,必须满足 @f$x \ge 1@f$ .
 * @param[out] out 输出 @f$\mathrm{acosh}(x)@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL;
 *         ::LMMC_STATUS_OUT_OF_RANGE 若 @f$x < 1@f$ .
 */
lmmc_status_t lmmc_acosh(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算反双曲正切 @f$\mathrm{atanh}(x)@f$ ,要求 @f$|x| < 1@f$ .
 *
 * @param[in]  x   输入值,必须满足 @f$|x| < 1@f$ .
 * @param[out] out 输出 @f$\mathrm{atanh}(x)@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL;
 *         ::LMMC_STATUS_OUT_OF_RANGE 若 @f$|x| \ge 1@f$ .
 */
lmmc_status_t lmmc_atanh(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算幂 @f$x^y@f$ .
 *
 * 当 @f$x = 0,y < 0@f$，或 @f$x < 0@f$ 且 @f$y@f$ 非整数时返回域错误。
 *
 * @param[in]  x   底数.
 * @param[in]  y   指数.
 * @param[out] out 输出 @f$x^y@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输入非有限或 out 为 NULL；
 *         ::LMMC_STATUS_OUT_OF_RANGE 若参数位于实数幂定义域外；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果不可表示。
 */
lmmc_status_t lmmc_pow(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out);

/**
 * @brief 计算向上取整 @f$\lceil x \rceil@f$ .
 *
 * @param[in]  x   输入值.
 * @param[out] out 输出 @f$\lceil x \rceil@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL.
 */
lmmc_status_t lmmc_ceil(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算向下取整 @f$\lfloor x \rfloor@f$ .
 *
 * @param[in]  x   输入值.
 * @param[out] out 输出 @f$\lfloor x \rfloor@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL.
 */
lmmc_status_t lmmc_floor(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算四舍五入 @f$\mathrm{round}(x)@f$ .
 *
 * @param[in]  x   输入值.
 * @param[out] out 输出四舍五入结果.
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL.
 */
lmmc_status_t lmmc_round(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算截断取整 @f$\mathrm{trunc}(x)@f$ (向零方向取整).
 *
 * @param[in]  x   输入值.
 * @param[out] out 输出截断取整结果.
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL.
 */
lmmc_status_t lmmc_trunc(lmmc_real_t x, lmmc_real_t* out);

#ifdef __cplusplus
}
#endif

#endif
