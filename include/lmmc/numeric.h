/**
 * @file numeric.h
 * @brief 数值常量,特殊值,近似比较与基本数学补充函数.
 *
 * 此模块包含 IEEE-754 特殊值生成 / 检测,扩展三角与对数函数,
 * 浮点拆分,近似相等判定,以及一个轻量的 radix-4 FFT 与
 * Lambert W 函数.
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

/* ===================== 特殊值生成 ===================== */

/** @brief 写入 IEEE-754 正无穷. */
lmmc_status_t lmmc_inf(lmmc_real_t* out_inf);
/** @brief 写入 IEEE-754 NaN. */
lmmc_status_t lmmc_nan(lmmc_real_t* out_nan);
/** @brief 写入机器精度(1 与下一个可表示数的差). */
lmmc_status_t lmmc_eps(lmmc_real_t* out_eps);

/* ===================== 浮点分类 ===================== */

/** @brief 检测是否为 NaN ,结果写入 @c *out_isnan (0/1). */
lmmc_status_t lmmc_isnan(lmmc_real_t x, int* out_isnan);
/** @brief 检测是否为 +/-infinity . */
lmmc_status_t lmmc_isinf(lmmc_real_t x, int* out_isinf);
/** @brief 检测是否为有限数(非 NaN 且非 +/-infinity). */
lmmc_status_t lmmc_isfinite(lmmc_real_t x, int* out_isfinite);
/** @brief 检测符号位(含 -0 ). */
lmmc_status_t lmmc_signbit(lmmc_real_t x, int* out_signbit);

/* ===================== 三角与几何 ===================== */

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
 * @brief 采用缩放计算求 @f$\sqrt{x^2+y^2}@f$，在各输入幅值范围内计算欧氏距离.
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或任一输入非有限;
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果上溢为非有限值.
 * 可表示的次正规结果正常返回；输出仅在成功时写入.
 */
lmmc_status_t lmmc_hypot(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res);

/* ===================== 指数与对数补充 ===================== */

/**
 * @brief 计算 @f$2^x@f$ .
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或 @p x 非有限;
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果上溢为非有限值.
 * 错误返回不修改 @p out_res.
 */
lmmc_status_t lmmc_exp2(lmmc_real_t x, lmmc_real_t* out_res);
/**
 * @brief 计算 @f$\log_2 x@f$ .
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若输出为空或 @p x 非有限;
 *         ::LMMC_STATUS_OUT_OF_RANGE 若 @p x 小于或等于零;
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若结果不可表示为有限值.
 * 错误返回不修改 @p out_res.
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
 * 错误返回不修改 @p out_res.
 */
lmmc_status_t lmmc_log1p(lmmc_real_t x, lmmc_real_t* out_res);

/* ===================== 浮点分解 ===================== */

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

/* ===================== 近似相等比较 ===================== */

/**
 * @brief 简单 epsilon 比较:@f$|a-b| \le \epsilon@f$ .
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
 * 适用于数值计算中需要同时处理接近零和远离零的数值比较场景.
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
 * @par 副作用
 * - 无.纯计算函数,不分配内存,不修改输入.
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

/* ===================== 离散傅里叶变换(radix-4) ===================== */

/**
 * @brief 计算不小于 @p n 的最近 4 的幂,用于 FFT 长度对齐.
 */
lmmc_status_t lmmc_fft_radix4_next_size(size_t n, size_t* out_nfft);

/**
 * @brief 严格 radix-4 就地 FFT.
 *
 * 对长度为 4 的幂的复数序列执行快速傅里叶变换.变换结果就地写回输入数组.
 * 逆变换时自动除以 N 进行归一化.
 *
 * @param[in,out] real    实部数组,长度 @p n ;变换后就地覆盖为频域实部.
 * @param[in,out] imag    虚部数组,长度 @p n ;变换后就地覆盖为频域虚部.
 * @param[in]     n       FFT 长度,必须为 4 的幂.
 * @param[in]     inverse 非 0 时执行逆变换并归一化(除以 n).
 *
 * @return ::LMMC_STATUS_OK 表示成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 表示 n 位于 4 的幂集合之外、指针为
 *         NULL、数组重叠或数组存储范围的地址计算溢出.
 *
 * @par 副作用
 * - 就地修改 @p real 和 @p imag 数组的全部 n 个元素.
 * - 不分配堆内存.
 * - @p real 和 @p imag 不可指向重叠的内存区域;重叠在写入前被拒绝.
 */
lmmc_status_t lmmc_fft_radix4(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse);

/** @brief 等价于 ::lmmc_fft_radix4(real, imag, n, 0) . */
lmmc_status_t lmmc_fft_radix4_forward(lmmc_real_t* real, lmmc_real_t* imag, size_t n);

/** @brief 等价于 ::lmmc_fft_radix4(real, imag, n, 1) . */
lmmc_status_t lmmc_fft_radix4_inverse(lmmc_real_t* real, lmmc_real_t* imag, size_t n);

/**
 * @brief 显式零填充辅助函数:将长度 @p n 的输入零填充到下一个 4 的幂.
 *
 * @param[in]  real_in   输入实部数组,长度 @p n .
 * @param[in]  imag_in   输入虚部数组,长度 @p n .
 * @param[in]  n         输入长度.
 * @param[out] real_out  输出实部数组,调用方需预分配至少由
 *                       ::lmmc_fft_radix4_next_size 得到的元素数.
 * @param[out] imag_out  输出虚部数组,容量要求同 @p real_out.
 * @param[out] out_nfft  写入实际填充后的长度(4 的幂).
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若参数无效、存储范围计算溢出，
 *         或任一输出与输入、另一输出、@p out_nfft 重叠.
 *
 * 所有参数和存储关系均在写入前验证；两个只读输入可以相互重叠。
 */
lmmc_status_t lmmc_fft_radix4_pad_into(
    const lmmc_real_t* real_in, const lmmc_real_t* imag_in, size_t n,
    lmmc_real_t* real_out, lmmc_real_t* imag_out, size_t* out_nfft);

/* ===================== 通用 N 点 FFT ===================== */

/**
 * @brief 计算任意长度 N 点 DFT(正变换或逆变换).
 *
 * 调度逻辑:
 * - N 为 4 的幂 -> radix-4 快速路径
 * - N 为 2 的幂 -> radix-2 Cooley-Tukey
 * - 其他 -> Bluestein chirp-z 算法(内部分配辅助缓冲区)
 *
 * @param[in,out] real    实部数组,长度 @p n ;变换后就地覆盖.
 * @param[in,out] imag    虚部数组,长度 @p n ;变换后就地覆盖.
 * @param[in]     n       FFT 长度,任意正整数.
 * @param[in]     inverse 非 0 时执行逆变换并归一化(除以 n).
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 n == 0、指针为 NULL、数组重叠
 *         或数组存储范围的地址计算溢出;
 *         ::LMMC_STATUS_ALLOCATION_FAILED 若 Bluestein 路径内存分配失败.
 *
 * @par 副作用
 * - 就地修改 @p real 和 @p imag 数组的全部 n 个元素.
 * - 当 n 非 2 的幂时,内部通过 Bluestein 算法分配临时缓冲区,函数返回前释放.
 * - @p real 和 @p imag 不可指向重叠的内存区域;重叠在写入前被拒绝.
 */
lmmc_status_t lmmc_fft(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse);

/** @brief 等价于 ::lmmc_fft(real, imag, n, 0) . */
lmmc_status_t lmmc_fft_forward(lmmc_real_t* real, lmmc_real_t* imag, size_t n);

/** @brief 等价于 ::lmmc_fft(real, imag, n, 1) . */
lmmc_status_t lmmc_fft_inverse(lmmc_real_t* real, lmmc_real_t* imag, size_t n);

/* ===================== 特殊函数(误差函数,伽马函数等) ===================== */

/**
 * @brief 计算误差函数 @f$\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt@f$ .
 *
 * @param[in]  x   输入值.
 * @param[out] out 输出 @f$\mathrm{erf}(x)@f$ .
 * @return LMMC_STATUS_OK 成功;LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL.
 */
lmmc_status_t lmmc_erf(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算互补误差函数 @f$\mathrm{erfc}(x) = 1 - \mathrm{erf}(x)@f$ .
 *
 * 对大 @f$|x|@f$ 使用直接近似以保持有效精度.
 *
 * @param[in]  x   输入值.
 * @param[out] out 输出 @f$\mathrm{erfc}(x)@f$ .
 * @return LMMC_STATUS_OK 成功;LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL.
 */
lmmc_status_t lmmc_erfc(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算对数伽马函数 @f$\ln\Gamma(x)@f$ ,要求 @f$x > 0@f$ .
 *
 * 使用 Lanczos 近似(g=7, n=9 系数),精度约 15 位有效数字.
 *
 * @param[in]  x   输入值,必须为正数.
 * @param[out] out 输出 @f$\ln\Gamma(x)@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 x <= 0 或 out 为 NULL.
 *
 * @par 副作用
 * - 无.纯计算函数,不分配内存,不修改输入.
 */
lmmc_status_t lmmc_lgamma(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算伽马函数 @f$\Gamma(x)@f$ ,要求 @f$x > 0@f$ .
 *
 * @param[in]  x   输入值,必须为正数.
 * @param[out] out 输出 @f$\Gamma(x)@f$ .
 * @return LMMC_STATUS_OK 成功;LMMC_STATUS_INVALID_ARGUMENT 若 x <= 0 或 out 为 NULL;
 *         LMMC_STATUS_NUMERICAL_FAILURE 若结果溢出.
 */
lmmc_status_t lmmc_tgamma(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算贝塔函数 @f$B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}@f$ ,要求 @f$a,b > 0@f$ .
 *
 * @param[in]  a   第一个参数,必须为正数.
 * @param[in]  b   第二个参数,必须为正数.
 * @param[out] out 输出 @f$B(a,b)@f$ .
 * @return LMMC_STATUS_OK 成功;LMMC_STATUS_INVALID_ARGUMENT 若 a <= 0 或 b <= 0 或 out 为 NULL.
 */
lmmc_status_t lmmc_beta(lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out);

/**
 * @brief 计算双伽马函数 @f$\psi(x) = \frac{d}{dx}\ln\Gamma(x)@f$ ,要求 @f$x > 0@f$ .
 *
 * 使用递推关系将 x 提升到大值区域后应用渐近展开.
 *
 * @param[in]  x   输入值,必须为正数.
 * @param[out] out 输出 @f$\psi(x)@f$ .
 * @return LMMC_STATUS_OK 成功;LMMC_STATUS_INVALID_ARGUMENT 若 x <= 0 或 out 为 NULL.
 */
lmmc_status_t lmmc_digamma(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 以下实标量包装器的公共状态约定。
 *
 * 所有输入必须有限，否则返回 LMMC_STATUS_INVALID_ARGUMENT。定义域或极点
 * 错误返回 LMMC_STATUS_OUT_OF_RANGE；有限输入产生不可表示结果时返回
 * LMMC_STATUS_NUMERICAL_FAILURE。非成功状态不写入输出参数。
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

/* ===================== Lambert W 函数 ===================== */

/**
 * @brief 计算 Lambert W 函数主分支 @f$W_0(z)@f$ .
 *
 * 满足 @f$W_0(z) e^{W_0(z)} = z@f$ ,使用 Halley 迭代求解.
 * 定义域为 @f$z \ge -1/e@f$ ,在 @f$z = -1/e@f$ 处 @f$W_0 = -1@f$ .
 *
 * @param[in]  z       输入值,必须满足 @f$z \ge -1/e@f$ .
 * @param[out] out_res 输出 @f$W_0(z)@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 z 超出定义域或 out_res 为 NULL.
 *
 * @par 副作用
 * - 无.纯计算函数,不分配内存,不修改输入.
 */
lmmc_status_t lmmc_lambertw(lmmc_real_t z, lmmc_real_t* out_res);

/**
 * @brief 计算 Lambert W 函数 @f$W_{-1}@f$ 分支,要求 @f$z \in [-1/e, 0)@f$ .
 *
 * @param[in]  z       输入值,必须满足 @f$-1/e \le z < 0@f$ .
 * @param[out] out_res 输出 @f$W_{-1}(z)@f$ .
 * @return LMMC_STATUS_OK 成功;LMMC_STATUS_INVALID_ARGUMENT 若 z 超出定义域.
 */
lmmc_status_t lmmc_lambertw_wm1(lmmc_real_t z, lmmc_real_t* out_res);

#ifdef __cplusplus
}
#endif

#endif
