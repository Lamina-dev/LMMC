/** @file stdlib_math.h */
#ifndef LMMC_STDLIB_MATH_H
#define LMMC_STDLIB_MATH_H

#include "lmmc/complex.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 由实部与虚部构造复数。 */
lmmc_status_t lmmc_std_math_complex(lmmc_real_t real,
                                    lmmc_real_t imag,
                                    lmmc_complex_t* out);
/** @brief 提取复数的实部。 */
lmmc_status_t lmmc_std_math_real(const lmmc_complex_t* z, lmmc_real_t* out);
/** @brief 提取复数的虚部。 */
lmmc_status_t lmmc_std_math_imag(const lmmc_complex_t* z, lmmc_real_t* out);
/** @brief 为 std.math.conj 计算复共轭。 */
lmmc_status_t lmmc_std_math_conj(const lmmc_complex_t* z, lmmc_complex_t* out);
/** @brief 为 std.math.abs 计算复数模。 */
lmmc_status_t lmmc_std_math_complex_abs(const lmmc_complex_t* z,
                                        lmmc_real_t* out);

lmmc_status_t lmmc_std_math_sin(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_cos(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_tan(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_pow(lmmc_real_t x, lmmc_real_t y,
                                lmmc_real_t* out);
lmmc_status_t lmmc_std_math_asin(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_acos(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_atan(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_sqrt(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_exp(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_ln(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_log(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_log_base(lmmc_real_t x, lmmc_real_t base,
                                     lmmc_real_t* out);
lmmc_status_t lmmc_std_math_log10(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_abs(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_floor(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_ceil(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_round(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_math_clamp(lmmc_real_t x, lmmc_real_t lo,
                                  lmmc_real_t hi, lmmc_real_t* out);

#ifdef __cplusplus
}
#endif

#endif
