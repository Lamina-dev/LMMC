/**
 * @file numeric.c
 * @brief 数值常量,特殊值,近似比较,FFT 与 LambertW 实现.
 */
#include <math.h>
#include <string.h>
#include <float.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/numeric.h"

static void lmmc_abs_inplace(lmmc_real_t* res, const lmmc_real_t* x) {
    LMMC_REAL_ABS(res, x);
}

static void lmmc_max_inplace(lmmc_real_t* res, const lmmc_real_t* a, const lmmc_real_t* b) {
    if (LMMC_REAL_CMP(a, b) > 0) {
        LMMC_REAL_SET(res, a);
    } else {
        LMMC_REAL_SET(res, b);
    }
}

static int lmmc_mul_overflow_size(size_t a, size_t b, size_t* out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }
    if (a > ((size_t)-1) / b) {
        return 1;
    }
    *out = a * b;
    return 0;
}

static void lmmc_swapd(lmmc_real_t* a, lmmc_real_t* b) {
    lmmc_real_t tmp;
    LMMC_REAL_INIT(&tmp);
    LMMC_REAL_SET(&tmp, a);
    LMMC_REAL_SET(a, b);
    LMMC_REAL_SET(b, &tmp);
    LMMC_REAL_CLEAR(&tmp);
}

static int lmmc_is_power_of_four(size_t n, unsigned* out_digits) {
    size_t t = n;
    unsigned digits = 0;

    if (n == 0) {
        return 0;
    }
    while ((t & (size_t)3) == 0) {
        t >>= 2;
        ++digits;
    }
    if (t != (size_t)1) {
        return 0;
    }

    if (out_digits != NULL) {
        *out_digits = digits;
    }
    return 1;
}

static size_t lmmc_reverse_base4(size_t value, unsigned digits) {
    size_t reversed = 0;
    unsigned i = 0;

    for (i = 0; i < digits; ++i) {
        reversed = (reversed << 2) | (value & (size_t)3);
        value >>= 2;
    }
    return reversed;
}

static void lmmc_complex_mul(const lmmc_real_t* ar, const lmmc_real_t* ai, const lmmc_real_t* br, const lmmc_real_t* bi, lmmc_real_t* out_r, lmmc_real_t* out_i) {
    lmmc_real_t tmp1, tmp2;
    LMMC_REAL_INIT(&tmp1);
    LMMC_REAL_INIT(&tmp2);

    LMMC_REAL_MUL(&tmp1, ar, br);
    LMMC_REAL_MUL(&tmp2, ai, bi);
    LMMC_REAL_SUB(out_r, &tmp1, &tmp2);

    LMMC_REAL_MUL(&tmp1, ar, bi);
    LMMC_REAL_MUL(&tmp2, ai, br);
    LMMC_REAL_ADD(out_i, &tmp1, &tmp2);

    LMMC_REAL_CLEAR(&tmp1);
    LMMC_REAL_CLEAR(&tmp2);
}

lmmc_status_t lmmc_inf(lmmc_real_t* out_inf) {
    if (out_inf == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
#ifdef INFINITY
    LMMC_REAL_SET_D(out_inf, INFINITY);
#else
    LMMC_REAL_SET_D(out_inf, HUGE_VAL);
#endif
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_nan(lmmc_real_t* out_nan) {
    if (out_nan == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
#ifdef NAN
    LMMC_REAL_SET_D(out_nan, NAN);
#else
    lmmc_real_t zero;
    LMMC_REAL_INIT(&zero);
    LMMC_REAL_SET_D(&zero, 0.0);
    LMMC_REAL_DIV(out_nan, &zero, &zero);
    LMMC_REAL_CLEAR(&zero);
#endif
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_eps(lmmc_real_t* out_eps) {
    if (out_eps == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    LMMC_REAL_SET_D(out_eps, LMMC_REAL_EPSILON);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_isnan(lmmc_real_t x, int* out_isnan) {
    if (out_isnan == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_isnan = isnan(x) ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_isinf(lmmc_real_t x, int* out_isinf) {
    if (out_isinf == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_isinf = isinf(x) ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_isfinite(lmmc_real_t x, int* out_isfinite) {
    if (out_isfinite == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_isfinite = isfinite(x) ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_signbit(lmmc_real_t x, int* out_signbit) {
    if (out_signbit == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_signbit = signbit(x) ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_atan2(lmmc_real_t y, lmmc_real_t x, lmmc_real_t* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    LMMC_REAL_SET_D(out_res, atan2(y, x));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sincos(lmmc_real_t x, lmmc_real_t* out_sin, lmmc_real_t* out_cos) {
    if (out_sin == NULL || out_cos == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    LMMC_REAL_SET_D(out_sin, sin(x));
    LMMC_REAL_SET_D(out_cos, cos(x));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_hypot(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    LMMC_REAL_SET_D(out_res, hypot(x, y));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_exp2(lmmc_real_t x, lmmc_real_t* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    LMMC_REAL_SET_D(out_res, exp2(x));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_log2(lmmc_real_t x, lmmc_real_t* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    LMMC_REAL_SET_D(out_res, log2(x));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_expm1(lmmc_real_t x, lmmc_real_t* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    LMMC_REAL_SET_D(out_res, expm1(x));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_log1p(lmmc_real_t x, lmmc_real_t* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    LMMC_REAL_SET_D(out_res, log1p(x));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_split_int_frac(lmmc_real_t x, lmmc_real_t* out_iptr, lmmc_real_t* out_frac) {
    if (out_iptr == NULL || out_frac == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    {
        double ip;
        double fr = modf(x, &ip);
        LMMC_REAL_SET_D(out_iptr, ip);
        LMMC_REAL_SET_D(out_frac, fr);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_fmod(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    LMMC_REAL_SET_D(out_res, fmod(x, y));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ldexp(lmmc_real_t x, int exp, lmmc_real_t* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    LMMC_REAL_SET_D(out_res, ldexp(x, exp));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_nextafter(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    LMMC_REAL_SET_D(out_res, nextafter(x, y));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_approx_eq(lmmc_real_t a, lmmc_real_t b, lmmc_real_t epsilon, int* out_equal) {
    lmmc_real_t diff, zero;
    if (out_equal == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (isnan(a) || isnan(b)) {
        *out_equal = 0;
        return LMMC_STATUS_OK;
    }
    if (isinf(a) || isinf(b)) {
        *out_equal = (LMMC_REAL_CMP(&a, &b) == 0) ? 1 : 0;
        return LMMC_STATUS_OK;
    }

    LMMC_REAL_INIT(&diff);
    LMMC_REAL_INIT(&zero);

    LMMC_REAL_SET_D(&zero, 0.0);
    LMMC_REAL_SUB(&diff, &a, &b);

    if (LMMC_REAL_CMP(&diff, &zero) < 0) {
        lmmc_real_t tmp;
        LMMC_REAL_INIT(&tmp);
        LMMC_REAL_SET(&tmp, &diff);
        LMMC_REAL_SUB(&diff, &zero, &tmp);
        LMMC_REAL_CLEAR(&tmp);
    }
    *out_equal = (LMMC_REAL_CMP(&diff, &epsilon) <= 0) ? 1 : 0;

    LMMC_REAL_CLEAR(&diff);
    LMMC_REAL_CLEAR(&zero);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_fft_radix4_next_size(size_t n, size_t* out_nfft) {
    size_t nfft = 1;

    if (out_nfft == NULL || n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    while (nfft < n) {
        if (nfft > ((size_t)-1) / (size_t)4) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        nfft *= (size_t)4;
    }

    *out_nfft = nfft;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_fft_radix4_core(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse) {
    unsigned digits = 0;
    size_t i = 0;
    size_t len = 0;

    if (!lmmc_is_power_of_four(n, &digits)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n <= 1) {
        return LMMC_STATUS_OK;
    }

    for (i = 0; i < n; ++i) {
        size_t r = lmmc_reverse_base4(i, digits);
        if (r > i) {
            lmmc_swapd(&real[i], &real[r]);
            lmmc_swapd(&imag[i], &imag[r]);
        }
    }

    len = 4;
    while (len <= n) {
        size_t group = len;
        size_t quarter = group / 4;

        lmmc_real_t angle_step, tmp1, tmp2, tmp3, tmp4;
        LMMC_REAL_INIT(&angle_step);
        LMMC_REAL_INIT(&tmp1);
        LMMC_REAL_INIT(&tmp2);
        LMMC_REAL_INIT(&tmp3);
        LMMC_REAL_INIT(&tmp4);

        LMMC_REAL_SET_D(&tmp1, (inverse ? 2.0 : -2.0) * LMMC_PI);
        LMMC_REAL_SET_D(&tmp2, (double)group);
        LMMC_REAL_DIV(&angle_step, &tmp1, &tmp2);

        for (i = 0; i < n; i += group) {
            size_t j = 0;
            for (j = 0; j < quarter; ++j) {
                size_t i0 = i + j;
                size_t i1 = i0 + quarter;
                size_t i2 = i1 + quarter;
                size_t i3 = i2 + quarter;

                lmmc_real_t a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
                lmmc_real_t ang1, ang2, ang3;
                lmmc_real_t c1, s1, c2, s2, c3, s3;
                double d_ang1, d_ang2, d_ang3;

                LMMC_REAL_INIT(&a0r); LMMC_REAL_INIT(&a0i);
                LMMC_REAL_INIT(&a1r); LMMC_REAL_INIT(&a1i);
                LMMC_REAL_INIT(&a2r); LMMC_REAL_INIT(&a2i);
                LMMC_REAL_INIT(&a3r); LMMC_REAL_INIT(&a3i);
                LMMC_REAL_INIT(&ang1); LMMC_REAL_INIT(&ang2); LMMC_REAL_INIT(&ang3);
                LMMC_REAL_INIT(&c1); LMMC_REAL_INIT(&s1);
                LMMC_REAL_INIT(&c2); LMMC_REAL_INIT(&s2);
                LMMC_REAL_INIT(&c3); LMMC_REAL_INIT(&s3);

                LMMC_REAL_SET(&a0r, &real[i0]);
                LMMC_REAL_SET(&a0i, &imag[i0]);
                LMMC_REAL_SET_D(&a1r, 0.0); LMMC_REAL_SET_D(&a1i, 0.0);
                LMMC_REAL_SET_D(&a2r, 0.0); LMMC_REAL_SET_D(&a2i, 0.0);
                LMMC_REAL_SET_D(&a3r, 0.0); LMMC_REAL_SET_D(&a3i, 0.0);

                LMMC_REAL_SET_D(&tmp1, (double)j);
                LMMC_REAL_MUL(&ang1, &angle_step, &tmp1);
                LMMC_REAL_SET_D(&tmp1, 2.0);
                LMMC_REAL_MUL(&ang2, &ang1, &tmp1);
                LMMC_REAL_SET_D(&tmp1, 3.0);
                LMMC_REAL_MUL(&ang3, &ang1, &tmp1);

                d_ang1 = ((inverse ? 2.0 : -2.0) * LMMC_PI / (double)group) * (double)j;
                d_ang2 = d_ang1 * 2.0;
                d_ang3 = d_ang1 * 3.0;

                LMMC_REAL_SET_D(&c1, cos(d_ang1)); LMMC_REAL_SET_D(&s1, sin(d_ang1));
                LMMC_REAL_SET_D(&c2, cos(d_ang2)); LMMC_REAL_SET_D(&s2, sin(d_ang2));
                LMMC_REAL_SET_D(&c3, cos(d_ang3)); LMMC_REAL_SET_D(&s3, sin(d_ang3));

                lmmc_complex_mul(&real[i1], &imag[i1], &c1, &s1, &a1r, &a1i);
                lmmc_complex_mul(&real[i2], &imag[i2], &c2, &s2, &a2r, &a2i);
                lmmc_complex_mul(&real[i3], &imag[i3], &c3, &s3, &a3r, &a3i);

                LMMC_REAL_ADD(&tmp1, &a0r, &a1r); LMMC_REAL_ADD(&tmp1, &tmp1, &a2r); LMMC_REAL_ADD(&real[i0], &tmp1, &a3r);
                LMMC_REAL_ADD(&tmp1, &a0i, &a1i); LMMC_REAL_ADD(&tmp1, &tmp1, &a2i); LMMC_REAL_ADD(&imag[i0], &tmp1, &a3i);

                LMMC_REAL_SUB(&tmp1, &a0r, &a1r); LMMC_REAL_ADD(&tmp1, &tmp1, &a2r); LMMC_REAL_SUB(&real[i2], &tmp1, &a3r);
                LMMC_REAL_SUB(&tmp1, &a0i, &a1i); LMMC_REAL_ADD(&tmp1, &tmp1, &a2i); LMMC_REAL_SUB(&imag[i2], &tmp1, &a3i);

                if (!inverse) {
                    LMMC_REAL_ADD(&tmp1, &a0r, &a1i); LMMC_REAL_SUB(&tmp1, &tmp1, &a2r); LMMC_REAL_SUB(&real[i1], &tmp1, &a3i);
                    LMMC_REAL_SUB(&tmp1, &a0i, &a1r); LMMC_REAL_SUB(&tmp1, &tmp1, &a2i); LMMC_REAL_ADD(&imag[i1], &tmp1, &a3r);
                    LMMC_REAL_SUB(&tmp1, &a0r, &a1i); LMMC_REAL_SUB(&tmp1, &tmp1, &a2r); LMMC_REAL_ADD(&real[i3], &tmp1, &a3i);
                    LMMC_REAL_ADD(&tmp1, &a0i, &a1r); LMMC_REAL_SUB(&tmp1, &tmp1, &a2i); LMMC_REAL_SUB(&imag[i3], &tmp1, &a3r);
                } else {
                    LMMC_REAL_SUB(&tmp1, &a0r, &a1i); LMMC_REAL_SUB(&tmp1, &tmp1, &a2r); LMMC_REAL_ADD(&real[i1], &tmp1, &a3i);
                    LMMC_REAL_ADD(&tmp1, &a0i, &a1r); LMMC_REAL_SUB(&tmp1, &tmp1, &a2i); LMMC_REAL_SUB(&imag[i1], &tmp1, &a3r);
                    LMMC_REAL_ADD(&tmp1, &a0r, &a1i); LMMC_REAL_SUB(&tmp1, &tmp1, &a2r); LMMC_REAL_SUB(&real[i3], &tmp1, &a3i);
                    LMMC_REAL_SUB(&tmp1, &a0i, &a1r); LMMC_REAL_SUB(&tmp1, &tmp1, &a2i); LMMC_REAL_ADD(&imag[i3], &tmp1, &a3r);
                }

                LMMC_REAL_CLEAR(&a0r); LMMC_REAL_CLEAR(&a0i);
                LMMC_REAL_CLEAR(&a1r); LMMC_REAL_CLEAR(&a1i);
                LMMC_REAL_CLEAR(&a2r); LMMC_REAL_CLEAR(&a2i);
                LMMC_REAL_CLEAR(&a3r); LMMC_REAL_CLEAR(&a3i);
                LMMC_REAL_CLEAR(&ang1); LMMC_REAL_CLEAR(&ang2); LMMC_REAL_CLEAR(&ang3);
                LMMC_REAL_CLEAR(&c1); LMMC_REAL_CLEAR(&s1);
                LMMC_REAL_CLEAR(&c2); LMMC_REAL_CLEAR(&s2);
                LMMC_REAL_CLEAR(&c3); LMMC_REAL_CLEAR(&s3);
            }
        }
        LMMC_REAL_CLEAR(&angle_step);
        LMMC_REAL_CLEAR(&tmp1);
        LMMC_REAL_CLEAR(&tmp2);
        LMMC_REAL_CLEAR(&tmp3);
        LMMC_REAL_CLEAR(&tmp4);

        if (len == n) {
            break;
        }
        len *= 4;
    }

    if (inverse) {
        lmmc_real_t scale, tmp1, tmp2;
        LMMC_REAL_INIT(&scale);
        LMMC_REAL_INIT(&tmp1);
        LMMC_REAL_INIT(&tmp2);

        LMMC_REAL_SET_D(&tmp1, 1.0);
        LMMC_REAL_SET_D(&tmp2, (double)n);
        LMMC_REAL_DIV(&scale, &tmp1, &tmp2);
        for (i = 0; i < n; ++i) {
            LMMC_REAL_MUL(&real[i], &real[i], &scale);
            LMMC_REAL_MUL(&imag[i], &imag[i], &scale);
        }

        LMMC_REAL_CLEAR(&scale);
        LMMC_REAL_CLEAR(&tmp1);
        LMMC_REAL_CLEAR(&tmp2);
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_fft_radix4(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse) {
    unsigned digits = 0;

    if (real == NULL || imag == NULL || (inverse != 0 && inverse != 1)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Strict radix-4: n must be an exact power of 4 */
    if (!lmmc_is_power_of_four(n, &digits)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    return lmmc_fft_radix4_core(real, imag, n, inverse);
}

lmmc_status_t lmmc_fft_radix4_forward(lmmc_real_t* real, lmmc_real_t* imag, size_t n) {
    return lmmc_fft_radix4(real, imag, n, 0);
}

lmmc_status_t lmmc_fft_radix4_inverse(lmmc_real_t* real, lmmc_real_t* imag, size_t n) {
    return lmmc_fft_radix4(real, imag, n, 1);
}

/**
 * @brief 误差函数 erf(x) - 使用 C 标准库 erf() 并包装为 LMMC 接口.
 *
 * C99 标准库提供了高精度的 erf 实现(通常 < 1 ULP 误差),
 * 直接使用以确保最佳精度.
 */
lmmc_status_t lmmc_erf(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = erf(x);
    return LMMC_STATUS_OK;
}

/**
 * @brief 互补误差函数 erfc(x) = 1 - erf(x) - 使用 C 标准库 erfc().
 *
 * 对大 |x| 直接计算以保持尾部概率的有效精度.
 */
lmmc_status_t lmmc_erfc(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = erfc(x);
    return LMMC_STATUS_OK;
}

/**
 * @brief 对数伽马函数 lgamma(x) - Lanczos 近似 (g=7, n=9).
 *
 * 使用 Lanczos 近似:
 * ln(Gamma(x)) = (x - 0.5) * ln(x + g - 0.5) - (x + g - 0.5) + 0.5*ln(2*pi) + ln(Ag(x))
 * 其中 Ag(x) 是 Lanczos 级数和.
 */
lmmc_status_t lmmc_lgamma(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Lanczos 系数 (g=7, n=9) - 来自 Numerical Recipes / Cephes */
    static const double coeff[9] = {
         0.99999999999980993,
       676.5203681218851,
      -1259.1392167224028,
        771.32342877765313,
       -176.61502916214059,
         12.507343278686905,
         -0.13857109526572012,
          9.9843695780195716e-6,
          1.5056327351493116e-7
    };

    double g = 7.0;
    double tmp, ser;
    int i;

    if (x < 0.5) {
        /* 使用反射公式: Gamma(x) * Gamma(1-x) = pi / sin(pi*x) */
        /* lgamma(x) = ln(pi) - ln(sin(pi*x)) - lgamma(1-x) */
        double sinpx = sin(LMMC_PI * x);
        if (fabs(sinpx) < 1e-300) return LMMC_STATUS_NUMERICAL_FAILURE;
        lmmc_real_t lg1mx;
        lmmc_status_t st = lmmc_lgamma(1.0 - x, &lg1mx);
        if (st != LMMC_STATUS_OK) return st;
        *out = log(LMMC_PI) - log(fabs(sinpx)) - lg1mx;
        return LMMC_STATUS_OK;
    }

    x -= 1.0;
    tmp = x + g + 0.5;
    ser = coeff[0];
    for (i = 1; i < 9; i++) {
        ser += coeff[i] / (x + (double)i);
    }

    *out = 0.5 * log(2.0 * LMMC_PI) + (x + 0.5) * log(tmp) - tmp + log(ser);
    return LMMC_STATUS_OK;
}

/**
 * @brief 伽马函数 tgamma(x) = exp(lgamma(x)),要求 x > 0.
 */
lmmc_status_t lmmc_tgamma(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    lmmc_real_t lg;
    lmmc_status_t st = lmmc_lgamma(x, &lg);
    if (st != LMMC_STATUS_OK) return st;

    /* 检查溢出 */
    if (lg > 709.0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    *out = exp(lg);
    return LMMC_STATUS_OK;
}

/**
 * @brief 贝塔函数 B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b).
 *
 * 使用对数伽马控制中间结果幅值:B(a,b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b)).
 */
lmmc_status_t lmmc_beta(lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a <= 0.0 || b <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    lmmc_real_t lga, lgb, lgab;
    lmmc_status_t st;

    st = lmmc_lgamma(a, &lga);
    if (st != LMMC_STATUS_OK) return st;

    st = lmmc_lgamma(b, &lgb);
    if (st != LMMC_STATUS_OK) return st;

    st = lmmc_lgamma(a + b, &lgab);
    if (st != LMMC_STATUS_OK) return st;

    double result = lga + lgb - lgab;
    if (result > 709.0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    *out = exp(result);
    return LMMC_STATUS_OK;
}

/**
 * @brief 双伽马函数 psi(x) = d/dx ln(Gamma(x)).
 *
 * 算法:
 * 1. 对 x < 6 使用递推关系 psi(x+1) = psi(x) + 1/x 将 x 提升到 >= 6.
 * 2. 对 x >= 6 使用渐近展开:
 *    psi(x) ~ ln(x) - 1/(2x) - sum_{k=1}^{N} B_{2k}/(2k * x^{2k})
 *    其中 B_{2k} 是 Bernoulli 数.
 */
lmmc_status_t lmmc_digamma(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    double result = 0.0;

    /* 递推:将 x 提升到 >= 7 以确保渐近展开精度 */
    while (x < 7.0) {
        result -= 1.0 / x;
        x += 1.0;
    }

    /* 渐近展开 psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ... */
    /* Bernoulli 数: B2=1/6, B4=-1/30, B6=1/42, B8=-1/30, B10=5/66, B12=-691/2730 */
    {
        double ix = 1.0 / x;
        double ix2 = ix * ix;

        /* 系数 = B_{2k} / (2k) */
        static const double bernoulli_coeff[] = {
            1.0 / 12.0,       /* B2/(2*1) = (1/6)/2 */
           -1.0 / 120.0,      /* B4/(2*2) = (-1/30)/4 */
            1.0 / 252.0,      /* B6/(2*3) = (1/42)/6 */
           -1.0 / 240.0,      /* B8/(2*4) = (-1/30)/8 */
            5.0 / 660.0,      /* B10/(2*5) = (5/66)/10 */
           -691.0 / 32760.0,  /* B12/(2*6) = (-691/2730)/12 */
            1.0 / 12.0        /* B14/(2*7) = (7/6)/14 = 1/12 */
        };

        double sum = 0.0;
        double ix2k = ix2; /* ix^(2k) starting at ix^2 */
        int k;
        for (k = 0; k < 7; k++) {
            sum -= bernoulli_coeff[k] * ix2k;
            ix2k *= ix2;
        }

        result += log(x) - 0.5 * ix + sum;
    }

    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lambertw(lmmc_real_t z, lmmc_real_t* out_res) {
    const double tol = LMMC_DEFAULT_REL_TOL;
    const int max_iter = 100;
    double w, expw, w_expw, diff, abs_diff, denom, f_prime, f_double_prime;
    double threshold;
    int i;

    if (!out_res) return LMMC_STATUS_INVALID_ARGUMENT;
    if (z < -LMMC_INV_E) return LMMC_STATUS_INVALID_ARGUMENT;
    if (z == 0.0) { *out_res = 0.0; return LMMC_STATUS_OK; }

    /* Initial approximation */
    if (z > 2.0) {
        double lnz = log(z);
        w = lnz - log(lnz);
    } else if (z > -LMMC_INV_E && z < -0.3) {
        /* Near the branch point: use a series expansion around -1/e */
        double p = sqrt(2.0 * (exp(1.0) * z + 1.0));
        w = -1.0 + p - p * p / 3.0;
    } else if (z == -LMMC_INV_E) {
        *out_res = -1.0;
        return LMMC_STATUS_OK;
    } else {
        w = z;
    }

    /* Halley iteration */
    for (i = 0; i < max_iter; ++i) {
        expw = exp(w);
        w_expw = w * expw;
        diff = w_expw - z;
        abs_diff = fabs(diff);
        threshold = tol * (1.0 + fabs(z));
        if (abs_diff <= threshold) {
            *out_res = w;
            return LMMC_STATUS_OK;
        }
        f_prime = expw * (w + 1.0);
        f_double_prime = expw * (w + 2.0);
        /* Halley step: w -= f / (f' - f * f'' / (2 * f')) */
        denom = f_prime - diff * f_double_prime / (2.0 * f_prime);
        if (fabs(denom) < 1e-300) {
            /* Fallback to Newton step if Halley denominator is too small */
            w -= diff / f_prime;
        } else {
            w -= diff / denom;
        }
    }
    *out_res = w;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lambertw_wm1(lmmc_real_t z, lmmc_real_t* out_res) {
    const double tol = LMMC_DEFAULT_REL_TOL;
    const int max_iter = 100;
    double w, expw, w_expw, diff, abs_diff, denom, f_prime, f_double_prime;
    double threshold;
    int i;

    if (!out_res) return LMMC_STATUS_INVALID_ARGUMENT;
    /* Domain: z in [-1/e, 0) */
    if (z < -LMMC_INV_E || z >= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Special case: z == -1/e => W_-_1(-1/e) = -1 */
    if (fabs(z + LMMC_INV_E) < 1e-300) {
        *out_res = -1.0;
        return LMMC_STATUS_OK;
    }

    /* Initial guess from asymptotic: log(-z) - log(-log(-z)) */
    {
        double lnmz = log(-z);
        double lnlnmz = log(-lnmz);
        w = lnmz - lnlnmz;
    }

    /* Halley iteration */
    for (i = 0; i < max_iter; ++i) {
        expw = exp(w);
        w_expw = w * expw;
        diff = w_expw - z;
        abs_diff = fabs(diff);
        threshold = tol * (1.0 + fabs(z));
        if (abs_diff <= threshold) {
            *out_res = w;
            return LMMC_STATUS_OK;
        }
        f_prime = expw * (w + 1.0);
        f_double_prime = expw * (w + 2.0);
        /* Halley step */
        denom = f_prime - diff * f_double_prime / (2.0 * f_prime);
        if (fabs(denom) < 1e-300) {
            w -= diff / f_prime;
        } else {
            w -= diff / denom;
        }
    }
    *out_res = w;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_double_nearly_equal_tol(
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t abs_tol,
    lmmc_real_t rel_tol,
    int* out_equal
) {
    lmmc_real_t diff, scale, threshold, zero, tmp1, tmp2;

    if (out_equal == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_INIT(&diff);
    LMMC_REAL_INIT(&scale);
    LMMC_REAL_INIT(&threshold);
    LMMC_REAL_INIT(&zero);
    LMMC_REAL_INIT(&tmp1);
    LMMC_REAL_INIT(&tmp2);

    LMMC_REAL_SET_D(&zero, 0.0);

    if (LMMC_REAL_CMP(&abs_tol, &zero) < 0 || LMMC_REAL_CMP(&rel_tol, &zero) < 0 || !LMMC_REAL_IS_FINITE(&abs_tol) || !LMMC_REAL_IS_FINITE(&rel_tol)) {
        LMMC_REAL_CLEAR(&diff); LMMC_REAL_CLEAR(&scale); LMMC_REAL_CLEAR(&threshold);
        LMMC_REAL_CLEAR(&zero); LMMC_REAL_CLEAR(&tmp1); LMMC_REAL_CLEAR(&tmp2);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (isnan(a) || isnan(b)) {
        LMMC_REAL_CLEAR(&diff); LMMC_REAL_CLEAR(&scale); LMMC_REAL_CLEAR(&threshold);
        LMMC_REAL_CLEAR(&zero); LMMC_REAL_CLEAR(&tmp1); LMMC_REAL_CLEAR(&tmp2);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (isinf(a) || isinf(b)) {
        *out_equal = (LMMC_REAL_CMP(&a, &b) == 0) ? 1 : 0;
        LMMC_REAL_CLEAR(&diff); LMMC_REAL_CLEAR(&scale); LMMC_REAL_CLEAR(&threshold);
        LMMC_REAL_CLEAR(&zero); LMMC_REAL_CLEAR(&tmp1); LMMC_REAL_CLEAR(&tmp2);
        return LMMC_STATUS_OK;
    }

    LMMC_REAL_SUB(&tmp1, &a, &b);
    lmmc_abs_inplace(&diff, &tmp1);

    lmmc_abs_inplace(&tmp1, &a);
    lmmc_abs_inplace(&tmp2, &b);
    lmmc_max_inplace(&scale, &tmp1, &tmp2);

    LMMC_REAL_MUL(&tmp1, &rel_tol, &scale);
    LMMC_REAL_ADD(&threshold, &abs_tol, &tmp1);

    if (isnan(diff) || isnan(threshold)) {
        LMMC_REAL_CLEAR(&diff); LMMC_REAL_CLEAR(&scale); LMMC_REAL_CLEAR(&threshold);
        LMMC_REAL_CLEAR(&zero); LMMC_REAL_CLEAR(&tmp1); LMMC_REAL_CLEAR(&tmp2);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (isinf(threshold)) {
        *out_equal = 1;
        LMMC_REAL_CLEAR(&diff); LMMC_REAL_CLEAR(&scale); LMMC_REAL_CLEAR(&threshold);
        LMMC_REAL_CLEAR(&zero); LMMC_REAL_CLEAR(&tmp1); LMMC_REAL_CLEAR(&tmp2);
        return LMMC_STATUS_OK;
    }

    *out_equal = (LMMC_REAL_CMP(&diff, &threshold) <= 0) ? 1 : 0;

    LMMC_REAL_CLEAR(&diff);
    LMMC_REAL_CLEAR(&scale);
    LMMC_REAL_CLEAR(&threshold);
    LMMC_REAL_CLEAR(&zero);
    LMMC_REAL_CLEAR(&tmp1);
    LMMC_REAL_CLEAR(&tmp2);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_double_nearly_equal(lmmc_real_t a, lmmc_real_t b, int* out_equal) {
    lmmc_real_t abs_tol, rel_tol;
    lmmc_status_t st;
    LMMC_REAL_INIT(&abs_tol);
    LMMC_REAL_INIT(&rel_tol);
    LMMC_REAL_SET_D(&abs_tol, LMMC_DEFAULT_ABS_TOL);
    LMMC_REAL_SET_D(&rel_tol, LMMC_DEFAULT_REL_TOL);
    st = lmmc_double_nearly_equal_tol(a, b, abs_tol, rel_tol, out_equal);
    LMMC_REAL_CLEAR(&abs_tol);
    LMMC_REAL_CLEAR(&rel_tol);
    return st;
}

lmmc_status_t lmmc_asin(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < -1.0 || x > 1.0) return LMMC_STATUS_OUT_OF_RANGE;
    *out = asin(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_acos(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < -1.0 || x > 1.0) return LMMC_STATUS_OUT_OF_RANGE;
    *out = acos(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_atan(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = atan(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sinh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = sinh(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_cosh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = cosh(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tanh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = tanh(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_asinh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = asinh(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_acosh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 1.0) return LMMC_STATUS_OUT_OF_RANGE;
    *out = acosh(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_atanh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= -1.0 || x >= 1.0) return LMMC_STATUS_OUT_OF_RANGE;
    *out = atanh(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_pow(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0 && floor(y) != y) return LMMC_STATUS_OUT_OF_RANGE;
    *out = pow(x, y);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ceil(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = ceil(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_floor(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = floor(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_round(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = round(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_trunc(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = trunc(x);
    return LMMC_STATUS_OK;
}
