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
#if (defined(_GNU_SOURCE) || defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
    {
        double s, c;
        sincos(x, &s, &c);
        LMMC_REAL_SET_D(out_sin, s);
        LMMC_REAL_SET_D(out_cos, c);
    }
#else
    LMMC_REAL_SET_D(out_sin, sin(x));
    LMMC_REAL_SET_D(out_cos, cos(x));
#endif
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
    lmmc_status_t st = LMMC_STATUS_OK;
    size_t nfft = 0;

    if (real == NULL || imag == NULL || (inverse != 0 && inverse != 1)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_fft_radix4_next_size(n, &nfft);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    if (nfft == n) {
        return lmmc_fft_radix4_core(real, imag, n, inverse);
    }

    {
        size_t nfft_bytes = 0;
        lmmc_real_t* tmp_real = NULL;
        lmmc_real_t* tmp_imag = NULL;

        if (lmmc_mul_overflow_size(nfft, sizeof(lmmc_real_t), &nfft_bytes)) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }

        tmp_real = (lmmc_real_t*)lmmc_alloc(nfft_bytes);
        tmp_imag = (lmmc_real_t*)lmmc_alloc(nfft_bytes);
        if (tmp_real == NULL || tmp_imag == NULL) {
            if (tmp_real != NULL) {
                lmmc_free(tmp_real);
            }
            if (tmp_imag != NULL) {
                lmmc_free(tmp_imag);
            }
            return LMMC_STATUS_ALLOCATION_FAILED;
        }

        memcpy(tmp_real, real, n * sizeof(lmmc_real_t));
        memcpy(tmp_imag, imag, n * sizeof(lmmc_real_t));
        memset(tmp_real + n, 0, (nfft - n) * sizeof(lmmc_real_t));
        memset(tmp_imag + n, 0, (nfft - n) * sizeof(lmmc_real_t));

        st = lmmc_fft_radix4_core(tmp_real, tmp_imag, nfft, inverse);
        if (st == LMMC_STATUS_OK) {
            memcpy(real, tmp_real, n * sizeof(lmmc_real_t));
            memcpy(imag, tmp_imag, n * sizeof(lmmc_real_t));
        }

        lmmc_free(tmp_real);
        lmmc_free(tmp_imag);
    }

    return st;
}

lmmc_status_t lmmc_fft_radix4_forward(lmmc_real_t* real, lmmc_real_t* imag, size_t n) {
    return lmmc_fft_radix4(real, imag, n, 0);
}

lmmc_status_t lmmc_fft_radix4_inverse(lmmc_real_t* real, lmmc_real_t* imag, size_t n) {
    return lmmc_fft_radix4(real, imag, n, 1);
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
