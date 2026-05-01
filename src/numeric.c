#include <math.h>
#include <string.h>
#include <float.h>
#include "memory_bridge.h"
#include "lmmc/numeric.h"

static double lmmc_absd(double x) {
    return x < 0.0 ? -x : x;
}

static double lmmc_maxd(double a, double b) {
    return a > b ? a : b;
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

static void lmmc_swapd(double* a, double* b) {
    double tmp = *a;
    *a = *b;
    *b = tmp;
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

static void lmmc_complex_mul(double ar, double ai, double br, double bi, double* out_r, double* out_i) {
    *out_r = ar * br - ai * bi;
    *out_i = ar * bi + ai * br;
}

lmmc_status_t lmmc_inf(double* out_inf) {
    if (out_inf == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
#ifdef INFINITY
    *out_inf = INFINITY;
#else
    *out_inf = HUGE_VAL;
#endif
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_nan(double* out_nan) {
    if (out_nan == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
#ifdef NAN
    *out_nan = NAN;
#else
    *out_nan = (0.0 / 0.0);
#endif
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_eps(double* out_eps) {
    if (out_eps == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_eps = DBL_EPSILON;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_isnan(double x, int* out_isnan) {
    if (out_isnan == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_isnan = isnan(x) ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_isinf(double x, int* out_isinf) {
    if (out_isinf == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_isinf = isinf(x) ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_isfinite(double x, int* out_isfinite) {
    if (out_isfinite == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_isfinite = isfinite(x) ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_signbit(double x, int* out_signbit) {
    if (out_signbit == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_signbit = signbit(x) ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_atan2(double y, double x, double* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_res = atan2(y, x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sincos(double x, double* out_sin, double* out_cos) {
    if (out_sin == NULL || out_cos == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
#if (defined(_GNU_SOURCE) || defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
    sincos(x, out_sin, out_cos);
#else
    *out_sin = sin(x);
    *out_cos = cos(x);
#endif
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_hypot(double x, double y, double* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_res = hypot(x, y);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_exp2(double x, double* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_res = exp2(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_log2(double x, double* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_res = log2(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_expm1(double x, double* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_res = expm1(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_log1p(double x, double* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_res = log1p(x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_modf(double x, double* out_iptr, double* out_frac) {
    if (out_iptr == NULL || out_frac == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_frac = modf(x, out_iptr);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_fmod(double x, double y, double* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_res = fmod(x, y);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ldexp(double x, int exp, double* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_res = ldexp(x, exp);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_nextafter(double x, double y, double* out_res) {
    if (out_res == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    *out_res = nextafter(x, y);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_approx_eq(double a, double b, double epsilon, int* out_equal) {
    if (out_equal == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (isnan(a) || isnan(b)) {
        *out_equal = 0;
        return LMMC_STATUS_OK;
    }
    if (isinf(a) || isinf(b)) {
        *out_equal = (a == b) ? 1 : 0;
        return LMMC_STATUS_OK;
    }
    double diff = a - b;
    if (diff < 0) diff = -diff;
    *out_equal = (diff <= epsilon) ? 1 : 0;
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

static lmmc_status_t lmmc_fft_radix4_core(double* real, double* imag, size_t n, int inverse) {
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
        double angle_step = (inverse ? 2.0 : -2.0) * LMMC_PI / (double)group;

        for (i = 0; i < n; i += group) {
            size_t j = 0;
            for (j = 0; j < quarter; ++j) {
                size_t i0 = i + j;
                size_t i1 = i0 + quarter;
                size_t i2 = i1 + quarter;
                size_t i3 = i2 + quarter;

                double a0r = real[i0];
                double a0i = imag[i0];
                double a1r = 0.0;
                double a1i = 0.0;
                double a2r = 0.0;
                double a2i = 0.0;
                double a3r = 0.0;
                double a3i = 0.0;
                double ang1 = angle_step * (double)j;
                double ang2 = ang1 * 2.0;
                double ang3 = ang1 * 3.0;

                lmmc_complex_mul(real[i1], imag[i1], cos(ang1), sin(ang1), &a1r, &a1i);
                lmmc_complex_mul(real[i2], imag[i2], cos(ang2), sin(ang2), &a2r, &a2i);
                lmmc_complex_mul(real[i3], imag[i3], cos(ang3), sin(ang3), &a3r, &a3i);

                real[i0] = a0r + a1r + a2r + a3r;
                imag[i0] = a0i + a1i + a2i + a3i;

                real[i2] = a0r - a1r + a2r - a3r;
                imag[i2] = a0i - a1i + a2i - a3i;

                if (!inverse) {
                    real[i1] = a0r + a1i - a2r - a3i;
                    imag[i1] = a0i - a1r - a2i + a3r;
                    real[i3] = a0r - a1i - a2r + a3i;
                    imag[i3] = a0i + a1r - a2i - a3r;
                } else {
                    real[i1] = a0r - a1i - a2r + a3i;
                    imag[i1] = a0i + a1r - a2i - a3r;
                    real[i3] = a0r + a1i - a2r - a3i;
                    imag[i3] = a0i - a1r - a2i + a3r;
                }
            }
        }

        if (len == n) {
            break;
        }
        len *= 4;
    }

    if (inverse) {
        double scale = 1.0 / (double)n;
        for (i = 0; i < n; ++i) {
            real[i] *= scale;
            imag[i] *= scale;
        }
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_fft_radix4(double* real, double* imag, size_t n, int inverse) {
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
        double* tmp_real = NULL;
        double* tmp_imag = NULL;

        if (lmmc_mul_overflow_size(nfft, sizeof(double), &nfft_bytes)) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }

        tmp_real = (double*)lmmc_alloc(nfft_bytes);
        tmp_imag = (double*)lmmc_alloc(nfft_bytes);
        if (tmp_real == NULL || tmp_imag == NULL) {
            if (tmp_real != NULL) {
                lmmc_free(tmp_real);
            }
            if (tmp_imag != NULL) {
                lmmc_free(tmp_imag);
            }
            return LMMC_STATUS_ALLOCATION_FAILED;
        }

        memcpy(tmp_real, real, n * sizeof(double));
        memcpy(tmp_imag, imag, n * sizeof(double));
        memset(tmp_real + n, 0, (nfft - n) * sizeof(double));
        memset(tmp_imag + n, 0, (nfft - n) * sizeof(double));

        st = lmmc_fft_radix4_core(tmp_real, tmp_imag, nfft, inverse);
        if (st == LMMC_STATUS_OK) {
            memcpy(real, tmp_real, n * sizeof(double));
            memcpy(imag, tmp_imag, n * sizeof(double));
        }

        lmmc_free(tmp_real);
        lmmc_free(tmp_imag);
    }

    return st;
}

lmmc_status_t lmmc_fft_radix4_forward(double* real, double* imag, size_t n) {
    return lmmc_fft_radix4(real, imag, n, 0);
}

lmmc_status_t lmmc_fft_radix4_inverse(double* real, double* imag, size_t n) {
    return lmmc_fft_radix4(real, imag, n, 1);
}

lmmc_status_t lmmc_double_nearly_equal_tol(
    double a,
    double b,
    double abs_tol,
    double rel_tol,
    int* out_equal
) {
    double diff = 0.0;
    double scale = 0.0;
    double threshold = 0.0;

    if (out_equal == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!(abs_tol >= 0.0) || !(rel_tol >= 0.0) || !isfinite(abs_tol) || !isfinite(rel_tol)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (isnan(a) || isnan(b)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (isinf(a) || isinf(b)) {
        *out_equal = (a == b) ? 1 : 0;
        return LMMC_STATUS_OK;
    }

    diff = lmmc_absd(a - b);
    scale = lmmc_maxd(lmmc_absd(a), lmmc_absd(b));
    threshold = abs_tol + rel_tol * scale;

    if (isnan(diff) || isnan(threshold)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (isinf(threshold)) {
        *out_equal = 1;
        return LMMC_STATUS_OK;
    }

    *out_equal = (diff <= threshold) ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_double_nearly_equal(double a, double b, int* out_equal) {
    return lmmc_double_nearly_equal_tol(a, b, LMMC_DEFAULT_ABS_TOL, LMMC_DEFAULT_REL_TOL, out_equal);
}
