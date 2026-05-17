#ifndef LMMC_NUMERIC_H
#define LMMC_NUMERIC_H

#include <stddef.h>
#include "lmmc/config.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LMMC_PI
#define LMMC_PI (3.14159265358979323846)
#endif
#ifndef LMMC_INV_PI
#define LMMC_INV_PI (0.31830988618379067154)
#endif
#ifndef LMMC_SQRT2
#define LMMC_SQRT2 (1.41421356237309504880)
#endif

#ifndef LMMC_DEFAULT_ABS_TOL
#define LMMC_DEFAULT_ABS_TOL 1e-12
#endif
#ifndef LMMC_DEFAULT_REL_TOL
#define LMMC_DEFAULT_REL_TOL 1e-10
#endif

/* Constants */
lmmc_status_t lmmc_inf(lmmc_real_t* out_inf);
lmmc_status_t lmmc_nan(lmmc_real_t* out_nan);
lmmc_status_t lmmc_eps(lmmc_real_t* out_eps);

/* Float Properties */
lmmc_status_t lmmc_isnan(lmmc_real_t x, int* out_isnan);
lmmc_status_t lmmc_isinf(lmmc_real_t x, int* out_isinf);
lmmc_status_t lmmc_isfinite(lmmc_real_t x, int* out_isfinite);
lmmc_status_t lmmc_signbit(lmmc_real_t x, int* out_signbit);

/* Enhanced Trig */
lmmc_status_t lmmc_atan2(lmmc_real_t y, lmmc_real_t x, lmmc_real_t* out_res);
lmmc_status_t lmmc_sincos(lmmc_real_t x, lmmc_real_t* out_sin, lmmc_real_t* out_cos);
lmmc_status_t lmmc_hypot(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res);

/* Power and Log */
lmmc_status_t lmmc_exp2(lmmc_real_t x, lmmc_real_t* out_res);
lmmc_status_t lmmc_log2(lmmc_real_t x, lmmc_real_t* out_res);
lmmc_status_t lmmc_expm1(lmmc_real_t x, lmmc_real_t* out_res);
lmmc_status_t lmmc_log1p(lmmc_real_t x, lmmc_real_t* out_res);

/* Float Manipulation */
lmmc_status_t lmmc_split_int_frac(lmmc_real_t x, lmmc_real_t* out_iptr, lmmc_real_t* out_frac);
lmmc_status_t lmmc_fmod(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res);
lmmc_status_t lmmc_ldexp(lmmc_real_t x, int exp, lmmc_real_t* out_res);
lmmc_status_t lmmc_nextafter(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res);

/* Comparison */
lmmc_status_t lmmc_approx_eq(lmmc_real_t a, lmmc_real_t b, lmmc_real_t epsilon, int* out_equal);

lmmc_status_t lmmc_double_nearly_equal_tol(
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t abs_tol,
    lmmc_real_t rel_tol,
    int* out_equal
);

lmmc_status_t lmmc_double_nearly_equal(lmmc_real_t a, lmmc_real_t b, int* out_equal);

lmmc_status_t lmmc_fft_radix4_next_size(size_t n, size_t* out_nfft);
lmmc_status_t lmmc_fft_radix4(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse);
lmmc_status_t lmmc_fft_radix4_forward(lmmc_real_t* real, lmmc_real_t* imag, size_t n);
lmmc_status_t lmmc_fft_radix4_inverse(lmmc_real_t* real, lmmc_real_t* imag, size_t n);

lmmc_status_t lmmc_lambertw(lmmc_real_t z, lmmc_real_t* out_res);

#ifdef __cplusplus
}
#endif

#endif
