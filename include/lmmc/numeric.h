#ifndef LMMC_NUMERIC_H
#define LMMC_NUMERIC_H

#include <stddef.h>
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LMMC_PI 3.14159265358979323846
#define LMMC_INV_PI 0.31830988618379067154
#define LMMC_SQRT2 1.41421356237309504880

#define LMMC_DEFAULT_ABS_TOL 1e-12
#define LMMC_DEFAULT_REL_TOL 1e-10

/* Constants */
lmmc_status_t lmmc_inf(double* out_inf);
lmmc_status_t lmmc_nan(double* out_nan);
lmmc_status_t lmmc_eps(double* out_eps);

/* Float Properties */
lmmc_status_t lmmc_isnan(double x, int* out_isnan);
lmmc_status_t lmmc_isinf(double x, int* out_isinf);
lmmc_status_t lmmc_isfinite(double x, int* out_isfinite);
lmmc_status_t lmmc_signbit(double x, int* out_signbit);

/* Enhanced Trig */
lmmc_status_t lmmc_atan2(double y, double x, double* out_res);
lmmc_status_t lmmc_sincos(double x, double* out_sin, double* out_cos);
lmmc_status_t lmmc_hypot(double x, double y, double* out_res);

/* Power and Log */
lmmc_status_t lmmc_exp2(double x, double* out_res);
lmmc_status_t lmmc_log2(double x, double* out_res);
lmmc_status_t lmmc_expm1(double x, double* out_res);
lmmc_status_t lmmc_log1p(double x, double* out_res);

/* Float Manipulation */
lmmc_status_t lmmc_modf(double x, double* out_iptr, double* out_frac);
lmmc_status_t lmmc_fmod(double x, double y, double* out_res);
lmmc_status_t lmmc_ldexp(double x, int exp, double* out_res);
lmmc_status_t lmmc_nextafter(double x, double y, double* out_res);

/* Comparison */
lmmc_status_t lmmc_approx_eq(double a, double b, double epsilon, int* out_equal);

lmmc_status_t lmmc_double_nearly_equal_tol(
    double a,
    double b,
    double abs_tol,
    double rel_tol,
    int* out_equal
);

lmmc_status_t lmmc_double_nearly_equal(double a, double b, int* out_equal);

lmmc_status_t lmmc_fft_radix4_next_size(size_t n, size_t* out_nfft);
lmmc_status_t lmmc_fft_radix4(double* real, double* imag, size_t n, int inverse);
lmmc_status_t lmmc_fft_radix4_forward(double* real, double* imag, size_t n);
lmmc_status_t lmmc_fft_radix4_inverse(double* real, double* imag, size_t n);

#ifdef __cplusplus
}
#endif

#endif
