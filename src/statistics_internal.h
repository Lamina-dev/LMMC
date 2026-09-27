#ifndef LMMC_STATISTICS_INTERNAL_H
#define LMMC_STATISTICS_INTERNAL_H

/* Private shared helpers for the statistics translation units. */
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "internal.h"
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"


static inline lmmc_status_t lmmc_validate_vec(const lmmc_vec_t* x) {
    return lmmc_vec_descriptor_is_valid(x)
        ? LMMC_STATUS_OK
        : LMMC_STATUS_INVALID_ARGUMENT;
}

static inline lmmc_status_t lmmc_validate_mat(const lmmc_mat_t* x) {
    return lmmc_mat_descriptor_is_valid(x)
        ? LMMC_STATUS_OK
        : LMMC_STATUS_INVALID_ARGUMENT;
}

static inline lmmc_status_t lmmc_finalize_nonnegative(lmmc_real_t value, lmmc_real_t* out_value) {
    lmmc_real_t zero, tol;
    if (!lmmc_is_finite(&value) || out_value == NULL) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    LMMC_REAL_INIT(&zero);
    LMMC_REAL_INIT(&tol);

    LMMC_REAL_SET_D(&zero, 0.0);
    LMMC_REAL_SET_D(&tol, LMMC_REAL_EPSILON);
    LMMC_REAL_NEG(&tol, &tol);

    if (LMMC_REAL_CMP(&value, &zero) < 0) {
        if (LMMC_REAL_CMP(&value, &tol) > 0) {
            LMMC_REAL_SET_D(&value, 0.0);
        } else {
            LMMC_REAL_CLEAR(&zero);
            LMMC_REAL_CLEAR(&tol);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    LMMC_REAL_SET(out_value, &value);

    LMMC_REAL_CLEAR(&zero);
    LMMC_REAL_CLEAR(&tol);
    return LMMC_STATUS_OK;
}

#ifndef LMMC_SQRT_2PI
#define LMMC_SQRT_2PI 2.5066282746310002
#endif

#ifndef LMMC_LOG_SQRT_2PI
#define LMMC_LOG_SQRT_2PI 0.9189385332046727
#endif

/* Stable log-factorial/log-gamma corrections and Poisson deviance primitives
 * shared by discrete and continuous distribution kernels. */
lmmc_real_t lmmc_stirling_error(lmmc_real_t x);
lmmc_real_t lmmc_log_gamma_stirling_error(lmmc_real_t x);
lmmc_real_t lmmc_deviance_part(lmmc_real_t x, lmmc_real_t mean);

/**
 * @brief 稳定计算 log B(a,b) 和 log(x^a complement_x^b / B(a,b))。
 * 独立传入补数以保留微小上尾；负无穷表示有效下溢，失败时输出保持不变。
 */
lmmc_status_t lmmc_log_beta(double a, double b, double* out_log_beta);
lmmc_status_t lmmc_beta_log_kernel(
    double x, double complement_x, double a, double b, double* out_log_kernel);

/* Regularized incomplete gamma/beta helpers (defined in statistics_internal.c).
 * Results are written only after a finite, in-range value has converged. */
lmmc_status_t regularized_gamma_lower(
    lmmc_real_t a, lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t regularized_gamma_upper_cf(
    lmmc_real_t a, lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t regularized_gamma_upper(
    lmmc_real_t a, lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t regularized_beta(
    lmmc_real_t x, lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out);
/** @brief 计算上尾；1 - x 舍入至端点时仍保留 x。 */
lmmc_status_t regularized_beta_upper(
    lmmc_real_t x, lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out);

#endif
