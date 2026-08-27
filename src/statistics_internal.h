#ifndef LMMC_STATISTICS_INTERNAL_H
#define LMMC_STATISTICS_INTERNAL_H

/* Private shared helpers for the statistics translation units. */
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"

static inline int lmmc_mul_overflow_size(size_t a, size_t b, size_t* out) {
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

static inline int lmmc_is_finite_number(lmmc_real_t v) {
    return LMMC_REAL_IS_FINITE(&v) ? 1 : 0;
}

static inline lmmc_status_t lmmc_validate_vec(const lmmc_vec_t* x) {
    if (x == NULL || x->data == NULL || x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_validate_mat(const lmmc_mat_t* x) {
    if (x == NULL || x->data == NULL || x->rows == 0 || x->cols == 0 || x->stride < x->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_finalize_nonnegative(lmmc_real_t value, lmmc_real_t* out_value) {
    lmmc_real_t zero, tol;
    if (!lmmc_is_finite_number(value) || out_value == NULL) {
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

/* Regularized incomplete gamma/beta helpers (defined in statistics_internal.c). */
lmmc_real_t regularized_gamma_lower(lmmc_real_t a, lmmc_real_t x);
lmmc_real_t regularized_gamma_upper_cf(lmmc_real_t a, lmmc_real_t x);
lmmc_real_t regularized_beta_cf(lmmc_real_t x, lmmc_real_t a, lmmc_real_t b);
lmmc_real_t regularized_beta(lmmc_real_t x, lmmc_real_t a, lmmc_real_t b);

#endif
