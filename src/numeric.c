/**
 * @file numeric.c
 * @brief 数值常量、标量分类与近似比较。
 */
#include <math.h>
#include <float.h>
#include "lmmc/numeric_scalar.h"

static void lmmc_max_inplace(lmmc_real_t* res, const lmmc_real_t* a, const lmmc_real_t* b) {
    if (LMMC_REAL_CMP(a, b) > 0) {
        LMMC_REAL_SET(res, a);
    } else {
        LMMC_REAL_SET(res, b);
    }
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
    LMMC_REAL_SET_D(out_eps, DBL_EPSILON);
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

lmmc_status_t lmmc_approx_eq(lmmc_real_t a, lmmc_real_t b, lmmc_real_t epsilon, int* out_equal) {
    lmmc_real_t diff, zero;
    if (out_equal == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(epsilon) || epsilon < 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
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
    LMMC_REAL_ABS(&diff, &tmp1);

    LMMC_REAL_ABS(&tmp1, &a);
    LMMC_REAL_ABS(&tmp2, &b);
    lmmc_max_inplace(&scale, &tmp1, &tmp2);

    LMMC_REAL_MUL(&tmp1, &rel_tol, &scale);
    lmmc_max_inplace(&threshold, &abs_tol, &tmp1);

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
