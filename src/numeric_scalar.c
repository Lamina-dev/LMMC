#include <math.h>
#include "lmmc/numeric_scalar.h"

lmmc_status_t lmmc_atan2(lmmc_real_t y, lmmc_real_t x, lmmc_real_t* out_res) {
    if (out_res == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    LMMC_REAL_SET_D(out_res, atan2(y, x));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sincos(lmmc_real_t x, lmmc_real_t* out_sin, lmmc_real_t* out_cos) {
    double sine, cosine;
    if (out_sin == NULL || out_cos == NULL || out_sin == out_cos || !isfinite(x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    sine = sin(x);
    cosine = cos(x);
    if (!isfinite(sine) || !isfinite(cosine)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_SET_D(out_sin, sine);
    LMMC_REAL_SET_D(out_cos, cosine);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_hypot(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res) {
    double result;
    if (out_res == NULL || !isfinite(x) || !isfinite(y)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    result = hypot(x, y);
    if (!isfinite(result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_SET_D(out_res, result);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_exp2(lmmc_real_t x, lmmc_real_t* out_res) {
    double result;
    if (out_res == NULL || !isfinite(x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    result = exp2(x);
    if (!isfinite(result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_SET_D(out_res, result);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_log2(lmmc_real_t x, lmmc_real_t* out_res) {
    double result;
    if (out_res == NULL || !isfinite(x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x <= 0.0) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    result = log2(x);
    if (!isfinite(result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_SET_D(out_res, result);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_expm1(lmmc_real_t x, lmmc_real_t* out_res) {
    double result;
    if (out_res == NULL || !isfinite(x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    result = expm1(x);
    if (!isfinite(result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_SET_D(out_res, result);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_log1p(lmmc_real_t x, lmmc_real_t* out_res) {
    double result;
    if (out_res == NULL || !isfinite(x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x <= -1.0) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    result = log1p(x);
    if (!isfinite(result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_SET_D(out_res, result);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_split_int_frac(lmmc_real_t x, lmmc_real_t* out_iptr, lmmc_real_t* out_frac) {
    if (out_iptr == NULL || out_frac == NULL || out_iptr == out_frac) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    {
        double ip;
        double fr = modf(x, &ip);
        LMMC_REAL_SET_D(out_iptr, ip);
        LMMC_REAL_SET_D(out_frac, fr);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_fmod(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res) {
    double result;
    if (out_res == NULL || isnan(x) || isnan(y)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (isinf(x) || y == 0.0) { return LMMC_STATUS_OUT_OF_RANGE; }
    result = fmod(x, y);
    if (!isfinite(result)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    LMMC_REAL_SET_D(out_res, result);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ldexp(lmmc_real_t x, int exp, lmmc_real_t* out_res) {
    double result;
    if (out_res == NULL || !isfinite(x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    result = ldexp(x, exp);
    if (!isfinite(result) || (x != 0.0 && result == 0.0)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_SET_D(out_res, result);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_nextafter(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out_res) {
    if (out_res == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    LMMC_REAL_SET_D(out_res, nextafter(x, y));
    return LMMC_STATUS_OK;
}
static lmmc_status_t store_finite_scalar_result(
    lmmc_real_t value, lmmc_real_t* out) {
    if (!isfinite(value)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *out = value;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_asin(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x < -1.0 || x > 1.0) { return LMMC_STATUS_OUT_OF_RANGE; }
    return store_finite_scalar_result(asin(x), out);
}

lmmc_status_t lmmc_acos(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x < -1.0 || x > 1.0) { return LMMC_STATUS_OUT_OF_RANGE; }
    return store_finite_scalar_result(acos(x), out);
}

lmmc_status_t lmmc_atan(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return store_finite_scalar_result(atan(x), out);
}

lmmc_status_t lmmc_sinh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return store_finite_scalar_result(sinh(x), out);
}

lmmc_status_t lmmc_cosh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return store_finite_scalar_result(cosh(x), out);
}

lmmc_status_t lmmc_tanh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return store_finite_scalar_result(tanh(x), out);
}

lmmc_status_t lmmc_asinh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return store_finite_scalar_result(asinh(x), out);
}

lmmc_status_t lmmc_acosh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x < 1.0) { return LMMC_STATUS_OUT_OF_RANGE; }
    return store_finite_scalar_result(acosh(x), out);
}

lmmc_status_t lmmc_atanh(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= -1.0 || x >= 1.0) { return LMMC_STATUS_OUT_OF_RANGE; }
    return store_finite_scalar_result(atanh(x), out);
}

lmmc_status_t lmmc_pow(lmmc_real_t x, lmmc_real_t y, lmmc_real_t* out) {
    if (!out || !isfinite(x) || !isfinite(y)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if ((x == 0.0 && y < 0.0) ||
        (x < 0.0 && trunc(y) != y)) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    return store_finite_scalar_result(pow(x, y), out);
}

lmmc_status_t lmmc_ceil(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return store_finite_scalar_result(ceil(x), out);
}

lmmc_status_t lmmc_floor(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return store_finite_scalar_result(floor(x), out);
}

lmmc_status_t lmmc_round(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return store_finite_scalar_result(round(x), out);
}

lmmc_status_t lmmc_trunc(lmmc_real_t x, lmmc_real_t* out) {
    if (!out || !isfinite(x)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return store_finite_scalar_result(trunc(x), out);
}
