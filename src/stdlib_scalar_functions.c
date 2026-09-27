#include "lmmc/stdlib.h"

#include <math.h>

#include "stdlib_internal.h"

const char* lmmc_std_error_name(lmmc_status_t status)
{
    static const char* const names[] = {
        [LMMC_STATUS_OK] = "Ok",
        [LMMC_STATUS_INVALID_ARGUMENT] = "InvalidArgument",
        [LMMC_STATUS_DIMENSION_MISMATCH] = "DimensionMismatch",
        [LMMC_STATUS_ALLOCATION_FAILED] = "ResourceLimit",
        [LMMC_STATUS_SINGULAR_MATRIX] = "SingularMatrix",
        [LMMC_STATUS_NOT_IMPLEMENTED] = "UnsupportedExpression",
        [LMMC_STATUS_NUMERICAL_FAILURE] = "NumericFailure",
        [LMMC_STATUS_NOT_POSITIVE_DEFINITE] = "DomainError",
        [LMMC_STATUS_CONVERGENCE_FAILED] = "NumericFailure",
        [LMMC_STATUS_OUT_OF_RANGE] = "DomainError",
        [LMMC_STATUS_INDEX_OUT_OF_BOUNDS] = "InvalidArgument",
        [LMMC_STATUS_WARNING_MAX_DEPTH] = "ResourceLimit",
        [LMMC_STATUS_EMPTY_INPUT] = "EmptyInput",
        [LMMC_STATUS_UNIT_STRIP_TYPE_MISMATCH] = "UnitStripTypeMismatch",
        [LMMC_STATUS_UNIT_STRIP_OVERFLOW] = "UnitStripOverflow",
        [LMMC_STATUS_UNIT_STRIP_INVALID] = "UnitStripInvalid",
        [LMMC_STATUS_UNIT_STRIP_LEGACY_SYNTAX] = "UnitStripLegacySyntax",
        [LMMC_STATUS_NOT_INITIALIZED] = "NotInitialized",
        [LMMC_STATUS_BUSY] = "Busy",
        [LMMC_STATUS_REFERENCE_LIMIT] = "ResourceLimit",
    };
    if ((unsigned int)status >= sizeof(names) / sizeof(names[0])) {
        return "InternalInvariant";
    }
    return names[(unsigned int)status];
}
lmmc_status_t lmmc_std_math_sin(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    return lmmc_std_store_finite_real((lmmc_real_t)sin((double)x), out);
}

lmmc_status_t lmmc_std_math_cos(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    return lmmc_std_store_finite_real((lmmc_real_t)cos((double)x), out);
}

lmmc_status_t lmmc_std_math_tan(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    return lmmc_std_store_finite_real((lmmc_real_t)tan((double)x), out);
}

lmmc_status_t lmmc_std_math_pow(lmmc_real_t x, lmmc_real_t y,
                                lmmc_real_t* out)
{
    double value;
    double integral_part;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x) || !lmmc_std_real_is_finite(y)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if ((double)x == 0.0 && (double)y < 0.0) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    if ((double)x < 0.0 && modf((double)y, &integral_part) != 0.0) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    value = pow((double)x, (double)y);
    if (!isfinite(value)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *out = (lmmc_real_t)value;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_math_asin(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    if (x < (lmmc_real_t)-1 || x > (lmmc_real_t)1) { return LMMC_STATUS_OUT_OF_RANGE; }
    return lmmc_std_store_finite_real((lmmc_real_t)asin((double)x), out);
}

lmmc_status_t lmmc_std_math_acos(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    if (x < (lmmc_real_t)-1 || x > (lmmc_real_t)1) { return LMMC_STATUS_OUT_OF_RANGE; }
    return lmmc_std_store_finite_real((lmmc_real_t)acos((double)x), out);
}

lmmc_status_t lmmc_std_math_atan(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    return lmmc_std_store_finite_real((lmmc_real_t)atan((double)x), out);
}

lmmc_status_t lmmc_std_math_sqrt(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    if (x < (lmmc_real_t)0) { return LMMC_STATUS_OUT_OF_RANGE; }
    return lmmc_std_store_finite_real((lmmc_real_t)sqrt((double)x), out);
}

lmmc_status_t lmmc_std_math_exp(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    return lmmc_std_store_finite_real((lmmc_real_t)exp((double)x), out);
}

lmmc_status_t lmmc_std_math_ln(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    if (x <= (lmmc_real_t)0) { return LMMC_STATUS_OUT_OF_RANGE; }
    return lmmc_std_store_finite_real((lmmc_real_t)log((double)x), out);
}

lmmc_status_t lmmc_std_math_log(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_std_math_ln(x, out);
}

lmmc_status_t lmmc_std_math_log_base(lmmc_real_t x, lmmc_real_t base,
                                     lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x) || !lmmc_std_real_is_finite(base)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (x <= (lmmc_real_t)0 || base <= (lmmc_real_t)0 ||
        base == (lmmc_real_t)1) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    return lmmc_std_store_finite_real(
        (lmmc_real_t)(log((double)x) / log((double)base)), out);
}

lmmc_status_t lmmc_std_math_log10(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    if (x <= (lmmc_real_t)0) { return LMMC_STATUS_OUT_OF_RANGE; }
    return lmmc_std_store_finite_real((lmmc_real_t)log10((double)x), out);
}

lmmc_status_t lmmc_std_math_abs(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    return lmmc_std_store_finite_real((lmmc_real_t)fabs((double)x), out);
}

lmmc_status_t lmmc_std_math_floor(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_std_store_finite_real((lmmc_real_t)floor((double)x), out);
}

lmmc_status_t lmmc_std_math_ceil(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_std_store_finite_real((lmmc_real_t)ceil((double)x), out);
}

lmmc_status_t lmmc_std_math_round(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_std_store_finite_real((lmmc_real_t)round((double)x), out);
}

lmmc_status_t lmmc_std_math_clamp(lmmc_real_t x, lmmc_real_t lo,
                                  lmmc_real_t hi, lmmc_real_t* out)
{
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(x) || !lmmc_std_real_is_finite(lo) ||
        !lmmc_std_real_is_finite(hi)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (lo > hi) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x < lo) *out = lo;
    else if (x > hi) *out = hi;
    else *out = x;
    return lmmc_std_store_finite_real(*out, out);
}
