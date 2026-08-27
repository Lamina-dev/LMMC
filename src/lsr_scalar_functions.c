#include "lmmc/lsr_stdlib.h"

#include <math.h>

#include "lsr_stdlib_internal.h"

const char* lmmc_lsr_error_name(lmmc_status_t status)
{
    switch (status) {
    case LMMC_STATUS_OK:
        return "Ok";
    case LMMC_STATUS_INVALID_ARGUMENT:
        return "InvalidArgument";
    case LMMC_STATUS_DIMENSION_MISMATCH:
        return "DimensionMismatch";
    case LMMC_STATUS_ALLOCATION_FAILED:
        return "ResourceLimit";
    case LMMC_STATUS_SINGULAR_MATRIX:
        return "SingularMatrix";
    case LMMC_STATUS_NOT_IMPLEMENTED:
        return "UnsupportedExpression";
    case LMMC_STATUS_NUMERICAL_FAILURE:
        return "NumericFailure";
    case LMMC_STATUS_NOT_POSITIVE_DEFINITE:
        return "DomainError";
    case LMMC_STATUS_CONVERGENCE_FAILED:
        return "NumericFailure";
    case LMMC_STATUS_OUT_OF_RANGE:
        return "DomainError";
    case LMMC_STATUS_INDEX_OUT_OF_BOUNDS:
        return "InvalidArgument";
    case LMMC_STATUS_WARNING_MAX_DEPTH:
        return "ResourceLimit";
    case LMMC_STATUS_EMPTY_INPUT:
        return "EmptyInput";
    case LMMC_STATUS_UNIT_STRIP_TYPE_MISMATCH:
        return "UnitStripTypeMismatch";
    case LMMC_STATUS_UNIT_STRIP_OVERFLOW:
        return "UnitStripOverflow";
    case LMMC_STATUS_UNIT_STRIP_INVALID:
        return "UnitStripInvalid";
    case LMMC_STATUS_UNIT_STRIP_LEGACY_SYNTAX:
        return "UnitStripLegacySyntax";
    }
    return "InternalInvariant";
}
lmmc_status_t lmmc_lsr_math_sin(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_store_finite_real((lmmc_real_t)sin((double)x), out);
}

lmmc_status_t lmmc_lsr_math_cos(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_store_finite_real((lmmc_real_t)cos((double)x), out);
}

lmmc_status_t lmmc_lsr_math_tan(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_store_finite_real((lmmc_real_t)tan((double)x), out);
}

lmmc_status_t lmmc_lsr_math_pow(lmmc_real_t x, lmmc_real_t y,
                                lmmc_real_t* out)
{
    double value;
    double integral_part;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x) || !lmmc_lsr_real_is_finite(y)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if ((double)x == 0.0 && (double)y < 0.0) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    if ((double)x < 0.0 && modf((double)y, &integral_part) != 0.0) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    value = pow((double)x, (double)y);
    if (!isfinite(value)) return LMMC_STATUS_NUMERICAL_FAILURE;
    *out = (lmmc_real_t)value;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_asin(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    if (x < (lmmc_real_t)-1 || x > (lmmc_real_t)1) return LMMC_STATUS_OUT_OF_RANGE;
    return lmmc_lsr_store_finite_real((lmmc_real_t)asin((double)x), out);
}

lmmc_status_t lmmc_lsr_math_acos(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    if (x < (lmmc_real_t)-1 || x > (lmmc_real_t)1) return LMMC_STATUS_OUT_OF_RANGE;
    return lmmc_lsr_store_finite_real((lmmc_real_t)acos((double)x), out);
}

lmmc_status_t lmmc_lsr_math_atan(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_store_finite_real((lmmc_real_t)atan((double)x), out);
}

lmmc_status_t lmmc_lsr_math_sqrt(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    if (x < (lmmc_real_t)0) return LMMC_STATUS_OUT_OF_RANGE;
    return lmmc_lsr_store_finite_real((lmmc_real_t)sqrt((double)x), out);
}

lmmc_status_t lmmc_lsr_math_exp(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_store_finite_real((lmmc_real_t)exp((double)x), out);
}

lmmc_status_t lmmc_lsr_math_ln(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    if (x <= (lmmc_real_t)0) return LMMC_STATUS_OUT_OF_RANGE;
    return lmmc_lsr_store_finite_real((lmmc_real_t)log((double)x), out);
}

lmmc_status_t lmmc_lsr_math_log(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_math_ln(x, out);
}

lmmc_status_t lmmc_lsr_math_log_base(lmmc_real_t x, lmmc_real_t base,
                                     lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x) || !lmmc_lsr_real_is_finite(base)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (x <= (lmmc_real_t)0 || base <= (lmmc_real_t)0 ||
        base == (lmmc_real_t)1) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    return lmmc_lsr_store_finite_real(
        (lmmc_real_t)(log((double)x) / log((double)base)), out);
}

lmmc_status_t lmmc_lsr_math_log10(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    if (x <= (lmmc_real_t)0) return LMMC_STATUS_OUT_OF_RANGE;
    return lmmc_lsr_store_finite_real((lmmc_real_t)log10((double)x), out);
}

lmmc_status_t lmmc_lsr_math_abs(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_store_finite_real((lmmc_real_t)fabs((double)x), out);
}

lmmc_status_t lmmc_lsr_math_floor(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_finite_real((lmmc_real_t)floor((double)x), out);
}

lmmc_status_t lmmc_lsr_math_ceil(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_finite_real((lmmc_real_t)ceil((double)x), out);
}

lmmc_status_t lmmc_lsr_math_round(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_finite_real((lmmc_real_t)round((double)x), out);
}

lmmc_status_t lmmc_lsr_math_clamp(lmmc_real_t x, lmmc_real_t lo,
                                  lmmc_real_t hi, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x) || !lmmc_lsr_real_is_finite(lo) ||
        !lmmc_lsr_real_is_finite(hi)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (lo > hi) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < lo) *out = lo;
    else if (x > hi) *out = hi;
    else *out = x;
    return lmmc_lsr_store_finite_real(*out, out);
}
