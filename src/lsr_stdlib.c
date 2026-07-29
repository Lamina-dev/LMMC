#include "lmmc/lsr_stdlib.h"

#include <math.h>
#include <stdlib.h>

#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/linear_algebra.h"
#include "lmmc/numeric.h"
#include "lmmc/random.h"
#include "lmmc/stats.h"

#include <ctype.h>
#include <string.h>

static lmmc_status_t lmmc_lsr_store_real(lmmc_real_t value, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = value;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_store_finite_real(lmmc_real_t value,
                                                lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite((double)value)) return LMMC_STATUS_NUMERICAL_FAILURE;
    *out = value;
    return LMMC_STATUS_OK;
}

typedef struct {
    const char* name;
    lmmc_real_t value;
    const char* unit;
} lmmc_lsr_constant_entry_t;

static const lmmc_lsr_constant_entry_t lmmc_lsr_constants[] = {
    {"EARTH_GRAVITY", (lmmc_real_t)9.80665, "m*s^-2"},
    {"MOON_GRAVITY", (lmmc_real_t)1.625, "m*s^-2"},
    {"MARS_GRAVITY", (lmmc_real_t)3.72076, "m*s^-2"},
    {"WATER_DENSITY", (lmmc_real_t)1000.0, "kg*m^-3"},
    {"STANDARD_PRESSURE", (lmmc_real_t)101325.0, "Pa"},
    {"STANDARD_TEMPERATURE", (lmmc_real_t)273.15, "K"},
    {"AIR_DENSITY", (lmmc_real_t)1.225, "kg*m^-3"},
    {"C", (lmmc_real_t)2.99792458e8, "m*s^-1"},
    {"G", (lmmc_real_t)6.67430e-11, "m^3*kg^-1*s^-2"},
    {"H", (lmmc_real_t)6.62607015e-34, "J*s"},
    {"KB", (lmmc_real_t)1.380649e-23, "J*K^-1"},
    {"EPSILON_0", (lmmc_real_t)8.8541878128e-12, "F*m^-1"},
    {"MU_0", (lmmc_real_t)1.25663706212e-6, "H*m^-1"},
    {"AVOGADRO", (lmmc_real_t)6.02214076e23, "mol^-1"},
    {"R", (lmmc_real_t)8.314462618, "J*mol^-1*K^-1"},
    {"FARADAY", (lmmc_real_t)9.648533212e4, "C*mol^-1"},
    {"AMU", (lmmc_real_t)1.66053906660e-27, "kg"},
    {"MOLAR_VOLUME_IDEAL", (lmmc_real_t)0.024465, "m^3*mol^-1"},
    {"ROOM_PRESSURE", (lmmc_real_t)1.0e5, "Pa"},
    {"ROOM_TEMPERATURE", (lmmc_real_t)297.15, "K"},
};

static size_t lmmc_lsr_constants_length(void)
{
    return sizeof(lmmc_lsr_constants) / sizeof(lmmc_lsr_constants[0]);
}

static const lmmc_lsr_constant_entry_t*
lmmc_lsr_find_constant(const char* name)
{
    if (!name) return NULL;
    for (size_t i = 0; i < lmmc_lsr_constants_length(); ++i) {
        if (strcmp(lmmc_lsr_constants[i].name, name) == 0) {
            return &lmmc_lsr_constants[i];
        }
    }
    return NULL;
}

typedef struct {
    double scale;
    int dims[7];
} lmmc_lsr_unit_sig_t;

typedef struct {
    const char* name;
    double scale;
    int dims[7];
} lmmc_lsr_unit_entry_t;

static const lmmc_lsr_unit_entry_t lmmc_lsr_units[] = {
    {"1", 1.0, {0, 0, 0, 0, 0, 0, 0}},
    {"m", 1.0, {1, 0, 0, 0, 0, 0, 0}},
    {"km", 1000.0, {1, 0, 0, 0, 0, 0, 0}},
    {"s", 1.0, {0, 0, 1, 0, 0, 0, 0}},
    {"h", 3600.0, {0, 0, 1, 0, 0, 0, 0}},
    {"kg", 1.0, {0, 1, 0, 0, 0, 0, 0}},
    {"g", 0.001, {0, 1, 0, 0, 0, 0, 0}},
    {"A", 1.0, {0, 0, 0, 1, 0, 0, 0}},
    {"K", 1.0, {0, 0, 0, 0, 1, 0, 0}},
    {"mol", 1.0, {0, 0, 0, 0, 0, 1, 0}},
    {"cd", 1.0, {0, 0, 0, 0, 0, 0, 1}},
    {"N", 1.0, {1, 1, -2, 0, 0, 0, 0}},
    {"Pa", 1.0, {-1, 1, -2, 0, 0, 0, 0}},
    {"J", 1.0, {2, 1, -2, 0, 0, 0, 0}},
    {"C", 1.0, {0, 0, 1, 1, 0, 0, 0}},
    {"F", 1.0, {-2, -1, 4, 2, 0, 0, 0}},
    {"H", 1.0, {2, 1, -2, -2, 0, 0, 0}},
};

static const lmmc_lsr_unit_entry_t* lmmc_lsr_find_unit(const char* name,
                                                       size_t length)
{
    for (size_t i = 0; i < sizeof(lmmc_lsr_units) / sizeof(lmmc_lsr_units[0]);
         ++i) {
        if (strlen(lmmc_lsr_units[i].name) == length &&
            strncmp(lmmc_lsr_units[i].name, name, length) == 0) {
            return &lmmc_lsr_units[i];
        }
    }
    return NULL;
}

static void lmmc_lsr_unit_identity(lmmc_lsr_unit_sig_t* sig)
{
    sig->scale = 1.0;
    for (size_t i = 0; i < 7; ++i) sig->dims[i] = 0;
}

static int lmmc_lsr_parse_int(const char** cursor, int* out)
{
    int sign = 1;
    int value = 0;
    int have_digit = 0;
    if (**cursor == '+') {
        ++(*cursor);
    } else if (**cursor == '-') {
        sign = -1;
        ++(*cursor);
    }
    while (isdigit((unsigned char)**cursor)) {
        have_digit = 1;
        value = value * 10 + (**cursor - '0');
        ++(*cursor);
    }
    if (!have_digit) return 0;
    *out = sign * value;
    return 1;
}

static int lmmc_lsr_parse_unit_expr(const char* text,
                                    lmmc_lsr_unit_sig_t* out)
{
    const char* cursor = text;
    int op_sign = 1;
    lmmc_lsr_unit_identity(out);
    if (!text || !*text) return 0;
    if (strcmp(text, "1") == 0) return 1;
    while (*cursor) {
        const char* name_begin = cursor;
        int exponent = 1;
        while (isalpha((unsigned char)*cursor) || *cursor == '_') ++cursor;
        if (cursor == name_begin) return 0;
        const lmmc_lsr_unit_entry_t* unit =
            lmmc_lsr_find_unit(name_begin, (size_t)(cursor - name_begin));
        if (!unit) return 0;
        if (*cursor == '^') {
            ++cursor;
            if (!lmmc_lsr_parse_int(&cursor, &exponent)) return 0;
        }
        exponent *= op_sign;
        out->scale *= pow(unit->scale, (double)exponent);
        for (size_t i = 0; i < 7; ++i) {
            out->dims[i] += unit->dims[i] * exponent;
        }
        if (*cursor == '\0') break;
        if (*cursor == '*') {
            op_sign = 1;
        } else if (*cursor == '/') {
            op_sign = -1;
        } else {
            return 0;
        }
        ++cursor;
    }
    return 1;
}

static int lmmc_lsr_same_dimension(const lmmc_lsr_unit_sig_t* lhs,
                                   const lmmc_lsr_unit_sig_t* rhs)
{
    for (size_t i = 0; i < 7; ++i) {
        if (lhs->dims[i] != rhs->dims[i]) return 0;
    }
    return 1;
}

static int lmmc_lsr_dimensionless_sig(const lmmc_lsr_unit_sig_t* sig)
{
    for (size_t i = 0; i < 7; ++i) {
        if (sig->dims[i] != 0) return 0;
    }
    return 1;
}

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
    }
    return "InternalInvariant";
}

lmmc_status_t lmmc_lsr_math_pi(lmmc_real_t* out)
{
    return lmmc_lsr_store_real(LMMC_CONST_PI, out);
}

lmmc_status_t lmmc_lsr_math_e(lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)2.71828182845904523536, out);
}

lmmc_status_t lmmc_lsr_math_phi(lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)1.61803398874989484820, out);
}

size_t lmmc_lsr_constants_count(void)
{
    return lmmc_lsr_constants_length();
}

const char* lmmc_lsr_constants_name(size_t index)
{
    if (index >= lmmc_lsr_constants_length()) return NULL;
    return lmmc_lsr_constants[index].name;
}

lmmc_status_t lmmc_lsr_constants_get(const char* name, lmmc_real_t* out)
{
    const lmmc_lsr_constant_entry_t* entry = lmmc_lsr_find_constant(name);
    if (!entry || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = entry->value;
    return LMMC_STATUS_OK;
}

const char* lmmc_lsr_constants_unit(const char* name)
{
    const lmmc_lsr_constant_entry_t* entry = lmmc_lsr_find_constant(name);
    return entry ? entry->unit : NULL;
}

lmmc_status_t lmmc_lsr_math_i(lmmc_complex_t* out)
{
    return lmmc_complex_create((lmmc_real_t)0, (lmmc_real_t)1, out);
}

lmmc_status_t lmmc_lsr_math_I(lmmc_complex_t* out)
{
    return lmmc_lsr_math_i(out);
}

lmmc_status_t lmmc_lsr_math_complex(lmmc_real_t real,
                                    lmmc_real_t imag,
                                    lmmc_complex_t* out)
{
    return lmmc_complex_create(real, imag, out);
}

lmmc_status_t lmmc_lsr_math_real(const lmmc_complex_t* z, lmmc_real_t* out)
{
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = z->real;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_imag(const lmmc_complex_t* z, lmmc_real_t* out)
{
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = z->imag;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_conj(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    return lmmc_complex_conj(z, out);
}

lmmc_status_t lmmc_lsr_math_complex_abs(const lmmc_complex_t* z,
                                        lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status = lmmc_complex_modulus(z, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_math_sin(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_finite_real((lmmc_real_t)sin((double)x), out);
}

lmmc_status_t lmmc_lsr_math_cos(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_finite_real((lmmc_real_t)cos((double)x), out);
}

lmmc_status_t lmmc_lsr_math_tan(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_finite_real((lmmc_real_t)tan((double)x), out);
}

lmmc_status_t lmmc_lsr_math_pow(lmmc_real_t x, lmmc_real_t y,
                                lmmc_real_t* out)
{
    double value;
    double integral_part;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
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
    return lmmc_asin(x, out);
}

lmmc_status_t lmmc_lsr_math_acos(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_acos(x, out);
}

lmmc_status_t lmmc_lsr_math_atan(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_atan(x, out);
}

lmmc_status_t lmmc_lsr_math_sqrt(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < (lmmc_real_t)0) return LMMC_STATUS_OUT_OF_RANGE;
    *out = (lmmc_real_t)sqrt((double)x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_exp(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_finite_real((lmmc_real_t)exp((double)x), out);
}

lmmc_status_t lmmc_lsr_math_ln(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= (lmmc_real_t)0) return LMMC_STATUS_OUT_OF_RANGE;
    *out = (lmmc_real_t)log((double)x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_log(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_math_ln(x, out);
}

lmmc_status_t lmmc_lsr_math_log_base(lmmc_real_t x, lmmc_real_t base,
                                     lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= (lmmc_real_t)0 || base <= (lmmc_real_t)0 ||
        base == (lmmc_real_t)1) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    *out = (lmmc_real_t)(log((double)x) / log((double)base));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_log10(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= (lmmc_real_t)0) return LMMC_STATUS_OUT_OF_RANGE;
    *out = (lmmc_real_t)log10((double)x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_abs(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_finite_real((lmmc_real_t)fabs((double)x), out);
}

lmmc_status_t lmmc_lsr_math_floor(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_floor(x, out);
}

lmmc_status_t lmmc_lsr_math_ceil(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_ceil(x, out);
}

lmmc_status_t lmmc_lsr_math_round(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_round(x, out);
}

lmmc_status_t lmmc_lsr_math_clamp(lmmc_real_t x, lmmc_real_t lo,
                                  lmmc_real_t hi, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lo > hi) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < lo) *out = lo;
    else if (x > hi) *out = hi;
    else *out = x;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_units_convert(lmmc_real_t x,
                                      const char* from_unit,
                                      const char* to_unit,
                                      lmmc_real_t* out)
{
    lmmc_lsr_unit_sig_t from_sig;
    lmmc_lsr_unit_sig_t to_sig;
    if (!out || !from_unit || !to_unit) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_parse_unit_expr(from_unit, &from_sig) ||
        !lmmc_lsr_parse_unit_expr(to_unit, &to_sig)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_lsr_same_dimension(&from_sig, &to_sig)) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    *out = (lmmc_real_t)((double)x * from_sig.scale / to_sig.scale);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_units_strip(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real(x, out);
}

lmmc_status_t lmmc_lsr_units_is_dimensionless(const char* unit, int* out)
{
    lmmc_lsr_unit_sig_t sig;
    if (!unit || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_parse_unit_expr(unit, &sig)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out = lmmc_lsr_dimensionless_sig(&sig);
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_wrap_const_vec(const lmmc_real_t* values,
                                             size_t count,
                                             lmmc_vec_t* out)
{
    if (!values || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (count == 0) return LMMC_STATUS_EMPTY_INPUT;
    out->size = count;
    out->data = (lmmc_real_t*)values;
    out->owns_data = 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_stats_mean(const lmmc_real_t* values, size_t count,
                                  lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_mean(&view, out);
}

lmmc_status_t lmmc_lsr_stats_median(const lmmc_real_t* values, size_t count,
                                    lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_median(&view, out);
}

lmmc_status_t lmmc_lsr_stats_var(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_variance_sample(&view, out);
}

lmmc_status_t lmmc_lsr_stats_std(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_stddev_sample(&view, out);
}

lmmc_status_t lmmc_lsr_stats_quantile(const lmmc_real_t* values, size_t count,
                                      lmmc_real_t q, lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_quantile(&view, q, out);
}

lmmc_status_t lmmc_lsr_stats_cov(const lmmc_real_t* x,
                                 const lmmc_real_t* y,
                                 size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t x_view;
    lmmc_vec_t y_view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(x, count, &x_view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_wrap_const_vec(y, count, &y_view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_covariance_sample(&x_view, &y_view, out);
}

lmmc_status_t lmmc_lsr_stats_corr(const lmmc_real_t* x,
                                  const lmmc_real_t* y,
                                  size_t count,
                                  lmmc_real_t* out)
{
    lmmc_vec_t x_view;
    lmmc_vec_t y_view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(x, count, &x_view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_wrap_const_vec(y, count, &y_view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_correlation_sample(&x_view, &y_view, out);
}

lmmc_status_t lmmc_lsr_random_seed(lmmc_rng_t* rng, uint64_t seed)
{
    return lmmc_rng_seed(rng, seed);
}

lmmc_status_t lmmc_lsr_random_rand(lmmc_rng_t* rng, lmmc_real_t* out)
{
    return lmmc_rng_uniform(rng, (lmmc_real_t)0, (lmmc_real_t)1, out);
}

lmmc_status_t lmmc_lsr_random_randint(lmmc_rng_t* rng, int64_t lo,
                                      int64_t hi, int64_t* out)
{
    return lmmc_rng_int_uniform(rng, lo, hi, out);
}

lmmc_status_t lmmc_lsr_random_normal(lmmc_rng_t* rng, lmmc_real_t mean,
                                     lmmc_real_t stddev, lmmc_real_t* out)
{
    return lmmc_rng_normal(rng, mean, stddev, out);
}

lmmc_status_t lmmc_lsr_random_choice(lmmc_rng_t* rng,
                                     const lmmc_real_t* values,
                                     size_t count,
                                     lmmc_real_t* out)
{
    int64_t index = 0;
    lmmc_status_t status;
    if (!values || !out || count == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (count > (size_t)INT64_MAX + 1u) return LMMC_STATUS_OUT_OF_RANGE;
    status = lmmc_rng_int_uniform(rng, 0, (int64_t)count - 1, &index);
    if (status != LMMC_STATUS_OK) return status;
    *out = values[index];
    return LMMC_STATUS_OK;
}

static int lmmc_lsr_mat_valid(const lmmc_mat_t* a)
{
    return a && a->data && a->rows > 0 && a->cols > 0 && a->stride >= a->cols;
}

static lmmc_status_t lmmc_lsr_copy_mat(const lmmc_mat_t* src, lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!lmmc_lsr_mat_valid(src) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_mat_create(src->rows, src->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_copy(src, out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

static lmmc_status_t lmmc_lsr_vec_to_column(const lmmc_vec_t* src,
                                            lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!src || !src->data || src->size == 0 || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_mat_create(src->size, 1, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < src->size; ++i) {
        LMMC_REAL_SET(&out->data[i * out->stride], &src->data[i]);
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_vec_to_diag(const lmmc_vec_t* src,
                                          size_t rows,
                                          size_t cols,
                                          lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!src || !src->data || src->size == 0 || rows == 0 || cols == 0 ||
        !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (src->size > rows || src->size > cols) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_mat_create(rows, cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < src->size; ++i) {
        LMMC_REAL_SET(&out->data[i * out->stride + i], &src->data[i]);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_shape(const lmmc_mat_t* a,
                                    size_t* out_rows,
                                    size_t* out_cols)
{
    if (!lmmc_lsr_mat_valid(a) || !out_rows || !out_cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out_rows = a->rows;
    *out_cols = a->cols;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_transpose(const lmmc_mat_t* a,
                                        lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_mat_create(a->cols, a->rows, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_transpose_to(a, out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_adjoint(const lmmc_mat_t* a,
                                      lmmc_mat_t* out)
{
    return lmmc_lsr_linalg_transpose(a, out);
}

lmmc_status_t lmmc_lsr_linalg_det(const lmmc_mat_t* a,
                                  lmmc_real_t* out)
{
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_mat_det(a, out);
}

lmmc_status_t lmmc_lsr_linalg_inv(const lmmc_mat_t* a,
                                  lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_inv(a, out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_rank(const lmmc_mat_t* a,
                                   size_t* out_rank)
{
    size_t rows, cols, count, rank, col;
    lmmc_real_t* work;
    lmmc_real_t scale = (lmmc_real_t)0;
    lmmc_real_t tol;

    if (!lmmc_lsr_mat_valid(a) || !out_rank) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != 0 && a->cols > ((size_t)-1) / a->rows) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    rows = a->rows;
    cols = a->cols;
    count = rows * cols;
    work = (lmmc_real_t*)malloc(count * sizeof(lmmc_real_t));
    if (!work) return LMMC_STATUS_ALLOCATION_FAILED;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            lmmc_real_t value = a->data[i * a->stride + j];
            work[i * cols + j] = value;
            if (fabs((double)value) > fabs((double)scale)) scale = value;
        }
    }

    if (scale == (lmmc_real_t)0) {
        free(work);
        *out_rank = 0;
        return LMMC_STATUS_OK;
    }
    tol = (lmmc_real_t)(1e-12 * fabs((double)scale) *
                        (double)(rows > cols ? rows : cols));

    rank = 0;
    for (col = 0; col < cols && rank < rows; ++col) {
        size_t pivot = rank;
        lmmc_real_t pivot_abs =
            (lmmc_real_t)fabs((double)work[pivot * cols + col]);
        for (size_t row = rank + 1; row < rows; ++row) {
            lmmc_real_t candidate =
                (lmmc_real_t)fabs((double)work[row * cols + col]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= tol) continue;

        if (pivot != rank) {
            for (size_t j = col; j < cols; ++j) {
                lmmc_real_t tmp = work[rank * cols + j];
                work[rank * cols + j] = work[pivot * cols + j];
                work[pivot * cols + j] = tmp;
            }
        }

        for (size_t row = rank + 1; row < rows; ++row) {
            lmmc_real_t factor = work[row * cols + col] /
                                 work[rank * cols + col];
            work[row * cols + col] = (lmmc_real_t)0;
            for (size_t j = col + 1; j < cols; ++j) {
                work[row * cols + j] -= factor * work[rank * cols + j];
            }
        }
        ++rank;
    }

    free(work);
    *out_rank = rank;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_trace(const lmmc_mat_t* a,
                                    lmmc_real_t* out)
{
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_mat_trace(a, out);
}

lmmc_status_t lmmc_lsr_linalg_solve_left(const lmmc_mat_t* a,
                                         const lmmc_mat_t* b,
                                         lmmc_mat_t* out)
{
    lmmc_mat_t lu = {0};
    lmmc_vec_t rhs = {0};
    lmmc_vec_t sol = {0};
    size_t* pivots = NULL;
    lmmc_status_t status;

    if (!lmmc_lsr_mat_valid(a) || !lmmc_lsr_mat_valid(b) || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols || b->rows != a->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    status = lmmc_mat_create(a->rows, a->cols, &lu);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_copy(a, &lu);
    if (status != LMMC_STATUS_OK) goto cleanup;

    pivots = (size_t*)malloc(a->rows * sizeof(size_t));
    if (!pivots) {
        status = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    status = lmmc_lu_decompose_inplace(&lu, pivots, NULL);
    if (status != LMMC_STATUS_OK) goto cleanup;

    status = lmmc_mat_create(a->cols, b->cols, out);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_vec_create(b->rows, &rhs);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(out);
        goto cleanup;
    }
    status = lmmc_vec_create(a->cols, &sol);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(out);
        goto cleanup;
    }

    for (size_t col = 0; col < b->cols; ++col) {
        for (size_t row = 0; row < b->rows; ++row) {
            rhs.data[row] = b->data[row * b->stride + col];
        }
        status = lmmc_lu_solve(&lu, pivots, &rhs, &sol);
        if (status != LMMC_STATUS_OK) {
            lmmc_mat_destroy(out);
            goto cleanup;
        }
        for (size_t row = 0; row < a->cols; ++row) {
            out->data[row * out->stride + col] = sol.data[row];
        }
    }

cleanup:
    lmmc_vec_destroy(&sol);
    lmmc_vec_destroy(&rhs);
    if (pivots) free(pivots);
    lmmc_mat_destroy(&lu);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_solve_right(const lmmc_mat_t* b,
                                          const lmmc_mat_t* a,
                                          lmmc_mat_t* out)
{
    lmmc_mat_t at = {0};
    lmmc_mat_t bt = {0};
    lmmc_mat_t xt = {0};
    lmmc_status_t status;

    if (!lmmc_lsr_mat_valid(a) || !lmmc_lsr_mat_valid(b) || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols || b->cols != a->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    status = lmmc_lsr_linalg_transpose(a, &at);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_linalg_transpose(b, &bt);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_linalg_solve_left(&at, &bt, &xt);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_linalg_transpose(&xt, out);

cleanup:
    lmmc_mat_destroy(&xt);
    lmmc_mat_destroy(&bt);
    lmmc_mat_destroy(&at);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_eig(const lmmc_mat_t* a,
                                  lmmc_eigen_gen_full_result_t* out)
{
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_eigen_general_full(a, out);
}

lmmc_status_t lmmc_lsr_linalg_svd(const lmmc_mat_t* a,
                                  lmmc_svd_result_t* out)
{
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_svd(a, out);
}

lmmc_status_t lmmc_lsr_linalg_eig_table(const lmmc_mat_t* a,
                                        lmmc_lsr_eig_table_t* out)
{
    lmmc_status_t status;
    lmmc_eigen_gen_full_result_t raw = {0};
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    status = lmmc_lsr_linalg_eig(a, &raw);
    if (status != LMMC_STATUS_OK) return status;

    status = lmmc_lsr_vec_to_column(&raw.real_parts, &out->values_real);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_vec_to_column(&raw.imag_parts, &out->values_imag);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_copy_mat(&raw.vectors_real, &out->vectors_real);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_copy_mat(&raw.vectors_imag, &out->vectors_imag);
    if (status != LMMC_STATUS_OK) goto cleanup;

cleanup:
    lmmc_eigen_gen_full_result_destroy(&raw);
    if (status != LMMC_STATUS_OK) lmmc_lsr_eig_table_destroy(out);
    return status;
}

const lmmc_mat_t* lmmc_lsr_eig_table_get(const lmmc_lsr_eig_table_t* table,
                                         const char* key)
{
    if (!table || !key) return NULL;
    if (strcmp(key, "values_real") == 0) return &table->values_real;
    if (strcmp(key, "values_imag") == 0) return &table->values_imag;
    if (strcmp(key, "vectors_real") == 0) return &table->vectors_real;
    if (strcmp(key, "vectors_imag") == 0) return &table->vectors_imag;
    return NULL;
}

void lmmc_lsr_eig_table_destroy(lmmc_lsr_eig_table_t* table)
{
    if (!table) return;
    lmmc_mat_destroy(&table->values_real);
    lmmc_mat_destroy(&table->values_imag);
    lmmc_mat_destroy(&table->vectors_real);
    lmmc_mat_destroy(&table->vectors_imag);
}

lmmc_status_t lmmc_lsr_linalg_svd_table(const lmmc_mat_t* a,
                                        lmmc_lsr_svd_table_t* out)
{
    lmmc_status_t status;
    lmmc_svd_result_t raw = {0};
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    status = lmmc_lsr_linalg_svd(a, &raw);
    if (status != LMMC_STATUS_OK) return status;

    status = lmmc_lsr_copy_mat(&raw.U, &out->U);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_vec_to_diag(&raw.sigma, raw.U.cols, raw.Vt.rows, &out->S);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_copy_mat(&raw.Vt, &out->Vt);
    if (status != LMMC_STATUS_OK) goto cleanup;

cleanup:
    lmmc_svd_result_destroy(&raw);
    if (status != LMMC_STATUS_OK) lmmc_lsr_svd_table_destroy(out);
    return status;
}

const lmmc_mat_t* lmmc_lsr_svd_table_get(const lmmc_lsr_svd_table_t* table,
                                         const char* key)
{
    if (!table || !key) return NULL;
    if (strcmp(key, "U") == 0) return &table->U;
    if (strcmp(key, "S") == 0) return &table->S;
    if (strcmp(key, "Vt") == 0) return &table->Vt;
    return NULL;
}

void lmmc_lsr_svd_table_destroy(lmmc_lsr_svd_table_t* table)
{
    if (!table) return;
    lmmc_mat_destroy(&table->U);
    lmmc_mat_destroy(&table->S);
    lmmc_mat_destroy(&table->Vt);
}
