#include "lmmc/lsr_stdlib.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lsr_stdlib_internal.h"

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
    int dims[9];
} lmmc_lsr_unit_sig_t;

enum {
    LMMC_LSR_MAX_UNIT_EXPONENT = 32,
    LMMC_LSR_UNIT_DIM_COUNT = 9
};

typedef struct {
    const char* name;
    double scale;
    int dims[9];
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
    {"score", 1.0, {0, 0, 0, 0, 0, 0, 0, 1, 0}},
    {"token", 1.0, {0, 0, 0, 0, 0, 0, 0, 0, 1}},
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
    for (size_t i = 0; i < LMMC_LSR_UNIT_DIM_COUNT; ++i) sig->dims[i] = 0;
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
        int digit = **cursor - '0';
        have_digit = 1;
        if (value > (LMMC_LSR_MAX_UNIT_EXPONENT - digit) / 10) {
            return 0;
        }
        value = value * 10 + digit;
        ++(*cursor);
    }
    if (!have_digit) return 0;
    *out = sign * value;
    if (*out < -LMMC_LSR_MAX_UNIT_EXPONENT ||
        *out > LMMC_LSR_MAX_UNIT_EXPONENT) {
        return 0;
    }
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
        if (!isfinite(out->scale)) return 0;
        for (size_t i = 0; i < LMMC_LSR_UNIT_DIM_COUNT; ++i) {
            out->dims[i] += unit->dims[i] * exponent;
            if (out->dims[i] < -LMMC_LSR_MAX_UNIT_EXPONENT ||
                out->dims[i] > LMMC_LSR_MAX_UNIT_EXPONENT) {
                return 0;
            }
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
        if (*cursor == '\0') return 0;
    }
    return 1;
}

static int lmmc_lsr_same_dimension(const lmmc_lsr_unit_sig_t* lhs,
                                   const lmmc_lsr_unit_sig_t* rhs)
{
    for (size_t i = 0; i < LMMC_LSR_UNIT_DIM_COUNT; ++i) {
        if (lhs->dims[i] != rhs->dims[i]) return 0;
    }
    return 1;
}

static int lmmc_lsr_dimensionless_sig(const lmmc_lsr_unit_sig_t* sig)
{
    for (size_t i = 0; i < LMMC_LSR_UNIT_DIM_COUNT; ++i) {
        if (sig->dims[i] != 0) return 0;
    }
    return 1;
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

lmmc_status_t lmmc_lsr_constants_entry(size_t index,
                                       const char** out_name,
                                       lmmc_real_t* out_value,
                                       const char** out_unit)
{
    if (!out_name || !out_value || !out_unit) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (index >= lmmc_lsr_constants_length()) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out_name = lmmc_lsr_constants[index].name;
    *out_value = lmmc_lsr_constants[index].value;
    *out_unit = lmmc_lsr_constants[index].unit;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_I(lmmc_complex_t* out)
{
    return lmmc_complex_create((lmmc_real_t)0, (lmmc_real_t)1, out);
}
lmmc_status_t lmmc_lsr_units_convert(lmmc_real_t x,
                                      const char* from_unit,
                                      const char* to_unit,
                                      lmmc_real_t* out)
{
    lmmc_lsr_unit_sig_t from_sig;
    lmmc_lsr_unit_sig_t to_sig;
    double converted;
    if (!out || !from_unit || !to_unit) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    if (!lmmc_lsr_parse_unit_expr(from_unit, &from_sig) ||
        !lmmc_lsr_parse_unit_expr(to_unit, &to_sig)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_lsr_same_dimension(&from_sig, &to_sig)) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    converted = (double)x * from_sig.scale / to_sig.scale;
    return lmmc_lsr_store_finite_real((lmmc_real_t)converted, out);
}

lmmc_status_t lmmc_lsr_units_convert_from_si(lmmc_real_t x,
                                             const char* to_unit,
                                             lmmc_real_t* out)
{
    lmmc_lsr_unit_sig_t to_sig;
    double converted;
    if (!out || !to_unit) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    if (!lmmc_lsr_parse_unit_expr(to_unit, &to_sig)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    converted = (double)x / to_sig.scale;
    return lmmc_lsr_store_finite_real((lmmc_real_t)converted, out);
}

lmmc_status_t lmmc_lsr_units_convert_num(lmmc_real_t x,
                                         const char* to_unit,
                                         lmmc_real_t* out)
{
    return lmmc_lsr_units_convert_from_si(x, to_unit, out);
}

lmmc_status_t lmmc_lsr_units_strip(lmmc_real_t x, lmmc_real_t* out)
{
    lmmc_status_t status = lmmc_lsr_store_finite_real(x, out);
    return status == LMMC_STATUS_NUMERICAL_FAILURE
               ? LMMC_STATUS_UNIT_STRIP_OVERFLOW
               : status;
}

lmmc_status_t lmmc_lsr_units_strip_num(lmmc_real_t x,
                                       const char* unit,
                                       lmmc_real_t* out)
{
    lmmc_lsr_unit_sig_t sig;
    double value;
    if (!unit || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (strstr(unit, "num<") != NULL || strstr(unit, "scalar<") != NULL) {
        return LMMC_STATUS_UNIT_STRIP_LEGACY_SYNTAX;
    }
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_UNIT_STRIP_OVERFLOW;
    if (!lmmc_lsr_parse_unit_expr(unit, &sig)) {
        return LMMC_STATUS_UNIT_STRIP_INVALID;
    }
    value = (double)x * sig.scale;
    {
        lmmc_status_t status =
            lmmc_lsr_store_finite_real((lmmc_real_t)value, out);
        return status == LMMC_STATUS_NUMERICAL_FAILURE
                   ? LMMC_STATUS_UNIT_STRIP_OVERFLOW
                   : status;
    }
}

lmmc_status_t lmmc_lsr_units_strip_scalar(lmmc_real_t x, lmmc_real_t* out)
{
    lmmc_status_t status = lmmc_lsr_store_finite_real(x, out);
    return status == LMMC_STATUS_NUMERICAL_FAILURE
               ? LMMC_STATUS_UNIT_STRIP_OVERFLOW
               : status;
}

lmmc_status_t lmmc_lsr_units_is_dimensionless_num(lmmc_real_t x, int* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(x)) return LMMC_STATUS_NUMERICAL_FAILURE;
    *out = 1;
    return LMMC_STATUS_OK;
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
