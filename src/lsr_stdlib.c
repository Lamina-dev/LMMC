#include "lmmc/lsr_stdlib.h"

#include <math.h>
#include <stdlib.h>

#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/linear_algebra.h"
#include "lmmc/numeric.h"
#include "lmmc/random.h"
#include "lmmc/stats.h"
#include "memory_bridge.h"

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

static int lmmc_lsr_real_is_finite(lmmc_real_t value)
{
    return isfinite((double)value);
}

static int lmmc_lsr_real_array_is_finite(const lmmc_real_t* values,
                                         size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (!lmmc_lsr_real_is_finite(values[i])) return 0;
    }
    return 1;
}

static int lmmc_lsr_mul_overflow_size(size_t a, size_t b, size_t* out)
{
    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }
    if (a > ((size_t)-1) / b) return 1;
    *out = a * b;
    return 0;
}

static int lmmc_lsr_complex_is_finite(const lmmc_complex_t* z)
{
    return z && lmmc_lsr_real_is_finite(z->real) &&
           lmmc_lsr_real_is_finite(z->imag);
}

static uint64_t lmmc_lsr_hash_mix64(uint64_t value)
{
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value;
}

static uint64_t lmmc_lsr_real_hash_bits(lmmc_real_t value)
{
    uint64_t bits = 0;
    if (value == (lmmc_real_t)0) value = (lmmc_real_t)0;
    memcpy(&bits, &value, sizeof(value));
    return lmmc_lsr_hash_mix64(bits);
}

static int lmmc_lsr_bool_is_valid(int value)
{
    return value == 0 || value == 1;
}

static uint64_t lmmc_lsr_text_hash_bytes(const char* value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    const unsigned char* cursor = (const unsigned char*)value;
    while (*cursor) {
        hash ^= (uint64_t)(*cursor);
        hash *= UINT64_C(1099511628211);
        ++cursor;
    }
    return lmmc_lsr_hash_mix64(hash);
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

lmmc_status_t lmmc_lsr_math_i(lmmc_complex_t* out)
{
    return lmmc_complex_create((lmmc_real_t)0, (lmmc_real_t)1, out);
}

lmmc_status_t lmmc_lsr_math_I(lmmc_complex_t* out)
{
    return lmmc_lsr_math_i(out);
}

lmmc_status_t lmmc_lsr_num_equal(lmmc_real_t lhs, lmmc_real_t rhs, int* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(lhs) || !lmmc_lsr_real_is_finite(rhs)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = lhs == rhs ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_num_hash(lmmc_real_t value, uint64_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(value)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = lmmc_lsr_real_hash_bits(value);
    return LMMC_STATUS_OK;
}

static lmmc_real_t lmmc_lsr_num_set_key(lmmc_real_t value)
{
    return value == (lmmc_real_t)0 ? (lmmc_real_t)0 : value;
}

static int lmmc_lsr_num_set_valid(const lmmc_lsr_num_set_t* set)
{
    return set && (set->size == 0 || set->data);
}

static lmmc_status_t lmmc_lsr_num_set_require_finite(
    const lmmc_lsr_num_set_t* set)
{
    if (!lmmc_lsr_num_set_valid(set)) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_array_is_finite(set->data, set->size)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    return LMMC_STATUS_OK;
}

static int lmmc_lsr_num_set_index_of(const lmmc_lsr_num_set_t* set,
                                     lmmc_real_t value)
{
    lmmc_real_t key = lmmc_lsr_num_set_key(value);
    for (size_t i = 0; i < set->size; ++i) {
        if (lmmc_lsr_num_set_key(set->data[i]) == key) return (int)i;
    }
    return -1;
}

static lmmc_status_t lmmc_lsr_num_set_alloc(size_t capacity,
                                            lmmc_lsr_num_set_t* out)
{
    size_t bytes = 0;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    out->size = 0;
    out->data = NULL;
    out->owns_data = 0;
    if (capacity == 0) return LMMC_STATUS_OK;
    if (lmmc_lsr_mul_overflow_size(capacity, sizeof(lmmc_real_t), &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out->data = (lmmc_real_t*)lmmc_alloc(bytes);
    if (!out->data) return LMMC_STATUS_ALLOCATION_FAILED;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_num_set_append_unique(
    lmmc_lsr_num_set_t* set,
    lmmc_real_t value)
{
    if (!set || !lmmc_lsr_real_is_finite(value)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (lmmc_lsr_num_set_index_of(set, value) >= 0) return LMMC_STATUS_OK;
    set->data[set->size++] = lmmc_lsr_num_set_key(value);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_num_set_make(const lmmc_real_t* values,
                                    size_t count,
                                    lmmc_lsr_num_set_t* out)
{
    lmmc_status_t status;
    if (!out || (count > 0 && !values)) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_array_is_finite(values, count)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_num_set_alloc(count, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < count; ++i) {
        status = lmmc_lsr_num_set_append_unique(out, values[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_lsr_num_set_destroy(out);
            return status;
        }
    }
    return LMMC_STATUS_OK;
}

void lmmc_lsr_num_set_destroy(lmmc_lsr_num_set_t* set)
{
    if (!set) return;
    if (set->owns_data && set->data) lmmc_free(set->data);
    set->size = 0;
    set->data = NULL;
    set->owns_data = 0;
}

lmmc_status_t lmmc_lsr_num_set_contains(const lmmc_lsr_num_set_t* set,
                                        lmmc_real_t value,
                                        int* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(value)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_num_set_require_finite(set);
    if (status != LMMC_STATUS_OK) return status;
    *out = lmmc_lsr_num_set_index_of(set, value) >= 0 ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_num_set_subset(const lmmc_lsr_num_set_t* lhs,
                                      const lmmc_lsr_num_set_t* rhs,
                                      int* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_num_set_require_finite(lhs);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_num_set_require_finite(rhs);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_num_set_index_of(rhs, lhs->data[i]) < 0) {
            *out = 0;
            return LMMC_STATUS_OK;
        }
    }
    *out = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_num_set_binary_alloc(
    const lmmc_lsr_num_set_t* lhs,
    const lmmc_lsr_num_set_t* rhs,
    lmmc_lsr_num_set_t* out,
    size_t* capacity)
{
    lmmc_status_t status;
    if (!out || (const void*)out == (const void*)lhs ||
        (const void*)out == (const void*)rhs || !capacity) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_lsr_num_set_require_finite(lhs);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_num_set_require_finite(rhs);
    if (status != LMMC_STATUS_OK) return status;
    if (lhs->size > ((size_t)-1) - rhs->size) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *capacity = lhs->size + rhs->size;
    return lmmc_lsr_num_set_alloc(*capacity, out);
}

lmmc_status_t lmmc_lsr_num_set_union(const lmmc_lsr_num_set_t* lhs,
                                     const lmmc_lsr_num_set_t* rhs,
                                     lmmc_lsr_num_set_t* out)
{
    size_t capacity = 0;
    lmmc_status_t status = lmmc_lsr_num_set_binary_alloc(lhs, rhs, out,
                                                         &capacity);
    (void)capacity;
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        status = lmmc_lsr_num_set_append_unique(out, lhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        status = lmmc_lsr_num_set_append_unique(out, rhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_num_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_num_set_intersection(const lmmc_lsr_num_set_t* lhs,
                                            const lmmc_lsr_num_set_t* rhs,
                                            lmmc_lsr_num_set_t* out)
{
    size_t capacity = 0;
    lmmc_status_t status = lmmc_lsr_num_set_binary_alloc(lhs, rhs, out,
                                                         &capacity);
    (void)capacity;
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_num_set_index_of(rhs, lhs->data[i]) >= 0) {
            status = lmmc_lsr_num_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) {
                lmmc_lsr_num_set_destroy(out);
                return status;
            }
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_num_set_difference(const lmmc_lsr_num_set_t* lhs,
                                          const lmmc_lsr_num_set_t* rhs,
                                          lmmc_lsr_num_set_t* out)
{
    size_t capacity = 0;
    lmmc_status_t status = lmmc_lsr_num_set_binary_alloc(lhs, rhs, out,
                                                         &capacity);
    (void)capacity;
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_num_set_index_of(rhs, lhs->data[i]) < 0) {
            status = lmmc_lsr_num_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) {
                lmmc_lsr_num_set_destroy(out);
                return status;
            }
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_num_set_symmetric_difference(
    const lmmc_lsr_num_set_t* lhs,
    const lmmc_lsr_num_set_t* rhs,
    lmmc_lsr_num_set_t* out)
{
    size_t capacity = 0;
    lmmc_status_t status = lmmc_lsr_num_set_binary_alloc(lhs, rhs, out,
                                                         &capacity);
    (void)capacity;
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_num_set_index_of(rhs, lhs->data[i]) < 0) {
            status = lmmc_lsr_num_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        if (lmmc_lsr_num_set_index_of(lhs, rhs->data[i]) < 0) {
            status = lmmc_lsr_num_set_append_unique(out, rhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_num_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_bool_equal(int lhs, int rhs, int* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_bool_is_valid(lhs) || !lmmc_lsr_bool_is_valid(rhs)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out = lhs == rhs ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_bool_hash(int value, uint64_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_bool_is_valid(value)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out = lmmc_lsr_hash_mix64(value ? UINT64_C(1) : UINT64_C(0));
    return LMMC_STATUS_OK;
}

static int lmmc_lsr_bool_set_valid(const lmmc_lsr_bool_set_t* set)
{
    return set && set->size <= 2 && (set->size == 0 || set->data);
}

static lmmc_status_t lmmc_lsr_bool_set_require_valid(
    const lmmc_lsr_bool_set_t* set)
{
    if (!lmmc_lsr_bool_set_valid(set)) return LMMC_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < set->size; ++i) {
        if (!lmmc_lsr_bool_is_valid(set->data[i])) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    return LMMC_STATUS_OK;
}

static int lmmc_lsr_bool_set_index_of(const lmmc_lsr_bool_set_t* set,
                                      int value)
{
    uint8_t key = value ? 1u : 0u;
    for (size_t i = 0; i < set->size; ++i) {
        if (set->data[i] == key) return (int)i;
    }
    return -1;
}

static lmmc_status_t lmmc_lsr_bool_set_alloc(size_t capacity,
                                             lmmc_lsr_bool_set_t* out)
{
    if (!out || capacity > 2) return LMMC_STATUS_INVALID_ARGUMENT;
    out->size = 0;
    out->data = NULL;
    out->owns_data = 0;
    if (capacity == 0) return LMMC_STATUS_OK;
    out->data = (uint8_t*)lmmc_alloc(capacity * sizeof(uint8_t));
    if (!out->data) return LMMC_STATUS_ALLOCATION_FAILED;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_bool_set_append_unique(
    lmmc_lsr_bool_set_t* set,
    int value)
{
    if (!set || !lmmc_lsr_bool_is_valid(value)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_lsr_bool_set_index_of(set, value) >= 0) return LMMC_STATUS_OK;
    if (set->size >= 2) return LMMC_STATUS_INVALID_ARGUMENT;
    set->data[set->size++] = value ? 1u : 0u;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_bool_set_make(const int* values,
                                     size_t count,
                                     lmmc_lsr_bool_set_t* out)
{
    lmmc_status_t status;
    if (!out || (count > 0 && !values)) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_bool_set_alloc(count > 2 ? 2 : count, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < count; ++i) {
        status = lmmc_lsr_bool_set_append_unique(out, values[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_lsr_bool_set_destroy(out);
            return status;
        }
    }
    return LMMC_STATUS_OK;
}

void lmmc_lsr_bool_set_destroy(lmmc_lsr_bool_set_t* set)
{
    if (!set) return;
    if (set->owns_data && set->data) lmmc_free(set->data);
    set->size = 0;
    set->data = NULL;
    set->owns_data = 0;
}

lmmc_status_t lmmc_lsr_bool_set_contains(const lmmc_lsr_bool_set_t* set,
                                         int value,
                                         int* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_bool_is_valid(value)) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_bool_set_require_valid(set);
    if (status != LMMC_STATUS_OK) return status;
    *out = lmmc_lsr_bool_set_index_of(set, value) >= 0 ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_bool_set_subset(const lmmc_lsr_bool_set_t* lhs,
                                       const lmmc_lsr_bool_set_t* rhs,
                                       int* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_bool_set_require_valid(lhs);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_bool_set_require_valid(rhs);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_bool_set_index_of(rhs, lhs->data[i]) < 0) {
            *out = 0;
            return LMMC_STATUS_OK;
        }
    }
    *out = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_bool_set_binary_alloc(
    const lmmc_lsr_bool_set_t* lhs,
    const lmmc_lsr_bool_set_t* rhs,
    lmmc_lsr_bool_set_t* out)
{
    lmmc_status_t status;
    if (!out || (const void*)out == (const void*)lhs ||
        (const void*)out == (const void*)rhs) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_lsr_bool_set_require_valid(lhs);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_bool_set_require_valid(rhs);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_bool_set_alloc(2, out);
}

lmmc_status_t lmmc_lsr_bool_set_union(const lmmc_lsr_bool_set_t* lhs,
                                      const lmmc_lsr_bool_set_t* rhs,
                                      lmmc_lsr_bool_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_bool_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        status = lmmc_lsr_bool_set_append_unique(out, lhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        status = lmmc_lsr_bool_set_append_unique(out, rhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_bool_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_bool_set_intersection(const lmmc_lsr_bool_set_t* lhs,
                                             const lmmc_lsr_bool_set_t* rhs,
                                             lmmc_lsr_bool_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_bool_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_bool_set_index_of(rhs, lhs->data[i]) >= 0) {
            status = lmmc_lsr_bool_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_bool_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_bool_set_difference(const lmmc_lsr_bool_set_t* lhs,
                                           const lmmc_lsr_bool_set_t* rhs,
                                           lmmc_lsr_bool_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_bool_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_bool_set_index_of(rhs, lhs->data[i]) < 0) {
            status = lmmc_lsr_bool_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_bool_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_bool_set_symmetric_difference(
    const lmmc_lsr_bool_set_t* lhs,
    const lmmc_lsr_bool_set_t* rhs,
    lmmc_lsr_bool_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_bool_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_bool_set_index_of(rhs, lhs->data[i]) < 0) {
            status = lmmc_lsr_bool_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        if (lmmc_lsr_bool_set_index_of(lhs, rhs->data[i]) < 0) {
            status = lmmc_lsr_bool_set_append_unique(out, rhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_bool_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_text_equal(const char* lhs, const char* rhs, int* out)
{
    if (!lhs || !rhs || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = strcmp(lhs, rhs) == 0 ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_text_hash(const char* value, uint64_t* out)
{
    if (!value || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = lmmc_lsr_text_hash_bytes(value);
    return LMMC_STATUS_OK;
}

static char* lmmc_lsr_text_copy(const char* value)
{
    size_t length = strlen(value);
    char* copy = NULL;
    if (length == (size_t)-1) return NULL;
    copy = (char*)lmmc_alloc(length + 1);
    if (!copy) return NULL;
    memcpy(copy, value, length + 1);
    return copy;
}

static int lmmc_lsr_text_set_valid(const lmmc_lsr_text_set_t* set)
{
    if (!set) return 0;
    if (set->size == 0) return 1;
    if (!set->data) return 0;
    for (size_t i = 0; i < set->size; ++i) {
        if (!set->data[i]) return 0;
    }
    return 1;
}

static int lmmc_lsr_text_set_index_of(const lmmc_lsr_text_set_t* set,
                                      const char* value)
{
    for (size_t i = 0; i < set->size; ++i) {
        if (strcmp(set->data[i], value) == 0) return (int)i;
    }
    return -1;
}

static lmmc_status_t lmmc_lsr_text_set_alloc(size_t capacity,
                                             lmmc_lsr_text_set_t* out)
{
    size_t bytes = 0;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    out->size = 0;
    out->data = NULL;
    out->owns_data = 0;
    if (capacity == 0) return LMMC_STATUS_OK;
    if (lmmc_lsr_mul_overflow_size(capacity, sizeof(char*), &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out->data = (char**)lmmc_alloc(bytes);
    if (!out->data) return LMMC_STATUS_ALLOCATION_FAILED;
    memset(out->data, 0, bytes);
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_text_set_append_unique(
    lmmc_lsr_text_set_t* set,
    const char* value)
{
    char* copy = NULL;
    if (!set || !value) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lmmc_lsr_text_set_index_of(set, value) >= 0) return LMMC_STATUS_OK;
    copy = lmmc_lsr_text_copy(value);
    if (!copy) return LMMC_STATUS_ALLOCATION_FAILED;
    set->data[set->size++] = copy;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_text_set_make(const char* const* values,
                                     size_t count,
                                     lmmc_lsr_text_set_t* out)
{
    lmmc_status_t status;
    if (!out || (count > 0 && !values)) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_text_set_alloc(count, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < count; ++i) {
        status = lmmc_lsr_text_set_append_unique(out, values[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_lsr_text_set_destroy(out);
            return status;
        }
    }
    return LMMC_STATUS_OK;
}

void lmmc_lsr_text_set_destroy(lmmc_lsr_text_set_t* set)
{
    if (!set) return;
    if (set->owns_data && set->data) {
        for (size_t i = 0; i < set->size; ++i) {
            if (set->data[i]) lmmc_free(set->data[i]);
        }
        lmmc_free(set->data);
    }
    set->size = 0;
    set->data = NULL;
    set->owns_data = 0;
}

lmmc_status_t lmmc_lsr_text_set_contains(const lmmc_lsr_text_set_t* set,
                                         const char* value,
                                         int* out)
{
    if (!value || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_text_set_valid(set)) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = lmmc_lsr_text_set_index_of(set, value) >= 0 ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_text_set_subset(const lmmc_lsr_text_set_t* lhs,
                                       const lmmc_lsr_text_set_t* rhs,
                                       int* out)
{
    if (!out || !lmmc_lsr_text_set_valid(lhs) ||
        !lmmc_lsr_text_set_valid(rhs)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_text_set_index_of(rhs, lhs->data[i]) < 0) {
            *out = 0;
            return LMMC_STATUS_OK;
        }
    }
    *out = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_text_set_binary_alloc(
    const lmmc_lsr_text_set_t* lhs,
    const lmmc_lsr_text_set_t* rhs,
    lmmc_lsr_text_set_t* out)
{
    if (!out || (const void*)out == (const void*)lhs ||
        (const void*)out == (const void*)rhs ||
        !lmmc_lsr_text_set_valid(lhs) || !lmmc_lsr_text_set_valid(rhs)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lhs->size > ((size_t)-1) - rhs->size) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return lmmc_lsr_text_set_alloc(lhs->size + rhs->size, out);
}

lmmc_status_t lmmc_lsr_text_set_union(const lmmc_lsr_text_set_t* lhs,
                                      const lmmc_lsr_text_set_t* rhs,
                                      lmmc_lsr_text_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_text_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        status = lmmc_lsr_text_set_append_unique(out, lhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        status = lmmc_lsr_text_set_append_unique(out, rhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_text_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_text_set_intersection(const lmmc_lsr_text_set_t* lhs,
                                             const lmmc_lsr_text_set_t* rhs,
                                             lmmc_lsr_text_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_text_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_text_set_index_of(rhs, lhs->data[i]) >= 0) {
            status = lmmc_lsr_text_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_text_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_text_set_difference(const lmmc_lsr_text_set_t* lhs,
                                           const lmmc_lsr_text_set_t* rhs,
                                           lmmc_lsr_text_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_text_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_text_set_index_of(rhs, lhs->data[i]) < 0) {
            status = lmmc_lsr_text_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_text_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_text_set_symmetric_difference(
    const lmmc_lsr_text_set_t* lhs,
    const lmmc_lsr_text_set_t* rhs,
    lmmc_lsr_text_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_text_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_text_set_index_of(rhs, lhs->data[i]) < 0) {
            status = lmmc_lsr_text_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        if (lmmc_lsr_text_set_index_of(lhs, rhs->data[i]) < 0) {
            status = lmmc_lsr_text_set_append_unique(out, rhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_text_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_math_complex(lmmc_real_t real,
                                    lmmc_real_t imag,
                                    lmmc_complex_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(real) || !lmmc_lsr_real_is_finite(imag)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    return lmmc_complex_create(real, imag, out);
}

lmmc_status_t lmmc_lsr_math_real(const lmmc_complex_t* z, lmmc_real_t* out)
{
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_complex_is_finite(z)) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_store_finite_real(z->real, out);
}

lmmc_status_t lmmc_lsr_math_imag(const lmmc_complex_t* z, lmmc_real_t* out)
{
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_complex_is_finite(z)) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_store_finite_real(z->imag, out);
}

lmmc_status_t lmmc_lsr_math_conj(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    lmmc_status_t status;
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_complex_is_finite(z)) return LMMC_STATUS_NUMERICAL_FAILURE;
    status = lmmc_complex_conj(z, out);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_complex_is_finite(out) ? LMMC_STATUS_OK
                                           : LMMC_STATUS_NUMERICAL_FAILURE;
}

lmmc_status_t lmmc_lsr_math_complex_abs(const lmmc_complex_t* z,
                                        lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_complex_is_finite(z)) return LMMC_STATUS_NUMERICAL_FAILURE;
    status = lmmc_complex_modulus(z, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_math_complex_equal(const lmmc_complex_t* lhs,
                                          const lmmc_complex_t* rhs,
                                          int* out)
{
    if (!lhs || !rhs || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_complex_is_finite(lhs) ||
        !lmmc_lsr_complex_is_finite(rhs)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = (lhs->real == rhs->real && lhs->imag == rhs->imag) ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_complex_hash(const lmmc_complex_t* z,
                                         uint64_t* out)
{
    uint64_t real_hash;
    uint64_t imag_hash;
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_complex_is_finite(z)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    real_hash = lmmc_lsr_real_hash_bits(z->real);
    imag_hash = lmmc_lsr_real_hash_bits(z->imag);
    *out = lmmc_lsr_hash_mix64(real_hash ^
                               (imag_hash + UINT64_C(0x9e3779b97f4a7c15) +
                                (real_hash << 6) + (real_hash >> 2)));
    return LMMC_STATUS_OK;
}

static lmmc_complex_t lmmc_lsr_complex_set_key(const lmmc_complex_t* value)
{
    lmmc_complex_t key = *value;
    key.real = lmmc_lsr_num_set_key(key.real);
    key.imag = lmmc_lsr_num_set_key(key.imag);
    return key;
}

static int lmmc_lsr_complex_set_valid(const lmmc_lsr_complex_set_t* set)
{
    return set && (set->size == 0 || set->data);
}

static lmmc_status_t lmmc_lsr_complex_set_require_finite(
    const lmmc_lsr_complex_set_t* set)
{
    if (!lmmc_lsr_complex_set_valid(set)) return LMMC_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < set->size; ++i) {
        if (!lmmc_lsr_complex_is_finite(&set->data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}

static int lmmc_lsr_complex_values_equal(const lmmc_complex_t* lhs,
                                         const lmmc_complex_t* rhs)
{
    lmmc_complex_t left = lmmc_lsr_complex_set_key(lhs);
    lmmc_complex_t right = lmmc_lsr_complex_set_key(rhs);
    return left.real == right.real && left.imag == right.imag;
}

static int lmmc_lsr_complex_set_index_of(const lmmc_lsr_complex_set_t* set,
                                         const lmmc_complex_t* value)
{
    for (size_t i = 0; i < set->size; ++i) {
        if (lmmc_lsr_complex_values_equal(&set->data[i], value)) return (int)i;
    }
    return -1;
}

static lmmc_status_t lmmc_lsr_complex_set_alloc(
    size_t capacity,
    lmmc_lsr_complex_set_t* out)
{
    size_t bytes = 0;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    out->size = 0;
    out->data = NULL;
    out->owns_data = 0;
    if (capacity == 0) return LMMC_STATUS_OK;
    if (lmmc_lsr_mul_overflow_size(capacity, sizeof(lmmc_complex_t),
                                   &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out->data = (lmmc_complex_t*)lmmc_alloc(bytes);
    if (!out->data) return LMMC_STATUS_ALLOCATION_FAILED;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_complex_set_append_unique(
    lmmc_lsr_complex_set_t* set,
    const lmmc_complex_t* value)
{
    if (!set || !lmmc_lsr_complex_is_finite(value)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (lmmc_lsr_complex_set_index_of(set, value) >= 0) {
        return LMMC_STATUS_OK;
    }
    set->data[set->size++] = lmmc_lsr_complex_set_key(value);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_complex_set_make(const lmmc_complex_t* values,
                                        size_t count,
                                        lmmc_lsr_complex_set_t* out)
{
    lmmc_status_t status;
    if (!out || (count > 0 && !values)) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_complex_set_alloc(count, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < count; ++i) {
        status = lmmc_lsr_complex_set_append_unique(out, &values[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_lsr_complex_set_destroy(out);
            return status;
        }
    }
    return LMMC_STATUS_OK;
}

void lmmc_lsr_complex_set_destroy(lmmc_lsr_complex_set_t* set)
{
    if (!set) return;
    if (set->owns_data && set->data) lmmc_free(set->data);
    set->size = 0;
    set->data = NULL;
    set->owns_data = 0;
}

lmmc_status_t lmmc_lsr_complex_set_contains(
    const lmmc_lsr_complex_set_t* set,
    const lmmc_complex_t* value,
    int* out)
{
    lmmc_status_t status;
    if (!value || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_complex_is_finite(value)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_complex_set_require_finite(set);
    if (status != LMMC_STATUS_OK) return status;
    *out = lmmc_lsr_complex_set_index_of(set, value) >= 0 ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_complex_set_subset(
    const lmmc_lsr_complex_set_t* lhs,
    const lmmc_lsr_complex_set_t* rhs,
    int* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_complex_set_require_finite(lhs);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_complex_set_require_finite(rhs);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_complex_set_index_of(rhs, &lhs->data[i]) < 0) {
            *out = 0;
            return LMMC_STATUS_OK;
        }
    }
    *out = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_complex_set_binary_alloc(
    const lmmc_lsr_complex_set_t* lhs,
    const lmmc_lsr_complex_set_t* rhs,
    lmmc_lsr_complex_set_t* out)
{
    lmmc_status_t status;
    if (!out || (const void*)out == (const void*)lhs ||
        (const void*)out == (const void*)rhs) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_lsr_complex_set_require_finite(lhs);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_complex_set_require_finite(rhs);
    if (status != LMMC_STATUS_OK) return status;
    if (lhs->size > ((size_t)-1) - rhs->size) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return lmmc_lsr_complex_set_alloc(lhs->size + rhs->size, out);
}

lmmc_status_t lmmc_lsr_complex_set_union(
    const lmmc_lsr_complex_set_t* lhs,
    const lmmc_lsr_complex_set_t* rhs,
    lmmc_lsr_complex_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_complex_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        status = lmmc_lsr_complex_set_append_unique(out, &lhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        status = lmmc_lsr_complex_set_append_unique(out, &rhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_complex_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_complex_set_intersection(
    const lmmc_lsr_complex_set_t* lhs,
    const lmmc_lsr_complex_set_t* rhs,
    lmmc_lsr_complex_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_complex_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_complex_set_index_of(rhs, &lhs->data[i]) >= 0) {
            status = lmmc_lsr_complex_set_append_unique(out, &lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_complex_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_complex_set_difference(
    const lmmc_lsr_complex_set_t* lhs,
    const lmmc_lsr_complex_set_t* rhs,
    lmmc_lsr_complex_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_complex_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_complex_set_index_of(rhs, &lhs->data[i]) < 0) {
            status = lmmc_lsr_complex_set_append_unique(out, &lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_complex_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_complex_set_symmetric_difference(
    const lmmc_lsr_complex_set_t* lhs,
    const lmmc_lsr_complex_set_t* rhs,
    lmmc_lsr_complex_set_t* out)
{
    lmmc_status_t status = lmmc_lsr_complex_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_lsr_complex_set_index_of(rhs, &lhs->data[i]) < 0) {
            status = lmmc_lsr_complex_set_append_unique(out, &lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        if (lmmc_lsr_complex_set_index_of(lhs, &rhs->data[i]) < 0) {
            status = lmmc_lsr_complex_set_append_unique(out, &rhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_lsr_complex_set_destroy(out);
    return status;
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

static lmmc_status_t lmmc_lsr_wrap_const_vec(const lmmc_real_t* values,
                                             size_t count,
                                             lmmc_vec_t* out)
{
    if (!values || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (count == 0) return LMMC_STATUS_EMPTY_INPUT;
    if (!lmmc_lsr_real_array_is_finite(values, count)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    out->size = count;
    out->data = (lmmc_real_t*)values;
    out->owns_data = 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_stats_mean(const lmmc_real_t* values, size_t count,
                                  lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_mean(&view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_stats_median(const lmmc_real_t* values, size_t count,
                                    lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_median(&view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_stats_var(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_variance_sample(&view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_stats_std(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_stddev_sample(&view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_stats_quantile(const lmmc_real_t* values, size_t count,
                                      lmmc_real_t q, lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(q)) return LMMC_STATUS_NUMERICAL_FAILURE;
    if (q < (lmmc_real_t)0 || q > (lmmc_real_t)1) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_quantile(&view, q, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_stats_cov(const lmmc_real_t* x,
                                 const lmmc_real_t* y,
                                 size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t x_view;
    lmmc_vec_t y_view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_wrap_const_vec(x, count, &x_view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_wrap_const_vec(y, count, &y_view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_covariance_sample(&x_view, &y_view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_stats_corr(const lmmc_real_t* x,
                                  const lmmc_real_t* y,
                                  size_t count,
                                  lmmc_real_t* out)
{
    lmmc_vec_t x_view;
    lmmc_vec_t y_view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_wrap_const_vec(x, count, &x_view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_wrap_const_vec(y, count, &y_view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_correlation_sample(&x_view, &y_view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

typedef lmmc_status_t (*lmmc_lsr_dist2_fn)(lmmc_real_t, lmmc_real_t,
                                           lmmc_real_t*);
typedef lmmc_status_t (*lmmc_lsr_dist3_fn)(lmmc_real_t, lmmc_real_t,
                                           lmmc_real_t, lmmc_real_t*);

static lmmc_status_t lmmc_lsr_stats_call_dist2(lmmc_lsr_dist2_fn fn,
                                               lmmc_real_t a,
                                               lmmc_real_t b,
                                               lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!fn || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(a) || !lmmc_lsr_real_is_finite(b)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = fn(a, b, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

static lmmc_status_t lmmc_lsr_stats_call_dist3(lmmc_lsr_dist3_fn fn,
                                               lmmc_real_t a,
                                               lmmc_real_t b,
                                               lmmc_real_t c,
                                               lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!fn || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(a) || !lmmc_lsr_real_is_finite(b) ||
        !lmmc_lsr_real_is_finite(c)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = fn(a, b, c, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_stats_normal_pdf(lmmc_real_t x, lmmc_real_t mean,
                                        lmmc_real_t stddev,
                                        lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_normal_pdf, x, mean, stddev,
                                     out);
}

lmmc_status_t lmmc_lsr_stats_normal_cdf(lmmc_real_t x, lmmc_real_t mean,
                                        lmmc_real_t stddev,
                                        lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_normal_cdf, x, mean, stddev,
                                     out);
}

lmmc_status_t lmmc_lsr_stats_normal_quantile(lmmc_real_t p,
                                             lmmc_real_t mean,
                                             lmmc_real_t stddev,
                                             lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_normal_quantile, p, mean,
                                     stddev, out);
}

lmmc_status_t lmmc_lsr_stats_t_pdf(lmmc_real_t x, lmmc_real_t df,
                                   lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist2(lmmc_dist_t_pdf, x, df, out);
}

lmmc_status_t lmmc_lsr_stats_t_cdf(lmmc_real_t x, lmmc_real_t df,
                                   lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist2(lmmc_dist_t_cdf, x, df, out);
}

lmmc_status_t lmmc_lsr_stats_t_quantile(lmmc_real_t p, lmmc_real_t df,
                                        lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist2(lmmc_dist_t_quantile, p, df, out);
}

lmmc_status_t lmmc_lsr_stats_chi2_pdf(lmmc_real_t x, lmmc_real_t df,
                                      lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist2(lmmc_dist_chi2_pdf, x, df, out);
}

lmmc_status_t lmmc_lsr_stats_chi2_cdf(lmmc_real_t x, lmmc_real_t df,
                                      lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist2(lmmc_dist_chi2_cdf, x, df, out);
}

lmmc_status_t lmmc_lsr_stats_chi2_quantile(lmmc_real_t p, lmmc_real_t df,
                                           lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist2(lmmc_dist_chi2_quantile, p, df, out);
}

lmmc_status_t lmmc_lsr_stats_f_pdf(lmmc_real_t x, lmmc_real_t df1,
                                   lmmc_real_t df2, lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_f_pdf, x, df1, df2, out);
}

lmmc_status_t lmmc_lsr_stats_f_cdf(lmmc_real_t x, lmmc_real_t df1,
                                   lmmc_real_t df2, lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_f_cdf, x, df1, df2, out);
}

lmmc_status_t lmmc_lsr_stats_f_quantile(lmmc_real_t p, lmmc_real_t df1,
                                        lmmc_real_t df2,
                                        lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_f_quantile, p, df1, df2,
                                     out);
}

lmmc_status_t lmmc_lsr_stats_gamma_pdf(lmmc_real_t x, lmmc_real_t shape,
                                       lmmc_real_t scale,
                                       lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_gamma_pdf, x, shape, scale,
                                     out);
}

lmmc_status_t lmmc_lsr_stats_gamma_cdf(lmmc_real_t x, lmmc_real_t shape,
                                       lmmc_real_t scale,
                                       lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_gamma_cdf, x, shape, scale,
                                     out);
}

lmmc_status_t lmmc_lsr_stats_gamma_quantile(lmmc_real_t p,
                                            lmmc_real_t shape,
                                            lmmc_real_t scale,
                                            lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_gamma_quantile, p, shape,
                                     scale, out);
}

lmmc_status_t lmmc_lsr_stats_beta_pdf(lmmc_real_t x, lmmc_real_t alpha,
                                      lmmc_real_t beta,
                                      lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_beta_pdf, x, alpha, beta,
                                     out);
}

lmmc_status_t lmmc_lsr_stats_beta_cdf(lmmc_real_t x, lmmc_real_t alpha,
                                      lmmc_real_t beta,
                                      lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_beta_cdf, x, alpha, beta,
                                     out);
}

lmmc_status_t lmmc_lsr_stats_beta_quantile(lmmc_real_t p,
                                           lmmc_real_t alpha,
                                           lmmc_real_t beta,
                                           lmmc_real_t* out)
{
    return lmmc_lsr_stats_call_dist3(lmmc_dist_beta_quantile, p, alpha,
                                     beta, out);
}

lmmc_status_t lmmc_lsr_stats_binomial_pmf(size_t k, size_t n,
                                          lmmc_real_t p,
                                          lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(p)) return LMMC_STATUS_NUMERICAL_FAILURE;
    status = lmmc_dist_binomial_pmf(k, n, p, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_stats_binomial_cdf(size_t k, size_t n,
                                          lmmc_real_t p,
                                          lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(p)) return LMMC_STATUS_NUMERICAL_FAILURE;
    status = lmmc_dist_binomial_cdf(k, n, p, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_stats_poisson_pmf(size_t k, lmmc_real_t lambda,
                                         lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(lambda)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_dist_poisson_pmf(k, lambda, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_stats_poisson_cdf(size_t k, lmmc_real_t lambda,
                                         lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(lambda)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_dist_poisson_cdf(k, lambda, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_random_seed(lmmc_rng_t* rng, uint64_t seed)
{
    if (!rng) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_rng_seed(rng, seed);
}

lmmc_status_t lmmc_lsr_random_rand(lmmc_rng_t* rng, lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!rng || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_rng_uniform(rng, (lmmc_real_t)0, (lmmc_real_t)1, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_random_randint(lmmc_rng_t* rng, int64_t lo,
                                      int64_t hi, int64_t* out)
{
    if (!rng || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_rng_int_uniform(rng, lo, hi, out);
}

lmmc_status_t lmmc_lsr_random_normal(lmmc_rng_t* rng, lmmc_real_t mean,
                                     lmmc_real_t stddev, lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!rng || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(mean) ||
        !lmmc_lsr_real_is_finite(stddev)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_rng_normal(rng, mean, stddev, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_random_choice(lmmc_rng_t* rng,
                                     const lmmc_real_t* values,
                                     size_t count,
                                     lmmc_real_t* out)
{
    int64_t index = 0;
    lmmc_status_t status;
    if (!rng || !values || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (count == 0) return LMMC_STATUS_EMPTY_INPUT;
    if (count > (size_t)INT64_MAX + 1u) return LMMC_STATUS_OUT_OF_RANGE;
    if (!lmmc_lsr_real_array_is_finite(values, count)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_rng_int_uniform(rng, 0, (int64_t)count - 1, &index);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(values[index], out);
}

static lmmc_rng_t* lmmc_lsr_default_rng = NULL;

static lmmc_status_t lmmc_lsr_default_rng_get(lmmc_rng_t** out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_default_rng) {
        status = lmmc_rng_create(&lmmc_lsr_default_rng);
        if (status != LMMC_STATUS_OK) return status;
    }
    *out = lmmc_lsr_default_rng;
    return LMMC_STATUS_OK;
}

void lmmc_lsr_random_default_deinit(void)
{
    lmmc_rng_destroy(lmmc_lsr_default_rng);
    lmmc_lsr_default_rng = NULL;
}

lmmc_status_t lmmc_lsr_random_default_seed(uint64_t seed)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status = lmmc_lsr_default_rng_get(&rng);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_random_seed(rng, seed);
}

lmmc_status_t lmmc_lsr_random_default_rand(lmmc_real_t* out)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status = lmmc_lsr_default_rng_get(&rng);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_random_rand(rng, out);
}

lmmc_status_t lmmc_lsr_random_default_randint(int64_t lo,
                                              int64_t hi,
                                              int64_t* out)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status = lmmc_lsr_default_rng_get(&rng);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_random_randint(rng, lo, hi, out);
}

lmmc_status_t lmmc_lsr_random_default_normal(lmmc_real_t mean,
                                             lmmc_real_t stddev,
                                             lmmc_real_t* out)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status = lmmc_lsr_default_rng_get(&rng);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_random_normal(rng, mean, stddev, out);
}

lmmc_status_t lmmc_lsr_random_default_choice(const lmmc_real_t* values,
                                             size_t count,
                                             lmmc_real_t* out)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status = lmmc_lsr_default_rng_get(&rng);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_random_choice(rng, values, count, out);
}

static int lmmc_lsr_mat_valid(const lmmc_mat_t* a)
{
    return a && a->data && a->rows > 0 && a->cols > 0 && a->stride >= a->cols;
}

static int lmmc_lsr_mat_is_finite(const lmmc_mat_t* a)
{
    if (!lmmc_lsr_mat_valid(a)) return 0;
    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            if (!lmmc_lsr_real_is_finite(a->data[i * a->stride + j])) {
                return 0;
            }
        }
    }
    return 1;
}

static lmmc_status_t lmmc_lsr_require_finite_mat(const lmmc_mat_t* a)
{
    if (!lmmc_lsr_mat_valid(a)) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_lsr_mat_is_finite(a) ? LMMC_STATUS_OK
                                     : LMMC_STATUS_NUMERICAL_FAILURE;
}

static int lmmc_lsr_vec_is_finite(const lmmc_vec_t* v)
{
    return v && v->data && v->size > 0 &&
           lmmc_lsr_real_array_is_finite(v->data, v->size);
}

static lmmc_status_t lmmc_lsr_require_finite_vec(const lmmc_vec_t* v)
{
    if (!v || !v->data || v->size == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_lsr_vec_is_finite(v) ? LMMC_STATUS_OK
                                     : LMMC_STATUS_NUMERICAL_FAILURE;
}

static int lmmc_lsr_compare_eval(lmmc_real_t lhs,
                                 lmmc_lsr_compare_op_t op,
                                 lmmc_real_t rhs,
                                 uint8_t* out)
{
    if (!out) return 0;
    switch (op) {
    case LMMC_LSR_COMPARE_EQ:
        *out = (uint8_t)(lhs == rhs);
        return 1;
    case LMMC_LSR_COMPARE_NE:
        *out = (uint8_t)(lhs != rhs);
        return 1;
    case LMMC_LSR_COMPARE_LT:
        *out = (uint8_t)(lhs < rhs);
        return 1;
    case LMMC_LSR_COMPARE_LE:
        *out = (uint8_t)(lhs <= rhs);
        return 1;
    case LMMC_LSR_COMPARE_GT:
        *out = (uint8_t)(lhs > rhs);
        return 1;
    case LMMC_LSR_COMPARE_GE:
        *out = (uint8_t)(lhs >= rhs);
        return 1;
    }
    return 0;
}

static lmmc_status_t lmmc_lsr_bool_vec_create(size_t size,
                                              lmmc_lsr_bool_vec_t* out)
{
    size_t bytes = 0;
    if (!out || size == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lmmc_lsr_mul_overflow_size(size, sizeof(uint8_t), &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out->data = (uint8_t*)lmmc_alloc(bytes);
    if (!out->data) return LMMC_STATUS_ALLOCATION_FAILED;
    memset(out->data, 0, bytes);
    out->size = size;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_bool_mat_create(size_t rows,
                                              size_t cols,
                                              lmmc_lsr_bool_mat_t* out)
{
    size_t count = 0;
    size_t bytes = 0;
    if (!out || rows == 0 || cols == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lmmc_lsr_mul_overflow_size(rows, cols, &count) ||
        lmmc_lsr_mul_overflow_size(count, sizeof(uint8_t), &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out->data = (uint8_t*)lmmc_alloc(bytes);
    if (!out->data) return LMMC_STATUS_ALLOCATION_FAILED;
    memset(out->data, 0, bytes);
    out->rows = rows;
    out->cols = cols;
    out->stride = cols;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_require_finite_eig_result(
    const lmmc_eigen_gen_full_result_t* result)
{
    lmmc_status_t status;
    if (!result) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(&result->real_parts);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(&result->imag_parts);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(&result->vectors_real);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_require_finite_mat(&result->vectors_imag);
}

static lmmc_status_t lmmc_lsr_require_finite_svd_result(
    const lmmc_svd_result_t* result)
{
    lmmc_status_t status;
    if (!result) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(&result->U);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(&result->sigma);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_require_finite_mat(&result->Vt);
}

static lmmc_status_t lmmc_lsr_copy_mat(const lmmc_mat_t* src, lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!lmmc_lsr_mat_valid(src) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(src);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_create(src->rows, src->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_copy(src, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

static lmmc_status_t lmmc_lsr_copy_vec(const lmmc_vec_t* src, lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(src);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(src->size, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_copy(src, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

static int lmmc_lsr_real_is_integer(lmmc_real_t value)
{
    double as_double = (double)value;
    return isfinite(as_double) && floor(as_double) == as_double;
}

static lmmc_status_t lmmc_lsr_pow_finite(lmmc_real_t base,
                                         lmmc_real_t exponent,
                                         lmmc_real_t* out)
{
    double value;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(base) ||
        !lmmc_lsr_real_is_finite(exponent)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (base == (lmmc_real_t)0 && exponent < (lmmc_real_t)0) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    if (base < (lmmc_real_t)0 && !lmmc_lsr_real_is_integer(exponent)) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    value = pow((double)base, (double)exponent);
    if (!isfinite(value)) return LMMC_STATUS_NUMERICAL_FAILURE;
    *out = (lmmc_real_t)value;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_vec_to_column(const lmmc_vec_t* src,
                                            lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!src || !src->data || src->size == 0 || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_lsr_real_array_is_finite(src->data, src->size)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
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
    if (!lmmc_lsr_real_array_is_finite(src->data, src->size)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
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

lmmc_status_t lmmc_lsr_linalg_shape_vec(const lmmc_mat_t* a,
                                        lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!lmmc_lsr_mat_valid(a) || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_vec_create(2, out);
    if (status != LMMC_STATUS_OK) return status;
    out->data[0] = (lmmc_real_t)a->rows;
    out->data[1] = (lmmc_real_t)a->cols;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_eye(size_t n, lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (n == 0 || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_mat_identity(n, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_diag(const lmmc_vec_t* diagonal,
                                   lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(diagonal);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_vec_to_diag(diagonal, diagonal->size, diagonal->size, out);
}

lmmc_status_t lmmc_lsr_linalg_dot(const lmmc_vec_t* a,
                                  const lmmc_vec_t* b,
                                  lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_dot(a, b, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_linalg_cross(const lmmc_vec_t* a,
                                    const lmmc_vec_t* b,
                                    lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != 3 || b->size != 3) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_create(3, out);
    if (status != LMMC_STATUS_OK) return status;
    out->data[0] = a->data[1] * b->data[2] - a->data[2] * b->data[1];
    out->data[1] = a->data[2] * b->data[0] - a->data[0] * b->data[2];
    out->data[2] = a->data[0] * b->data[1] - a->data[1] * b->data[0];
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_norm(const lmmc_vec_t* x,
                                   lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(x);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_norm2(x, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_linalg_vec_add(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_lsr_copy_vec(a, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_axpy((lmmc_real_t)1, b, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_add_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_copy_vec(x, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        out->data[i] += scalar;
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_sub(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_lsr_copy_vec(a, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_axpy((lmmc_real_t)-1, b, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_sub_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    return lmmc_lsr_linalg_vec_add_scalar(x, -scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_scalar_sub_vec(lmmc_real_t scalar,
                                             const lmmc_vec_t* x,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_require_finite_vec(x);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(x->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        out->data[i] = scalar - x->data[i];
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_mul(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_create(a->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        out->data[i] = a->data[i] * b->data[i];
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_div(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_create(a->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        if (b->data[i] == (lmmc_real_t)0) {
            lmmc_vec_destroy(out);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        out->data[i] = a->data[i] / b->data[i];
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_div_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    if (scalar == (lmmc_real_t)0) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_linalg_vec_scale(x, (lmmc_real_t)1 / scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_scalar_div_vec(lmmc_real_t scalar,
                                             const lmmc_vec_t* x,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_require_finite_vec(x);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(x->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        if (x->data[i] == (lmmc_real_t)0) {
            lmmc_vec_destroy(out);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        out->data[i] = scalar / x->data[i];
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_pow(const lmmc_vec_t* base,
                                      const lmmc_vec_t* exponent,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(base);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(exponent);
    if (status != LMMC_STATUS_OK) return status;
    if (base->size != exponent->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_create(base->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        status =
            lmmc_lsr_pow_finite(base->data[i], exponent->data[i], &out->data[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(out);
            return status;
        }
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_pow_scalar(const lmmc_vec_t* base,
                                             lmmc_real_t exponent,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(base);
    if (status != LMMC_STATUS_OK) return status;
    if (!lmmc_lsr_real_is_finite(exponent)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_vec_create(base->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        status = lmmc_lsr_pow_finite(base->data[i], exponent, &out->data[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(out);
            return status;
        }
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_scale(const lmmc_vec_t* x,
                                        lmmc_real_t alpha,
                                        lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(alpha)) return LMMC_STATUS_NUMERICAL_FAILURE;
    status = lmmc_lsr_copy_vec(x, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_scale(out, alpha);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_mul_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    return lmmc_lsr_linalg_vec_scale(x, scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_mat_add(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_add(a, b, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_add_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_copy_mat(a, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            out->data[i * out->stride + j] += scalar;
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_sub(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_sub(a, b, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_sub_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    return lmmc_lsr_linalg_mat_add_scalar(a, -scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_scalar_sub_mat(lmmc_real_t scalar,
                                             const lmmc_mat_t* a,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            out->data[i * out->stride + j] =
                scalar - a->data[i * a->stride + j];
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_mul_elem(const lmmc_mat_t* a,
                                           const lmmc_mat_t* b,
                                           lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            out->data[i * out->stride + j] =
                a->data[i * a->stride + j] * b->data[i * b->stride + j];
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_div(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            lmmc_real_t denom = b->data[i * b->stride + j];
            if (denom == (lmmc_real_t)0) {
                lmmc_mat_destroy(out);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            out->data[i * out->stride + j] =
                a->data[i * a->stride + j] / denom;
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_div_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    if (scalar == (lmmc_real_t)0) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_linalg_mat_scale(a, (lmmc_real_t)1 / scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_scalar_div_mat(lmmc_real_t scalar,
                                             const lmmc_mat_t* a,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            lmmc_real_t denom = a->data[i * a->stride + j];
            if (denom == (lmmc_real_t)0) {
                lmmc_mat_destroy(out);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            out->data[i * out->stride + j] = scalar / denom;
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_pow_elem(const lmmc_mat_t* base,
                                           const lmmc_mat_t* exponent,
                                           lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(base);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(exponent);
    if (status != LMMC_STATUS_OK) return status;
    if (base->rows != exponent->rows || base->cols != exponent->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(base->rows, base->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            status = lmmc_lsr_pow_finite(base->data[i * base->stride + j],
                                         exponent->data[i * exponent->stride + j],
                                         &out->data[i * out->stride + j]);
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(out);
                return status;
            }
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_pow_scalar(const lmmc_mat_t* base,
                                             lmmc_real_t exponent,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(base);
    if (status != LMMC_STATUS_OK) return status;
    if (!lmmc_lsr_real_is_finite(exponent)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_mat_create(base->rows, base->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            status = lmmc_lsr_pow_finite(base->data[i * base->stride + j],
                                         exponent,
                                         &out->data[i * out->stride + j]);
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(out);
                return status;
            }
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_scale(const lmmc_mat_t* a,
                                        lmmc_real_t alpha,
                                        lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(alpha)) return LMMC_STATUS_NUMERICAL_FAILURE;
    status = lmmc_lsr_copy_mat(a, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_scale(out, alpha);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_mul_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    return lmmc_lsr_linalg_mat_scale(a, scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_vec_compare(const lmmc_vec_t* a,
                                          const lmmc_vec_t* b,
                                          lmmc_lsr_compare_op_t op,
                                          lmmc_lsr_bool_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_lsr_bool_vec_create(a->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        if (!lmmc_lsr_compare_eval(a->data[i], op, b->data[i],
                                   &out->data[i])) {
            lmmc_lsr_bool_vec_destroy(out);
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_vec_compare_scalar(const lmmc_vec_t* a,
                                                 lmmc_lsr_compare_op_t op,
                                                 lmmc_real_t scalar,
                                                 lmmc_lsr_bool_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_bool_vec_create(a->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        if (!lmmc_lsr_compare_eval(a->data[i], op, scalar, &out->data[i])) {
            lmmc_lsr_bool_vec_destroy(out);
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_mat_compare(const lmmc_mat_t* a,
                                          const lmmc_mat_t* b,
                                          lmmc_lsr_compare_op_t op,
                                          lmmc_lsr_bool_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_lsr_bool_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            if (!lmmc_lsr_compare_eval(a->data[i * a->stride + j],
                                       op,
                                       b->data[i * b->stride + j],
                                       &out->data[i * out->stride + j])) {
                lmmc_lsr_bool_mat_destroy(out);
                return LMMC_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_mat_compare_scalar(const lmmc_mat_t* a,
                                                 lmmc_lsr_compare_op_t op,
                                                 lmmc_real_t scalar,
                                                 lmmc_lsr_bool_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_bool_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            if (!lmmc_lsr_compare_eval(a->data[i * a->stride + j],
                                       op,
                                       scalar,
                                       &out->data[i * out->stride + j])) {
                lmmc_lsr_bool_mat_destroy(out);
                return LMMC_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return LMMC_STATUS_OK;
}

void lmmc_lsr_bool_vec_destroy(lmmc_lsr_bool_vec_t* vec)
{
    if (!vec) return;
    if (vec->owns_data && vec->data) lmmc_free(vec->data);
    vec->size = 0;
    vec->data = NULL;
    vec->owns_data = 0;
}

void lmmc_lsr_bool_mat_destroy(lmmc_lsr_bool_mat_t* mat)
{
    if (!mat) return;
    if (mat->owns_data && mat->data) lmmc_free(mat->data);
    mat->rows = 0;
    mat->cols = 0;
    mat->stride = 0;
    mat->data = NULL;
    mat->owns_data = 0;
}

lmmc_status_t lmmc_lsr_linalg_matmul(const lmmc_mat_t* a,
                                     const lmmc_mat_t* b,
                                     lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->cols != b->rows) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_mat_create(a->rows, b->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_mul(a, b, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_matvec(const lmmc_mat_t* a,
                                     const lmmc_vec_t* x,
                                     lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(x);
    if (status != LMMC_STATUS_OK) return status;
    if (a->cols != x->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_create(a->rows, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_vec_mul(a, x, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_transpose(const lmmc_mat_t* a,
                                        lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_create(a->cols, a->rows, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_transpose_to(a, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
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
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_det(a, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_linalg_inv(const lmmc_mat_t* a,
                                  lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_inv(a, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
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

    if (!out_rank) return LMMC_STATUS_INVALID_ARGUMENT;
    lmmc_status_t input_status = lmmc_lsr_require_finite_mat(a);
    if (input_status != LMMC_STATUS_OK) return input_status;
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
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_trace(a, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
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

    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
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

    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
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
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_eigen_general_full(a, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_eig_result(out);
    if (status != LMMC_STATUS_OK) {
        lmmc_eigen_gen_full_result_destroy(out);
        return status;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_svd(const lmmc_mat_t* a,
                                  lmmc_svd_result_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_svd(a, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_svd_result(out);
    if (status != LMMC_STATUS_OK) {
        lmmc_svd_result_destroy(out);
        return status;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_eig_table(const lmmc_mat_t* a,
                                        lmmc_lsr_eig_table_t* out)
{
    lmmc_status_t status;
    lmmc_eigen_gen_full_result_t raw = {0};
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
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

size_t lmmc_lsr_eig_table_count(const lmmc_lsr_eig_table_t* table)
{
    return table ? 4u : 0u;
}

const char* lmmc_lsr_eig_table_key(const lmmc_lsr_eig_table_t* table,
                                   size_t index)
{
    static const char* keys[] = {
        "values_real",
        "values_imag",
        "vectors_real",
        "vectors_imag",
    };
    if (!table || index >= sizeof(keys) / sizeof(keys[0])) return NULL;
    return keys[index];
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
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
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

size_t lmmc_lsr_svd_table_count(const lmmc_lsr_svd_table_t* table)
{
    return table ? 3u : 0u;
}

const char* lmmc_lsr_svd_table_key(const lmmc_lsr_svd_table_t* table,
                                   size_t index)
{
    static const char* keys[] = {"U", "S", "Vt"};
    if (!table || index >= sizeof(keys) / sizeof(keys[0])) return NULL;
    return keys[index];
}

void lmmc_lsr_svd_table_destroy(lmmc_lsr_svd_table_t* table)
{
    if (!table) return;
    lmmc_mat_destroy(&table->U);
    lmmc_mat_destroy(&table->S);
    lmmc_mat_destroy(&table->Vt);
}
