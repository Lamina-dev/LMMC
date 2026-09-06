#ifndef LMMC_STD_STDLIB_INTERNAL_H
#define LMMC_STD_STDLIB_INTERNAL_H

/* Private shared helpers for the LMMC standard-library translation units. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/numeric.h"
#include "lmmc/stdlib.h"

static inline lmmc_status_t lmmc_std_store_real(lmmc_real_t value, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = value;
    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_std_store_finite_real(lmmc_real_t value,
                                                lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite((double)value)) return LMMC_STATUS_NUMERICAL_FAILURE;
    *out = value;
    return LMMC_STATUS_OK;
}

static inline int lmmc_std_real_is_finite(lmmc_real_t value)
{
    return isfinite((double)value);
}

static inline int lmmc_std_real_array_is_finite(const lmmc_real_t* values,
                                         size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (!lmmc_std_real_is_finite(values[i])) return 0;
    }
    return 1;
}

static inline int lmmc_std_mul_overflow_size(size_t a, size_t b, size_t* out)
{
    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }
    if (a > ((size_t)-1) / b) return 1;
    *out = a * b;
    return 0;
}

static inline int lmmc_std_complex_is_finite(const lmmc_complex_t* z)
{
    return z && lmmc_std_real_is_finite(z->real) &&
           lmmc_std_real_is_finite(z->imag);
}

static inline uint64_t lmmc_std_hash_mix64(uint64_t value)
{
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return value;
}

static inline uint64_t lmmc_std_real_hash_bits(lmmc_real_t value)
{
    uint64_t bits = 0;
    if (value == (lmmc_real_t)0) value = (lmmc_real_t)0;
    memcpy(&bits, &value, sizeof(value));
    return lmmc_std_hash_mix64(bits);
}

static inline int lmmc_std_bool_is_valid(int value)
{
    return value == 0 || value == 1;
}

static inline uint64_t lmmc_std_text_hash_bytes(const char* value)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    const unsigned char* cursor = (const unsigned char*)value;
    while (*cursor) {
        hash ^= (uint64_t)(*cursor);
        hash *= UINT64_C(1099511628211);
        ++cursor;
    }
    return lmmc_std_hash_mix64(hash);
}

static inline int lmmc_std_mat_valid(const lmmc_mat_t* a)
{
    return lmmc_mat_descriptor_is_valid(a);
}

static inline int lmmc_std_mat_is_finite(const lmmc_mat_t* a)
{
    if (!lmmc_std_mat_valid(a)) return 0;
    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            if (!lmmc_std_real_is_finite(a->data[i * a->stride + j])) {
                return 0;
            }
        }
    }
    return 1;
}

static inline lmmc_status_t lmmc_std_require_finite_mat(const lmmc_mat_t* a)
{
    if (!lmmc_std_mat_valid(a)) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_std_mat_is_finite(a) ? LMMC_STATUS_OK
                                     : LMMC_STATUS_NUMERICAL_FAILURE;
}

static inline int lmmc_std_vec_is_finite(const lmmc_vec_t* v)
{
    return lmmc_vec_descriptor_is_valid(v) &&
           lmmc_std_real_array_is_finite(v->data, v->size);
}

static inline lmmc_status_t lmmc_std_require_finite_vec(const lmmc_vec_t* v)
{
    if (!lmmc_vec_descriptor_is_valid(v)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return lmmc_std_vec_is_finite(v) ? LMMC_STATUS_OK
                                     : LMMC_STATUS_NUMERICAL_FAILURE;
}

static inline int lmmc_std_compare_eval(lmmc_real_t lhs,
                                 lmmc_std_compare_op_t op,
                                 lmmc_real_t rhs,
                                 uint8_t* out)
{
    if (!out) return 0;
    switch (op) {
    case LMMC_STD_COMPARE_EQ:
        *out = (uint8_t)(lhs == rhs);
        return 1;
    case LMMC_STD_COMPARE_NE:
        *out = (uint8_t)(lhs != rhs);
        return 1;
    case LMMC_STD_COMPARE_LT:
        *out = (uint8_t)(lhs < rhs);
        return 1;
    case LMMC_STD_COMPARE_LE:
        *out = (uint8_t)(lhs <= rhs);
        return 1;
    case LMMC_STD_COMPARE_GT:
        *out = (uint8_t)(lhs > rhs);
        return 1;
    case LMMC_STD_COMPARE_GE:
        *out = (uint8_t)(lhs >= rhs);
        return 1;
    }
    return 0;
}

static inline lmmc_status_t lmmc_std_bool_vec_create(size_t size,
                                              lmmc_std_bool_vec_t* out)
{
    size_t bytes = 0;
    if (!out || size == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lmmc_std_mul_overflow_size(size, sizeof(uint8_t), &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out->data = (uint8_t*)lmmc_alloc(bytes);
    if (!out->data) return LMMC_STATUS_ALLOCATION_FAILED;
    memset(out->data, 0, bytes);
    out->size = size;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_std_bool_mat_create(size_t rows,
                                              size_t cols,
                                              lmmc_std_bool_mat_t* out)
{
    size_t count = 0;
    size_t bytes = 0;
    if (!out || rows == 0 || cols == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lmmc_std_mul_overflow_size(rows, cols, &count) ||
        lmmc_std_mul_overflow_size(count, sizeof(uint8_t), &bytes)) {
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

static inline lmmc_status_t lmmc_std_require_finite_eig_result(
    const lmmc_eigen_gen_full_result_t* result)
{
    lmmc_status_t status;
    if (!result) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_require_finite_vec(&result->real_parts);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_std_require_finite_vec(&result->imag_parts);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_std_require_finite_mat(&result->vectors_real);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_require_finite_mat(&result->vectors_imag);
}

static inline lmmc_status_t lmmc_std_require_finite_svd_result(
    const lmmc_svd_result_t* result)
{
    lmmc_status_t status;
    if (!result) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_require_finite_mat(&result->U);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_std_require_finite_vec(&result->sigma);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_require_finite_mat(&result->Vt);
}

static inline lmmc_status_t lmmc_std_copy_mat(const lmmc_mat_t* src, lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!lmmc_std_mat_valid(src) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_require_finite_mat(src);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_create(src->rows, src->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_copy(src, out);
    if (status == LMMC_STATUS_OK) status = lmmc_std_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

static inline lmmc_status_t lmmc_std_copy_vec(const lmmc_vec_t* src, lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_require_finite_vec(src);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(src->size, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_copy(src, out);
    if (status == LMMC_STATUS_OK) status = lmmc_std_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

static inline int lmmc_std_real_is_integer(lmmc_real_t value)
{
    double as_double = (double)value;
    return isfinite(as_double) && floor(as_double) == as_double;
}

static inline lmmc_status_t lmmc_std_pow_finite(lmmc_real_t base,
                                         lmmc_real_t exponent,
                                         lmmc_real_t* out)
{
    double value;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(base) ||
        !lmmc_std_real_is_finite(exponent)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (base == (lmmc_real_t)0 && exponent < (lmmc_real_t)0) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    if (base < (lmmc_real_t)0 && !lmmc_std_real_is_integer(exponent)) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    value = pow((double)base, (double)exponent);
    if (!isfinite(value)) return LMMC_STATUS_NUMERICAL_FAILURE;
    *out = (lmmc_real_t)value;
    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_std_vec_to_column(const lmmc_vec_t* src,
                                            lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!src || !src->data || src->size == 0 || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_std_real_array_is_finite(src->data, src->size)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_mat_create(src->size, 1, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < src->size; ++i) {
        LMMC_REAL_SET(&out->data[i * out->stride], &src->data[i]);
    }
    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_std_vec_to_diag(const lmmc_vec_t* src,
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
    if (!lmmc_std_real_array_is_finite(src->data, src->size)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_mat_create(rows, cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < src->size; ++i) {
        LMMC_REAL_SET(&out->data[i * out->stride + i], &src->data[i]);
    }
    return LMMC_STATUS_OK;
}

static inline lmmc_real_t lmmc_std_num_set_key(lmmc_real_t value)
{
    return value == (lmmc_real_t)0 ? (lmmc_real_t)0 : value;
}

#endif
