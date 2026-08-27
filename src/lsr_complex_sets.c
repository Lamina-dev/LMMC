#include "lmmc/lsr_stdlib.h"

#include "memory_bridge.h"

#include "lsr_stdlib_internal.h"

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
