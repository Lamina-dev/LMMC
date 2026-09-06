#include "lmmc/stdlib.h"

#include "memory_bridge.h"

#include "stdlib_internal.h"

lmmc_status_t lmmc_std_num_equal(lmmc_real_t lhs, lmmc_real_t rhs, int* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(lhs) || !lmmc_std_real_is_finite(rhs)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = lhs == rhs ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_num_hash(lmmc_real_t value, uint64_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(value)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = lmmc_std_real_hash_bits(value);
    return LMMC_STATUS_OK;
}

static int lmmc_std_num_set_valid(const lmmc_std_num_set_t* set)
{
    return set && (set->size == 0 || set->data);
}

static lmmc_status_t lmmc_std_num_set_require_finite(
    const lmmc_std_num_set_t* set)
{
    if (!lmmc_std_num_set_valid(set)) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_array_is_finite(set->data, set->size)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    return LMMC_STATUS_OK;
}

static int lmmc_std_num_set_contains_value(const lmmc_std_num_set_t* set,
                                           lmmc_real_t value)
{
    lmmc_real_t key = lmmc_std_num_set_key(value);
    for (size_t i = 0; i < set->size; ++i) {
        if (lmmc_std_num_set_key(set->data[i]) == key) return 1;
    }
    return 0;
}

static lmmc_status_t lmmc_std_num_set_alloc(size_t capacity,
                                            lmmc_std_num_set_t* out)
{
    size_t bytes = 0;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    out->size = 0;
    out->data = NULL;
    out->owns_data = 0;
    if (capacity == 0) return LMMC_STATUS_OK;
    if (lmmc_std_mul_overflow_size(capacity, sizeof(lmmc_real_t), &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out->data = (lmmc_real_t*)lmmc_alloc(bytes);
    if (!out->data) return LMMC_STATUS_ALLOCATION_FAILED;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_std_num_set_append_unique(
    lmmc_std_num_set_t* set,
    lmmc_real_t value)
{
    if (!set || !lmmc_std_real_is_finite(value)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (lmmc_std_num_set_contains_value(set, value)) return LMMC_STATUS_OK;
    if (!set->data) return LMMC_STATUS_INVALID_ARGUMENT;
    set->data[set->size++] = lmmc_std_num_set_key(value);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_num_set_make(const lmmc_real_t* values,
                                    size_t count,
                                    lmmc_std_num_set_t* out)
{
    lmmc_status_t status;
    if (!out || (count > 0 && !values)) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_array_is_finite(values, count)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_std_num_set_alloc(count, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < count; ++i) {
        status = lmmc_std_num_set_append_unique(out, values[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_std_num_set_destroy(out);
            return status;
        }
    }
    return LMMC_STATUS_OK;
}

void lmmc_std_num_set_destroy(lmmc_std_num_set_t* set)
{
    if (!set) return;
    if (set->owns_data && set->data) lmmc_free(set->data);
    set->size = 0;
    set->data = NULL;
    set->owns_data = 0;
}

lmmc_status_t lmmc_std_num_set_contains(const lmmc_std_num_set_t* set,
                                        lmmc_real_t value,
                                        int* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(value)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_std_num_set_require_finite(set);
    if (status != LMMC_STATUS_OK) return status;
    *out = lmmc_std_num_set_contains_value(set, value);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_num_set_subset(const lmmc_std_num_set_t* lhs,
                                      const lmmc_std_num_set_t* rhs,
                                      int* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_num_set_require_finite(lhs);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_std_num_set_require_finite(rhs);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (!lmmc_std_num_set_contains_value(rhs, lhs->data[i])) {
            *out = 0;
            return LMMC_STATUS_OK;
        }
    }
    *out = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_std_num_set_binary_alloc(
    const lmmc_std_num_set_t* lhs,
    const lmmc_std_num_set_t* rhs,
    lmmc_std_num_set_t* out,
    size_t* capacity)
{
    lmmc_status_t status;
    if (!out || (const void*)out == (const void*)lhs ||
        (const void*)out == (const void*)rhs || !capacity) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_std_num_set_require_finite(lhs);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_std_num_set_require_finite(rhs);
    if (status != LMMC_STATUS_OK) return status;
    if (lhs->size > ((size_t)-1) - rhs->size) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *capacity = lhs->size + rhs->size;
    return lmmc_std_num_set_alloc(*capacity, out);
}

lmmc_status_t lmmc_std_num_set_union(const lmmc_std_num_set_t* lhs,
                                     const lmmc_std_num_set_t* rhs,
                                     lmmc_std_num_set_t* out)
{
    size_t capacity = 0;
    lmmc_status_t status = lmmc_std_num_set_binary_alloc(lhs, rhs, out,
                                                         &capacity);
    (void)capacity;
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        status = lmmc_std_num_set_append_unique(out, lhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        status = lmmc_std_num_set_append_unique(out, rhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_std_num_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_std_num_set_intersection(const lmmc_std_num_set_t* lhs,
                                            const lmmc_std_num_set_t* rhs,
                                            lmmc_std_num_set_t* out)
{
    size_t capacity = 0;
    lmmc_status_t status = lmmc_std_num_set_binary_alloc(lhs, rhs, out,
                                                         &capacity);
    (void)capacity;
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_std_num_set_contains_value(rhs, lhs->data[i])) {
            status = lmmc_std_num_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) {
                lmmc_std_num_set_destroy(out);
                return status;
            }
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_num_set_difference(const lmmc_std_num_set_t* lhs,
                                          const lmmc_std_num_set_t* rhs,
                                          lmmc_std_num_set_t* out)
{
    size_t capacity = 0;
    lmmc_status_t status = lmmc_std_num_set_binary_alloc(lhs, rhs, out,
                                                         &capacity);
    (void)capacity;
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (!lmmc_std_num_set_contains_value(rhs, lhs->data[i])) {
            status = lmmc_std_num_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) {
                lmmc_std_num_set_destroy(out);
                return status;
            }
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_num_set_symmetric_difference(
    const lmmc_std_num_set_t* lhs,
    const lmmc_std_num_set_t* rhs,
    lmmc_std_num_set_t* out)
{
    size_t capacity = 0;
    lmmc_status_t status = lmmc_std_num_set_binary_alloc(lhs, rhs, out,
                                                         &capacity);
    (void)capacity;
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (!lmmc_std_num_set_contains_value(rhs, lhs->data[i])) {
            status = lmmc_std_num_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        if (!lmmc_std_num_set_contains_value(lhs, rhs->data[i])) {
            status = lmmc_std_num_set_append_unique(out, rhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_std_num_set_destroy(out);
    return status;
}
