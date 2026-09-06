#include "lmmc/stdlib.h"

#include "memory_bridge.h"

#include "stdlib_internal.h"

lmmc_status_t lmmc_std_bool_equal(int lhs, int rhs, int* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_bool_is_valid(lhs) || !lmmc_std_bool_is_valid(rhs)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out = lhs == rhs ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_bool_hash(int value, uint64_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_bool_is_valid(value)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out = lmmc_std_hash_mix64(value ? UINT64_C(1) : UINT64_C(0));
    return LMMC_STATUS_OK;
}

static int lmmc_std_bool_set_valid(const lmmc_std_bool_set_t* set)
{
    return set && set->size <= 2 && (set->size == 0 || set->data);
}

static lmmc_status_t lmmc_std_bool_set_require_valid(
    const lmmc_std_bool_set_t* set)
{
    if (!lmmc_std_bool_set_valid(set)) return LMMC_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < set->size; ++i) {
        if (!lmmc_std_bool_is_valid(set->data[i])) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    return LMMC_STATUS_OK;
}

static int lmmc_std_bool_set_contains_value(const lmmc_std_bool_set_t* set,
                                            int value)
{
    uint8_t key = value ? 1u : 0u;
    for (size_t i = 0; i < set->size; ++i) {
        if (set->data[i] == key) return 1;
    }
    return 0;
}

static lmmc_status_t lmmc_std_bool_set_alloc(size_t capacity,
                                             lmmc_std_bool_set_t* out)
{
    if (!out || capacity > 2) return LMMC_STATUS_INVALID_ARGUMENT;
    out->size = 0;
    out->data = NULL;
    out->owns_data = 0;
    if (capacity == 0) return LMMC_STATUS_OK;
    out->data = (uint8_t*)lmmc_alloc_array(capacity, sizeof(uint8_t));
    if (!out->data) return LMMC_STATUS_ALLOCATION_FAILED;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_std_bool_set_append_unique(
    lmmc_std_bool_set_t* set,
    int value)
{
    if (!set || !lmmc_std_bool_is_valid(value)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_std_bool_set_contains_value(set, value)) return LMMC_STATUS_OK;
    if (set->size >= 2) return LMMC_STATUS_INVALID_ARGUMENT;
    set->data[set->size++] = value ? 1u : 0u;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_bool_set_make(const int* values,
                                     size_t count,
                                     lmmc_std_bool_set_t* out)
{
    lmmc_status_t status;
    if (!out || (count > 0 && !values)) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_bool_set_alloc(count > 2 ? 2 : count, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < count; ++i) {
        status = lmmc_std_bool_set_append_unique(out, values[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_std_bool_set_destroy(out);
            return status;
        }
    }
    return LMMC_STATUS_OK;
}

void lmmc_std_bool_set_destroy(lmmc_std_bool_set_t* set)
{
    if (!set) return;
    if (set->owns_data && set->data) lmmc_free(set->data);
    set->size = 0;
    set->data = NULL;
    set->owns_data = 0;
}

lmmc_status_t lmmc_std_bool_set_contains(const lmmc_std_bool_set_t* set,
                                         int value,
                                         int* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_bool_is_valid(value)) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_bool_set_require_valid(set);
    if (status != LMMC_STATUS_OK) return status;
    *out = lmmc_std_bool_set_contains_value(set, value);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_bool_set_subset(const lmmc_std_bool_set_t* lhs,
                                       const lmmc_std_bool_set_t* rhs,
                                       int* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_bool_set_require_valid(lhs);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_std_bool_set_require_valid(rhs);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (!lmmc_std_bool_set_contains_value(rhs, lhs->data[i])) {
            *out = 0;
            return LMMC_STATUS_OK;
        }
    }
    *out = 1;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_std_bool_set_binary_alloc(
    const lmmc_std_bool_set_t* lhs,
    const lmmc_std_bool_set_t* rhs,
    lmmc_std_bool_set_t* out)
{
    lmmc_status_t status;
    if (!out || (const void*)out == (const void*)lhs ||
        (const void*)out == (const void*)rhs) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_std_bool_set_require_valid(lhs);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_std_bool_set_require_valid(rhs);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_bool_set_alloc(2, out);
}

lmmc_status_t lmmc_std_bool_set_union(const lmmc_std_bool_set_t* lhs,
                                      const lmmc_std_bool_set_t* rhs,
                                      lmmc_std_bool_set_t* out)
{
    lmmc_status_t status = lmmc_std_bool_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        status = lmmc_std_bool_set_append_unique(out, lhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        status = lmmc_std_bool_set_append_unique(out, rhs->data[i]);
        if (status != LMMC_STATUS_OK) goto fail;
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_std_bool_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_std_bool_set_intersection(const lmmc_std_bool_set_t* lhs,
                                             const lmmc_std_bool_set_t* rhs,
                                             lmmc_std_bool_set_t* out)
{
    lmmc_status_t status = lmmc_std_bool_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (lmmc_std_bool_set_contains_value(rhs, lhs->data[i])) {
            status = lmmc_std_bool_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_std_bool_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_std_bool_set_difference(const lmmc_std_bool_set_t* lhs,
                                           const lmmc_std_bool_set_t* rhs,
                                           lmmc_std_bool_set_t* out)
{
    lmmc_status_t status = lmmc_std_bool_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (!lmmc_std_bool_set_contains_value(rhs, lhs->data[i])) {
            status = lmmc_std_bool_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_std_bool_set_destroy(out);
    return status;
}

lmmc_status_t lmmc_std_bool_set_symmetric_difference(
    const lmmc_std_bool_set_t* lhs,
    const lmmc_std_bool_set_t* rhs,
    lmmc_std_bool_set_t* out)
{
    lmmc_status_t status = lmmc_std_bool_set_binary_alloc(lhs, rhs, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < lhs->size; ++i) {
        if (!lmmc_std_bool_set_contains_value(rhs, lhs->data[i])) {
            status = lmmc_std_bool_set_append_unique(out, lhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    for (size_t i = 0; i < rhs->size; ++i) {
        if (!lmmc_std_bool_set_contains_value(lhs, rhs->data[i])) {
            status = lmmc_std_bool_set_append_unique(out, rhs->data[i]);
            if (status != LMMC_STATUS_OK) goto fail;
        }
    }
    return LMMC_STATUS_OK;
fail:
    lmmc_std_bool_set_destroy(out);
    return status;
}
