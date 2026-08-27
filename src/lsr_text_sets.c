#include "lmmc/lsr_stdlib.h"

#include <string.h>

#include "memory_bridge.h"

#include "lsr_stdlib_internal.h"

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
