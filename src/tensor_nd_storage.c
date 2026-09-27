/**
 * @file tensor_nd_storage.c
 * @brief N 维张量的分配、释放与元素访问。
 */
#include "internal/tensor_nd_internal.h"

#include <string.h>

lmmc_status_t lmmc_tensor_nd_create(size_t ndim, const size_t* dims, lmmc_tensor_nd_t* out) {
    return lmmc_tensor_nd_allocate_result(ndim, dims, out);
}

lmmc_status_t lmmc_tensor_nd_get(const lmmc_tensor_nd_t* t, const size_t* idx, lmmc_real_t* out) {
    size_t i;
    size_t offset = 0;

    if (!lmmc_tensor_nd_descriptor_is_valid(t) || idx == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < t->ndim; ++i) {
        if (idx[i] >= t->dims[i]) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        offset += idx[i] * t->strides[i];
    }

    LMMC_REAL_SET(out, &t->data[offset]);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_nd_set(lmmc_tensor_nd_t* t, const size_t* idx, lmmc_real_t value) {
    size_t i;
    size_t offset = 0;

    if (!lmmc_tensor_nd_descriptor_is_valid(t) || idx == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < t->ndim; ++i) {
        if (idx[i] >= t->dims[i]) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        offset += idx[i] * t->strides[i];
    }

    LMMC_REAL_SET(&t->data[offset], &value);
    return LMMC_STATUS_OK;
}

void lmmc_tensor_nd_destroy(lmmc_tensor_nd_t* t) {
    if (t == NULL) return;

    if (t->owns_data && t->data != NULL) {
        size_t total;
        if (lmmc_tensor_nd_layout(t->ndim, t->dims, NULL, &total) == LMMC_STATUS_OK) {
            for (size_t i = 0; i < total; ++i) {
                LMMC_REAL_CLEAR(&t->data[i]);
            }
        }
        lmmc_memory_free(t->data);
    }

    t->ndim = 0;
    memset(t->dims, 0, sizeof(t->dims));
    memset(t->strides, 0, sizeof(t->strides));
    t->data = NULL;
    t->owns_data = 0;
}
