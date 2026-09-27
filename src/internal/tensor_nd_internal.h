/**
 * @file tensor_nd_internal.h
 * @brief N 维张量结果存储、索引遍历与乘积累加。
 * @internal
 */
#ifndef LMMC_TENSOR_ND_INTERNAL_H
#define LMMC_TENSOR_ND_INTERNAL_H

#include <string.h>
#include "internal.h"
#include "memory_bridge.h"
#include "lmmc/tensor_nd.h"

static inline lmmc_status_t lmmc_tensor_nd_layout(
    size_t ndim, const size_t* dims, size_t* strides, size_t* total) {
    size_t count = 1;
    if (dims == NULL || total == NULL ||
        ndim == 0 || ndim > LMMC_TENSOR_MAX_NDIM) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = ndim; i > 0; --i) {
        const size_t axis = i - 1;
        if (dims[axis] == 0) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        if (strides != NULL) {
            strides[axis] = count;
        }
        if (!lmmc_safe_mul_size(count, dims[axis], &count)) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    *total = count;
    return LMMC_STATUS_OK;
}

static inline int lmmc_tensor_nd_descriptor_is_valid(const lmmc_tensor_nd_t* tensor) {
    size_t total;
    size_t logical_bytes;
    size_t max_offset = 0;
    size_t elements;
    size_t bytes;
    if (tensor == NULL || tensor->data == NULL ||
        lmmc_tensor_nd_layout(tensor->ndim, tensor->dims, NULL, &total) != LMMC_STATUS_OK ||
        !lmmc_safe_mul_size(total, sizeof(lmmc_real_t), &logical_bytes)) {
        return 0;
    }
    for (size_t i = 0; i < tensor->ndim; ++i) {
        size_t axis_offset;
        if (tensor->strides[i] == 0 ||
            !lmmc_safe_mul_size(tensor->dims[i] - 1, tensor->strides[i], &axis_offset) ||
            !lmmc_safe_add_size(max_offset, axis_offset, &max_offset)) {
            return 0;
        }
    }
    return lmmc_safe_add_size(max_offset, 1, &elements) &&
           lmmc_safe_mul_size(elements, sizeof(lmmc_real_t), &bytes) &&
           (uintptr_t)bytes <= UINTPTR_MAX - (uintptr_t)tensor->data;
}

/** @brief 为乘积与置换结果分配缓冲区，并仅在成功后提交输出描述符。 */
static inline lmmc_status_t lmmc_tensor_nd_allocate_result(
    size_t ndim, const size_t* dims, lmmc_tensor_nd_t* out) {
    size_t total;
    size_t bytes;
    lmmc_tensor_nd_t result = {0};
    if (out == NULL ||
        lmmc_tensor_nd_layout(ndim, dims, result.strides, &total) != LMMC_STATUS_OK ||
        !lmmc_safe_mul_size(total, sizeof(lmmc_real_t), &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < ndim; ++i) {
        result.dims[i] = dims[i];
    }
    lmmc_real_t* data = (lmmc_real_t*)lmmc_memory_alloc(bytes);
    if (data == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, bytes);
    for (size_t i = 0; i < total; ++i) {
        LMMC_REAL_INIT(&data[i]);
    }

    result.ndim = ndim;
    result.data = data;
    result.owns_data = 1;
    *out = result;
    return LMMC_STATUS_OK;
}

static inline size_t lmmc_tensor_nd_offset(
    size_t ndim, const size_t* strides, const size_t* idx) {
    size_t offset = 0;
    for (size_t i = 0; i < ndim; ++i) {
        offset += idx[i] * strides[i];
    }
    return offset;
}

/** @brief 按行主序推进多维索引，末元素回绕后返回零。 */
static inline int lmmc_tensor_nd_next_index(
    size_t ndim, const size_t* dims, size_t* idx) {
    while (ndim > 0) {
        --ndim;
        idx[ndim]++;
        if (idx[ndim] < dims[ndim]) {
            return 1;
        }
        idx[ndim] = 0;
    }
    return 0;
}

static inline void lmmc_tensor_nd_add_product(
    lmmc_real_t* sum, const lmmc_real_t* a, const lmmc_real_t* b) {
    lmmc_real_t va, vb, prod, new_sum;
    LMMC_REAL_INIT(&va);
    LMMC_REAL_INIT(&vb);
    LMMC_REAL_INIT(&prod);
    LMMC_REAL_INIT(&new_sum);
    LMMC_REAL_SET(&va, a);
    LMMC_REAL_SET(&vb, b);
    LMMC_REAL_MUL(&prod, &va, &vb);
    LMMC_REAL_ADD(&new_sum, sum, &prod);
    LMMC_REAL_SET(sum, &new_sum);
    LMMC_REAL_CLEAR(&va);
    LMMC_REAL_CLEAR(&vb);
    LMMC_REAL_CLEAR(&prod);
    LMMC_REAL_CLEAR(&new_sum);
}

#endif
