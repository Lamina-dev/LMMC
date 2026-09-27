/**
 * @file tensor3_internal.h
 * @brief 三阶张量描述符的公共检查。
 * @internal
 */
#ifndef LMMC_TENSOR3_INTERNAL_H
#define LMMC_TENSOR3_INTERNAL_H

#include "internal.h"
#include "lmmc/tensor3.h"

static inline int lmmc_tensor3_descriptor_fields_are_valid(
    const lmmc_tensor3_t* tensor) {
    if (tensor == NULL) {
        return 0;
    }
    if (tensor->data == NULL) {
        return 0;
    }
    if (tensor->dim0 == 0 || tensor->dim1 == 0 || tensor->dim2 == 0) {
        return 0;
    }
    return tensor->stride0 != 0 && tensor->stride1 != 0 &&
           tensor->stride2 != 0;
}

static inline int lmmc_tensor3_logical_size_is_valid(
    const lmmc_tensor3_t* tensor) {
    size_t total;
    size_t bytes;
    if (!lmmc_safe_mul_size(tensor->dim0, tensor->dim1, &total)) {
        return 0;
    }
    if (!lmmc_safe_mul_size(total, tensor->dim2, &total)) {
        return 0;
    }
    return lmmc_safe_mul_size(total, sizeof(lmmc_real_t), &bytes);
}

static inline int lmmc_tensor3_add_axis_span(
    size_t dim, size_t stride, size_t* max_offset) {
    size_t axis_offset;
    if (!lmmc_safe_mul_size(dim - 1, stride, &axis_offset)) {
        return 0;
    }
    return lmmc_safe_add_size(*max_offset, axis_offset, max_offset);
}

static inline int lmmc_tensor3_span_is_valid(const lmmc_tensor3_t* tensor) {
    size_t max_offset = 0;
    size_t elements;
    size_t bytes;
    if (!lmmc_tensor3_add_axis_span(
            tensor->dim0, tensor->stride0, &max_offset)) {
        return 0;
    }
    if (!lmmc_tensor3_add_axis_span(
            tensor->dim1, tensor->stride1, &max_offset)) {
        return 0;
    }
    if (!lmmc_tensor3_add_axis_span(
            tensor->dim2, tensor->stride2, &max_offset)) {
        return 0;
    }
    if (!lmmc_safe_add_size(max_offset, 1, &elements)) {
        return 0;
    }
    if (!lmmc_safe_mul_size(elements, sizeof(lmmc_real_t), &bytes)) {
        return 0;
    }
    return (uintptr_t)bytes <= UINTPTR_MAX - (uintptr_t)tensor->data;
}

static inline lmmc_status_t lmmc_tensor3_validate(
    const lmmc_tensor3_t* tensor) {
    if (!lmmc_tensor3_descriptor_fields_are_valid(tensor)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_tensor3_logical_size_is_valid(tensor)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_tensor3_span_is_valid(tensor)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}


static inline int lmmc_tensor3_same_shape(const lmmc_tensor3_t* a, const lmmc_tensor3_t* b) {
    return a->dim0 == b->dim0 && a->dim1 == b->dim1 && a->dim2 == b->dim2;
}

static inline lmmc_status_t lmmc_tensor3_validate_binary(
    const lmmc_tensor3_t* a,
    const lmmc_tensor3_t* b,
    const lmmc_tensor3_t* out_tensor
) {
    if (lmmc_tensor3_validate(a) != LMMC_STATUS_OK ||
        lmmc_tensor3_validate(b) != LMMC_STATUS_OK ||
        lmmc_tensor3_validate(out_tensor) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_tensor3_same_shape(a, b) || !lmmc_tensor3_same_shape(a, out_tensor)) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    return LMMC_STATUS_OK;
}

#endif
