/**
 * @file tensor3_storage.c
 * @brief 三阶张量的分配、包装、释放与元素访问.
 */
#include "memory_bridge.h"
#include "internal/tensor3_internal.h"
#include "lmmc/tensor_nd.h"

lmmc_status_t lmmc_tensor3_create(size_t dim0, size_t dim1, size_t dim2, lmmc_tensor3_t* out_tensor) {
    lmmc_tensor_nd_t nd = {0};
    size_t dims[3];
    lmmc_status_t st;

    if (out_tensor == NULL || dim0 == 0 || dim1 == 0 || dim2 == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    dims[0] = dim0;
    dims[1] = dim1;
    dims[2] = dim2;
    st = lmmc_tensor_nd_create(3, dims, &nd);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    out_tensor->dim0 = nd.dims[0];
    out_tensor->dim1 = nd.dims[1];
    out_tensor->dim2 = nd.dims[2];
    out_tensor->stride0 = nd.strides[0];
    out_tensor->stride1 = nd.strides[1];
    out_tensor->stride2 = nd.strides[2];
    out_tensor->data = nd.data;
    out_tensor->owns_data = nd.owns_data;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor3_wrap(
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t stride0,
    size_t stride1,
    size_t stride2,
    lmmc_real_t* data,
    lmmc_tensor3_t* out_tensor
) {
    lmmc_tensor3_t candidate = {0};
    if (out_tensor == NULL || data == NULL || dim0 == 0 || dim1 == 0 || dim2 == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (stride0 == 0 || stride1 == 0 || stride2 == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    candidate.dim0 = dim0;
    candidate.dim1 = dim1;
    candidate.dim2 = dim2;
    candidate.stride0 = stride0;
    candidate.stride1 = stride1;
    candidate.stride2 = stride2;
    candidate.data = data;
    candidate.owns_data = 0;

    if (lmmc_tensor3_validate(&candidate) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    *out_tensor = candidate;
    return LMMC_STATUS_OK;
}

void lmmc_tensor3_destroy(lmmc_tensor3_t* tensor) {
    if (tensor == NULL) {
        return;
    }
    if (tensor->owns_data && tensor->data != NULL) {
        size_t total = 0;
        size_t total_part = 0;
        if (lmmc_safe_mul_size(tensor->dim0, tensor->dim1, &total_part) &&
            lmmc_safe_mul_size(total_part, tensor->dim2, &total)) {
            for (size_t i = 0; i < total; ++i) {
                LMMC_REAL_CLEAR(&tensor->data[i]);
            }
        }
        lmmc_memory_free(tensor->data);
    }
    tensor->dim0 = 0;
    tensor->dim1 = 0;
    tensor->dim2 = 0;
    tensor->stride0 = 0;
    tensor->stride1 = 0;
    tensor->stride2 = 0;
    tensor->data = NULL;
    tensor->owns_data = 0;
}

lmmc_status_t lmmc_tensor3_set(
    lmmc_tensor3_t* tensor, size_t i, size_t j, size_t k,
    lmmc_real_t value)
{
    if (lmmc_tensor3_validate(tensor) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (i >= tensor->dim0 || j >= tensor->dim1 || k >= tensor->dim2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    const size_t offset =
        i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2;
    LMMC_REAL_SET(&tensor->data[offset], &value);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor3_get(
    const lmmc_tensor3_t* tensor, size_t i, size_t j, size_t k,
    lmmc_real_t* out_value)
{
    if (lmmc_tensor3_validate(tensor) != LMMC_STATUS_OK ||
        out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (i >= tensor->dim0 || j >= tensor->dim1 || k >= tensor->dim2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    const size_t offset =
        i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2;
    LMMC_REAL_SET(out_value, &tensor->data[offset]);
    return LMMC_STATUS_OK;
}
