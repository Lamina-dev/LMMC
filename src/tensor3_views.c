/**
 * @file tensor3_views.c
 * @brief 三阶张量的零拷贝重塑与切片。
 */
#include "internal/tensor3_internal.h"
#include "lmmc/tensor_nd.h"

static lmmc_status_t lmmc_tensor3_reshape_error(
    const lmmc_tensor3_t* tensor, const size_t* new_dims, lmmc_status_t status) {
    /** @brief 元素总数均可表示且不相等时返回 DIMENSION_MISMATCH。 */
    size_t old_total, new_total;
    if (!lmmc_safe_mul_size(tensor->dim0, tensor->dim1, &old_total) ||
        !lmmc_safe_mul_size(old_total, tensor->dim2, &old_total) ||
        !lmmc_safe_mul_size(new_dims[0], new_dims[1], &new_total) ||
        !lmmc_safe_mul_size(new_total, new_dims[2], &new_total)) {
        return status;
    }
    return old_total != new_total ? LMMC_STATUS_DIMENSION_MISMATCH : status;
}

lmmc_status_t lmmc_tensor3_reshape_view(
    const lmmc_tensor3_t* tensor,
    size_t new_dim0,
    size_t new_dim1,
    size_t new_dim2,
    lmmc_tensor3_t* out_view
) {
    lmmc_tensor_nd_t nd_src;
    lmmc_tensor_nd_t nd_view = {0};
    size_t new_dims[3];
    lmmc_status_t st;

    if (lmmc_tensor3_validate(tensor) != LMMC_STATUS_OK || out_view == NULL ||
        new_dim0 == 0 || new_dim1 == 0 || new_dim2 == 0 ||
        (tensor == out_view && tensor->owns_data)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    nd_src.ndim = 3;
    nd_src.dims[0] = tensor->dim0;
    nd_src.dims[1] = tensor->dim1;
    nd_src.dims[2] = tensor->dim2;
    nd_src.strides[0] = tensor->stride0;
    nd_src.strides[1] = tensor->stride1;
    nd_src.strides[2] = tensor->stride2;
    nd_src.data = tensor->data;
    nd_src.owns_data = 0;

    new_dims[0] = new_dim0;
    new_dims[1] = new_dim1;
    new_dims[2] = new_dim2;

    st = lmmc_tensor_nd_reshape_view(&nd_src, 3, new_dims, &nd_view);
    if (st != LMMC_STATUS_OK) {
        return lmmc_tensor3_reshape_error(tensor, new_dims, st);
    }

    out_view->dim0 = nd_view.dims[0];
    out_view->dim1 = nd_view.dims[1];
    out_view->dim2 = nd_view.dims[2];
    out_view->stride0 = nd_view.strides[0];
    out_view->stride1 = nd_view.strides[1];
    out_view->stride2 = nd_view.strides[2];
    out_view->data = nd_view.data;
    out_view->owns_data = 0;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_tensor3_slice_offset(
    const lmmc_tensor3_t* tensor, size_t begin0, size_t begin1, size_t begin2,
    size_t* out_offset) {
    size_t off0 = 0;
    size_t off1 = 0;
    size_t off2 = 0;
    size_t offset = 0;

    if (!lmmc_safe_mul_size(begin0, tensor->stride0, &off0) ||
        !lmmc_safe_mul_size(begin1, tensor->stride1, &off1) ||
        !lmmc_safe_mul_size(begin2, tensor->stride2, &off2)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_safe_add_size(off0, off1, &offset) ||
        !lmmc_safe_add_size(offset, off2, &offset)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out_offset = offset;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor3_slice_view(
    const lmmc_tensor3_t* tensor,
    size_t begin0,
    size_t end0,
    size_t begin1,
    size_t end1,
    size_t begin2,
    size_t end2,
    lmmc_tensor3_t* out_view
) {
    size_t offset = 0;
    lmmc_tensor3_t view;

    if (lmmc_tensor3_validate(tensor) != LMMC_STATUS_OK || out_view == NULL ||
        (tensor == out_view && tensor->owns_data)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (begin0 >= end0 || begin1 >= end1 || begin2 >= end2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (end0 > tensor->dim0 || end1 > tensor->dim1 || end2 > tensor->dim2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (lmmc_tensor3_slice_offset(tensor, begin0, begin1, begin2, &offset) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    view.dim0 = end0 - begin0;
    view.dim1 = end1 - begin1;
    view.dim2 = end2 - begin2;
    view.stride0 = tensor->stride0;
    view.stride1 = tensor->stride1;
    view.stride2 = tensor->stride2;
    view.data = tensor->data + offset;
    view.owns_data = 0;
    *out_view = view;
    return LMMC_STATUS_OK;
}
