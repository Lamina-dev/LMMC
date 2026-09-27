/**
 * @file tensor_nd_views.c
 * @brief N-D 张量的轴置换与零拷贝重塑.
 */
#include "internal.h"
#include "internal/tensor_nd_internal.h"

static lmmc_status_t lmmc_tensor_nd_permutation_shape(
    const lmmc_tensor_nd_t* in, const size_t* perm, const lmmc_tensor_nd_t* out,
    size_t* new_dims) {
    int seen[LMMC_TENSOR_MAX_NDIM] = {0};
    if (!lmmc_tensor_nd_descriptor_is_valid(in) || perm == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (in == out && in->owns_data) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < in->ndim; ++i) {
        if (perm[i] >= in->ndim || seen[perm[i]]) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        seen[perm[i]] = 1;
        new_dims[i] = in->dims[perm[i]];
    }
    return LMMC_STATUS_OK;
}

static void lmmc_tensor_nd_copy_permuted(
    const lmmc_tensor_nd_t* in, const size_t* perm, lmmc_tensor_nd_t* out) {
    size_t idx[LMMC_TENSOR_MAX_NDIM] = {0};
    /** @brief 按输入顺序遍历，以置换后的索引复制数据。 */
    do {
        const size_t src_offset = lmmc_tensor_nd_offset(in->ndim, in->strides, idx);
        size_t dst_offset = 0;
        for (size_t i = 0; i < in->ndim; ++i) {
            dst_offset += idx[perm[i]] * out->strides[i];
        }
        LMMC_REAL_SET(&out->data[dst_offset], &in->data[src_offset]);
    } while (lmmc_tensor_nd_next_index(in->ndim, in->dims, idx));
}

lmmc_status_t lmmc_tensor_nd_permute(
    const lmmc_tensor_nd_t* in, const size_t* perm, lmmc_tensor_nd_t* out) {
    lmmc_tensor_nd_t input_snapshot;
    lmmc_tensor_nd_t result = {0};
    size_t new_dims[LMMC_TENSOR_MAX_NDIM];
    lmmc_status_t st = lmmc_tensor_nd_permutation_shape(in, perm, out, new_dims);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (in == out) {
        input_snapshot = *in;
        in = &input_snapshot;
    }
    st = lmmc_tensor_nd_allocate_result(in->ndim, new_dims, &result);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    lmmc_tensor_nd_copy_permuted(in, perm, &result);
    *out = result;
    return LMMC_STATUS_OK;
}

/** @brief 检查 N-D 张量是否为行优先连续存储。 */
static int lmmc_tensor_nd_is_contiguous(const lmmc_tensor_nd_t* t) {
    size_t i;
    size_t expected_stride;
    if (t == NULL || t->ndim == 0) return 0;

    expected_stride = 1;
    for (i = t->ndim; i > 0; --i) {
        if (t->strides[i - 1] != expected_stride) return 0;
        if (t->dims[i - 1] == 0 ||
            !lmmc_safe_mul_size(expected_stride, t->dims[i - 1], &expected_stride)) return 0;
    }
    return 1;
}

static lmmc_status_t lmmc_tensor_nd_validate_reshape(
    const lmmc_tensor_nd_t* src, size_t new_ndim, const size_t* new_dims,
    const lmmc_tensor_nd_t* out_view) {
    if (!lmmc_tensor_nd_descriptor_is_valid(src) ||
        new_dims == NULL || out_view == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    /** @brief 拒绝将拥有数据的源对象覆写为非拥有视图，以保留缓冲区所有者。 */
    if (src == out_view && src->owns_data) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (new_ndim == 0 || new_ndim > LMMC_TENSOR_MAX_NDIM) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}


lmmc_status_t lmmc_tensor_nd_reshape_view(const lmmc_tensor_nd_t* src,
    size_t new_ndim, const size_t* new_dims, lmmc_tensor_nd_t* out_view) {
    size_t old_total, new_total;
    lmmc_tensor_nd_t view = {0};
    if (lmmc_tensor_nd_validate_reshape(src, new_ndim, new_dims, out_view) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_tensor_nd_layout(new_ndim, new_dims, view.strides, &new_total) !=
            LMMC_STATUS_OK ||
        lmmc_tensor_nd_layout(src->ndim, src->dims, NULL, &old_total) !=
            LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (old_total != new_total) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_tensor_nd_is_contiguous(src)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /** @brief 形状数组可与 out_view 别名；读取全部输入元数据后再写入输出。 */
    for (size_t i = 0; i < new_ndim; ++i) {
        view.dims[i] = new_dims[i];
    }
    view.ndim = new_ndim;
    view.data = src->data;
    view.owns_data = 0;
    *out_view = view;
    return LMMC_STATUS_OK;
}
