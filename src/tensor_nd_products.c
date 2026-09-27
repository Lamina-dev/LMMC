/**
 * @file tensor_nd_products.c
 * @brief N 维张量的轴收缩与模 n 乘积。
 */
#include "internal/tensor_nd_internal.h"

typedef struct {
    size_t naxes;
    size_t contract_size;
    const size_t* axes_a;
    const size_t* axes_b;
    size_t a_free_axes[LMMC_TENSOR_MAX_NDIM];
    size_t b_free_axes[LMMC_TENSOR_MAX_NDIM];
    size_t n_a_free;
    size_t n_b_free;
} lmmc_tensor_nd_contraction_t;

static int lmmc_tensor_nd_contraction_inputs_are_valid(
    const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    const lmmc_tensor_nd_t* out) {
    if (!lmmc_tensor_nd_descriptor_is_valid(a)) {
        return 0;
    }
    if (!lmmc_tensor_nd_descriptor_is_valid(b)) {
        return 0;
    }
    return out != NULL;
}

static int lmmc_tensor_nd_contraction_alias_is_valid(
    const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    const lmmc_tensor_nd_t* out) {
    if (out == a && a->owns_data) {
        return 0;
    }
    if (out == b && b->owns_data) {
        return 0;
    }
    return 1;
}

static int lmmc_tensor_nd_contraction_axes_are_valid(
    const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    const lmmc_tensor_nd_contraction_t* contraction) {
    if (contraction->naxes > a->ndim) {
        return 0;
    }
    if (contraction->naxes > b->ndim) {
        return 0;
    }
    if (contraction->naxes == 0) {
        return 1;
    }
    if (contraction->axes_a == NULL) {
        return 0;
    }
    return contraction->axes_b != NULL;
}

static lmmc_status_t lmmc_tensor_nd_validate_contraction(
    const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    const lmmc_tensor_nd_contraction_t* contraction,
    const lmmc_tensor_nd_t* out) {
    if (!lmmc_tensor_nd_contraction_inputs_are_valid(a, b, out)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_tensor_nd_contraction_alias_is_valid(a, b, out)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_tensor_nd_contraction_axes_are_valid(a, b, contraction)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

static size_t lmmc_tensor_nd_free_axes(size_t ndim, const int* contracted, size_t* axes) {
    size_t count = 0;
    /** @brief 自由轴保持输入顺序。 */
    for (size_t i = 0; i < ndim; ++i) {
        if (!contracted[i]) {
            axes[count++] = i;
        }
    }
    return count;
}

static lmmc_status_t lmmc_tensor_nd_contraction_shape(
    const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    lmmc_tensor_nd_contraction_t* contraction, size_t* out_ndim, size_t* out_dims) {
    int a_contracted[LMMC_TENSOR_MAX_NDIM] = {0};
    int b_contracted[LMMC_TENSOR_MAX_NDIM] = {0};
    contraction->contract_size = 1;
    for (size_t i = 0; i < contraction->naxes; ++i) {
        const size_t axis_a = contraction->axes_a[i];
        const size_t axis_b = contraction->axes_b[i];
        if (axis_a >= a->ndim || axis_b >= b->ndim) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        if (a_contracted[axis_a] || b_contracted[axis_b]) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        if (a->dims[axis_a] != b->dims[axis_b] ||
            !lmmc_safe_mul_size(
                contraction->contract_size, a->dims[axis_a],
                &contraction->contract_size)) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        a_contracted[axis_a] = 1;
        b_contracted[axis_b] = 1;
    }
    contraction->n_a_free = lmmc_tensor_nd_free_axes(a->ndim, a_contracted, contraction->a_free_axes);
    contraction->n_b_free = lmmc_tensor_nd_free_axes(b->ndim, b_contracted, contraction->b_free_axes);
    const size_t free_count = contraction->n_a_free + contraction->n_b_free;
    /** @brief 标量结果表示为长度 1 的一维张量。 */
    *out_ndim = free_count == 0 ? 1 : free_count;
    if (*out_ndim > LMMC_TENSOR_MAX_NDIM) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    size_t j = 0;
    for (size_t i = 0; i < contraction->n_a_free; ++i) {
        out_dims[j++] = a->dims[contraction->a_free_axes[i]];
    }
    for (size_t i = 0; i < contraction->n_b_free; ++i) {
        out_dims[j++] = b->dims[contraction->b_free_axes[i]];
    }
    if (free_count == 0) {
        out_dims[0] = 1;
    }
    return LMMC_STATUS_OK;
}

static size_t lmmc_tensor_nd_contracted_offset(
    const lmmc_tensor_nd_t* tensor, const size_t* free_axes, size_t n_free,
    const size_t* out_idx, const size_t* contracted_axes, size_t naxes,
    const size_t* contract_idx) {
    size_t full_idx[LMMC_TENSOR_MAX_NDIM] = {0};
    for (size_t i = 0; i < n_free; ++i) {
        full_idx[free_axes[i]] = out_idx[i];
    }
    for (size_t i = 0; i < naxes; ++i) {
        full_idx[contracted_axes[i]] = contract_idx[i];
    }
    return lmmc_tensor_nd_offset(tensor->ndim, tensor->strides, full_idx);
}

static void lmmc_tensor_nd_next_contracted_index(
    const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_contraction_t* contraction,
    size_t* contract_idx) {
    size_t ci = contraction->naxes;
    /** @brief 仅在收缩维度内递增并回绕 contract_idx。 */
    while (ci > 0) {
        --ci;
        contract_idx[ci]++;
        if (contract_idx[ci] < a->dims[contraction->axes_a[ci]]) {
            break;
        }
        contract_idx[ci] = 0;
    }
}

static void lmmc_tensor_nd_contract_element(
    const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    const lmmc_tensor_nd_contraction_t* contraction, size_t contract_size,
    const size_t* out_idx, lmmc_real_t* sum) {
    size_t contract_idx[LMMC_TENSOR_MAX_NDIM] = {0};
    LMMC_REAL_SET_D(sum, 0.0);
    for (size_t k = 0; k < contract_size; ++k) {
        const size_t a_offset = lmmc_tensor_nd_contracted_offset(
            a, contraction->a_free_axes, contraction->n_a_free, out_idx,
            contraction->axes_a, contraction->naxes, contract_idx);
        const size_t b_offset = lmmc_tensor_nd_contracted_offset(
            b, contraction->b_free_axes, contraction->n_b_free,
            out_idx + contraction->n_a_free, contraction->axes_b,
            contraction->naxes, contract_idx);
        lmmc_tensor_nd_add_product(sum, &a->data[a_offset], &b->data[b_offset]);
        lmmc_tensor_nd_next_contracted_index(a, contraction, contract_idx);
    }
}

static void lmmc_tensor_nd_execute_contraction(
    const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    const lmmc_tensor_nd_contraction_t* contraction, lmmc_tensor_nd_t* out) {
    size_t out_idx[LMMC_TENSOR_MAX_NDIM] = {0};
    do {
        lmmc_real_t sum;
        LMMC_REAL_INIT(&sum);
        lmmc_tensor_nd_contract_element(
            a, b, contraction, contraction->contract_size, out_idx, &sum);
        const size_t offset = lmmc_tensor_nd_offset(out->ndim, out->strides, out_idx);
        LMMC_REAL_SET(&out->data[offset], &sum);
        LMMC_REAL_CLEAR(&sum);
    } while (lmmc_tensor_nd_next_index(out->ndim, out->dims, out_idx));
}

lmmc_status_t lmmc_tensor_nd_contract(const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    const size_t* axes_a, const size_t* axes_b, size_t naxes, lmmc_tensor_nd_t* out) {
    lmmc_tensor_nd_contraction_t contraction;
    lmmc_tensor_nd_t a_snapshot;
    lmmc_tensor_nd_t b_snapshot;
    lmmc_tensor_nd_t result = {0};
    size_t out_ndim;
    size_t out_dims[LMMC_TENSOR_MAX_NDIM];
    contraction.naxes = naxes;
    contraction.axes_a = axes_a;
    contraction.axes_b = axes_b;
    lmmc_status_t st = lmmc_tensor_nd_validate_contraction(a, b, &contraction, out);
    if (st == LMMC_STATUS_OK) {
        st = lmmc_tensor_nd_contraction_shape(a, b, &contraction, &out_ndim, out_dims);
    }
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (out == a) {
        a_snapshot = *a;
        a = &a_snapshot;
    }
    if (out == b) {
        b_snapshot = *b;
        b = &b_snapshot;
    }
    st = lmmc_tensor_nd_allocate_result(out_ndim, out_dims, &result);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    lmmc_tensor_nd_execute_contraction(a, b, &contraction, &result);
    *out = result;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_tensor_nd_validate_mode_product(
    const lmmc_tensor_nd_t* t, const lmmc_mat_t* mat, size_t mode,
    const lmmc_tensor_nd_t* out) {
    if (!lmmc_tensor_nd_descriptor_is_valid(t) ||
        !lmmc_mat_descriptor_is_valid(mat) || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (out == t && t->owns_data) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (mode >= t->ndim || mat->cols != t->dims[mode]) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

static void lmmc_tensor_nd_mode_product_element(
    const lmmc_tensor_nd_t* t, const lmmc_mat_t* mat, size_t mode,
    const size_t* out_idx, lmmc_real_t* sum) {
    const size_t j_dim = t->dims[mode];
    LMMC_REAL_SET_D(sum, 0.0);
    for (size_t i = 0; i < j_dim; ++i) {
        /** @brief 源索引沿用 out_idx，仅将模维替换为 i。 */
        size_t src_offset = 0;
        for (size_t d = 0; d < t->ndim; ++d) {
            if (d == mode) {
                src_offset += i * t->strides[d];
            } else {
                src_offset += out_idx[d] * t->strides[d];
            }
        }
        lmmc_tensor_nd_add_product(
            sum, &mat->data[out_idx[mode] * mat->stride + i], &t->data[src_offset]);
    }
}

static void lmmc_tensor_nd_execute_mode_product(
    const lmmc_tensor_nd_t* t, const lmmc_mat_t* mat, size_t mode,
    lmmc_tensor_nd_t* out) {
    /** @brief 模乘积：out[i0,...,i_mode,...] = sum_j mat[i_mode,j] * t[i0,...,j,...]。 */
    size_t out_idx[LMMC_TENSOR_MAX_NDIM] = {0};
    do {
        lmmc_real_t sum;
        LMMC_REAL_INIT(&sum);
        lmmc_tensor_nd_mode_product_element(t, mat, mode, out_idx, &sum);
        const size_t offset = lmmc_tensor_nd_offset(out->ndim, out->strides, out_idx);
        LMMC_REAL_SET(&out->data[offset], &sum);
        LMMC_REAL_CLEAR(&sum);
    } while (lmmc_tensor_nd_next_index(out->ndim, out->dims, out_idx));
}

lmmc_status_t lmmc_tensor_nd_mode_n_product(const lmmc_tensor_nd_t* t, const lmmc_mat_t* mat,
    size_t mode, lmmc_tensor_nd_t* out) {
    lmmc_tensor_nd_t input_snapshot;
    lmmc_tensor_nd_t result = {0};
    size_t out_dims[LMMC_TENSOR_MAX_NDIM];
    lmmc_status_t st = lmmc_tensor_nd_validate_mode_product(t, mat, mode, out);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    for (size_t i = 0; i < t->ndim; ++i) {
        out_dims[i] = i == mode ? mat->rows : t->dims[i];
    }
    if (out == t) {
        input_snapshot = *t;
        t = &input_snapshot;
    }
    st = lmmc_tensor_nd_allocate_result(t->ndim, out_dims, &result);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    lmmc_tensor_nd_execute_mode_product(t, mat, mode, &result);
    *out = result;
    return LMMC_STATUS_OK;
}
