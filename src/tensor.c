/**
 * @file tensor.c
 * @brief N-D 张量结构与基本运算实现（含向后兼容的三阶接口）。
 */
#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/tensor.h"


static lmmc_status_t lmmc_tensor_validate(const lmmc_tensor_t* tensor) {
    if (tensor == NULL || tensor->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (tensor->dim0 == 0 || tensor->dim1 == 0 || tensor->dim2 == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (tensor->stride0 == 0 || tensor->stride1 == 0 || tensor->stride2 == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}


static int lmmc_tensor_same_shape(const lmmc_tensor_t* a, const lmmc_tensor_t* b) {
    return a->dim0 == b->dim0 && a->dim1 == b->dim1 && a->dim2 == b->dim2;
}

static lmmc_status_t lmmc_tensor_validate_binary(
    const lmmc_tensor_t* a,
    const lmmc_tensor_t* b,
    const lmmc_tensor_t* out_tensor
) {
    if (lmmc_tensor_validate(a) != LMMC_STATUS_OK ||
        lmmc_tensor_validate(b) != LMMC_STATUS_OK ||
        lmmc_tensor_validate(out_tensor) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_tensor_same_shape(a, b) || !lmmc_tensor_same_shape(a, out_tensor)) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_mat_validate(const lmmc_mat_t* mat) {
    return lmmc_mat_descriptor_is_valid(mat)
        ? LMMC_STATUS_OK
        : LMMC_STATUS_INVALID_ARGUMENT;
}

lmmc_status_t lmmc_tensor3_create(size_t dim0, size_t dim1, size_t dim2, lmmc_tensor_t* out_tensor) {
    lmmc_tensor_nd_t nd = {0};
    size_t dims[3];
    lmmc_status_t st;

    if (out_tensor == NULL || dim0 == 0 || dim1 == 0 || dim2 == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    dims[0] = dim0;
    dims[1] = dim1;
    dims[2] = dim2;
    st = lmmc_tensor_create(3, dims, &nd);
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
    lmmc_tensor_t* out_tensor
) {
    lmmc_tensor_t candidate = {0};
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

    if (lmmc_tensor_validate(&candidate) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    *out_tensor = candidate;
    return LMMC_STATUS_OK;
}

void lmmc_tensor_destroy(lmmc_tensor_t* tensor) {
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
        lmmc_free(tensor->data);
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

lmmc_status_t lmmc_tensor_fill(lmmc_tensor_t* tensor, lmmc_real_t value) {
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                LMMC_REAL_SET(&tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2], &value);
            }
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_set(lmmc_tensor_t* tensor, size_t i, size_t j, size_t k, lmmc_real_t value) {
    lmmc_tensor_nd_t nd;
    size_t idx[3];

    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (i >= tensor->dim0 || j >= tensor->dim1 || k >= tensor->dim2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    nd.ndim = 3;
    nd.dims[0] = tensor->dim0;
    nd.dims[1] = tensor->dim1;
    nd.dims[2] = tensor->dim2;
    nd.strides[0] = tensor->stride0;
    nd.strides[1] = tensor->stride1;
    nd.strides[2] = tensor->stride2;
    nd.data = tensor->data;
    nd.owns_data = 0;

    idx[0] = i;
    idx[1] = j;
    idx[2] = k;
    return lmmc_tensor_set_nd(&nd, idx, value);
}

lmmc_status_t lmmc_tensor_get(const lmmc_tensor_t* tensor, size_t i, size_t j, size_t k, lmmc_real_t* out_value) {
    lmmc_tensor_nd_t nd;
    size_t idx[3];

    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (i >= tensor->dim0 || j >= tensor->dim1 || k >= tensor->dim2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    nd.ndim = 3;
    nd.dims[0] = tensor->dim0;
    nd.dims[1] = tensor->dim1;
    nd.dims[2] = tensor->dim2;
    nd.strides[0] = tensor->stride0;
    nd.strides[1] = tensor->stride1;
    nd.strides[2] = tensor->stride2;
    nd.data = tensor->data;
    nd.owns_data = 0;

    idx[0] = i;
    idx[1] = j;
    idx[2] = k;
    return lmmc_tensor_get_nd(&nd, idx, out_value);
}

lmmc_status_t lmmc_tensor_norm_fro(const lmmc_tensor_t* tensor, lmmc_real_t* out_norm) {
    lmmc_scaled_sumsq_t acc;
    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || out_norm == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_scaled_sumsq_init(&acc);
    for (size_t i = 0; i < tensor->dim0; ++i) {
        for (size_t j = 0; j < tensor->dim1; ++j) {
            for (size_t k = 0; k < tensor->dim2; ++k) {
                lmmc_scaled_sumsq_add(
                    &acc,
                    tensor->data[
                        i * tensor->stride0 +
                        j * tensor->stride1 +
                        k * tensor->stride2]);
            }
        }
    }
    *out_norm = lmmc_scaled_sumsq_norm(&acc);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_add(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor) {
    size_t i = 0, j = 0, k = 0;
    lmmc_status_t st = lmmc_tensor_validate_binary(a, b, out_tensor);
    lmmc_real_t vr;
    LMMC_REAL_INIT(&vr);

    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&vr);
        return st;
    }

    for (i = 0; i < a->dim0; ++i) {
        for (j = 0; j < a->dim1; ++j) {
            for (k = 0; k < a->dim2; ++k) {
                lmmc_real_t* va = &a->data[i * a->stride0 + j * a->stride1 + k * a->stride2];
                lmmc_real_t* vb = &b->data[i * b->stride0 + j * b->stride1 + k * b->stride2];
                if (!lmmc_is_finite(va) || !lmmc_is_finite(vb)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_ADD(&vr, va, vb);
                if (!lmmc_is_finite(&vr)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_SET(&out_tensor->data[i * out_tensor->stride0 + j * out_tensor->stride1 + k * out_tensor->stride2], &vr);
            }
        }
    }
    LMMC_REAL_CLEAR(&vr);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_sub(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor) {
    size_t i = 0, j = 0, k = 0;
    lmmc_status_t st = lmmc_tensor_validate_binary(a, b, out_tensor);
    lmmc_real_t vr;
    LMMC_REAL_INIT(&vr);

    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&vr);
        return st;
    }

    for (i = 0; i < a->dim0; ++i) {
        for (j = 0; j < a->dim1; ++j) {
            for (k = 0; k < a->dim2; ++k) {
                lmmc_real_t* va = &a->data[i * a->stride0 + j * a->stride1 + k * a->stride2];
                lmmc_real_t* vb = &b->data[i * b->stride0 + j * b->stride1 + k * b->stride2];
                if (!lmmc_is_finite(va) || !lmmc_is_finite(vb)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_SUB(&vr, va, vb);
                if (!lmmc_is_finite(&vr)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_SET(&out_tensor->data[i * out_tensor->stride0 + j * out_tensor->stride1 + k * out_tensor->stride2], &vr);
            }
        }
    }
    LMMC_REAL_CLEAR(&vr);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_mul(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor) {
    size_t i = 0, j = 0, k = 0;
    lmmc_status_t st = lmmc_tensor_validate_binary(a, b, out_tensor);
    lmmc_real_t vr;
    LMMC_REAL_INIT(&vr);

    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&vr);
        return st;
    }

    for (i = 0; i < a->dim0; ++i) {
        for (j = 0; j < a->dim1; ++j) {
            for (k = 0; k < a->dim2; ++k) {
                lmmc_real_t* va = &a->data[i * a->stride0 + j * a->stride1 + k * a->stride2];
                lmmc_real_t* vb = &b->data[i * b->stride0 + j * b->stride1 + k * b->stride2];
                if (!lmmc_is_finite(va) || !lmmc_is_finite(vb)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_MUL(&vr, va, vb);
                if (!lmmc_is_finite(&vr)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_SET(&out_tensor->data[i * out_tensor->stride0 + j * out_tensor->stride1 + k * out_tensor->stride2], &vr);
            }
        }
    }
    LMMC_REAL_CLEAR(&vr);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_div(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor) {
    size_t i = 0, j = 0, k = 0;
    lmmc_status_t st = lmmc_tensor_validate_binary(a, b, out_tensor);
    lmmc_real_t vr, zero;
    LMMC_REAL_INIT(&vr);
    LMMC_REAL_INIT(&zero);

    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&vr);
        LMMC_REAL_CLEAR(&zero);
        return st;
    }
    LMMC_REAL_SET_D(&zero, 0.0);

    for (i = 0; i < a->dim0; ++i) {
        for (j = 0; j < a->dim1; ++j) {
            for (k = 0; k < a->dim2; ++k) {
                lmmc_real_t* va = &a->data[i * a->stride0 + j * a->stride1 + k * a->stride2];
                lmmc_real_t* vb = &b->data[i * b->stride0 + j * b->stride1 + k * b->stride2];
                if (!lmmc_is_finite(va) || !lmmc_is_finite(vb) || LMMC_REAL_CMP(vb, &zero) == 0) {
                    LMMC_REAL_CLEAR(&vr);
                    LMMC_REAL_CLEAR(&zero);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_DIV(&vr, va, vb);
                if (!lmmc_is_finite(&vr)) {
                    LMMC_REAL_CLEAR(&vr);
                    LMMC_REAL_CLEAR(&zero);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_SET(&out_tensor->data[i * out_tensor->stride0 + j * out_tensor->stride1 + k * out_tensor->stride2], &vr);
            }
        }
    }
    LMMC_REAL_CLEAR(&vr);
    LMMC_REAL_CLEAR(&zero);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_scale(const lmmc_tensor_t* tensor, lmmc_real_t alpha, lmmc_tensor_t* out_tensor) {
    size_t i = 0, j = 0, k = 0;
    lmmc_real_t vr;
    LMMC_REAL_INIT(&vr);
    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || lmmc_tensor_validate(out_tensor) != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&vr);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_tensor_same_shape(tensor, out_tensor)) {
        LMMC_REAL_CLEAR(&vr);
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    if (!lmmc_is_finite(&alpha)) {
        LMMC_REAL_CLEAR(&vr);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                lmmc_real_t* v = &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2];
                if (!lmmc_is_finite(v)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_MUL(&vr, v, &alpha);
                if (!lmmc_is_finite(&vr)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_SET(&out_tensor->data[i * out_tensor->stride0 + j * out_tensor->stride1 + k * out_tensor->stride2], &vr);
            }
        }
    }
    LMMC_REAL_CLEAR(&vr);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_sum(const lmmc_tensor_t* tensor, lmmc_real_t* out_sum) {
    size_t i = 0, j = 0, k = 0;
    lmmc_real_t sum, tmp_sum;
    LMMC_REAL_INIT(&sum);
    LMMC_REAL_INIT(&tmp_sum);
    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || out_sum == NULL) {
        LMMC_REAL_CLEAR(&sum);
        LMMC_REAL_CLEAR(&tmp_sum);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_SET_D(&sum, 0.0);

    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                lmmc_real_t* v = &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2];
                if (!lmmc_is_finite(v)) {
                    LMMC_REAL_CLEAR(&sum);
                    LMMC_REAL_CLEAR(&tmp_sum);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_ADD(&tmp_sum, &sum, v);
                LMMC_REAL_SET(&sum, &tmp_sum);
                if (!lmmc_is_finite(&sum)) {
                    LMMC_REAL_CLEAR(&sum);
                    LMMC_REAL_CLEAR(&tmp_sum);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
            }
        }
    }
    LMMC_REAL_SET(out_sum, &sum);
    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_max(const lmmc_tensor_t* tensor, lmmc_real_t* out_max) {
    size_t i = 0, j = 0, k = 0;
    lmmc_real_t max_v;
    LMMC_REAL_INIT(&max_v);
    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || out_max == NULL) {
        LMMC_REAL_CLEAR(&max_v);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_SET(&max_v, &tensor->data[0]);
    if (!lmmc_is_finite(&max_v)) {
        LMMC_REAL_CLEAR(&max_v);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                lmmc_real_t* v = &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2];
                if (!lmmc_is_finite(v)) {
                    LMMC_REAL_CLEAR(&max_v);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                if (LMMC_REAL_CMP(v, &max_v) > 0) {
                    LMMC_REAL_SET(&max_v, v);
                }
            }
        }
    }
    LMMC_REAL_SET(out_max, &max_v);
    LMMC_REAL_CLEAR(&max_v);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_min(const lmmc_tensor_t* tensor, lmmc_real_t* out_min) {
    size_t i = 0, j = 0, k = 0;
    lmmc_real_t min_v;
    LMMC_REAL_INIT(&min_v);
    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || out_min == NULL) {
        LMMC_REAL_CLEAR(&min_v);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_SET(&min_v, &tensor->data[0]);
    if (!lmmc_is_finite(&min_v)) {
        LMMC_REAL_CLEAR(&min_v);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                lmmc_real_t* v = &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2];
                if (!lmmc_is_finite(v)) {
                    LMMC_REAL_CLEAR(&min_v);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                if (LMMC_REAL_CMP(v, &min_v) < 0) {
                    LMMC_REAL_SET(&min_v, v);
                }
            }
        }
    }
    LMMC_REAL_SET(out_min, &min_v);
    LMMC_REAL_CLEAR(&min_v);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_sum_axis(const lmmc_tensor_t* tensor, size_t axis, lmmc_mat_t* out_matrix) {
    size_t i = 0, j = 0, k = 0;
    lmmc_real_t sum, tmp_sum;
    LMMC_REAL_INIT(&sum);
    LMMC_REAL_INIT(&tmp_sum);

    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || lmmc_mat_validate(out_matrix) != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&sum);
        LMMC_REAL_CLEAR(&tmp_sum);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (axis > 2) {
        LMMC_REAL_CLEAR(&sum);
        LMMC_REAL_CLEAR(&tmp_sum);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (axis == 0 && (out_matrix->rows != tensor->dim1 || out_matrix->cols != tensor->dim2)) {
        LMMC_REAL_CLEAR(&sum);
        LMMC_REAL_CLEAR(&tmp_sum);
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    if (axis == 1 && (out_matrix->rows != tensor->dim0 || out_matrix->cols != tensor->dim2)) {
        LMMC_REAL_CLEAR(&sum);
        LMMC_REAL_CLEAR(&tmp_sum);
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    if (axis == 2 && (out_matrix->rows != tensor->dim0 || out_matrix->cols != tensor->dim1)) {
        LMMC_REAL_CLEAR(&sum);
        LMMC_REAL_CLEAR(&tmp_sum);
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    if (axis == 0) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                LMMC_REAL_SET_D(&sum, 0.0);
                for (i = 0; i < tensor->dim0; ++i) {
                    lmmc_real_t* v = &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2];
                    if (!lmmc_is_finite(v)) {
                        LMMC_REAL_CLEAR(&sum);
                        LMMC_REAL_CLEAR(&tmp_sum);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }
                    LMMC_REAL_ADD(&tmp_sum, &sum, v);
                    LMMC_REAL_SET(&sum, &tmp_sum);
                    if (!lmmc_is_finite(&sum)) {
                        LMMC_REAL_CLEAR(&sum);
                        LMMC_REAL_CLEAR(&tmp_sum);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }
                }
                LMMC_REAL_SET(&out_matrix->data[j * out_matrix->stride + k], &sum);
            }
        }
    } else if (axis == 1) {
        for (i = 0; i < tensor->dim0; ++i) {
            for (k = 0; k < tensor->dim2; ++k) {
                LMMC_REAL_SET_D(&sum, 0.0);
                for (j = 0; j < tensor->dim1; ++j) {
                    lmmc_real_t* v = &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2];
                    if (!lmmc_is_finite(v)) {
                        LMMC_REAL_CLEAR(&sum);
                        LMMC_REAL_CLEAR(&tmp_sum);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }
                    LMMC_REAL_ADD(&tmp_sum, &sum, v);
                    LMMC_REAL_SET(&sum, &tmp_sum);
                    if (!lmmc_is_finite(&sum)) {
                        LMMC_REAL_CLEAR(&sum);
                        LMMC_REAL_CLEAR(&tmp_sum);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }
                }
                LMMC_REAL_SET(&out_matrix->data[i * out_matrix->stride + k], &sum);
            }
        }
    } else {
        for (i = 0; i < tensor->dim0; ++i) {
            for (j = 0; j < tensor->dim1; ++j) {
                LMMC_REAL_SET_D(&sum, 0.0);
                for (k = 0; k < tensor->dim2; ++k) {
                    lmmc_real_t* v = &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2];
                    if (!lmmc_is_finite(v)) {
                        LMMC_REAL_CLEAR(&sum);
                        LMMC_REAL_CLEAR(&tmp_sum);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }
                    LMMC_REAL_ADD(&tmp_sum, &sum, v);
                    LMMC_REAL_SET(&sum, &tmp_sum);
                    if (!lmmc_is_finite(&sum)) {
                        LMMC_REAL_CLEAR(&sum);
                        LMMC_REAL_CLEAR(&tmp_sum);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }
                }
                LMMC_REAL_SET(&out_matrix->data[i * out_matrix->stride + j], &sum);
            }
        }
    }

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_reshape_view(
    const lmmc_tensor_t* tensor,
    size_t new_dim0,
    size_t new_dim1,
    size_t new_dim2,
    lmmc_tensor_t* out_view
) {
    lmmc_tensor_nd_t nd_src;
    lmmc_tensor_nd_t nd_view = {0};
    size_t new_dims[3];
    lmmc_status_t st;

    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || out_view == NULL ||
        new_dim0 == 0 || new_dim1 == 0 || new_dim2 == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Construct N-D view of source */
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
        /* Map INVALID_ARGUMENT to DIMENSION_MISMATCH for total-size mismatch (backward compat) */
        size_t old_total = tensor->dim0 * tensor->dim1 * tensor->dim2;
        size_t new_total = new_dim0 * new_dim1 * new_dim2;
        if (old_total != new_total) {
            return LMMC_STATUS_DIMENSION_MISMATCH;
        }
        return st;
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

lmmc_status_t lmmc_tensor_slice_view(
    const lmmc_tensor_t* tensor,
    size_t begin0,
    size_t end0,
    size_t begin1,
    size_t end1,
    size_t begin2,
    size_t end2,
    lmmc_tensor_t* out_view
) {
    size_t off0 = 0;
    size_t off1 = 0;
    size_t off2 = 0;
    size_t offset = 0;

    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || out_view == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (begin0 >= end0 || begin1 >= end1 || begin2 >= end2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (end0 > tensor->dim0 || end1 > tensor->dim1 || end2 > tensor->dim2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_safe_mul_size(begin0, tensor->stride0, &off0) ||
        !lmmc_safe_mul_size(begin1, tensor->stride1, &off1) ||
        !lmmc_safe_mul_size(begin2, tensor->stride2, &off2)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    offset = off0 + off1;
    if (offset < off0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    offset += off2;
    if (offset < off2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    out_view->dim0 = end0 - begin0;
    out_view->dim1 = end1 - begin1;
    out_view->dim2 = end2 - begin2;
    out_view->stride0 = tensor->stride0;
    out_view->stride1 = tensor->stride1;
    out_view->stride2 = tensor->stride2;
    out_view->data = tensor->data + offset;
    out_view->owns_data = 0;
    return LMMC_STATUS_OK;
}

/**
 * @brief Check if an N-D tensor is contiguous (row-major).
 */
static int lmmc_tensor_nd_is_contiguous(const lmmc_tensor_nd_t* t) {
    size_t i;
    size_t expected_stride;
    if (t == NULL || t->ndim == 0) return 0;

    expected_stride = 1;
    for (i = t->ndim; i > 0; --i) {
        if (t->strides[i - 1] != expected_stride) return 0;
        expected_stride *= t->dims[i - 1];
    }
    return 1;
}

lmmc_status_t lmmc_tensor_create(size_t ndim, const size_t* dims, lmmc_tensor_nd_t* out) {
    size_t total = 1;
    size_t bytes = 0;
    size_t i;
    lmmc_real_t* data = NULL;

    if (out == NULL || dims == NULL || ndim == 0 || ndim > LMMC_TENSOR_MAX_NDIM) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < ndim; ++i) {
        if (dims[i] == 0) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        /* Overflow check */
        if (total > ((size_t)-1) / dims[i]) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        total *= dims[i];
    }

    /* Overflow check for bytes */
    if (total > ((size_t)-1) / sizeof(lmmc_real_t)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    bytes = total * sizeof(lmmc_real_t);

    data = (lmmc_real_t*)lmmc_alloc(bytes);
    if (data == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, bytes);

    for (i = 0; i < total; ++i) {
        LMMC_REAL_INIT(&data[i]);
    }

    out->ndim = ndim;
    memset(out->dims, 0, sizeof(out->dims));
    memset(out->strides, 0, sizeof(out->strides));

    for (i = 0; i < ndim; ++i) {
        out->dims[i] = dims[i];
    }

    /* Row-major strides */
    out->strides[ndim - 1] = 1;
    for (i = ndim - 1; i > 0; --i) {
        out->strides[i - 1] = out->strides[i] * out->dims[i];
    }

    out->data = data;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_get_nd(const lmmc_tensor_nd_t* t, const size_t* idx, lmmc_real_t* out) {
    size_t i;
    size_t offset = 0;

    if (t == NULL || t->data == NULL || idx == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (t->ndim == 0 || t->ndim > LMMC_TENSOR_MAX_NDIM) {
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

lmmc_status_t lmmc_tensor_set_nd(lmmc_tensor_nd_t* t, const size_t* idx, lmmc_real_t value) {
    size_t i;
    size_t offset = 0;

    if (t == NULL || t->data == NULL || idx == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (t->ndim == 0 || t->ndim > LMMC_TENSOR_MAX_NDIM) {
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

lmmc_status_t lmmc_tensor_permute(const lmmc_tensor_nd_t* in, const size_t* perm, lmmc_tensor_nd_t* out) {
    size_t i, ndim;
    size_t new_dims[LMMC_TENSOR_MAX_NDIM];
    int seen[LMMC_TENSOR_MAX_NDIM] = {0};
    size_t total = 1;
    size_t bytes;
    lmmc_real_t* data = NULL;
    size_t idx[LMMC_TENSOR_MAX_NDIM] = {0};

    if (in == NULL || in->data == NULL || perm == NULL || out == NULL) {
        if (out != NULL) { out->data = NULL; out->owns_data = 0; }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    ndim = in->ndim;
    if (ndim == 0 || ndim > LMMC_TENSOR_MAX_NDIM) {
        out->data = NULL; out->owns_data = 0;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Validate permutation */
    for (i = 0; i < ndim; ++i) {
        if (perm[i] >= ndim || seen[perm[i]]) {
            out->data = NULL; out->owns_data = 0;
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        seen[perm[i]] = 1;
        new_dims[i] = in->dims[perm[i]];
    }

    /* Compute total elements */
    for (i = 0; i < ndim; ++i) {
        total *= new_dims[i];
    }
    bytes = total * sizeof(lmmc_real_t);

    data = (lmmc_real_t*)lmmc_alloc(bytes);
    if (data == NULL) {
        out->data = NULL; out->owns_data = 0;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, bytes);
    for (i = 0; i < total; ++i) {
        LMMC_REAL_INIT(&data[i]);
    }

    /* Set up output tensor metadata */
    out->ndim = ndim;
    memset(out->dims, 0, sizeof(out->dims));
    memset(out->strides, 0, sizeof(out->strides));
    for (i = 0; i < ndim; ++i) {
        out->dims[i] = new_dims[i];
    }
    out->strides[ndim - 1] = 1;
    for (i = ndim - 1; i > 0; --i) {
        out->strides[i - 1] = out->strides[i] * out->dims[i];
    }
    out->data = data;
    out->owns_data = 1;

    /* Copy data with permuted indexing */
    /* Iterate over all elements of the input tensor */
    memset(idx, 0, sizeof(idx));
    for (;;) {
        /* Compute source offset */
        size_t src_offset = 0;
        size_t dst_offset = 0;
        size_t perm_idx[LMMC_TENSOR_MAX_NDIM];

        for (i = 0; i < ndim; ++i) {
            src_offset += idx[i] * in->strides[i];
        }
        /* Compute permuted index: out[perm_idx] where perm_idx[j] = idx[perm[j]] */
        for (i = 0; i < ndim; ++i) {
            perm_idx[i] = idx[perm[i]];
        }
        for (i = 0; i < ndim; ++i) {
            dst_offset += perm_idx[i] * out->strides[i];
        }

        LMMC_REAL_SET(&out->data[dst_offset], &in->data[src_offset]);

        /* Increment multi-index (input order) */
        i = ndim;
        while (i > 0) {
            --i;
            idx[i]++;
            if (idx[i] < in->dims[i]) break;
            idx[i] = 0;
            if (i == 0) goto permute_done;
        }
    }
permute_done:
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_contract(const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    const size_t* axes_a, const size_t* axes_b, size_t naxes, lmmc_tensor_nd_t* out) {

    size_t i, j, k;
    size_t out_ndim;
    size_t out_dims[LMMC_TENSOR_MAX_NDIM];
    /* Track which axes are contracted */
    int a_contracted[LMMC_TENSOR_MAX_NDIM] = {0};
    int b_contracted[LMMC_TENSOR_MAX_NDIM] = {0};
    size_t a_free_axes[LMMC_TENSOR_MAX_NDIM];
    size_t b_free_axes[LMMC_TENSOR_MAX_NDIM];
    size_t n_a_free = 0, n_b_free = 0;
    size_t total_out = 1;
    size_t bytes;
    lmmc_real_t* data = NULL;

    if (a == NULL || a->data == NULL || b == NULL || b->data == NULL || out == NULL) {
        if (out != NULL) { out->data = NULL; out->owns_data = 0; }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (naxes > a->ndim || naxes > b->ndim) {
        out->data = NULL; out->owns_data = 0;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (naxes > 0 && (axes_a == NULL || axes_b == NULL)) {
        out->data = NULL; out->owns_data = 0;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Validate axes */
    for (i = 0; i < naxes; ++i) {
        if (axes_a[i] >= a->ndim || axes_b[i] >= b->ndim) {
            out->data = NULL; out->owns_data = 0;
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        if (a_contracted[axes_a[i]] || b_contracted[axes_b[i]]) {
            out->data = NULL; out->owns_data = 0;
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        if (a->dims[axes_a[i]] != b->dims[axes_b[i]]) {
            out->data = NULL; out->owns_data = 0;
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        a_contracted[axes_a[i]] = 1;
        b_contracted[axes_b[i]] = 1;
    }

    /* Determine free axes */
    for (i = 0; i < a->ndim; ++i) {
        if (!a_contracted[i]) {
            a_free_axes[n_a_free++] = i;
        }
    }
    for (i = 0; i < b->ndim; ++i) {
        if (!b_contracted[i]) {
            b_free_axes[n_b_free++] = i;
        }
    }

    out_ndim = n_a_free + n_b_free;
    if (out_ndim == 0) out_ndim = 1; /* scalar result stored as 1-D tensor of size 1 */
    if (out_ndim > LMMC_TENSOR_MAX_NDIM) {
        out->data = NULL; out->owns_data = 0;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Build output dims */
    j = 0;
    for (i = 0; i < n_a_free; ++i) {
        out_dims[j++] = a->dims[a_free_axes[i]];
    }
    for (i = 0; i < n_b_free; ++i) {
        out_dims[j++] = b->dims[b_free_axes[i]];
    }
    if (n_a_free + n_b_free == 0) {
        out_dims[0] = 1;
    }

    /* Allocate output */
    total_out = 1;
    for (i = 0; i < out_ndim; ++i) {
        total_out *= out_dims[i];
    }
    bytes = total_out * sizeof(lmmc_real_t);
    data = (lmmc_real_t*)lmmc_alloc(bytes);
    if (data == NULL) {
        out->data = NULL; out->owns_data = 0;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, bytes);
    for (i = 0; i < total_out; ++i) {
        LMMC_REAL_INIT(&data[i]);
    }

    out->ndim = out_ndim;
    memset(out->dims, 0, sizeof(out->dims));
    memset(out->strides, 0, sizeof(out->strides));
    for (i = 0; i < out_ndim; ++i) {
        out->dims[i] = out_dims[i];
    }
    out->strides[out_ndim - 1] = 1;
    for (i = out_ndim - 1; i > 0; --i) {
        out->strides[i - 1] = out->strides[i] * out->dims[i];
    }
    out->data = data;
    out->owns_data = 1;

    /* Perform contraction via nested loops */
    {
        /* Compute contraction dimension product */
        size_t contract_size = 1;
        for (i = 0; i < naxes; ++i) {
            contract_size *= a->dims[axes_a[i]];
        }

        /* Iterate over all output elements */
        size_t out_idx[LMMC_TENSOR_MAX_NDIM] = {0};
        for (;;) {
            /* For this output element, sum over contracted indices */
            size_t contract_idx[LMMC_TENSOR_MAX_NDIM] = {0}; /* indices into contracted dims */
            lmmc_real_t sum;
            LMMC_REAL_INIT(&sum);
            LMMC_REAL_SET_D(&sum, 0.0);

            for (k = 0; k < contract_size; ++k) {
                /* Compute a's full index */
                size_t a_full_idx[LMMC_TENSOR_MAX_NDIM] = {0};
                size_t b_full_idx[LMMC_TENSOR_MAX_NDIM] = {0};
                size_t a_offset = 0, b_offset = 0;
                lmmc_real_t va, vb, prod, new_sum;
                LMMC_REAL_INIT(&va);
                LMMC_REAL_INIT(&vb);
                LMMC_REAL_INIT(&prod);
                LMMC_REAL_INIT(&new_sum);

                /* Fill a's free indices from output */
                for (i = 0; i < n_a_free; ++i) {
                    a_full_idx[a_free_axes[i]] = out_idx[i];
                }
                /* Fill a's contracted indices */
                for (i = 0; i < naxes; ++i) {
                    a_full_idx[axes_a[i]] = contract_idx[i];
                }
                /* Fill b's free indices from output */
                for (i = 0; i < n_b_free; ++i) {
                    b_full_idx[b_free_axes[i]] = out_idx[n_a_free + i];
                }
                /* Fill b's contracted indices */
                for (i = 0; i < naxes; ++i) {
                    b_full_idx[axes_b[i]] = contract_idx[i];
                }

                /* Compute offsets */
                for (i = 0; i < a->ndim; ++i) {
                    a_offset += a_full_idx[i] * a->strides[i];
                }
                for (i = 0; i < b->ndim; ++i) {
                    b_offset += b_full_idx[i] * b->strides[i];
                }

                LMMC_REAL_SET(&va, &a->data[a_offset]);
                LMMC_REAL_SET(&vb, &b->data[b_offset]);
                LMMC_REAL_MUL(&prod, &va, &vb);
                LMMC_REAL_ADD(&new_sum, &sum, &prod);
                LMMC_REAL_SET(&sum, &new_sum);

                LMMC_REAL_CLEAR(&va);
                LMMC_REAL_CLEAR(&vb);
                LMMC_REAL_CLEAR(&prod);
                LMMC_REAL_CLEAR(&new_sum);

                /* Increment contract_idx */
                if (naxes > 0) {
                    size_t ci = naxes;
                    while (ci > 0) {
                        --ci;
                        contract_idx[ci]++;
                        if (contract_idx[ci] < a->dims[axes_a[ci]]) break;
                        contract_idx[ci] = 0;
                        if (ci == 0) break;
                    }
                }
            }

            /* Write sum to output */
            {
                size_t out_offset = 0;
                for (i = 0; i < out_ndim; ++i) {
                    out_offset += out_idx[i] * out->strides[i];
                }
                LMMC_REAL_SET(&out->data[out_offset], &sum);
            }
            LMMC_REAL_CLEAR(&sum);

            /* Increment out_idx */
            i = out_ndim;
            while (i > 0) {
                --i;
                out_idx[i]++;
                if (out_idx[i] < out->dims[i]) break;
                out_idx[i] = 0;
                if (i == 0) goto contract_done;
            }
        }
    }
contract_done:
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_mode_n_product(const lmmc_tensor_nd_t* t, const lmmc_mat_t* mat,
    size_t mode, lmmc_tensor_nd_t* out) {

    size_t i;
    size_t out_dims[LMMC_TENSOR_MAX_NDIM];
    size_t total_out = 1;
    size_t bytes;
    lmmc_real_t* data = NULL;

    if (t == NULL || t->data == NULL || mat == NULL || mat->data == NULL || out == NULL) {
        if (out != NULL) { out->data = NULL; out->owns_data = 0; }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (t->ndim == 0 || t->ndim > LMMC_TENSOR_MAX_NDIM) {
        out->data = NULL; out->owns_data = 0;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (mode >= t->ndim) {
        out->data = NULL; out->owns_data = 0;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (mat->cols != t->dims[mode]) {
        out->data = NULL; out->owns_data = 0;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (mat->rows == 0 || mat->cols == 0) {
        out->data = NULL; out->owns_data = 0;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Build output dims */
    for (i = 0; i < t->ndim; ++i) {
        if (i == mode) {
            out_dims[i] = mat->rows;
        } else {
            out_dims[i] = t->dims[i];
        }
    }

    /* Allocate output */
    for (i = 0; i < t->ndim; ++i) {
        total_out *= out_dims[i];
    }
    bytes = total_out * sizeof(lmmc_real_t);
    data = (lmmc_real_t*)lmmc_alloc(bytes);
    if (data == NULL) {
        out->data = NULL; out->owns_data = 0;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, bytes);
    for (i = 0; i < total_out; ++i) {
        LMMC_REAL_INIT(&data[i]);
    }

    out->ndim = t->ndim;
    memset(out->dims, 0, sizeof(out->dims));
    memset(out->strides, 0, sizeof(out->strides));
    for (i = 0; i < t->ndim; ++i) {
        out->dims[i] = out_dims[i];
    }
    out->strides[t->ndim - 1] = 1;
    for (i = t->ndim - 1; i > 0; --i) {
        out->strides[i - 1] = out->strides[i] * out->dims[i];
    }
    out->data = data;
    out->owns_data = 1;

    /* Perform mode-n product: out[i0,...,i_{mode},...] = sum_j mat[i_mode, j] * t[i0,...,j,...] */
    {
        size_t out_idx[LMMC_TENSOR_MAX_NDIM] = {0};
        for (;;) {
            /* For this output element, sum over the mode dimension of t */
            lmmc_real_t sum;
            size_t j_dim = t->dims[mode];
            LMMC_REAL_INIT(&sum);
            LMMC_REAL_SET_D(&sum, 0.0);

            for (i = 0; i < j_dim; ++i) {
                /* Compute source index: same as out_idx but with mode dimension = i */
                size_t src_offset = 0;
                size_t d;
                lmmc_real_t mat_val, t_val, prod, new_sum;
                LMMC_REAL_INIT(&mat_val);
                LMMC_REAL_INIT(&t_val);
                LMMC_REAL_INIT(&prod);
                LMMC_REAL_INIT(&new_sum);

                for (d = 0; d < t->ndim; ++d) {
                    if (d == mode) {
                        src_offset += i * t->strides[d];
                    } else {
                        src_offset += out_idx[d] * t->strides[d];
                    }
                }

                /* mat[out_idx[mode], i] */
                LMMC_REAL_SET(&mat_val, &mat->data[out_idx[mode] * mat->stride + i]);
                LMMC_REAL_SET(&t_val, &t->data[src_offset]);
                LMMC_REAL_MUL(&prod, &mat_val, &t_val);
                LMMC_REAL_ADD(&new_sum, &sum, &prod);
                LMMC_REAL_SET(&sum, &new_sum);

                LMMC_REAL_CLEAR(&mat_val);
                LMMC_REAL_CLEAR(&t_val);
                LMMC_REAL_CLEAR(&prod);
                LMMC_REAL_CLEAR(&new_sum);
            }

            /* Write to output */
            {
                size_t out_offset = 0;
                size_t d;
                for (d = 0; d < out->ndim; ++d) {
                    out_offset += out_idx[d] * out->strides[d];
                }
                LMMC_REAL_SET(&out->data[out_offset], &sum);
            }
            LMMC_REAL_CLEAR(&sum);

            /* Increment out_idx */
            {
                size_t dim_i = out->ndim;
                while (dim_i > 0) {
                    --dim_i;
                    out_idx[dim_i]++;
                    if (out_idx[dim_i] < out->dims[dim_i]) break;
                    out_idx[dim_i] = 0;
                    if (dim_i == 0) goto mode_n_done;
                }
            }
        }
    }
mode_n_done:
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_nd_reshape_view(const lmmc_tensor_nd_t* src,
    size_t new_ndim, const size_t* new_dims, lmmc_tensor_nd_t* out_view) {

    size_t i;
    size_t old_total = 1, new_total = 1;

    if (src == NULL || src->data == NULL || new_dims == NULL || out_view == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (new_ndim == 0 || new_ndim > LMMC_TENSOR_MAX_NDIM) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (src->ndim == 0 || src->ndim > LMMC_TENSOR_MAX_NDIM) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < new_ndim; ++i) {
        if (new_dims[i] == 0) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        new_total *= new_dims[i];
    }
    for (i = 0; i < src->ndim; ++i) {
        old_total *= src->dims[i];
    }
    if (old_total != new_total) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Check if source is contiguous (required for reshape view) */
    if (!lmmc_tensor_nd_is_contiguous(src)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Create non-owning view */
    out_view->ndim = new_ndim;
    memset(out_view->dims, 0, sizeof(out_view->dims));
    memset(out_view->strides, 0, sizeof(out_view->strides));
    for (i = 0; i < new_ndim; ++i) {
        out_view->dims[i] = new_dims[i];
    }
    out_view->strides[new_ndim - 1] = 1;
    for (i = new_ndim - 1; i > 0; --i) {
        out_view->strides[i - 1] = out_view->strides[i] * out_view->dims[i];
    }
    out_view->data = src->data;
    out_view->owns_data = 0;
    return LMMC_STATUS_OK;
}

void lmmc_tensor_nd_destroy(lmmc_tensor_nd_t* t) {
    if (t == NULL) return;

    if (t->owns_data && t->data != NULL) {
        size_t total = 1;
        size_t i;
        for (i = 0; i < t->ndim; ++i) {
            total *= t->dims[i];
        }
        for (i = 0; i < total; ++i) {
            LMMC_REAL_CLEAR(&t->data[i]);
        }
        lmmc_free(t->data);
    }

    t->ndim = 0;
    memset(t->dims, 0, sizeof(t->dims));
    memset(t->strides, 0, sizeof(t->strides));
    t->data = NULL;
    t->owns_data = 0;
}
