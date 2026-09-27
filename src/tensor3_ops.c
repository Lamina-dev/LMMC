/**
 * @file tensor3_ops.c
 * @brief 三阶张量的逐元素运算、范数与归约。
 */
#include "internal/tensor3_internal.h"

lmmc_status_t lmmc_tensor3_fill(lmmc_tensor3_t* tensor, lmmc_real_t value) {
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    if (lmmc_tensor3_validate(tensor) != LMMC_STATUS_OK) {
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

lmmc_status_t lmmc_tensor3_norm_fro(const lmmc_tensor3_t* tensor, lmmc_real_t* out_norm) {
    lmmc_scaled_sumsq_t acc;
    if (lmmc_tensor3_validate(tensor) != LMMC_STATUS_OK || out_norm == NULL) {
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

lmmc_status_t lmmc_tensor3_add(const lmmc_tensor3_t* a, const lmmc_tensor3_t* b, lmmc_tensor3_t* out_tensor) {
    size_t i = 0, j = 0, k = 0;
    lmmc_status_t st = lmmc_tensor3_validate_binary(a, b, out_tensor);
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

lmmc_status_t lmmc_tensor3_sub(const lmmc_tensor3_t* a, const lmmc_tensor3_t* b, lmmc_tensor3_t* out_tensor) {
    size_t i = 0, j = 0, k = 0;
    lmmc_status_t st = lmmc_tensor3_validate_binary(a, b, out_tensor);
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

lmmc_status_t lmmc_tensor3_mul(const lmmc_tensor3_t* a, const lmmc_tensor3_t* b, lmmc_tensor3_t* out_tensor) {
    size_t i = 0, j = 0, k = 0;
    lmmc_status_t st = lmmc_tensor3_validate_binary(a, b, out_tensor);
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

lmmc_status_t lmmc_tensor3_div(const lmmc_tensor3_t* a, const lmmc_tensor3_t* b, lmmc_tensor3_t* out_tensor) {
    size_t i = 0, j = 0, k = 0;
    lmmc_status_t st = lmmc_tensor3_validate_binary(a, b, out_tensor);
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

lmmc_status_t lmmc_tensor3_scale(const lmmc_tensor3_t* tensor, lmmc_real_t alpha, lmmc_tensor3_t* out_tensor) {
    size_t i = 0, j = 0, k = 0;
    lmmc_real_t vr;
    LMMC_REAL_INIT(&vr);
    if (lmmc_tensor3_validate(tensor) != LMMC_STATUS_OK || lmmc_tensor3_validate(out_tensor) != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&vr);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_tensor3_same_shape(tensor, out_tensor)) {
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

lmmc_status_t lmmc_tensor3_sum(const lmmc_tensor3_t* tensor, lmmc_real_t* out_sum) {
    size_t i = 0, j = 0, k = 0;
    lmmc_real_t sum, tmp_sum;
    LMMC_REAL_INIT(&sum);
    LMMC_REAL_INIT(&tmp_sum);
    if (lmmc_tensor3_validate(tensor) != LMMC_STATUS_OK || out_sum == NULL) {
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

lmmc_status_t lmmc_tensor3_max(const lmmc_tensor3_t* tensor, lmmc_real_t* out_max) {
    size_t i = 0, j = 0, k = 0;
    lmmc_real_t max_v;
    LMMC_REAL_INIT(&max_v);
    if (lmmc_tensor3_validate(tensor) != LMMC_STATUS_OK || out_max == NULL) {
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

lmmc_status_t lmmc_tensor3_min(const lmmc_tensor3_t* tensor, lmmc_real_t* out_min) {
    size_t i = 0, j = 0, k = 0;
    lmmc_real_t min_v;
    LMMC_REAL_INIT(&min_v);
    if (lmmc_tensor3_validate(tensor) != LMMC_STATUS_OK || out_min == NULL) {
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

static lmmc_status_t lmmc_tensor3_validate_axis_sum(
    const lmmc_tensor3_t* tensor, size_t axis, const lmmc_mat_t* out_matrix) {
    if (lmmc_tensor3_validate(tensor) != LMMC_STATUS_OK ||
        !lmmc_mat_descriptor_is_valid(out_matrix)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (axis > 2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    const size_t rows = axis == 0 ? tensor->dim1 : tensor->dim0;
    const size_t cols = axis == 2 ? tensor->dim1 : tensor->dim2;
    if (out_matrix->rows != rows || out_matrix->cols != cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_tensor3_sum_line(
    const lmmc_real_t* data, size_t count, size_t stride,
    lmmc_real_t* sum, lmmc_real_t* tmp_sum) {
    LMMC_REAL_SET_D(sum, 0.0);
    for (size_t i = 0; i < count; ++i) {
        const lmmc_real_t* v = &data[i * stride];
        if (!lmmc_is_finite(v)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        LMMC_REAL_ADD(tmp_sum, sum, v);
        LMMC_REAL_SET(sum, tmp_sum);
        if (!lmmc_is_finite(sum)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor3_sum_axis(
    const lmmc_tensor3_t* tensor, size_t axis, lmmc_mat_t* out_matrix) {
    lmmc_real_t sum, tmp_sum;
    LMMC_REAL_INIT(&sum);
    LMMC_REAL_INIT(&tmp_sum);
    lmmc_status_t st = lmmc_tensor3_validate_axis_sum(tensor, axis, out_matrix);
    if (st == LMMC_STATUS_OK) {
        const size_t dims[3] = {tensor->dim0, tensor->dim1, tensor->dim2};
        const size_t strides[3] = {tensor->stride0, tensor->stride1, tensor->stride2};
        const size_t row_stride = axis == 0 ? tensor->stride1 : tensor->stride0;
        const size_t col_stride = axis == 2 ? tensor->stride1 : tensor->stride2;
        for (size_t row = 0; row < out_matrix->rows; ++row) {
            for (size_t col = 0; col < out_matrix->cols; ++col) {
                const size_t offset = row * row_stride + col * col_stride;
                st = lmmc_tensor3_sum_line(
                    &tensor->data[offset], dims[axis], strides[axis], &sum, &tmp_sum);
                if (st != LMMC_STATUS_OK) {
                    goto done;
                }
                LMMC_REAL_SET(&out_matrix->data[row * out_matrix->stride + col], &sum);
            }
        }
    }
done:
    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_sum);
    return st;
}
