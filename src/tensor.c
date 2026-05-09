#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/tensor.h"

static int lmmc_mul_overflow_size(size_t a, size_t b, size_t* out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }
    if (a > ((size_t)-1) / b) {
        return 1;
    }
    *out = a * b;
    return 0;
}

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

static int lmmc_is_finite_number(const lmmc_real_t* v) {
    return LMMC_REAL_IS_FINITE(v) ? 1 : 0;
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
    if (mat == NULL || mat->data == NULL || mat->rows == 0 || mat->cols == 0 || mat->stride < mat->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

static int lmmc_tensor_is_contiguous(const lmmc_tensor_t* tensor) {
    if (tensor == NULL) {
        return 0;
    }
    if (tensor->stride2 != 1) {
        return 0;
    }
    if (tensor->stride1 != tensor->dim2) {
        return 0;
    }
    if (tensor->stride0 != tensor->dim1 * tensor->dim2) {
        return 0;
    }
    return 1;
}

lmmc_status_t lmmc_tensor3_create(size_t dim0, size_t dim1, size_t dim2, lmmc_tensor_t* out_tensor) {
    size_t total = 0;
    size_t bytes = 0;
    lmmc_real_t* data = NULL;

    if (out_tensor == NULL || dim0 == 0 || dim1 == 0 || dim2 == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_mul_overflow_size(dim0, dim1, &total) ||
        lmmc_mul_overflow_size(total, dim2, &total) ||
        lmmc_mul_overflow_size(total, sizeof(lmmc_real_t), &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    data = (lmmc_real_t*)lmmc_alloc(bytes);
    if (data == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, bytes);

    for (size_t i = 0; i < total; ++i) {
        LMMC_REAL_INIT(&data[i]);
    }

    out_tensor->dim0 = dim0;
    out_tensor->dim1 = dim1;
    out_tensor->dim2 = dim2;
    out_tensor->stride2 = 1;
    out_tensor->stride1 = dim2;
    out_tensor->stride0 = dim1 * dim2;
    out_tensor->data = data;
    out_tensor->owns_data = 1;
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
        size_t total = tensor->dim0 * tensor->dim1 * tensor->dim2;
        for (size_t i = 0; i < total; ++i) {
            LMMC_REAL_CLEAR(&tensor->data[i]);
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
    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (i >= tensor->dim0 || j >= tensor->dim1 || k >= tensor->dim2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_SET(&tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2], &value);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_get(const lmmc_tensor_t* tensor, size_t i, size_t j, size_t k, lmmc_real_t* out_value) {
    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (i >= tensor->dim0 || j >= tensor->dim1 || k >= tensor->dim2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_SET(out_value, &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2]);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_tensor_norm_fro(const lmmc_tensor_t* tensor, lmmc_real_t* out_norm) {
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    lmmc_real_t sum, tmp_mul, tmp_sum;
    LMMC_REAL_INIT(&sum);
    LMMC_REAL_INIT(&tmp_mul);
    LMMC_REAL_INIT(&tmp_sum);
    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || out_norm == NULL) {
        LMMC_REAL_CLEAR(&sum);
        LMMC_REAL_CLEAR(&tmp_mul);
        LMMC_REAL_CLEAR(&tmp_sum);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_SET_D(&sum, 0.0);

    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                lmmc_real_t* v = &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2];
                LMMC_REAL_MUL(&tmp_mul, v, v);
                LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
                LMMC_REAL_SET(&sum, &tmp_sum);
            }
        }
    }
    LMMC_REAL_SQRT(out_norm, &sum);

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
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
                if (!lmmc_is_finite_number(va) || !lmmc_is_finite_number(vb)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_ADD(&vr, va, vb);
                if (!lmmc_is_finite_number(&vr)) {
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
                if (!lmmc_is_finite_number(va) || !lmmc_is_finite_number(vb)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_SUB(&vr, va, vb);
                if (!lmmc_is_finite_number(&vr)) {
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
                if (!lmmc_is_finite_number(va) || !lmmc_is_finite_number(vb)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_MUL(&vr, va, vb);
                if (!lmmc_is_finite_number(&vr)) {
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
                if (!lmmc_is_finite_number(va) || !lmmc_is_finite_number(vb) || LMMC_REAL_CMP(vb, &zero) == 0) {
                    LMMC_REAL_CLEAR(&vr);
                    LMMC_REAL_CLEAR(&zero);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_DIV(&vr, va, vb);
                if (!lmmc_is_finite_number(&vr)) {
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
    if (!lmmc_is_finite_number(&alpha)) {
        LMMC_REAL_CLEAR(&vr);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                lmmc_real_t* v = &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2];
                if (!lmmc_is_finite_number(v)) {
                    LMMC_REAL_CLEAR(&vr);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_MUL(&vr, v, &alpha);
                if (!lmmc_is_finite_number(&vr)) {
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
                if (!lmmc_is_finite_number(v)) {
                    LMMC_REAL_CLEAR(&sum);
                    LMMC_REAL_CLEAR(&tmp_sum);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_ADD(&tmp_sum, &sum, v);
                LMMC_REAL_SET(&sum, &tmp_sum);
                if (!lmmc_is_finite_number(&sum)) {
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
    if (!lmmc_is_finite_number(&max_v)) {
        LMMC_REAL_CLEAR(&max_v);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                lmmc_real_t* v = &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2];
                if (!lmmc_is_finite_number(v)) {
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
    if (!lmmc_is_finite_number(&min_v)) {
        LMMC_REAL_CLEAR(&min_v);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                lmmc_real_t* v = &tensor->data[i * tensor->stride0 + j * tensor->stride1 + k * tensor->stride2];
                if (!lmmc_is_finite_number(v)) {
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
                    if (!lmmc_is_finite_number(v)) {
                        LMMC_REAL_CLEAR(&sum);
                        LMMC_REAL_CLEAR(&tmp_sum);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }
                    LMMC_REAL_ADD(&tmp_sum, &sum, v);
                    LMMC_REAL_SET(&sum, &tmp_sum);
                    if (!lmmc_is_finite_number(&sum)) {
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
                    if (!lmmc_is_finite_number(v)) {
                        LMMC_REAL_CLEAR(&sum);
                        LMMC_REAL_CLEAR(&tmp_sum);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }
                    LMMC_REAL_ADD(&tmp_sum, &sum, v);
                    LMMC_REAL_SET(&sum, &tmp_sum);
                    if (!lmmc_is_finite_number(&sum)) {
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
                    if (!lmmc_is_finite_number(v)) {
                        LMMC_REAL_CLEAR(&sum);
                        LMMC_REAL_CLEAR(&tmp_sum);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }
                    LMMC_REAL_ADD(&tmp_sum, &sum, v);
                    LMMC_REAL_SET(&sum, &tmp_sum);
                    if (!lmmc_is_finite_number(&sum)) {
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
    size_t old_total = 0;
    size_t new_total = 0;

    if (lmmc_tensor_validate(tensor) != LMMC_STATUS_OK || out_view == NULL ||
        new_dim0 == 0 || new_dim1 == 0 || new_dim2 == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_tensor_is_contiguous(tensor)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (lmmc_mul_overflow_size(tensor->dim0, tensor->dim1, &old_total) ||
        lmmc_mul_overflow_size(old_total, tensor->dim2, &old_total) ||
        lmmc_mul_overflow_size(new_dim0, new_dim1, &new_total) ||
        lmmc_mul_overflow_size(new_total, new_dim2, &new_total)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (old_total != new_total) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    out_view->dim0 = new_dim0;
    out_view->dim1 = new_dim1;
    out_view->dim2 = new_dim2;
    out_view->stride2 = 1;
    out_view->stride1 = new_dim2;
    out_view->stride0 = new_dim1 * new_dim2;
    out_view->data = tensor->data;
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

    if (lmmc_mul_overflow_size(begin0, tensor->stride0, &off0) ||
        lmmc_mul_overflow_size(begin1, tensor->stride1, &off1) ||
        lmmc_mul_overflow_size(begin2, tensor->stride2, &off2)) {
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
