#include "lmmc/stdlib.h"

#include "lmmc/dense.h"

#include "stdlib_internal.h"

lmmc_status_t lmmc_std_linalg_vec_add(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    lmmc_vec_t result = {0};
    if (!lmmc_std_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) { return status; }
    if (a->size != b->size) { return LMMC_STATUS_DIMENSION_MISMATCH; }
    status = lmmc_std_copy_vec(a, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_vec_axpy((lmmc_real_t)1, b, &result);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_std_require_finite_vec(&result);
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_vec_add_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    lmmc_vec_t result = {0};
    if (!lmmc_std_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_std_copy_vec(x, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.size; ++i) {
        result.data[i] += scalar;
    }
    status = lmmc_std_require_finite_vec(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_vec_sub(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    lmmc_vec_t result = {0};
    if (!lmmc_std_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) { return status; }
    if (a->size != b->size) { return LMMC_STATUS_DIMENSION_MISMATCH; }
    status = lmmc_std_copy_vec(a, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_vec_axpy((lmmc_real_t)-1, b, &result);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_std_require_finite_vec(&result);
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_vec_sub_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    return lmmc_std_linalg_vec_add_scalar(x, -scalar, out);
}

lmmc_status_t lmmc_std_linalg_scalar_sub_vec(lmmc_real_t scalar,
                                             const lmmc_vec_t* x,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    lmmc_vec_t result = {0};
    if (!lmmc_std_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_std_require_finite_vec(x);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_vec_create(x->size, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.size; ++i) {
        result.data[i] = scalar - x->data[i];
    }
    status = lmmc_std_require_finite_vec(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_vec_mul(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    lmmc_vec_t result = {0};
    if (!lmmc_std_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) { return status; }
    if (a->size != b->size) { return LMMC_STATUS_DIMENSION_MISMATCH; }
    status = lmmc_vec_create(a->size, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.size; ++i) {
        result.data[i] = a->data[i] * b->data[i];
    }
    status = lmmc_std_require_finite_vec(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_vec_div(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    lmmc_vec_t result = {0};
    if (!lmmc_std_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) { return status; }
    if (a->size != b->size) { return LMMC_STATUS_DIMENSION_MISMATCH; }
    for (size_t i = 0; i < b->size; ++i) {
        if (b->data[i] == (lmmc_real_t)0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    }
    status = lmmc_vec_create(a->size, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.size; ++i) {
        result.data[i] = a->data[i] / b->data[i];
    }
    status = lmmc_std_require_finite_vec(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_vec_div_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    lmmc_vec_t result = {0};
    if (!lmmc_std_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (scalar == (lmmc_real_t)0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    if (!lmmc_std_real_is_finite(scalar)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    status = lmmc_std_require_finite_vec(x);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_vec_create(x->size, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.size; ++i) {
        result.data[i] = x->data[i] / scalar;
    }
    status = lmmc_std_require_finite_vec(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_scalar_div_vec(lmmc_real_t scalar,
                                             const lmmc_vec_t* x,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    lmmc_vec_t result = {0};
    if (!lmmc_std_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_std_require_finite_vec(x);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < x->size; ++i) {
        if (x->data[i] == (lmmc_real_t)0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    }
    status = lmmc_vec_create(x->size, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.size; ++i) {
        result.data[i] = scalar / x->data[i];
    }
    status = lmmc_std_require_finite_vec(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_vec_pow(const lmmc_vec_t* base,
                                      const lmmc_vec_t* exponent,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    lmmc_vec_t result = {0};
    if (!lmmc_std_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_vec(base);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_vec(exponent);
    if (status != LMMC_STATUS_OK) { return status; }
    if (base->size != exponent->size) { return LMMC_STATUS_DIMENSION_MISMATCH; }
    status = lmmc_vec_create(base->size, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.size; ++i) {
        status =
            lmmc_std_pow_finite(base->data[i], exponent->data[i], &result.data[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(&result);
            return status;
        }
    }
    status = lmmc_std_require_finite_vec(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_vec_pow_scalar(const lmmc_vec_t* base,
                                             lmmc_real_t exponent,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    lmmc_vec_t result = {0};
    if (!lmmc_std_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_vec(base);
    if (status != LMMC_STATUS_OK) { return status; }
    if (!lmmc_std_real_is_finite(exponent)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_vec_create(base->size, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.size; ++i) {
        status = lmmc_std_pow_finite(base->data[i], exponent, &result.data[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(&result);
            return status;
        }
    }
    status = lmmc_std_require_finite_vec(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_vec_scale(const lmmc_vec_t* x,
                                        lmmc_real_t alpha,
                                        lmmc_vec_t* out)
{
    lmmc_status_t status;
    lmmc_vec_t result = {0};
    if (!lmmc_std_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(alpha)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    status = lmmc_std_copy_vec(x, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_vec_scale(&result, alpha);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_std_require_finite_vec(&result);
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_vec_mul_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    return lmmc_std_linalg_vec_scale(x, scalar, out);
}

lmmc_status_t lmmc_std_linalg_mat_add(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t result = {0};
    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) { return status; }
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(a->rows, a->cols, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_mat_add(a, b, &result);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_std_require_finite_mat(&result);
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_mat_add_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t result = {0};
    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_std_copy_mat(a, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            result.data[i * result.stride + j] += scalar;
        }
    }
    status = lmmc_std_require_finite_mat(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_mat_sub(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t result = {0};
    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) { return status; }
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(a->rows, a->cols, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_mat_sub(a, b, &result);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_std_require_finite_mat(&result);
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_mat_sub_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    return lmmc_std_linalg_mat_add_scalar(a, -scalar, out);
}

lmmc_status_t lmmc_std_linalg_scalar_sub_mat(lmmc_real_t scalar,
                                             const lmmc_mat_t* a,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t result = {0};
    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_std_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_mat_create(a->rows, a->cols, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            result.data[i * result.stride + j] =
                scalar - a->data[i * a->stride + j];
        }
    }
    status = lmmc_std_require_finite_mat(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_mat_mul_elem(const lmmc_mat_t* a,
                                           const lmmc_mat_t* b,
                                           lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t result = {0};
    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) { return status; }
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(a->rows, a->cols, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            result.data[i * result.stride + j] =
                a->data[i * a->stride + j] * b->data[i * b->stride + j];
        }
    }
    status = lmmc_std_require_finite_mat(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_mat_div(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t result = {0};
    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) { return status; }
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    for (size_t i = 0; i < b->rows; ++i) {
        for (size_t j = 0; j < b->cols; ++j) {
            if (b->data[i * b->stride + j] == (lmmc_real_t)0) {
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }
    }
    status = lmmc_mat_create(a->rows, a->cols, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            lmmc_real_t denom = b->data[i * b->stride + j];
            result.data[i * result.stride + j] =
                a->data[i * a->stride + j] / denom;
        }
    }
    status = lmmc_std_require_finite_mat(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_mat_div_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t result = {0};
    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (scalar == (lmmc_real_t)0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    if (!lmmc_std_real_is_finite(scalar)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    status = lmmc_std_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_mat_create(a->rows, a->cols, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            result.data[i * result.stride + j] =
                a->data[i * a->stride + j] / scalar;
        }
    }
    status = lmmc_std_require_finite_mat(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_scalar_div_mat(lmmc_real_t scalar,
                                             const lmmc_mat_t* a,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t result = {0};
    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_std_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            if (a->data[i * a->stride + j] == (lmmc_real_t)0) {
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }
    }
    status = lmmc_mat_create(a->rows, a->cols, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            lmmc_real_t denom = a->data[i * a->stride + j];
            result.data[i * result.stride + j] = scalar / denom;
        }
    }
    status = lmmc_std_require_finite_mat(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_mat_pow_elem(const lmmc_mat_t* base,
                                           const lmmc_mat_t* exponent,
                                           lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t result = {0};
    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_mat(base);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_mat(exponent);
    if (status != LMMC_STATUS_OK) { return status; }
    if (base->rows != exponent->rows || base->cols != exponent->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(base->rows, base->cols, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            status = lmmc_std_pow_finite(base->data[i * base->stride + j],
                                         exponent->data[i * exponent->stride + j],
                                         &result.data[i * result.stride + j]);
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(&result);
                return status;
            }
        }
    }
    status = lmmc_std_require_finite_mat(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_mat_pow_scalar(const lmmc_mat_t* base,
                                             lmmc_real_t exponent,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t result = {0};
    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_mat(base);
    if (status != LMMC_STATUS_OK) { return status; }
    if (!lmmc_std_real_is_finite(exponent)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_mat_create(base->rows, base->cols, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            status = lmmc_std_pow_finite(base->data[i * base->stride + j],
                                         exponent,
                                         &result.data[i * result.stride + j]);
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(&result);
                return status;
            }
        }
    }
    status = lmmc_std_require_finite_mat(&result);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

static lmmc_status_t lmmc_std_replace_with_product(
    lmmc_mat_t* target, const lmmc_mat_t* factor)
{
    lmmc_mat_t product = {0, 0, 0, NULL, 0};
    lmmc_status_t status = lmmc_mat_create(target->rows, factor->cols, &product);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_mat_mul(target, factor, &product);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_std_require_finite_mat(&product);
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&product);
        return status;
    }
    lmmc_mat_destroy(target);
    *target = product;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_linalg_mat_pow_int(const lmmc_mat_t* base,
                                          int64_t exponent,
                                          lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t factor = {0, 0, 0, NULL, 0};
    lmmc_mat_t result = {0, 0, 0, NULL, 0};

    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_mat(base);
    if (status != LMMC_STATUS_OK) { return status; }
    if (base->rows != base->cols) { return LMMC_STATUS_DIMENSION_MISMATCH; }

    uint64_t power;
    if (exponent < 0) {
        status = lmmc_std_linalg_inv(base, &factor);
        if (status != LMMC_STATUS_OK) { return status; }
        power = (uint64_t)(-(exponent + 1)) + UINT64_C(1);
    } else {
        status = lmmc_std_copy_mat(base, &factor);
        if (status != LMMC_STATUS_OK) { return status; }
        power = (uint64_t)exponent;
    }

    status = lmmc_mat_identity(base->rows, &result);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&factor);
        return status;
    }

    while (power > 0) {
        if ((power & UINT64_C(1)) != 0) {
            status = lmmc_std_replace_with_product(&result, &factor);
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(&result);
                lmmc_mat_destroy(&factor);
                return status;
            }
        }

        power >>= 1;
        if (power > 0) {
            status = lmmc_std_replace_with_product(&factor, &factor);
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(&result);
                lmmc_mat_destroy(&factor);
                return status;
            }
        }
    }

    lmmc_mat_destroy(&factor);
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_linalg_mat_scale(const lmmc_mat_t* a,
                                        lmmc_real_t alpha,
                                        lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t result = {0};
    if (!lmmc_std_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(alpha)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    status = lmmc_std_copy_mat(a, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_mat_scale(&result, alpha);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_std_require_finite_mat(&result);
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result);
    } else {
        *out = result;
    }
    return status;
}

lmmc_status_t lmmc_std_linalg_mat_mul_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    return lmmc_std_linalg_mat_scale(a, scalar, out);
}

lmmc_status_t lmmc_std_linalg_vec_compare(const lmmc_vec_t* a,
                                          const lmmc_vec_t* b,
                                          lmmc_std_compare_op_t op,
                                          lmmc_std_bool_vec_t* out)
{
    lmmc_status_t status;
    lmmc_std_bool_vec_t result = {0};
    uint8_t ignored;
    if (!lmmc_std_bool_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) { return status; }
    if (a->size != b->size) { return LMMC_STATUS_DIMENSION_MISMATCH; }
    if (!lmmc_std_compare_eval(0, op, 0, &ignored)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_std_bool_vec_create(a->size, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.size; ++i) {
        (void)lmmc_std_compare_eval(a->data[i], op, b->data[i], &result.data[i]);
    }
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_linalg_vec_compare_scalar(const lmmc_vec_t* a,
                                                 lmmc_std_compare_op_t op,
                                                 lmmc_real_t scalar,
                                                 lmmc_std_bool_vec_t* out)
{
    lmmc_status_t status;
    lmmc_std_bool_vec_t result = {0};
    uint8_t ignored;
    if (!lmmc_std_bool_vec_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) { return status; }
    if (!lmmc_std_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (!lmmc_std_compare_eval(0, op, 0, &ignored)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_std_bool_vec_create(a->size, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.size; ++i) {
        (void)lmmc_std_compare_eval(a->data[i], op, scalar, &result.data[i]);
    }
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_linalg_mat_compare(const lmmc_mat_t* a,
                                          const lmmc_mat_t* b,
                                          lmmc_std_compare_op_t op,
                                          lmmc_std_bool_mat_t* out)
{
    lmmc_status_t status;
    lmmc_std_bool_mat_t result = {0};
    uint8_t ignored;
    if (!lmmc_std_bool_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_std_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) { return status; }
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    if (!lmmc_std_compare_eval(0, op, 0, &ignored)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_std_bool_mat_create(a->rows, a->cols, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            (void)lmmc_std_compare_eval(a->data[i * a->stride + j],
                                        op, b->data[i * b->stride + j],
                                        &result.data[i * result.stride + j]);
        }
    }
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_linalg_mat_compare_scalar(const lmmc_mat_t* a,
                                                 lmmc_std_compare_op_t op,
                                                 lmmc_real_t scalar,
                                                 lmmc_std_bool_mat_t* out)
{
    lmmc_status_t status;
    lmmc_std_bool_mat_t result = {0};
    uint8_t ignored;
    if (!lmmc_std_bool_mat_is_empty(out)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_std_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) { return status; }
    if (!lmmc_std_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (!lmmc_std_compare_eval(0, op, 0, &ignored)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_std_bool_mat_create(a->rows, a->cols, &result);
    if (status != LMMC_STATUS_OK) { return status; }
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            (void)lmmc_std_compare_eval(a->data[i * a->stride + j],
                                        op, scalar,
                                        &result.data[i * result.stride + j]);
        }
    }
    *out = result;
    return LMMC_STATUS_OK;
}

