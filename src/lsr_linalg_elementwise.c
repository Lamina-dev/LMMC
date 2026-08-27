#include "lmmc/lsr_stdlib.h"

#include <math.h>
#include <stdlib.h>

#include "lmmc/dense.h"
#include "lmmc/linear_algebra.h"

#include "lsr_stdlib_internal.h"

lmmc_status_t lmmc_lsr_linalg_vec_add(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_lsr_copy_vec(a, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_axpy((lmmc_real_t)1, b, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_add_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_copy_vec(x, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        out->data[i] += scalar;
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_sub(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_lsr_copy_vec(a, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_axpy((lmmc_real_t)-1, b, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_sub_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    return lmmc_lsr_linalg_vec_add_scalar(x, -scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_scalar_sub_vec(lmmc_real_t scalar,
                                             const lmmc_vec_t* x,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_require_finite_vec(x);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(x->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        out->data[i] = scalar - x->data[i];
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_mul(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_create(a->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        out->data[i] = a->data[i] * b->data[i];
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_div(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_create(a->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        if (b->data[i] == (lmmc_real_t)0) {
            lmmc_vec_destroy(out);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        out->data[i] = a->data[i] / b->data[i];
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_div_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    if (scalar == (lmmc_real_t)0) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_linalg_vec_scale(x, (lmmc_real_t)1 / scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_scalar_div_vec(lmmc_real_t scalar,
                                             const lmmc_vec_t* x,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_require_finite_vec(x);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(x->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        if (x->data[i] == (lmmc_real_t)0) {
            lmmc_vec_destroy(out);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        out->data[i] = scalar / x->data[i];
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_pow(const lmmc_vec_t* base,
                                      const lmmc_vec_t* exponent,
                                      lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(base);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(exponent);
    if (status != LMMC_STATUS_OK) return status;
    if (base->size != exponent->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_create(base->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        status =
            lmmc_lsr_pow_finite(base->data[i], exponent->data[i], &out->data[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(out);
            return status;
        }
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_pow_scalar(const lmmc_vec_t* base,
                                             lmmc_real_t exponent,
                                             lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(base);
    if (status != LMMC_STATUS_OK) return status;
    if (!lmmc_lsr_real_is_finite(exponent)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_vec_create(base->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        status = lmmc_lsr_pow_finite(base->data[i], exponent, &out->data[i]);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(out);
            return status;
        }
    }
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_scale(const lmmc_vec_t* x,
                                        lmmc_real_t alpha,
                                        lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(alpha)) return LMMC_STATUS_NUMERICAL_FAILURE;
    status = lmmc_lsr_copy_vec(x, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_scale(out, alpha);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_vec_mul_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out)
{
    return lmmc_lsr_linalg_vec_scale(x, scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_mat_add(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_add(a, b, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_add_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_copy_mat(a, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            out->data[i * out->stride + j] += scalar;
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_sub(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_sub(a, b, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_sub_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    return lmmc_lsr_linalg_mat_add_scalar(a, -scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_scalar_sub_mat(lmmc_real_t scalar,
                                             const lmmc_mat_t* a,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            out->data[i * out->stride + j] =
                scalar - a->data[i * a->stride + j];
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_mul_elem(const lmmc_mat_t* a,
                                           const lmmc_mat_t* b,
                                           lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            out->data[i * out->stride + j] =
                a->data[i * a->stride + j] * b->data[i * b->stride + j];
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_div(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            lmmc_real_t denom = b->data[i * b->stride + j];
            if (denom == (lmmc_real_t)0) {
                lmmc_mat_destroy(out);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            out->data[i * out->stride + j] =
                a->data[i * a->stride + j] / denom;
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_div_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    if (scalar == (lmmc_real_t)0) return LMMC_STATUS_NUMERICAL_FAILURE;
    return lmmc_lsr_linalg_mat_scale(a, (lmmc_real_t)1 / scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_scalar_div_mat(lmmc_real_t scalar,
                                             const lmmc_mat_t* a,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            lmmc_real_t denom = a->data[i * a->stride + j];
            if (denom == (lmmc_real_t)0) {
                lmmc_mat_destroy(out);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            out->data[i * out->stride + j] = scalar / denom;
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_pow_elem(const lmmc_mat_t* base,
                                           const lmmc_mat_t* exponent,
                                           lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(base);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(exponent);
    if (status != LMMC_STATUS_OK) return status;
    if (base->rows != exponent->rows || base->cols != exponent->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_mat_create(base->rows, base->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            status = lmmc_lsr_pow_finite(base->data[i * base->stride + j],
                                         exponent->data[i * exponent->stride + j],
                                         &out->data[i * out->stride + j]);
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(out);
                return status;
            }
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_pow_scalar(const lmmc_mat_t* base,
                                             lmmc_real_t exponent,
                                             lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(base);
    if (status != LMMC_STATUS_OK) return status;
    if (!lmmc_lsr_real_is_finite(exponent)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_mat_create(base->rows, base->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            status = lmmc_lsr_pow_finite(base->data[i * base->stride + j],
                                         exponent,
                                         &out->data[i * out->stride + j]);
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(out);
                return status;
            }
        }
    }
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_pow_int(const lmmc_mat_t* base,
                                          int64_t exponent,
                                          lmmc_mat_t* out)
{
    lmmc_status_t status;
    lmmc_mat_t factor = {0, 0, 0, NULL, 0};
    lmmc_mat_t result = {0, 0, 0, NULL, 0};

    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(base);
    if (status != LMMC_STATUS_OK) return status;
    if (base->rows != base->cols) return LMMC_STATUS_DIMENSION_MISMATCH;

    uint64_t power;
    if (exponent < 0) {
        status = lmmc_lsr_linalg_inv(base, &factor);
        if (status != LMMC_STATUS_OK) return status;
        power = (uint64_t)(-(exponent + 1)) + UINT64_C(1);
    } else {
        status = lmmc_lsr_copy_mat(base, &factor);
        if (status != LMMC_STATUS_OK) return status;
        power = (uint64_t)exponent;
    }

    status = lmmc_mat_identity(base->rows, &result);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&factor);
        return status;
    }

    while (power > 0) {
        if ((power & UINT64_C(1)) != 0) {
            lmmc_mat_t product = {0, 0, 0, NULL, 0};
            status = lmmc_mat_create(result.rows, factor.cols, &product);
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(&result);
                lmmc_mat_destroy(&factor);
                return status;
            }
            status = lmmc_mat_mul(&result, &factor, &product);
            if (status == LMMC_STATUS_OK) {
                status = lmmc_lsr_require_finite_mat(&product);
            }
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(&product);
                lmmc_mat_destroy(&result);
                lmmc_mat_destroy(&factor);
                return status;
            }
            lmmc_mat_destroy(&result);
            result = product;
        }

        power >>= 1;
        if (power > 0) {
            lmmc_mat_t square = {0, 0, 0, NULL, 0};
            status = lmmc_mat_create(factor.rows, factor.cols, &square);
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(&result);
                lmmc_mat_destroy(&factor);
                return status;
            }
            status = lmmc_mat_mul(&factor, &factor, &square);
            if (status == LMMC_STATUS_OK) {
                status = lmmc_lsr_require_finite_mat(&square);
            }
            if (status != LMMC_STATUS_OK) {
                lmmc_mat_destroy(&square);
                lmmc_mat_destroy(&result);
                lmmc_mat_destroy(&factor);
                return status;
            }
            lmmc_mat_destroy(&factor);
            factor = square;
        }
    }

    lmmc_mat_destroy(&factor);
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_mat_scale(const lmmc_mat_t* a,
                                        lmmc_real_t alpha,
                                        lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_lsr_real_is_finite(alpha)) return LMMC_STATUS_NUMERICAL_FAILURE;
    status = lmmc_lsr_copy_mat(a, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_scale(out, alpha);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_mat_mul_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out)
{
    return lmmc_lsr_linalg_mat_scale(a, scalar, out);
}

lmmc_status_t lmmc_lsr_linalg_vec_compare(const lmmc_vec_t* a,
                                          const lmmc_vec_t* b,
                                          lmmc_lsr_compare_op_t op,
                                          lmmc_lsr_bool_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_lsr_bool_vec_create(a->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        if (!lmmc_lsr_compare_eval(a->data[i], op, b->data[i],
                                   &out->data[i])) {
            lmmc_lsr_bool_vec_destroy(out);
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_vec_compare_scalar(const lmmc_vec_t* a,
                                                 lmmc_lsr_compare_op_t op,
                                                 lmmc_real_t scalar,
                                                 lmmc_lsr_bool_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_bool_vec_create(a->size, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->size; ++i) {
        if (!lmmc_lsr_compare_eval(a->data[i], op, scalar, &out->data[i])) {
            lmmc_lsr_bool_vec_destroy(out);
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_mat_compare(const lmmc_mat_t* a,
                                          const lmmc_mat_t* b,
                                          lmmc_lsr_compare_op_t op,
                                          lmmc_lsr_bool_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != b->rows || a->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    status = lmmc_lsr_bool_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            if (!lmmc_lsr_compare_eval(a->data[i * a->stride + j],
                                       op,
                                       b->data[i * b->stride + j],
                                       &out->data[i * out->stride + j])) {
                lmmc_lsr_bool_mat_destroy(out);
                return LMMC_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_mat_compare_scalar(const lmmc_mat_t* a,
                                                 lmmc_lsr_compare_op_t op,
                                                 lmmc_real_t scalar,
                                                 lmmc_lsr_bool_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    if (!lmmc_lsr_real_is_finite(scalar)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_lsr_bool_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            if (!lmmc_lsr_compare_eval(a->data[i * a->stride + j],
                                       op,
                                       scalar,
                                       &out->data[i * out->stride + j])) {
                lmmc_lsr_bool_mat_destroy(out);
                return LMMC_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return LMMC_STATUS_OK;
}

