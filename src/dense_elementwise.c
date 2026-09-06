/**
 * @file dense_elementwise.c
 * @brief 稠密逐元素运算实现（Hadamard、比较、apply 及其数学函数包装）。
 */

#include <math.h>
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/linear_algebra.h"

lmmc_status_t lmmc_vec_hadamard(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c) {
    size_t i;
    lmmc_real_t tmp_mul;

    if (!lmmc_vec_descriptor_is_valid(a) ||
        !lmmc_vec_descriptor_is_valid(b) ||
        !lmmc_vec_descriptor_is_valid(c)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->size != b->size || a->size != c->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    LMMC_REAL_INIT(&tmp_mul);

    for (i = 0; i < a->size; ++i) {
        LMMC_REAL_MUL(&tmp_mul, &a->data[i], &b->data[i]);
        LMMC_REAL_SET(&c->data[i], &tmp_mul);
    }

    LMMC_REAL_CLEAR(&tmp_mul);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_elementwise_div(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c) {
    size_t i;
    lmmc_real_t zero;
    lmmc_real_t tmp_div;

    if (!lmmc_vec_descriptor_is_valid(a) ||
        !lmmc_vec_descriptor_is_valid(b) ||
        !lmmc_vec_descriptor_is_valid(c)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->size != b->size || a->size != c->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    /* Pre-scan b for zeros before writing any output */
    LMMC_REAL_INIT(&zero);
    LMMC_REAL_SET_D(&zero, 0.0);
    for (i = 0; i < b->size; ++i) {
        if (LMMC_REAL_CMP(&b->data[i], &zero) == 0) {
            LMMC_REAL_CLEAR(&zero);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    LMMC_REAL_CLEAR(&zero);

    LMMC_REAL_INIT(&tmp_div);

    for (i = 0; i < a->size; ++i) {
        LMMC_REAL_DIV(&tmp_div, &a->data[i], &b->data[i]);
        LMMC_REAL_SET(&c->data[i], &tmp_div);
    }

    LMMC_REAL_CLEAR(&tmp_div);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_elementwise_pow(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c) {
    size_t i;

    if (!lmmc_vec_descriptor_is_valid(a) ||
        !lmmc_vec_descriptor_is_valid(b) ||
        !lmmc_vec_descriptor_is_valid(c)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->size != b->size || a->size != c->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    for (i = 0; i < a->size; ++i) {
        c->data[i] = pow(a->data[i], b->data[i]);
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_hadamard(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c) {
    size_t i, j;
    lmmc_real_t tmp_mul;

    if (!lmmc_mat_descriptor_is_valid(a) ||
        !lmmc_mat_descriptor_is_valid(b) ||
        !lmmc_mat_descriptor_is_valid(c)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != c->rows || a->cols != c->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    LMMC_REAL_INIT(&tmp_mul);

    for (i = 0; i < a->rows; ++i) {
        for (j = 0; j < a->cols; ++j) {
            LMMC_REAL_MUL(&tmp_mul,
                          &a->data[i * a->stride + j],
                          &b->data[i * b->stride + j]);
            LMMC_REAL_SET(&c->data[i * c->stride + j], &tmp_mul);
        }
    }

    LMMC_REAL_CLEAR(&tmp_mul);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_elementwise_div(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c) {
    size_t i, j;
    lmmc_real_t zero;
    lmmc_real_t tmp_div;

    if (!lmmc_mat_descriptor_is_valid(a) ||
        !lmmc_mat_descriptor_is_valid(b) ||
        !lmmc_mat_descriptor_is_valid(c)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != c->rows || a->cols != c->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    /* Pre-scan all elements of b for zeros before writing any output */
    LMMC_REAL_INIT(&zero);
    LMMC_REAL_SET_D(&zero, 0.0);
    for (i = 0; i < b->rows; ++i) {
        for (j = 0; j < b->cols; ++j) {
            if (LMMC_REAL_CMP(&b->data[i * b->stride + j], &zero) == 0) {
                LMMC_REAL_CLEAR(&zero);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }
    }
    LMMC_REAL_CLEAR(&zero);

    LMMC_REAL_INIT(&tmp_div);

    for (i = 0; i < a->rows; ++i) {
        for (j = 0; j < a->cols; ++j) {
            LMMC_REAL_DIV(&tmp_div,
                          &a->data[i * a->stride + j],
                          &b->data[i * b->stride + j]);
            LMMC_REAL_SET(&c->data[i * c->stride + j], &tmp_div);
        }
    }

    LMMC_REAL_CLEAR(&tmp_div);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_elementwise_pow(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c) {
    size_t i, j;

    if (!lmmc_mat_descriptor_is_valid(a) ||
        !lmmc_mat_descriptor_is_valid(b) ||
        !lmmc_mat_descriptor_is_valid(c)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != c->rows || a->cols != c->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    for (i = 0; i < a->rows; ++i) {
        for (j = 0; j < a->cols; ++j) {
            c->data[i * c->stride + j] = pow(
                a->data[i * a->stride + j],
                b->data[i * b->stride + j]);
        }
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_cmp_gt(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out) {
    size_t i;

    if (!lmmc_vec_descriptor_is_valid(a) ||
        !lmmc_vec_descriptor_is_valid(b) || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->size != b->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    for (i = 0; i < a->size; ++i) {
        out[i] = (LMMC_REAL_CMP(&a->data[i], &b->data[i]) > 0) ? 1 : 0;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_cmp_lt(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out) {
    size_t i;

    if (!lmmc_vec_descriptor_is_valid(a) ||
        !lmmc_vec_descriptor_is_valid(b) || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->size != b->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    for (i = 0; i < a->size; ++i) {
        out[i] = (LMMC_REAL_CMP(&a->data[i], &b->data[i]) < 0) ? 1 : 0;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_cmp_ge(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out) {
    size_t i;

    if (!lmmc_vec_descriptor_is_valid(a) ||
        !lmmc_vec_descriptor_is_valid(b) || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->size != b->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    for (i = 0; i < a->size; ++i) {
        out[i] = (LMMC_REAL_CMP(&a->data[i], &b->data[i]) >= 0) ? 1 : 0;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_cmp_le(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out) {
    size_t i;

    if (!lmmc_vec_descriptor_is_valid(a) ||
        !lmmc_vec_descriptor_is_valid(b) || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->size != b->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    for (i = 0; i < a->size; ++i) {
        out[i] = (LMMC_REAL_CMP(&a->data[i], &b->data[i]) <= 0) ? 1 : 0;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_cmp_eq(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t tol, int* out) {
    size_t i;
    lmmc_real_t diff;
    lmmc_real_t abs_diff;

    if (!lmmc_vec_descriptor_is_valid(a) ||
        !lmmc_vec_descriptor_is_valid(b) || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->size != b->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    LMMC_REAL_INIT(&diff);
    LMMC_REAL_INIT(&abs_diff);

    for (i = 0; i < a->size; ++i) {
        LMMC_REAL_SUB(&diff, &a->data[i], &b->data[i]);
        LMMC_REAL_ABS(&abs_diff, &diff);
        out[i] = (LMMC_REAL_CMP(&abs_diff, &tol) <= 0) ? 1 : 0;
    }

    LMMC_REAL_CLEAR(&diff);
    LMMC_REAL_CLEAR(&abs_diff);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_cross(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c)
{
    lmmc_real_t t0, t1, t2;
    lmmc_real_t mul1, mul2;

    if (!lmmc_vec_descriptor_is_valid(a) ||
        !lmmc_vec_descriptor_is_valid(b) ||
        !lmmc_vec_descriptor_is_valid(c)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->size != 3 || b->size != 3 || c->size != 3) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    LMMC_REAL_INIT(&t0);
    LMMC_REAL_INIT(&t1);
    LMMC_REAL_INIT(&t2);
    LMMC_REAL_INIT(&mul1);
    LMMC_REAL_INIT(&mul2);

    /* c[0] = a[1]*b[2] - a[2]*b[1] */
    LMMC_REAL_MUL(&mul1, &a->data[1], &b->data[2]);
    LMMC_REAL_MUL(&mul2, &a->data[2], &b->data[1]);
    LMMC_REAL_SUB(&t0, &mul1, &mul2);

    /* c[1] = a[2]*b[0] - a[0]*b[2] */
    LMMC_REAL_MUL(&mul1, &a->data[2], &b->data[0]);
    LMMC_REAL_MUL(&mul2, &a->data[0], &b->data[2]);
    LMMC_REAL_SUB(&t1, &mul1, &mul2);

    /* c[2] = a[0]*b[1] - a[1]*b[0] */
    LMMC_REAL_MUL(&mul1, &a->data[0], &b->data[1]);
    LMMC_REAL_MUL(&mul2, &a->data[1], &b->data[0]);
    LMMC_REAL_SUB(&t2, &mul1, &mul2);

    /* Write results (safe for aliasing since we used temporaries) */
    LMMC_REAL_SET(&c->data[0], &t0);
    LMMC_REAL_SET(&c->data[1], &t1);
    LMMC_REAL_SET(&c->data[2], &t2);

    LMMC_REAL_CLEAR(&t0);
    LMMC_REAL_CLEAR(&t1);
    LMMC_REAL_CLEAR(&t2);
    LMMC_REAL_CLEAR(&mul1);
    LMMC_REAL_CLEAR(&mul2);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_apply(const lmmc_vec_t* in, lmmc_real_t (*func)(lmmc_real_t), lmmc_vec_t* out)
{
    size_t i;

    if (!lmmc_vec_descriptor_is_valid(in) ||
        !lmmc_vec_descriptor_is_valid(out) || func == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (in->size != out->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    for (i = 0; i < in->size; ++i) {
        out->data[i] = func(in->data[i]);
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_apply(const lmmc_mat_t* in, lmmc_real_t (*func)(lmmc_real_t), lmmc_mat_t* out)
{
    size_t i;
    size_t j;

    if (!lmmc_mat_descriptor_is_valid(in) ||
        !lmmc_mat_descriptor_is_valid(out) || func == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (in->rows != out->rows || in->cols != out->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    for (i = 0; i < in->rows; ++i) {
        for (j = 0; j < in->cols; ++j) {
            out->data[i * out->stride + j] = func(in->data[i * in->stride + j]);
        }
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_apply_sin(const lmmc_vec_t* in, lmmc_vec_t* out)
{
    return lmmc_vec_apply(in, sin, out);
}

lmmc_status_t lmmc_vec_apply_cos(const lmmc_vec_t* in, lmmc_vec_t* out)
{
    return lmmc_vec_apply(in, cos, out);
}

lmmc_status_t lmmc_vec_apply_exp(const lmmc_vec_t* in, lmmc_vec_t* out)
{
    return lmmc_vec_apply(in, exp, out);
}

lmmc_status_t lmmc_vec_apply_log(const lmmc_vec_t* in, lmmc_vec_t* out)
{
    return lmmc_vec_apply(in, log, out);
}

lmmc_status_t lmmc_vec_apply_sqrt(const lmmc_vec_t* in, lmmc_vec_t* out)
{
    return lmmc_vec_apply(in, sqrt, out);
}

lmmc_status_t lmmc_vec_apply_abs(const lmmc_vec_t* in, lmmc_vec_t* out)
{
    return lmmc_vec_apply(in, fabs, out);
}

lmmc_status_t lmmc_mat_apply_sin(const lmmc_mat_t* in, lmmc_mat_t* out)
{
    return lmmc_mat_apply(in, sin, out);
}

lmmc_status_t lmmc_mat_apply_cos(const lmmc_mat_t* in, lmmc_mat_t* out)
{
    return lmmc_mat_apply(in, cos, out);
}

lmmc_status_t lmmc_mat_apply_exp(const lmmc_mat_t* in, lmmc_mat_t* out)
{
    return lmmc_mat_apply(in, exp, out);
}

lmmc_status_t lmmc_mat_apply_log(const lmmc_mat_t* in, lmmc_mat_t* out)
{
    return lmmc_mat_apply(in, log, out);
}

lmmc_status_t lmmc_mat_apply_sqrt(const lmmc_mat_t* in, lmmc_mat_t* out)
{
    return lmmc_mat_apply(in, sqrt, out);
}

lmmc_status_t lmmc_mat_apply_abs(const lmmc_mat_t* in, lmmc_mat_t* out)
{
    return lmmc_mat_apply(in, fabs, out);
}

