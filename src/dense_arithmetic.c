/**
 * @file dense_arithmetic.c
 * @brief 稠密矩阵 / 向量算术与范数实现（加、减、缩放、范数、迹等）。
 */

#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "blas_backend.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/linear_algebra.h"

lmmc_status_t lmmc_mat_norm_fro(const lmmc_mat_t* a, lmmc_real_t* out_norm) {
    size_t i = 0;
    size_t j = 0;
    lmmc_real_t sum;
    lmmc_real_t tmp_mul;
    lmmc_real_t tmp_sum;

    if (a == NULL || out_norm == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_INIT(&sum);
    LMMC_REAL_INIT(&tmp_mul);
    LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_SET_D(&sum, 0.0);

    for (i = 0; i < a->rows; ++i) {
        for (j = 0; j < a->cols; ++j) {
            LMMC_REAL_MUL(&tmp_mul, &a->data[i * a->stride + j], &a->data[i * a->stride + j]);
            LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
            LMMC_REAL_SET(&sum, &tmp_sum);
        }
    }
    LMMC_REAL_SQRT(out_norm, &sum);

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_norm2(const lmmc_vec_t* x, lmmc_real_t* out_norm) {
    if (x == NULL || out_norm == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

#ifdef LMMC_USE_BLAS

    *out_norm = lmmc_blas_dnrm2(x->size, x->data, 1);
    return LMMC_STATUS_OK;
#else
    lmmc_real_t sum; LMMC_REAL_INIT(&sum);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_SET_D(&sum, 0.0);

    for (size_t i = 0; i < x->size; ++i) {
        LMMC_REAL_MUL(&tmp_mul, &x->data[i], &x->data[i]);
        LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
        LMMC_REAL_SET(&sum, &tmp_sum);
    }
    LMMC_REAL_SQRT(out_norm, &sum);

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
#endif
}

lmmc_status_t lmmc_vec_norm_inf(const lmmc_vec_t* x, lmmc_real_t* out_norm) {
    if (x == NULL || out_norm == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t max_val; LMMC_REAL_INIT(&max_val);
    lmmc_real_t abs_val; LMMC_REAL_INIT(&abs_val);
    LMMC_REAL_SET_D(&max_val, 0.0);

    for (size_t i = 0; i < x->size; ++i) {
        LMMC_REAL_ABS(&abs_val, &x->data[i]);
        max_val = lmmc_max(max_val, abs_val);
    }
    LMMC_REAL_SET(out_norm, &max_val);

    LMMC_REAL_CLEAR(&max_val);
    LMMC_REAL_CLEAR(&abs_val);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_scale(lmmc_vec_t* x, lmmc_real_t alpha) {
    if (x == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);

    for (size_t i = 0; i < x->size; ++i) {
        LMMC_REAL_MUL(&tmp_mul, &x->data[i], &alpha);
        LMMC_REAL_SET(&x->data[i], &tmp_mul);
    }

    LMMC_REAL_CLEAR(&tmp_mul);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_asum(const lmmc_vec_t* x, lmmc_real_t* out_asum) {
    if (x == NULL || out_asum == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t sum; LMMC_REAL_INIT(&sum);
    lmmc_real_t abs_val; LMMC_REAL_INIT(&abs_val);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_SET_D(&sum, 0.0);

    for (size_t i = 0; i < x->size; ++i) {
        LMMC_REAL_ABS(&abs_val, &x->data[i]);
        LMMC_REAL_ADD(&tmp_sum, &sum, &abs_val);
        LMMC_REAL_SET(&sum, &tmp_sum);
    }
    LMMC_REAL_SET(out_asum, &sum);

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&abs_val);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_iamax(const lmmc_vec_t* x, size_t* out_idx) {
    if (x == NULL || out_idx == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t max_val; LMMC_REAL_INIT(&max_val);
    lmmc_real_t abs_val; LMMC_REAL_INIT(&abs_val);
    size_t max_idx = 0;

    LMMC_REAL_ABS(&max_val, &x->data[0]);

    for (size_t i = 1; i < x->size; ++i) {
        LMMC_REAL_ABS(&abs_val, &x->data[i]);
        if (LMMC_REAL_CMP(&abs_val, &max_val) > 0) {
            LMMC_REAL_SET(&max_val, &abs_val);
            max_idx = i;
        }
    }
    *out_idx = max_idx;

    LMMC_REAL_CLEAR(&max_val);
    LMMC_REAL_CLEAR(&abs_val);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_add(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c) {
    if (a == NULL || b == NULL || c == NULL ||
        a->data == NULL || b->data == NULL || c->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != c->rows || a->cols != c->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);

    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            LMMC_REAL_ADD(&tmp_sum,
                          &a->data[i * a->stride + j],
                          &b->data[i * b->stride + j]);
            LMMC_REAL_SET(&c->data[i * c->stride + j], &tmp_sum);
        }
    }

    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_sub(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c) {
    if (a == NULL || b == NULL || c == NULL ||
        a->data == NULL || b->data == NULL || c->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != c->rows || a->cols != c->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    lmmc_real_t tmp_sub; LMMC_REAL_INIT(&tmp_sub);

    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            LMMC_REAL_SUB(&tmp_sub,
                          &a->data[i * a->stride + j],
                          &b->data[i * b->stride + j]);
            LMMC_REAL_SET(&c->data[i * c->stride + j], &tmp_sub);
        }
    }

    LMMC_REAL_CLEAR(&tmp_sub);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_scale(lmmc_mat_t* a, lmmc_real_t alpha) {
    if (a == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);

    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            LMMC_REAL_MUL(&tmp_mul, &a->data[i * a->stride + j], &alpha);
            LMMC_REAL_SET(&a->data[i * a->stride + j], &tmp_mul);
        }
    }

    LMMC_REAL_CLEAR(&tmp_mul);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_identity(size_t n, lmmc_mat_t* out_mat) {
    if (out_mat == NULL || n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_status_t status = lmmc_mat_create(n, n, out_mat);
    if (status != LMMC_STATUS_OK) {
        return status;
    }

    lmmc_real_t one; LMMC_REAL_INIT(&one);
    LMMC_REAL_SET_D(&one, 1.0);

    for (size_t i = 0; i < n; ++i) {
        LMMC_REAL_SET(&out_mat->data[i * out_mat->stride + i], &one);
    }

    LMMC_REAL_CLEAR(&one);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_trace(const lmmc_mat_t* a, lmmc_real_t* out_trace) {
    if (a == NULL || out_trace == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t sum; LMMC_REAL_INIT(&sum);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_SET_D(&sum, 0.0);

    for (size_t i = 0; i < a->rows; ++i) {
        LMMC_REAL_ADD(&tmp_sum, &sum, &a->data[i * a->stride + i]);
        LMMC_REAL_SET(&sum, &tmp_sum);
    }
    LMMC_REAL_SET(out_trace, &sum);

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_norm1(const lmmc_mat_t* a, lmmc_real_t* out_norm) {
    size_t i, j;
    lmmc_real_t col_sum;
    lmmc_real_t abs_val;
    lmmc_real_t tmp_sum;
    lmmc_real_t max_val;

    if (a == NULL || out_norm == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_INIT(&col_sum);
    LMMC_REAL_INIT(&abs_val);
    LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_INIT(&max_val);
    LMMC_REAL_SET_D(&max_val, 0.0);

    for (j = 0; j < a->cols; ++j) {
        LMMC_REAL_SET_D(&col_sum, 0.0);
        for (i = 0; i < a->rows; ++i) {
            LMMC_REAL_ABS(&abs_val, &a->data[i * a->stride + j]);
            LMMC_REAL_ADD(&tmp_sum, &col_sum, &abs_val);
            LMMC_REAL_SET(&col_sum, &tmp_sum);
        }
        if (LMMC_REAL_CMP(&col_sum, &max_val) > 0) {
            LMMC_REAL_SET(&max_val, &col_sum);
        }
    }

    LMMC_REAL_SET(out_norm, &max_val);

    LMMC_REAL_CLEAR(&col_sum);
    LMMC_REAL_CLEAR(&abs_val);
    LMMC_REAL_CLEAR(&tmp_sum);
    LMMC_REAL_CLEAR(&max_val);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_norm_inf(const lmmc_mat_t* a, lmmc_real_t* out_norm) {
    size_t i, j;
    lmmc_real_t row_sum;
    lmmc_real_t abs_val;
    lmmc_real_t tmp_sum;
    lmmc_real_t max_val;

    if (a == NULL || out_norm == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_INIT(&row_sum);
    LMMC_REAL_INIT(&abs_val);
    LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_INIT(&max_val);
    LMMC_REAL_SET_D(&max_val, 0.0);

    for (i = 0; i < a->rows; ++i) {
        LMMC_REAL_SET_D(&row_sum, 0.0);
        for (j = 0; j < a->cols; ++j) {
            LMMC_REAL_ABS(&abs_val, &a->data[i * a->stride + j]);
            LMMC_REAL_ADD(&tmp_sum, &row_sum, &abs_val);
            LMMC_REAL_SET(&row_sum, &tmp_sum);
        }
        if (LMMC_REAL_CMP(&row_sum, &max_val) > 0) {
            LMMC_REAL_SET(&max_val, &row_sum);
        }
    }

    LMMC_REAL_SET(out_norm, &max_val);

    LMMC_REAL_CLEAR(&row_sum);
    LMMC_REAL_CLEAR(&abs_val);
    LMMC_REAL_CLEAR(&tmp_sum);
    LMMC_REAL_CLEAR(&max_val);
    return LMMC_STATUS_OK;
}

