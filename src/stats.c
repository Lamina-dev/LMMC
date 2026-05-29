/**
 * @file stats.c
 * @brief 基础统计量与组合数学实现。
 */
#include <math.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"
#include "lammp/lmmp.h"
#include "lammp/numth.h"

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

static int lmmc_is_finite_number(lmmc_real_t v) {
    return LMMC_REAL_IS_FINITE(&v) ? 1 : 0;
}

static lmmc_status_t lmmc_validate_vec(const lmmc_vec_t* x) {
    if (x == NULL || x->data == NULL || x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_validate_mat(const lmmc_mat_t* x) {
    if (x == NULL || x->data == NULL || x->rows == 0 || x->cols == 0 || x->stride < x->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_finalize_nonnegative(lmmc_real_t value, lmmc_real_t* out_value) {
    lmmc_real_t zero, tol;
    if (!lmmc_is_finite_number(value) || out_value == NULL) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    LMMC_REAL_INIT(&zero);
    LMMC_REAL_INIT(&tol);

    LMMC_REAL_SET_D(&zero, 0.0);
    LMMC_REAL_SET_D(&tol, LMMC_REAL_EPSILON);
    LMMC_REAL_NEG(&tol, &tol);

    if (LMMC_REAL_CMP(&value, &zero) < 0) {
        if (LMMC_REAL_CMP(&value, &tol) > 0) {
            LMMC_REAL_SET_D(&value, 0.0);
        } else {
            LMMC_REAL_CLEAR(&zero);
            LMMC_REAL_CLEAR(&tol);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    LMMC_REAL_SET(out_value, &value);

    LMMC_REAL_CLEAR(&zero);
    LMMC_REAL_CLEAR(&tol);
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_mean_m2(const lmmc_vec_t* x, lmmc_real_t* out_mean, lmmc_real_t* out_m2) {
    size_t i = 0;
    lmmc_real_t mean, m2;
    lmmc_status_t st = lmmc_validate_vec(x);

    if (out_mean == NULL || out_m2 == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    LMMC_REAL_INIT(&mean);
    LMMC_REAL_INIT(&m2);

    LMMC_REAL_SET_D(&mean, 0.0);
    LMMC_REAL_SET_D(&m2, 0.0);

    for (i = 0; i < x->size; ++i) {
        lmmc_real_t v, n, delta, delta2, tmp;

        LMMC_REAL_INIT(&v);
        LMMC_REAL_INIT(&n);
        LMMC_REAL_INIT(&delta);
        LMMC_REAL_INIT(&delta2);
        LMMC_REAL_INIT(&tmp);

        LMMC_REAL_SET(&v, &x->data[i]);
        LMMC_REAL_SET_D(&n, (double)(i + 1));

        if (!lmmc_is_finite_number(v)) {
            LMMC_REAL_CLEAR(&tmp);
            LMMC_REAL_CLEAR(&delta2);
            LMMC_REAL_CLEAR(&delta);
            LMMC_REAL_CLEAR(&n);
            LMMC_REAL_CLEAR(&v);
            LMMC_REAL_CLEAR(&mean);
            LMMC_REAL_CLEAR(&m2);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        LMMC_REAL_SUB(&delta, &v, &mean);
        LMMC_REAL_DIV(&tmp, &delta, &n);
        LMMC_REAL_ADD(&mean, &mean, &tmp);
        LMMC_REAL_SUB(&delta2, &v, &mean);
        LMMC_REAL_MUL(&tmp, &delta, &delta2);
        LMMC_REAL_ADD(&m2, &m2, &tmp);

        LMMC_REAL_CLEAR(&tmp);
        LMMC_REAL_CLEAR(&delta2);
        LMMC_REAL_CLEAR(&delta);
        LMMC_REAL_CLEAR(&n);
        LMMC_REAL_CLEAR(&v);
    }

    if (!lmmc_is_finite_number(mean) || !lmmc_is_finite_number(m2)) {
        LMMC_REAL_CLEAR(&mean);
        LMMC_REAL_CLEAR(&m2);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    LMMC_REAL_SET(out_mean, &mean);
    LMMC_REAL_SET(out_m2, &m2);

    LMMC_REAL_CLEAR(&mean);
    LMMC_REAL_CLEAR(&m2);
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_cov_accumulate(
    const lmmc_vec_t* x,
    const lmmc_vec_t* y,
    lmmc_real_t* out_c,
    lmmc_real_t* out_m2x,
    lmmc_real_t* out_m2y
) {
    size_t i = 0;
    lmmc_real_t mean_x, mean_y, c, m2x, m2y;
    lmmc_status_t stx = lmmc_validate_vec(x);
    lmmc_status_t sty = lmmc_validate_vec(y);

    if (out_c == NULL || out_m2x == NULL || out_m2y == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (stx != LMMC_STATUS_OK || sty != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size != y->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    LMMC_REAL_INIT(&mean_x);
    LMMC_REAL_INIT(&mean_y);
    LMMC_REAL_INIT(&c);
    LMMC_REAL_INIT(&m2x);
    LMMC_REAL_INIT(&m2y);

    LMMC_REAL_SET_D(&mean_x, 0.0);
    LMMC_REAL_SET_D(&mean_y, 0.0);
    LMMC_REAL_SET_D(&c, 0.0);
    LMMC_REAL_SET_D(&m2x, 0.0);
    LMMC_REAL_SET_D(&m2y, 0.0);

    for (i = 0; i < x->size; ++i) {
        lmmc_real_t vx, vy, n, dx, dy, tmp1, tmp2;

        LMMC_REAL_INIT(&vx);
        LMMC_REAL_INIT(&vy);
        LMMC_REAL_INIT(&n);
        LMMC_REAL_INIT(&dx);
        LMMC_REAL_INIT(&dy);
        LMMC_REAL_INIT(&tmp1);
        LMMC_REAL_INIT(&tmp2);

        LMMC_REAL_SET(&vx, &x->data[i]);
        LMMC_REAL_SET(&vy, &y->data[i]);
        LMMC_REAL_SET_D(&n, (double)(i + 1));

        if (!lmmc_is_finite_number(vx) || !lmmc_is_finite_number(vy)) {
            LMMC_REAL_CLEAR(&tmp2);
            LMMC_REAL_CLEAR(&tmp1);
            LMMC_REAL_CLEAR(&dy);
            LMMC_REAL_CLEAR(&dx);
            LMMC_REAL_CLEAR(&n);
            LMMC_REAL_CLEAR(&vy);
            LMMC_REAL_CLEAR(&vx);
            LMMC_REAL_CLEAR(&m2y);
            LMMC_REAL_CLEAR(&m2x);
            LMMC_REAL_CLEAR(&c);
            LMMC_REAL_CLEAR(&mean_y);
            LMMC_REAL_CLEAR(&mean_x);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        LMMC_REAL_SUB(&dx, &vx, &mean_x);
        LMMC_REAL_SUB(&dy, &vy, &mean_y);

        LMMC_REAL_DIV(&tmp1, &dx, &n);
        LMMC_REAL_ADD(&mean_x, &mean_x, &tmp1);

        LMMC_REAL_DIV(&tmp1, &dy, &n);
        LMMC_REAL_ADD(&mean_y, &mean_y, &tmp1);

        LMMC_REAL_SUB(&tmp2, &vy, &mean_y);
        LMMC_REAL_MUL(&tmp1, &dx, &tmp2);
        LMMC_REAL_ADD(&c, &c, &tmp1);

        LMMC_REAL_SUB(&tmp2, &vx, &mean_x);
        LMMC_REAL_MUL(&tmp1, &dx, &tmp2);
        LMMC_REAL_ADD(&m2x, &m2x, &tmp1);

        LMMC_REAL_SUB(&tmp2, &vy, &mean_y);
        LMMC_REAL_MUL(&tmp1, &dy, &tmp2);
        LMMC_REAL_ADD(&m2y, &m2y, &tmp1);

        LMMC_REAL_CLEAR(&tmp2);
        LMMC_REAL_CLEAR(&tmp1);
        LMMC_REAL_CLEAR(&dy);
        LMMC_REAL_CLEAR(&dx);
        LMMC_REAL_CLEAR(&n);
        LMMC_REAL_CLEAR(&vy);
        LMMC_REAL_CLEAR(&vx);
    }

    if (!lmmc_is_finite_number(c) || !lmmc_is_finite_number(m2x) || !lmmc_is_finite_number(m2y)) {
        LMMC_REAL_CLEAR(&m2y);
        LMMC_REAL_CLEAR(&m2x);
        LMMC_REAL_CLEAR(&c);
        LMMC_REAL_CLEAR(&mean_y);
        LMMC_REAL_CLEAR(&mean_x);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    LMMC_REAL_SET(out_c, &c);
    LMMC_REAL_SET(out_m2x, &m2x);
    LMMC_REAL_SET(out_m2y, &m2y);

    LMMC_REAL_CLEAR(&m2y);
    LMMC_REAL_CLEAR(&m2x);
    LMMC_REAL_CLEAR(&c);
    LMMC_REAL_CLEAR(&mean_y);
    LMMC_REAL_CLEAR(&mean_x);
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_variance_common(const lmmc_vec_t* x, int sample, lmmc_real_t* out_variance) {
    lmmc_real_t mean, m2, denom, variance;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (out_variance == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_INIT(&mean);
    LMMC_REAL_INIT(&m2);
    LMMC_REAL_INIT(&denom);
    LMMC_REAL_INIT(&variance);

    LMMC_REAL_SET_D(&mean, 0.0);
    LMMC_REAL_SET_D(&m2, 0.0);
    LMMC_REAL_SET_D(&denom, 0.0);
    LMMC_REAL_SET_D(&variance, 0.0);

    st = lmmc_vec_mean_m2(x, &mean, &m2);
    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&variance);
        LMMC_REAL_CLEAR(&denom);
        LMMC_REAL_CLEAR(&m2);
        LMMC_REAL_CLEAR(&mean);
        return st;
    }

    if (sample) {
        if (x->size < 2) {
            LMMC_REAL_CLEAR(&variance);
            LMMC_REAL_CLEAR(&denom);
            LMMC_REAL_CLEAR(&m2);
            LMMC_REAL_CLEAR(&mean);
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        LMMC_REAL_SET_D(&denom, (double)(x->size - 1));
    } else {
        LMMC_REAL_SET_D(&denom, (double)x->size);
    }

    LMMC_REAL_DIV(&variance, &m2, &denom);
    st = lmmc_finalize_nonnegative(variance, out_variance);

    LMMC_REAL_CLEAR(&variance);
    LMMC_REAL_CLEAR(&denom);
    LMMC_REAL_CLEAR(&m2);
    LMMC_REAL_CLEAR(&mean);
    return st;
}

static lmmc_status_t lmmc_vec_covariance_common(
    const lmmc_vec_t* x,
    const lmmc_vec_t* y,
    int sample,
    lmmc_real_t* out_covariance
) {
    lmmc_real_t c, m2x, m2y, denom, cov;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (out_covariance == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_INIT(&c);
    LMMC_REAL_INIT(&m2x);
    LMMC_REAL_INIT(&m2y);
    LMMC_REAL_INIT(&denom);
    LMMC_REAL_INIT(&cov);

    LMMC_REAL_SET_D(&c, 0.0);
    LMMC_REAL_SET_D(&m2x, 0.0);
    LMMC_REAL_SET_D(&m2y, 0.0);
    LMMC_REAL_SET_D(&denom, 0.0);
    LMMC_REAL_SET_D(&cov, 0.0);

    st = lmmc_vec_cov_accumulate(x, y, &c, &m2x, &m2y);
    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&cov);
        LMMC_REAL_CLEAR(&denom);
        LMMC_REAL_CLEAR(&m2y);
        LMMC_REAL_CLEAR(&m2x);
        LMMC_REAL_CLEAR(&c);
        return st;
    }

    if (sample) {
        if (x->size < 2) {
            LMMC_REAL_CLEAR(&cov);
            LMMC_REAL_CLEAR(&denom);
            LMMC_REAL_CLEAR(&m2y);
            LMMC_REAL_CLEAR(&m2x);
            LMMC_REAL_CLEAR(&c);
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        LMMC_REAL_SET_D(&denom, (double)(x->size - 1));
    } else {
        LMMC_REAL_SET_D(&denom, (double)x->size);
    }

    LMMC_REAL_DIV(&cov, &c, &denom);
    if (!lmmc_is_finite_number(cov)) {
        LMMC_REAL_CLEAR(&cov);
        LMMC_REAL_CLEAR(&denom);
        LMMC_REAL_CLEAR(&m2y);
        LMMC_REAL_CLEAR(&m2x);
        LMMC_REAL_CLEAR(&c);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    LMMC_REAL_SET(out_covariance, &cov);

    LMMC_REAL_CLEAR(&cov);
    LMMC_REAL_CLEAR(&denom);
    LMMC_REAL_CLEAR(&m2y);
    LMMC_REAL_CLEAR(&m2x);
    LMMC_REAL_CLEAR(&c);
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_correlation_common(
    const lmmc_vec_t* x,
    const lmmc_vec_t* y,
    int sample,
    lmmc_real_t* out_correlation
) {
    lmmc_real_t c, m2x, m2y, corr, tmp, zero, one, min_one, tol, upper, lower;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (out_correlation == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_INIT(&c);
    LMMC_REAL_INIT(&m2x);
    LMMC_REAL_INIT(&m2y);
    LMMC_REAL_INIT(&corr);
    LMMC_REAL_INIT(&tmp);
    LMMC_REAL_INIT(&zero);
    LMMC_REAL_INIT(&one);
    LMMC_REAL_INIT(&min_one);
    LMMC_REAL_INIT(&tol);
    LMMC_REAL_INIT(&upper);
    LMMC_REAL_INIT(&lower);

    LMMC_REAL_SET_D(&c, 0.0);
    LMMC_REAL_SET_D(&m2x, 0.0);
    LMMC_REAL_SET_D(&m2y, 0.0);
    LMMC_REAL_SET_D(&corr, 0.0);
    LMMC_REAL_SET_D(&zero, 0.0);
    LMMC_REAL_SET_D(&one, 1.0);
    LMMC_REAL_SET_D(&min_one, -1.0);
    LMMC_REAL_SET_D(&tol, 1e-12);

    st = lmmc_vec_cov_accumulate(x, y, &c, &m2x, &m2y);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }

    if (sample && x->size < 2) {
        st = LMMC_STATUS_INVALID_ARGUMENT;
        goto cleanup;
    }

    if (LMMC_REAL_CMP(&m2x, &zero) <= 0 || LMMC_REAL_CMP(&m2y, &zero) <= 0) {
        st = LMMC_STATUS_NUMERICAL_FAILURE;
        goto cleanup;
    }

    LMMC_REAL_MUL(&tmp, &m2x, &m2y);
    LMMC_REAL_SQRT(&tmp, &tmp);
    LMMC_REAL_DIV(&corr, &c, &tmp);

    if (!lmmc_is_finite_number(corr)) {
        st = LMMC_STATUS_NUMERICAL_FAILURE;
        goto cleanup;
    }

    LMMC_REAL_ADD(&upper, &one, &tol);
    if (LMMC_REAL_CMP(&corr, &one) > 0 && LMMC_REAL_CMP(&corr, &upper) < 0) {
        LMMC_REAL_SET(&corr, &one);
    }

    LMMC_REAL_SUB(&lower, &min_one, &tol);
    if (LMMC_REAL_CMP(&corr, &min_one) < 0 && LMMC_REAL_CMP(&corr, &lower) > 0) {
        LMMC_REAL_SET(&corr, &min_one);
    }

    if (LMMC_REAL_CMP(&corr, &min_one) < 0 || LMMC_REAL_CMP(&corr, &one) > 0) {
        st = LMMC_STATUS_NUMERICAL_FAILURE;
        goto cleanup;
    }

    LMMC_REAL_SET(out_correlation, &corr);

cleanup:
    LMMC_REAL_CLEAR(&lower);
    LMMC_REAL_CLEAR(&upper);
    LMMC_REAL_CLEAR(&tol);
    LMMC_REAL_CLEAR(&min_one);
    LMMC_REAL_CLEAR(&one);
    LMMC_REAL_CLEAR(&zero);
    LMMC_REAL_CLEAR(&tmp);
    LMMC_REAL_CLEAR(&corr);
    LMMC_REAL_CLEAR(&m2y);
    LMMC_REAL_CLEAR(&m2x);
    LMMC_REAL_CLEAR(&c);

    return st;
}

static lmmc_status_t lmmc_mat_column_means_to_buffer(const lmmc_mat_t* x, lmmc_real_t* means) {
    size_t col = 0;
    lmmc_status_t st = lmmc_validate_mat(x);

    if (st != LMMC_STATUS_OK || means == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (col = 0; col < x->cols; ++col) {
        size_t row = 0;
        lmmc_real_t mean;
        LMMC_REAL_INIT(&mean);
        LMMC_REAL_SET_D(&mean, 0.0);

        for (row = 0; row < x->rows; ++row) {
            lmmc_real_t v, n, delta, tmp;
            LMMC_REAL_INIT(&v);
            LMMC_REAL_INIT(&n);
            LMMC_REAL_INIT(&delta);
            LMMC_REAL_INIT(&tmp);

            LMMC_REAL_SET(&v, &x->data[row * x->stride + col]);
            LMMC_REAL_SET_D(&n, (double)(row + 1));

            if (!lmmc_is_finite_number(v)) {
                LMMC_REAL_CLEAR(&tmp);
                LMMC_REAL_CLEAR(&delta);
                LMMC_REAL_CLEAR(&n);
                LMMC_REAL_CLEAR(&v);
                LMMC_REAL_CLEAR(&mean);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            LMMC_REAL_SUB(&delta, &v, &mean);
            LMMC_REAL_DIV(&tmp, &delta, &n);
            LMMC_REAL_ADD(&mean, &mean, &tmp);

            LMMC_REAL_CLEAR(&tmp);
            LMMC_REAL_CLEAR(&delta);
            LMMC_REAL_CLEAR(&n);
            LMMC_REAL_CLEAR(&v);
        }

        if (!lmmc_is_finite_number(mean)) {
            LMMC_REAL_CLEAR(&mean);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        LMMC_REAL_SET(&means[col], &mean);
        LMMC_REAL_CLEAR(&mean);
    }

    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_mat_covariance_or_correlation(
    const lmmc_mat_t* x,
    int sample,
    int correlation,
    lmmc_mat_t* out_matrix
) {
    size_t bytes = 0;
    size_t col_i = 0;
    size_t col_j = 0;
    lmmc_real_t denom;
    lmmc_real_t* means = NULL;
    lmmc_real_t* stddev = NULL;
    lmmc_status_t st = lmmc_validate_mat(x);

    if (st != LMMC_STATUS_OK || out_matrix == NULL || out_matrix->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (out_matrix->rows != x->cols || out_matrix->cols != x->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    if (sample && x->rows < 2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_INIT(&denom);
    if (sample) {
        LMMC_REAL_SET_D(&denom, (double)(x->rows - 1));
    } else {
        LMMC_REAL_SET_D(&denom, (double)x->rows);
    }

    if (lmmc_mul_overflow_size(x->cols, sizeof(lmmc_real_t), &bytes)) {
        LMMC_REAL_CLEAR(&denom);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    means = (lmmc_real_t*)lmmc_alloc(bytes);
    if (means == NULL) {
        LMMC_REAL_CLEAR(&denom);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_INIT(&means[k]);

    if (correlation) {
        stddev = (lmmc_real_t*)lmmc_alloc(bytes);
        if (stddev == NULL) {
            for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&means[k]);
            lmmc_free(means);
            LMMC_REAL_CLEAR(&denom);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_INIT(&stddev[k]);
    }

    st = lmmc_mat_column_means_to_buffer(x, means);
    if (st != LMMC_STATUS_OK) {
        if (correlation) {
            for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&stddev[k]);
            lmmc_free(stddev);
        }
        for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&means[k]);
        lmmc_free(means);
        LMMC_REAL_CLEAR(&denom);
        return st;
    }

    if (correlation) {
        for (col_i = 0; col_i < x->cols; ++col_i) {
            size_t row = 0;
            lmmc_real_t sumsq, variance, d, tmp;
            LMMC_REAL_INIT(&sumsq);
            LMMC_REAL_INIT(&variance);
            LMMC_REAL_INIT(&d);
            LMMC_REAL_INIT(&tmp);

            LMMC_REAL_SET_D(&sumsq, 0.0);

            for (row = 0; row < x->rows; ++row) {
                LMMC_REAL_SUB(&d, &x->data[row * x->stride + col_i], &means[col_i]);
                LMMC_REAL_MUL(&tmp, &d, &d);
                LMMC_REAL_ADD(&sumsq, &sumsq, &tmp);
            }

            LMMC_REAL_DIV(&variance, &sumsq, &denom);
            st = lmmc_finalize_nonnegative(variance, &variance);

            lmmc_real_t zero;
            LMMC_REAL_INIT(&zero);
            LMMC_REAL_SET_D(&zero, 0.0);

            if (st != LMMC_STATUS_OK || LMMC_REAL_CMP(&variance, &zero) <= 0) {
                LMMC_REAL_CLEAR(&zero);
                LMMC_REAL_CLEAR(&tmp);
                LMMC_REAL_CLEAR(&d);
                LMMC_REAL_CLEAR(&variance);
                LMMC_REAL_CLEAR(&sumsq);
                if (correlation) {
                    for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&stddev[k]);
                    lmmc_free(stddev);
                }
                for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&means[k]);
                lmmc_free(means);
                LMMC_REAL_CLEAR(&denom);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            LMMC_REAL_CLEAR(&zero);

            LMMC_REAL_SQRT(&stddev[col_i], &variance);

            if (!lmmc_is_finite_number(stddev[col_i])) {
                LMMC_REAL_CLEAR(&tmp);
                LMMC_REAL_CLEAR(&d);
                LMMC_REAL_CLEAR(&variance);
                LMMC_REAL_CLEAR(&sumsq);
                for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&stddev[k]);
                lmmc_free(stddev);
                for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&means[k]);
                lmmc_free(means);
                LMMC_REAL_CLEAR(&denom);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            LMMC_REAL_CLEAR(&tmp);
            LMMC_REAL_CLEAR(&d);
            LMMC_REAL_CLEAR(&variance);
            LMMC_REAL_CLEAR(&sumsq);
        }
    }

    for (col_i = 0; col_i < x->cols; ++col_i) {
        for (col_j = col_i; col_j < x->cols; ++col_j) {
            size_t row = 0;
            lmmc_real_t sum, cov, out_value, di, dj, tmp;
            LMMC_REAL_INIT(&sum);
            LMMC_REAL_INIT(&cov);
            LMMC_REAL_INIT(&out_value);
            LMMC_REAL_INIT(&di);
            LMMC_REAL_INIT(&dj);
            LMMC_REAL_INIT(&tmp);

            LMMC_REAL_SET_D(&sum, 0.0);

            for (row = 0; row < x->rows; ++row) {
                LMMC_REAL_SUB(&di, &x->data[row * x->stride + col_i], &means[col_i]);
                LMMC_REAL_SUB(&dj, &x->data[row * x->stride + col_j], &means[col_j]);
                LMMC_REAL_MUL(&tmp, &di, &dj);
                LMMC_REAL_ADD(&sum, &sum, &tmp);
            }

            LMMC_REAL_DIV(&cov, &sum, &denom);
            if (!lmmc_is_finite_number(cov)) {
                LMMC_REAL_CLEAR(&tmp);
                LMMC_REAL_CLEAR(&dj);
                LMMC_REAL_CLEAR(&di);
                LMMC_REAL_CLEAR(&out_value);
                LMMC_REAL_CLEAR(&cov);
                LMMC_REAL_CLEAR(&sum);
                if (correlation) {
                    for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&stddev[k]);
                    lmmc_free(stddev);
                }
                for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&means[k]);
                lmmc_free(means);
                LMMC_REAL_CLEAR(&denom);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            if (correlation) {
                if (col_i == col_j) {
                    LMMC_REAL_SET_D(&out_value, 1.0);
                } else {
                    LMMC_REAL_MUL(&tmp, &stddev[col_i], &stddev[col_j]);
                    LMMC_REAL_DIV(&out_value, &cov, &tmp);
                    if (!lmmc_is_finite_number(out_value)) {
                        LMMC_REAL_CLEAR(&tmp);
                        LMMC_REAL_CLEAR(&dj);
                        LMMC_REAL_CLEAR(&di);
                        LMMC_REAL_CLEAR(&out_value);
                        LMMC_REAL_CLEAR(&cov);
                        LMMC_REAL_CLEAR(&sum);
                        for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&stddev[k]);
                        lmmc_free(stddev);
                        for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&means[k]);
                        lmmc_free(means);
                        LMMC_REAL_CLEAR(&denom);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }

                    lmmc_real_t one, min_one, tol, upper, lower;
                    LMMC_REAL_INIT(&one);
                    LMMC_REAL_INIT(&min_one);
                    LMMC_REAL_INIT(&tol);
                    LMMC_REAL_INIT(&upper);
                    LMMC_REAL_INIT(&lower);

                    LMMC_REAL_SET_D(&one, 1.0);
                    LMMC_REAL_SET_D(&min_one, -1.0);
                    LMMC_REAL_SET_D(&tol, 1e-12);

                    LMMC_REAL_ADD(&upper, &one, &tol);
                    if (LMMC_REAL_CMP(&out_value, &one) > 0 && LMMC_REAL_CMP(&out_value, &upper) < 0) {
                        LMMC_REAL_SET(&out_value, &one);
                    }

                    LMMC_REAL_SUB(&lower, &min_one, &tol);
                    if (LMMC_REAL_CMP(&out_value, &min_one) < 0 && LMMC_REAL_CMP(&out_value, &lower) > 0) {
                        LMMC_REAL_SET(&out_value, &min_one);
                    }

                    if (LMMC_REAL_CMP(&out_value, &min_one) < 0 || LMMC_REAL_CMP(&out_value, &one) > 0) {
                        LMMC_REAL_CLEAR(&lower);
                        LMMC_REAL_CLEAR(&upper);
                        LMMC_REAL_CLEAR(&tol);
                        LMMC_REAL_CLEAR(&min_one);
                        LMMC_REAL_CLEAR(&one);

                        LMMC_REAL_CLEAR(&tmp);
                        LMMC_REAL_CLEAR(&dj);
                        LMMC_REAL_CLEAR(&di);
                        LMMC_REAL_CLEAR(&out_value);
                        LMMC_REAL_CLEAR(&cov);
                        LMMC_REAL_CLEAR(&sum);
                        for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&stddev[k]);
                        lmmc_free(stddev);
                        for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&means[k]);
                        lmmc_free(means);
                        LMMC_REAL_CLEAR(&denom);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }

                    LMMC_REAL_CLEAR(&lower);
                    LMMC_REAL_CLEAR(&upper);
                    LMMC_REAL_CLEAR(&tol);
                    LMMC_REAL_CLEAR(&min_one);
                    LMMC_REAL_CLEAR(&one);
                }
            } else {
                LMMC_REAL_SET(&out_value, &cov);
            }

            LMMC_REAL_SET(&out_matrix->data[col_i * out_matrix->stride + col_j], &out_value);
            LMMC_REAL_SET(&out_matrix->data[col_j * out_matrix->stride + col_i], &out_value);

            LMMC_REAL_CLEAR(&tmp);
            LMMC_REAL_CLEAR(&dj);
            LMMC_REAL_CLEAR(&di);
            LMMC_REAL_CLEAR(&out_value);
            LMMC_REAL_CLEAR(&cov);
            LMMC_REAL_CLEAR(&sum);
        }
    }

    if (correlation) {
        for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&stddev[k]);
        lmmc_free(stddev);
    }
    for (size_t k = 0; k < x->cols; ++k) LMMC_REAL_CLEAR(&means[k]);
    lmmc_free(means);
    LMMC_REAL_CLEAR(&denom);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_mean(const lmmc_vec_t* x, lmmc_real_t* out_mean) {
    lmmc_real_t mean = 0.0;
    lmmc_real_t m2 = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (out_mean == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_vec_mean_m2(x, &mean, &m2);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    *out_mean = mean;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_variance_population(const lmmc_vec_t* x, lmmc_real_t* out_variance) {
    return lmmc_vec_variance_common(x, 0, out_variance);
}

lmmc_status_t lmmc_vec_variance_sample(const lmmc_vec_t* x, lmmc_real_t* out_variance) {
    return lmmc_vec_variance_common(x, 1, out_variance);
}

lmmc_status_t lmmc_vec_stddev_population(const lmmc_vec_t* x, lmmc_real_t* out_stddev) {
    lmmc_real_t variance = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (out_stddev == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_vec_variance_common(x, 0, &variance);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    *out_stddev = variance;
    LMMC_REAL_SQRT(out_stddev, out_stddev);
    if (!lmmc_is_finite_number(*out_stddev)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_stddev_sample(const lmmc_vec_t* x, lmmc_real_t* out_stddev) {
    lmmc_real_t variance = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (out_stddev == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_vec_variance_common(x, 1, &variance);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    *out_stddev = variance;
    LMMC_REAL_SQRT(out_stddev, out_stddev);
    if (!lmmc_is_finite_number(*out_stddev)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_covariance_population(const lmmc_vec_t* x, const lmmc_vec_t* y, lmmc_real_t* out_covariance) {
    return lmmc_vec_covariance_common(x, y, 0, out_covariance);
}

lmmc_status_t lmmc_vec_covariance_sample(const lmmc_vec_t* x, const lmmc_vec_t* y, lmmc_real_t* out_covariance) {
    return lmmc_vec_covariance_common(x, y, 1, out_covariance);
}

lmmc_status_t lmmc_vec_correlation_population(const lmmc_vec_t* x, const lmmc_vec_t* y, lmmc_real_t* out_correlation) {
    return lmmc_vec_correlation_common(x, y, 0, out_correlation);
}

lmmc_status_t lmmc_vec_correlation_sample(const lmmc_vec_t* x, const lmmc_vec_t* y, lmmc_real_t* out_correlation) {
    return lmmc_vec_correlation_common(x, y, 1, out_correlation);
}

lmmc_status_t lmmc_mat_column_mean(const lmmc_mat_t* x, lmmc_vec_t* out_means) {
    lmmc_status_t st = lmmc_validate_mat(x);

    if (st != LMMC_STATUS_OK || out_means == NULL || out_means->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (out_means->size != x->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    return lmmc_mat_column_means_to_buffer(x, out_means->data);
}

lmmc_status_t lmmc_mat_covariance_population(const lmmc_mat_t* x, lmmc_mat_t* out_covariance) {
    return lmmc_mat_covariance_or_correlation(x, 0, 0, out_covariance);
}

lmmc_status_t lmmc_mat_covariance_sample(const lmmc_mat_t* x, lmmc_mat_t* out_covariance) {
    return lmmc_mat_covariance_or_correlation(x, 1, 0, out_covariance);
}

lmmc_status_t lmmc_mat_correlation_population(const lmmc_mat_t* x, lmmc_mat_t* out_correlation) {
    return lmmc_mat_covariance_or_correlation(x, 0, 1, out_correlation);
}

lmmc_status_t lmmc_mat_correlation_sample(const lmmc_mat_t* x, lmmc_mat_t* out_correlation) {
    return lmmc_mat_covariance_or_correlation(x, 1, 1, out_correlation);
}

static double lmmp_convert_to_double(mp_srcptr src, mp_size_t size) {
    double res = 0.0;
    if (size == 0) return 0.0;

    double scale = 1.0;
    double limb_radix = 1.0;
    int k;
    for (k = 0; k < LIMB_BITS; ++k) {
        limb_radix *= 2.0;
    }

    for (mp_size_t i = 0; i < size; ++i) {
        res += (double)src[i] * scale;
        scale *= limb_radix;
    }
    return res;
}

static void lmmp_to_lmmc_real(lmmc_real_t* dst, mp_srcptr src, mp_size_t size) {

    LMMC_REAL_SET_D(dst, lmmp_convert_to_double(src, size));


}

void lmmc_stats_factorial(lmmc_real_t* out_val, uint32_t n) {
    mp_bitcnt_t bits = 0;
    mp_size_t len = lmmp_factorial_size_(n, &bits);
    mp_ptr dst = (mp_ptr)lmmc_alloc(len * sizeof(mp_limb_t));
    if (dst == NULL) {
        LMMC_REAL_SET_D(out_val, -1.0);
        return;
    }

    mp_size_t an = lmmp_factorial_(dst, bits, len, n);
    lmmp_to_lmmc_real(out_val, dst, an);

    lmmc_free(dst);
}

void lmmc_stats_nPr(lmmc_real_t* out_val, uint32_t n, uint32_t r) {
    if (r > n) {
        LMMC_REAL_SET_D(out_val, 0.0);
        return;
    }
    mp_bitcnt_t bits = 0;
    mp_size_t len = lmmp_nPr_size_(n, r, &bits);
    mp_ptr dst = (mp_ptr)lmmc_alloc(len * sizeof(mp_limb_t));
    if (dst == NULL) {
        LMMC_REAL_SET_D(out_val, -1.0);
        return;
    }

    mp_size_t an = lmmp_nPr_(dst, bits, len, n, r);
    lmmp_to_lmmc_real(out_val, dst, an);

    lmmc_free(dst);
}

void lmmc_stats_nCr(lmmc_real_t* out_val, uint32_t n, uint32_t r) {
    if (r > n) {
        LMMC_REAL_SET_D(out_val, 0.0);
        return;
    }
    mp_bitcnt_t bits = 0;
    mp_size_t len = lmmp_nCr_size_(n, r, &bits);
    mp_ptr dst = (mp_ptr)lmmc_alloc(len * sizeof(mp_limb_t));
    if (dst == NULL) {
        LMMC_REAL_SET_D(out_val, -1.0);
        return;
    }

    mp_size_t an = lmmp_nCr_(dst, bits, len, n, r);
    lmmp_to_lmmc_real(out_val, dst, an);

    lmmc_free(dst);
}

/* ===================== 描述性统计 ===================== */

/**
 * @brief 比较函数，用于 qsort 排序 lmmc_real_t 数组。
 */
static int lmmc_real_compare(const void* a, const void* b) {
    lmmc_real_t va = *(const lmmc_real_t*)a;
    lmmc_real_t vb = *(const lmmc_real_t*)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

lmmc_status_t lmmc_vec_median(const lmmc_vec_t* x, lmmc_real_t* out) {
    lmmc_real_t* sorted = NULL;
    size_t n;

    if (out == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x == NULL || x->data == NULL || x->size == 0) return LMMC_STATUS_INVALID_ARGUMENT;

    n = x->size;
    sorted = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (sorted == NULL) return LMMC_STATUS_ALLOCATION_FAILED;

    for (size_t i = 0; i < n; ++i) {
        sorted[i] = x->data[i];
    }
    qsort(sorted, n, sizeof(lmmc_real_t), lmmc_real_compare);

    if (n % 2 == 1) {
        *out = sorted[n / 2];
    } else {
        *out = (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5;
    }

    lmmc_free(sorted);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_quantile(const lmmc_vec_t* x, lmmc_real_t p, lmmc_real_t* out) {
    lmmc_real_t* sorted = NULL;
    size_t n;
    double h, frac;
    size_t lo, hi;

    if (out == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x == NULL || x->data == NULL || x->size == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p < 0.0 || p > 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    n = x->size;
    sorted = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (sorted == NULL) return LMMC_STATUS_ALLOCATION_FAILED;

    for (size_t i = 0; i < n; ++i) {
        sorted[i] = x->data[i];
    }
    qsort(sorted, n, sizeof(lmmc_real_t), lmmc_real_compare);

    /* Linear interpolation method (same as numpy default) */
    h = p * (double)(n - 1);
    lo = (size_t)h;
    hi = lo + 1;
    frac = h - (double)lo;

    if (hi >= n) {
        *out = sorted[n - 1];
    } else {
        *out = sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }

    lmmc_free(sorted);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_histogram(const lmmc_vec_t* x, size_t nbins, lmmc_real_t* edges, size_t* counts) {
    size_t n, i;
    lmmc_real_t xmin, xmax, width;

    if (x == NULL || x->data == NULL || x->size == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (nbins == 0 || edges == NULL || counts == NULL) return LMMC_STATUS_INVALID_ARGUMENT;

    n = x->size;

    /* Find min and max */
    xmin = x->data[0];
    xmax = x->data[0];
    for (i = 1; i < n; ++i) {
        if (x->data[i] < xmin) xmin = x->data[i];
        if (x->data[i] > xmax) xmax = x->data[i];
    }

    /* Handle constant data */
    if (xmax == xmin) {
        xmin -= 0.5;
        xmax += 0.5;
    }

    width = (xmax - xmin) / (lmmc_real_t)nbins;

    /* Compute edges */
    for (i = 0; i <= nbins; ++i) {
        edges[i] = xmin + width * (lmmc_real_t)i;
    }

    /* Initialize counts */
    for (i = 0; i < nbins; ++i) {
        counts[i] = 0;
    }

    /* Bin the data */
    for (i = 0; i < n; ++i) {
        size_t bin;
        if (x->data[i] >= xmax) {
            bin = nbins - 1; /* last bin includes the right edge */
        } else {
            bin = (size_t)((x->data[i] - xmin) / width);
            if (bin >= nbins) bin = nbins - 1;
        }
        counts[bin]++;
    }

    return LMMC_STATUS_OK;
}

/* ===================== 概率分布辅助函数 ===================== */

#ifndef LMMC_SQRT_2PI
#define LMMC_SQRT_2PI 2.5066282746310002
#endif

#ifndef LMMC_LOG_SQRT_2PI
#define LMMC_LOG_SQRT_2PI 0.9189385332046727
#endif

/* Forward declaration */
static lmmc_real_t regularized_gamma_upper_cf(lmmc_real_t a, lmmc_real_t x);

/**
 * @brief 正则化不完全伽马函数下尾 P(a, x) = gamma(a, x) / Gamma(a)。
 * 使用级数展开或连分数展开。
 */
static lmmc_real_t regularized_gamma_lower(lmmc_real_t a, lmmc_real_t x) {
    if (x <= 0.0) return 0.0;
    if (x < a + 1.0) {
        /* Series expansion */
        lmmc_real_t sum = 1.0 / a;
        lmmc_real_t term = 1.0 / a;
        lmmc_real_t ap = a;
        int i;
        for (i = 0; i < 200; ++i) {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if (fabs(term) < fabs(sum) * 1e-15) break;
        }
        return sum * exp(-x + a * log(x) - lgamma(a));
    } else {
        /* Use complement: P = 1 - Q, where Q uses continued fraction */
        return 1.0 - regularized_gamma_upper_cf(a, x);
    }
}

/**
 * @brief 正则化不完全伽马函数上尾 Q(a, x) = 1 - P(a, x)。
 * 使用 Lentz 连分数展开。
 */
static lmmc_real_t regularized_gamma_upper_cf(lmmc_real_t a, lmmc_real_t x) {
    /* Lentz continued fraction for Q(a, x) */
    lmmc_real_t f, c, d, delta;
    int i;
    lmmc_real_t tiny = 1e-30;

    f = tiny;
    c = tiny;
    d = 0.0;

    /* CF: b0 = 0, a1 = 1, b1 = x - a + 1, a_i = (i-1)*(a - (i-1)), b_i = x - a + 2*i - 1 */
    /* Modified Lentz: start with b0 = x - a + 1 */
    {
        lmmc_real_t b0 = x - a + 1.0;
        if (fabs(b0) < tiny) b0 = tiny;
        f = b0;
        c = b0;
        d = 0.0;
    }

    for (i = 1; i <= 200; ++i) {
        lmmc_real_t an = (lmmc_real_t)i * (a - (lmmc_real_t)i);
        lmmc_real_t bn = x - a + 2.0 * (lmmc_real_t)i + 1.0;

        d = bn + an * d;
        if (fabs(d) < tiny) d = tiny;
        d = 1.0 / d;

        c = bn + an / c;
        if (fabs(c) < tiny) c = tiny;

        delta = c * d;
        f *= delta;
        if (fabs(delta - 1.0) < 1e-15) break;
    }

    return exp(-x + a * log(x) - lgamma(a)) / f;
}

/**
 * @brief 正则化不完全贝塔函数 I_x(a, b)。
 * 使用连分数展开（Lentz 方法）。
 */
static lmmc_real_t regularized_beta_cf(lmmc_real_t x, lmmc_real_t a, lmmc_real_t b) {
    lmmc_real_t qab, qap, qam, c, d, f;
    int m, m2;
    lmmc_real_t tiny = 1e-30;

    qab = a + b;
    qap = a + 1.0;
    qam = a - 1.0;

    c = 1.0;
    d = 1.0 - qab * x / qap;
    if (fabs(d) < tiny) d = tiny;
    d = 1.0 / d;
    f = d;

    for (m = 1; m <= 200; ++m) {
        lmmc_real_t aa, del;
        m2 = 2 * m;

        /* Even step */
        aa = (lmmc_real_t)m * (b - (lmmc_real_t)m) * x /
             ((qam + (lmmc_real_t)m2) * (a + (lmmc_real_t)m2));
        d = 1.0 + aa * d;
        if (fabs(d) < tiny) d = tiny;
        c = 1.0 + aa / c;
        if (fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        f *= c * d;

        /* Odd step */
        aa = -(a + (lmmc_real_t)m) * (qab + (lmmc_real_t)m) * x /
             ((a + (lmmc_real_t)m2) * (qap + (lmmc_real_t)m2));
        d = 1.0 + aa * d;
        if (fabs(d) < tiny) d = tiny;
        c = 1.0 + aa / c;
        if (fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        del = c * d;
        f *= del;

        if (fabs(del - 1.0) < 1e-15) break;
    }

    return f;
}

/**
 * @brief 正则化不完全贝塔函数 I_x(a, b)。
 */
static lmmc_real_t regularized_beta(lmmc_real_t x, lmmc_real_t a, lmmc_real_t b) {
    lmmc_real_t bt;
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) +
             a * log(x) + b * log(1.0 - x));

    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * regularized_beta_cf(x, a, b) / a;
    } else {
        return 1.0 - bt * regularized_beta_cf(1.0 - x, b, a) / b;
    }
}

/* ===================== 正态分布 ===================== */

lmmc_status_t lmmc_dist_normal_pdf(lmmc_real_t x, lmmc_real_t mu,
                                    lmmc_real_t sigma, lmmc_real_t* out) {
    lmmc_real_t z, exponent;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (sigma <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    z = (x - mu) / sigma;
    exponent = -0.5 * z * z;
    *out = exp(exponent) / (sigma * LMMC_SQRT_2PI);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_normal_cdf(lmmc_real_t x, lmmc_real_t mu,
                                    lmmc_real_t sigma, lmmc_real_t* out) {
    lmmc_real_t z;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (sigma <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    z = (x - mu) / sigma;
    *out = 0.5 * (1.0 + erf(z / sqrt(2.0)));
    return LMMC_STATUS_OK;
}

/**
 * @brief Rational approximation for the normal quantile (Beasley-Springer-Moro).
 */
static lmmc_real_t normal_quantile_rational(lmmc_real_t p) {
    /* Coefficients for rational approximation */
    static const double a[] = {
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00
    };

    double q, r;
    double plow = 0.02425;
    double phigh = 1.0 - plow;

    if (p < plow) {
        /* Lower tail */
        q = sqrt(-2.0 * log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= phigh) {
        /* Central region */
        q = p - 0.5;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        /* Upper tail */
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

lmmc_status_t lmmc_dist_normal_quantile(lmmc_real_t p, lmmc_real_t mu,
                                         lmmc_real_t sigma, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (sigma <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    *out = mu + sigma * normal_quantile_rational(p);
    return LMMC_STATUS_OK;
}

/* ===================== t 分布 ===================== */

lmmc_status_t lmmc_dist_t_pdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    lmmc_real_t coeff, base;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    coeff = exp(lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0)) /
            sqrt(df * LMMC_PI);
    base = 1.0 + x * x / df;
    *out = coeff * pow(base, -(df + 1.0) / 2.0);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_t_cdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    lmmc_real_t t_val, ib;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    t_val = df / (df + x * x);
    ib = regularized_beta(t_val, df / 2.0, 0.5);

    if (x >= 0.0) {
        *out = 1.0 - 0.5 * ib;
    } else {
        *out = 0.5 * ib;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_t_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out) {
    /* Newton's method using t CDF and PDF */
    lmmc_real_t x, cdf_val, pdf_val;
    int iter;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Initial guess from normal quantile */
    x = normal_quantile_rational(p);

    for (iter = 0; iter < 100; ++iter) {
        lmmc_real_t dx;
        lmmc_dist_t_cdf(x, df, &cdf_val);
        lmmc_dist_t_pdf(x, df, &pdf_val);
        if (pdf_val < 1e-300) break;
        dx = (cdf_val - p) / pdf_val;
        x -= dx;
        if (fabs(dx) < 1e-12 * (1.0 + fabs(x))) break;
    }

    *out = x;
    return LMMC_STATUS_OK;
}

/* ===================== χ² 分布 ===================== */

lmmc_status_t lmmc_dist_chi2_pdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    lmmc_real_t k2;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (df < 2.0) { *out = 1.0 / 0.0; return LMMC_STATUS_OK; }
        if (df == 2.0) { *out = 0.5; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    k2 = df / 2.0;
    *out = exp((k2 - 1.0) * log(x) - x / 2.0 - k2 * log(2.0) - lgamma(k2));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_chi2_cdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }

    *out = regularized_gamma_lower(df / 2.0, x / 2.0);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_chi2_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out) {
    /* Newton's method on chi2 CDF */
    lmmc_real_t x, cdf_val, pdf_val, dx;
    int iter;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Initial guess: Wilson-Hilferty approximation */
    {
        lmmc_real_t z = normal_quantile_rational(p);
        lmmc_real_t tmp = 1.0 - 2.0 / (9.0 * df) + z * sqrt(2.0 / (9.0 * df));
        x = df * tmp * tmp * tmp;
        if (x <= 0.0) x = 0.01;
    }

    for (iter = 0; iter < 100; ++iter) {
        lmmc_dist_chi2_cdf(x, df, &cdf_val);
        lmmc_dist_chi2_pdf(x, df, &pdf_val);
        if (pdf_val < 1e-300) break;
        dx = (cdf_val - p) / pdf_val;
        x -= dx;
        if (x <= 0.0) x = 1e-10;
        if (fabs(dx) < 1e-12 * (1.0 + fabs(x))) break;
    }

    *out = x;
    return LMMC_STATUS_OK;
}

/* ===================== F 分布 ===================== */

lmmc_status_t lmmc_dist_f_pdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2,
                               lmmc_real_t* out) {
    lmmc_real_t num, den;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df1 <= 0.0 || df2 <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (df1 < 2.0) { *out = 1.0 / 0.0; return LMMC_STATUS_OK; }
        if (df1 == 2.0) { *out = 1.0; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    num = (df1 / 2.0) * log(df1 / df2) + (df1 / 2.0 - 1.0) * log(x);
    den = ((df1 + df2) / 2.0) * log(1.0 + df1 * x / df2);
    *out = exp(num - den - lgamma(df1 / 2.0) - lgamma(df2 / 2.0) +
               lgamma((df1 + df2) / 2.0));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_f_cdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2,
                               lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df1 <= 0.0 || df2 <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }

    *out = regularized_beta(df1 * x / (df1 * x + df2), df1 / 2.0, df2 / 2.0);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_f_quantile(lmmc_real_t p, lmmc_real_t df1, lmmc_real_t df2,
                                    lmmc_real_t* out) {
    /* Newton's method on F CDF */
    lmmc_real_t x, cdf_val, pdf_val, dx;
    int iter;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df1 <= 0.0 || df2 <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Initial guess */
    x = df2 / (df2 - 2.0 > 0.1 ? df2 - 2.0 : 0.1);
    if (x <= 0.0) x = 1.0;

    for (iter = 0; iter < 100; ++iter) {
        lmmc_dist_f_cdf(x, df1, df2, &cdf_val);
        lmmc_dist_f_pdf(x, df1, df2, &pdf_val);
        if (pdf_val < 1e-300) break;
        dx = (cdf_val - p) / pdf_val;
        x -= dx;
        if (x <= 0.0) x = 1e-10;
        if (fabs(dx) < 1e-12 * (1.0 + fabs(x))) break;
    }

    *out = x;
    return LMMC_STATUS_OK;
}

/* ===================== 伽马分布 ===================== */

lmmc_status_t lmmc_dist_gamma_pdf(lmmc_real_t x, lmmc_real_t shape,
                                   lmmc_real_t scale, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (shape <= 0.0 || scale <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (shape < 1.0) { *out = 1.0 / 0.0; return LMMC_STATUS_OK; }
        if (shape == 1.0) { *out = 1.0 / scale; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    *out = exp((shape - 1.0) * log(x) - x / scale -
               shape * log(scale) - lgamma(shape));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_gamma_cdf(lmmc_real_t x, lmmc_real_t shape,
                                   lmmc_real_t scale, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (shape <= 0.0 || scale <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }

    *out = regularized_gamma_lower(shape, x / scale);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_gamma_quantile(lmmc_real_t p, lmmc_real_t shape,
                                        lmmc_real_t scale, lmmc_real_t* out) {
    /* Newton's method on gamma CDF */
    lmmc_real_t x, cdf_val, pdf_val, dx;
    int iter;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (shape <= 0.0 || scale <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Initial guess: use chi2 approximation */
    x = shape * scale;
    if (x <= 0.0) x = 1.0;

    for (iter = 0; iter < 100; ++iter) {
        lmmc_dist_gamma_cdf(x, shape, scale, &cdf_val);
        lmmc_dist_gamma_pdf(x, shape, scale, &pdf_val);
        if (pdf_val < 1e-300) break;
        dx = (cdf_val - p) / pdf_val;
        x -= dx;
        if (x <= 0.0) x = 1e-10;
        if (fabs(dx) < 1e-12 * (1.0 + fabs(x))) break;
    }

    *out = x;
    return LMMC_STATUS_OK;
}

/* ===================== 贝塔分布 ===================== */

lmmc_status_t lmmc_dist_beta_pdf(lmmc_real_t x, lmmc_real_t alpha,
                                  lmmc_real_t beta_param, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (alpha <= 0.0 || beta_param <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0 || x > 1.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (alpha < 1.0) { *out = 1.0 / 0.0; return LMMC_STATUS_OK; }
        if (alpha == 1.0) {
            *out = exp(lgamma(alpha + beta_param) - lgamma(alpha) - lgamma(beta_param));
            return LMMC_STATUS_OK;
        }
        *out = 0.0; return LMMC_STATUS_OK;
    }
    if (x == 1.0) {
        if (beta_param < 1.0) { *out = 1.0 / 0.0; return LMMC_STATUS_OK; }
        if (beta_param == 1.0) {
            *out = exp(lgamma(alpha + beta_param) - lgamma(alpha) - lgamma(beta_param));
            return LMMC_STATUS_OK;
        }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    *out = exp(lgamma(alpha + beta_param) - lgamma(alpha) - lgamma(beta_param) +
               (alpha - 1.0) * log(x) + (beta_param - 1.0) * log(1.0 - x));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_beta_cdf(lmmc_real_t x, lmmc_real_t alpha,
                                  lmmc_real_t beta_param, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (alpha <= 0.0 || beta_param <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x >= 1.0) { *out = 1.0; return LMMC_STATUS_OK; }

    *out = regularized_beta(x, alpha, beta_param);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_beta_quantile(lmmc_real_t p, lmmc_real_t alpha,
                                       lmmc_real_t beta_param, lmmc_real_t* out) {
    /* Newton's method on beta CDF */
    lmmc_real_t x, cdf_val, pdf_val, dx;
    int iter;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (alpha <= 0.0 || beta_param <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Initial guess: mean of beta distribution */
    x = alpha / (alpha + beta_param);

    for (iter = 0; iter < 100; ++iter) {
        lmmc_dist_beta_cdf(x, alpha, beta_param, &cdf_val);
        lmmc_dist_beta_pdf(x, alpha, beta_param, &pdf_val);
        if (pdf_val < 1e-300) break;
        dx = (cdf_val - p) / pdf_val;
        x -= dx;
        if (x <= 0.0) x = 1e-10;
        if (x >= 1.0) x = 1.0 - 1e-10;
        if (fabs(dx) < 1e-12 * (1.0 + fabs(x))) break;
    }

    *out = x;
    return LMMC_STATUS_OK;
}

/* ===================== 二项分布 ===================== */

lmmc_status_t lmmc_dist_binomial_pmf(size_t k, size_t n, lmmc_real_t p,
                                      lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p < 0.0 || p > 1.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (k > n) { *out = 0.0; return LMMC_STATUS_OK; }

    /* Use log to avoid overflow: log(C(n,k)) + k*log(p) + (n-k)*log(1-p) */
    if (p == 0.0) {
        *out = (k == 0) ? 1.0 : 0.0;
        return LMMC_STATUS_OK;
    }
    if (p == 1.0) {
        *out = (k == n) ? 1.0 : 0.0;
        return LMMC_STATUS_OK;
    }

    {
        lmmc_real_t log_pmf = lgamma((double)(n + 1)) - lgamma((double)(k + 1)) -
                              lgamma((double)(n - k + 1)) +
                              (double)k * log(p) + (double)(n - k) * log(1.0 - p);
        *out = exp(log_pmf);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_binomial_cdf(size_t k, size_t n, lmmc_real_t p_param,
                                      lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p_param < 0.0 || p_param > 1.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (k >= n) { *out = 1.0; return LMMC_STATUS_OK; }

    /* Use regularized incomplete beta: CDF = I_{1-p}(n-k, k+1) */
    *out = regularized_beta(1.0 - p_param, (lmmc_real_t)(n - k), (lmmc_real_t)(k + 1));
    return LMMC_STATUS_OK;
}

/* ===================== 泊松分布 ===================== */

lmmc_status_t lmmc_dist_poisson_pmf(size_t k, lmmc_real_t lambda,
                                     lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lambda < 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lambda == 0.0) {
        *out = (k == 0) ? 1.0 : 0.0;
        return LMMC_STATUS_OK;
    }

    {
        lmmc_real_t log_pmf = (double)k * log(lambda) - lambda -
                              lgamma((double)(k + 1));
        *out = exp(log_pmf);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_poisson_cdf(size_t k, lmmc_real_t lambda,
                                     lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lambda < 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lambda == 0.0) { *out = 1.0; return LMMC_STATUS_OK; }

    /* CDF = Q(k+1, lambda) = 1 - P(k+1, lambda) = upper regularized gamma */
    *out = 1.0 - regularized_gamma_lower((lmmc_real_t)(k + 1), lambda);
    return LMMC_STATUS_OK;
}
