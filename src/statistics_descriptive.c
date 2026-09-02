#include <math.h>
#include <stdlib.h>

#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"

#include "statistics_internal.h"

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

        if (!lmmc_is_finite(&v)) {
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

    if (!lmmc_is_finite(&mean) || !lmmc_is_finite(&m2)) {
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

        if (!lmmc_is_finite(&vx) || !lmmc_is_finite(&vy)) {
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

    if (!lmmc_is_finite(&c) || !lmmc_is_finite(&m2x) || !lmmc_is_finite(&m2y)) {
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
    if (!lmmc_is_finite(&cov)) {
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

    if (!lmmc_is_finite(&corr)) {
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

            if (!lmmc_is_finite(&v)) {
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

        if (!lmmc_is_finite(&mean)) {
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

    if (!lmmc_safe_mul_size(x->cols, sizeof(lmmc_real_t), &bytes)) {
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

            if (!lmmc_is_finite(&stddev[col_i])) {
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
            if (!lmmc_is_finite(&cov)) {
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
                    if (!lmmc_is_finite(&out_value)) {
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
    if (!lmmc_is_finite(&*out_stddev)) {
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
    if (!lmmc_is_finite(&*out_stddev)) {
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

