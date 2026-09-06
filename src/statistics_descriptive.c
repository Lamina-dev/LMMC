#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"

#include "statistics_internal.h"

static lmmc_status_t lmmc_mean_strided(
    const lmmc_real_t* data,
    size_t count,
    size_t stride,
    lmmc_real_t* out_mean
) {
    lmmc_real_t mean;
    size_t i;

    if (data == NULL || count == 0 || out_mean == NULL)
        return LMMC_STATUS_INVALID_ARGUMENT;
    mean = data[0];
    if (!isfinite(mean)) return LMMC_STATUS_NUMERICAL_FAILURE;

    for (i = 1; i < count; ++i) {
        const lmmc_real_t value = data[i * stride];
        const lmmc_real_t sample_count = (lmmc_real_t)(i + 1);
        const lmmc_real_t delta = value - mean;
        if (!isfinite(value)) return LMMC_STATUS_NUMERICAL_FAILURE;
        if (isfinite(delta)) {
            mean = fma(delta, 1.0 / sample_count, mean);
        } else {
            mean = mean * ((sample_count - 1.0) / sample_count) +
                   value / sample_count;
        }
        if (!isfinite(mean)) return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out_mean = mean;
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

static lmmc_status_t lmmc_correlation_scaled(
    const lmmc_real_t* x,
    size_t x_stride,
    const lmmc_real_t* y,
    size_t y_stride,
    size_t count,
    lmmc_real_t* out_correlation
) {
    lmmc_real_t scale_x = 0.0;
    lmmc_real_t scale_y = 0.0;
    lmmc_real_t mean_x = 0.0;
    lmmc_real_t mean_y = 0.0;
    lmmc_real_t c = 0.0;
    lmmc_real_t m2x = 0.0;
    lmmc_real_t m2y = 0.0;
    lmmc_real_t correlation;
    size_t i;

    for (i = 0; i < count; ++i) {
        const lmmc_real_t value_x = x[i * x_stride];
        const lmmc_real_t value_y = y[i * y_stride];
        if (!isfinite(value_x) || !isfinite(value_y))
            return LMMC_STATUS_NUMERICAL_FAILURE;
        scale_x = fmax(scale_x, fabs(value_x));
        scale_y = fmax(scale_y, fabs(value_y));
    }
    if (scale_x == 0.0 || scale_y == 0.0)
        return LMMC_STATUS_NUMERICAL_FAILURE;

    for (i = 0; i < count; ++i) {
        const lmmc_real_t sample_count = (lmmc_real_t)(i + 1);
        const lmmc_real_t value_x = x[i * x_stride] / scale_x;
        const lmmc_real_t value_y = y[i * y_stride] / scale_y;
        const lmmc_real_t delta_x = value_x - mean_x;
        const lmmc_real_t delta_y = value_y - mean_y;

        mean_x += delta_x / sample_count;
        mean_y += delta_y / sample_count;
        c += delta_x * (value_y - mean_y);
        m2x += delta_x * (value_x - mean_x);
        m2y += delta_y * (value_y - mean_y);
    }
    if (!(m2x > 0.0) || !(m2y > 0.0) ||
        !isfinite(c) || !isfinite(m2x) || !isfinite(m2y))
        return LMMC_STATUS_NUMERICAL_FAILURE;

    correlation = (c / sqrt(m2x)) / sqrt(m2y);
    if (!isfinite(correlation)) return LMMC_STATUS_NUMERICAL_FAILURE;
    if (correlation > 1.0 && correlation < 1.0 + 1e-12)
        correlation = 1.0;
    if (correlation < -1.0 && correlation > -1.0 - 1e-12)
        correlation = -1.0;
    if (correlation < -1.0 || correlation > 1.0)
        return LMMC_STATUS_NUMERICAL_FAILURE;

    *out_correlation = correlation;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_correlation_common(
    const lmmc_vec_t* x,
    const lmmc_vec_t* y,
    int sample,
    lmmc_real_t* out_correlation
) {
    lmmc_status_t status_x;
    lmmc_status_t status_y;

    if (out_correlation == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    status_x = lmmc_validate_vec(x);
    status_y = lmmc_validate_vec(y);
    if (status_x != LMMC_STATUS_OK || status_y != LMMC_STATUS_OK)
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (x->size != y->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    if (sample && x->size < 2) return LMMC_STATUS_INVALID_ARGUMENT;

    return lmmc_correlation_scaled(
        x->data, 1, y->data, 1, x->size, out_correlation);
}

static lmmc_status_t lmmc_mat_column_means_to_buffer(
    const lmmc_mat_t* x, lmmc_real_t* means) {
    lmmc_status_t status = lmmc_validate_mat(x);
    size_t column;

    if (status != LMMC_STATUS_OK || means == NULL)
        return LMMC_STATUS_INVALID_ARGUMENT;
    for (column = 0; column < x->cols; ++column) {
        status = lmmc_mean_strided(
            &x->data[column], x->rows, x->stride, &means[column]);
        if (status != LMMC_STATUS_OK) return status;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_mat_covariance_common(
    const lmmc_mat_t* x,
    int sample,
    lmmc_mat_t* out_matrix
) {
    size_t bytes = 0;
    lmmc_real_t* means = NULL;
    lmmc_status_t status = lmmc_validate_mat(x);
    lmmc_real_t denominator;
    size_t col_i;
    size_t col_j;

    if (status != LMMC_STATUS_OK ||
        out_matrix == NULL || out_matrix->data == NULL)
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (out_matrix->rows != x->cols || out_matrix->cols != x->cols)
        return LMMC_STATUS_DIMENSION_MISMATCH;
    if (sample && x->rows < 2) return LMMC_STATUS_INVALID_ARGUMENT;
    denominator = sample ? (lmmc_real_t)(x->rows - 1)
                         : (lmmc_real_t)x->rows;
    if (!lmmc_safe_mul_size(x->cols, sizeof(lmmc_real_t), &bytes))
        return LMMC_STATUS_INVALID_ARGUMENT;

    means = (lmmc_real_t*)lmmc_alloc(bytes);
    if (means == NULL) return LMMC_STATUS_ALLOCATION_FAILED;
    status = lmmc_mat_column_means_to_buffer(x, means);
    if (status != LMMC_STATUS_OK) goto cleanup;

    for (col_i = 0; col_i < x->cols; ++col_i) {
        for (col_j = col_i; col_j < x->cols; ++col_j) {
            lmmc_real_t sum = 0.0;
            lmmc_real_t covariance;
            size_t row;

            for (row = 0; row < x->rows; ++row) {
                const lmmc_real_t delta_i =
                    x->data[row * x->stride + col_i] - means[col_i];
                const lmmc_real_t delta_j =
                    x->data[row * x->stride + col_j] - means[col_j];
                sum += delta_i * delta_j;
                if (!isfinite(sum)) {
                    status = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }
            }
            covariance = sum / denominator;
            if (!isfinite(covariance)) {
                status = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }
            out_matrix->data[col_i * out_matrix->stride + col_j] = covariance;
            out_matrix->data[col_j * out_matrix->stride + col_i] = covariance;
        }
    }

cleanup:
    lmmc_free(means);
    return status;
}

static lmmc_status_t lmmc_mat_correlation_common(
    const lmmc_mat_t* x,
    int sample,
    lmmc_mat_t* out_matrix
) {
    lmmc_status_t status = lmmc_validate_mat(x);
    size_t col_i;
    size_t col_j;

    if (status != LMMC_STATUS_OK ||
        out_matrix == NULL || out_matrix->data == NULL)
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (out_matrix->rows != x->cols || out_matrix->cols != x->cols)
        return LMMC_STATUS_DIMENSION_MISMATCH;
    if (sample && x->rows < 2) return LMMC_STATUS_INVALID_ARGUMENT;

    for (col_i = 0; col_i < x->cols; ++col_i) {
        for (col_j = col_i; col_j < x->cols; ++col_j) {
            lmmc_real_t correlation;
            status = lmmc_correlation_scaled(
                &x->data[col_i], x->stride,
                &x->data[col_j], x->stride,
                x->rows, &correlation);
            if (status != LMMC_STATUS_OK) return status;
            if (col_i == col_j) correlation = 1.0;
            out_matrix->data[col_i * out_matrix->stride + col_j] =
                correlation;
            out_matrix->data[col_j * out_matrix->stride + col_i] =
                correlation;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_mean(const lmmc_vec_t* x, lmmc_real_t* out_mean) {
    lmmc_status_t status;
    if (out_mean == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_validate_vec(x);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_mean_strided(x->data, x->size, 1, out_mean);
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
    return lmmc_mat_covariance_common(x, 0, out_covariance);
}

lmmc_status_t lmmc_mat_covariance_sample(const lmmc_mat_t* x, lmmc_mat_t* out_covariance) {
    return lmmc_mat_covariance_common(x, 1, out_covariance);
}

lmmc_status_t lmmc_mat_correlation_population(const lmmc_mat_t* x, lmmc_mat_t* out_correlation) {
    return lmmc_mat_correlation_common(x, 0, out_correlation);
}

lmmc_status_t lmmc_mat_correlation_sample(const lmmc_mat_t* x, lmmc_mat_t* out_correlation) {
    return lmmc_mat_correlation_common(x, 1, out_correlation);
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

static lmmc_real_t lmmc_real_midpoint(
    lmmc_real_t a, lmmc_real_t b) {
    if ((a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0)) {
        return a * 0.5 + b * 0.5;
    }
    return a + (b - a) * 0.5;
}

static int lmmc_vec_values_are_finite(const lmmc_vec_t* x) {
    for (size_t i = 0; i < x->size; ++i) {
        if (!lmmc_is_finite(&x->data[i])) return 0;
    }
    return 1;
}

lmmc_status_t lmmc_vec_median(const lmmc_vec_t* x, lmmc_real_t* out) {
    lmmc_real_t* sorted = NULL;
    size_t n;

    if (out == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x == NULL || x->data == NULL || x->size == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_vec_values_are_finite(x)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    n = x->size;
    sorted = (lmmc_real_t*)lmmc_alloc_array(n, sizeof(lmmc_real_t));
    if (sorted == NULL) return LMMC_STATUS_ALLOCATION_FAILED;

    for (size_t i = 0; i < n; ++i) {
        sorted[i] = x->data[i];
    }
    qsort(sorted, n, sizeof(lmmc_real_t), lmmc_real_compare);

    if (n % 2 == 1) {
        *out = sorted[n / 2];
    } else {
        *out = lmmc_real_midpoint(
            sorted[n / 2 - 1], sorted[n / 2]);
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
    if (!lmmc_is_finite(&p) || p < 0.0 || p > 1.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_vec_values_are_finite(x)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    n = x->size;
    sorted = (lmmc_real_t*)lmmc_alloc_array(n, sizeof(lmmc_real_t));
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
    lmmc_real_t xmin, xmax, range_scale = 1.0;
    lmmc_real_t scaled_min = 0.0, scaled_span = 0.0;
    int crosses_zero;

    if (x == NULL || x->data == NULL || x->size == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (nbins == 0 || edges == NULL || counts == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_vec_values_are_finite(x)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    n = x->size;

    xmin = x->data[0];
    xmax = x->data[0];
    for (i = 1; i < n; ++i) {
        if (x->data[i] < xmin) xmin = x->data[i];
        if (x->data[i] > xmax) xmax = x->data[i];
    }

    if (xmax == xmin) {
        const lmmc_real_t value = xmin;
        const lmmc_real_t lower = value - 0.5;
        const lmmc_real_t upper = value + 0.5;
        if (lower < value && upper > value &&
            lmmc_is_finite(&lower) && lmmc_is_finite(&upper)) {
            xmin = lower;
            xmax = upper;
        } else if (value > 0.0) {
            xmin = value * 0.5;
            xmax = value <= DBL_MAX / 1.5 ? value * 1.5 : value;
        } else if (value < 0.0) {
            xmin = value >= -DBL_MAX / 1.5 ? value * 1.5 : value;
            xmax = value * 0.5;
        } else {
            xmin = -0.5;
            xmax = 0.5;
        }
    }
    crosses_zero = xmin < 0.0 && xmax > 0.0;
    if (crosses_zero) {
        range_scale = fmax(-xmin, xmax);
        scaled_min = xmin / range_scale;
        scaled_span = xmax / range_scale - scaled_min;
    }


    edges[0] = xmin;
    for (i = 1; i < nbins; ++i) {
        const lmmc_real_t t =
            (lmmc_real_t)i / (lmmc_real_t)nbins;
        if ((xmin < 0.0 && xmax > 0.0) ||
            (xmin > 0.0 && xmax < 0.0)) {
            edges[i] = (1.0 - t) * xmin + t * xmax;
        } else {
            edges[i] = xmin + t * (xmax - xmin);
        }
    }
    edges[nbins] = xmax;

    for (i = 0; i < nbins; ++i) {
        counts[i] = 0;
    }

    for (i = 0; i < n; ++i) {
        size_t bin;
        if (x->data[i] <= xmin) {
            bin = 0;
        } else if (x->data[i] >= xmax) {
            bin = nbins - 1;
        } else {
            lmmc_real_t position;
            lmmc_real_t bin_position;
            if (crosses_zero) {
                position =
                    (x->data[i] / range_scale - scaled_min) /
                    scaled_span;
            } else {
                position =
                    (x->data[i] - xmin) / (xmax - xmin);
            }
            bin_position = position * (lmmc_real_t)nbins;
            if (position >= 1.0 ||
                bin_position >= (lmmc_real_t)(nbins - 1)) {
                bin = nbins - 1;
            } else if (position <= 0.0) {
                bin = 0;
            } else {
                bin = (size_t)bin_position;
            }
        }
        counts[bin]++;
    }

    return LMMC_STATUS_OK;
}

