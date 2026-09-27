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

    if (data == NULL || count == 0 || out_mean == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    mean = data[0];
    if (!isfinite(mean)) { return LMMC_STATUS_NUMERICAL_FAILURE; }

    for (i = 1; i < count; ++i) {
        const lmmc_real_t value = data[i * stride];
        const lmmc_real_t sample_count = (lmmc_real_t)(i + 1);
        const lmmc_real_t delta = value - mean;
        if (!isfinite(value)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        if (isfinite(delta)) {
            mean = fma(delta, 1.0 / sample_count, mean);
        } else {
            mean = mean * ((sample_count - 1.0) / sample_count) +
                   value / sample_count;
        }
        if (!isfinite(mean)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    }
    *out_mean = mean;
    return LMMC_STATUS_OK;
}

typedef struct {
    lmmc_real_t origin;
    lmmc_real_t mean;
    int exponent;
} lmmc_scaled_column_t;

static lmmc_status_t lmmc_scaled_column(
    const lmmc_real_t* data,
    size_t count,
    size_t stride,
    lmmc_scaled_column_t* column,
    lmmc_real_t* out_m2
) {
    lmmc_real_t scale = 0.0;
    lmmc_real_t m2 = 0.0;
    size_t i;

    for (i = 0; i < count; ++i) {
        const lmmc_real_t value = data[i * stride];
        if (!isfinite(value)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        scale = fmax(scale, fabs(value));
    }
    column->exponent = 0;
    if (scale != 0.0) {
        frexp(scale, &column->exponent);
    }
    column->origin = scalbn(data[0], -column->exponent);
    column->mean = 0.0;

    /**
     * @brief 用精确的二进制幂缩放约束矩，并平移原点以保留离差。
     * 输入中点不可表示时仍可保留离差。
     */
    for (i = 1; i < count; ++i) {
        const lmmc_real_t value =
            scalbn(data[i * stride], -column->exponent) - column->origin;
        const lmmc_real_t delta = value - column->mean;
        const lmmc_real_t sample_count = (lmmc_real_t)(i + 1);
        column->mean = fma(delta, 1.0 / sample_count, column->mean);
        if (out_m2 != NULL) {
            m2 = fma(delta, value - column->mean, m2);
        }
    }
    if (out_m2 != NULL) {
        *out_m2 = m2;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_covariance_scaled(
    const lmmc_real_t* x,
    size_t x_stride,
    const lmmc_scaled_column_t* column_x,
    const lmmc_real_t* y,
    size_t y_stride,
    const lmmc_scaled_column_t* column_y,
    size_t count,
    lmmc_real_t denominator,
    lmmc_real_t* out_covariance
) {
    lmmc_real_t sum = 0.0;
    lmmc_real_t covariance;
    size_t i;

    for (i = 0; i < count; ++i) {
        const lmmc_real_t delta_x =
            (scalbn(x[i * x_stride], -column_x->exponent) - column_x->origin) -
            column_x->mean;
        const lmmc_real_t delta_y =
            (scalbn(y[i * y_stride], -column_y->exponent) - column_y->origin) -
            column_y->mean;
        sum = fma(delta_x, delta_y, sum);
    }
    /** @brief 归一化后分别还原尺度，使未归一化协方差和边缘方差可超出原尺度表示范围。 */
    covariance = scalbn(sum / denominator,
                        column_x->exponent + column_y->exponent);
    if (!isfinite(covariance)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *out_covariance = covariance;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_variance_common(const lmmc_vec_t* x, int sample, lmmc_real_t* out_variance) {
    lmmc_scaled_column_t column;
    lmmc_real_t m2, variance;
    lmmc_status_t status;

    if (out_variance == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_validate_vec(x);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_scaled_column(x->data, x->size, 1, &column, &m2);
    if (status != LMMC_STATUS_OK) { return status; }
    if (sample && x->size < 2) { return LMMC_STATUS_INVALID_ARGUMENT; }

    variance = scalbn(m2 / (lmmc_real_t)(x->size - (sample != 0)),
                      2 * column.exponent);
    return lmmc_finalize_nonnegative(variance, out_variance);
}

static lmmc_status_t lmmc_vec_stddev_common(
    const lmmc_vec_t* x,
    int sample,
    lmmc_real_t* out_stddev
) {
    lmmc_scaled_column_t column;
    lmmc_real_t m2, stddev;
    lmmc_status_t status;

    if (out_stddev == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_validate_vec(x);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_scaled_column(x->data, x->size, 1, &column, &m2);
    if (status != LMMC_STATUS_OK) { return status; }
    if (sample && x->size < 2) { return LMMC_STATUS_INVALID_ARGUMENT; }

    /** @brief 开方后还原尺度，允许方差溢出而标准差有限。 */
    stddev = scalbn(sqrt(m2 / (lmmc_real_t)(x->size - (sample != 0))),
                    column.exponent);
    if (!isfinite(stddev)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *out_stddev = stddev;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_covariance_common(
    const lmmc_vec_t* x,
    const lmmc_vec_t* y,
    int sample,
    lmmc_real_t* out_covariance
) {
    lmmc_scaled_column_t column_x, column_y;
    lmmc_status_t status;

    if (out_covariance == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (lmmc_validate_vec(x) != LMMC_STATUS_OK ||
        lmmc_validate_vec(y) != LMMC_STATUS_OK) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x->size != y->size) { return LMMC_STATUS_DIMENSION_MISMATCH; }
    status = lmmc_scaled_column(x->data, x->size, 1, &column_x, NULL);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_scaled_column(y->data, y->size, 1, &column_y, NULL);
    if (status != LMMC_STATUS_OK) { return status; }
    if (sample && x->size < 2) { return LMMC_STATUS_INVALID_ARGUMENT; }

    return lmmc_covariance_scaled(
        x->data, 1, &column_x, y->data, 1, &column_y, x->size,
        (lmmc_real_t)(x->size - (sample != 0)), out_covariance);
}

static lmmc_status_t lmmc_finalize_correlation(
    lmmc_real_t c, lmmc_real_t m2x, lmmc_real_t m2y,
    size_t count, lmmc_real_t* out_correlation) {
    lmmc_real_t correlation;
    const lmmc_real_t rounding_bound =
        64.0 * DBL_EPSILON + 8.0 * (lmmc_real_t)count * DBL_EPSILON * DBL_EPSILON;
    if (!(m2x > 0.0) || !(m2y > 0.0) ||
        !isfinite(c) || !isfinite(m2x) || !isfinite(m2y)) { return LMMC_STATUS_NUMERICAL_FAILURE; }

    correlation = (c / sqrt(m2x)) / sqrt(m2y);
    if (!isfinite(correlation)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    if (correlation > 1.0 && correlation <= 1.0 + rounding_bound) {
        correlation = 1.0;
    }
    if (correlation < -1.0 && correlation >= -1.0 - rounding_bound) {
        correlation = -1.0;
    }
    if (correlation < -1.0 || correlation > 1.0) { return LMMC_STATUS_NUMERICAL_FAILURE; }

    *out_correlation = correlation;
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
    lmmc_scaled_column_t column_x, column_y;
    lmmc_real_t spread_x = 0.0, spread_y = 0.0;
    lmmc_real_t sx = 0.0, sy = 0.0, qx = 0.0, qy = 0.0, c = 0.0;
    lmmc_real_t sx_error = 0.0, sy_error = 0.0;
    lmmc_real_t qx_error = 0.0, qy_error = 0.0, c_error = 0.0;
    int exponent_x, exponent_y;
    size_t i;
    lmmc_status_t status = lmmc_scaled_column(x, count, x_stride, &column_x, NULL);
    if (status != LMMC_STATUS_OK) { return status; }
    status = lmmc_scaled_column(y, count, y_stride, &column_y, NULL);
    if (status != LMMC_STATUS_OK) { return status; }

    for (i = 0; i < count; ++i) {
        const lmmc_real_t dx =
            (scalbn(x[i * x_stride], -column_x.exponent) - column_x.origin) - column_x.mean;
        const lmmc_real_t dy =
            (scalbn(y[i * y_stride], -column_y.exponent) - column_y.origin) - column_y.mean;
        spread_x = fmax(spread_x, fabs(dx));
        spread_y = fmax(spread_y, fabs(dy));
    }
    if (spread_x == 0.0 || spread_y == 0.0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    frexp(spread_x, &exponent_x);
    frexp(spread_y, &exponent_y);

    /**
     * @brief 平方前缩放中心化离差，使次正规差值的矩保持正常量级。
     * Pearson 相关系数与各列单位无关。
     */
    for (i = 0; i < count; ++i) {
        const lmmc_real_t dx = scalbn(
            (scalbn(x[i * x_stride], -column_x.exponent) - column_x.origin) - column_x.mean,
            -exponent_x);
        const lmmc_real_t dy = scalbn(
            (scalbn(y[i * y_stride], -column_y.exponent) - column_y.origin) - column_y.mean,
            -exponent_y);
        const lmmc_real_t xx = dx * dx, yy = dy * dy, xy = dx * dy;
        lmmc_compensated_add(dx, &sx, &sx_error);
        lmmc_compensated_add(dy, &sy, &sy_error);
        lmmc_compensated_add(xx, &qx, &qx_error);
        lmmc_compensated_add(fma(dx, dx, -xx), &qx, &qx_error);
        lmmc_compensated_add(yy, &qy, &qy_error);
        lmmc_compensated_add(fma(dy, dy, -yy), &qy, &qy_error);
        lmmc_compensated_add(xy, &c, &c_error);
        lmmc_compensated_add(fma(dx, dy, -xy), &c, &c_error);
    }
    sx += sx_error;
    sy += sy_error;
    /** @brief 统一修正残余均值，使协方差与方差使用相同的舍入修正公式。 */
    qx = fma(-sx, sx / (lmmc_real_t)count, qx + qx_error);
    qy = fma(-sy, sy / (lmmc_real_t)count, qy + qy_error);
    c = fma(-sx, sy / (lmmc_real_t)count, c + c_error);
    return lmmc_finalize_correlation(c, qx, qy, count, out_correlation);
}

static lmmc_status_t lmmc_vec_correlation_common(
    const lmmc_vec_t* x,
    const lmmc_vec_t* y,
    int sample,
    lmmc_real_t* out_correlation
) {
    lmmc_status_t status_x;
    lmmc_status_t status_y;

    if (out_correlation == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status_x = lmmc_validate_vec(x);
    status_y = lmmc_validate_vec(y);
    if (status_x != LMMC_STATUS_OK || status_y != LMMC_STATUS_OK) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x->size != y->size) { return LMMC_STATUS_DIMENSION_MISMATCH; }
    if (sample && x->size < 2) { return LMMC_STATUS_INVALID_ARGUMENT; }

    return lmmc_correlation_scaled(
        x->data, 1, y->data, 1, x->size, out_correlation);
}

static lmmc_status_t lmmc_mat_column_means_to_buffer(
    const lmmc_mat_t* x, lmmc_real_t* means) {
    lmmc_status_t status = lmmc_validate_mat(x);
    size_t column;

    if (status != LMMC_STATUS_OK || means == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    for (column = 0; column < x->cols; ++column) {
        status = lmmc_mean_strided(
            &x->data[column], x->rows, x->stride, &means[column]);
        if (status != LMMC_STATUS_OK) { return status; }
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_scale_matrix_columns(
    const lmmc_mat_t* x, lmmc_scaled_column_t* columns) {
    size_t col_i;
    lmmc_status_t status;
    for (col_i = 0; col_i < x->cols; ++col_i) {
        status = lmmc_scaled_column(
            &x->data[col_i], x->rows, x->stride, &columns[col_i], NULL);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_mat_covariance_common(
    const lmmc_mat_t* x,
    int sample,
    lmmc_mat_t* out_matrix
) {
    size_t bytes = 0;
    lmmc_scaled_column_t* columns = NULL;
    lmmc_status_t status = lmmc_validate_mat(x);
    lmmc_real_t denominator;
    size_t col_i;
    size_t col_j;

    if (status != LMMC_STATUS_OK ||
        out_matrix == NULL || out_matrix->data == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (out_matrix->rows != x->cols || out_matrix->cols != x->cols) { return LMMC_STATUS_DIMENSION_MISMATCH; }
    if (sample && x->rows < 2) { return LMMC_STATUS_INVALID_ARGUMENT; }
    denominator = sample ? (lmmc_real_t)(x->rows - 1)
                         : (lmmc_real_t)x->rows;
    if (!lmmc_safe_mul_size(x->cols, sizeof(*columns), &bytes)) { return LMMC_STATUS_INVALID_ARGUMENT; }

    columns = (lmmc_scaled_column_t*)lmmc_memory_alloc(bytes);
    if (columns == NULL) { return LMMC_STATUS_ALLOCATION_FAILED; }
    status = lmmc_scale_matrix_columns(x, columns);
    if (status != LMMC_STATUS_OK) {
        goto cleanup;
    }

    for (col_i = 0; col_i < x->cols; ++col_i) {
        for (col_j = col_i; col_j < x->cols; ++col_j) {
            lmmc_real_t covariance;
            status = lmmc_covariance_scaled(
                &x->data[col_i], x->stride, &columns[col_i],
                &x->data[col_j], x->stride, &columns[col_j],
                x->rows, denominator, &covariance);
            if (status != LMMC_STATUS_OK) {
                goto cleanup;
            }
            out_matrix->data[col_i * out_matrix->stride + col_j] = covariance;
            out_matrix->data[col_j * out_matrix->stride + col_i] = covariance;
        }
    }

cleanup:
    lmmc_memory_free(columns);
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
        out_matrix == NULL || out_matrix->data == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (out_matrix->rows != x->cols || out_matrix->cols != x->cols) { return LMMC_STATUS_DIMENSION_MISMATCH; }
    if (sample && x->rows < 2) { return LMMC_STATUS_INVALID_ARGUMENT; }

    for (col_i = 0; col_i < x->cols; ++col_i) {
        for (col_j = col_i; col_j < x->cols; ++col_j) {
            lmmc_real_t correlation;
            status = lmmc_correlation_scaled(
                &x->data[col_i], x->stride,
                &x->data[col_j], x->stride,
                x->rows, &correlation);
            if (status != LMMC_STATUS_OK) { return status; }
            if (col_i == col_j) {
                correlation = 1.0;
            }
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
    if (out_mean == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_validate_vec(x);
    if (status != LMMC_STATUS_OK) { return status; }
    return lmmc_mean_strided(x->data, x->size, 1, out_mean);
}

lmmc_status_t lmmc_vec_variance_population(const lmmc_vec_t* x, lmmc_real_t* out_variance) {
    return lmmc_vec_variance_common(x, 0, out_variance);
}

lmmc_status_t lmmc_vec_variance_sample(const lmmc_vec_t* x, lmmc_real_t* out_variance) {
    return lmmc_vec_variance_common(x, 1, out_variance);
}

lmmc_status_t lmmc_vec_stddev_population(const lmmc_vec_t* x, lmmc_real_t* out_stddev) {
    return lmmc_vec_stddev_common(x, 0, out_stddev);
}

lmmc_status_t lmmc_vec_stddev_sample(const lmmc_vec_t* x, lmmc_real_t* out_stddev) {
    return lmmc_vec_stddev_common(x, 1, out_stddev);
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

