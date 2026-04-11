#include <math.h>
#include "memory_bridge.h"
#include "lmmc/stats.h"

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

static int lmmc_is_finite_number(double v) {
    return isfinite(v) ? 1 : 0;
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

static lmmc_status_t lmmc_finalize_nonnegative(double value, double* out_value) {
    if (!lmmc_is_finite_number(value) || out_value == NULL) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (value < 0.0) {
        if (value > -1e-15) {
            value = 0.0;
        } else {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    *out_value = value;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_mean_m2(const lmmc_vec_t* x, double* out_mean, double* out_m2) {
    size_t i = 0;
    double mean = 0.0;
    double m2 = 0.0;
    lmmc_status_t st = lmmc_validate_vec(x);

    if (out_mean == NULL || out_m2 == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    for (i = 0; i < x->size; ++i) {
        double v = x->data[i];
        double n = (double)(i + 1);
        double delta = 0.0;
        double delta2 = 0.0;

        if (!lmmc_is_finite_number(v)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        delta = v - mean;
        mean += delta / n;
        delta2 = v - mean;
        m2 += delta * delta2;
    }

    if (!lmmc_is_finite_number(mean) || !lmmc_is_finite_number(m2)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    *out_mean = mean;
    *out_m2 = m2;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_cov_accumulate(
    const lmmc_vec_t* x,
    const lmmc_vec_t* y,
    double* out_c,
    double* out_m2x,
    double* out_m2y
) {
    size_t i = 0;
    double mean_x = 0.0;
    double mean_y = 0.0;
    double c = 0.0;
    double m2x = 0.0;
    double m2y = 0.0;
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

    for (i = 0; i < x->size; ++i) {
        double vx = x->data[i];
        double vy = y->data[i];
        double n = (double)(i + 1);
        double dx = 0.0;
        double dy = 0.0;

        if (!lmmc_is_finite_number(vx) || !lmmc_is_finite_number(vy)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        dx = vx - mean_x;
        dy = vy - mean_y;

        mean_x += dx / n;
        mean_y += dy / n;

        c += dx * (vy - mean_y);
        m2x += dx * (vx - mean_x);
        m2y += dy * (vy - mean_y);
    }

    if (!lmmc_is_finite_number(c) || !lmmc_is_finite_number(m2x) || !lmmc_is_finite_number(m2y)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    *out_c = c;
    *out_m2x = m2x;
    *out_m2y = m2y;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_variance_common(const lmmc_vec_t* x, int sample, double* out_variance) {
    double mean = 0.0;
    double m2 = 0.0;
    double denom = 0.0;
    double variance = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (out_variance == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_vec_mean_m2(x, &mean, &m2);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    if (sample) {
        if (x->size < 2) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        denom = (double)(x->size - 1);
    } else {
        denom = (double)x->size;
    }

    variance = m2 / denom;
    return lmmc_finalize_nonnegative(variance, out_variance);
}

static lmmc_status_t lmmc_vec_covariance_common(
    const lmmc_vec_t* x,
    const lmmc_vec_t* y,
    int sample,
    double* out_covariance
) {
    double c = 0.0;
    double m2x = 0.0;
    double m2y = 0.0;
    double denom = 0.0;
    double cov = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (out_covariance == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_vec_cov_accumulate(x, y, &c, &m2x, &m2y);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    if (sample) {
        if (x->size < 2) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        denom = (double)(x->size - 1);
    } else {
        denom = (double)x->size;
    }

    cov = c / denom;
    if (!lmmc_is_finite_number(cov)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    *out_covariance = cov;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_correlation_common(
    const lmmc_vec_t* x,
    const lmmc_vec_t* y,
    int sample,
    double* out_correlation
) {
    double c = 0.0;
    double m2x = 0.0;
    double m2y = 0.0;
    double corr = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (out_correlation == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_vec_cov_accumulate(x, y, &c, &m2x, &m2y);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    if (sample && x->size < 2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (m2x <= 0.0 || m2y <= 0.0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    corr = c / sqrt(m2x * m2y);
    if (!lmmc_is_finite_number(corr)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (corr > 1.0 && corr < 1.0 + 1e-12) {
        corr = 1.0;
    }
    if (corr < -1.0 && corr > -1.0 - 1e-12) {
        corr = -1.0;
    }
    if (corr < -1.0 || corr > 1.0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    *out_correlation = corr;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_mat_column_means_to_buffer(const lmmc_mat_t* x, double* means) {
    size_t col = 0;
    lmmc_status_t st = lmmc_validate_mat(x);

    if (st != LMMC_STATUS_OK || means == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (col = 0; col < x->cols; ++col) {
        size_t row = 0;
        double mean = 0.0;

        for (row = 0; row < x->rows; ++row) {
            double v = x->data[row * x->stride + col];
            double n = (double)(row + 1);
            double delta = 0.0;

            if (!lmmc_is_finite_number(v)) {
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            delta = v - mean;
            mean += delta / n;
        }

        if (!lmmc_is_finite_number(mean)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        means[col] = mean;
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
    double denom = 0.0;
    double* means = NULL;
    double* stddev = NULL;
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

    denom = sample ? (double)(x->rows - 1) : (double)x->rows;

    if (lmmc_mul_overflow_size(x->cols, sizeof(double), &bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    means = (double*)lmmc_alloc(bytes);
    if (means == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    if (correlation) {
        stddev = (double*)lmmc_alloc(bytes);
        if (stddev == NULL) {
            lmmc_free(means);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
    }

    st = lmmc_mat_column_means_to_buffer(x, means);
    if (st != LMMC_STATUS_OK) {
        lmmc_free(stddev);
        lmmc_free(means);
        return st;
    }

    if (correlation) {
        for (col_i = 0; col_i < x->cols; ++col_i) {
            size_t row = 0;
            double sumsq = 0.0;
            double variance = 0.0;

            for (row = 0; row < x->rows; ++row) {
                double d = x->data[row * x->stride + col_i] - means[col_i];
                sumsq += d * d;
            }

            variance = sumsq / denom;
            st = lmmc_finalize_nonnegative(variance, &variance);
            if (st != LMMC_STATUS_OK || variance <= 0.0) {
                lmmc_free(stddev);
                lmmc_free(means);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            stddev[col_i] = sqrt(variance);

            if (!lmmc_is_finite_number(stddev[col_i])) {
                lmmc_free(stddev);
                lmmc_free(means);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }
    }

    for (col_i = 0; col_i < x->cols; ++col_i) {
        for (col_j = col_i; col_j < x->cols; ++col_j) {
            size_t row = 0;
            double sum = 0.0;
            double cov = 0.0;
            double out_value = 0.0;

            for (row = 0; row < x->rows; ++row) {
                double di = x->data[row * x->stride + col_i] - means[col_i];
                double dj = x->data[row * x->stride + col_j] - means[col_j];
                sum += di * dj;
            }

            cov = sum / denom;
            if (!lmmc_is_finite_number(cov)) {
                lmmc_free(stddev);
                lmmc_free(means);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            if (correlation) {
                if (col_i == col_j) {
                    out_value = 1.0;
                } else {
                    out_value = cov / (stddev[col_i] * stddev[col_j]);
                    if (!lmmc_is_finite_number(out_value)) {
                        lmmc_free(stddev);
                        lmmc_free(means);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }
                    if (out_value > 1.0 && out_value < 1.0 + 1e-12) {
                        out_value = 1.0;
                    }
                    if (out_value < -1.0 && out_value > -1.0 - 1e-12) {
                        out_value = -1.0;
                    }
                    if (out_value < -1.0 || out_value > 1.0) {
                        lmmc_free(stddev);
                        lmmc_free(means);
                        return LMMC_STATUS_NUMERICAL_FAILURE;
                    }
                }
            } else {
                out_value = cov;
            }

            out_matrix->data[col_i * out_matrix->stride + col_j] = out_value;
            out_matrix->data[col_j * out_matrix->stride + col_i] = out_value;
        }
    }

    lmmc_free(stddev);
    lmmc_free(means);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_mean(const lmmc_vec_t* x, double* out_mean) {
    double mean = 0.0;
    double m2 = 0.0;
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

lmmc_status_t lmmc_vec_variance_population(const lmmc_vec_t* x, double* out_variance) {
    return lmmc_vec_variance_common(x, 0, out_variance);
}

lmmc_status_t lmmc_vec_variance_sample(const lmmc_vec_t* x, double* out_variance) {
    return lmmc_vec_variance_common(x, 1, out_variance);
}

lmmc_status_t lmmc_vec_stddev_population(const lmmc_vec_t* x, double* out_stddev) {
    double variance = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (out_stddev == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_vec_variance_common(x, 0, &variance);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    *out_stddev = sqrt(variance);
    if (!lmmc_is_finite_number(*out_stddev)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_stddev_sample(const lmmc_vec_t* x, double* out_stddev) {
    double variance = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (out_stddev == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_vec_variance_common(x, 1, &variance);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    *out_stddev = sqrt(variance);
    if (!lmmc_is_finite_number(*out_stddev)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_covariance_population(const lmmc_vec_t* x, const lmmc_vec_t* y, double* out_covariance) {
    return lmmc_vec_covariance_common(x, y, 0, out_covariance);
}

lmmc_status_t lmmc_vec_covariance_sample(const lmmc_vec_t* x, const lmmc_vec_t* y, double* out_covariance) {
    return lmmc_vec_covariance_common(x, y, 1, out_covariance);
}

lmmc_status_t lmmc_vec_correlation_population(const lmmc_vec_t* x, const lmmc_vec_t* y, double* out_correlation) {
    return lmmc_vec_correlation_common(x, y, 0, out_correlation);
}

lmmc_status_t lmmc_vec_correlation_sample(const lmmc_vec_t* x, const lmmc_vec_t* y, double* out_correlation) {
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
