#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    lmmc_vec_t x = {0};
    lmmc_vec_t y = {0};
    lmmc_vec_t means = {0};
    lmmc_vec_t one = {0};
    lmmc_vec_t const_x = {0};
    lmmc_vec_t lin_y = {0};
    lmmc_vec_t short_y = {0};
    lmmc_vec_t nan_vec = {0};

    lmmc_mat_t data = {0};
    lmmc_mat_t cov_p = {0};
    lmmc_mat_t cov_s = {0};
    lmmc_mat_t corr_p = {0};
    lmmc_mat_t corr_s = {0};
    lmmc_mat_t one_row = {0};
    lmmc_mat_t one_row_cov = {0};
    lmmc_mat_t bad_cov_shape = {0};
    lmmc_mat_t nan_mat = {0};

    lmmc_status_t st = LMMC_STATUS_OK;
    double mean = 0.0;
    double var = 0.0;
    double std = 0.0;
    double cov = 0.0;
    double corr = 0.0;
    int rc = 0;

    st = lmmc_vec_create(4, &x);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(4, &y);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    x.data[0] = 1.0;
    x.data[1] = 2.0;
    x.data[2] = 3.0;
    x.data[3] = 4.0;

    y.data[0] = 2.0;
    y.data[1] = 4.0;
    y.data[2] = 6.0;
    y.data[3] = 8.0;

    st = lmmc_vec_mean(&x, &mean);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(mean, 2.5, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_variance_population(&x, &var);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(var, 1.25, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_variance_sample(&x, &var);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(var, 1.6666666666666667, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_stddev_population(&x, &std);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(std, 1.118033988749895, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_stddev_sample(&x, &std);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(std, 1.2909944487358056, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_covariance_population(&x, &y, &cov);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(cov, 2.5, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_covariance_sample(&x, &y, &cov);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(cov, 3.3333333333333335, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_correlation_population(&x, &y, &corr);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(corr, 1.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_correlation_sample(&x, &y, &corr);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(corr, 1.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(4, 2, &data);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(2, &means);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &cov_p);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &cov_s);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &corr_p);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &corr_s);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    data.data[0] = 1.0; data.data[1] = 2.0;
    data.data[2] = 2.0; data.data[3] = 4.0;
    data.data[4] = 3.0; data.data[5] = 6.0;
    data.data[6] = 4.0; data.data[7] = 8.0;

    st = lmmc_mat_column_mean(&data, &means);
    if (st != LMMC_STATUS_OK ||
        !lmmc_test_nearly_equal(means.data[0], 2.5, 1e-12) ||
        !lmmc_test_nearly_equal(means.data[1], 5.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_covariance_population(&data, &cov_p);
    if (st != LMMC_STATUS_OK ||
        !lmmc_test_nearly_equal(cov_p.data[0], 1.25, 1e-12) ||
        !lmmc_test_nearly_equal(cov_p.data[1], 2.5, 1e-12) ||
        !lmmc_test_nearly_equal(cov_p.data[2], 2.5, 1e-12) ||
        !lmmc_test_nearly_equal(cov_p.data[3], 5.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_covariance_sample(&data, &cov_s);
    if (st != LMMC_STATUS_OK ||
        !lmmc_test_nearly_equal(cov_s.data[0], 1.6666666666666667, 1e-12) ||
        !lmmc_test_nearly_equal(cov_s.data[1], 3.3333333333333335, 1e-12) ||
        !lmmc_test_nearly_equal(cov_s.data[2], 3.3333333333333335, 1e-12) ||
        !lmmc_test_nearly_equal(cov_s.data[3], 6.666666666666667, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_correlation_population(&data, &corr_p);
    if (st != LMMC_STATUS_OK ||
        !lmmc_test_nearly_equal(corr_p.data[0], 1.0, 1e-12) ||
        !lmmc_test_nearly_equal(corr_p.data[1], 1.0, 1e-12) ||
        !lmmc_test_nearly_equal(corr_p.data[2], 1.0, 1e-12) ||
        !lmmc_test_nearly_equal(corr_p.data[3], 1.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_correlation_sample(&data, &corr_s);
    if (st != LMMC_STATUS_OK ||
        !lmmc_test_nearly_equal(corr_s.data[0], 1.0, 1e-12) ||
        !lmmc_test_nearly_equal(corr_s.data[1], 1.0, 1e-12) ||
        !lmmc_test_nearly_equal(corr_s.data[2], 1.0, 1e-12) ||
        !lmmc_test_nearly_equal(corr_s.data[3], 1.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(1, &one);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    one.data[0] = 42.0;

    st = lmmc_vec_variance_sample(&one, &var);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_variance_population(&one, &var);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(var, 0.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(3, &const_x);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(3, &lin_y);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    const_x.data[0] = 1.0;
    const_x.data[1] = 1.0;
    const_x.data[2] = 1.0;
    lin_y.data[0] = 1.0;
    lin_y.data[1] = 2.0;
    lin_y.data[2] = 3.0;

    st = lmmc_vec_correlation_sample(&const_x, &lin_y, &corr);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(3, &short_y);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_covariance_population(&x, &short_y, &cov);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(2, &nan_vec);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    nan_vec.data[0] = 1.0;
    nan_vec.data[1] = NAN;

    st = lmmc_vec_mean(&nan_vec, &mean);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(1, 2, &one_row);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &one_row_cov);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    one_row.data[0] = 1.0;
    one_row.data[1] = 2.0;

    st = lmmc_mat_covariance_sample(&one_row, &one_row_cov);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(3, 3, &bad_cov_shape);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_covariance_population(&data, &bad_cov_shape);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(2, 2, &nan_mat);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    nan_mat.data[0] = 1.0;
    nan_mat.data[1] = 2.0;
    nan_mat.data[2] = 3.0;
    nan_mat.data[3] = NAN;

    st = lmmc_mat_column_mean(&nan_mat, &means);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        rc = 1;
        goto cleanup;
    }

cleanup:
    lmmc_mat_destroy(&nan_mat);
    lmmc_mat_destroy(&bad_cov_shape);
    lmmc_mat_destroy(&one_row_cov);
    lmmc_mat_destroy(&one_row);
    lmmc_vec_destroy(&nan_vec);
    lmmc_vec_destroy(&short_y);
    lmmc_vec_destroy(&lin_y);
    lmmc_vec_destroy(&const_x);
    lmmc_vec_destroy(&one);
    lmmc_mat_destroy(&corr_s);
    lmmc_mat_destroy(&corr_p);
    lmmc_mat_destroy(&cov_s);
    lmmc_mat_destroy(&cov_p);
    lmmc_vec_destroy(&means);
    lmmc_mat_destroy(&data);
    lmmc_vec_destroy(&y);
    lmmc_vec_destroy(&x);

    if (rc != 0) {
        printf("stats test failed\n");
    }
    return rc;
}
