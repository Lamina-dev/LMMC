#include <stdio.h>
#include "lmmc/lmmc.h"

int main(void) {
    lmmc_vec_t x = {0};
    lmmc_vec_t y = {0};
    lmmc_vec_t means = {0};
    lmmc_mat_t data = {0};
    lmmc_mat_t cov = {0};
    lmmc_mat_t corr = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    double mean_x = 0.0;
    double var_x_s = 0.0;
    double std_x_s = 0.0;
    double cov_xy_s = 0.0;
    double corr_xy_s = 0.0;
    int rc = 0;

    st = lmmc_vec_create(5, &x);
    if (st != LMMC_STATUS_OK) {
        printf("vec_create x failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(5, &y);
    if (st != LMMC_STATUS_OK) {
        printf("vec_create y failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    x.data[0] = 1.0; y.data[0] = 2.0;
    x.data[1] = 2.0; y.data[1] = 4.0;
    x.data[2] = 3.0; y.data[2] = 6.0;
    x.data[3] = 4.0; y.data[3] = 8.0;
    x.data[4] = 5.0; y.data[4] = 10.0;

    st = lmmc_vec_mean(&x, &mean_x);
    if (st != LMMC_STATUS_OK) {
        printf("vec_mean failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_variance_sample(&x, &var_x_s);
    if (st != LMMC_STATUS_OK) {
        printf("vec_variance_sample failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_stddev_sample(&x, &std_x_s);
    if (st != LMMC_STATUS_OK) {
        printf("vec_stddev_sample failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_covariance_sample(&x, &y, &cov_xy_s);
    if (st != LMMC_STATUS_OK) {
        printf("vec_covariance_sample failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_correlation_sample(&x, &y, &corr_xy_s);
    if (st != LMMC_STATUS_OK) {
        printf("vec_correlation_sample failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(5, 2, &data);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create data failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(2, &means);
    if (st != LMMC_STATUS_OK) {
        printf("vec_create means failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &cov);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create cov failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &corr);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create corr failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    data.data[0] = x.data[0]; data.data[1] = y.data[0];
    data.data[2] = x.data[1]; data.data[3] = y.data[1];
    data.data[4] = x.data[2]; data.data[5] = y.data[2];
    data.data[6] = x.data[3]; data.data[7] = y.data[3];
    data.data[8] = x.data[4]; data.data[9] = y.data[4];

    st = lmmc_mat_column_mean(&data, &means);
    if (st != LMMC_STATUS_OK) {
        printf("mat_column_mean failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_covariance_sample(&data, &cov);
    if (st != LMMC_STATUS_OK) {
        printf("mat_covariance_sample failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_correlation_sample(&data, &corr);
    if (st != LMMC_STATUS_OK) {
        printf("mat_correlation_sample failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    printf("Vector stats (sample):\n");
    printf("  mean(x)        = %.6f\n", mean_x);
    printf("  var_sample(x)  = %.6f\n", var_x_s);
    printf("  std_sample(x)  = %.6f\n", std_x_s);
    printf("  cov_sample(x,y)= %.6f\n", cov_xy_s);
    printf("  corr_sample(x,y)= %.6f\n", corr_xy_s);

    printf("\nMatrix column means:\n");
    printf("  mean(col0)= %.6f, mean(col1)= %.6f\n", means.data[0], means.data[1]);

    printf("\nSample covariance matrix:\n");
    printf("  [%.6f, %.6f]\n", cov.data[0], cov.data[1]);
    printf("  [%.6f, %.6f]\n", cov.data[2], cov.data[3]);

    printf("\nSample correlation matrix:\n");
    printf("  [%.6f, %.6f]\n", corr.data[0], corr.data[1]);
    printf("  [%.6f, %.6f]\n", corr.data[2], corr.data[3]);

cleanup:
    lmmc_mat_destroy(&corr);
    lmmc_mat_destroy(&cov);
    lmmc_vec_destroy(&means);
    lmmc_mat_destroy(&data);
    lmmc_vec_destroy(&y);
    lmmc_vec_destroy(&x);
    return rc;
}
