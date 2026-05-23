#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    int rc = 0;
    lmmc_vec_t v1 = {0};
    lmmc_vec_t v2 = {0};
    lmmc_vec_t big_v1 = {0};
    lmmc_vec_t big_v2 = {0};
    lmmc_mat_t data = {0};
    lmmc_mat_t cov_mat = {0};
    lmmc_mat_t corr_mat = {0};

    /* ===== Requirement 15.1: factorial known values ===== */
    {
        lmmc_real_t val = 0.0;

        lmmc_stats_factorial(&val, 0);
        if (!lmmc_test_nearly_equal(val, 1.0, 1e-12)) { rc = 1; goto done; }

        lmmc_stats_factorial(&val, 1);
        if (!lmmc_test_nearly_equal(val, 1.0, 1e-12)) { rc = 1; goto done; }

        lmmc_stats_factorial(&val, 10);
        if (!lmmc_test_nearly_equal(val, 3628800.0, 1e-6)) { rc = 1; goto done; }

        lmmc_stats_factorial(&val, 20);
        if (!lmmc_test_nearly_equal(val, 2432902008176640000.0, 1e6)) { rc = 1; goto done; }
    }

    /* ===== Requirement 15.2: nCr known values ===== */
    {
        lmmc_real_t val = 0.0;

        lmmc_stats_nCr(&val, 10, 3);
        if (!lmmc_test_nearly_equal(val, 120.0, 1e-10)) { rc = 1; goto done; }

        lmmc_stats_nCr(&val, 20, 10);
        if (!lmmc_test_nearly_equal(val, 184756.0, 1e-6)) { rc = 1; goto done; }

        /* C(n, 0) = 1 for various n */
        lmmc_stats_nCr(&val, 5, 0);
        if (!lmmc_test_nearly_equal(val, 1.0, 1e-12)) { rc = 1; goto done; }

        lmmc_stats_nCr(&val, 100, 0);
        if (!lmmc_test_nearly_equal(val, 1.0, 1e-12)) { rc = 1; goto done; }

        /* C(n, n) = 1 for various n */
        lmmc_stats_nCr(&val, 5, 5);
        if (!lmmc_test_nearly_equal(val, 1.0, 1e-12)) { rc = 1; goto done; }

        lmmc_stats_nCr(&val, 50, 50);
        if (!lmmc_test_nearly_equal(val, 1.0, 1e-12)) { rc = 1; goto done; }
    }

    /* ===== Requirement 15.3: nPr known values ===== */
    {
        lmmc_real_t val = 0.0;

        lmmc_stats_nPr(&val, 5, 3);
        if (!lmmc_test_nearly_equal(val, 60.0, 1e-10)) { rc = 1; goto done; }

        lmmc_stats_nPr(&val, 10, 5);
        if (!lmmc_test_nearly_equal(val, 30240.0, 1e-6)) { rc = 1; goto done; }
    }

    /* ===== Requirement 15.4: constant vector variance = 0 ===== */
    {
        lmmc_status_t st;
        lmmc_real_t var = 0.0;

        st = lmmc_vec_create(50, &v1);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        /* Fill with constant value 7.5 */
        for (size_t i = 0; i < 50; i++) {
            LMMC_REAL_SET_D(&v1.data[i], 7.5);
        }

        st = lmmc_vec_variance_population(&v1, &var);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(var, 0.0, 1e-12)) {
            rc = 1; goto done;
        }

        lmmc_vec_destroy(&v1);
        memset(&v1, 0, sizeof(v1));
    }

    /* ===== Requirement 15.5: negative correlation = -1 ===== */
    {
        lmmc_status_t st;
        lmmc_real_t corr = 0.0;

        st = lmmc_vec_create(5, &v1);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_vec_create(5, &v2);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        /* x = [1, 2, 3, 4, 5], y = [-1, -2, -3, -4, -5] => perfect negative correlation */
        for (size_t i = 0; i < 5; i++) {
            LMMC_REAL_SET_D(&v1.data[i], (double)(i + 1));
            LMMC_REAL_SET_D(&v2.data[i], -(double)(i + 1));
        }

        st = lmmc_vec_correlation_sample(&v1, &v2, &corr);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(corr, -1.0, 1e-12)) {
            rc = 1; goto done;
        }

        st = lmmc_vec_correlation_population(&v1, &v2, &corr);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(corr, -1.0, 1e-12)) {
            rc = 1; goto done;
        }

        lmmc_vec_destroy(&v2);
        memset(&v2, 0, sizeof(v2));
        lmmc_vec_destroy(&v1);
        memset(&v1, 0, sizeof(v1));
    }

    /* ===== Requirement 15.7: large vector (size=1000) numerical stability ===== */
    {
        lmmc_status_t st;
        lmmc_real_t mean_val = 0.0;
        lmmc_real_t var_val = 0.0;
        const size_t n = 1000;

        st = lmmc_vec_create(n, &big_v1);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        /* Fill with values 1, 2, ..., 1000 */
        for (size_t i = 0; i < n; i++) {
            LMMC_REAL_SET_D(&big_v1.data[i], (double)(i + 1));
        }

        /* Mean of 1..1000 = 500.5 */
        st = lmmc_vec_mean(&big_v1, &mean_val);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(mean_val, 500.5, 1e-10)) {
            rc = 1; goto done;
        }

        /* Population variance of 1..n = (n^2 - 1) / 12 = (1000000 - 1) / 12 = 83333.25 */
        st = lmmc_vec_variance_population(&big_v1, &var_val);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(var_val, 83333.25, 1e-6)) {
            rc = 1; goto done;
        }

        lmmc_vec_destroy(&big_v1);
        memset(&big_v1, 0, sizeof(big_v1));
    }

    /* ===== Requirement 15.8: covariance matrix symmetry ===== */
    {
        lmmc_status_t st;
        const size_t rows = 10;
        const size_t cols = 3;

        st = lmmc_mat_create(rows, cols, &data);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_mat_create(cols, cols, &cov_mat);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        /* Fill data matrix with some values */
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double val = (double)(i * cols + j) * 0.7 + (double)(j * j) * 1.3;
                LMMC_REAL_SET_D(&data.data[i * cols + j], val);
            }
        }

        st = lmmc_mat_covariance_population(&data, &cov_mat);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        /* Verify symmetry: Cov[i][j] == Cov[j][i] */
        for (size_t i = 0; i < cols; i++) {
            for (size_t j = 0; j < cols; j++) {
                double cij = cov_mat.data[i * cols + j];
                double cji = cov_mat.data[j * cols + i];
                if (!lmmc_test_nearly_equal(cij, cji, 1e-12)) {
                    rc = 1; goto done;
                }
            }
        }

        /* Also verify with sample covariance */
        st = lmmc_mat_covariance_sample(&data, &cov_mat);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        for (size_t i = 0; i < cols; i++) {
            for (size_t j = 0; j < cols; j++) {
                double cij = cov_mat.data[i * cols + j];
                double cji = cov_mat.data[j * cols + i];
                if (!lmmc_test_nearly_equal(cij, cji, 1e-12)) {
                    rc = 1; goto done;
                }
            }
        }

        lmmc_mat_destroy(&cov_mat);
        memset(&cov_mat, 0, sizeof(cov_mat));
        lmmc_mat_destroy(&data);
        memset(&data, 0, sizeof(data));
    }

    /* ===== Requirement 15.9: correlation matrix diagonal = 1 ===== */
    {
        lmmc_status_t st;
        const size_t rows = 10;
        const size_t cols = 4;

        st = lmmc_mat_create(rows, cols, &data);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_mat_create(cols, cols, &corr_mat);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        /* Fill with non-constant columns (important: each column must have variance > 0) */
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double val = (double)(i + 1) * (double)(j + 1) + (double)(i * i) * 0.1;
                LMMC_REAL_SET_D(&data.data[i * cols + j], val);
            }
        }

        st = lmmc_mat_correlation_population(&data, &corr_mat);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        /* Verify diagonal elements are 1.0 */
        for (size_t i = 0; i < cols; i++) {
            double diag_val = corr_mat.data[i * cols + i];
            if (!lmmc_test_nearly_equal(diag_val, 1.0, 1e-12)) {
                rc = 1; goto done;
            }
        }

        /* Also verify with sample correlation */
        st = lmmc_mat_correlation_sample(&data, &corr_mat);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        for (size_t i = 0; i < cols; i++) {
            double diag_val = corr_mat.data[i * cols + i];
            if (!lmmc_test_nearly_equal(diag_val, 1.0, 1e-12)) {
                rc = 1; goto done;
            }
        }

        lmmc_mat_destroy(&corr_mat);
        memset(&corr_mat, 0, sizeof(corr_mat));
        lmmc_mat_destroy(&data);
        memset(&data, 0, sizeof(data));
    }

    /* ===== Requirement 15.10: nCr(n, r) with r > n returns 0 ===== */
    {
        lmmc_real_t val = 999.0; /* initialize to non-zero to verify it gets set to 0 */

        lmmc_stats_nCr(&val, 5, 6);
        if (!lmmc_test_nearly_equal(val, 0.0, 1e-12)) { rc = 1; goto done; }

        lmmc_stats_nCr(&val, 3, 10);
        if (!lmmc_test_nearly_equal(val, 0.0, 1e-12)) { rc = 1; goto done; }

        lmmc_stats_nCr(&val, 0, 1);
        if (!lmmc_test_nearly_equal(val, 0.0, 1e-12)) { rc = 1; goto done; }
    }

done:
    lmmc_mat_destroy(&corr_mat);
    lmmc_mat_destroy(&cov_mat);
    lmmc_mat_destroy(&data);
    lmmc_vec_destroy(&big_v2);
    lmmc_vec_destroy(&big_v1);
    lmmc_vec_destroy(&v2);
    lmmc_vec_destroy(&v1);

    if (rc != 0) {
        printf("stats extended test failed\n");
    }
    return rc;
}
