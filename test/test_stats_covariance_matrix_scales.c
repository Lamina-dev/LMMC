/**
 * @file test_stats_covariance_matrix_scales.c
 * @brief 协方差矩阵的尺度边界测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"
static void test_large_variance_self_covariance(void **state) {
    (void)state;
    lmmc_status_t st;

    double var, cov;
    const double a = 0x1p511;
    lmmc_real_t large_data[] = {a, -a, a, -a};
    lmmc_vec_t large = {4, large_data, 0};
    const double expected[] = {0x1p1022};
    for (int sample = 0; sample <= 1; ++sample) {
        const double factor = sample ? 4.0 / 3.0 : 1.0;
        if (sample) {
            st = lmmc_vec_variance_sample(&large, &var);
        } else {
            st = lmmc_vec_variance_population(&large, &var);
        }
        assert_false(st != LMMC_STATUS_OK || !isfinite(var) || !(fabs(var / (expected[0] * factor) - 1.0) <= 1e-12));
        if (sample) {
            st = lmmc_vec_covariance_sample(&large, &large, &cov);
        } else {
            st = lmmc_vec_covariance_population(&large, &large, &cov);
        }
        assert_false(st != LMMC_STATUS_OK || !isfinite(cov) || !(fabs(cov / var - 1.0) <= 1e-12));
    }
}
static void test_signed_cross_covariance(void **state) {
    (void)state;
    lmmc_status_t st;

    double cov;
    const double a = 0x1p511;
    lmmc_real_t large_data[] = {a, -a, a, -a};
    lmmc_vec_t large = {4, large_data, 0};
    const double expected[] = {0x1p1022};
    const double b = 0x1p-511;
    lmmc_real_t negative_data[] = {-a, a, -a, a};
    lmmc_real_t small_data[] = {b, -b, b, -b};
    lmmc_vec_t negative = {4, negative_data, 0};
    lmmc_vec_t small = {4, small_data, 0};
    for (int sample = 0; sample <= 1; ++sample) {
        const double factor = sample ? 4.0 / 3.0 : 1.0;
        const double var = expected[0] * factor;
        if (sample) {
            st = lmmc_vec_covariance_sample(&large, &negative, &cov);
        } else {
            st = lmmc_vec_covariance_population(&large, &negative, &cov);
        }
        assert_false(st != LMMC_STATUS_OK || !isfinite(cov) || !(fabs(cov / -var - 1.0) <= 1e-12));
        if (sample) {
            st = lmmc_vec_covariance_sample(&large, &small, &cov);
        } else {
            st = lmmc_vec_covariance_population(&large, &small, &cov);
        }
        assert_false(st != LMMC_STATUS_OK || !isfinite(cov) || !(fabs(cov / factor - 1.0) <= 1e-12));
    }
}
static void test_large_matrix_covariance(void **state) {
    (void)state;
    lmmc_status_t st;

    const double a = 0x1p511;
    const double b = 0x1p-511;
    lmmc_real_t large_data[] = {a, -a, a, -a};
    lmmc_real_t matrix_data[] = {
        a, -a, b, -a, a, -b, a, -a, b, -a, a, -b};
    lmmc_real_t matrix_output[9];
    lmmc_real_t single_output;
    const double expected[] = {
        0x1p1022, -0x1p1022, 1.0,
        -0x1p1022, 0x1p1022, -1.0,
        1.0, -1.0, 0x1p-1022};
    lmmc_mat_t matrix = {4, 3, 3, matrix_data, 0};
    lmmc_mat_t matrix_cov = {3, 3, 3, matrix_output, 0};
    lmmc_mat_t single = {4, 1, 1, large_data, 0};
    lmmc_mat_t single_cov = {1, 1, 1, &single_output, 0};
    for (int sample = 0; sample <= 1; ++sample) {
        const double factor = sample ? 4.0 / 3.0 : 1.0;
        const double var = expected[0] * factor;
        st = sample ? lmmc_mat_covariance_sample(&single, &single_cov)
                    : lmmc_mat_covariance_population(&single, &single_cov);
        assert_int_equal(st, LMMC_STATUS_OK);
        assert_true(isfinite(single_output));
        assert_true(fabs(single_output / var - 1.0) <= 1e-12);
        st = sample ? lmmc_mat_covariance_sample(&matrix, &matrix_cov)
                    : lmmc_mat_covariance_population(&matrix, &matrix_cov);
        assert_int_equal(st, LMMC_STATUS_OK);
        for (size_t i = 0; i < 9; ++i) {
            assert_true(isfinite(matrix_output[i]));
            assert_true(fabs(matrix_output[i] / (expected[i] * factor) - 1.0) <= 1e-12);
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_large_variance_self_covariance),
        cmocka_unit_test(test_signed_cross_covariance),
        cmocka_unit_test(test_large_matrix_covariance),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
