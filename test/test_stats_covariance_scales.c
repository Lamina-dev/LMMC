/**
 * @file test_stats_covariance_scales.c
 * @brief 协方差尺度边界测试。
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
static void test_opposing_scales(void **state) {
    (void)state;
    lmmc_status_t st;

    double cov;
    lmmc_real_t large_data[] = {0x1p600, -0x1p600};
    lmmc_real_t small_data[] = {0x1p-600, -0x1p-600};
    lmmc_vec_t large = {2, large_data, 0};
    lmmc_vec_t small = {2, small_data, 0};

    for (int sample = 0; sample <= 1; ++sample) {
        const double expected = sample ? 2.0 : 1.0;
        if (sample) {
            st = lmmc_vec_covariance_sample(&large, &small, &cov);
        } else {
            st = lmmc_vec_covariance_population(&large, &small, &cov);
        }
        assert_false(st != LMMC_STATUS_OK || !isfinite(cov) || !(fabs(cov / expected - 1.0) <= 1e-12));
        if (sample) {
            st = lmmc_vec_covariance_sample(&small, &large, &cov);
        } else {
            st = lmmc_vec_covariance_population(&small, &large, &cov);
        }
        assert_false(st != LMMC_STATUS_OK || !isfinite(cov) || !(fabs(cov / expected - 1.0) <= 1e-12));
    }
}
static void test_adjacent_vector(void **state) {
    (void)state;
    lmmc_status_t st;

    double var, cov;
    const double base = 0x1p550;
    const double adjacent = nextafter(base, INFINITY);
    const double spacing = adjacent - base;
    lmmc_real_t nearby_data[] = {base, adjacent};
    lmmc_vec_t nearby = {2, nearby_data, 0};
    for (int sample = 0; sample <= 1; ++sample) {
        const double expected = spacing * spacing * (sample ? 0.5 : 0.25);
        if (sample) {
            st = lmmc_vec_variance_sample(&nearby, &var);
        } else {
            st = lmmc_vec_variance_population(&nearby, &var);
        }
        assert_false(st != LMMC_STATUS_OK || !isfinite(var) || !(fabs(var / expected - 1.0) <= 1e-12));
        if (sample) {
            st = lmmc_vec_covariance_sample(&nearby, &nearby, &cov);
        } else {
            st = lmmc_vec_covariance_population(&nearby, &nearby, &cov);
        }
        assert_false(st != LMMC_STATUS_OK || !isfinite(cov) || !(fabs(cov / expected - 1.0) <= 1e-12));
    }
}
static void test_adjacent_matrix(void **state) {
    (void)state;
    lmmc_status_t st;

    const double base = 0x1p550;
    const double adjacent = nextafter(base, INFINITY);
    const double spacing = adjacent - base;
    lmmc_real_t nearby_data[] = {base, adjacent};
    lmmc_real_t matrix_output;
    lmmc_mat_t matrix = {2, 1, 1, nearby_data, 0};
    lmmc_mat_t matrix_cov = {1, 1, 1, &matrix_output, 0};
    for (int sample = 0; sample <= 1; ++sample) {
        const double expected = spacing * spacing * (sample ? 0.5 : 0.25);
        st = sample ? lmmc_mat_covariance_sample(&matrix, &matrix_cov)
                    : lmmc_mat_covariance_population(&matrix, &matrix_cov);
        assert_false(st != LMMC_STATUS_OK || !isfinite(matrix_output) || !(fabs(matrix_output / expected - 1.0) <= 1e-12));
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_opposing_scales),
        cmocka_unit_test(test_adjacent_vector),
        cmocka_unit_test(test_adjacent_matrix),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
