/**
 * @file test_stats_dist_boundaries.c
 * @brief 概率分布与描述性统计的边界测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static int setup(void **state) {
    (void)state;
    assert_int_equal(lmmc_init(), LMMC_STATUS_OK);
    return 0;
}

static int teardown(void **state) {
    (void)state;
    assert_int_equal(lmmc_deinit(), LMMC_STATUS_OK);
    return 0;
}

static void test_density_failure_contracts(void **state) {
    (void)state;
    lmmc_real_t val = 123.0;

    assert_true(lmmc_dist_chi2_pdf(0.0, 1.0, &val) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(val == 123.0);
    assert_true(lmmc_dist_f_pdf(0.0, 1.0, 2.0, &val) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(val == 123.0);
    assert_true(lmmc_dist_gamma_pdf(0.0, 0.5, 1.0, &val) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(val == 123.0);
    assert_true(lmmc_dist_beta_pdf(0.0, 0.5, 1.0, &val) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(val == 123.0);
}

static void test_cdf_large_parameter_contracts(void **state) {
    (void)state;
    const lmmc_real_t gamma_center =
        0.5 + 1.0 / (3.0 * sqrt(2.0 * acos(-1.0) * 100000.0));
    lmmc_real_t val = 123.0;
    assert_true(lmmc_dist_beta_cdf(0.5, DBL_MAX, DBL_MAX, &val) ==
                LMMC_STATUS_OK);
    assert_true(val == 0.5);
    assert_true(lmmc_dist_f_cdf(1.0, DBL_MAX, DBL_MAX, &val) ==
                LMMC_STATUS_OK);
    assert_true(val == 0.5);

    assert_true(lmmc_dist_gamma_cdf(100000.0, 100000.0, 1.0, &val) ==
                LMMC_STATUS_OK);
    assert_true(fabs(val - gamma_center) < 1e-8);

    assert_true(lmmc_dist_beta_cdf(0.5, 10000000.0, 10000000.0, &val) ==
                LMMC_STATUS_OK);
    assert_true(val == 0.5);

    assert_true(lmmc_dist_binomial_cdf(9999999, 19999999, 0.5, &val) ==
                LMMC_STATUS_OK);
    assert_true(val == 0.5);

    assert_true(lmmc_dist_poisson_cdf(99999, 100000.0, &val) ==
                LMMC_STATUS_OK);
    assert_true(fabs(val - (1.0 - gamma_center)) < 1e-8);
}

static void test_continuous_dist_rejects_nonfinite(void **state) {
    (void)state;
    lmmc_real_t val;

    assert_true(lmmc_dist_normal_pdf(NAN, 0.0, 1.0, &val) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_dist_normal_cdf(0.0, NAN, 1.0, &val) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_dist_normal_quantile(NAN, 0.0, 1.0, &val) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_dist_normal_quantile(
                    0.75, DBL_MAX, DBL_MAX, &val) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_dist_t_cdf(0.0, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_dist_chi2_pdf(NAN, 2.0, &val) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_dist_f_cdf(1.0, INFINITY, 2.0, &val) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_dist_gamma_quantile(0.5, 1.0, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_dist_beta_cdf(0.5, NAN, 1.0, &val) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_dist_binomial_pmf(1, 2, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_dist_binomial_cdf(1, 2, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_dist_poisson_pmf(1, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_dist_poisson_cdf(1, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_density_failure_contracts, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cdf_large_parameter_contracts, setup, teardown),
        cmocka_unit_test_setup_teardown(test_continuous_dist_rejects_nonfinite, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
