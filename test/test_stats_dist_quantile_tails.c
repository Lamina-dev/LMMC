/**
 * @file test_stats_dist_quantile_tails.c
 * @brief 分位数尾部边界测试。
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

static void test_quantile_tail_regressions(void **state) {
    (void)state;
    lmmc_real_t q, cdf;
    lmmc_status_t status;

    status = lmmc_dist_chi2_quantile(0.01, 0.1, &q);
    assert_int_equal(status, LMMC_STATUS_OK);
    status = lmmc_dist_chi2_cdf(q, 0.1, &cdf);
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(fabs(cdf - 0.01) < 1e-10);

    status = lmmc_dist_gamma_quantile(0.01, 2.0, 2.0, &q);
    assert_int_equal(status, LMMC_STATUS_OK);
    status = lmmc_dist_gamma_cdf(q, 2.0, 2.0, &cdf);
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(fabs(cdf - 0.01) < 1e-10);

    status = lmmc_dist_f_quantile(0.5, 100.0, 1.0, &q);
    assert_int_equal(status, LMMC_STATUS_OK);
    status = lmmc_dist_f_cdf(q, 100.0, 1.0, &cdf);
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(fabs(cdf - 0.5) < 1e-10);

    status = lmmc_dist_beta_quantile(0.99, 100.0, 2.0, &q);
    assert_int_equal(status, LMMC_STATUS_OK);
    status = lmmc_dist_beta_cdf(q, 100.0, 2.0, &cdf);
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(fabs(cdf - 0.99) < 1e-10);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_quantile_tail_regressions, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
