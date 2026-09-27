/**
 * @file test_stats_dist_normal.c
 * @brief 正态分布的密度、累积概率及分位数测试。
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

static void test_normal_density_cdf(void **state) {
    (void)state;
    lmmc_real_t val;
    assert_true(lmmc_dist_normal_pdf(0.0, 0.0, 1.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.3989422804014327, 1e-10));

    val = 123.0;
    assert_true(lmmc_dist_normal_pdf(
                    0.0, 0.0, DBL_MIN * 0.0625, &val) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(val == 123.0);

    assert_true(lmmc_dist_normal_cdf(0.0, 0.0, 1.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.5, 1e-10));

    assert_true(lmmc_dist_normal_cdf(1.96, 0.0, 1.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.9750021048517796, 1e-6));

    {
        const lmmc_real_t expected = 0.5 * erfc(10.0 / sqrt(2.0));
        assert_true(lmmc_dist_normal_cdf(-10.0, 0.0, 1.0, &val) ==
                    LMMC_STATUS_OK);
        assert_true(fabs(val - expected) <= expected * 1e-12);
    }

    {
        const lmmc_real_t expected_cdf =
            0.5 * erfc(-2.0 / sqrt(2.0));
        const lmmc_real_t expected_pdf =
            exp(-2.0) / sqrt(2.0 * LMMC_PI) / DBL_MAX;
        assert_true(lmmc_dist_normal_cdf(
                        DBL_MAX, -DBL_MAX, DBL_MAX, &val) ==
                    LMMC_STATUS_OK);
        assert_true(fabs(val - expected_cdf) < 1e-15);
        assert_true(lmmc_dist_normal_pdf(
                        DBL_MAX, -DBL_MAX, DBL_MAX, &val) ==
                    LMMC_STATUS_OK);
        assert_true(val > 0.0 &&
                    fabs(val - expected_pdf) <= expected_pdf * 1e-12);
    }
}
static void test_normal_quantile(void **state) {
    (void)state;
    lmmc_real_t val;
    assert_true(lmmc_dist_normal_quantile(0.5, 0.0, 1.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.0, 1e-8));

    lmmc_real_t cdf_val;
    assert_int_equal(lmmc_dist_normal_cdf(1.5, 0.0, 1.0, &cdf_val),
                     LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_dist_normal_quantile(cdf_val, 0.0, 1.0, &val),
        LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 1.5, 1e-8));

    {
        const lmmc_real_t sigma = DBL_MAX * 0.5;
        const lmmc_real_t p = 0.0013498980316300933;
        assert_true(lmmc_dist_normal_quantile(
                        p, DBL_MAX, sigma, &val) == LMMC_STATUS_OK);
        assert_true(isfinite(val) &&
                    fabs(val / DBL_MAX + 0.5) < 1e-8);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_normal_density_cdf, setup, teardown),
        cmocka_unit_test_setup_teardown(test_normal_quantile, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
