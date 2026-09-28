/**
 * @file test_stats_dist_chi_square_f.c
 * @brief 卡方分布与 F 分布的密度、累积概率及分位数测试。
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

static void test_chi2_dist(void **state) {
    (void)state;
    lmmc_real_t val;

    {
        const lmmc_real_t expected =
            0.5 / sqrt(LMMC_PI) / sqrt(DBL_MAX);
        assert_true(lmmc_dist_chi2_pdf(DBL_MAX, DBL_MAX, &val) ==
                    LMMC_STATUS_OK);
        assert_true(val > 0.0 &&
                    fabs(val / expected - 1.0) < 1e-12);
    }

    assert_true(lmmc_dist_chi2_cdf(0.0, 5.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.0, 1e-10));

    lmmc_real_t cdf_val;
    assert_int_equal(lmmc_dist_chi2_cdf(5.0, 3.0, &cdf_val),
                     LMMC_STATUS_OK);
    assert_int_equal(lmmc_dist_chi2_quantile(cdf_val, 3.0, &val),
                     LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 5.0, 1e-6));

    {
        const lmmc_real_t expected =
            erf(sqrt(DBL_TRUE_MIN) / sqrt(2.0));
        assert_true(lmmc_dist_chi2_cdf(DBL_TRUE_MIN, 1.0, &val) ==
                    LMMC_STATUS_OK);
        assert_true(val > 0.0 &&
                    fabs(val / expected - 1.0) < 1e-12);
    }
}

static void test_chi2_subnormal_degrees(void **state) {
    (void)state;
    lmmc_real_t val = 42.0;

    assert_int_equal(lmmc_dist_chi2_cdf(1.0, DBL_TRUE_MIN, &val),
                     LMMC_STATUS_OK);
    assert_true(val == 1.0);

    assert_int_equal(lmmc_dist_chi2_pdf(DBL_TRUE_MIN, DBL_TRUE_MIN, &val),
                     LMMC_STATUS_OK);
    assert_true(fabs(val - 0.5) < 1e-12);
    assert_int_equal(lmmc_dist_chi2_pdf(2.0 * DBL_TRUE_MIN, DBL_TRUE_MIN, &val),
                     LMMC_STATUS_OK);
    assert_true(fabs(val - 0.25) < 1e-12);
}

static void test_f_dist(void **state) {
    (void)state;
    lmmc_real_t val;

    assert_true(lmmc_dist_f_cdf(0.0, 5.0, 10.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.0, 1e-10));
    {
        const double degrees[] = {2.0, 31.0, 32.0, 10000.0, 2e16, DBL_MAX};
        size_t i;
        for (i = 0; i < sizeof(degrees) / sizeof(degrees[0]); ++i) {
            /**
             * @brief df2=2 时对 (x/(x+2/df1))^(df1/2) 求导。
             * 在 x=1 处为 a/(a+1) * (1+1/a)^(-a)。
             */
            const double a = degrees[i] / 2.0;
            const double expected = (a / (a + 1.0)) * exp(-a * log1p(1.0 / a));
            assert_true(lmmc_dist_f_pdf(1.0, degrees[i], 2.0, &val) == LMMC_STATUS_OK);
            assert_true(isfinite(val) && fabs(val / expected - 1.0) <= 2e-12);
        }
    }

    {
        const lmmc_real_t expected = exp(-500.0 * log1p(0.2));
        assert_true(lmmc_dist_f_cdf(0.01, 1000.0, 2.0, &val) ==
                    LMMC_STATUS_OK);
        assert_true(val > 0.0 && fabs(val / expected - 1.0) < 1e-12);
    }

    lmmc_real_t cdf_val;
    assert_true(lmmc_dist_f_cdf(2.0, 5.0, 10.0, &cdf_val) == LMMC_STATUS_OK);
    assert_true(lmmc_dist_f_quantile(cdf_val, 5.0, 10.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 2.0, 1e-5));

    {
        const lmmc_real_t expected =
            sqrt(DBL_MAX / (8.0 * LMMC_PI));
        assert_true(lmmc_dist_f_pdf(1.0, DBL_MAX, DBL_MAX, &val) ==
                    LMMC_STATUS_OK);
        assert_true(isfinite(val) &&
                    fabs(val / expected - 1.0) < 1e-12);
    }

    assert_true(lmmc_dist_f_pdf(0.8, 32.0, 40.0, &val) ==
                LMMC_STATUS_OK);
    assert_true(fabs(val - 1.1859038610805410002) <
                1.1859038610805410002e-12);
}

static void test_f_subnormal_degrees(void **state) {
    (void)state;
    lmmc_real_t val = 42.0;

    assert_int_equal(lmmc_dist_f_cdf(1.0, DBL_TRUE_MIN, 1.0, &val),
                     LMMC_STATUS_OK);
    assert_true(val == 1.0);
    assert_int_equal(lmmc_dist_f_cdf(1.0, DBL_TRUE_MIN, DBL_TRUE_MIN, &val),
                     LMMC_STATUS_OK);
    assert_true(fabs(val - 0.5) < 1e-12);
    assert_int_equal(lmmc_dist_f_cdf(1.0, DBL_TRUE_MIN,
                                     2.0 * DBL_TRUE_MIN, &val),
                     LMMC_STATUS_OK);
    assert_true(fabs(val - 2.0 / 3.0) < 1e-12);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_chi2_dist, setup, teardown),
        cmocka_unit_test_setup_teardown(test_chi2_subnormal_degrees, setup, teardown),
        cmocka_unit_test_setup_teardown(test_f_dist, setup, teardown),
        cmocka_unit_test_setup_teardown(test_f_subnormal_degrees, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
