/**
 * @file test_stats_dist_beta.c
 * @brief Beta 分布测试。
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

static void test_beta_dist(void **state) {
    (void)state;
    lmmc_real_t val;

    assert_true(lmmc_dist_beta_cdf(0.0, 2.0, 3.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.0, 1e-10));
    assert_true(lmmc_dist_beta_cdf(1.0, 2.0, 3.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 1.0, 1e-10));

    assert_true(lmmc_dist_beta_cdf(0.5, 1.0, 1.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.5, 1e-10));

    {
        const lmmc_real_t shape = 1.0e18;
        const lmmc_real_t expected =
            sqrt(4.0 * shape / LMMC_PI);
        assert_true(lmmc_dist_beta_pdf(0.5, shape, shape, &val) ==
                    LMMC_STATUS_OK);
        assert_true(isfinite(val) &&
                    fabs(val / expected - 1.0) < 1e-12);
    }

    {
        const lmmc_real_t shape = DBL_MAX;
        const lmmc_real_t expected =
            2.0 * sqrt(shape / LMMC_PI);
        assert_true(lmmc_dist_beta_pdf(0.5, shape, shape, &val) ==
                    LMMC_STATUS_OK);
        assert_true(isfinite(val) &&
                    fabs(val / expected - 1.0) < 1e-12);
    }

    assert_true(lmmc_dist_beta_pdf(0.4, 16.0, 20.0, &val) ==
                LMMC_STATUS_OK);
    assert_true(fabs(val - 4.2502261912068876) <
                4.2502261912068876e-12);
}

static void test_beta_mixed_shapes_and_roundtrip(void **state) {
    (void)state;
    lmmc_real_t val;
    {
        const double a = 1e16;
        const double x = nextafter(1.0, 0.0);
        const double complement = 1.0 - x;
        const double shapes[] = {1.0, 2.0};
        size_t i;
        for (i = 0; i < 2; ++i) {
            const double b = shapes[i];
            const double reference = b == 1.0
                                         ? a * exp((a - 1.0) * log(x))
                                         : a * ((a + 1.0) * complement) * exp((a - 1.0) * log(x));
            assert_true(lmmc_dist_beta_pdf(x, a, b, &val) == LMMC_STATUS_OK);
            assert_true(isfinite(val) && fabs(val / reference - 1.0) <= 2e-12);
            assert_true(lmmc_dist_beta_pdf(complement, b, a, &val) == LMMC_STATUS_OK);
            assert_true(isfinite(val) && fabs(val / reference - 1.0) <= 2e-12);
        }
        assert_true(lmmc_dist_beta_cdf(x, a, 1.0, &val) == LMMC_STATUS_OK);
        assert_true(isfinite(val) && fabs(val / exp(a * log(x)) - 1.0) <= 2e-12);
    }
    lmmc_real_t cdf_val;
    assert_true(lmmc_dist_beta_cdf(0.3, 2.0, 5.0, &cdf_val) == LMMC_STATUS_OK);
    assert_true(lmmc_dist_beta_quantile(cdf_val, 2.0, 5.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.3, 1e-6));
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_beta_dist, setup, teardown),
        cmocka_unit_test_setup_teardown(test_beta_mixed_shapes_and_roundtrip, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
