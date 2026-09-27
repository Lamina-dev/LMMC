/**
 * @file test_stats_dist_gamma.c
 * @brief Gamma 分布测试。
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

static void test_gamma_cdf(void **state) {
    (void)state;
    lmmc_real_t val;
    assert_true(lmmc_dist_gamma_cdf(0.0, 2.0, 1.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.0, 1e-10));

    assert_true(lmmc_dist_gamma_cdf(1.0, 1.0, 1.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 1.0 - exp(-1.0), 1e-10));

    lmmc_real_t cdf_val;
    assert_int_equal(lmmc_dist_gamma_cdf(3.0, 2.0, 1.5, &cdf_val),
                     LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_dist_gamma_quantile(cdf_val, 2.0, 1.5, &val),
        LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 3.0, 1e-6));

    {
        const lmmc_real_t expected =
            erf(sqrt(DBL_MIN) / sqrt(DBL_MAX));
        assert_true(lmmc_dist_gamma_cdf(
                        DBL_MIN, 0.5, DBL_MAX, &val) ==
                    LMMC_STATUS_OK);
        assert_true(val > 0.0 &&
                    fabs(val / expected - 1.0) < 1e-12);
    }
}
static void test_gamma_density(void **state) {
    (void)state;
    lmmc_real_t val;
    {
        const lmmc_real_t shape = 1.0e18;
        const lmmc_real_t expected =
            1.0 / sqrt(2.0 * LMMC_PI * shape);
        assert_true(lmmc_dist_gamma_pdf(shape, shape, 1.0, &val) ==
                    LMMC_STATUS_OK);
        assert_true(isfinite(val) &&
                    fabs(val / expected - 1.0) < 1e-12);
    }

    {
        const lmmc_real_t shape = 0x1.8p1023;
        const lmmc_real_t center =
            1.0 / (sqrt(2.0 * LMMC_PI) * sqrt(shape));
        const lmmc_real_t far_x = 0x1p1023;
        const lmmc_real_t near_x = 0x1.6p1023;
        const lmmc_real_t far_ratio = far_x / shape;
        const lmmc_real_t near_ratio = near_x / shape;
        /**
         * @brief 参考密度比 f(x)/f(a) = exp(a * (log(x/a) + 1 - x/a) - log(x/a))。
         * 两个偏离中心的尾部值均下溢，包括偏差级数区域内的点；两处 x 与形状参数之和均溢出。
         */
        const lmmc_real_t far_expected = center * exp(
                                                      shape * (log(far_ratio) + 1.0 - far_ratio) - log(far_ratio));
        const lmmc_real_t near_expected = center * exp(
                                                       shape * (log(near_ratio) + 1.0 - near_ratio) - log(near_ratio));
        assert_true(lmmc_dist_gamma_pdf(shape, shape, 1.0, &val) ==
                    LMMC_STATUS_OK);
        assert_true(fabs(val / center - 1.0) < 1e-12);
        assert_true(lmmc_dist_gamma_pdf(far_x, shape, 1.0, &val) ==
                    LMMC_STATUS_OK);
        assert_true(val == far_expected && val == 0.0);
        assert_true(lmmc_dist_gamma_pdf(near_x, shape, 1.0, &val) ==
                    LMMC_STATUS_OK);
        assert_true(val == near_expected && val == 0.0);
    }

    assert_true(lmmc_dist_gamma_pdf(30.0, 16.0, 2.0, &val) ==
                LMMC_STATUS_OK);
    assert_true(fabs(val - 0.051217933332267094) <
                0.051217933332267094e-12);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_gamma_cdf, setup, teardown),
        cmocka_unit_test_setup_teardown(test_gamma_density, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
