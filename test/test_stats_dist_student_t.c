/**
 * @file test_stats_dist_student_t.c
 * @brief Student t 分布测试。
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

static void test_t_dist(void **state) {
    (void)state;
    lmmc_real_t val;

    assert_true(lmmc_dist_t_cdf(0.0, 5.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.5, 1e-10));
    {
        const double degrees[] = {nextafter(32.0, 0.0), 32.0,
                                  nextafter(32.0, INFINITY), 1e8};
        size_t i;
        for (i = 0; i < 4; ++i) {
            const double expected = i == 3 ? 0.398942279404077 : 0.39583818969061102;
            assert_true(lmmc_dist_t_pdf(0.0, degrees[i], &val) == LMMC_STATUS_OK);
            assert_true(isfinite(val) && fabs(val / expected - 1.0) <= 2e-12);
        }
    }

    lmmc_real_t cdf_val;
    assert_true(lmmc_dist_t_cdf(2.0, 10.0, &cdf_val) == LMMC_STATUS_OK);
    assert_true(lmmc_dist_t_quantile(cdf_val, 10.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 2.0, 1e-6));
}

static void test_t_extreme_tails(void **state) {
    (void)state;
    lmmc_real_t val;

    {
        const lmmc_real_t extreme = DBL_MAX;
        const lmmc_real_t expected = atan(1.0 / extreme) / LMMC_PI;
        assert_true(lmmc_dist_t_cdf(-extreme, 1.0, &val) == LMMC_STATUS_OK);
        assert_true(val > 0.0 &&
                    fabs(val - expected) <= expected * 1e-12);
    }

    {
        const lmmc_real_t extreme = 1.0e200;
        const lmmc_real_t df = 0.1;
        const lmmc_real_t log_power =
            2.0 * log(extreme) - log(df);
        const lmmc_real_t expected = exp(
            lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0) -
            0.5 * (log(df) + log(LMMC_PI)) -
            ((df + 1.0) / 2.0) * log_power);
        assert_true(lmmc_dist_t_pdf(extreme, df, &val) == LMMC_STATUS_OK);
        assert_true(val > 0.0 &&
                    fabs(val - expected) <= expected * 1e-12);
    }

    {
        const lmmc_real_t expected_cdf =
            0.84134474606854294859;
        const lmmc_real_t expected_quantile =
            1.95996398454005423552;
        assert_true(lmmc_dist_t_pdf(0.0, DBL_MAX, &val) ==
                    LMMC_STATUS_OK);
        assert_true(fabs(val - 1.0 / sqrt(2.0 * LMMC_PI)) < 1e-15);
        assert_true(lmmc_dist_t_cdf(1.0, DBL_MAX, &val) ==
                    LMMC_STATUS_OK);
        assert_true(fabs(val - expected_cdf) < 1e-15);
        assert_true(lmmc_dist_t_quantile(0.975, DBL_MAX, &val) ==
                    LMMC_STATUS_OK);
        assert_true(fabs(val - expected_quantile) < 5e-8);
    }
}

static void test_t_subnormal_degrees(void **state) {
    (void)state;
    lmmc_real_t val = 42.0;
    const lmmc_real_t expected_density = sqrt(DBL_TRUE_MIN) / 2.0;

    assert_int_equal(lmmc_dist_t_pdf(0.0, DBL_TRUE_MIN, &val),
                     LMMC_STATUS_OK);
    assert_true(val > 0.0 && fabs(val / expected_density - 1.0) < 1e-12);
    assert_int_equal(lmmc_dist_t_pdf(sqrt(DBL_TRUE_MIN), DBL_TRUE_MIN, &val),
                     LMMC_STATUS_OK);
    assert_true(val > 0.0 &&
                fabs(val / (expected_density / sqrt(2.0)) - 1.0) < 1e-12);
    assert_int_equal(lmmc_dist_t_cdf(1.0, DBL_TRUE_MIN, &val),
                     LMMC_STATUS_OK);
    assert_true(val == 0.5);
    assert_int_equal(lmmc_dist_t_cdf(-1.0, DBL_TRUE_MIN, &val),
                     LMMC_STATUS_OK);
    assert_true(val == 0.5);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_t_dist, setup, teardown),
        cmocka_unit_test_setup_teardown(test_t_extreme_tails, setup, teardown),
        cmocka_unit_test_setup_teardown(test_t_subnormal_degrees, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
