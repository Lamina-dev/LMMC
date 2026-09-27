/**
 * @file test_stats_dist_discrete.c
 * @brief 离散概率分布测试。
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

static void test_binomial_mass(void **state) {
    (void)state;
    lmmc_real_t val;
    assert_true(lmmc_dist_binomial_pmf(5, 10, 0.5, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.24609375, 1e-10));

    assert_true(lmmc_dist_binomial_pmf(0, SIZE_MAX, 0.5, &val) ==
                LMMC_STATUS_OK);
    assert_true(isfinite(val) && val == 0.0);

#if SIZE_MAX >= UINT64_C(1000000000000000000)
    {
        const size_t large_n = (size_t)UINT64_C(1000000000000000000);
        const lmmc_real_t expected =
            sqrt(2.0 / (acos(-1.0) * (lmmc_real_t)large_n));
        assert_true(lmmc_dist_binomial_pmf(
                        large_n / 2, large_n, 0.5, &val) ==
                    LMMC_STATUS_OK);
        assert_true(isfinite(val) &&
                    fabs(val / expected - 1.0) < 1e-12);
    }
#endif
}
static void test_binomial_tiny_probability(void **state) {
    (void)state;
    lmmc_real_t val;
#if SIZE_MAX >= UINT64_C(1152921504606846976)
    {
        const size_t n = (size_t)UINT64_C(1152921504606846976);
        const lmmc_real_t p = 0x1p-60;
        const lmmc_real_t zero_probability =
            exp((lmmc_real_t)n * log1p(-p));
        const lmmc_real_t one_cdf = zero_probability *
                                    (1.0 + (lmmc_real_t)n * p / (1.0 - p));
        assert_true(lmmc_dist_binomial_cdf(0, n, p, &val) == LMMC_STATUS_OK);
        assert_true(fabs(val / zero_probability - 1.0) < 1e-12);
        assert_true(lmmc_dist_binomial_cdf(1, n, p, &val) == LMMC_STATUS_OK);
        assert_true(fabs(val / one_cdf - 1.0) < 1e-12);
    }

    {
        const size_t n = (size_t)UINT64_C(1152921504606846976);
        const lmmc_real_t p = 0x1p-54;
        lmmc_real_t term = exp((lmmc_real_t)n * log1p(-p));
        lmmc_real_t expected = term;
        size_t k;
        for (k = 1; k <= 4; ++k) {
            term *= ((lmmc_real_t)(n - k + 1) * p) /
                    ((lmmc_real_t)k * (1.0 - p));
            expected += term;
        }
        assert_true(lmmc_dist_binomial_cdf(4, n, p, &val) == LMMC_STATUS_OK);
        assert_true(val > 0.0 && fabs(val / expected - 1.0) < 1e-12);
    }
#endif
}
static void test_binomial_support(void **state) {
    (void)state;
    lmmc_real_t val;
    assert_true(lmmc_dist_binomial_cdf(10, 10, 0.5, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 1.0, 1e-10));
}
static void test_poisson_dist(void **state) {
    (void)state;
    lmmc_real_t val;

    assert_true(lmmc_dist_poisson_pmf(0, 3.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, exp(-3.0), 1e-10));

    assert_true(lmmc_dist_poisson_pmf(3, 3.0, &val) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 27.0 * exp(-3.0) / 6.0, 1e-10));

#if SIZE_MAX >= UINT64_C(1000000000000000000)
    {
        const size_t large_k = (size_t)UINT64_C(1000000000000000000);
        const lmmc_real_t lambda = 1.0e18;
        const lmmc_real_t expected =
            1.0 / sqrt(2.0 * acos(-1.0) * lambda);
        assert_true(lmmc_dist_poisson_pmf(
                        large_k, lambda, &val) == LMMC_STATUS_OK);
        assert_true(isfinite(val) &&
                    fabs(val / expected - 1.0) < 1e-12);
    }
#endif
    assert_true(lmmc_dist_poisson_cdf(0, 40.0, &val) == LMMC_STATUS_OK);
    assert_true(fabs(val - exp(-40.0)) <= exp(-40.0) * 1e-12);
    assert_true(lmmc_dist_poisson_cdf(SIZE_MAX, 40.0, &val) == LMMC_STATUS_OK);
    assert_true(val == 1.0);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_binomial_mass, setup, teardown),
        cmocka_unit_test_setup_teardown(test_binomial_tiny_probability, setup, teardown),
        cmocka_unit_test_setup_teardown(test_binomial_support, setup, teardown),
        cmocka_unit_test_setup_teardown(test_poisson_dist, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
