#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <float.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

struct test_fixture {
    lmmc_rng_t *rng;
    lmmc_rng_t *reference;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_rng_destroy(fixture->rng);
    lmmc_rng_destroy(fixture->reference);
    free(fixture);
    *state = NULL;
    return 0;
}

static void test_large_binomial_moments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;
    const size_t samples = 10000;
    const size_t n = 1000000;
    const double p = 0.37;
    const double expected_mean = (double)n * p;
    const double expected_variance = (double)n * p * (1.0 - p);
    double sum = 0.0;
    double sum_sq = 0.0;

    size_t value = 0;

    assert_false(lmmc_rng_create(&fixture->rng) != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, UINT64_C(0x42545045313036)), LMMC_STATUS_OK);
    for (size_t i = 0; i < samples; ++i) {
        st = lmmc_rng_binomial(fixture->rng, n, p, &value);
        assert_int_equal(st, LMMC_STATUS_OK);
        assert_true(value <= n);
        sum += (double)value;
        sum_sq += (double)value * (double)value;
    }
    {
        const double mean = sum / (double)samples;
        const double variance = sum_sq / (double)samples - mean * mean;
        const double mean_se = sqrt(expected_variance / (double)samples);
        const double variance_se = expected_variance *
                                   sqrt(2.0 / (double)(samples - 1));
        assert_false(!(fabs(mean - expected_mean) <= 6.0 * mean_se) || !(fabs(variance - expected_variance) <= 6.0 * variance_se));
    }

    st = lmmc_rng_binomial(fixture->rng, SIZE_MAX, 0.5, &value);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(value > 0 && value < SIZE_MAX);
}

static void test_binomial_goodness_of_fit(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;
    const size_t samples = 100000;
    const size_t n = 8;
    const double p = 0.3;
    size_t observed[9] = {0};
    double probability[9];
    double chi_square = 0.0;
    const double q = 1.0 - p;
    const double threshold_df = 7.0;

    probability[0] = pow(q, (double)n);
    for (size_t k = 1; k <= n; ++k) {
        probability[k] = probability[k - 1] *
                         ((double)(n - k + 1) / (double)k) * (p / q);
    }
    assert_false(lmmc_rng_create(&fixture->rng) != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, UINT64_C(0x504D4643484932)), LMMC_STATUS_OK);
    for (size_t i = 0; i < samples; ++i) {
        size_t value;
        st = lmmc_rng_binomial(fixture->rng, n, p, &value);
        assert_int_equal(st, LMMC_STATUS_OK);
        assert_true(value <= n);
        ++observed[value];
    }
    for (size_t k = 0; k < 7; ++k) {
        const double expected = (double)samples * probability[k];
        const double delta = (double)observed[k] - expected;
        chi_square += delta * delta / expected;
    }
    {
        const double expected = (double)samples * (probability[7] + probability[8]);
        const double delta = (double)(observed[7] + observed[8]) - expected;
        chi_square += delta * delta / expected;
    }
    assert_true((chi_square < threshold_df + 6.0 * sqrt(2.0 * threshold_df)));
}

static void test_beta_extreme_support(void **state) {
    struct test_fixture *fixture = *state;
    const double tiny = nextafter(0.0, 1.0);
    const double shapes[][2] = {{0.001, 0.001}, {tiny, tiny}, {tiny, 0.001}, {0.001, tiny}, {tiny, 2.0}, {2.0, tiny}};

    size_t shape, i;
    assert_false(lmmc_rng_create(&fixture->rng) != LMMC_STATUS_OK);
    for (shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); ++shape) {
        assert_int_equal(lmmc_rng_seed(fixture->rng, 1), LMMC_STATUS_OK);
        for (i = 0; i < 100; ++i) {
            double value;
            assert_int_equal(
                lmmc_rng_beta(fixture->rng, shapes[shape][0],
                              shapes[shape][1], &value),
                LMMC_STATUS_OK);
            assert_true(isfinite(value));
            assert_true(value >= 0.0 && value <= 1.0);
        }
    }
}

static void test_beta_invalid_atomicity(void **state) {
    struct test_fixture *fixture = *state;
    const double invalid[] = {0.0, -1.0, NAN, INFINITY, -INFINITY};

    size_t i;

    double value = 42.0;
    assert_false(lmmc_rng_create(&fixture->rng) != LMMC_STATUS_OK);
    assert_false(lmmc_rng_create(&fixture->reference) != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 123), LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->reference, 123), LMMC_STATUS_OK);
    for (i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        assert_int_equal(
            lmmc_rng_beta(fixture->rng, invalid[i], 2.0, &value),
            LMMC_STATUS_INVALID_ARGUMENT);
        assert_true(value == 42.0);
        assert_int_equal(
            lmmc_rng_beta(fixture->rng, 2.0, invalid[i], &value),
            LMMC_STATUS_INVALID_ARGUMENT);
        assert_true(value == 42.0);
    }
    assert_int_equal(lmmc_rng_beta(NULL, 1.0, 1.0, &value),
                     LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(value == 42.0);
    assert_int_equal(lmmc_rng_beta(fixture->rng, 1.0, 1.0, NULL),
                     LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(value == 42.0);
    for (i = 0; i < 16; ++i) {
        assert_false(lmmc_rng_next_u64(fixture->rng) != lmmc_rng_next_u64(fixture->reference));
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_large_binomial_moments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_binomial_goodness_of_fit, setup, teardown),
        cmocka_unit_test_setup_teardown(test_beta_extreme_support, setup, teardown),
        cmocka_unit_test_setup_teardown(test_beta_invalid_atomicity, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
