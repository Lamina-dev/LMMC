/**
 * @file test_random_extended.c
 * 针对 LMMC 中 random extended 相关接口的单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define NUM_SAMPLES 10000

struct test_fixture {
    lmmc_rng_t *rng;
    lmmc_rng_t *rng1;
    lmmc_rng_t *rng2;
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
    lmmc_rng_destroy(fixture->rng1);
    lmmc_rng_destroy(fixture->rng2);
    free(fixture);
    *state = NULL;
    return 0;
}

static int cmp_int(const void *a, const void *b) {
    int va = *(const int *)a;
    int vb = *(const int *)b;
    return (va > vb) - (va < vb);
}

static void test_poisson_output_range(void **state) {
    struct test_fixture *fixture = *state;

    size_t sample = 123;
    assert_false(lmmc_rng_create(&fixture->rng) != LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_rng_seed(fixture->rng, UINT64_C(0x504f4953534f4e)),
        LMMC_STATUS_OK);
    const lmmc_status_t status =
        lmmc_rng_poisson(fixture->rng, DBL_MAX, &sample);
    assert_int_equal(status, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(sample, 123);
}

static void test_seed_reproducibility(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int i;

    st = lmmc_rng_create(&fixture->rng1);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_rng_create(&fixture->rng2);
    assert_false(st != LMMC_STATUS_OK);

    assert_int_equal(lmmc_rng_seed(fixture->rng1, 12345),
                     LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng2, 12345),
                     LMMC_STATUS_OK);

    for (i = 0; i < 100; i++) {
        uint64_t v1 = lmmc_rng_next_u64(fixture->rng1);
        uint64_t v2 = lmmc_rng_next_u64(fixture->rng2);
        assert_false(v1 != v2);
    }

    assert_int_equal(lmmc_rng_seed(fixture->rng1, 99999),
                     LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng2, 99999),
                     LMMC_STATUS_OK);
    for (i = 0; i < 50; i++) {
        lmmc_real_t u1, u2;
        assert_int_equal(
            lmmc_rng_uniform(fixture->rng1, 0.0, 1.0, &u1),
            LMMC_STATUS_OK);
        assert_int_equal(
            lmmc_rng_uniform(fixture->rng2, 0.0, 1.0, &u2),
            LMMC_STATUS_OK);
        assert_false(u1 != u2);
    }
}

static void test_uniform_moments(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int i;
    double sum = 0.0;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 42), LMMC_STATUS_OK);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_uniform(fixture->rng, 0.0, 1.0, &val);
        assert_false(st != LMMC_STATUS_OK);
        assert_false(!(val >= 0.0) || !(val <= 1.0));
        sum += val;
    }

    double mean = sum / NUM_SAMPLES;
    assert_true((fabs(mean - 0.5) <= 0.02));
}

static void test_normal_moments(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int i;
    double sum = 0.0;
    double sum_sq = 0.0;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 77), LMMC_STATUS_OK);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_normal(fixture->rng, 0.0, 1.0, &val);
        assert_false(st != LMMC_STATUS_OK);
        sum += val;
        sum_sq += val * val;
    }

    double mean = sum / NUM_SAMPLES;
    double variance = (sum_sq / NUM_SAMPLES) - (mean * mean);
    double stddev = sqrt(variance);

    assert_true((fabs(mean) <= 0.05));
    assert_true((fabs(stddev - 1.0) <= 0.05));
}

static void test_exponential_moments(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int i;
    double sum = 0.0;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 123), LMMC_STATUS_OK);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_exponential(fixture->rng, 2.0, &val);
        assert_false(st != LMMC_STATUS_OK);
        assert_true((val >= 0.0));
        sum += val;
    }

    double mean = sum / NUM_SAMPLES;
    assert_true((fabs(mean - 0.5) <= 0.05));
}

static void test_shuffle_permutation(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int sorted_orig[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int i;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 555), LMMC_STATUS_OK);

    st = lmmc_rng_shuffle(fixture->rng, arr, 10, sizeof(int));
    assert_false(st != LMMC_STATUS_OK);

    qsort(arr, 10, sizeof(int), cmp_int);
    for (i = 0; i < 10; i++) {
        assert_false(arr[i] != sorted_orig[i]);
    }
}

static void test_singleton_shuffle(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int val = 42;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 777), LMMC_STATUS_OK);

    st = lmmc_rng_shuffle(fixture->rng, &val, 1, sizeof(int));
    assert_false(st != LMMC_STATUS_OK);
    assert_false(val != 42);
}

static void test_uniform_bounds(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    lmmc_real_t val;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 1), LMMC_STATUS_OK);

    st = lmmc_rng_uniform(fixture->rng, 5.0, 5.0, &val);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_rng_uniform(fixture->rng, 10.0, 3.0, &val);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_rng_uniform(fixture->rng, -DBL_MAX, DBL_MAX, &val);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(isfinite(val));
    assert_true(val >= -DBL_MAX && val < DBL_MAX);

    st = lmmc_rng_uniform(fixture->rng, NAN, 1.0, &val);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_exponential_domain(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    lmmc_real_t val;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 2), LMMC_STATUS_OK);

    st = lmmc_rng_exponential(fixture->rng, 0.0, &val);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_rng_exponential(fixture->rng, -1.0, &val);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_rng_exponential(fixture->rng, NAN, &val);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_normal_gamma_domain(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    lmmc_real_t val;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 3), LMMC_STATUS_OK);

    st = lmmc_rng_normal(fixture->rng, 0.0, -1.0, &val);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_rng_normal(fixture->rng, NAN, 1.0, &val);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_rng_normal(fixture->rng, 0.0, NAN, &val);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_rng_gamma(fixture->rng, 1.0, NAN, &val);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void check_full_integer_range(lmmc_rng_t *rng1, lmmc_rng_t *rng2) {
    uint64_t raw = lmmc_rng_next_u64(rng1);
    uint64_t bits = (uint64_t)INT64_MIN + raw;
    int64_t actual;
    int64_t expected = bits <= (uint64_t)INT64_MAX
                           ? (int64_t)bits
                           : INT64_MIN + (int64_t)(bits - (UINT64_C(1) << 63));
    lmmc_status_t st =
        lmmc_rng_int_uniform(rng2, INT64_MIN, INT64_MAX, &actual);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_int_equal(actual, expected);
}

static void test_integer_sampling_extremes(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int64_t actual;
    size_t binomial = 23;

    assert_int_equal(lmmc_rng_create(&fixture->rng1), LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_create(&fixture->rng2), LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_rng_seed(fixture->rng1, UINT64_C(0x1122334455667788)),
        LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_rng_seed(fixture->rng2, UINT64_C(0x1122334455667788)),
        LMMC_STATUS_OK);

    check_full_integer_range(fixture->rng1, fixture->rng2);

    assert_int_equal(lmmc_rng_seed(fixture->rng1, 991), LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng2, 991), LMMC_STATUS_OK);
    st = lmmc_rng_int_uniform(fixture->rng1, -7, -7, &actual);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_int_equal(actual, -7);
    assert_int_equal(lmmc_rng_next_u64(fixture->rng1),
                     lmmc_rng_next_u64(fixture->rng2));

    assert_int_equal(lmmc_rng_seed(fixture->rng1, 992), LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng2, 992), LMMC_STATUS_OK);
    st = lmmc_rng_binomial(fixture->rng1, 10, NAN, &binomial);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(binomial, 23);
    assert_int_equal(lmmc_rng_next_u64(fixture->rng1),
                     lmmc_rng_next_u64(fixture->rng2));

    st = lmmc_rng_binomial(fixture->rng1, SIZE_MAX, DBL_MIN, &binomial);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_int_equal(binomial, 0);
}

static void test_shuffle_uniformity(void **state) {
    struct test_fixture *fixture = *state;

    size_t counts[3] = {0, 0, 0};
    size_t trial;

    assert_false(lmmc_rng_create(&fixture->rng) != LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_rng_seed(fixture->rng, UINT64_C(0x53485546464C45)),
        LMMC_STATUS_OK);
    for (trial = 0; trial < 60000; ++trial) {
        int values[3] = {0, 1, 2};
        assert_int_equal(
            lmmc_rng_shuffle(fixture->rng, values, 3, sizeof(values[0])),
            LMMC_STATUS_OK);
        ++counts[(size_t)values[0]];
    }
    for (trial = 0; trial < 3; ++trial) {
        assert_true(counts[trial] >= 19000 && counts[trial] <= 21000);
    }
}

static void test_seed_null_argument(void **state) {
    (void)state;
    lmmc_status_t st;

    st = lmmc_rng_seed(NULL, 42);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_poisson_output_range, setup, teardown),
        cmocka_unit_test_setup_teardown(test_seed_reproducibility, setup, teardown),
        cmocka_unit_test_setup_teardown(test_uniform_moments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_normal_moments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_exponential_moments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_shuffle_permutation, setup, teardown),
        cmocka_unit_test_setup_teardown(test_singleton_shuffle, setup, teardown),
        cmocka_unit_test_setup_teardown(test_uniform_bounds, setup, teardown),
        cmocka_unit_test_setup_teardown(test_exponential_domain, setup, teardown),
        cmocka_unit_test_setup_teardown(test_normal_gamma_domain, setup, teardown),
        cmocka_unit_test_setup_teardown(test_integer_sampling_extremes, setup, teardown),
        cmocka_unit_test_setup_teardown(test_shuffle_uniformity, setup, teardown),
        cmocka_unit_test_setup_teardown(test_seed_null_argument, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
