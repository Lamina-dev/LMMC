/**
 * @file test_random_dist.c
 * 针对 LMMC 中 random dist 相关接口的单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "lmmc/config.h"
#include "lmmc/random.h"

#define NUM_SAMPLES 1200

#define NUM_CONFIGS 50

struct test_fixture {
    lmmc_rng_t *rng;
    lmmc_real_t *array;
    lmmc_real_t *sorted_before;
    int *array_int;
    int *sorted_before_int;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    srand(0x52444953u);
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_rng_destroy(fixture->rng);
    free(fixture->array);
    free(fixture->sorted_before);
    free(fixture->array_int);
    free(fixture->sorted_before_int);
    free(fixture);
    *state = NULL;
    return 0;
}

static double rand_in_range(double lo, double hi) {
    double u = (double)rand() / (double)RAND_MAX;
    return lo + (hi - lo) * u;
}

static int cmp_real(const void *a, const void *b) {
    lmmc_real_t va = *(const lmmc_real_t *)a;
    lmmc_real_t vb = *(const lmmc_real_t *)b;
    if (va < vb) {
        return -1;
    }
    if (va > vb) {
        return 1;
    }
    return 0;
}

static int cmp_int(const void *a, const void *b) {
    int va = *(const int *)a;
    int vb = *(const int *)b;
    return va - vb;
}

static void test_uniform_range_single(void **state) {
    struct test_fixture *fixture = *state;
    int cfg;

    lmmc_status_t status;

    status = lmmc_rng_create(&fixture->rng);
    assert_true(status == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, (uint64_t)42), LMMC_STATUS_OK);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {
        double a = rand_in_range(-1000.0, 1000.0);
        double b = a + rand_in_range(0.001, 2000.0);
        int i;

        for (i = 0; i < NUM_SAMPLES; i++) {
            lmmc_real_t value;
            status = lmmc_rng_uniform(fixture->rng, (lmmc_real_t)a, (lmmc_real_t)b, &value);
            assert_true(status == LMMC_STATUS_OK);
            assert_true(value >= a);
            assert_true(value < b);
        }
    }
}

static void test_uniform_range_fill(void **state) {
    struct test_fixture *fixture = *state;
    int cfg;

    lmmc_status_t status;

    status = lmmc_rng_create(&fixture->rng);
    assert_true(status == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, (uint64_t)123), LMMC_STATUS_OK);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {
        double a = rand_in_range(-500.0, 500.0);
        double b = a + rand_in_range(0.01, 1000.0);
        lmmc_real_t array[NUM_SAMPLES];
        size_t i;

        status = lmmc_rng_fill_uniform(fixture->rng, (lmmc_real_t)a, (lmmc_real_t)b,
                                       array, NUM_SAMPLES);
        assert_true(status == LMMC_STATUS_OK);

        for (i = 0; i < NUM_SAMPLES; i++) {
            assert_true(array[i] >= a);
            assert_true(array[i] < b);
        }
    }
}

static void test_uniform_range_tiny_interval(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t status;
    int i;

    status = lmmc_rng_create(&fixture->rng);
    assert_true(status == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, (uint64_t)999), LMMC_STATUS_OK);

    double a = 1.0;
    double b = 1.0 + 1e-10;

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t value;
        status = lmmc_rng_uniform(fixture->rng, (lmmc_real_t)a, (lmmc_real_t)b, &value);
        assert_true(status == LMMC_STATUS_OK);
        assert_true(value >= a);
        assert_true(value < b);
    }
}

static void test_uniform_range_large_interval(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t status;
    int i;

    status = lmmc_rng_create(&fixture->rng);
    assert_true(status == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, (uint64_t)7777), LMMC_STATUS_OK);

    double a = -1e15;
    double b = 1e15;

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t value;
        status = lmmc_rng_uniform(fixture->rng, (lmmc_real_t)a, (lmmc_real_t)b, &value);
        assert_true(status == LMMC_STATUS_OK);
        assert_true(value >= a);
        assert_true(value < b);
    }
}

static void test_exponential_nonneg(void **state) {
    struct test_fixture *fixture = *state;
    int cfg;

    lmmc_status_t status;

    status = lmmc_rng_create(&fixture->rng);
    assert_true(status == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, (uint64_t)314159), LMMC_STATUS_OK);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {
        double rate = rand_in_range(0.001, 1000.0);
        int i;

        for (i = 0; i < NUM_SAMPLES; i++) {
            lmmc_real_t value;
            status = lmmc_rng_exponential(fixture->rng, (lmmc_real_t)rate, &value);
            assert_true(status == LMMC_STATUS_OK);
            assert_true(value >= 0.0);
        }
    }
}

static void test_exponential_nonneg_small_rate(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t status;
    int i;

    status = lmmc_rng_create(&fixture->rng);
    assert_true(status == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, (uint64_t)271828), LMMC_STATUS_OK);

    double rate = 1e-10;

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t value;
        status = lmmc_rng_exponential(fixture->rng, (lmmc_real_t)rate, &value);
        assert_true(status == LMMC_STATUS_OK);
        assert_true(value >= 0.0);
    }
}

static void test_exponential_nonneg_large_rate(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t status;
    int i;

    status = lmmc_rng_create(&fixture->rng);
    assert_true(status == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, (uint64_t)161803), LMMC_STATUS_OK);

    double rate = 1e10;

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t value;
        status = lmmc_rng_exponential(fixture->rng, (lmmc_real_t)rate, &value);
        assert_true(status == LMMC_STATUS_OK);
        assert_true(value >= 0.0);
    }
}

static void test_shuffle_preserves_elements_real(void **state) {
    struct test_fixture *fixture = *state;
    int cfg;

    lmmc_status_t status;

    status = lmmc_rng_create(&fixture->rng);
    assert_true(status == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, (uint64_t)55555), LMMC_STATUS_OK);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {
        size_t n = 2 + (size_t)(rand() % 199);
        fixture->array = (lmmc_real_t *)malloc(n * sizeof(lmmc_real_t));
        fixture->sorted_before = (lmmc_real_t *)malloc(n * sizeof(lmmc_real_t));
        size_t i;

        assert_true(fixture->array != NULL && fixture->sorted_before != NULL);

        for (i = 0; i < n; i++) {
            fixture->array[i] = rand_in_range(-1000.0, 1000.0);
        }

        memcpy(fixture->sorted_before, fixture->array, n * sizeof(lmmc_real_t));
        qsort(fixture->sorted_before, n, sizeof(lmmc_real_t), cmp_real);

        status = lmmc_rng_shuffle(fixture->rng, fixture->array, n, sizeof(lmmc_real_t));
        assert_true(status == LMMC_STATUS_OK);

        qsort(fixture->array, n, sizeof(lmmc_real_t), cmp_real);

        for (i = 0; i < n; i++) {
            assert_true(fixture->array[i] == fixture->sorted_before[i]);
        }

        free(fixture->array);
        fixture->array = NULL;
        free(fixture->sorted_before);
        fixture->sorted_before = NULL;
    }
}

static void test_shuffle_preserves_elements_int(void **state) {
    struct test_fixture *fixture = *state;
    int cfg;

    lmmc_status_t status;

    status = lmmc_rng_create(&fixture->rng);
    assert_true(status == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, (uint64_t)88888), LMMC_STATUS_OK);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {
        size_t n = 2 + (size_t)(rand() % 199);
        fixture->array_int = (int *)malloc(n * sizeof(int));
        fixture->sorted_before_int = (int *)malloc(n * sizeof(int));
        size_t i;

        assert_true(fixture->array_int != NULL && fixture->sorted_before_int != NULL);

        for (i = 0; i < n; i++) {
            fixture->array_int[i] = rand() % 1000 - 500;
        }

        memcpy(fixture->sorted_before_int, fixture->array_int, n * sizeof(int));
        qsort(fixture->sorted_before_int, n, sizeof(int), cmp_int);

        status = lmmc_rng_shuffle(fixture->rng, fixture->array_int, n, sizeof(int));
        assert_true(status == LMMC_STATUS_OK);

        qsort(fixture->array_int, n, sizeof(int), cmp_int);

        for (i = 0; i < n; i++) {
            assert_true(fixture->array_int[i] == fixture->sorted_before_int[i]);
        }

        free(fixture->array_int);
        fixture->array_int = NULL;
        free(fixture->sorted_before_int);
        fixture->sorted_before_int = NULL;
    }
}

static void test_shuffle_single_element(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t status;

    status = lmmc_rng_create(&fixture->rng);
    assert_true(status == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, (uint64_t)11111), LMMC_STATUS_OK);

    lmmc_real_t val = 42.0;
    status = lmmc_rng_shuffle(fixture->rng, &val, 1, sizeof(lmmc_real_t));
    assert_true(status == LMMC_STATUS_OK);
    assert_true(val == 42.0);
}

static void test_shuffle_with_duplicates(void **state) {
    struct test_fixture *fixture = *state;
    int cfg;

    lmmc_status_t status;

    status = lmmc_rng_create(&fixture->rng);
    assert_true(status == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, (uint64_t)22222), LMMC_STATUS_OK);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {
        size_t n = 50 + (size_t)(rand() % 151);
        fixture->array_int = (int *)malloc(n * sizeof(int));
        fixture->sorted_before_int = (int *)malloc(n * sizeof(int));
        size_t i;

        assert_true(fixture->array_int != NULL && fixture->sorted_before_int != NULL);

        for (i = 0; i < n; i++) {
            fixture->array_int[i] = rand() % 10;
        }

        memcpy(fixture->sorted_before_int, fixture->array_int, n * sizeof(int));
        qsort(fixture->sorted_before_int, n, sizeof(int), cmp_int);

        status = lmmc_rng_shuffle(fixture->rng, fixture->array_int, n, sizeof(int));
        assert_true(status == LMMC_STATUS_OK);

        qsort(fixture->array_int, n, sizeof(int), cmp_int);

        for (i = 0; i < n; i++) {
            assert_true(fixture->array_int[i] == fixture->sorted_before_int[i]);
        }

        free(fixture->array_int);
        fixture->array_int = NULL;
        free(fixture->sorted_before_int);
        fixture->sorted_before_int = NULL;
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_uniform_range_single, setup, teardown),
        cmocka_unit_test_setup_teardown(test_uniform_range_fill, setup, teardown),
        cmocka_unit_test_setup_teardown(test_uniform_range_tiny_interval, setup, teardown),
        cmocka_unit_test_setup_teardown(test_uniform_range_large_interval, setup, teardown),
        cmocka_unit_test_setup_teardown(test_exponential_nonneg, setup, teardown),
        cmocka_unit_test_setup_teardown(test_exponential_nonneg_small_rate, setup, teardown),
        cmocka_unit_test_setup_teardown(test_exponential_nonneg_large_rate, setup, teardown),
        cmocka_unit_test_setup_teardown(test_shuffle_preserves_elements_real, setup, teardown),
        cmocka_unit_test_setup_teardown(test_shuffle_preserves_elements_int, setup, teardown),
        cmocka_unit_test_setup_teardown(test_shuffle_single_element, setup, teardown),
        cmocka_unit_test_setup_teardown(test_shuffle_with_duplicates, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
