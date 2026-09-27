/**
 * @file test_dense_vec.c
 * 针对 LMMC 中 dense vec 相关接口的单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"

#define NUM_ITERATIONS 150

#define TOL 1e-12

struct test_fixture {
    lmmc_vec_t x;
    lmmc_vec_t y;
    lmmc_vec_t scratch;
    double *orig_x;
    double *orig_y;
};

static double rand_double(double range) {
    return ((double)rand() / (double)RAND_MAX) * 2.0 * range - range;
}

static size_t rand_vec_size(size_t min_size, size_t max_size) {
    return min_size + (size_t)(rand() % (int)(max_size - min_size + 1));
}

static void fill_random_vec(lmmc_vec_t *v, double range) {
    for (size_t i = 0; i < v->size; ++i) {
        v->data[i] = rand_double(range);
    }
}

static void fill_zero_vec(lmmc_vec_t *v) {
    for (size_t i = 0; i < v->size; ++i) {
        v->data[i] = 0.0;
    }
}

static void test_axpy_correctness(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_vec_size(1, 64);

        assert_int_equal(lmmc_vec_create(n, &fixture->x), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(n, &fixture->y), LMMC_STATUS_OK);

        double range = (iter < 10) ? 1e15 : 100.0;
        fill_random_vec(&fixture->x, range);
        fill_random_vec(&fixture->y, range);

        assert_int_equal(lmmc_vec_create(n, &fixture->scratch), LMMC_STATUS_OK);
        for (size_t i = 0; i < n; ++i) {
            fixture->scratch.data[i] = fixture->y.data[i];
        }

        lmmc_real_t alpha = rand_double(10.0);

        lmmc_status_t status = lmmc_vec_axpy(alpha, &fixture->x, &fixture->y);
        assert_int_equal(status, LMMC_STATUS_OK);

        for (size_t i = 0; i < n; ++i) {
            double expected = alpha * fixture->x.data[i] + fixture->scratch.data[i];
            double diff = fabs(fixture->y.data[i] - expected);
            double scale = fabs(expected) + 1.0;
            assert_true(isfinite(diff) && isfinite(scale) && diff <= TOL * scale);
        }

        lmmc_vec_destroy(&fixture->x);
        lmmc_vec_destroy(&fixture->y);
        lmmc_vec_destroy(&fixture->scratch);
    }
}

static void test_axpy_single_element(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_vec_create(1, &fixture->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(1, &fixture->y), LMMC_STATUS_OK);

    fixture->x.data[0] = 3.0;
    fixture->y.data[0] = 7.0;
    lmmc_real_t alpha = 2.0;

    lmmc_status_t status = lmmc_vec_axpy(alpha, &fixture->x, &fixture->y);
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(fabs(fixture->y.data[0] - 13.0) < TOL);
}

static void test_axpy_alpha_zero(void **state) {
    struct test_fixture *fixture = *state;
    size_t n = 10;

    assert_int_equal(lmmc_vec_create(n, &fixture->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &fixture->y), LMMC_STATUS_OK);
    fill_random_vec(&fixture->x, 100.0);
    fill_random_vec(&fixture->y, 100.0);

    double old_y[10];
    for (size_t i = 0; i < n; ++i)
        old_y[i] = fixture->y.data[i];

    lmmc_real_t alpha = 0.0;
    lmmc_status_t status = lmmc_vec_axpy(alpha, &fixture->x, &fixture->y);
    assert_int_equal(status, LMMC_STATUS_OK);

    for (size_t i = 0; i < n; ++i) {
        assert_true(fixture->y.data[i] == old_y[i]);
    }
}

static void test_norm_properties(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_vec_size(1, 64);

        assert_int_equal(lmmc_vec_create(n, &fixture->x), LMMC_STATUS_OK);

        double range = (iter < 10) ? 1e10 : 50.0;
        fill_random_vec(&fixture->x, range);

        lmmc_real_t norm2_val, norm_inf_val;
        lmmc_status_t s1 = lmmc_vec_norm2(&fixture->x, &norm2_val);
        lmmc_status_t s2 = lmmc_vec_norm_inf(&fixture->x, &norm_inf_val);
        assert_int_equal(s1, LMMC_STATUS_OK);
        assert_int_equal(s2, LMMC_STATUS_OK);

        assert_true(isfinite(norm2_val) && norm2_val >= 0.0);
        assert_true(isfinite(norm_inf_val) && norm_inf_val >= 0.0);

        int is_nonzero = 0;
        for (size_t i = 0; i < n; ++i) {
            if (fixture->x.data[i] != 0.0) {
                is_nonzero = 1;
                break;
            }
        }

        if (is_nonzero) {
            assert_true(norm2_val > 0.0);
            assert_true(norm_inf_val > 0.0);
        }

        double sqrt_n = sqrt((double)n);
        assert_true(norm_inf_val <= norm2_val + TOL * norm2_val);
        assert_true(norm2_val <= sqrt_n * norm_inf_val + TOL * norm2_val);

        lmmc_vec_destroy(&fixture->x);
    }
}

static void test_norm_zero_vector(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < 10; iter++) {
        size_t n = rand_vec_size(1, 32);

        assert_int_equal(lmmc_vec_create(n, &fixture->x), LMMC_STATUS_OK);
        fill_zero_vec(&fixture->x);

        lmmc_real_t norm2_val, norm_inf_val;
        assert_int_equal(lmmc_vec_norm2(&fixture->x, &norm2_val),
                         LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_norm_inf(&fixture->x, &norm_inf_val),
                         LMMC_STATUS_OK);

        assert_true(norm2_val == 0.0);
        assert_true(norm_inf_val == 0.0);

        lmmc_vec_destroy(&fixture->x);
    }
}

static void test_norm_single_element(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_vec_create(1, &fixture->x), LMMC_STATUS_OK);
    fixture->x.data[0] = -5.0;

    lmmc_real_t norm2_val, norm_inf_val;
    assert_int_equal(lmmc_vec_norm2(&fixture->x, &norm2_val),
                     LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_norm_inf(&fixture->x, &norm_inf_val),
                     LMMC_STATUS_OK);

    assert_true(fabs(norm2_val - 5.0) < TOL);
    assert_true(fabs(norm_inf_val - 5.0) < TOL);
}

static void test_copy_correctness(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_vec_size(1, 64);

        assert_int_equal(lmmc_vec_create(n, &fixture->x), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(n, &fixture->scratch), LMMC_STATUS_OK);

        double range = (iter < 10) ? 1e15 : 100.0;
        fill_random_vec(&fixture->x, range);
        fill_random_vec(&fixture->scratch, range);

        lmmc_status_t status = lmmc_vec_copy(&fixture->x, &fixture->scratch);
        assert_int_equal(status, LMMC_STATUS_OK);

        for (size_t i = 0; i < n; ++i) {
            assert_true(fixture->scratch.data[i] == fixture->x.data[i]);
        }

        lmmc_vec_destroy(&fixture->x);
        lmmc_vec_destroy(&fixture->scratch);
    }
}

static void test_swap_correctness(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_vec_size(1, 64);

        assert_int_equal(lmmc_vec_create(n, &fixture->x), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(n, &fixture->y), LMMC_STATUS_OK);

        double range = (iter < 10) ? 1e15 : 100.0;
        fill_random_vec(&fixture->x, range);
        fill_random_vec(&fixture->y, range);

        fixture->orig_x = (double *)malloc(n * sizeof(double));
        assert_non_null(fixture->orig_x);
        fixture->orig_y = (double *)malloc(n * sizeof(double));
        assert_non_null(fixture->orig_y);
        for (size_t i = 0; i < n; ++i) {
            fixture->orig_x[i] = fixture->x.data[i];
            fixture->orig_y[i] = fixture->y.data[i];
        }

        lmmc_status_t status = lmmc_vec_swap(&fixture->x, &fixture->y);
        assert_int_equal(status, LMMC_STATUS_OK);

        for (size_t i = 0; i < n; ++i) {
            assert_true(fixture->x.data[i] == fixture->orig_y[i]);
            assert_true(fixture->y.data[i] == fixture->orig_x[i]);
        }

        free(fixture->orig_x);
        fixture->orig_x = NULL;
        free(fixture->orig_y);
        fixture->orig_y = NULL;
        lmmc_vec_destroy(&fixture->x);
        lmmc_vec_destroy(&fixture->y);
    }
}

static void test_copy_swap_single_element(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_vec_create(1, &fixture->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(1, &fixture->y), LMMC_STATUS_OK);

    fixture->x.data[0] = 42.0;
    fixture->y.data[0] = -99.0;

    assert_int_equal(lmmc_vec_create(1, &fixture->scratch), LMMC_STATUS_OK);
    fixture->scratch.data[0] = 0.0;
    assert_int_equal(lmmc_vec_copy(&fixture->x, &fixture->scratch),
                     LMMC_STATUS_OK);
    assert_true(fixture->scratch.data[0] == 42.0);

    assert_int_equal(lmmc_vec_swap(&fixture->x, &fixture->y),
                     LMMC_STATUS_OK);
    assert_true(fixture->x.data[0] == -99.0);
    assert_true(fixture->y.data[0] == 42.0);
}

static void test_swap_double_swap(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < 50; iter++) {
        size_t n = rand_vec_size(2, 32);

        assert_int_equal(lmmc_vec_create(n, &fixture->x), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(n, &fixture->y), LMMC_STATUS_OK);
        fill_random_vec(&fixture->x, 100.0);
        fill_random_vec(&fixture->y, 100.0);

        fixture->orig_x = (double *)malloc(n * sizeof(double));
        assert_non_null(fixture->orig_x);
        fixture->orig_y = (double *)malloc(n * sizeof(double));
        assert_non_null(fixture->orig_y);
        for (size_t i = 0; i < n; ++i) {
            fixture->orig_x[i] = fixture->x.data[i];
            fixture->orig_y[i] = fixture->y.data[i];
        }

        assert_int_equal(lmmc_vec_swap(&fixture->x, &fixture->y),
                         LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_swap(&fixture->x, &fixture->y),
                         LMMC_STATUS_OK);

        for (size_t i = 0; i < n; ++i) {
            assert_true(fixture->x.data[i] == fixture->orig_x[i]);
            assert_true(fixture->y.data[i] == fixture->orig_y[i]);
        }

        free(fixture->orig_x);
        fixture->orig_x = NULL;
        free(fixture->orig_y);
        fixture->orig_y = NULL;
        lmmc_vec_destroy(&fixture->x);
        lmmc_vec_destroy(&fixture->y);
    }
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    assert_non_null(fixture);
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->y);
    lmmc_vec_destroy(&fixture->scratch);
    free(fixture->orig_x);
    free(fixture->orig_y);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const unsigned int seed = 0x44564543u;
    srand(seed);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_axpy_correctness, setup, teardown),
        cmocka_unit_test_setup_teardown(test_axpy_single_element, setup, teardown),
        cmocka_unit_test_setup_teardown(test_axpy_alpha_zero, setup, teardown),
        cmocka_unit_test_setup_teardown(test_norm_properties, setup, teardown),
        cmocka_unit_test_setup_teardown(test_norm_zero_vector, setup, teardown),
        cmocka_unit_test_setup_teardown(test_norm_single_element, setup, teardown),
        cmocka_unit_test_setup_teardown(test_copy_correctness, setup, teardown),
        cmocka_unit_test_setup_teardown(test_swap_correctness, setup, teardown),
        cmocka_unit_test_setup_teardown(test_copy_swap_single_element, setup, teardown),
        cmocka_unit_test_setup_teardown(test_swap_double_swap, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
