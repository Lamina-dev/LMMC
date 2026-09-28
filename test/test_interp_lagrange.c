/**
 * @file test_interp_lagrange.c
 * 针对 LMMC 中 interp lagrange 相关接口的单元测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "lmmc/config.h"
#include "lmmc/status.h"
#include "lmmc/interp.h"
#include "test_common.h"

typedef struct {
    lmmc_interp_lagrange_t *interp_lagrange_lag;
} test_fixture_t;

static int setup(void **state) {
    test_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag);
    free(fixture);
    return 0;
}

static void test_lagrange_null_args(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0};

    lmmc_status_t st;

    st = lmmc_interp_lagrange_create(NULL, ys, 3, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_lagrange_create(xs, NULL, 3, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_lagrange_create(xs, ys, 3, NULL);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_lagrange_too_few_points(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {1.0};
    lmmc_real_t ys[] = {2.0};

    lmmc_status_t st;

    st = lmmc_interp_lagrange_create(xs, ys, 0, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_lagrange_single_point(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {3.0};
    lmmc_real_t ys[] = {7.0};

    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-12;

    st = lmmc_interp_lagrange_create(xs, ys, 1, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag, 3.0, &result);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(result, 7.0, eps));

    st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag, 5.0, &result);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(result, 7.0, eps));

    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag);
    fixture->interp_lagrange_lag = NULL;
}

static void test_lagrange_passes_through_data_points(void **state) {
    test_fixture_t *fixture = *state;

    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};

    lmmc_status_t st;
    lmmc_real_t result;
    size_t i;
    double eps = 1e-12;

    st = lmmc_interp_lagrange_create(xs, ys, 5, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < 5; i++) {
        st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag, xs[i], &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, ys[i], eps));
    }

    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag);
    fixture->interp_lagrange_lag = NULL;
}

static void test_lagrange_quadratic_exact(void **state) {
    test_fixture_t *fixture = *state;

    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {1.0, 0.0, 3.0};

    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-10;
    double x;

    st = lmmc_interp_lagrange_create(xs, ys, 3, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_OK);

    for (x = -1.0; x <= 3.0; x += 0.5) {
        lmmc_real_t expected = 2.0 * x * x - 3.0 * x + 1.0;
        st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag, (lmmc_real_t)x, &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, expected, eps));
    }

    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag);
    fixture->interp_lagrange_lag = NULL;
}

static void test_lagrange_cubic_exact(void **state) {
    test_fixture_t *fixture = *state;

    lmmc_real_t xs[] = {-1.0, 0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {2.0, 1.0, 0.0, 5.0};

    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-10;
    double x;

    st = lmmc_interp_lagrange_create(xs, ys, 4, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_OK);

    for (x = -1.0; x <= 2.0; x += 0.25) {
        lmmc_real_t expected = x * x * x - 2.0 * x + 1.0;
        st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag, (lmmc_real_t)x, &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, expected, eps));
    }

    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag);
    fixture->interp_lagrange_lag = NULL;
}

static void test_lagrange_eval_null(void **state) {
    (void)state;
    lmmc_real_t result;
    lmmc_status_t st;

    st = lmmc_interp_lagrange_eval(NULL, 1.0, &result);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_lagrange_eval_null_out(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0};

    lmmc_status_t st;

    st = lmmc_interp_lagrange_create(xs, ys, 3, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag, 0.5, NULL);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag);
    fixture->interp_lagrange_lag = NULL;
}

static void test_lagrange_destroy_null(void **state) {
    (void)state;

    lmmc_interp_lagrange_destroy(NULL);
}

static void test_lagrange_non_uniform_spacing(void **state) {
    test_fixture_t *fixture = *state;

    lmmc_real_t xs[] = {0.0, 0.5, 1.5, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 0.25, 2.25, 9.0, 16.0};

    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-9;

    st = lmmc_interp_lagrange_create(xs, ys, 5, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag, 2.0, &result);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(result, 4.0, eps));

    st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag, 1.0, &result);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(result, 1.0, eps));

    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag);
    fixture->interp_lagrange_lag = NULL;
}

static void test_lagrange_weight_range(void **state) {
    test_fixture_t *fixture = *state;
    const lmmc_real_t repeated[] = {0.0, DBL_MAX, -DBL_MAX, 1.0, 1.0};
    const lmmc_real_t unrepresentable[] = {-DBL_MAX, -1.0, 0.0, 1.0, DBL_MAX};
    const lmmc_real_t tiny[] = {0.0, DBL_MIN, 2.0 * DBL_MIN};
    const lmmc_real_t ys[5] = {0};
    lmmc_real_t result = 42.0;

    assert_int_equal(lmmc_interp_lagrange_create(
        repeated, ys, 5, &fixture->interp_lagrange_lag), LMMC_STATUS_INVALID_ARGUMENT);
    assert_null(fixture->interp_lagrange_lag);
    assert_int_equal(lmmc_interp_lagrange_create(
        unrepresentable, ys, 5, &fixture->interp_lagrange_lag), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_null(fixture->interp_lagrange_lag);
    assert_int_equal(lmmc_interp_lagrange_create(
        tiny, ys, 3, &fixture->interp_lagrange_lag), LMMC_STATUS_OK);
    assert_int_equal(lmmc_interp_lagrange_eval(
        fixture->interp_lagrange_lag, DBL_MIN / 2.0, &result), LMMC_STATUS_OK);
    assert_true(result == 0.0);
}

static void test_lagrange_large_constant(void **state) {
    test_fixture_t *fixture = *state;
    const lmmc_real_t xs[] = {0.0, 1.0};
    const lmmc_real_t ys[] = {DBL_MAX, DBL_MAX};
    lmmc_real_t result = 42.0;
    assert_int_equal(lmmc_interp_lagrange_create(
        xs, ys, 2, &fixture->interp_lagrange_lag), LMMC_STATUS_OK);
    assert_int_equal(lmmc_interp_lagrange_eval(
        fixture->interp_lagrange_lag, 0.5, &result), LMMC_STATUS_OK);
    assert_true(result == DBL_MAX);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_lagrange_null_args, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lagrange_too_few_points, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lagrange_single_point, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lagrange_passes_through_data_points, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lagrange_quadratic_exact, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lagrange_cubic_exact, setup, teardown),
        cmocka_unit_test(test_lagrange_eval_null),
        cmocka_unit_test_setup_teardown(test_lagrange_eval_null_out, setup, teardown),
        cmocka_unit_test(test_lagrange_destroy_null),
        cmocka_unit_test_setup_teardown(test_lagrange_non_uniform_spacing, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lagrange_weight_range, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lagrange_large_constant, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
