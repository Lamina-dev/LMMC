/**
 * @file test_interp_cspline.c
 * 针对 LMMC 中 interp cspline 相关接口的单元测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/config.h"
#include "lmmc/status.h"
#include "lmmc/interp.h"
#include "test_common.h"

typedef struct {
    lmmc_interp_cspline_t *interp_cspline_spline;
} test_fixture_t;

static int setup(void **state) {
    test_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    free(fixture);
    return 0;
}

static void test_cspline_null_args(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0};

    lmmc_status_t st;

    st = lmmc_interp_cspline_create(NULL, ys, 3, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_cspline_create(xs, NULL, 3, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_cspline_create(xs, ys, 3, NULL);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_cspline_too_few_points(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0, 1.0};
    lmmc_real_t ys[] = {0.0, 1.0};

    lmmc_status_t st;

    st = lmmc_interp_cspline_create(xs, ys, 2, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_cspline_create(xs, ys, 0, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_cspline_non_increasing_xs(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs_equal[] = {0.0, 1.0, 1.0};
    lmmc_real_t xs_decreasing[] = {0.0, 2.0, 1.0};
    lmmc_real_t ys[] = {0.0, 1.0, 2.0};

    lmmc_status_t st;

    st = lmmc_interp_cspline_create(xs_equal, ys, 3, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_cspline_create(xs_decreasing, ys, 3, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_cspline_passes_through_data_points(void **state) {
    test_fixture_t *fixture = *state;

    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};

    lmmc_status_t st;
    lmmc_real_t result;
    size_t i;
    double eps = 1e-12;

    st = lmmc_interp_cspline_create(xs, ys, 5, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < 5; i++) {
        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, xs[i], &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, ys[i], eps));
    }

    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

static void test_cspline_out_of_range(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {1.0, 4.0, 9.0, 16.0};

    lmmc_status_t st;
    lmmc_real_t result;

    st = lmmc_interp_cspline_create(xs, ys, 4, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, 0.5, &result);
    assert_true(st == LMMC_STATUS_OUT_OF_RANGE);

    st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, 4.5, &result);
    assert_true(st == LMMC_STATUS_OUT_OF_RANGE);

    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

static void test_cspline_eval_null(void **state) {
    (void)state;
    lmmc_real_t result;
    lmmc_status_t st;

    st = lmmc_interp_cspline_eval(NULL, 1.0, &result);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_cspline_derivative_continuity(void **state) {
    test_fixture_t *fixture = *state;

    lmmc_real_t xs[6], ys[6];

    lmmc_status_t st;
    size_t i;
    double h = 1e-5;
    double tol_d1 = 1e-4;
    double tol_d2 = 1e-2;

    for (i = 0; i < 6; i++) {
        xs[i] = (lmmc_real_t)i;
        ys[i] = sin((double)i);
    }

    st = lmmc_interp_cspline_create(xs, ys, 6, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 1; i < 5; i++) {
        lmmc_real_t x_knot = xs[i];
        lmmc_real_t y_left, y_right, y_center;
        lmmc_real_t deriv_left, deriv_right;
        lmmc_real_t deriv2_left, deriv2_right;
        lmmc_real_t y_ll, y_rr;

        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, x_knot - h, &y_left);
        assert_true(st == LMMC_STATUS_OK);
        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, x_knot + h, &y_right);
        assert_true(st == LMMC_STATUS_OK);
        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, x_knot, &y_center);
        assert_true(st == LMMC_STATUS_OK);

        deriv_left = (y_center - y_left) / h;
        deriv_right = (y_right - y_center) / h;
        assert_true(fabs(deriv_left - deriv_right) < tol_d1);

        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, x_knot - 2 * h, &y_ll);
        assert_true(st == LMMC_STATUS_OK);
        deriv2_left = (y_center - 2.0 * y_left + y_ll) / (h * h);

        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, x_knot + 2 * h, &y_rr);
        assert_true(st == LMMC_STATUS_OK);
        deriv2_right = (y_rr - 2.0 * y_right + y_center) / (h * h);

        assert_true(fabs(deriv2_left - deriv2_right) < tol_d2);
    }

    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

static void test_cspline_linear_function(void **state) {
    test_fixture_t *fixture = *state;

    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {1.0, 3.0, 5.0, 7.0, 9.0};

    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-12;
    double x;

    st = lmmc_interp_cspline_create(xs, ys, 5, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    for (x = 0.0; x <= 4.0; x += 0.25) {
        lmmc_real_t expected = 2.0 * x + 1.0;
        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, (lmmc_real_t)x, &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, expected, eps));
    }

    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

static void test_cspline_destroy_null(void **state) {
    (void)state;

    lmmc_interp_cspline_destroy(NULL);
}

static void test_cspline_three_points(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {0.0, 1.0, 0.0};

    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-12;

    st = lmmc_interp_cspline_create(xs, ys, 3, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, 0.0, &result);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(result, 0.0, eps));

    st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, 1.0, &result);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(result, 1.0, eps));

    st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, 2.0, &result);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(result, 0.0, eps));

    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_cspline_null_args, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cspline_too_few_points, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cspline_non_increasing_xs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cspline_passes_through_data_points, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cspline_out_of_range, setup, teardown),
        cmocka_unit_test(test_cspline_eval_null),
        cmocka_unit_test_setup_teardown(test_cspline_derivative_continuity, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cspline_linear_function, setup, teardown),
        cmocka_unit_test(test_cspline_destroy_null),
        cmocka_unit_test_setup_teardown(test_cspline_three_points, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
