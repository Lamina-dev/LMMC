/**
 * @file test_interp_extended.c
 * 针对 LMMC 中 interp extended 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define TEST_EPS_TIGHT 1e-12
#define TEST_EPS_NORMAL 1e-10
#define TEST_EPS_LOOSE 1e-4

typedef struct {
    lmmc_interp_cspline_t *interp_cspline_spline;
    lmmc_interp_lagrange_t *interp_lagrange_lag_equi;
    lmmc_interp_lagrange_t *interp_lagrange_lag_cheb;
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
    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag_cheb);
    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag_equi);
    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    free(fixture);
    return 0;
}

static void test_cspline_sin_20_nodes(void **state) {
    test_fixture_t *fixture = *state;
    const size_t n = 20;
    lmmc_real_t xs[20], ys[20];

    lmmc_status_t st;
    size_t i;

    for (i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i * 2.0 * LMMC_CONST_PI / (lmmc_real_t)(n - 1);
        ys[i] = sin(xs[i]);
    }

    st = lmmc_interp_cspline_create(xs, ys, n, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < 50; i++) {
        lmmc_real_t query_x = (lmmc_real_t)(i + 1) * 2.0 * LMMC_CONST_PI / 51.0;
        lmmc_real_t result;
        lmmc_real_t expected = sin(query_x);

        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, query_x, &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, expected, TEST_EPS_LOOSE));
    }

    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

static void test_cspline_linear_exact(void **state) {
    test_fixture_t *fixture = *state;
    const size_t n = 10;
    lmmc_real_t xs[10], ys[10];

    lmmc_status_t st;
    size_t i;

    const lmmc_real_t a = 3.5, b = -2.7;

    for (i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i;
        ys[i] = a * xs[i] + b;
    }

    st = lmmc_interp_cspline_create(xs, ys, n, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < 50; i++) {
        lmmc_real_t query_x = (lmmc_real_t)i * 9.0 / 50.0;
        lmmc_real_t result;
        lmmc_real_t expected = a * query_x + b;

        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, query_x, &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, expected, TEST_EPS_TIGHT));
    }

    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

static void test_lagrange_runge_chebyshev_vs_equidistant(void **state) {
    test_fixture_t *fixture = *state;
    const size_t n = 15;
    lmmc_real_t xs_equi[15], ys_equi[15];
    lmmc_real_t xs_cheb[15], ys_cheb[15];

    lmmc_status_t st;
    size_t i;
    double max_err_equi = 0.0, max_err_cheb = 0.0;

    for (i = 0; i < n; i++) {
        xs_equi[i] = -1.0 + 2.0 * (lmmc_real_t)i / (lmmc_real_t)(n - 1);
        ys_equi[i] = 1.0 / (1.0 + 25.0 * xs_equi[i] * xs_equi[i]);
    }

    for (i = 0; i < n; i++) {
        xs_cheb[i] = cos((2.0 * (lmmc_real_t)i + 1.0) * LMMC_CONST_PI / (2.0 * (lmmc_real_t)n));
        ys_cheb[i] = 1.0 / (1.0 + 25.0 * xs_cheb[i] * xs_cheb[i]);
    }

    st = lmmc_interp_lagrange_create(xs_equi, ys_equi, n, &fixture->interp_lagrange_lag_equi);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_interp_lagrange_create(xs_cheb, ys_cheb, n, &fixture->interp_lagrange_lag_cheb);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < 100; i++) {
        lmmc_real_t x = -0.95 + 1.9 * (lmmc_real_t)i / 99.0;
        lmmc_real_t exact = 1.0 / (1.0 + 25.0 * x * x);
        lmmc_real_t result_equi, result_cheb;
        double err_equi, err_cheb;

        st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag_equi, x, &result_equi);
        assert_true(st == LMMC_STATUS_OK);

        st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag_cheb, x, &result_cheb);
        assert_true(st == LMMC_STATUS_OK);

        err_equi = fabs(result_equi - exact);
        err_cheb = fabs(result_cheb - exact);

        if (err_equi > max_err_equi)
            max_err_equi = err_equi;
        if (err_cheb > max_err_cheb)
            max_err_cheb = err_cheb;
    }

    assert_true(max_err_cheb < max_err_equi);

    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag_equi);
    fixture->interp_lagrange_lag_equi = NULL;
    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag_cheb);
    fixture->interp_lagrange_lag_cheb = NULL;
}

static void test_lagrange_3node_quadratic(void **state) {
    test_fixture_t *fixture = *state;

    lmmc_real_t xs[] = {-1.0, 0.0, 2.0};
    lmmc_real_t ys[3];

    lmmc_status_t st;
    size_t i;

    for (i = 0; i < 3; i++) {
        ys[i] = 2.0 * xs[i] * xs[i] - 3.0 * xs[i] + 1.0;
    }

    st = lmmc_interp_lagrange_create(xs, ys, 3, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i <= 20; i++) {
        lmmc_real_t x = -1.0 + 3.0 * (lmmc_real_t)i / 20.0;
        lmmc_real_t expected = 2.0 * x * x - 3.0 * x + 1.0;
        lmmc_real_t result;

        st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag, x, &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, expected, TEST_EPS_NORMAL));
    }

    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag);
    fixture->interp_lagrange_lag = NULL;
}

static void test_cspline_exact_at_nodes(void **state) {
    test_fixture_t *fixture = *state;
    const size_t n = 10;
    lmmc_real_t xs[10], ys[10];

    lmmc_status_t st;
    size_t i;

    for (i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i * 0.5;
        ys[i] = sin(xs[i]) + 0.1 * cos(3.0 * xs[i]);
    }

    st = lmmc_interp_cspline_create(xs, ys, n, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < n; i++) {
        lmmc_real_t result;
        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, xs[i], &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, ys[i], TEST_EPS_TIGHT));
    }

    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

static void test_lagrange_exact_at_nodes(void **state) {
    test_fixture_t *fixture = *state;
    const size_t n = 8;
    lmmc_real_t xs[8], ys[8];

    lmmc_status_t st;
    size_t i;

    for (i = 0; i < n; i++) {
        xs[i] = -2.0 + (lmmc_real_t)i * 0.7;
        ys[i] = exp(-xs[i] * xs[i]);
    }

    st = lmmc_interp_lagrange_create(xs, ys, n, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < n; i++) {
        lmmc_real_t result;
        st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_lag, xs[i], &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, ys[i], TEST_EPS_TIGHT));
    }

    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_lag);
    fixture->interp_lagrange_lag = NULL;
}

static void test_insufficient_nodes_cspline(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0, 1.0};
    lmmc_real_t ys[] = {0.0, 1.0};

    lmmc_status_t st;

    st = lmmc_interp_cspline_create(xs, ys, 2, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_cspline_create(xs, ys, 1, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_cspline_create(xs, ys, 0, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_insufficient_nodes_lagrange(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0};
    lmmc_real_t ys[] = {0.0};

    lmmc_status_t st;

    st = lmmc_interp_lagrange_create(xs, ys, 0, &fixture->interp_lagrange_lag);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_cspline_100_nodes_continuity(void **state) {
    test_fixture_t *fixture = *state;
    const size_t n = 100;
    lmmc_real_t xs[100];
    lmmc_real_t ys[100];

    lmmc_status_t st;
    size_t i;
    double h = 1e-6;
    double tol_continuity = 1e-4;

    for (i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i * 10.0 / (lmmc_real_t)(n - 1);
        ys[i] = sin(xs[i]) * exp(-0.1 * xs[i]);
    }

    st = lmmc_interp_cspline_create(xs, ys, n, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 1; i < n - 1; i++) {
        lmmc_real_t x_knot = xs[i];
        lmmc_real_t y_left, y_right, y_center;
        lmmc_real_t deriv_left, deriv_right;

        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, x_knot - h, &y_left);
        assert_int_equal(st, LMMC_STATUS_OK);
        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, x_knot + h, &y_right);
        assert_int_equal(st, LMMC_STATUS_OK);
        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, x_knot, &y_center);
        assert_int_equal(st, LMMC_STATUS_OK);

        deriv_left = (y_center - y_left) / h;
        deriv_right = (y_right - y_center) / h;

        assert_true(fabs(deriv_left - deriv_right) < tol_continuity);
    }

    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_cspline_sin_20_nodes, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cspline_linear_exact, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lagrange_runge_chebyshev_vs_equidistant, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lagrange_3node_quadratic, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cspline_exact_at_nodes, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lagrange_exact_at_nodes, setup, teardown),
        cmocka_unit_test_setup_teardown(test_insufficient_nodes_cspline, setup, teardown),
        cmocka_unit_test_setup_teardown(test_insufficient_nodes_lagrange, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cspline_100_nodes_continuity, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
