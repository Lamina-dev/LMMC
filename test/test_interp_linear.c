/**
 * @file test_interp_linear.c
 * 针对 LMMC 中 interp linear 相关接口的单元测试。
 */
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static void test_affine_reproduction(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t out_y = 0.0;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    lmmc_real_t ys[] = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0};
    size_t n = 6;
    lmmc_real_t queries[] = {0.5, 1.5, 2.5, 3.5, 4.5};
    size_t nq = 5;

    for (size_t i = 0; i < nq; i++) {
        lmmc_real_t expected = 2.0 * queries[i] + 1.0;
        st = lmmc_interp_linear(xs, ys, n, queries[i], &out_y);
        assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, expected, 1e-14));
    }
}

static void test_exact_nodes(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t out_y = 0.0;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    size_t n = 5;

    for (size_t i = 0; i < n; i++) {
        st = lmmc_interp_linear(xs, ys, n, xs[i], &out_y);
        assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, ys[i], 1e-14));
    }
}

static void test_two_point_endpoints(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t out_y = 0.0;
    lmmc_real_t xs[] = {1.0, 3.0};
    lmmc_real_t ys[] = {5.0, 11.0};
    size_t n = 2;
    lmmc_real_t query = 2.0;
    lmmc_real_t expected = 8.0;

    st = lmmc_interp_linear(xs, ys, n, query, &out_y);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, expected, 1e-14));

    st = lmmc_interp_linear(xs, ys, n, 1.0, &out_y);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, 5.0, 1e-14));

    st = lmmc_interp_linear(xs, ys, n, 3.0, &out_y);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, 11.0, 1e-14));
}

static void test_extreme_midpoint(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t out_y = 0.0;
    const lmmc_real_t xs[] = {-DBL_MAX, DBL_MAX};
    const lmmc_real_t ys[] = {-DBL_MAX, DBL_MAX};

    st = lmmc_interp_linear(xs, ys, 2, 0.0, &out_y);
    assert_false(st != LMMC_STATUS_OK || out_y != 0.0);
}

static void test_dense_grid_reproduction(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t out_y = 0.0;
    lmmc_real_t xs[100];
    lmmc_real_t ys[100];
    size_t n = 100;

    for (size_t i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i;
        ys[i] = 2.0 * xs[i] + 1.0;
    }

    for (size_t i = 0; i < n - 1; i++) {
        lmmc_real_t qx = xs[i] + 0.5;
        lmmc_real_t expected = 2.0 * qx + 1.0;
        st = lmmc_interp_linear(xs, ys, n, qx, &out_y);
        assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, expected, 1e-12));
    }

    for (size_t i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i / (lmmc_real_t)(n - 1);
        ys[i] = xs[i] * xs[i];
    }

    for (size_t i = 0; i < n - 1; i++) {
        lmmc_real_t qx = (xs[i] + xs[i + 1]) / 2.0;
        lmmc_real_t interp_expected = (ys[i] + ys[i + 1]) / 2.0;
        st = lmmc_interp_linear(xs, ys, n, qx, &out_y);
        assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, interp_expected, 1e-12));
    }
}

static void test_insufficient_nodes(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t out_y = 0.0;
    lmmc_real_t xs[] = {1.0};
    lmmc_real_t ys[] = {2.0};

    st = lmmc_interp_linear(xs, ys, 1, 1.0, &out_y);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_linear(xs, ys, 0, 1.0, &out_y);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_null_arguments(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t out_y = 0.0;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {0.0, 1.0, 2.0};

    st = lmmc_interp_linear(NULL, ys, 3, 1.0, &out_y);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_linear(xs, NULL, 3, 1.0, &out_y);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_linear(xs, ys, 3, 1.0, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_query_range(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t out_y = 0.0;
    lmmc_real_t xs[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    lmmc_real_t ys[] = {2.0, 4.0, 6.0, 8.0, 10.0};

    st = lmmc_interp_linear(xs, ys, 5, 0.5, &out_y);
    assert_false(st != LMMC_STATUS_OUT_OF_RANGE);

    st = lmmc_interp_linear(xs, ys, 5, 5.5, &out_y);
    assert_false(st != LMMC_STATUS_OUT_OF_RANGE);
}

static void test_invalid_coordinates(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t out_y = 0.0;
    lmmc_real_t xs_dup[] = {1.0, 2.0, 2.0, 3.0};
    lmmc_real_t ys_dup[] = {1.0, 2.0, 3.0, 4.0};
    lmmc_real_t xs_dec[] = {3.0, 2.0, 1.0};
    lmmc_real_t ys_dec[] = {6.0, 4.0, 2.0};
    lmmc_real_t xs_valid[] = {1.0, 2.0, 3.0};
    lmmc_real_t xs_nan[] = {1.0, NAN, 3.0};
    lmmc_real_t ys_nan[] = {1.0, NAN, 3.0};

    st = lmmc_interp_linear(xs_dup, ys_dup, 4, 2.5, &out_y);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_linear(xs_dec, ys_dec, 3, 2.0, &out_y);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_linear(xs_nan, ys_dup, 3, 2.0, &out_y);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_linear(xs_valid, ys_nan, 3, 2.0, &out_y);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_linear(xs_dup, ys_dup, 4, NAN, &out_y);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_piecewise_values(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t out_y = 0.0;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
    lmmc_real_t ys[] = {0.0, 10.0, 4.0, 7.0};

    st = lmmc_interp_linear(xs, ys, 4, 0.3, &out_y);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, 3.0, 1e-14));

    st = lmmc_interp_linear(xs, ys, 4, 1.5, &out_y);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, 7.0, 1e-14));

    st = lmmc_interp_linear(xs, ys, 4, 2.7, &out_y);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, 6.1, 1e-14));
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_affine_reproduction),
        cmocka_unit_test(test_exact_nodes),
        cmocka_unit_test(test_two_point_endpoints),
        cmocka_unit_test(test_extreme_midpoint),
        cmocka_unit_test(test_dense_grid_reproduction),
        cmocka_unit_test(test_insufficient_nodes),
        cmocka_unit_test(test_null_arguments),
        cmocka_unit_test(test_query_range),
        cmocka_unit_test(test_invalid_coordinates),
        cmocka_unit_test(test_piecewise_values),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
