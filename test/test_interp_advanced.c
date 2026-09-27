/**
 * @file test_interp_advanced.c
 * 测试扩展插值功能：边界条件、PCHIP、Akima、二维插值。
 */
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define EPS_TIGHT 1e-12
#define EPS_NORMAL 1e-10
#define EPS_LOOSE 1e-4

typedef struct {
    lmmc_interp_cspline_t *interp_cspline_spline;
    lmmc_interp_pchip_t *interp_pchip_p;
    lmmc_interp_akima_t *interp_akima_a;
    lmmc_interp_cspline_t *interp_cspline_s;
    lmmc_interp_lagrange_t *interp_lagrange_l;
} test_fixture_t;

static int setup(void **state) {
    test_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_l);
    lmmc_interp_cspline_destroy(fixture->interp_cspline_s);
    lmmc_interp_akima_destroy(fixture->interp_akima_a);
    lmmc_interp_pchip_destroy(fixture->interp_pchip_p);
    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    free(fixture);
    return 0;
}

static void test_cspline_clamped_linear(void **state) {
    test_fixture_t *fixture = *state;
    /* A linear function with clamped BC matching the true derivative
     * should reproduce the function exactly. */
    const size_t n = 5;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {1.0, 3.0, 5.0, 7.0, 9.0}; /* y = 2x + 1 */

    lmmc_status_t st;
    size_t i;

    st = lmmc_interp_cspline_create_ex(xs, ys, n,
                                       LMMC_SPLINE_CLAMPED, 2.0, 2.0, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < 20; i++) {
        lmmc_real_t x = (lmmc_real_t)i * 4.0 / 19.0;
        lmmc_real_t result, expected = 2.0 * x + 1.0;
        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, x, &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(fabs(result - expected) < EPS_TIGHT);
    }
    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

static void test_cspline_not_a_knot_quadratic(void **state) {
    test_fixture_t *fixture = *state;
    /* A quadratic function with not-a-knot BC should be reproduced exactly
     * (cubic spline can represent quadratics exactly). */
    const size_t n = 5;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[5];

    lmmc_status_t st;
    size_t i;

    for (i = 0; i < n; i++)
        ys[i] = xs[i] * xs[i]; /* y = x^2 */

    st = lmmc_interp_cspline_create_ex(xs, ys, n,
                                       LMMC_SPLINE_NOT_A_KNOT, 0.0, 0.0, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < 20; i++) {
        lmmc_real_t x = (lmmc_real_t)i * 4.0 / 19.0;
        lmmc_real_t result, expected = x * x;
        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, x, &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(fabs(result - expected) < EPS_NORMAL);
    }
    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

static void test_cspline_periodic_sin(void **state) {
    test_fixture_t *fixture = *state;
    /* sin(x) on [0, 2*pi] is periodic with matching endpoints. */
    const size_t n = 21;
    lmmc_real_t xs[21], ys[21];

    lmmc_status_t st;
    size_t i;

    for (i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i * 2.0 * LMMC_CONST_PI / (lmmc_real_t)(n - 1);
        ys[i] = sin(xs[i]);
    }
    /* Force exact periodicity */
    ys[n - 1] = ys[0];

    st = lmmc_interp_cspline_create_ex(xs, ys, n,
                                       LMMC_SPLINE_PERIODIC, 0.0, 0.0, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < 50; i++) {
        lmmc_real_t x = (lmmc_real_t)(i + 1) * 2.0 * LMMC_CONST_PI / 51.0;
        lmmc_real_t result, expected = sin(x);
        st = lmmc_interp_cspline_eval(fixture->interp_cspline_spline, x, &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(fabs(result - expected) < EPS_LOOSE);
    }
    lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
    fixture->interp_cspline_spline = NULL;
}

static void test_cspline_periodic_reject_mismatch(void **state) {
    test_fixture_t *fixture = *state;
    /* Periodic BC should reject if endpoints differ by > 1e-12 */
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
    lmmc_real_t ys[] = {1.0, 2.0, 3.0, 1.5}; /* ys[0] != ys[3] */

    lmmc_status_t st;

    st = lmmc_interp_cspline_create_ex(xs, ys, 4,
                                       LMMC_SPLINE_PERIODIC, 0.0, 0.0, &fixture->interp_cspline_spline);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_pchip_monotone_increasing(void **state) {
    test_fixture_t *fixture = *state;
    /* PCHIP should preserve monotonicity of monotone data. */
    const size_t n = 6;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    lmmc_real_t ys[] = {0.0, 0.5, 1.0, 2.0, 4.0, 8.0}; /* strictly increasing */

    lmmc_status_t st;
    size_t i;
    lmmc_real_t prev_y;

    st = lmmc_interp_pchip_create(xs, ys, n, &fixture->interp_pchip_p);
    assert_true(st == LMMC_STATUS_OK);

    /* Evaluate at many points and check monotonicity */
    st = lmmc_interp_pchip_eval(fixture->interp_pchip_p, 0.0, &prev_y);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 1; i <= 100; i++) {
        lmmc_real_t x = (lmmc_real_t)i * 5.0 / 100.0;
        lmmc_real_t y;
        st = lmmc_interp_pchip_eval(fixture->interp_pchip_p, x, &y);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(y >= prev_y - 1e-15);
        prev_y = y;
    }
    lmmc_interp_pchip_destroy(fixture->interp_pchip_p);
    fixture->interp_pchip_p = NULL;
}

static void test_pchip_exact_at_nodes(void **state) {
    test_fixture_t *fixture = *state;
    const size_t n = 5;
    lmmc_real_t xs[] = {0.0, 1.0, 3.0, 5.0, 7.0};
    lmmc_real_t ys[] = {1.0, 2.5, 0.5, 3.0, 2.0};

    lmmc_status_t st;
    size_t i;

    st = lmmc_interp_pchip_create(xs, ys, n, &fixture->interp_pchip_p);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < n; i++) {
        lmmc_real_t result;
        st = lmmc_interp_pchip_eval(fixture->interp_pchip_p, xs[i], &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(fabs(result - ys[i]) < EPS_TIGHT);
    }
    lmmc_interp_pchip_destroy(fixture->interp_pchip_p);
    fixture->interp_pchip_p = NULL;
}

static void test_pchip_two_points(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0, 1.0};
    lmmc_real_t ys[] = {2.0, 5.0};

    lmmc_status_t st;
    lmmc_real_t result;

    st = lmmc_interp_pchip_create(xs, ys, 2, &fixture->interp_pchip_p);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_interp_pchip_eval(fixture->interp_pchip_p, 0.5, &result);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fabs(result - 3.5) < EPS_TIGHT);

    lmmc_interp_pchip_destroy(fixture->interp_pchip_p);
    fixture->interp_pchip_p = NULL;
}

static void test_akima_exact_at_nodes(void **state) {
    test_fixture_t *fixture = *state;
    const size_t n = 7;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    lmmc_real_t ys[] = {1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5};

    lmmc_status_t st;
    size_t i;

    st = lmmc_interp_akima_create(xs, ys, n, &fixture->interp_akima_a);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < n; i++) {
        lmmc_real_t result;
        st = lmmc_interp_akima_eval(fixture->interp_akima_a, xs[i], &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(fabs(result - ys[i]) < EPS_TIGHT);
    }
    lmmc_interp_akima_destroy(fixture->interp_akima_a);
    fixture->interp_akima_a = NULL;
}

static void test_akima_linear_exact(void **state) {
    test_fixture_t *fixture = *state;
    /* Akima should reproduce a linear function exactly. */
    const size_t n = 6;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    lmmc_real_t ys[6];

    lmmc_status_t st;
    size_t i;

    for (i = 0; i < n; i++)
        ys[i] = 2.0 * xs[i] + 1.0;

    st = lmmc_interp_akima_create(xs, ys, n, &fixture->interp_akima_a);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < 20; i++) {
        lmmc_real_t x = (lmmc_real_t)i * 5.0 / 19.0;
        lmmc_real_t result, expected = 2.0 * x + 1.0;
        st = lmmc_interp_akima_eval(fixture->interp_akima_a, x, &result);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(fabs(result - expected) < EPS_NORMAL);
    }
    lmmc_interp_akima_destroy(fixture->interp_akima_a);
    fixture->interp_akima_a = NULL;
}

static void test_akima_too_few_points(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
    lmmc_real_t ys[] = {0.0, 1.0, 2.0, 3.0};

    lmmc_status_t st;

    st = lmmc_interp_akima_create(xs, ys, 4, &fixture->interp_akima_a);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_bilinear_exact_plane(void **state) {
    (void)state;
    /* Bilinear should reproduce a bilinear function z = ax + by + c exactly. */
    const size_t nx = 3, ny = 4;
    lmmc_real_t xs[] = {0.0, 2.0, 5.0};
    lmmc_real_t ys[] = {0.0, 1.0, 3.0, 4.0};
    lmmc_real_t zs[3 * 4]; /* row-major: zs[i*ny + j] */
    lmmc_status_t st;
    size_t i, j;

    /* z = 3*x + 2*y + 1 */
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            zs[i * ny + j] = 3.0 * xs[i] + 2.0 * ys[j] + 1.0;

    /* Test at several interior points */
    for (i = 0; i < 10; i++) {
        lmmc_real_t qx = (lmmc_real_t)i * 5.0 / 9.0;
        for (j = 0; j < 10; j++) {
            lmmc_real_t qy = (lmmc_real_t)j * 4.0 / 9.0;
            lmmc_real_t result, expected = 3.0 * qx + 2.0 * qy + 1.0;
            st = lmmc_interp_bilinear(xs, nx, ys, ny, zs, qx, qy, &result);
            assert_true(st == LMMC_STATUS_OK);
            assert_true(fabs(result - expected) < EPS_NORMAL);
        }
    }
}

static void test_bilinear_out_of_range(void **state) {
    (void)state;
    lmmc_real_t xs[] = {0.0, 1.0};
    lmmc_real_t ys[] = {0.0, 1.0};
    lmmc_real_t zs[] = {1.0, 2.0, 3.0, 4.0};
    lmmc_real_t result;
    lmmc_status_t st;

    st = lmmc_interp_bilinear(xs, 2, ys, 2, zs, -0.1, 0.5, &result);
    assert_true(st == LMMC_STATUS_OUT_OF_RANGE);

    st = lmmc_interp_bilinear(xs, 2, ys, 2, zs, 0.5, 1.1, &result);
    assert_true(st == LMMC_STATUS_OUT_OF_RANGE);
}

static void test_bicubic_quadratic(void **state) {
    (void)state;
    const size_t nx = 5, ny = 5;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t zs[5 * 5];
    lmmc_status_t st;
    size_t i, j;

    /* z = x^2 + y^2 */
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            zs[i * ny + j] = xs[i] * xs[i] + ys[j] * ys[j];

    /* Test at interior points (away from boundaries for best accuracy) */
    for (i = 1; i < 8; i++) {
        lmmc_real_t qx = 1.0 + (lmmc_real_t)i * 2.0 / 8.0;
        for (j = 1; j < 8; j++) {
            lmmc_real_t qy = 1.0 + (lmmc_real_t)j * 2.0 / 8.0;
            lmmc_real_t result, expected = qx * qx + qy * qy;
            st = lmmc_interp_bicubic(xs, nx, ys, ny, zs, qx, qy, &result);
            assert_true(st == LMMC_STATUS_OK);
            assert_true(fabs(result - expected) < EPS_NORMAL);
        }
    }
}

static void test_bicubic_too_few_points(void **state) {
    (void)state;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {0.0, 1.0, 2.0, 3.0};
    lmmc_real_t zs[12] = {0};
    lmmc_real_t result;
    lmmc_status_t st;

    st = lmmc_interp_bicubic(xs, 3, ys, 4, zs, 1.0, 1.0, &result);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_non_increasing_abscissae_rejected(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0, 2.0, 1.0, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 1.0, 2.0, 3.0, 4.0};

    lmmc_status_t st;

    st = lmmc_interp_pchip_create(xs, ys, 5, &fixture->interp_pchip_p);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_akima_create(xs, ys, 5, &fixture->interp_akima_a);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_cspline_create_ex(xs, ys, 5,
                                       LMMC_SPLINE_NATURAL, 0.0, 0.0, &fixture->interp_cspline_s);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_invalid_and_nonfinite_inputs_rejected(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};
    lmmc_real_t xs_nan[] = {0.0, 1.0, NAN, 3.0, 4.0};
    lmmc_real_t ys_nan[] = {0.0, 1.0, NAN, 9.0, 16.0};

    lmmc_status_t st;

    st = lmmc_interp_cspline_create_ex(
        xs, ys, 5, (lmmc_spline_bc_t)99, 0.0, 0.0, &fixture->interp_cspline_s);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_cspline_create_ex(
        xs, ys, 5, LMMC_SPLINE_CLAMPED, NAN, 0.0, &fixture->interp_cspline_s);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_pchip_create(xs_nan, ys, 5, &fixture->interp_pchip_p);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_akima_create(xs, ys_nan, 5, &fixture->interp_akima_a);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_interp_lagrange_create(xs_nan, ys, 5, &fixture->interp_lagrange_l);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_nonfinite_evaluation_inputs_rejected(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};
    lmmc_real_t grid[] = {
        0.0, 1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0, 5.0,
        2.0, 3.0, 4.0, 5.0, 6.0,
        3.0, 4.0, 5.0, 6.0, 7.0,
        4.0, 5.0, 6.0, 7.0, 8.0};

    lmmc_real_t result = 42.0;
    lmmc_status_t st;

    assert_true(lmmc_interp_cspline_create(xs, ys, 5, &fixture->interp_cspline_s) == LMMC_STATUS_OK);
    assert_true(lmmc_interp_pchip_create(xs, ys, 5, &fixture->interp_pchip_p) == LMMC_STATUS_OK);
    assert_true(lmmc_interp_akima_create(xs, ys, 5, &fixture->interp_akima_a) == LMMC_STATUS_OK);
    assert_true(lmmc_interp_lagrange_create(xs, ys, 5, &fixture->interp_lagrange_l) == LMMC_STATUS_OK);

    st = lmmc_interp_cspline_eval(fixture->interp_cspline_s, NAN, &result);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
    st = lmmc_interp_pchip_eval(fixture->interp_pchip_p, NAN, &result);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
    st = lmmc_interp_akima_eval(fixture->interp_akima_a, NAN, &result);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
    st = lmmc_interp_lagrange_eval(fixture->interp_lagrange_l, NAN, &result);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
    st = lmmc_interp_bilinear(xs, 5, xs, 5, grid, NAN, 1.0, &result);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    grid[12] = NAN;
    st = lmmc_interp_bicubic(xs, 5, xs, 5, grid, 1.0, 1.0, &result);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_interp_cspline_destroy(fixture->interp_cspline_s);
    fixture->interp_cspline_s = NULL;
    lmmc_interp_pchip_destroy(fixture->interp_pchip_p);
    fixture->interp_pchip_p = NULL;
    lmmc_interp_akima_destroy(fixture->interp_akima_a);
    fixture->interp_akima_a = NULL;
    lmmc_interp_lagrange_destroy(fixture->interp_lagrange_l);
    fixture->interp_lagrange_l = NULL;
}

static void test_nonfinite_coefficients_rejected(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_real_t xs_extreme[] = {-DBL_MAX, -1.0, 0.0, 1.0, DBL_MAX};
    lmmc_real_t xs_regular[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys_regular[] = {0.0, 1.0, 4.0, 9.0, 16.0};
    lmmc_real_t ys_extreme[] = {-DBL_MAX, DBL_MAX, 0.0, 1.0, 2.0};

    assert_true(lmmc_interp_cspline_create(xs_regular, ys_extreme, 5, &fixture->interp_cspline_s) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(fixture->interp_cspline_s == NULL);

    assert_true(lmmc_interp_pchip_create(xs_regular, ys_extreme, 5, &fixture->interp_pchip_p) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(fixture->interp_pchip_p == NULL);

    assert_true(lmmc_interp_akima_create(xs_regular, ys_extreme, 5, &fixture->interp_akima_a) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(fixture->interp_akima_a == NULL);

    assert_true(lmmc_interp_lagrange_create(xs_extreme, ys_regular, 5, &fixture->interp_lagrange_l) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(fixture->interp_lagrange_l == NULL);
}

static void test_not_a_knot_small_nonuniform(void **state) {
    test_fixture_t *fixture = *state;
    const lmmc_real_t xs[] = {-1.0, 0.25, 2.0, 5.0};
    lmmc_real_t ys[4];
    size_t n, i;
    for (n = 3; n <= 4; ++n) {

        for (i = 0; i < n; ++i) {
            ys[i] = n == 3 ? xs[i] * xs[i] : xs[i] * xs[i] * xs[i];
        }
        assert_true(lmmc_interp_cspline_create_ex(xs, ys, n,
                                                  LMMC_SPLINE_NOT_A_KNOT, 0.0, 0.0, &fixture->interp_cspline_spline) == LMMC_STATUS_OK);
        for (i = 0; i <= 20; ++i) {
            lmmc_real_t query = xs[0] + (xs[n - 1] - xs[0]) * (lmmc_real_t)i / 20.0;
            lmmc_real_t expected = n == 3 ? query * query : query * query * query;
            lmmc_real_t result;
            assert_true(lmmc_interp_cspline_eval(fixture->interp_cspline_spline, query, &result) == LMMC_STATUS_OK);
            assert_true(isfinite(result) && fabs(result - expected) <= EPS_NORMAL);
        }
        lmmc_interp_cspline_destroy(fixture->interp_cspline_spline);
        fixture->interp_cspline_spline = NULL;
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_cspline_clamped_linear, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cspline_not_a_knot_quadratic, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cspline_periodic_sin, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cspline_periodic_reject_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_pchip_monotone_increasing, setup, teardown),
        cmocka_unit_test_setup_teardown(test_pchip_exact_at_nodes, setup, teardown),
        cmocka_unit_test_setup_teardown(test_pchip_two_points, setup, teardown),
        cmocka_unit_test_setup_teardown(test_akima_exact_at_nodes, setup, teardown),
        cmocka_unit_test_setup_teardown(test_akima_linear_exact, setup, teardown),
        cmocka_unit_test_setup_teardown(test_akima_too_few_points, setup, teardown),
        cmocka_unit_test(test_bilinear_exact_plane),
        cmocka_unit_test(test_bilinear_out_of_range),
        cmocka_unit_test(test_bicubic_quadratic),
        cmocka_unit_test(test_bicubic_too_few_points),
        cmocka_unit_test_setup_teardown(test_non_increasing_abscissae_rejected, setup, teardown),
        cmocka_unit_test_setup_teardown(test_invalid_and_nonfinite_inputs_rejected, setup, teardown),
        cmocka_unit_test_setup_teardown(test_nonfinite_evaluation_inputs_rejected, setup, teardown),
        cmocka_unit_test_setup_teardown(test_nonfinite_coefficients_rejected, setup, teardown),
        cmocka_unit_test_setup_teardown(test_not_a_knot_small_nonuniform, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
