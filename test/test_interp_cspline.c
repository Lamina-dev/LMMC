/**
 * @file test_interp_cspline.c
 * @brief Unit tests for cubic spline interpolation (natural boundary conditions).
 *
 * Tests cover:
 * - Parameter validation (NULL pointers, n < 3, non-increasing xs)
 * - Interpolation passes through data points (Property 15)
 * - Out-of-range query returns LMMC_STATUS_OUT_OF_RANGE
 * - Derivative continuity at internal knots (Property 16)
 * - Correct interpolation of known functions
 *
 * Validates: Requirements 11.1–11.7
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/config.h"
#include "lmmc/status.h"
#include "lmmc/interp.h"
#include "test_common.h"

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)

/* ========================================================================
 * Test: Parameter validation
 * ======================================================================== */

static int test_cspline_null_args(void)
{
    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0};
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;

    /* NULL xs */
    st = lmmc_interp_cspline_create(NULL, ys, 3, &spline);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "NULL xs should fail");

    /* NULL ys */
    st = lmmc_interp_cspline_create(xs, NULL, 3, &spline);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "NULL ys should fail");

    /* NULL out_spline */
    st = lmmc_interp_cspline_create(xs, ys, 3, NULL);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "NULL out_spline should fail");

    return 0;
}

static int test_cspline_too_few_points(void)
{
    lmmc_real_t xs[] = {0.0, 1.0};
    lmmc_real_t ys[] = {0.0, 1.0};
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;

    /* n < 3 */
    st = lmmc_interp_cspline_create(xs, ys, 2, &spline);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "n=2 should fail (need >= 3)");

    st = lmmc_interp_cspline_create(xs, ys, 0, &spline);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "n=0 should fail");

    return 0;
}

static int test_cspline_non_increasing_xs(void)
{
    lmmc_real_t xs_equal[] = {0.0, 1.0, 1.0};
    lmmc_real_t xs_decreasing[] = {0.0, 2.0, 1.0};
    lmmc_real_t ys[] = {0.0, 1.0, 2.0};
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;

    st = lmmc_interp_cspline_create(xs_equal, ys, 3, &spline);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "equal xs should fail");

    st = lmmc_interp_cspline_create(xs_decreasing, ys, 3, &spline);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "decreasing xs should fail");

    return 0;
}

/* ========================================================================
 * Test: Spline passes through data points (Property 15)
 * ======================================================================== */

static int test_cspline_passes_through_data_points(void)
{
    /* Use f(x) = x^2 sampled at 5 points */
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    lmmc_real_t result;
    size_t i;
    double eps = 1e-12;

    st = lmmc_interp_cspline_create(xs, ys, 5, &spline);
    CHECK(st == LMMC_STATUS_OK, "create should succeed");

    for (i = 0; i < 5; i++) {
        st = lmmc_interp_cspline_eval(spline, xs[i], &result);
        CHECK(st == LMMC_STATUS_OK, "eval at data point %zu should succeed", i);
        CHECK(lmmc_test_nearly_equal(result, ys[i], eps),
              "eval at x=%.1f: expected %.6f, got %.6f", xs[i], ys[i], result);
    }

    lmmc_interp_cspline_destroy(spline);
    return 0;
}

/* ========================================================================
 * Test: Out-of-range query
 * ======================================================================== */

static int test_cspline_out_of_range(void)
{
    lmmc_real_t xs[] = {1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {1.0, 4.0, 9.0, 16.0};
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    lmmc_real_t result;

    st = lmmc_interp_cspline_create(xs, ys, 4, &spline);
    CHECK(st == LMMC_STATUS_OK, "create should succeed");

    /* Below range */
    st = lmmc_interp_cspline_eval(spline, 0.5, &result);
    CHECK(st == LMMC_STATUS_OUT_OF_RANGE, "below range should return OUT_OF_RANGE");

    /* Above range */
    st = lmmc_interp_cspline_eval(spline, 4.5, &result);
    CHECK(st == LMMC_STATUS_OUT_OF_RANGE, "above range should return OUT_OF_RANGE");

    lmmc_interp_cspline_destroy(spline);
    return 0;
}

/* ========================================================================
 * Test: Eval with NULL spline
 * ======================================================================== */

static int test_cspline_eval_null(void)
{
    lmmc_real_t result;
    lmmc_status_t st;

    st = lmmc_interp_cspline_eval(NULL, 1.0, &result);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "NULL spline should fail");

    return 0;
}

/* ========================================================================
 * Test: Derivative continuity at internal knots (Property 16)
 *
 * Approximate first and second derivatives from left and right using
 * finite differences and verify they match at internal knots.
 * ======================================================================== */

static int test_cspline_derivative_continuity(void)
{
    /* Use sin(x) sampled at 6 points in [0, 5] */
    lmmc_real_t xs[6], ys[6];
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    size_t i;
    double h = 1e-5;
    double tol_d1 = 1e-4; /* tolerance for first derivative finite difference */
    double tol_d2 = 1e-2; /* tolerance for second derivative finite difference (less accurate) */

    for (i = 0; i < 6; i++) {
        xs[i] = (lmmc_real_t)i;
        ys[i] = sin((double)i);
    }

    st = lmmc_interp_cspline_create(xs, ys, 6, &spline);
    CHECK(st == LMMC_STATUS_OK, "create should succeed");

    /* Check continuity at internal knots (indices 1..4) */
    for (i = 1; i < 5; i++) {
        lmmc_real_t x_knot = xs[i];
        lmmc_real_t y_left, y_right, y_center;
        lmmc_real_t deriv_left, deriv_right;
        lmmc_real_t deriv2_left, deriv2_right;
        lmmc_real_t y_ll, y_rr;

        /* First derivative: f'(x) ≈ (f(x+h) - f(x-h)) / (2h) */
        st = lmmc_interp_cspline_eval(spline, x_knot - h, &y_left);
        CHECK(st == LMMC_STATUS_OK, "eval left of knot %zu", i);
        st = lmmc_interp_cspline_eval(spline, x_knot + h, &y_right);
        CHECK(st == LMMC_STATUS_OK, "eval right of knot %zu", i);
        st = lmmc_interp_cspline_eval(spline, x_knot, &y_center);
        CHECK(st == LMMC_STATUS_OK, "eval at knot %zu", i);

        /* Approximate first derivative from left and right */
        deriv_left = (y_center - y_left) / h;
        deriv_right = (y_right - y_center) / h;
        CHECK(fabs(deriv_left - deriv_right) < tol_d1,
              "first derivative discontinuity at knot %zu: left=%.8f, right=%.8f",
              i, deriv_left, deriv_right);

        /* Second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2 */
        /* From left side */
        st = lmmc_interp_cspline_eval(spline, x_knot - 2*h, &y_ll);
        CHECK(st == LMMC_STATUS_OK, "eval far left of knot %zu", i);
        deriv2_left = (y_center - 2.0*y_left + y_ll) / (h*h);

        /* From right side */
        st = lmmc_interp_cspline_eval(spline, x_knot + 2*h, &y_rr);
        CHECK(st == LMMC_STATUS_OK, "eval far right of knot %zu", i);
        deriv2_right = (y_rr - 2.0*y_right + y_center) / (h*h);

        CHECK(fabs(deriv2_left - deriv2_right) < tol_d2,
              "second derivative discontinuity at knot %zu: left=%.8f, right=%.8f",
              i, deriv2_left, deriv2_right);
    }

    lmmc_interp_cspline_destroy(spline);
    return 0;
}

/* ========================================================================
 * Test: Linear function should be interpolated exactly
 * ======================================================================== */

static int test_cspline_linear_function(void)
{
    /* f(x) = 2x + 1, natural spline should reproduce linear exactly */
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {1.0, 3.0, 5.0, 7.0, 9.0};
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-12;
    double x;

    st = lmmc_interp_cspline_create(xs, ys, 5, &spline);
    CHECK(st == LMMC_STATUS_OK, "create should succeed");

    /* Test at intermediate points */
    for (x = 0.0; x <= 4.0; x += 0.25) {
        lmmc_real_t expected = 2.0 * x + 1.0;
        st = lmmc_interp_cspline_eval(spline, (lmmc_real_t)x, &result);
        CHECK(st == LMMC_STATUS_OK, "eval at x=%.2f should succeed", x);
        CHECK(lmmc_test_nearly_equal(result, expected, eps),
              "linear at x=%.2f: expected %.6f, got %.6f", x, expected, result);
    }

    lmmc_interp_cspline_destroy(spline);
    return 0;
}

/* ========================================================================
 * Test: Destroy NULL is safe
 * ======================================================================== */

static int test_cspline_destroy_null(void)
{
    /* Should not crash */
    lmmc_interp_cspline_destroy(NULL);
    return 0;
}

/* ========================================================================
 * Test: Minimum 3 points (boundary case)
 * ======================================================================== */

static int test_cspline_three_points(void)
{
    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {0.0, 1.0, 0.0};
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-12;

    st = lmmc_interp_cspline_create(xs, ys, 3, &spline);
    CHECK(st == LMMC_STATUS_OK, "create with 3 points should succeed");

    /* Check passes through data points */
    st = lmmc_interp_cspline_eval(spline, 0.0, &result);
    CHECK(st == LMMC_STATUS_OK, "eval at x=0");
    CHECK(lmmc_test_nearly_equal(result, 0.0, eps), "at x=0: got %.6f", result);

    st = lmmc_interp_cspline_eval(spline, 1.0, &result);
    CHECK(st == LMMC_STATUS_OK, "eval at x=1");
    CHECK(lmmc_test_nearly_equal(result, 1.0, eps), "at x=1: got %.6f", result);

    st = lmmc_interp_cspline_eval(spline, 2.0, &result);
    CHECK(st == LMMC_STATUS_OK, "eval at x=2");
    CHECK(lmmc_test_nearly_equal(result, 0.0, eps), "at x=2: got %.6f", result);

    lmmc_interp_cspline_destroy(spline);
    return 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(void)
{
    int failed = 0;

    printf("=== Cubic Spline Interpolation Tests ===\n\n");

    printf("test_cspline_null_args... ");
    if (test_cspline_null_args() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_too_few_points... ");
    if (test_cspline_too_few_points() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_non_increasing_xs... ");
    if (test_cspline_non_increasing_xs() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_passes_through_data_points... ");
    if (test_cspline_passes_through_data_points() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_out_of_range... ");
    if (test_cspline_out_of_range() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_eval_null... ");
    if (test_cspline_eval_null() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_derivative_continuity... ");
    if (test_cspline_derivative_continuity() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_linear_function... ");
    if (test_cspline_linear_function() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_destroy_null... ");
    if (test_cspline_destroy_null() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_three_points... ");
    if (test_cspline_three_points() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("\n=== Results: %d/%d tests passed ===\n", 10 - failed, 10);

    return (failed > 0) ? 1 : 0;
}
