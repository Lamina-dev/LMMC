/**
 * @file test_interp_extended.c
 * 针对 LMMC 中 interp extended 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define TEST_EPS_TIGHT   1e-12
#define TEST_EPS_NORMAL  1e-10
#define TEST_EPS_LOOSE   1e-4

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)


static int test_cspline_sin_20_nodes(void)
{
    const size_t n = 20;
    lmmc_real_t xs[20], ys[20];
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    size_t i;


    for (i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i * 2.0 * LMMC_CONST_PI / (lmmc_real_t)(n - 1);
        ys[i] = sin(xs[i]);
    }

    st = lmmc_interp_cspline_create(xs, ys, n, &spline);
    CHECK(st == LMMC_STATUS_OK, "cspline create with 20 sin nodes should succeed");


    for (i = 0; i < 50; i++) {
        lmmc_real_t query_x = (lmmc_real_t)(i + 1) * 2.0 * LMMC_CONST_PI / 51.0;
        lmmc_real_t result;
        lmmc_real_t expected = sin(query_x);

        st = lmmc_interp_cspline_eval(spline, query_x, &result);
        CHECK(st == LMMC_STATUS_OK, "eval at x=%.4f should succeed", query_x);
        CHECK(lmmc_test_nearly_equal(result, expected, TEST_EPS_LOOSE),
              "sin(%.4f): expected %.10f, got %.10f, error=%.2e",
              query_x, expected, result, fabs(result - expected));
    }

    lmmc_interp_cspline_destroy(spline);
    return 0;
}


static int test_cspline_linear_exact(void)
{
    const size_t n = 10;
    lmmc_real_t xs[10], ys[10];
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    size_t i;

    const lmmc_real_t a = 3.5, b = -2.7;

    for (i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i;
        ys[i] = a * xs[i] + b;
    }

    st = lmmc_interp_cspline_create(xs, ys, n, &spline);
    CHECK(st == LMMC_STATUS_OK, "cspline create for linear function should succeed");


    for (i = 0; i < 50; i++) {
        lmmc_real_t query_x = (lmmc_real_t)i * 9.0 / 50.0;
        lmmc_real_t result;
        lmmc_real_t expected = a * query_x + b;

        st = lmmc_interp_cspline_eval(spline, query_x, &result);
        CHECK(st == LMMC_STATUS_OK, "eval at x=%.4f should succeed", query_x);
        CHECK(lmmc_test_nearly_equal(result, expected, TEST_EPS_TIGHT),
              "linear(%.4f): expected %.12f, got %.12f, error=%.2e",
              query_x, expected, result, fabs(result - expected));
    }

    lmmc_interp_cspline_destroy(spline);
    return 0;
}


static int test_lagrange_runge_chebyshev_vs_equidistant(void)
{
    const size_t n = 15;
    lmmc_real_t xs_equi[15], ys_equi[15];
    lmmc_real_t xs_cheb[15], ys_cheb[15];
    lmmc_interp_lagrange_t* lag_equi = NULL;
    lmmc_interp_lagrange_t* lag_cheb = NULL;
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

    st = lmmc_interp_lagrange_create(xs_equi, ys_equi, n, &lag_equi);
    CHECK(st == LMMC_STATUS_OK, "lagrange create with equidistant nodes should succeed");

    st = lmmc_interp_lagrange_create(xs_cheb, ys_cheb, n, &lag_cheb);
    CHECK(st == LMMC_STATUS_OK, "lagrange create with Chebyshev nodes should succeed");


    for (i = 0; i < 100; i++) {
        lmmc_real_t x = -0.95 + 1.9 * (lmmc_real_t)i / 99.0;
        lmmc_real_t exact = 1.0 / (1.0 + 25.0 * x * x);
        lmmc_real_t result_equi, result_cheb;
        double err_equi, err_cheb;

        st = lmmc_interp_lagrange_eval(lag_equi, x, &result_equi);
        CHECK(st == LMMC_STATUS_OK, "eval equidistant at x=%.4f should succeed", x);

        st = lmmc_interp_lagrange_eval(lag_cheb, x, &result_cheb);
        CHECK(st == LMMC_STATUS_OK, "eval Chebyshev at x=%.4f should succeed", x);

        err_equi = fabs(result_equi - exact);
        err_cheb = fabs(result_cheb - exact);

        if (err_equi > max_err_equi) max_err_equi = err_equi;
        if (err_cheb > max_err_cheb) max_err_cheb = err_cheb;
    }


    CHECK(max_err_cheb < max_err_equi,
          "Chebyshev max error (%.6e) should be less than equidistant max error (%.6e)",
          max_err_cheb, max_err_equi);

    lmmc_interp_lagrange_destroy(lag_equi);
    lmmc_interp_lagrange_destroy(lag_cheb);
    return 0;
}


static int test_lagrange_3node_quadratic(void)
{

    lmmc_real_t xs[] = {-1.0, 0.0, 2.0};
    lmmc_real_t ys[3];
    lmmc_interp_lagrange_t* lag = NULL;
    lmmc_status_t st;
    size_t i;


    for (i = 0; i < 3; i++) {
        ys[i] = 2.0 * xs[i] * xs[i] - 3.0 * xs[i] + 1.0;
    }

    st = lmmc_interp_lagrange_create(xs, ys, 3, &lag);
    CHECK(st == LMMC_STATUS_OK, "lagrange create with 3 nodes should succeed");


    for (i = 0; i <= 20; i++) {
        lmmc_real_t x = -1.0 + 3.0 * (lmmc_real_t)i / 20.0;
        lmmc_real_t expected = 2.0 * x * x - 3.0 * x + 1.0;
        lmmc_real_t result;

        st = lmmc_interp_lagrange_eval(lag, x, &result);
        CHECK(st == LMMC_STATUS_OK, "eval at x=%.4f should succeed", x);
        CHECK(lmmc_test_nearly_equal(result, expected, TEST_EPS_NORMAL),
              "quadratic at x=%.4f: expected %.12f, got %.12f, error=%.2e",
              x, expected, result, fabs(result - expected));
    }

    lmmc_interp_lagrange_destroy(lag);
    return 0;
}


static int test_cspline_exact_at_nodes(void)
{
    const size_t n = 10;
    lmmc_real_t xs[10], ys[10];
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    size_t i;


    for (i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i * 0.5;
        ys[i] = sin(xs[i]) + 0.1 * cos(3.0 * xs[i]);
    }

    st = lmmc_interp_cspline_create(xs, ys, n, &spline);
    CHECK(st == LMMC_STATUS_OK, "cspline create should succeed");

    for (i = 0; i < n; i++) {
        lmmc_real_t result;
        st = lmmc_interp_cspline_eval(spline, xs[i], &result);
        CHECK(st == LMMC_STATUS_OK, "eval at node %zu should succeed", i);
        CHECK(lmmc_test_nearly_equal(result, ys[i], TEST_EPS_TIGHT),
              "cspline at node x=%.4f: expected %.15f, got %.15f",
              xs[i], ys[i], result);
    }

    lmmc_interp_cspline_destroy(spline);
    return 0;
}


static int test_lagrange_exact_at_nodes(void)
{
    const size_t n = 8;
    lmmc_real_t xs[8], ys[8];
    lmmc_interp_lagrange_t* lag = NULL;
    lmmc_status_t st;
    size_t i;


    for (i = 0; i < n; i++) {
        xs[i] = -2.0 + (lmmc_real_t)i * 0.7;
        ys[i] = exp(-xs[i] * xs[i]);
    }

    st = lmmc_interp_lagrange_create(xs, ys, n, &lag);
    CHECK(st == LMMC_STATUS_OK, "lagrange create should succeed");

    for (i = 0; i < n; i++) {
        lmmc_real_t result;
        st = lmmc_interp_lagrange_eval(lag, xs[i], &result);
        CHECK(st == LMMC_STATUS_OK, "eval at node %zu should succeed", i);
        CHECK(lmmc_test_nearly_equal(result, ys[i], TEST_EPS_TIGHT),
              "lagrange at node x=%.4f: expected %.15f, got %.15f",
              xs[i], ys[i], result);
    }

    lmmc_interp_lagrange_destroy(lag);
    return 0;
}


static int test_insufficient_nodes_cspline(void)
{
    lmmc_real_t xs[] = {0.0, 1.0};
    lmmc_real_t ys[] = {0.0, 1.0};
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;


    st = lmmc_interp_cspline_create(xs, ys, 2, &spline);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "cspline with n=2 should return INVALID_ARGUMENT, got %d", (int)st);

    st = lmmc_interp_cspline_create(xs, ys, 1, &spline);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "cspline with n=1 should return INVALID_ARGUMENT, got %d", (int)st);

    st = lmmc_interp_cspline_create(xs, ys, 0, &spline);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "cspline with n=0 should return INVALID_ARGUMENT, got %d", (int)st);

    return 0;
}

static int test_insufficient_nodes_lagrange(void)
{
    lmmc_real_t xs[] = {0.0};
    lmmc_real_t ys[] = {0.0};
    lmmc_interp_lagrange_t* lag = NULL;
    lmmc_status_t st;


    st = lmmc_interp_lagrange_create(xs, ys, 0, &lag);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "lagrange with n=0 should return INVALID_ARGUMENT, got %d", (int)st);

    return 0;
}


static int test_cspline_100_nodes_continuity(void)
{
    const size_t n = 100;
    lmmc_real_t* xs = NULL;
    lmmc_real_t* ys = NULL;
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    size_t i;
    double h = 1e-6;
    double tol_continuity = 1e-4;

    xs = (lmmc_real_t*)malloc(n * sizeof(lmmc_real_t));
    ys = (lmmc_real_t*)malloc(n * sizeof(lmmc_real_t));
    if (!xs || !ys) {
        free(xs);
        free(ys);
        CHECK(0, "memory allocation failed");
    }


    for (i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i * 10.0 / (lmmc_real_t)(n - 1);
        ys[i] = sin(xs[i]) * exp(-0.1 * xs[i]);
    }

    st = lmmc_interp_cspline_create(xs, ys, n, &spline);
    CHECK(st == LMMC_STATUS_OK, "cspline create with 100 nodes should succeed");


    for (i = 1; i < n - 1; i++) {
        lmmc_real_t x_knot = xs[i];
        lmmc_real_t y_left, y_right, y_center;
        lmmc_real_t deriv_left, deriv_right;

        st = lmmc_interp_cspline_eval(spline, x_knot - h, &y_left);
        if (st != LMMC_STATUS_OK) continue;
        st = lmmc_interp_cspline_eval(spline, x_knot + h, &y_right);
        if (st != LMMC_STATUS_OK) continue;
        st = lmmc_interp_cspline_eval(spline, x_knot, &y_center);
        if (st != LMMC_STATUS_OK) continue;

        deriv_left = (y_center - y_left) / h;
        deriv_right = (y_right - y_center) / h;

        CHECK(fabs(deriv_left - deriv_right) < tol_continuity,
              "derivative discontinuity at knot %zu (x=%.4f): left=%.8f, right=%.8f, diff=%.2e",
              i, x_knot, deriv_left, deriv_right, fabs(deriv_left - deriv_right));
    }

    lmmc_interp_cspline_destroy(spline);
    free(xs);
    free(ys);
    return 0;
}


int main(void)
{
    int failed = 0;

    printf("=== Interpolation Extended Tests ===\n\n");

    printf("test_cspline_sin_20_nodes... ");
    if (test_cspline_sin_20_nodes() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_linear_exact... ");
    if (test_cspline_linear_exact() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_runge_chebyshev_vs_equidistant... ");
    if (test_lagrange_runge_chebyshev_vs_equidistant() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_3node_quadratic... ");
    if (test_lagrange_3node_quadratic() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_exact_at_nodes... ");
    if (test_cspline_exact_at_nodes() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_exact_at_nodes... ");
    if (test_lagrange_exact_at_nodes() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_insufficient_nodes_cspline... ");
    if (test_insufficient_nodes_cspline() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_insufficient_nodes_lagrange... ");
    if (test_insufficient_nodes_lagrange() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_cspline_100_nodes_continuity... ");
    if (test_cspline_100_nodes_continuity() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("\n=== Results: %d/%d tests passed ===\n", 9 - failed, 9);

    if (failed > 0) {
        printf("interp_extended test failed\n");
    }

    return (failed > 0) ? 1 : 0;
}
