/**
 * @file test_interp_advanced.c
 * @brief 测试扩展插值功能：边界条件、PCHIP、Akima、二维插值。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define EPS_TIGHT  1e-12
#define EPS_NORMAL 1e-10
#define EPS_LOOSE  1e-4

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)

/* ======== Boundary Condition Tests ======== */

static int test_cspline_clamped_linear(void)
{
    /* A linear function with clamped BC matching the true derivative
     * should reproduce the function exactly. */
    const size_t n = 5;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {1.0, 3.0, 5.0, 7.0, 9.0}; /* y = 2x + 1 */
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    size_t i;

    st = lmmc_interp_cspline_create_ex(xs, ys, n,
        LMMC_SPLINE_CLAMPED, 2.0, 2.0, &spline);
    CHECK(st == LMMC_STATUS_OK, "clamped cspline create should succeed");

    for (i = 0; i < 20; i++) {
        lmmc_real_t x = (lmmc_real_t)i * 4.0 / 19.0;
        lmmc_real_t result, expected = 2.0 * x + 1.0;
        st = lmmc_interp_cspline_eval(spline, x, &result);
        CHECK(st == LMMC_STATUS_OK, "eval should succeed");
        CHECK(fabs(result - expected) < EPS_TIGHT,
              "clamped linear at x=%.3f: got %.12f, expected %.12f",
              x, result, expected);
    }
    lmmc_interp_cspline_destroy(spline);
    return 0;
}

static int test_cspline_not_a_knot_quadratic(void)
{
    /* A quadratic function with not-a-knot BC should be reproduced exactly
     * (cubic spline can represent quadratics exactly). */
    const size_t n = 5;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[5];
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    size_t i;

    for (i = 0; i < n; i++) ys[i] = xs[i] * xs[i]; /* y = x^2 */

    st = lmmc_interp_cspline_create_ex(xs, ys, n,
        LMMC_SPLINE_NOT_A_KNOT, 0.0, 0.0, &spline);
    CHECK(st == LMMC_STATUS_OK, "not-a-knot cspline create should succeed");

    for (i = 0; i < 20; i++) {
        lmmc_real_t x = (lmmc_real_t)i * 4.0 / 19.0;
        lmmc_real_t result, expected = x * x;
        st = lmmc_interp_cspline_eval(spline, x, &result);
        CHECK(st == LMMC_STATUS_OK, "eval should succeed");
        CHECK(fabs(result - expected) < EPS_NORMAL,
              "not-a-knot quadratic at x=%.3f: got %.12f, expected %.12f, err=%.2e",
              x, result, expected, fabs(result - expected));
    }
    lmmc_interp_cspline_destroy(spline);
    return 0;
}

static int test_cspline_periodic_sin(void)
{
    /* sin(x) on [0, 2*pi] is periodic with matching endpoints. */
    const size_t n = 21;
    lmmc_real_t xs[21], ys[21];
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;
    size_t i;

    for (i = 0; i < n; i++) {
        xs[i] = (lmmc_real_t)i * 2.0 * LMMC_CONST_PI / (lmmc_real_t)(n - 1);
        ys[i] = sin(xs[i]);
    }
    /* Force exact periodicity */
    ys[n - 1] = ys[0];

    st = lmmc_interp_cspline_create_ex(xs, ys, n,
        LMMC_SPLINE_PERIODIC, 0.0, 0.0, &spline);
    CHECK(st == LMMC_STATUS_OK, "periodic cspline create should succeed");

    for (i = 0; i < 50; i++) {
        lmmc_real_t x = (lmmc_real_t)(i + 1) * 2.0 * LMMC_CONST_PI / 51.0;
        lmmc_real_t result, expected = sin(x);
        st = lmmc_interp_cspline_eval(spline, x, &result);
        CHECK(st == LMMC_STATUS_OK, "eval should succeed");
        CHECK(fabs(result - expected) < EPS_LOOSE,
              "periodic sin at x=%.3f: got %.8f, expected %.8f, err=%.2e",
              x, result, expected, fabs(result - expected));
    }
    lmmc_interp_cspline_destroy(spline);
    return 0;
}

static int test_cspline_periodic_reject_mismatch(void)
{
    /* Periodic BC should reject if endpoints differ by > 1e-12 */
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
    lmmc_real_t ys[] = {1.0, 2.0, 3.0, 1.5}; /* ys[0] != ys[3] */
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_status_t st;

    st = lmmc_interp_cspline_create_ex(xs, ys, 4,
        LMMC_SPLINE_PERIODIC, 0.0, 0.0, &spline);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "periodic with mismatched endpoints should fail, got %d", (int)st);
    return 0;
}

/* ======== PCHIP Tests ======== */

static int test_pchip_monotone_increasing(void)
{
    /* PCHIP should preserve monotonicity of monotone data. */
    const size_t n = 6;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    lmmc_real_t ys[] = {0.0, 0.5, 1.0, 2.0, 4.0, 8.0}; /* strictly increasing */
    lmmc_interp_pchip_t* p = NULL;
    lmmc_status_t st;
    size_t i;
    lmmc_real_t prev_y;

    st = lmmc_interp_pchip_create(xs, ys, n, &p);
    CHECK(st == LMMC_STATUS_OK, "pchip create should succeed");

    /* Evaluate at many points and check monotonicity */
    st = lmmc_interp_pchip_eval(p, 0.0, &prev_y);
    CHECK(st == LMMC_STATUS_OK, "eval at 0 should succeed");

    for (i = 1; i <= 100; i++) {
        lmmc_real_t x = (lmmc_real_t)i * 5.0 / 100.0;
        lmmc_real_t y;
        st = lmmc_interp_pchip_eval(p, x, &y);
        CHECK(st == LMMC_STATUS_OK, "eval at x=%.3f should succeed", x);
        CHECK(y >= prev_y - 1e-15,
              "pchip monotonicity violated at x=%.3f: y=%.10f < prev=%.10f",
              x, y, prev_y);
        prev_y = y;
    }
    lmmc_interp_pchip_destroy(p);
    return 0;
}

static int test_pchip_exact_at_nodes(void)
{
    const size_t n = 5;
    lmmc_real_t xs[] = {0.0, 1.0, 3.0, 5.0, 7.0};
    lmmc_real_t ys[] = {1.0, 2.5, 0.5, 3.0, 2.0};
    lmmc_interp_pchip_t* p = NULL;
    lmmc_status_t st;
    size_t i;

    st = lmmc_interp_pchip_create(xs, ys, n, &p);
    CHECK(st == LMMC_STATUS_OK, "pchip create should succeed");

    for (i = 0; i < n; i++) {
        lmmc_real_t result;
        st = lmmc_interp_pchip_eval(p, xs[i], &result);
        CHECK(st == LMMC_STATUS_OK, "eval at node %zu should succeed", i);
        CHECK(fabs(result - ys[i]) < EPS_TIGHT,
              "pchip at node x=%.1f: got %.12f, expected %.12f",
              xs[i], result, ys[i]);
    }
    lmmc_interp_pchip_destroy(p);
    return 0;
}

static int test_pchip_two_points(void)
{
    lmmc_real_t xs[] = {0.0, 1.0};
    lmmc_real_t ys[] = {2.0, 5.0};
    lmmc_interp_pchip_t* p = NULL;
    lmmc_status_t st;
    lmmc_real_t result;

    st = lmmc_interp_pchip_create(xs, ys, 2, &p);
    CHECK(st == LMMC_STATUS_OK, "pchip with 2 points should succeed");

    st = lmmc_interp_pchip_eval(p, 0.5, &result);
    CHECK(st == LMMC_STATUS_OK, "eval should succeed");
    CHECK(fabs(result - 3.5) < EPS_TIGHT,
          "pchip linear at 0.5: got %.12f, expected 3.5", result);

    lmmc_interp_pchip_destroy(p);
    return 0;
}

/* ======== Akima Tests ======== */

static int test_akima_exact_at_nodes(void)
{
    const size_t n = 7;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    lmmc_real_t ys[] = {1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5};
    lmmc_interp_akima_t* a = NULL;
    lmmc_status_t st;
    size_t i;

    st = lmmc_interp_akima_create(xs, ys, n, &a);
    CHECK(st == LMMC_STATUS_OK, "akima create should succeed");

    for (i = 0; i < n; i++) {
        lmmc_real_t result;
        st = lmmc_interp_akima_eval(a, xs[i], &result);
        CHECK(st == LMMC_STATUS_OK, "eval at node %zu should succeed", i);
        CHECK(fabs(result - ys[i]) < EPS_TIGHT,
              "akima at node x=%.1f: got %.12f, expected %.12f",
              xs[i], result, ys[i]);
    }
    lmmc_interp_akima_destroy(a);
    return 0;
}

static int test_akima_linear_exact(void)
{
    /* Akima should reproduce a linear function exactly. */
    const size_t n = 6;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    lmmc_real_t ys[6];
    lmmc_interp_akima_t* a = NULL;
    lmmc_status_t st;
    size_t i;

    for (i = 0; i < n; i++) ys[i] = 2.0 * xs[i] + 1.0;

    st = lmmc_interp_akima_create(xs, ys, n, &a);
    CHECK(st == LMMC_STATUS_OK, "akima create should succeed");

    for (i = 0; i < 20; i++) {
        lmmc_real_t x = (lmmc_real_t)i * 5.0 / 19.0;
        lmmc_real_t result, expected = 2.0 * x + 1.0;
        st = lmmc_interp_akima_eval(a, x, &result);
        CHECK(st == LMMC_STATUS_OK, "eval should succeed");
        CHECK(fabs(result - expected) < EPS_NORMAL,
              "akima linear at x=%.3f: got %.12f, expected %.12f, err=%.2e",
              x, result, expected, fabs(result - expected));
    }
    lmmc_interp_akima_destroy(a);
    return 0;
}

static int test_akima_too_few_points(void)
{
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
    lmmc_real_t ys[] = {0.0, 1.0, 2.0, 3.0};
    lmmc_interp_akima_t* a = NULL;
    lmmc_status_t st;

    st = lmmc_interp_akima_create(xs, ys, 4, &a);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "akima with 4 points should fail, got %d", (int)st);
    return 0;
}

/* ======== 2D Interpolation Tests ======== */

static int test_bilinear_exact_plane(void)
{
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
            CHECK(st == LMMC_STATUS_OK, "bilinear eval should succeed");
            CHECK(fabs(result - expected) < EPS_NORMAL,
                  "bilinear at (%.2f,%.2f): got %.8f, expected %.8f, err=%.2e",
                  qx, qy, result, expected, fabs(result - expected));
        }
    }
    return 0;
}

static int test_bilinear_out_of_range(void)
{
    lmmc_real_t xs[] = {0.0, 1.0};
    lmmc_real_t ys[] = {0.0, 1.0};
    lmmc_real_t zs[] = {1.0, 2.0, 3.0, 4.0};
    lmmc_real_t result;
    lmmc_status_t st;

    st = lmmc_interp_bilinear(xs, 2, ys, 2, zs, -0.1, 0.5, &result);
    CHECK(st == LMMC_STATUS_OUT_OF_RANGE, "bilinear out of range x should fail");

    st = lmmc_interp_bilinear(xs, 2, ys, 2, zs, 0.5, 1.1, &result);
    CHECK(st == LMMC_STATUS_OUT_OF_RANGE, "bilinear out of range y should fail");
    return 0;
}

static int test_bicubic_quadratic(void)
{
    /* Bicubic should reproduce a quadratic function reasonably well. */
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
            CHECK(st == LMMC_STATUS_OK, "bicubic eval should succeed");
            /* Catmull-Rom is exact for quadratics on uniform grids */
            CHECK(fabs(result - expected) < 0.5,
                  "bicubic at (%.2f,%.2f): got %.6f, expected %.6f, err=%.2e",
                  qx, qy, result, expected, fabs(result - expected));
        }
    }
    return 0;
}

static int test_bicubic_too_few_points(void)
{
    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {0.0, 1.0, 2.0, 3.0};
    lmmc_real_t zs[12] = {0};
    lmmc_real_t result;
    lmmc_status_t st;

    st = lmmc_interp_bicubic(xs, 3, ys, 4, zs, 1.0, 1.0, &result);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "bicubic with nx=3 should fail, got %d", (int)st);
    return 0;
}

/* ======== Validation Tests ======== */

static int test_non_increasing_abscissae_rejected(void)
{
    lmmc_real_t xs[] = {0.0, 2.0, 1.0, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_interp_pchip_t* p = NULL;
    lmmc_interp_akima_t* a = NULL;
    lmmc_interp_cspline_t* s = NULL;
    lmmc_status_t st;

    st = lmmc_interp_pchip_create(xs, ys, 5, &p);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "pchip with non-increasing xs should fail");

    st = lmmc_interp_akima_create(xs, ys, 5, &a);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "akima with non-increasing xs should fail");

    st = lmmc_interp_cspline_create_ex(xs, ys, 5,
        LMMC_SPLINE_NATURAL, 0.0, 0.0, &s);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "cspline_ex with non-increasing xs should fail");
    return 0;
}

/* ======== Main ======== */

int main(void)
{
    int failed = 0;
    int total = 0;

    printf("=== Interpolation Advanced Tests ===\n\n");

#define RUN_TEST(fn) do { \
    total++; \
    printf("%-45s ", #fn "..."); \
    if (fn() == 0) printf("PASS\n"); \
    else { printf("\n"); failed++; } \
} while (0)

    RUN_TEST(test_cspline_clamped_linear);
    RUN_TEST(test_cspline_not_a_knot_quadratic);
    RUN_TEST(test_cspline_periodic_sin);
    RUN_TEST(test_cspline_periodic_reject_mismatch);
    RUN_TEST(test_pchip_monotone_increasing);
    RUN_TEST(test_pchip_exact_at_nodes);
    RUN_TEST(test_pchip_two_points);
    RUN_TEST(test_akima_exact_at_nodes);
    RUN_TEST(test_akima_linear_exact);
    RUN_TEST(test_akima_too_few_points);
    RUN_TEST(test_bilinear_exact_plane);
    RUN_TEST(test_bilinear_out_of_range);
    RUN_TEST(test_bicubic_quadratic);
    RUN_TEST(test_bicubic_too_few_points);
    RUN_TEST(test_non_increasing_abscissae_rejected);

#undef RUN_TEST

    printf("\n=== Results: %d/%d tests passed ===\n", total - failed, total);
    return (failed > 0) ? 1 : 0;
}
