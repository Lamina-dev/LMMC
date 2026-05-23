/**
 * @file test_interp_lagrange.c
 * @brief 针对 LMMC 中 interp lagrange 相关接口的单元测试。
 *
 * @internal
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


static int test_lagrange_null_args(void)
{
    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0};
    lmmc_interp_lagrange_t* lag = NULL;
    lmmc_status_t st;


    st = lmmc_interp_lagrange_create(NULL, ys, 3, &lag);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "NULL xs should fail");


    st = lmmc_interp_lagrange_create(xs, NULL, 3, &lag);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "NULL ys should fail");


    st = lmmc_interp_lagrange_create(xs, ys, 3, NULL);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "NULL out_lagrange should fail");

    return 0;
}

static int test_lagrange_too_few_points(void)
{
    lmmc_real_t xs[] = {1.0};
    lmmc_real_t ys[] = {2.0};
    lmmc_interp_lagrange_t* lag = NULL;
    lmmc_status_t st;


    st = lmmc_interp_lagrange_create(xs, ys, 0, &lag);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "n=0 should fail (need >= 1)");

    return 0;
}


static int test_lagrange_single_point(void)
{
    lmmc_real_t xs[] = {3.0};
    lmmc_real_t ys[] = {7.0};
    lmmc_interp_lagrange_t* lag = NULL;
    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-12;

    st = lmmc_interp_lagrange_create(xs, ys, 1, &lag);
    CHECK(st == LMMC_STATUS_OK, "create with 1 point should succeed");


    st = lmmc_interp_lagrange_eval(lag, 3.0, &result);
    CHECK(st == LMMC_STATUS_OK, "eval at data point should succeed");
    CHECK(lmmc_test_nearly_equal(result, 7.0, eps),
          "at x=3.0: expected 7.0, got %.15f", result);


    st = lmmc_interp_lagrange_eval(lag, 5.0, &result);
    CHECK(st == LMMC_STATUS_OK, "eval at x=5.0 should succeed");
    CHECK(lmmc_test_nearly_equal(result, 7.0, eps),
          "at x=5.0: expected 7.0, got %.15f", result);

    lmmc_interp_lagrange_destroy(lag);
    return 0;
}


static int test_lagrange_passes_through_data_points(void)
{

    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};
    lmmc_interp_lagrange_t* lag = NULL;
    lmmc_status_t st;
    lmmc_real_t result;
    size_t i;
    double eps = 1e-12;

    st = lmmc_interp_lagrange_create(xs, ys, 5, &lag);
    CHECK(st == LMMC_STATUS_OK, "create should succeed");

    for (i = 0; i < 5; i++) {
        st = lmmc_interp_lagrange_eval(lag, xs[i], &result);
        CHECK(st == LMMC_STATUS_OK, "eval at data point %zu should succeed", i);
        CHECK(lmmc_test_nearly_equal(result, ys[i], eps),
              "eval at x=%.1f: expected %.6f, got %.15f", xs[i], ys[i], result);
    }

    lmmc_interp_lagrange_destroy(lag);
    return 0;
}


static int test_lagrange_quadratic_exact(void)
{

    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {1.0, 0.0, 3.0};
    lmmc_interp_lagrange_t* lag = NULL;
    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-10;
    double x;

    st = lmmc_interp_lagrange_create(xs, ys, 3, &lag);
    CHECK(st == LMMC_STATUS_OK, "create should succeed");


    for (x = -1.0; x <= 3.0; x += 0.5) {
        lmmc_real_t expected = 2.0 * x * x - 3.0 * x + 1.0;
        st = lmmc_interp_lagrange_eval(lag, (lmmc_real_t)x, &result);
        CHECK(st == LMMC_STATUS_OK, "eval at x=%.2f should succeed", x);
        CHECK(lmmc_test_nearly_equal(result, expected, eps),
              "quadratic at x=%.2f: expected %.10f, got %.10f", x, expected, result);
    }

    lmmc_interp_lagrange_destroy(lag);
    return 0;
}

static int test_lagrange_cubic_exact(void)
{

    lmmc_real_t xs[] = {-1.0, 0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {2.0, 1.0, 0.0, 5.0};
    lmmc_interp_lagrange_t* lag = NULL;
    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-10;
    double x;

    st = lmmc_interp_lagrange_create(xs, ys, 4, &lag);
    CHECK(st == LMMC_STATUS_OK, "create should succeed");


    for (x = -1.0; x <= 2.0; x += 0.25) {
        lmmc_real_t expected = x * x * x - 2.0 * x + 1.0;
        st = lmmc_interp_lagrange_eval(lag, (lmmc_real_t)x, &result);
        CHECK(st == LMMC_STATUS_OK, "eval at x=%.2f should succeed", x);
        CHECK(lmmc_test_nearly_equal(result, expected, eps),
              "cubic at x=%.2f: expected %.10f, got %.10f", x, expected, result);
    }

    lmmc_interp_lagrange_destroy(lag);
    return 0;
}


static int test_lagrange_eval_null(void)
{
    lmmc_real_t result;
    lmmc_status_t st;

    st = lmmc_interp_lagrange_eval(NULL, 1.0, &result);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "NULL lagrange should fail");

    return 0;
}

static int test_lagrange_eval_null_out(void)
{
    lmmc_real_t xs[] = {0.0, 1.0, 2.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0};
    lmmc_interp_lagrange_t* lag = NULL;
    lmmc_status_t st;

    st = lmmc_interp_lagrange_create(xs, ys, 3, &lag);
    CHECK(st == LMMC_STATUS_OK, "create should succeed");

    st = lmmc_interp_lagrange_eval(lag, 0.5, NULL);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT, "NULL out_y should fail");

    lmmc_interp_lagrange_destroy(lag);
    return 0;
}


static int test_lagrange_destroy_null(void)
{

    lmmc_interp_lagrange_destroy(NULL);
    return 0;
}


static int test_lagrange_non_uniform_spacing(void)
{

    lmmc_real_t xs[] = {0.0, 0.5, 1.5, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 0.25, 2.25, 9.0, 16.0};
    lmmc_interp_lagrange_t* lag = NULL;
    lmmc_status_t st;
    lmmc_real_t result;
    double eps = 1e-9;

    st = lmmc_interp_lagrange_create(xs, ys, 5, &lag);
    CHECK(st == LMMC_STATUS_OK, "create should succeed");


    st = lmmc_interp_lagrange_eval(lag, 2.0, &result);
    CHECK(st == LMMC_STATUS_OK, "eval at x=2.0 should succeed");
    CHECK(lmmc_test_nearly_equal(result, 4.0, eps),
          "at x=2.0: expected 4.0, got %.15f", result);

    st = lmmc_interp_lagrange_eval(lag, 1.0, &result);
    CHECK(st == LMMC_STATUS_OK, "eval at x=1.0 should succeed");
    CHECK(lmmc_test_nearly_equal(result, 1.0, eps),
          "at x=1.0: expected 1.0, got %.15f", result);

    lmmc_interp_lagrange_destroy(lag);
    return 0;
}


int main(void)
{
    int failed = 0;

    printf("=== Lagrange Interpolation (Barycentric Form) Tests ===\n\n");

    printf("test_lagrange_null_args... ");
    if (test_lagrange_null_args() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_too_few_points... ");
    if (test_lagrange_too_few_points() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_single_point... ");
    if (test_lagrange_single_point() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_passes_through_data_points... ");
    if (test_lagrange_passes_through_data_points() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_quadratic_exact... ");
    if (test_lagrange_quadratic_exact() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_cubic_exact... ");
    if (test_lagrange_cubic_exact() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_eval_null... ");
    if (test_lagrange_eval_null() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_eval_null_out... ");
    if (test_lagrange_eval_null_out() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_destroy_null... ");
    if (test_lagrange_destroy_null() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("test_lagrange_non_uniform_spacing... ");
    if (test_lagrange_non_uniform_spacing() == 0) printf("PASS\n"); else { printf("\n"); failed++; }

    printf("\n=== Results: %d/%d tests passed ===\n", 10 - failed, 10);

    return (failed > 0) ? 1 : 0;
}
