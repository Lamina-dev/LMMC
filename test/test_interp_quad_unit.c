/**
 * @file test_interp_quad_unit.c
 * @brief Unit tests for interpolation (PCHIP monotonicity) and quadrature
 *        (Tanh-Sinh endpoint singularity, Romberg smooth function).
 *
 * Validates: Requirements 17.6, 17.7
 */
#include <math.h>
#include <stdio.h>

#include "lmmc/interp.h"
#include "lmmc/quadrature.h"
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
 * Test 1: PCHIP monotonicity preservation on monotone increasing data
 * ======================================================================== */

static int test_pchip_monotonicity(void)
{
    /* Create monotone increasing data with varying slopes */
    const size_t n = 10;
    lmmc_real_t xs[] = {0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 7.0, 8.5, 10.0};
    lmmc_real_t ys[] = {0.0, 0.1, 0.3, 0.8, 1.5, 2.5, 4.0, 5.5, 7.0, 10.0};
    lmmc_interp_pchip_t* p = NULL;
    lmmc_status_t st;
    size_t i;
    lmmc_real_t prev_y;
    const size_t num_eval = 500; /* many intermediate points */

    st = lmmc_interp_pchip_create(xs, ys, n, &p);
    CHECK(st == LMMC_STATUS_OK, "pchip create should succeed, got %d", (int)st);

    /* Evaluate at first point */
    st = lmmc_interp_pchip_eval(p, xs[0], &prev_y);
    CHECK(st == LMMC_STATUS_OK, "eval at x=0 should succeed");

    /* Evaluate at many intermediate points and verify monotonicity */
    for (i = 1; i <= num_eval; i++) {
        lmmc_real_t x = xs[0] + (lmmc_real_t)i * (xs[n - 1] - xs[0]) / (lmmc_real_t)num_eval;
        lmmc_real_t y;
        st = lmmc_interp_pchip_eval(p, x, &y);
        CHECK(st == LMMC_STATUS_OK, "eval at x=%.4f should succeed", x);
        CHECK(y >= prev_y - 1e-15,
              "PCHIP monotonicity violated at x=%.6f: y=%.12f < prev_y=%.12f (diff=%.2e)",
              x, y, prev_y, prev_y - y);
        prev_y = y;
    }

    lmmc_interp_pchip_destroy(p);
    return 0;
}

/* ========================================================================
 * Test 2: Tanh-Sinh on endpoint singularity integral x^{-0.5} from 0 to 1
 * ======================================================================== */

static lmmc_real_t fn_inv_sqrt(lmmc_real_t x, void* ud)
{
    (void)ud;
    if (x <= 0.0) return 0.0;
    return 1.0 / sqrt(x);
}

static int test_tanh_sinh_endpoint_singularity(void)
{
    /* integral of x^{-0.5} from 0 to 1 = 2*sqrt(1) - 2*sqrt(0) = 2.0
     * This is a challenging integral with an endpoint singularity at x=0.
     * The Tanh-Sinh method handles it well, achieving ~1e-8 accuracy. */
    const lmmc_real_t exact = 2.0;
    const lmmc_real_t tol = 1e-8;
    lmmc_quad_result_t result;
    lmmc_status_t st;

    st = lmmc_quad_tanh_sinh(fn_inv_sqrt, NULL, 0.0, 1.0, 1e-10, 1000000, &result);
    CHECK(st == LMMC_STATUS_OK || st == LMMC_STATUS_CONVERGENCE_FAILED,
          "tanh_sinh should return OK or convergence status, got %d", (int)st);

    printf("    Tanh-Sinh result: %.15f, exact: %.15f, error: %.2e\n",
           result.value, exact, fabs(result.value - exact));

    CHECK(fabs(result.value - exact) <= tol,
          "Tanh-Sinh integral of x^{-0.5} on [0,1]: error %.2e exceeds tol %.2e",
          fabs(result.value - exact), tol);

    return 0;
}

/* ========================================================================
 * Test 3: Romberg on smooth function exp(x) from 0 to 1
 * ======================================================================== */

static lmmc_real_t fn_exp(lmmc_real_t x, void* ud)
{
    (void)ud;
    return exp(x);
}

static int test_romberg_smooth_exp(void)
{
    /* integral of exp(x) from 0 to 1 = e - 1 */
    const lmmc_real_t exact = exp(1.0) - 1.0;
    const lmmc_real_t tol = 1e-10;
    lmmc_quad_result_t result;
    lmmc_status_t st;

    st = lmmc_quad_romberg(fn_exp, NULL, 0.0, 1.0, 1e-12, 30, &result);
    CHECK(st == LMMC_STATUS_OK,
          "romberg should return OK, got %d", (int)st);

    printf("    Romberg result: %.15f, exact: %.15f, error: %.2e\n",
           result.value, exact, fabs(result.value - exact));

    CHECK(fabs(result.value - exact) <= tol,
          "Romberg integral of exp(x) on [0,1]: error %.2e exceeds tol %.2e",
          fabs(result.value - exact), tol);

    return 0;
}

/* ======== Main ======== */

int main(void)
{
    int failed = 0;
    int total = 0;

    printf("=== Interpolation & Quadrature Unit Tests ===\n\n");

#define RUN_TEST(fn) do { \
    total++; \
    printf("%-50s ", #fn "..."); \
    if (fn() == 0) printf("PASS\n"); \
    else { printf("\n"); failed++; } \
} while (0)

    RUN_TEST(test_pchip_monotonicity);
    RUN_TEST(test_tanh_sinh_endpoint_singularity);
    RUN_TEST(test_romberg_smooth_exp);

#undef RUN_TEST

    printf("\n=== Results: %d/%d tests passed ===\n", total - failed, total);
    return (failed > 0) ? 1 : 0;
}
