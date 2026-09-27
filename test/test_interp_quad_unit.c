#include <stdlib.h>
/**
 * @file test_interp_quad_unit.c
 * 插值（PCHIP 单调性）与积分（Tanh-Sinh、Romberg）单元测试。
 */
#include <math.h>
#include <stdio.h>

#include "lmmc/interp.h"
#include "lmmc/quadrature.h"
#include "test_common.h"

/* ========================================================================
 * Test 1: PCHIP monotonicity preservation on monotone increasing data
 * ======================================================================== */

typedef struct {
    lmmc_interp_pchip_t *interp_pchip_p;
} test_fixture_t;

static int setup(void **state) {
    test_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_interp_pchip_destroy(fixture->interp_pchip_p);
    free(fixture);
    return 0;
}

static void test_pchip_monotonicity(void **state) {
    test_fixture_t *fixture = *state;
    /* Create monotone increasing data with varying slopes */
    const size_t n = 10;
    lmmc_real_t xs[] = {0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 7.0, 8.5, 10.0};
    lmmc_real_t ys[] = {0.0, 0.1, 0.3, 0.8, 1.5, 2.5, 4.0, 5.5, 7.0, 10.0};

    lmmc_status_t st;
    size_t i;
    lmmc_real_t prev_y;
    const size_t num_eval = 500; /* many intermediate points */

    st = lmmc_interp_pchip_create(xs, ys, n, &fixture->interp_pchip_p);
    assert_true(st == LMMC_STATUS_OK);

    /* Evaluate at first point */
    st = lmmc_interp_pchip_eval(fixture->interp_pchip_p, xs[0], &prev_y);
    assert_true(st == LMMC_STATUS_OK);

    /* Evaluate at many intermediate points and verify monotonicity */
    for (i = 1; i <= num_eval; i++) {
        lmmc_real_t x = xs[0] + (lmmc_real_t)i * (xs[n - 1] - xs[0]) / (lmmc_real_t)num_eval;
        lmmc_real_t y;
        st = lmmc_interp_pchip_eval(fixture->interp_pchip_p, x, &y);
        assert_true(st == LMMC_STATUS_OK);
        assert_true(y >= prev_y - 1e-15);
        prev_y = y;
    }

    lmmc_interp_pchip_destroy(fixture->interp_pchip_p);
    fixture->interp_pchip_p = NULL;
}

/* ========================================================================
 * Test 2: Tanh-Sinh on endpoint singularity integral x^{-0.5} from 0 to 1
 * ======================================================================== */

static lmmc_real_t fn_inv_sqrt(lmmc_real_t x, void *ud) {
    (void)ud;
    if (x <= 0.0)
        return 0.0;
    return 1.0 / sqrt(x);
}

static void test_tanh_sinh_endpoint_singularity(void **state) {
    (void)state;
    /* integral of x^{-0.5} from 0 to 1 = 2*sqrt(1) - 2*sqrt(0) = 2.0
     * This is a challenging integral with an endpoint singularity at x=0.
     * The Tanh-Sinh method handles it well, achieving ~1e-8 accuracy. */
    const lmmc_real_t exact = 2.0;
    const lmmc_real_t tol = 1e-8;
    lmmc_quad_result_t result;
    lmmc_status_t st;

    st = lmmc_quad_tanh_sinh(fn_inv_sqrt, NULL, 0.0, 1.0, 1e-10, 1000000, &result);
    assert_true(st == LMMC_STATUS_OK || st == LMMC_STATUS_CONVERGENCE_FAILED);

    printf("    Tanh-Sinh result: %.15f, exact: %.15f, error: %.2e\n",
           result.value, exact, fabs(result.value - exact));

    assert_true(fabs(result.value - exact) <= tol);
}

/* ========================================================================
 * Test 3: Romberg on smooth function exp(x) from 0 to 1
 * ======================================================================== */

static lmmc_real_t fn_exp(lmmc_real_t x, void *ud) {
    (void)ud;
    return exp(x);
}

static void test_romberg_smooth_exp(void **state) {
    (void)state;
    /* integral of exp(x) from 0 to 1 = e - 1 */
    const lmmc_real_t exact = exp(1.0) - 1.0;
    const lmmc_real_t tol = 1e-10;
    lmmc_quad_result_t result;
    lmmc_status_t st;

    st = lmmc_quad_romberg(fn_exp, NULL, 0.0, 1.0, 1e-12, 30, &result);
    assert_true(st == LMMC_STATUS_OK);

    printf("    Romberg result: %.15f, exact: %.15f, error: %.2e\n",
           result.value, exact, fabs(result.value - exact));

    assert_true(fabs(result.value - exact) <= tol);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_pchip_monotonicity, setup, teardown),
        cmocka_unit_test(test_tanh_sinh_endpoint_singularity),
        cmocka_unit_test(test_romberg_smooth_exp),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
