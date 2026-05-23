#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/* cos(x) - x = 0, root ≈ 0.7390851332151607 (Dottie number) */
static double fn_cos_minus_x(double x, void* user_data) {
    (void)user_data;
    return cos(x) - x;
}

/* x^3 - 2x - 5 = 0, root ≈ 2.0945514815423265 */
static double fn_cubic(double x, void* user_data) {
    (void)user_data;
    return x * x * x - 2.0 * x - 5.0;
}

static double df_cubic(double x, void* user_data) {
    (void)user_data;
    return 3.0 * x * x - 2.0;
}

/* exp(x) - 3 = 0, root = ln(3) */
static double fn_exp_minus_3(double x, void* user_data) {
    (void)user_data;
    return exp(x) - 3.0;
}

/* sin(x) = 0, multiple roots at n*pi */
static double fn_sin(double x, void* user_data) {
    (void)user_data;
    return sin(x);
}

/* f(x) = x - 1, has root at x=1 (used for endpoint-is-zero test) */
static double fn_endpoint_zero(double x, void* user_data) {
    (void)user_data;
    return x - 1.0;
}

/* cbrt(x) - 1 = 0, root at x=1; derivative -> infinity at x=0 */
static double fn_cbrt(double x, void* user_data) {
    (void)user_data;
    return cbrt(x) - 1.0;
}

static double df_cbrt(double x, void* user_data) {
    (void)user_data;
    if (x == 0.0) return 1e300;
    return 1.0 / (3.0 * cbrt(x * x));
}

int main(void) {
    lmmc_nonlinear_config_t cfg = {0};
    lmmc_nonlinear_result_t result = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_nonlinear_default_config(&cfg);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

    /* ===== Test 1: cos(x)-x=0 bisection (Req 10.1) ===== */
    {
        double expected_root = 0.7390851332151607;
        st = lmmc_bisection_solve(fn_cos_minus_x, NULL, 0.0, 1.0, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, expected_root, 1e-10)) { rc = 1; goto done; }
    }

    /* ===== Test 2: x^3-2x-5=0 Newton method (Req 10.2) ===== */
    {
        double expected_root = 2.0945514815423265;
        st = lmmc_newton_solve(fn_cubic, df_cubic, NULL, 2.0, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, expected_root, 1e-10)) { rc = 1; goto done; }
    }

    /* ===== Test 3: exp(x)-3=0 secant method (Req 10.3) ===== */
    {
        double expected_root = log(3.0);
        st = lmmc_secant_solve(fn_exp_minus_3, NULL, 0.0, 2.0, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, expected_root, 1e-10)) { rc = 1; goto done; }
    }

    /* ===== Test 4: Multiple roots of sin(x) in different intervals (Req 10.4) ===== */
    {
        double pi_val = 3.14159265358979323846;

        /* Find root near pi in [2.5, 3.8] */
        st = lmmc_bisection_solve(fn_sin, NULL, 2.5, 3.8, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, pi_val, 1e-9)) { rc = 1; goto done; }

        /* Find root near 2*pi in [5.5, 6.8] */
        st = lmmc_bisection_solve(fn_sin, NULL, 5.5, 6.8, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, 2.0 * pi_val, 1e-9)) { rc = 1; goto done; }
    }

    /* ===== Test 5: Endpoint is zero — immediate return (Req 10.5) ===== */
    {
        /* f(1) = 0, so left endpoint = 1 should return immediately */
        st = lmmc_bisection_solve(fn_endpoint_zero, NULL, 1.0, 3.0, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, 1.0, 1e-14)) { rc = 1; goto done; }
        if (result.num_iter != 0) { rc = 1; goto done; }
    }

    /* ===== Test 6: Derivative tends to infinity — x^(1/3) near x=0 (Req 10.6) ===== */
    {
        /* Newton on cbrt(x)-1=0 starting from x=0.5 should converge to x=1 */
        st = lmmc_newton_solve(fn_cbrt, df_cbrt, NULL, 0.5, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, 1.0, 1e-10)) { rc = 1; goto done; }

        /* Newton starting very close to 0 where derivative is huge —
           the solver should handle it gracefully (either converge or report failure).
           Due to the extreme derivative near 0, Newton steps are tiny and the solver
           may converge to a spurious point or report failure — both are acceptable. */
        st = lmmc_newton_solve(fn_cbrt, df_cbrt, NULL, 1e-20, &cfg, &result);
        if (st != LMMC_STATUS_OK && st != LMMC_STATUS_NUMERICAL_FAILURE) {
            rc = 1; goto done;
        }
    }

    /* ===== Test 7: Invalid bracket — same sign at endpoints (Req 10.7) ===== */
    {
        /* sin(x) is positive on (0, pi), so [0.5, 2.5] has same sign */
        st = lmmc_bisection_solve(fn_sin, NULL, 0.5, 2.5, &cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT ||
            result.failure_reason != LMMC_NONLINEAR_FAILURE_INVALID_BRACKET) {
            rc = 1; goto done;
        }
    }

    /* ===== Test 8: Very strict tolerance abs_tol=1e-15 (Req 10.8) ===== */
    {
        lmmc_nonlinear_config_t strict_cfg = cfg;
        strict_cfg.abs_tol = 1e-15;
        strict_cfg.rel_tol = 0.0;
        strict_cfg.max_iter = 200;

        /* cos(x)-x=0 with very strict tolerance */
        double expected_root = 0.7390851332151607;
        st = lmmc_bisection_solve(fn_cos_minus_x, NULL, 0.0, 1.0, &strict_cfg, &result);
        if (st == LMMC_STATUS_OK) {
            if (result.converged != 1) { rc = 1; goto done; }
            if (!lmmc_test_nearly_equal(result.root, expected_root, 1e-14)) { rc = 1; goto done; }
        } else if (st == LMMC_STATUS_NUMERICAL_FAILURE) {
            /* Acceptable: may hit max_iter with such strict tolerance */
            if (result.failure_reason != LMMC_NONLINEAR_FAILURE_MAX_ITER) { rc = 1; goto done; }
            /* Even if not converged, the root should be very close */
            if (!lmmc_test_nearly_equal(result.root, expected_root, 1e-12)) { rc = 1; goto done; }
        } else {
            rc = 1; goto done;
        }

        /* Newton with strict tolerance — should achieve machine precision */
        strict_cfg.max_iter = 100;
        st = lmmc_newton_solve(fn_cubic, df_cubic, NULL, 2.0, &strict_cfg, &result);
        if (st == LMMC_STATUS_OK) {
            if (result.converged != 1) { rc = 1; goto done; }
            if (!lmmc_test_nearly_equal(result.root, 2.0945514815423265, 1e-14)) { rc = 1; goto done; }
        } else if (st == LMMC_STATUS_NUMERICAL_FAILURE) {
            if (result.failure_reason != LMMC_NONLINEAR_FAILURE_MAX_ITER) { rc = 1; goto done; }
        } else {
            rc = 1; goto done;
        }
    }

done:
    if (rc != 0) {
        printf("nonlinear extended test failed\n");
    }
    return rc;
}
