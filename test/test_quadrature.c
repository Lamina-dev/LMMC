#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/* ========================================================================
 * Helper integrand functions
 * ======================================================================== */

static lmmc_real_t fn_const(lmmc_real_t x, void* ud) {
    (void)x; (void)ud;
    return 1.0;
}

static lmmc_real_t fn_x1(lmmc_real_t x, void* ud) {
    (void)ud;
    return x;
}

static lmmc_real_t fn_x2(lmmc_real_t x, void* ud) {
    (void)ud;
    return x * x;
}

static lmmc_real_t fn_x3(lmmc_real_t x, void* ud) {
    (void)ud;
    return x * x * x;
}

static lmmc_real_t fn_x4(lmmc_real_t x, void* ud) {
    (void)ud;
    return x * x * x * x;
}

static lmmc_real_t fn_sin(lmmc_real_t x, void* ud) {
    (void)ud;
    return sin(x);
}

static lmmc_real_t fn_exp(lmmc_real_t x, void* ud) {
    (void)ud;
    return exp(x);
}

static lmmc_real_t fn_inv_sqrt(lmmc_real_t x, void* ud) {
    (void)ud;
    if (x <= 0.0) return 0.0; /* avoid singularity at x=0 */
    return 1.0 / sqrt(x);
}

int main(void) {
    lmmc_status_t st;
    lmmc_real_t result;
    lmmc_quad_result_t adaptive_result;
    int rc = 0;

    /* ====================================================================
     * 1. Trapezoidal rule: polynomial convergence on [0,1]
     * Requirement 1.1: x^0..x^4 convergence verification
     * ==================================================================== */

    /* Analytical integrals of x^n on [0,1]: 1/(n+1) */
    {
        lmmc_quad_func_t poly_funcs[] = { fn_const, fn_x1, fn_x2, fn_x3, fn_x4 };
        double exact_vals[] = { 1.0, 0.5, 1.0/3.0, 0.25, 0.2 };

        for (int p = 0; p < 5; p++) {
            /* Verify convergence: error with n=100 < error with n=10 */
            lmmc_real_t res_10, res_100;
            st = lmmc_quad_trapezoid(poly_funcs[p], NULL, 0.0, 1.0, 10, &res_10);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            st = lmmc_quad_trapezoid(poly_funcs[p], NULL, 0.0, 1.0, 100, &res_100);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

            double err_10 = fabs(res_10 - exact_vals[p]);
            double err_100 = fabs(res_100 - exact_vals[p]);

            /* For constant and linear, trapezoid is exact */
            if (p <= 1) {
                if (!lmmc_test_nearly_equal(res_10, exact_vals[p], 1e-12)) {
                    rc = 1; goto done;
                }
            } else {
                /* For higher degree, error should decrease with more subintervals */
                if (err_100 >= err_10) { rc = 1; goto done; }
            }
        }
    }

    /* ====================================================================
     * 2. Simpson's rule: exact for degree <= 3 polynomials
     * Requirement 1.2: error < 1e-12 for cubic and below
     * ==================================================================== */
    {
        lmmc_quad_func_t poly_funcs[] = { fn_const, fn_x1, fn_x2, fn_x3 };
        double exact_vals[] = { 1.0, 0.5, 1.0/3.0, 0.25 };

        for (int p = 0; p < 4; p++) {
            /* Simpson with n=2 (minimum even) should be exact for degree <= 3 */
            st = lmmc_quad_simpson(poly_funcs[p], NULL, 0.0, 1.0, 2, &result);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            if (!lmmc_test_nearly_equal(result, exact_vals[p], 1e-12)) {
                rc = 1; goto done;
            }

            /* Also verify with n=4 */
            st = lmmc_quad_simpson(poly_funcs[p], NULL, 0.0, 1.0, 4, &result);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            if (!lmmc_test_nearly_equal(result, exact_vals[p], 1e-12)) {
                rc = 1; goto done;
            }
        }
    }

    /* ====================================================================
     * 3. Gauss-Legendre: sin(x) on [0, pi] accuracy
     * Requirement 1.3: higher order GL achieves error < 1e-10 vs exact = 2.0
     * Note: sin(x) is not a polynomial, so order >= 7 is needed for 1e-10
     * ==================================================================== */
    {
        double exact_sin = 2.0; /* integral of sin(x) from 0 to pi */

        /* Orders 7-10 should achieve < 1e-10 accuracy */
        for (size_t order = 7; order <= 10; order++) {
            st = lmmc_quad_gauss_legendre(fn_sin, NULL, 0.0, LMMC_CONST_PI, order, &result);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            if (!lmmc_test_nearly_equal(result, exact_sin, 1e-10)) {
                rc = 1; goto done;
            }
        }

        /* Verify that increasing order improves accuracy */
        lmmc_real_t res4, res8;
        st = lmmc_quad_gauss_legendre(fn_sin, NULL, 0.0, LMMC_CONST_PI, 4, &res4);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_quad_gauss_legendre(fn_sin, NULL, 0.0, LMMC_CONST_PI, 8, &res8);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (fabs(res8 - exact_sin) >= fabs(res4 - exact_sin)) {
            rc = 1; goto done;
        }
    }

    /* ====================================================================
     * 4. Gauss-Legendre: polynomial exactness property
     * Requirement 1.10: n-point rule exact for degree <= 2n-1
     * ==================================================================== */
    {
        /* Order 2: exact for degree <= 3 */
        st = lmmc_quad_gauss_legendre(fn_x3, NULL, 0.0, 1.0, 2, &result);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result, 0.25, 1e-11)) {
            rc = 1; goto done;
        }

        /* Order 3: exact for degree <= 5, test x^4 (degree 4) */
        st = lmmc_quad_gauss_legendre(fn_x4, NULL, 0.0, 1.0, 3, &result);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result, 0.2, 1e-11)) {
            rc = 1; goto done;
        }

        /* Order 5: exact for degree <= 9, test x^4 (degree 4) */
        st = lmmc_quad_gauss_legendre(fn_x4, NULL, 0.0, 1.0, 5, &result);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result, 0.2, 1e-11)) {
            rc = 1; goto done;
        }
    }

    /* ====================================================================
     * 5. Adaptive integration: exp(x) on [0,1]
     * Requirement 1.4: result vs exact = e - 1
     * ==================================================================== */
    {
        double exact_exp = exp(1.0) - 1.0; /* e - 1 ≈ 1.71828... */

        st = lmmc_quad_adaptive(fn_exp, NULL, 0.0, 1.0, 1e-10, 1e-10, 50, &adaptive_result);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(adaptive_result.value, exact_exp, 1e-10)) {
            rc = 1; goto done;
        }
        /* Verify function evaluations were performed */
        if (adaptive_result.num_evals == 0) { rc = 1; goto done; }
    }

    /* ====================================================================
     * 6. Adaptive integration: 1/sqrt(x) on [0,1] (integrable singularity)
     * Requirement 1.5: exact = 2.0
     * ==================================================================== */
    {
        double exact_inv_sqrt = 2.0; /* integral of 1/sqrt(x) from 0 to 1 = 2 */

        /* Use a small offset to avoid the singularity at x=0 */
        st = lmmc_quad_adaptive(fn_inv_sqrt, NULL, 1e-10, 1.0, 1e-4, 1e-4, 50, &adaptive_result);
        /* Accept OK or WARNING_MAX_DEPTH */
        if (st != LMMC_STATUS_OK && st != LMMC_STATUS_WARNING_MAX_DEPTH) {
            rc = 1; goto done;
        }
        /* The result should be close to 2.0 (with some tolerance due to singularity) */
        if (!lmmc_test_nearly_equal(adaptive_result.value, exact_inv_sqrt, 1e-2)) {
            rc = 1; goto done;
        }
    }

    /* ====================================================================
     * 7. Error handling: NULL func
     * Requirement 1.6
     * ==================================================================== */
    {
        st = lmmc_quad_trapezoid(NULL, NULL, 0.0, 1.0, 10, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_simpson(NULL, NULL, 0.0, 1.0, 2, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_gauss_legendre(NULL, NULL, 0.0, 1.0, 5, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_adaptive(NULL, NULL, 0.0, 1.0, 1e-10, 1e-10, 50, &adaptive_result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ====================================================================
     * 8. Error handling: n=0
     * Requirement 1.7
     * ==================================================================== */
    {
        st = lmmc_quad_trapezoid(fn_const, NULL, 0.0, 1.0, 0, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_simpson(fn_const, NULL, 0.0, 1.0, 0, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ====================================================================
     * 9. Error handling: a == b (equal limits)
     * Requirement 1.8: The implementation returns INVALID_ARGUMENT for a >= b
     * ==================================================================== */
    {
        st = lmmc_quad_trapezoid(fn_const, NULL, 1.0, 1.0, 10, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_simpson(fn_const, NULL, 1.0, 1.0, 2, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_gauss_legendre(fn_const, NULL, 1.0, 1.0, 5, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_adaptive(fn_const, NULL, 1.0, 1.0, 1e-10, 1e-10, 50, &adaptive_result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ====================================================================
     * 10. Error handling: reversed interval (a > b)
     * Requirement 1.9: The implementation returns INVALID_ARGUMENT for a >= b
     * ==================================================================== */
    {
        st = lmmc_quad_trapezoid(fn_const, NULL, 2.0, 1.0, 10, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_simpson(fn_const, NULL, 2.0, 1.0, 2, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_gauss_legendre(fn_const, NULL, 2.0, 1.0, 5, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_adaptive(fn_const, NULL, 2.0, 1.0, 1e-10, 1e-10, 50, &adaptive_result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ====================================================================
     * 11. Error handling: NULL out_result
     * ==================================================================== */
    {
        st = lmmc_quad_trapezoid(fn_const, NULL, 0.0, 1.0, 10, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_simpson(fn_const, NULL, 0.0, 1.0, 2, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_gauss_legendre(fn_const, NULL, 0.0, 1.0, 5, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_adaptive(fn_const, NULL, 0.0, 1.0, 1e-10, 1e-10, 50, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ====================================================================
     * 12. Simpson: odd n should be rejected
     * ==================================================================== */
    {
        st = lmmc_quad_simpson(fn_const, NULL, 0.0, 1.0, 3, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ====================================================================
     * 13. Gauss-Legendre: invalid order (< 2 or > 20)
     * ==================================================================== */
    {
        st = lmmc_quad_gauss_legendre(fn_const, NULL, 0.0, 1.0, 1, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_gauss_legendre(fn_const, NULL, 0.0, 1.0, 21, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ====================================================================
     * 14. Adaptive: negative tolerance
     * ==================================================================== */
    {
        st = lmmc_quad_adaptive(fn_const, NULL, 0.0, 1.0, -1.0, 1e-10, 50, &adaptive_result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_quad_adaptive(fn_const, NULL, 0.0, 1.0, 1e-10, -1.0, 50, &adaptive_result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

done:
    if (rc != 0) {
        printf("quadrature test failed\n");
    }
    return rc;
}
