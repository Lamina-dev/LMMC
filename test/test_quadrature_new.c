/**
 * @file test_quadrature_new.c
 * @brief Tests for new quadrature functions: Romberg, Tanh-Sinh, Gauss-Hermite, Gauss-Laguerre.
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/quadrature.h"
#include "test_common.h"

/* Test functions */
static lmmc_real_t fn_exp(lmmc_real_t x, void* ud) {
    (void)ud;
    return exp(x);
}

static lmmc_real_t fn_sin(lmmc_real_t x, void* ud) {
    (void)ud;
    return sin(x);
}

/* f(x) = 1/sqrt(x) - has endpoint singularity at x=0 */
static lmmc_real_t fn_inv_sqrt(lmmc_real_t x, void* ud) {
    (void)ud;
    if (x <= 0.0) return 0.0;
    return 1.0 / sqrt(x);
}

/* f(x) = 1 for Gauss-Hermite: integral of exp(-x^2) = sqrt(pi) */
static lmmc_real_t fn_one(lmmc_real_t x, void* ud) {
    (void)x; (void)ud;
    return 1.0;
}

/* f(x) = x^2 for Gauss-Hermite: integral of x^2*exp(-x^2) = sqrt(pi)/2 */
static lmmc_real_t fn_x2(lmmc_real_t x, void* ud) {
    (void)ud;
    return x * x;
}

/* f(x) = 1 for Gauss-Laguerre: integral of exp(-x) = 1 */
static lmmc_real_t fn_one_lag(lmmc_real_t x, void* ud) {
    (void)x; (void)ud;
    return 1.0;
}

/* f(x) = x for Gauss-Laguerre: integral of x*exp(-x) = 1 */
static lmmc_real_t fn_x_lag(lmmc_real_t x, void* ud) {
    (void)ud;
    return x;
}

int main(void) {
    lmmc_status_t st;
    lmmc_quad_result_t result;
    lmmc_real_t val;
    int rc = 0;

    printf("=== Romberg Tests ===\n");

    /* Romberg: integral of exp(x) from 0 to 1 = e - 1 */
    {
        double exact = exp(1.0) - 1.0;
        st = lmmc_quad_romberg(fn_exp, NULL, 0.0, 1.0, 1e-12, 20, &result);
        printf("Romberg exp: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
               st, result.value, exact, fabs(result.value - exact));
        if (st != LMMC_STATUS_OK) { rc = 1; }
        if (fabs(result.value - exact) > 1e-10) { rc = 1; }
    }

    /* Romberg: integral of sin(x) from 0 to pi = 2 */
    {
        double exact = 2.0;
        double pi = 3.14159265358979323846;
        st = lmmc_quad_romberg(fn_sin, NULL, 0.0, pi, 1e-12, 20, &result);
        printf("Romberg sin: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
               st, result.value, exact, fabs(result.value - exact));
        if (st != LMMC_STATUS_OK) { rc = 1; }
        if (fabs(result.value - exact) > 1e-10) { rc = 1; }
    }

    /* Romberg: invalid arguments */
    {
        st = lmmc_quad_romberg(NULL, NULL, 0.0, 1.0, 1e-10, 20, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; }
        st = lmmc_quad_romberg(fn_exp, NULL, 1.0, 0.0, 1e-10, 20, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; }
    }

    printf("\n=== Tanh-Sinh Tests ===\n");

    /* Tanh-Sinh: integral of 1/sqrt(x) from 0 to 1 = 2 */
    {
        double exact = 2.0;
        st = lmmc_quad_tanh_sinh(fn_inv_sqrt, NULL, 0.0, 1.0, 1e-10, 100000, &result);
        printf("Tanh-Sinh 1/sqrt(x): status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
               st, result.value, exact, fabs(result.value - exact));
        if (st != LMMC_STATUS_OK && st != LMMC_STATUS_CONVERGENCE_FAILED) { rc = 1; }
        if (fabs(result.value - exact) > 1e-8) { rc = 1; }
    }

    /* Tanh-Sinh: integral of exp(x) from 0 to 1 = e - 1 */
    {
        double exact = exp(1.0) - 1.0;
        st = lmmc_quad_tanh_sinh(fn_exp, NULL, 0.0, 1.0, 1e-12, 100000, &result);
        printf("Tanh-Sinh exp: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
               st, result.value, exact, fabs(result.value - exact));
        if (st != LMMC_STATUS_OK) { rc = 1; }
        if (fabs(result.value - exact) > 1e-9) { rc = 1; }
    }

    printf("\n=== Gauss-Hermite Tests ===\n");

    /* Gauss-Hermite: integral of 1 * exp(-x^2) = sqrt(pi) */
    {
        double exact = sqrt(3.14159265358979323846);
        st = lmmc_quad_gauss_hermite(fn_one, NULL, 1, &val);
        printf("GH order=1, f=1: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
               st, val, exact, fabs(val - exact));
        if (st != LMMC_STATUS_OK) { rc = 1; }
        if (fabs(val - exact) > 1e-10) { rc = 1; }
    }

    /* Gauss-Hermite: integral of x^2 * exp(-x^2) = sqrt(pi)/2 */
    {
        double exact = sqrt(3.14159265358979323846) / 2.0;
        st = lmmc_quad_gauss_hermite(fn_x2, NULL, 5, &val);
        printf("GH order=5, f=x^2: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
               st, val, exact, fabs(val - exact));
        if (st != LMMC_STATUS_OK) { rc = 1; }
        if (fabs(val - exact) > 1e-10) { rc = 1; }
    }

    /* Gauss-Hermite: invalid arguments */
    {
        st = lmmc_quad_gauss_hermite(NULL, NULL, 5, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; }
        st = lmmc_quad_gauss_hermite(fn_one, NULL, 0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; }
        st = lmmc_quad_gauss_hermite(fn_one, NULL, 21, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; }
    }

    printf("\n=== Gauss-Laguerre Tests ===\n");

    /* Gauss-Laguerre: integral of 1 * exp(-x) from 0 to inf = 1 */
    {
        double exact = 1.0;
        st = lmmc_quad_gauss_laguerre(fn_one_lag, NULL, 1, &val);
        printf("GL order=1, f=1: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
               st, val, exact, fabs(val - exact));
        if (st != LMMC_STATUS_OK) { rc = 1; }
        if (fabs(val - exact) > 1e-10) { rc = 1; }
    }

    /* Gauss-Laguerre: integral of x * exp(-x) from 0 to inf = 1 */
    {
        double exact = 1.0;
        st = lmmc_quad_gauss_laguerre(fn_x_lag, NULL, 5, &val);
        printf("GL order=5, f=x: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
               st, val, exact, fabs(val - exact));
        if (st != LMMC_STATUS_OK) { rc = 1; }
        if (fabs(val - exact) > 1e-10) { rc = 1; }
    }

    /* Gauss-Laguerre: invalid arguments */
    {
        st = lmmc_quad_gauss_laguerre(NULL, NULL, 5, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; }
        st = lmmc_quad_gauss_laguerre(fn_one_lag, NULL, 0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; }
        st = lmmc_quad_gauss_laguerre(fn_one_lag, NULL, 21, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; }
    }

    printf("\n=== Summary ===\n");
    if (rc == 0) {
        printf("All quadrature tests PASSED\n");
    } else {
        printf("Some quadrature tests FAILED\n");
    }
    return rc;
}
