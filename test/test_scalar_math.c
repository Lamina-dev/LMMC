/**
 * @file test_scalar_math.c
 * @brief Property-based and unit tests for scalar math wrappers
 *        (inverse trig, hyperbolic, power/rounding).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define PBT_ITERATIONS 100

static int test_failures = 0;
static int test_count = 0;

#define REPORT(name, result) do { \
    test_count++; \
    if (result) { \
        printf("  FAIL: %s\n", name); \
        test_failures++; \
    } else { \
        printf("  PASS: %s\n", name); \
    } \
} while (0)

static double rand_double(double lo, double hi)
{
    return ((double)rand() / RAND_MAX) * (hi - lo) + lo;
}

static int test_property18_inverse_trig_roundtrip(void)
{
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        double x = rand_double(-1.0, 1.0);
        lmmc_real_t asin_x, acos_x;
        lmmc_real_t sin_result, cos_result;
        lmmc_status_t st;

        /* asin round-trip: sin(asin(x)) ~ x */
        st = lmmc_asin(x, &asin_x);
        if (st != LMMC_STATUS_OK) {
            printf("    asin(%g) failed at iteration %d, status=%d\n", x, i, (int)st);
            return 1;
        }
        sin_result = sin(asin_x);
        if (!lmmc_test_nearly_equal(sin_result, x, eps)) {
            printf("    sin(asin(%g)) = %g, expected %g at iter %d\n",
                   x, sin_result, x, i);
            return 1;
        }

        /* acos round-trip: cos(acos(x)) ~ x */
        st = lmmc_acos(x, &acos_x);
        if (st != LMMC_STATUS_OK) {
            printf("    acos(%g) failed at iteration %d, status=%d\n", x, i, (int)st);
            return 1;
        }
        cos_result = cos(acos_x);
        if (!lmmc_test_nearly_equal(cos_result, x, eps)) {
            printf("    cos(acos(%g)) = %g, expected %g at iter %d\n",
                   x, cos_result, x, i);
            return 1;
        }
    }
    return 0;
}

static int test_property19_hyperbolic_roundtrips(void)
{
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        /* sinh(asinh(x)) ~ x for x in [-10, 10] */
        double x = rand_double(-10.0, 10.0);
        lmmc_real_t asinh_x, sinh_result;
        lmmc_status_t st;

        st = lmmc_asinh(x, &asinh_x);
        if (st != LMMC_STATUS_OK) {
            printf("    asinh(%g) failed at iteration %d, status=%d\n", x, i, (int)st);
            return 1;
        }

        st = lmmc_sinh(asinh_x, &sinh_result);
        if (st != LMMC_STATUS_OK) {
            printf("    sinh(%g) failed at iteration %d, status=%d\n", asinh_x, i, (int)st);
            return 1;
        }

        if (!lmmc_test_nearly_equal(sinh_result, x, eps)) {
            printf("    sinh(asinh(%g)) = %g, expected %g at iter %d\n",
                   x, sinh_result, x, i);
            return 1;
        }

        /* cosh(acosh(x)) ~ x for x in [1, 10] */
        {
            double y = rand_double(1.0, 10.0);
            lmmc_real_t acosh_y, cosh_result;

            st = lmmc_acosh(y, &acosh_y);
            if (st != LMMC_STATUS_OK) {
                printf("    acosh(%g) failed at iteration %d, status=%d\n", y, i, (int)st);
                return 1;
            }

            st = lmmc_cosh(acosh_y, &cosh_result);
            if (st != LMMC_STATUS_OK) {
                printf("    cosh(%g) failed at iteration %d, status=%d\n", acosh_y, i, (int)st);
                return 1;
            }

            if (!lmmc_test_nearly_equal(cosh_result, y, eps)) {
                printf("    cosh(acosh(%g)) = %g, expected %g at iter %d\n",
                       y, cosh_result, y, i);
                return 1;
            }
        }
    }
    return 0;
}

static int test_property20_floor_ceil_invariants(void)
{
    int i;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        double x = rand_double(-100.0, 100.0);
        lmmc_real_t f, c;
        lmmc_status_t st;

        st = lmmc_floor(x, &f);
        if (st != LMMC_STATUS_OK) {
            printf("    floor(%g) failed at iteration %d, status=%d\n", x, i, (int)st);
            return 1;
        }

        st = lmmc_ceil(x, &c);
        if (st != LMMC_STATUS_OK) {
            printf("    ceil(%g) failed at iteration %d, status=%d\n", x, i, (int)st);
            return 1;
        }

        /* floor(x) <= x */
        if (f > x) {
            printf("    floor(%g) = %g > x at iter %d\n", x, f, i);
            return 1;
        }

        /* ceil(x) >= x */
        if (c < x) {
            printf("    ceil(%g) = %g < x at iter %d\n", x, c, i);
            return 1;
        }

        /* ceil(x) - floor(x) <= 1 */
        if ((c - f) > 1.0 + 1e-15) {
            printf("    ceil(%g) - floor(%g) = %g > 1 at iter %d\n",
                   x, x, c - f, i);
            return 1;
        }
    }
    return 0;
}

/* ---- Unit Tests ---- */

/* Domain errors */
static int test_unit_domain_errors(void)
{
    lmmc_real_t out;
    lmmc_status_t st;

    /* asin(2.0) -> OUT_OF_RANGE */
    st = lmmc_asin(2.0, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        printf("    asin(2.0) expected OUT_OF_RANGE, got %d\n", (int)st);
        return 1;
    }

    /* asin(-2.0) -> OUT_OF_RANGE */
    st = lmmc_asin(-2.0, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        printf("    asin(-2.0) expected OUT_OF_RANGE, got %d\n", (int)st);
        return 1;
    }

    /* acos(2.0) -> OUT_OF_RANGE */
    st = lmmc_acos(2.0, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        printf("    acos(2.0) expected OUT_OF_RANGE, got %d\n", (int)st);
        return 1;
    }

    /* acosh(0.5) -> OUT_OF_RANGE */
    st = lmmc_acosh(0.5, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        printf("    acosh(0.5) expected OUT_OF_RANGE, got %d\n", (int)st);
        return 1;
    }

    /* atanh(1.0) -> OUT_OF_RANGE */
    st = lmmc_atanh(1.0, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        printf("    atanh(1.0) expected OUT_OF_RANGE, got %d\n", (int)st);
        return 1;
    }

    /* atanh(-1.0) -> OUT_OF_RANGE */
    st = lmmc_atanh(-1.0, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        printf("    atanh(-1.0) expected OUT_OF_RANGE, got %d\n", (int)st);
        return 1;
    }

    /* pow(-2.0, 0.5) -> OUT_OF_RANGE (negative base, non-integer exponent) */
    st = lmmc_pow(-2.0, 0.5, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        printf("    pow(-2.0, 0.5) expected OUT_OF_RANGE, got %d\n", (int)st);
        return 1;
    }

    return 0;
}

/* Boundary values */
static int test_unit_boundary_values(void)
{
    lmmc_real_t out;
    lmmc_status_t st;
    double eps = 1e-12;
    double pi_half = LMMC_CONST_PI / 2.0;

    /* asin(0) = 0 */
    st = lmmc_asin(0.0, &out);
    if (st != LMMC_STATUS_OK) {
        printf("    asin(0) failed, status=%d\n", (int)st);
        return 1;
    }
    if (!lmmc_test_nearly_equal(out, 0.0, eps)) {
        printf("    asin(0) = %g, expected 0\n", out);
        return 1;
    }

    /* acos(1) = 0 */
    st = lmmc_acos(1.0, &out);
    if (st != LMMC_STATUS_OK) {
        printf("    acos(1) failed, status=%d\n", (int)st);
        return 1;
    }
    if (!lmmc_test_nearly_equal(out, 0.0, eps)) {
        printf("    acos(1) = %g, expected 0\n", out);
        return 1;
    }

    /* asin(1) = pi/2 */
    st = lmmc_asin(1.0, &out);
    if (st != LMMC_STATUS_OK) {
        printf("    asin(1) failed, status=%d\n", (int)st);
        return 1;
    }
    if (!lmmc_test_nearly_equal(out, pi_half, eps)) {
        printf("    asin(1) = %g, expected %g\n", out, pi_half);
        return 1;
    }

    /* asin(-1) = -pi/2 */
    st = lmmc_asin(-1.0, &out);
    if (st != LMMC_STATUS_OK) {
        printf("    asin(-1) failed, status=%d\n", (int)st);
        return 1;
    }
    if (!lmmc_test_nearly_equal(out, -pi_half, eps)) {
        printf("    asin(-1) = %g, expected %g\n", out, -pi_half);
        return 1;
    }

    /* acosh(1) = 0 */
    st = lmmc_acosh(1.0, &out);
    if (st != LMMC_STATUS_OK) {
        printf("    acosh(1) failed, status=%d\n", (int)st);
        return 1;
    }
    if (!lmmc_test_nearly_equal(out, 0.0, eps)) {
        printf("    acosh(1) = %g, expected 0\n", out);
        return 1;
    }

    /* atanh(0) = 0 */
    st = lmmc_atanh(0.0, &out);
    if (st != LMMC_STATUS_OK) {
        printf("    atanh(0) failed, status=%d\n", (int)st);
        return 1;
    }
    if (!lmmc_test_nearly_equal(out, 0.0, eps)) {
        printf("    atanh(0) = %g, expected 0\n", out);
        return 1;
    }

    /* floor(2.7) = 2.0 */
    st = lmmc_floor(2.7, &out);
    if (st != LMMC_STATUS_OK) {
        printf("    floor(2.7) failed, status=%d\n", (int)st);
        return 1;
    }
    if (!lmmc_test_nearly_equal(out, 2.0, eps)) {
        printf("    floor(2.7) = %g, expected 2.0\n", out);
        return 1;
    }

    /* ceil(2.3) = 3.0 */
    st = lmmc_ceil(2.3, &out);
    if (st != LMMC_STATUS_OK) {
        printf("    ceil(2.3) failed, status=%d\n", (int)st);
        return 1;
    }
    if (!lmmc_test_nearly_equal(out, 3.0, eps)) {
        printf("    ceil(2.3) = %g, expected 3.0\n", out);
        return 1;
    }

    /* pow(2.0, 3.0) = 8.0 */
    st = lmmc_pow(2.0, 3.0, &out);
    if (st != LMMC_STATUS_OK) {
        printf("    pow(2,3) failed, status=%d\n", (int)st);
        return 1;
    }
    if (!lmmc_test_nearly_equal(out, 8.0, eps)) {
        printf("    pow(2,3) = %g, expected 8.0\n", out);
        return 1;
    }

    /* pow(-2.0, 3.0) = -8.0 (negative base, integer exponent is OK) */
    st = lmmc_pow(-2.0, 3.0, &out);
    if (st != LMMC_STATUS_OK) {
        printf("    pow(-2,3) failed, status=%d\n", (int)st);
        return 1;
    }
    if (!lmmc_test_nearly_equal(out, -8.0, eps)) {
        printf("    pow(-2,3) = %g, expected -8.0\n", out);
        return 1;
    }

    return 0;
}

/* NULL pointer checks */
static int test_unit_null_pointer(void)
{
    lmmc_status_t st;

    /* asin with NULL out */
    st = lmmc_asin(0.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    asin(0.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* acos with NULL out */
    st = lmmc_acos(0.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    acos(0.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* atan with NULL out */
    st = lmmc_atan(0.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    atan(0.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* sinh with NULL out */
    st = lmmc_sinh(1.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    sinh(1.0, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* cosh with NULL out */
    st = lmmc_cosh(1.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    cosh(1.0, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* tanh with NULL out */
    st = lmmc_tanh(1.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    tanh(1.0, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* asinh with NULL out */
    st = lmmc_asinh(1.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    asinh(1.0, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* acosh with NULL out */
    st = lmmc_acosh(2.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    acosh(2.0, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* atanh with NULL out */
    st = lmmc_atanh(0.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    atanh(0.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* pow with NULL out */
    st = lmmc_pow(2.0, 3.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    pow(2,3,NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* ceil with NULL out */
    st = lmmc_ceil(1.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    ceil(1.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* floor with NULL out */
    st = lmmc_floor(1.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    floor(1.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* round with NULL out */
    st = lmmc_round(1.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    round(1.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* trunc with NULL out */
    st = lmmc_trunc(1.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    trunc(1.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    return 0;
}

/* ---- Main ---- */

int main(void)
{
    srand(12345);
    if (lmmc_init() != LMMC_STATUS_OK) return 1;

    printf("=== Scalar Math Property Tests ===\n");
    REPORT("Property 18: Inverse trigonometric round-trip",
           test_property18_inverse_trig_roundtrip());
    REPORT("Property 19: Hyperbolic function round-trips",
           test_property19_hyperbolic_roundtrips());
    REPORT("Property 20: Floor and ceil invariants",
           test_property20_floor_ceil_invariants());

    printf("\n=== Scalar Math Unit Tests ===\n");
    REPORT("Unit: Domain errors",
           test_unit_domain_errors());
    REPORT("Unit: Boundary values",
           test_unit_boundary_values());
    REPORT("Unit: NULL pointer checks",
           test_unit_null_pointer());

    printf("\n=== Results: %d/%d passed ===\n",
           test_count - test_failures, test_count);

    if (lmmc_deinit() != LMMC_STATUS_OK) return 1;
    return test_failures > 0 ? 1 : 0;
}
