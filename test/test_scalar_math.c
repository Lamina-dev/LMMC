/**
 * @file test_scalar_math.c
 * @brief Property-based and unit tests for scalar math wrappers
 *        (inverse trig, hyperbolic, power/rounding).
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define PBT_ITERATIONS 100

static int setup(void **state) {
    (void)state;
    assert_int_equal(lmmc_init(), LMMC_STATUS_OK);
    srand(12345);
    return 0;
}

static int teardown(void **state) {
    (void)state;
    assert_int_equal(lmmc_deinit(), LMMC_STATUS_OK);
    return 0;
}

static double rand_double(double lo, double hi) {
    return ((double)rand() / RAND_MAX) * (hi - lo) + lo;
}

static void test_property18_inverse_trig_roundtrip(void **state) {
    (void)state;
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
            fail_msg("    asin(%g) failed at iteration %d, status=%d\n", x, i, (int)st);
        }
        sin_result = sin(asin_x);
        if (!lmmc_test_nearly_equal(sin_result, x, eps)) {
            fail_msg("    sin(asin(%g)) = %g, expected %g at iter %d\n", x, sin_result, x, i);
        }

        /* acos round-trip: cos(acos(x)) ~ x */
        st = lmmc_acos(x, &acos_x);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    acos(%g) failed at iteration %d, status=%d\n", x, i, (int)st);
        }
        cos_result = cos(acos_x);
        if (!lmmc_test_nearly_equal(cos_result, x, eps)) {
            fail_msg("    cos(acos(%g)) = %g, expected %g at iter %d\n", x, cos_result, x, i);
        }
    }
}

static void test_property19_hyperbolic_roundtrips(void **state) {
    (void)state;
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        /* sinh(asinh(x)) ~ x for x in [-10, 10] */
        double x = rand_double(-10.0, 10.0);
        lmmc_real_t asinh_x, sinh_result;
        lmmc_status_t st;

        st = lmmc_asinh(x, &asinh_x);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    asinh(%g) failed at iteration %d, status=%d\n", x, i, (int)st);
        }

        st = lmmc_sinh(asinh_x, &sinh_result);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    sinh(%g) failed at iteration %d, status=%d\n", asinh_x, i, (int)st);
        }

        if (!lmmc_test_nearly_equal(sinh_result, x, eps)) {
            fail_msg("    sinh(asinh(%g)) = %g, expected %g at iter %d\n", x, sinh_result, x, i);
        }

        /* cosh(acosh(x)) ~ x for x in [1, 10] */
        {
            double y = rand_double(1.0, 10.0);
            lmmc_real_t acosh_y, cosh_result;

            st = lmmc_acosh(y, &acosh_y);
            if (st != LMMC_STATUS_OK) {
                fail_msg("    acosh(%g) failed at iteration %d, status=%d\n", y, i, (int)st);
            }

            st = lmmc_cosh(acosh_y, &cosh_result);
            if (st != LMMC_STATUS_OK) {
                fail_msg("    cosh(%g) failed at iteration %d, status=%d\n", acosh_y, i, (int)st);
            }

            if (!lmmc_test_nearly_equal(cosh_result, y, eps)) {
                fail_msg("    cosh(acosh(%g)) = %g, expected %g at iter %d\n", y, cosh_result, y, i);
            }
        }
    }
}

static void test_property20_floor_ceil_invariants(void **state) {
    (void)state;
    int i;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        double x = rand_double(-100.0, 100.0);
        lmmc_real_t f, c;
        lmmc_status_t st;

        st = lmmc_floor(x, &f);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    floor(%g) failed at iteration %d, status=%d\n", x, i, (int)st);
        }

        st = lmmc_ceil(x, &c);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    ceil(%g) failed at iteration %d, status=%d\n", x, i, (int)st);
        }

        /* floor(x) <= x */
        if (!(f <= x)) {
            fail_msg("    floor(%g) = %g > x at iter %d\n", x, f, i);
        }

        /* ceil(x) >= x */
        if (!(c >= x)) {
            fail_msg("    ceil(%g) = %g < x at iter %d\n", x, c, i);
        }

        /* ceil(x) - floor(x) <= 1 */
        if (!((c - f) <= 1.0 + 1e-15)) {
            fail_msg("    ceil(%g) - floor(%g) = %g > 1 at iter %d\n", x, x, c - f, i);
        }
    }
}

/* ---- Unit Tests ---- */

/* Domain errors */
static void test_inverse_trig_domain(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t out;
    /* asin(2.0) -> OUT_OF_RANGE */
    st = lmmc_asin(2.0, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        fail_msg("    asin(2.0) expected OUT_OF_RANGE, got %d\n", (int)st);
    }

    /* asin(-2.0) -> OUT_OF_RANGE */
    st = lmmc_asin(-2.0, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        fail_msg("    asin(-2.0) expected OUT_OF_RANGE, got %d\n", (int)st);
    }

    /* acos(2.0) -> OUT_OF_RANGE */
    st = lmmc_acos(2.0, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        fail_msg("    acos(2.0) expected OUT_OF_RANGE, got %d\n", (int)st);
    }
}

static void test_inverse_hyperbolic_domain(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t out;
    /* acosh(0.5) -> OUT_OF_RANGE */
    st = lmmc_acosh(0.5, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        fail_msg("    acosh(0.5) expected OUT_OF_RANGE, got %d\n", (int)st);
    }

    /* atanh(1.0) -> OUT_OF_RANGE */
    st = lmmc_atanh(1.0, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        fail_msg("    atanh(1.0) expected OUT_OF_RANGE, got %d\n", (int)st);
    }

    /* atanh(-1.0) -> OUT_OF_RANGE */
    st = lmmc_atanh(-1.0, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        fail_msg("    atanh(-1.0) expected OUT_OF_RANGE, got %d\n", (int)st);
    }
}

static void test_power_domain_and_range(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t out;
    /* pow(-2.0, 0.5) -> OUT_OF_RANGE (negative base, non-integer exponent) */
    st = lmmc_pow(-2.0, 0.5, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        fail_msg("    pow(-2.0, 0.5) expected OUT_OF_RANGE, got %d\n", (int)st);
    }

    out = 123.0;
    st = lmmc_pow(0.0, -1.0, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE || out != 123.0) {
        fail_msg("    pow(0,-1) must report a pole without writing output\n");
    }

    out = 123.0;
    st = lmmc_pow(DBL_MAX, 2.0, &out);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || out != 123.0) {
        fail_msg("    overflowing pow must report numerical failure\n");
    }
}

static void test_scalar_nonfinite_errors(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t out;
    out = 123.0;
    st = lmmc_sinh(DBL_MAX, &out);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || out != 123.0) {
        fail_msg("    overflowing sinh must report numerical failure\n");
    }

    out = 123.0;
    st = lmmc_asin(NAN, &out);
    if (st != LMMC_STATUS_INVALID_ARGUMENT || out != 123.0) {
        fail_msg("    asin must reject NaN without writing output\n");
    }
}

/* Boundary values */
static void test_inverse_trig_boundaries(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t out;
    const double eps = 1e-12;
    const double pi_half = LMMC_CONST_PI / 2.0;
    /* asin(0) = 0 */
    st = lmmc_asin(0.0, &out);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    asin(0) failed, status=%d\n", (int)st);
    }
    if (!lmmc_test_nearly_equal(out, 0.0, eps)) {
        fail_msg("    asin(0) = %g, expected 0\n", out);
    }

    /* acos(1) = 0 */
    st = lmmc_acos(1.0, &out);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    acos(1) failed, status=%d\n", (int)st);
    }
    if (!lmmc_test_nearly_equal(out, 0.0, eps)) {
        fail_msg("    acos(1) = %g, expected 0\n", out);
    }

    /* asin(1) = pi/2 */
    st = lmmc_asin(1.0, &out);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    asin(1) failed, status=%d\n", (int)st);
    }
    if (!lmmc_test_nearly_equal(out, pi_half, eps)) {
        fail_msg("    asin(1) = %g, expected %g\n", out, pi_half);
    }

    /* asin(-1) = -pi/2 */
    st = lmmc_asin(-1.0, &out);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    asin(-1) failed, status=%d\n", (int)st);
    }
    if (!lmmc_test_nearly_equal(out, -pi_half, eps)) {
        fail_msg("    asin(-1) = %g, expected %g\n", out, -pi_half);
    }
}

static void test_inverse_hyperbolic_boundaries(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t out;
    const double eps = 1e-12;
    /* acosh(1) = 0 */
    st = lmmc_acosh(1.0, &out);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    acosh(1) failed, status=%d\n", (int)st);
    }
    if (!lmmc_test_nearly_equal(out, 0.0, eps)) {
        fail_msg("    acosh(1) = %g, expected 0\n", out);
    }

    /* atanh(0) = 0 */
    st = lmmc_atanh(0.0, &out);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    atanh(0) failed, status=%d\n", (int)st);
    }
    if (!lmmc_test_nearly_equal(out, 0.0, eps)) {
        fail_msg("    atanh(0) = %g, expected 0\n", out);
    }
}

static void test_rounding_boundaries(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t out;
    const double eps = 1e-12;
    /* floor(2.7) = 2.0 */
    st = lmmc_floor(2.7, &out);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    floor(2.7) failed, status=%d\n", (int)st);
    }
    if (!lmmc_test_nearly_equal(out, 2.0, eps)) {
        fail_msg("    floor(2.7) = %g, expected 2.0\n", out);
    }

    /* ceil(2.3) = 3.0 */
    st = lmmc_ceil(2.3, &out);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    ceil(2.3) failed, status=%d\n", (int)st);
    }
    if (!lmmc_test_nearly_equal(out, 3.0, eps)) {
        fail_msg("    ceil(2.3) = %g, expected 3.0\n", out);
    }
}

static void test_power_sign(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t out;
    const double eps = 1e-12;
    /* pow(2.0, 3.0) = 8.0 */
    st = lmmc_pow(2.0, 3.0, &out);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    pow(2,3) failed, status=%d\n", (int)st);
    }
    if (!lmmc_test_nearly_equal(out, 8.0, eps)) {
        fail_msg("    pow(2,3) = %g, expected 8.0\n", out);
    }

    /* pow(-2.0, 3.0) = -8.0 (negative base, integer exponent is OK) */
    st = lmmc_pow(-2.0, 3.0, &out);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    pow(-2,3) failed, status=%d\n", (int)st);
    }
    if (!lmmc_test_nearly_equal(out, -8.0, eps)) {
        fail_msg("    pow(-2,3) = %g, expected -8.0\n", out);
    }
}

/* NULL pointer checks */
static void test_inverse_trig_null_outputs(void **state) {
    (void)state;
    lmmc_status_t st;
    /* asin with NULL out */
    st = lmmc_asin(0.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    asin(0.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    /* acos with NULL out */
    st = lmmc_acos(0.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    acos(0.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    /* atan with NULL out */
    st = lmmc_atan(0.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    atan(0.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }
}

static void test_hyperbolic_null_outputs(void **state) {
    (void)state;
    lmmc_status_t st;
    /* sinh with NULL out */
    st = lmmc_sinh(1.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    sinh(1.0, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    /* cosh with NULL out */
    st = lmmc_cosh(1.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    cosh(1.0, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    /* tanh with NULL out */
    st = lmmc_tanh(1.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    tanh(1.0, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }
}

static void test_inverse_hyperbolic_null_outputs(void **state) {
    (void)state;
    lmmc_status_t st;
    /* asinh with NULL out */
    st = lmmc_asinh(1.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    asinh(1.0, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    /* acosh with NULL out */
    st = lmmc_acosh(2.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    acosh(2.0, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    /* atanh with NULL out */
    st = lmmc_atanh(0.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    atanh(0.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }
}

static void test_power_rounding_null_outputs(void **state) {
    (void)state;
    lmmc_status_t st;
    /* pow with NULL out */
    st = lmmc_pow(2.0, 3.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    pow(2,3,NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    /* ceil with NULL out */
    st = lmmc_ceil(1.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    ceil(1.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    /* floor with NULL out */
    st = lmmc_floor(1.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    floor(1.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    /* round with NULL out */
    st = lmmc_round(1.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    round(1.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    /* trunc with NULL out */
    st = lmmc_trunc(1.5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    trunc(1.5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }
}

/* ---- Main ---- */

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_property18_inverse_trig_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property19_hyperbolic_roundtrips, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property20_floor_ceil_invariants, setup, teardown),
        cmocka_unit_test_setup_teardown(test_inverse_trig_domain, setup, teardown),
        cmocka_unit_test_setup_teardown(test_inverse_hyperbolic_domain, setup, teardown),
        cmocka_unit_test_setup_teardown(test_power_domain_and_range, setup, teardown),
        cmocka_unit_test_setup_teardown(test_scalar_nonfinite_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_inverse_trig_boundaries, setup, teardown),
        cmocka_unit_test_setup_teardown(test_inverse_hyperbolic_boundaries, setup, teardown),
        cmocka_unit_test_setup_teardown(test_rounding_boundaries, setup, teardown),
        cmocka_unit_test_setup_teardown(test_power_sign, setup, teardown),
        cmocka_unit_test_setup_teardown(test_inverse_trig_null_outputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_hyperbolic_null_outputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_inverse_hyperbolic_null_outputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_power_rounding_null_outputs, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
