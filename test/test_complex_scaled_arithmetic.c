/**
 * @file test_complex_scaled_arithmetic.c
 * @brief 复数缩放运算的性质与单元测试。
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
#define TEST_PI LMMC_CONST_PI

static int setup(void **state) {
    (void)state;
    assert_int_equal(lmmc_init(), LMMC_STATUS_OK);
    return 0;
}

static int teardown(void **state) {
    (void)state;
    assert_int_equal(lmmc_deinit(), LMMC_STATUS_OK);
    return 0;
}

static void test_extreme_modulus_log(void **state) {
    (void)state;
    lmmc_complex_t z = {DBL_MAX, 1.0}, out;
    lmmc_real_t modulus;
    lmmc_status_t st;
    st = lmmc_complex_modulus(&z, &modulus);
    if (st != LMMC_STATUS_OK || !isfinite(modulus) || modulus != DBL_MAX) {
        fail_msg("    extreme modulus overflowed: status=%d, value=%g\n", (int)st, modulus);
    }

    st = lmmc_complex_log(&z, &out);
    if (st != LMMC_STATUS_OK || !isfinite(out.real) || !isfinite(out.imag) || !(fabs(out.real - log(DBL_MAX)) <= 1e-12)) {
        fail_msg("    extreme logarithm overflowed: status=%d, value=(%g,%g)\n", (int)st, out.real, out.imag);
    }
}

static void test_extreme_sqrt(void **state) {
    (void)state;
    lmmc_complex_t z = {DBL_MAX, 1.0}, out;
    lmmc_status_t st;
    st = lmmc_complex_sqrt(&z, &out);
    if (st != LMMC_STATUS_OK || !isfinite(out.real) || !isfinite(out.imag) || !(fabs(out.real / sqrt(DBL_MAX) - 1.0) <= 1e-12)) {
        fail_msg("    extreme square root overflowed: status=%d, value=(%g,%g)\n", (int)st, out.real, out.imag);
    }
}

static void test_extreme_diagonal_log(void **state) {
    (void)state;
    lmmc_complex_t out;
    lmmc_status_t st;
    const lmmc_complex_t diagonal_extreme = {DBL_MAX, DBL_MAX};
    const double expected_log_real = log(DBL_MAX) + 0.5 * log(2.0);
    st = lmmc_complex_log(&diagonal_extreme, &out);
    if (st != LMMC_STATUS_OK || !isfinite(out.real) || !isfinite(out.imag) || !(fabs(out.real - expected_log_real) <= 1e-12) || !(fabs(out.imag - TEST_PI / 4.0) <= 1e-12)) {
        fail_msg("    diagonal extreme logarithm failed: "
                 "status=%d, value=(%g,%g)\n",
                 (int)st, out.real, out.imag);
    }
}

static void test_extreme_diagonal_sqrt(void **state) {
    (void)state;
    lmmc_complex_t out;
    lmmc_status_t st;
    const lmmc_complex_t diagonal_extreme = {DBL_MAX, DBL_MAX};
    const double root_max = sqrt(DBL_MAX);
    const double expected_real_ratio =
        sqrt((sqrt(2.0) + 1.0) * 0.5);
    const double expected_imag_ratio =
        sqrt((sqrt(2.0) - 1.0) * 0.5);
    st = lmmc_complex_sqrt(&diagonal_extreme, &out);
    if (st != LMMC_STATUS_OK || !isfinite(out.real) || !isfinite(out.imag) || !(fabs(out.real / root_max - expected_real_ratio) <= 1e-12) || !(fabs(out.imag / root_max - expected_imag_ratio) <= 1e-12)) {
        fail_msg("    diagonal extreme square root failed: "
                 "status=%d, value=(%g,%g)\n",
                 (int)st, out.real, out.imag);
    }
}

static void test_extreme_subnormal_division(void **state) {
    (void)state;
    lmmc_complex_t out;
    lmmc_status_t st;
    const lmmc_complex_t numerator = {1.0, 0.0};
    const lmmc_complex_t denominator = {DBL_MAX, DBL_MAX};
    st = lmmc_complex_div(&numerator, &denominator, &out);
    if (st != LMMC_STATUS_OK || !isfinite(out.real) || !isfinite(out.imag) || !(out.real > 0.0) || !(out.imag < 0.0)) {
        fail_msg("    extreme division lost finite subnormal result: "
                 "status=%d, value=(%g,%g)\n",
                 (int)st, out.real, out.imag);
    }
}

static void test_extreme_negative_radius(void **state) {
    (void)state;
    lmmc_complex_t out;
    lmmc_status_t st;
    st = lmmc_complex_from_polar(-1.0, 0.0, &out);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    negative polar radius returned %d instead of INVALID_ARGUMENT\n", (int)st);
    }
}
static void test_unit_scaled_multiplication(void **state) {
    (void)state;
    const struct {
        lmmc_complex_t a, b, expected;
    } cases[] = {
        {{1.0 + 0x1p-52, 1.0}, {1.0 - 0x1p-52, 1.0}, {-0x1p-104, 2.0}},
        {{0x1.8p1023, 0x1p1022}, {1.375, 0.5}, {0x1.dp1023, 0x1.7p1023}},
        {{0x1p-1074, 0x1p-1074}, {0.5, 0.5}, {0.0, 0x1p-1074}},
        {{0x1p1023, 0x1p-1022}, {0x1p-52, 0.0}, {0x1p971, 0x1p-1074}}};

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        lmmc_complex_t out = {7.0, 9.0};
        lmmc_status_t st = lmmc_complex_mul(&cases[i].a, &cases[i].b, &out);
        if (st != LMMC_STATUS_OK || out.real != cases[i].expected.real || out.imag != cases[i].expected.imag) {
            fail_msg("    scaled multiplication case %" PRIuMAX ": status=%d, value=(%a,%a)\n", (uintmax_t)(i), (int)st, out.real, out.imag);
        }
        lmmc_complex_t in_place = cases[i].a;
        st = lmmc_complex_mul(&in_place, &cases[i].b, &in_place);
        if (st != LMMC_STATUS_OK || in_place.real != cases[i].expected.real || in_place.imag != cases[i].expected.imag) {
            fail_msg("    in-place scaled multiplication case %" PRIuMAX " failed\n", (uintmax_t)(i));
        }
    }
}
static void test_unit_scaled_division(void **state) {
    (void)state;
    const struct {
        lmmc_complex_t numerator;
        lmmc_complex_t denominator;
        lmmc_complex_t expected;
    } cases[] = {
        {{DBL_MAX, DBL_MAX}, {1.0, 1.0}, {DBL_MAX, 0.0}},
        {{0.0, 0x1p-1074}, {0.0, 0x1p-1074}, {1.0, 0.0}},
        {{DBL_MAX, 0x1p-1074}, {1.0, 0.0}, {DBL_MAX, 0x1p-1074}},
        {{DBL_MAX, 0.0}, {1.0, 0x1p-1074}, {DBL_MAX, -0x1.fffffffffffffp-51}}};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        for (int alias = 0; alias < 3; ++alias) {
            lmmc_complex_t numerator = cases[i].numerator;
            lmmc_complex_t denominator = cases[i].denominator;
            lmmc_complex_t separate = {7.0, 9.0};
            lmmc_complex_t *out = alias == 1 ? &numerator : alias == 2 ? &denominator
                                                                       : &separate;
            const lmmc_status_t st = lmmc_complex_div(&numerator, &denominator, out);
            if (st != LMMC_STATUS_OK || out->real != cases[i].expected.real || out->imag != cases[i].expected.imag) {
                fail_msg("    scaled division case %" PRIuMAX ", alias %d: status=%d, value=(%a,%a)\n", (uintmax_t)(i), alias, (int)st, out->real, out->imag);
            }
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_extreme_modulus_log, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_sqrt, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_diagonal_log, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_diagonal_sqrt, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_subnormal_division, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_negative_radius, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_scaled_multiplication, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_scaled_division, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
