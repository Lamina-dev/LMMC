/**
 * @file test_complex_principal_branches.c
 * @brief 复数模块的性质测试与单元测试。
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

static int complex_nearly_equal(const lmmc_complex_t *a, const lmmc_complex_t *b, double eps) {
    return lmmc_test_nearly_equal(a->real, b->real, eps) &&
           lmmc_test_nearly_equal(a->imag, b->imag, eps);
}
static void test_in_place_log(void **state) {
    (void)state;

    for (int side = -1; side <= 1; side += 2) {
        const double imaginary_zero = copysign(0.0, (double)side);
        lmmc_complex_t z = {-4.0, imaginary_zero};
        lmmc_status_t st = lmmc_complex_log(&z, &z);
        if (st != LMMC_STATUS_OK || !isfinite(z.real) || !isfinite(z.imag) || !(fabs(z.real - log(4.0)) <= 1e-14) || !(fabs(z.imag - side * TEST_PI) <= 1e-14)) {
            fail_msg("    in-place logarithm on branch side %d: status=%d, value=(%a,%a)\n", side, (int)st, z.real, z.imag);
        }
    }
}
static void test_in_place_sqrt(void **state) {
    (void)state;

    for (int side = -1; side <= 1; side += 2) {
        const double imaginary_zero = copysign(0.0, (double)side);
        lmmc_complex_t z;
        lmmc_status_t st;
        z.real = -4.0;
        z.imag = imaginary_zero;
        st = lmmc_complex_sqrt(&z, &z);
        if (st != LMMC_STATUS_OK || z.real != 0.0 || signbit(z.real) || z.imag != side * 2.0) {
            fail_msg("    in-place square root on branch side %d: status=%d, value=(%a,%a)\n", side, (int)st, z.real, z.imag);
        }
    }
}
static void test_in_place_signed_zero(void **state) {
    (void)state;

    for (int side = -1; side <= 1; side += 2) {
        const double imaginary_zero = copysign(0.0, (double)side);
        lmmc_complex_t z;
        lmmc_status_t st;
        z.real = -0.0;
        z.imag = imaginary_zero;
        st = lmmc_complex_sqrt(&z, &z);
        if (st != LMMC_STATUS_OK || z.real != 0.0 || signbit(z.real) || z.imag != 0.0 || !!signbit(z.imag) != !!signbit(imaginary_zero)) {
            fail_msg("    square root of signed zero on side %d: status=%d, value=(%a,%a)\n", side, (int)st, z.real, z.imag);
        }
    }
}
static void test_in_place_subnormal(void **state) {
    (void)state;

    for (int side = -1; side <= 1; side += 2) {
        lmmc_complex_t z;
        lmmc_status_t st;
        for (int real_side = -1; real_side <= 1; real_side += 2) {
            z.real = real_side * 0.64;
            z.imag = side * 0x1p-1074;
            st = lmmc_complex_sqrt(&z, &z);
            const double expected_real = real_side < 0 ? 0x1p-1074 : 0.8;
            const double expected_imag = side * (real_side < 0 ? 0.8 : 0x1p-1074);
            if (st != LMMC_STATUS_OK || z.real != expected_real || z.imag != expected_imag) {
                fail_msg("    square root subnormal component on sides %d,%d: "
                         "status=%d, value=(%a,%a)\n",
                         real_side, side, (int)st, z.real, z.imag);
            }
        }
    }
}
static void test_in_place_log_regular(void **state) {
    (void)state;

    {
        lmmc_complex_t z = {3.0, 4.0};
        const lmmc_complex_t expected = {log(5.0), atan2(4.0, 3.0)};
        lmmc_status_t st = lmmc_complex_log(&z, &z);
        if (st != LMMC_STATUS_OK || !complex_nearly_equal(&z, &expected, 1e-14)) {
            fail_msg("    in-place logarithm: status=%d, value=(%a,%a)\n", (int)st, z.real, z.imag);
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_in_place_log, setup, teardown),
        cmocka_unit_test_setup_teardown(test_in_place_sqrt, setup, teardown),
        cmocka_unit_test_setup_teardown(test_in_place_signed_zero, setup, teardown),
        cmocka_unit_test_setup_teardown(test_in_place_subnormal, setup, teardown),
        cmocka_unit_test_setup_teardown(test_in_place_log_regular, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
