/**
 * @file test_complex_transcendental.c
 * @brief 复数超越函数单元测试：exp, log, sqrt, sin, cos, pow。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int nearly_equal(double a, double b, double tol) {
    return isfinite(a) && isfinite(b) && fabs(a - b) <= tol * (1.0 + fabs(b));
}

static int complex_nearly_equal(const lmmc_complex_t *actual,
                                double real, double imag, double tol) {
    return nearly_equal(actual->real, real, tol) &&
           nearly_equal(actual->imag, imag, tol);
}

static void test_power_boundary_contract(void **state) {
    (void)state;
    const struct {
        lmmc_complex_t base, exponent;
        lmmc_status_t status;
        lmmc_complex_t expected;
    } cases[] = {
        {{0.0, 0.0}, {0.0, 1.0}, LMMC_STATUS_OUT_OF_RANGE, {7.0, 9.0}},
        {{0.0, 0.0}, {NAN, 0.0}, LMMC_STATUS_NUMERICAL_FAILURE, {7.0, 9.0}},
        {{NAN, 0.0}, {0.0, 0.0}, LMMC_STATUS_NUMERICAL_FAILURE, {7.0, 9.0}},
        {{0.0, 0.0}, {-1.0, INFINITY}, LMMC_STATUS_NUMERICAL_FAILURE, {7.0, 9.0}},
        {{0.0, 0.0}, {1.0, 2.0}, LMMC_STATUS_OK, {0.0, 0.0}},
        {{2.0, 0.0}, {2048.0, 0.0}, LMMC_STATUS_NUMERICAL_FAILURE, {7.0, 9.0}}};

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        lmmc_complex_t out = {7.0, 9.0};
        lmmc_status_t st = lmmc_complex_pow(&cases[i].base, &cases[i].exponent, &out);
        if (st != cases[i].status || out.real != cases[i].expected.real || out.imag != cases[i].expected.imag) {
            fail_msg("FAIL: power boundary case %" PRIuMAX ": status=%d, value=(%a,%a)\n", (uintmax_t)(i), (int)st, out.real, out.imag);
        }
    }
}

static void test_integer_power_contract(void **state) {
    (void)state;
    const struct {
        lmmc_complex_t base, exponent, expected;
    } cases[] = {
        {{-DBL_MAX, 0.0}, {1.0, 0.0}, {-DBL_MAX, 0.0}},
        {{DBL_MAX, DBL_MAX}, {1.0, 0.0}, {DBL_MAX, DBL_MAX}},
        {{-1.0, 0.0}, {2.0, 0.0}, {1.0, 0.0}},
        {{0.0, 1.0}, {64.0, 0.0}, {1.0, 0.0}},
        {{0.0, 1.0}, {0x1.fffffffffffffp52, 0.0}, {0.0, -1.0}},
        {{0.0, 1.0}, {DBL_MAX, 0.0}, {1.0, 0.0}},
        {{0x1p512, 0x1p512}, {-2.0, 0.0}, {0.0, -0x1p-1025}}};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        for (int alias = 0; alias < 3; ++alias) {
            lmmc_complex_t base = cases[i].base;
            lmmc_complex_t exponent = cases[i].exponent;
            lmmc_complex_t separate = {7.0, 9.0};
            lmmc_complex_t *out = alias == 1 ? &base : alias == 2 ? &exponent
                                                                  : &separate;
            const lmmc_status_t st = lmmc_complex_pow(&base, &exponent, out);
            if (st != LMMC_STATUS_OK || out->real != cases[i].expected.real || out->imag != cases[i].expected.imag) {
                fail_msg("FAIL: integer power case %" PRIuMAX ", alias %d: status=%d, value=(%a,%a)\n", (uintmax_t)(i), alias, (int)st, out->real, out->imag);
            }
        }
    }
}

static void test_exponential_range_contract(void **state) {
    (void)state;
    lmmc_complex_t z = {710.0, M_PI / 4.0};
    lmmc_complex_t out = {7.0, 9.0};
    const double expected = 1.5796728482882014e308;
    lmmc_status_t st = lmmc_complex_exp(&z, &out);

    if (st != LMMC_STATUS_OK || !nearly_equal(out.real, expected, 1e-14) || !nearly_equal(out.imag, expected, 1e-14)) {
        fail_msg("FAIL: finite large complex exponential: status=%d, value=(%a,%a)\n", (int)st, out.real, out.imag);
    }

    z.real = -744.25;
    z.imag = -acos(0.49);
    st = lmmc_complex_exp(&z, &out);
    if (st != LMMC_STATUS_OK || out.real != 0x1p-1074 || out.imag != -0x1p-1074) {
        fail_msg("FAIL: subnormal complex exponential: status=%d, value=(%a,%a)\n", (int)st, out.real, out.imag);
    }

    z.real = 710.0;
    z.imag = 0.0;
    out.real = 7.0;
    out.imag = 9.0;
    st = lmmc_complex_exp(&z, &out);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || out.real != 7.0 || out.imag != 9.0) {
        fail_msg("FAIL: overflowing complex exponential: status=%d, value=(%a,%a)\n", (int)st, out.real, out.imag);
    }
}

static void check_finite_trigonometric_range(int cosine) {
    const double expected = 1.3022201128601071e308;
    lmmc_status_t (*operation)(const lmmc_complex_t *, lmmc_complex_t *) =
        cosine ? lmmc_complex_cos : lmmc_complex_sin;

    for (int side = -1; side <= 1; side += 2) {
        lmmc_complex_t z = {M_PI / 4.0, side * 710.5};
        lmmc_complex_t out = {7.0, 9.0};
        lmmc_status_t st = operation(&z, &out);
        const double expected_imag = (cosine ? -side : side) * expected;
        if (st != LMMC_STATUS_OK || !nearly_equal(out.real, expected, 1e-14) || !nearly_equal(out.imag, expected_imag, 1e-14)) {
            fail_msg("FAIL: finite complex %s on side %d: status=%d, value=(%a,%a)\n", cosine ? "cosine" : "sine", side, (int)st, out.real, out.imag);
        }
    }
}

static void check_trigonometric_overflow(int cosine) {
    lmmc_status_t (*operation)(const lmmc_complex_t *, lmmc_complex_t *) =
        cosine ? lmmc_complex_cos : lmmc_complex_sin;

    {
        const lmmc_complex_t z = {0.0, 710.5};
        lmmc_complex_t out = {7.0, 9.0};
        lmmc_status_t st = operation(&z, &out);
        if (st != LMMC_STATUS_NUMERICAL_FAILURE || out.real != 7.0 || out.imag != 9.0) {
            fail_msg("FAIL: overflowing complex %s: status=%d, value=(%a,%a)\n", cosine ? "cosine" : "sine", (int)st, out.real, out.imag);
        }
    }
}

static void test_trigonometric_range_contract(void **state) {
    (void)state;

    for (int cosine = 0; cosine <= 1; ++cosine) {
        check_finite_trigonometric_range(cosine);
        check_trigonometric_overflow(cosine);
    }
}

static void test_logarithm_near_unit_modulus(void **state) {
    (void)state;
    const struct {
        lmmc_complex_t input;
        double expected_real;
    } cases[] = {
        {{0.6, 0.8}, 0x1.999999999999ap-56},
        {{0.6, 0x1.9999999999999p-1}, -0x1.3333333333333p-54},
        {{-0.9862343144953257, 0.16535379315859358}, 1.7510020490391953e-19},
        {{1.0, 0x1p-27}, 0x1p-55}};

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        lmmc_complex_t out = {7.0, 9.0};
        lmmc_status_t st = lmmc_complex_log(&cases[i].input, &out);
        if (st != LMMC_STATUS_OK || !isfinite(out.real) || !(fabs(out.real - cases[i].expected_real) <= 4.0 * fabs(nextafter(cases[i].expected_real, INFINITY) - cases[i].expected_real))) {
            fail_msg("FAIL: logarithm near unit modulus case %" PRIuMAX ": status=%d, real=%a\n", (uintmax_t)(i), (int)st, out.real);
        }
    }
}

static void test_exponential_values(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_complex_t z;
    lmmc_complex_t result;
    /* exp(0+0i) = 1+0i */
    lmmc_complex_create(0.0, 0.0, &z);
    st = lmmc_complex_exp(&z, &result);
    if (st != LMMC_STATUS_OK || !complex_nearly_equal(&result, 1.0, 0.0, 1e-12)) {
        fail_msg("FAIL: exp(0+0i) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }

    /* exp(1+0i) = e+0i */
    lmmc_complex_create(1.0, 0.0, &z);
    st = lmmc_complex_exp(&z, &result);
    if (st != LMMC_STATUS_OK || !complex_nearly_equal(&result, exp(1.0), 0.0, 1e-12)) {
        fail_msg("FAIL: exp(1+0i) expected e+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }

    /* exp(0+pi*i) = -1+0i (Euler's identity) */
    lmmc_complex_create(0.0, M_PI, &z);
    st = lmmc_complex_exp(&z, &result);
    if (st != LMMC_STATUS_OK || !complex_nearly_equal(&result, -1.0, 0.0, 1e-12)) {
        fail_msg("FAIL: exp(i*pi) expected -1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }

    /* exp(NULL) -> INVALID_ARGUMENT */
    st = lmmc_complex_exp(NULL, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("FAIL: exp(NULL) expected INVALID_ARGUMENT, got %d\n", st);
    }
}

static void test_logarithm_values(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_complex_t z;
    lmmc_complex_t result;
    /* log(1+0i) = 0+0i */
    lmmc_complex_create(1.0, 0.0, &z);
    st = lmmc_complex_log(&z, &result);
    if (st != LMMC_STATUS_OK || !complex_nearly_equal(&result, 0.0, 0.0, 1e-12)) {
        fail_msg("FAIL: log(1+0i) expected 0+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }

    /* log(e+0i) = 1+0i */
    lmmc_complex_create(exp(1.0), 0.0, &z);
    st = lmmc_complex_log(&z, &result);
    if (st != LMMC_STATUS_OK || !complex_nearly_equal(&result, 1.0, 0.0, 1e-12)) {
        fail_msg("FAIL: log(e+0i) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }

    /* log(-1+0i) = 0+pi*i */
    lmmc_complex_create(-1.0, 0.0, &z);
    st = lmmc_complex_log(&z, &result);
    if (st != LMMC_STATUS_OK || !complex_nearly_equal(&result, 0.0, M_PI, 1e-12)) {
        fail_msg("FAIL: log(-1+0i) expected 0+pi*i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }

    /* log(0+0i) -> OUT_OF_RANGE */
    lmmc_complex_create(0.0, 0.0, &z);
    st = lmmc_complex_log(&z, &result);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        fail_msg("FAIL: log(0+0i) expected OUT_OF_RANGE, got %d\n", st);
    }

    /* log(NULL) -> INVALID_ARGUMENT */
    st = lmmc_complex_log(NULL, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("FAIL: log(NULL) expected INVALID_ARGUMENT, got %d\n", st);
    }
}

static void test_sqrt_positive_values(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_complex_t z;
    lmmc_complex_t result;
    /* sqrt(1+0i) = 1+0i */
    lmmc_complex_create(1.0, 0.0, &z);
    st = lmmc_complex_sqrt(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        fail_msg("FAIL: sqrt(1+0i) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }

    /* sqrt(4+0i) = 2+0i */
    lmmc_complex_create(4.0, 0.0, &z);
    st = lmmc_complex_sqrt(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 2.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        fail_msg("FAIL: sqrt(4+0i) expected 2+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }
}

static void test_sqrt_branch_values(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_complex_t z;
    lmmc_complex_t result;
    /* sqrt(-1+0i) = 0+1i */
    lmmc_complex_create(-1.0, 0.0, &z);
    st = lmmc_complex_sqrt(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 0.0, 1e-12) || !nearly_equal(result.imag, 1.0, 1e-12)) {
        fail_msg("FAIL: sqrt(-1+0i) expected 0+1i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }

    /* sqrt(0+0i) = 0+0i */
    lmmc_complex_create(0.0, 0.0, &z);
    st = lmmc_complex_sqrt(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 0.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        fail_msg("FAIL: sqrt(0+0i) expected 0+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }
}

static void test_sine_values(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_complex_t z;
    lmmc_complex_t result;
    /* sin(0+0i) = 0+0i */
    lmmc_complex_create(0.0, 0.0, &z);
    st = lmmc_complex_sin(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 0.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        fail_msg("FAIL: sin(0+0i) expected 0+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }

    /* sin(pi/2 + 0i) = 1+0i */
    lmmc_complex_create(M_PI / 2.0, 0.0, &z);
    st = lmmc_complex_sin(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        fail_msg("FAIL: sin(pi/2+0i) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }
}

static void test_cosine_values(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_complex_t z;
    lmmc_complex_t result;
    /* cos(0+0i) = 1+0i */
    lmmc_complex_create(0.0, 0.0, &z);
    st = lmmc_complex_cos(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        fail_msg("FAIL: cos(0+0i) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }

    /* cos(pi+0i) = -1+0i */
    lmmc_complex_create(M_PI, 0.0, &z);
    st = lmmc_complex_cos(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, -1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        fail_msg("FAIL: cos(pi+0i) expected -1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
    }
}

static void test_trigonometric_identity(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_complex_t z;
    /* sin^2(z) + cos^2(z) = 1 */
    {
        lmmc_complex_t sin_z, cos_z, sin2, cos2, sum;
        lmmc_complex_create(1.5, 2.3, &z);
        st = lmmc_complex_sin(&z, &sin_z);
        if (st != LMMC_STATUS_OK) {
            fail_msg("FAIL: sin(1.5+2.3i) returned st=%d\n", st);
        }
        st = lmmc_complex_cos(&z, &cos_z);
        if (st != LMMC_STATUS_OK) {
            fail_msg("FAIL: cos(1.5+2.3i) returned st=%d\n", st);
        }
        st = lmmc_complex_mul(&sin_z, &sin_z, &sin2);
        if (st != LMMC_STATUS_OK) {
            fail_msg("FAIL: sin^2 mul returned st=%d\n", st);
        }
        st = lmmc_complex_mul(&cos_z, &cos_z, &cos2);
        if (st != LMMC_STATUS_OK) {
            fail_msg("FAIL: cos^2 mul returned st=%d\n", st);
        }
        st = lmmc_complex_add(&sin2, &cos2, &sum);
        if (st != LMMC_STATUS_OK) {
            fail_msg("FAIL: sin^2+cos^2 add returned st=%d\n", st);
        }
        if (!nearly_equal(sum.real, 1.0, 1e-10) || !nearly_equal(sum.imag, 0.0, 1e-10)) {
            fail_msg("FAIL: sin^2(1.5+2.3i)+cos^2(1.5+2.3i) expected 1+0i, got %g+%gi\n", sum.real, sum.imag);
        }
    }
}

static void test_power_values(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_complex_t result;
    /* pow(2+0i, 3+0i) = 8+0i */
    {
        lmmc_complex_t base, exponent;
        lmmc_complex_create(2.0, 0.0, &base);
        lmmc_complex_create(3.0, 0.0, &exponent);
        st = lmmc_complex_pow(&base, &exponent, &result);
        if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 8.0, 1e-10) || !nearly_equal(result.imag, 0.0, 1e-10)) {
            fail_msg("FAIL: pow(2,3) expected 8+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        }
    }

    /* pow(0+0i, -1+0i) -> OUT_OF_RANGE */
    {
        lmmc_complex_t base, exponent;
        lmmc_complex_create(0.0, 0.0, &base);
        lmmc_complex_create(-1.0, 0.0, &exponent);
        st = lmmc_complex_pow(&base, &exponent, &result);
        if (st != LMMC_STATUS_OUT_OF_RANGE) {
            fail_msg("FAIL: pow(0, -1) expected OUT_OF_RANGE, got %d\n", st);
        }
    }
}

static void test_power_zero_values(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_complex_t result;
    /* pow(0+0i, 2+0i) = 0+0i */
    {
        lmmc_complex_t base, exponent;
        lmmc_complex_create(0.0, 0.0, &base);
        lmmc_complex_create(2.0, 0.0, &exponent);
        st = lmmc_complex_pow(&base, &exponent, &result);
        if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 0.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
            fail_msg("FAIL: pow(0,2) expected 0+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        }
    }

    /* The public power API defines pow(0+0i, 0+0i) = 1+0i. */
    {
        lmmc_complex_t base, exponent;
        lmmc_complex_create(0.0, 0.0, &base);
        lmmc_complex_create(0.0, 0.0, &exponent);
        st = lmmc_complex_pow(&base, &exponent, &result);
        if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
            fail_msg("FAIL: pow(0,0) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        }
    }

    /* pow(NULL, ...) -> INVALID_ARGUMENT */
    {
        lmmc_complex_t exponent;
        lmmc_complex_create(1.0, 0.0, &exponent);
        st = lmmc_complex_pow(NULL, &exponent, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            fail_msg("FAIL: pow(NULL,...) expected INVALID_ARGUMENT, got %d\n", st);
        }
    }
}

static void test_log_exp_inverse(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_complex_t z;
    /* exp(log(z)) ≈ z */
    {
        lmmc_complex_t log_z, exp_log_z;
        lmmc_complex_create(3.0, 4.0, &z);
        st = lmmc_complex_log(&z, &log_z);
        if (st != LMMC_STATUS_OK) {
            fail_msg("FAIL: log(3+4i) returned st=%d\n", st);
        }
        st = lmmc_complex_exp(&log_z, &exp_log_z);
        if (st != LMMC_STATUS_OK) {
            fail_msg("FAIL: exp(log(3+4i)) returned st=%d\n", st);
        }
        if (!nearly_equal(exp_log_z.real, 3.0, 1e-10) || !nearly_equal(exp_log_z.imag, 4.0, 1e-10)) {
            fail_msg("FAIL: exp(log(3+4i)) expected 3+4i, got %g+%gi\n", exp_log_z.real, exp_log_z.imag);
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_power_boundary_contract),
        cmocka_unit_test(test_integer_power_contract),
        cmocka_unit_test(test_exponential_range_contract),
        cmocka_unit_test(test_trigonometric_range_contract),
        cmocka_unit_test(test_logarithm_near_unit_modulus),
        cmocka_unit_test(test_exponential_values),
        cmocka_unit_test(test_logarithm_values),
        cmocka_unit_test(test_sqrt_positive_values),
        cmocka_unit_test(test_sqrt_branch_values),
        cmocka_unit_test(test_sine_values),
        cmocka_unit_test(test_cosine_values),
        cmocka_unit_test(test_trigonometric_identity),
        cmocka_unit_test(test_power_values),
        cmocka_unit_test(test_power_zero_values),
        cmocka_unit_test(test_log_exp_inverse),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
