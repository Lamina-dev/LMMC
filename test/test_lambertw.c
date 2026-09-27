#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"

static int forward_close(double actual, double expected) {
    return isfinite(actual) && fabs(actual - expected) <=
                                   64.0 * DBL_EPSILON * fabs(expected) + 2.0 * nextafter(0.0, 1.0);
}

static void test_reference_values(void **state) {
    (void)state;
    /**
     * @brief 以精确 binary64 输入计算的 mpmath 100 位十进制参考值。
     * 端点代表值不作为精确双精度输入的数学真值，其两个分支另行测试。
     */
    const struct {
        double z;
        int branch;
        double expected;
    } cases[] = {
        {1.0, 0, 0.56714329040978384},
        {-0x1.70a3d70a3d70ap-2, 0, -0.80608431597081764},
        {-0x1.70a3d70a3d70ap-2, -1, -1.2227701339785062},
        {-0x1.999999999999ap-4, -1, -3.5771520639572971},
        {-0x1.47ae147ae147bp-7, -1, -6.4727751243940048},
        {0x1.4e718d7d7625ap+664, 0, 454.39804503371403},
        {0x1.249ad2594c37dp+332, 0, 224.84310644511851},
        {DBL_MAX, 0, 703.22703310477016},
        {-0x1.79ca10c924223p-67, -1, -49.962984276674476},
        {-0x1.78b56362cef37p-2, 0, -0.99999998469574591},
        {-0x1.78b56362cef37p-2, -1, -1.0000000153042543},
        {-0x0.0000000000001p-1022, -1, -751.06155953987911},
        {-0x1.3333333333333p-2, 0, -0.48940222718021492},
        {-0x1.3333333333333p-2, -1, -1.7813370234216277}};
    size_t i;

    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        double result = 123.0;
        lmmc_status_t status = cases[i].branch == 0
                                   ? lmmc_lambertw(cases[i].z, &result)
                                   : lmmc_lambertw_wm1(cases[i].z, &result);
        if (status != LMMC_STATUS_OK || !forward_close(result, cases[i].expected)) {
            fail_msg("W branch %d z=%.17g: status=%d actual=%.17g expected=%.17g\n", cases[i].branch, cases[i].z, status, result, cases[i].expected);
        }
    }
}

static void test_tiny_values(void **state) {
    (void)state;
    const double tiny[] = {0.0, -0.0, DBL_MIN, -DBL_MIN,
                           0x1p-54, -0x1p-54, 0x0.0000000000001p-1022, -0x0.0000000000001p-1022};
    size_t i;
    for (i = 0; i < sizeof(tiny) / sizeof(tiny[0]); ++i) {
        double result;
        assert_false(lmmc_lambertw(tiny[i], &result) != LMMC_STATUS_OK || result != tiny[i] || !!signbit(result) != !!signbit(tiny[i]));
    }
}

static void test_constructed_values(void **state) {
    (void)state;
    const double roots[] = {-2.0, -8.0, -32.0, -100.0, -700.0};
    size_t i;
    for (i = 0; i < sizeof(roots) / sizeof(roots[0]); ++i) {
        double result;
        const double z = roots[i] * exp(roots[i]);
        assert_false(lmmc_lambertw_wm1(z, &result) != LMMC_STATUS_OK || !forward_close(result, roots[i]));
    }
}

static void test_series_boundary(void **state) {
    (void)state;
    size_t i;
    for (i = 0; i < 6; ++i) {
        const double magnitude = i % 3 == 0 ? nextafter(0x1p-26, 0.0) : (i % 3 == 1 ? 0x1p-26 : nextafter(0x1p-26, INFINITY));
        const double z = i < 3 ? magnitude : -magnitude;
        const double reference = z - z * z + 1.5 * z * z * z - (8.0 / 3.0) * z * z * z * z;
        double result;
        assert_false(lmmc_lambertw(z, &result) != LMMC_STATUS_OK || !forward_close(result, reference));
    }
}

static void check_branch_domains(int branch) {
    const double invalid[] = {-0.5, nextafter(-LMMC_INV_E, -INFINITY), NAN, INFINITY, -INFINITY};
    size_t i;
    double result = 123.0;
    lmmc_status_t (*function)(double, double *) = branch == 0 ? lmmc_lambertw : lmmc_lambertw_wm1;
    assert_false(function(-LMMC_INV_E, &result) != LMMC_STATUS_OK || result != -1.0);
    assert_false(function(-0.1, NULL) != LMMC_STATUS_INVALID_ARGUMENT);
    for (i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        const lmmc_status_t expected = isfinite(invalid[i])
                                           ? LMMC_STATUS_INVALID_ARGUMENT
                                           : LMMC_STATUS_NUMERICAL_FAILURE;
        result = 123.0;
        assert_false(function(invalid[i], &result) != expected || result != 123.0);
    }
}

static void test_domains_and_atomic_outputs(void **state) {
    (void)state;
    size_t i;
    for (int branch = 0; branch >= -1; --branch) {
        check_branch_domains(branch);
    }
    for (i = 0; i < 3; ++i) {
        double result = 123.0;
        const double z = i == 0 ? 0.0 : (i == 1 ? -0.0 : 1.0);
        assert_false(lmmc_lambertw_wm1(z, &result) != LMMC_STATUS_INVALID_ARGUMENT || result != 123.0);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_reference_values),
        cmocka_unit_test(test_tiny_values),
        cmocka_unit_test(test_constructed_values),
        cmocka_unit_test(test_series_boundary),
        cmocka_unit_test(test_domains_and_atomic_outputs),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
