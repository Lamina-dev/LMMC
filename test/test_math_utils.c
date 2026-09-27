/**
 * @file test_math_utils.c
 * 针对 LMMC 中 math utils 相关接口的单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include "lmmc/lmmc.h"

#include "test_common.h"

static void check_numeric_constants(void **state) {
    (void)state;
    lmmc_real_t val;
    assert_true(lmmc_inf(&val) == LMMC_STATUS_OK && isinf(val));
    assert_true(lmmc_nan(&val) == LMMC_STATUS_OK && isnan(val));
    assert_true(lmmc_eps(&val) == LMMC_STATUS_OK &&
                val == nextafter(1.0, 2.0) - 1.0);
    assert_true(1.0 + val > 1.0);
    assert_true(lmmc_eps(NULL) == LMMC_STATUS_INVALID_ARGUMENT);
}

static void check_classification(void **state) {
    (void)state;
    int flag;
    assert_true(lmmc_isnan(NAN, &flag) == LMMC_STATUS_OK);
    assert_true(flag == 1);
    assert_true(lmmc_isnan(1.0, &flag) == LMMC_STATUS_OK);
    assert_true(flag == 0);
    assert_true(lmmc_isinf(INFINITY, &flag) == LMMC_STATUS_OK);
    assert_true(flag == 1);
    assert_true(lmmc_isinf(1.0, &flag) == LMMC_STATUS_OK);
    assert_true(flag == 0);
    assert_true(lmmc_isfinite(1.0, &flag) == LMMC_STATUS_OK);
    assert_true(flag == 1);
    assert_true(lmmc_isfinite(INFINITY, &flag) == LMMC_STATUS_OK);
    assert_true(flag == 0);
    assert_true(lmmc_signbit(-1.0, &flag) == LMMC_STATUS_OK);
    assert_true(flag == 1);
    assert_true(lmmc_signbit(1.0, &flag) == LMMC_STATUS_OK);
    assert_true(flag == 0);
}

static void check_geometric_functions(void **state) {
    (void)state;
    lmmc_real_t val;
    lmmc_real_t val2;
    assert_true(lmmc_atan2(1.0, 1.0, &val) == LMMC_STATUS_OK && fabs(val - 0.7853981633974483) < 1e5 * LMMC_REAL_EPSILON);
    assert_true(lmmc_sincos(0.5, &val, &val2) == LMMC_STATUS_OK && fabs(val - sin(0.5)) < 1e5 * LMMC_REAL_EPSILON && fabs(val2 - cos(0.5)) < 1e5 * LMMC_REAL_EPSILON);
    assert_true(lmmc_hypot(3.0, 4.0, &val) == LMMC_STATUS_OK && fabs(val - 5.0) < 1e5 * LMMC_REAL_EPSILON);
}

static void check_logarithmic_functions(void **state) {
    (void)state;
    lmmc_real_t val;
    assert_true(lmmc_exp2(3.0, &val) == LMMC_STATUS_OK && fabs(val - 8.0) < 1e5 * LMMC_REAL_EPSILON);
    assert_true(lmmc_log2(8.0, &val) == LMMC_STATUS_OK && fabs(val - 3.0) < 1e5 * LMMC_REAL_EPSILON);
    assert_true(lmmc_expm1(1e-10, &val) == LMMC_STATUS_OK && fabs(val - 1e-10) < 1e5 * LMMC_REAL_EPSILON);
    assert_true(lmmc_log1p(1e-10, &val) == LMMC_STATUS_OK && fabs(val - 1e-10) < 1e5 * LMMC_REAL_EPSILON);
}

static void check_floating_decomposition(void **state) {
    (void)state;
    lmmc_real_t val;
    lmmc_real_t val2;
    assert_true(lmmc_split_int_frac(3.14, &val, &val2) == LMMC_STATUS_OK && val == 3.0 && fabs(val2 - 0.14) < 1e5 * LMMC_REAL_EPSILON);
    assert_true(lmmc_fmod(5.3, 2.0, &val) == LMMC_STATUS_OK && fabs(val - 1.3) < 1e5 * LMMC_REAL_EPSILON);
    assert_true(lmmc_ldexp(0.5, 3, &val) == LMMC_STATUS_OK && val == 4.0);
    assert_true(lmmc_nextafter(1.0, 2.0, &val) == LMMC_STATUS_OK && val > 1.0);
}

static void check_approximate_comparison(void **state) {
    (void)state;
    int flag;
    assert_true(lmmc_approx_eq(1.0, 1.0 + 10.0 * LMMC_REAL_EPSILON, 20.0 * LMMC_REAL_EPSILON, &flag) == LMMC_STATUS_OK && flag == 1);
    assert_true(lmmc_approx_eq(1.0, 1.0 + 10.0 * LMMC_REAL_EPSILON, 5.0 * LMMC_REAL_EPSILON, &flag) == LMMC_STATUS_OK && flag == 0);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(check_numeric_constants),
        cmocka_unit_test(check_classification),
        cmocka_unit_test(check_geometric_functions),
        cmocka_unit_test(check_logarithmic_functions),
        cmocka_unit_test(check_floating_decomposition),
        cmocka_unit_test(check_approximate_comparison),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
