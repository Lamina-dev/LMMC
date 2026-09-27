#include <float.h>
#include <math.h>
#include <stdio.h>

#include "lmmc/quadrature.h"
#include "test_common.h"

static lmmc_real_t fn_exp(lmmc_real_t x, void *data) {
    (void)data;
    return exp(x);
}

static lmmc_real_t fn_sin(lmmc_real_t x, void *data) {
    (void)data;
    return sin(x);
}

static lmmc_real_t fn_scaled_const(lmmc_real_t x, void *data) {
    (void)x;
    return *(const lmmc_real_t *)data;
}

static lmmc_real_t fn_nan(lmmc_real_t x, void *data) {
    (void)x;
    (void)data;
    return NAN;
}

static void test_romberg_exp(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    const lmmc_real_t exact = exp(1.0) - 1.0;
    const lmmc_status_t status = lmmc_quad_romberg(
        fn_exp, NULL, 0.0, 1.0, 1e-12, 20, &result);
    printf("Romberg exp: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
           status, result.value, exact, fabs(result.value - exact));
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(result.value, exact, 1e-10));
}

static void test_romberg_sin(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    const lmmc_real_t exact = 2.0;
    const lmmc_real_t pi = 3.14159265358979323846;
    const lmmc_status_t status = lmmc_quad_romberg(
        fn_sin, NULL, 0.0, pi, 1e-12, 20, &result);
    printf("Romberg sin: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
           status, result.value, exact, fabs(result.value - exact));
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(result.value, exact, 1e-10));
}

static void test_romberg_wide(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    const lmmc_real_t scale = 1.0 / DBL_MAX;
    const lmmc_status_t status = lmmc_quad_romberg(
        fn_scaled_const, (void *)&scale,
        -DBL_MAX, DBL_MAX, 1e-12, 4, &result);
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(result.value, 2.0, 1e-12));
}

static void test_romberg_large(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    const lmmc_real_t scale = 0.5;
    const lmmc_status_t status = lmmc_quad_romberg(
        fn_scaled_const, (void *)&scale,
        0.0, DBL_MAX, 1e-12, 4, &result);
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(result.value == DBL_MAX * 0.5);
}

static void test_romberg_invalid(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    assert_int_equal(
        lmmc_quad_romberg(NULL, NULL, 0.0, 1.0, 1e-10, 20, &result),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_romberg(fn_exp, NULL, 1.0, 0.0, 1e-10, 20, &result),
        LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_romberg_failures(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    lmmc_status_t status = lmmc_quad_romberg(
        fn_exp, NULL, 0.0, 1.0, 1e-12, 1, &result);
    assert_int_equal(status, LMMC_STATUS_CONVERGENCE_FAILED);
    assert_true(isinf(result.error));

    status = lmmc_quad_romberg(
        fn_exp, NULL, 0.0, 1.0, 1e-12, 31, &result);
    assert_int_equal(status, LMMC_STATUS_INVALID_ARGUMENT);

    status = lmmc_quad_romberg(
        fn_nan, NULL, 0.0, 1.0, 1e-12, 4, &result);
    assert_int_equal(status, LMMC_STATUS_NUMERICAL_FAILURE);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_romberg_exp),
        cmocka_unit_test(test_romberg_sin),
        cmocka_unit_test(test_romberg_wide),
        cmocka_unit_test(test_romberg_large),
        cmocka_unit_test(test_romberg_invalid),
        cmocka_unit_test(test_romberg_failures),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
