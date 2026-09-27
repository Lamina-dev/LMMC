#include <math.h>
#include <stdio.h>

#include "lmmc/quadrature.h"
#include "test_common.h"
#include "internal_test_hooks.h"

static lmmc_real_t fn_one(lmmc_real_t x, void *data) {
    (void)x;
    (void)data;
    return 1.0;
}

static lmmc_real_t fn_x2(lmmc_real_t x, void *data) {
    (void)data;
    return x * x;
}

static lmmc_real_t fn_x(lmmc_real_t x, void *data) {
    (void)data;
    return x;
}

static lmmc_real_t fn_nan(lmmc_real_t x, void *data) {
    (void)x;
    (void)data;
    return NAN;
}
static lmmc_real_t fn_counted_one(lmmc_real_t x, void *data) {
    (void)x;
    size_t *calls = data;
    ++*calls;
    return 1.0;
}


static void test_hermite_constant(void **state) {
    (void)state;
    lmmc_real_t value;
    const lmmc_real_t exact = sqrt(3.14159265358979323846);
    const lmmc_status_t status = lmmc_quad_gauss_hermite(
        fn_one, NULL, 1, &value);
    printf("GH order=1, f=1: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
           status, value, exact, fabs(value - exact));
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(value, exact, 1e-10));
}

static void test_hermite_quadratic(void **state) {
    (void)state;
    lmmc_real_t value;
    const lmmc_real_t exact = sqrt(3.14159265358979323846) / 2.0;
    const lmmc_status_t status = lmmc_quad_gauss_hermite(
        fn_x2, NULL, 5, &value);
    printf("GH order=5, f=x^2: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
           status, value, exact, fabs(value - exact));
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(value, exact, 1e-10));
}

static void test_hermite_failures(void **state) {
    (void)state;
    lmmc_real_t value;
    assert_int_equal(
        lmmc_quad_gauss_hermite(NULL, NULL, 5, &value),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_gauss_hermite(fn_one, NULL, 0, &value),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_gauss_hermite(fn_one, NULL, 21, &value),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_gauss_hermite(fn_nan, NULL, 5, &value),
        LMMC_STATUS_NUMERICAL_FAILURE);
}
static void test_hermite_iteration_exhaustion(void **state) {
    (void)state;
    size_t calls = 0;
    lmmc_real_t value = 17.0;
    assert_int_equal(
        lmmc_quad_gauss_hermite_with_iteration_limit_for_test(
            fn_counted_one, &calls, 5, 0, &value),
        LMMC_STATUS_CONVERGENCE_FAILED);
    assert_int_equal(calls, 0);
    assert_true(value == 17.0);
}


static void test_laguerre_constant(void **state) {
    (void)state;
    lmmc_real_t value;
    const lmmc_status_t status = lmmc_quad_gauss_laguerre(
        fn_one, NULL, 1, &value);
    printf("GL order=1, f=1: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
           status, value, 1.0, fabs(value - 1.0));
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(value, 1.0, 1e-10));
}

static void test_laguerre_linear(void **state) {
    (void)state;
    lmmc_real_t value;
    const lmmc_status_t status = lmmc_quad_gauss_laguerre(
        fn_x, NULL, 5, &value);
    printf("GL order=5, f=x: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
           status, value, 1.0, fabs(value - 1.0));
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(value, 1.0, 1e-10));
}

static void test_laguerre_invalid(void **state) {
    (void)state;
    lmmc_real_t value;
    assert_int_equal(
        lmmc_quad_gauss_laguerre(NULL, NULL, 5, &value),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_gauss_laguerre(fn_one, NULL, 0, &value),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_gauss_laguerre(fn_one, NULL, 21, &value),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_gauss_laguerre(fn_nan, NULL, 5, &value),
        LMMC_STATUS_NUMERICAL_FAILURE);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_hermite_constant),
        cmocka_unit_test(test_hermite_quadratic),
        cmocka_unit_test(test_hermite_failures),
        cmocka_unit_test(test_hermite_iteration_exhaustion),
        cmocka_unit_test(test_laguerre_constant),
        cmocka_unit_test(test_laguerre_linear),
        cmocka_unit_test(test_laguerre_invalid),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
