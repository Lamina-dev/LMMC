/**
 * @file test_nonlinear_extended.c
 * 针对 LMMC 中 nonlinear extended 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static double fn_cos_minus_x(double x, void *user_data) {
    (void)user_data;
    return cos(x) - x;
}

static double fn_cubic(double x, void *user_data) {
    (void)user_data;
    return x * x * x - 2.0 * x - 5.0;
}

static double df_cubic(double x, void *user_data) {
    (void)user_data;
    return 3.0 * x * x - 2.0;
}

static double fn_exp_minus_3(double x, void *user_data) {
    (void)user_data;
    return exp(x) - 3.0;
}

static double fn_sin(double x, void *user_data) {
    (void)user_data;
    return sin(x);
}

static double fn_endpoint_zero(double x, void *user_data) {
    (void)user_data;
    return x - 1.0;
}

static double fn_cbrt(double x, void *user_data) {
    (void)user_data;
    return cbrt(x) - 1.0;
}

static double df_cbrt(double x, void *user_data) {
    (void)user_data;
    if (x == 0.0) {
        return 1e300;
    }
    return 1.0 / (3.0 * cbrt(x * x));
}

static void test_cosine_fixed_point(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    double expected_root = 0.7390851332151607;
    st = lmmc_bisection_solve(fn_cos_minus_x, NULL, 0.0, 1.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(lmmc_test_nearly_equal(result.root, expected_root, 1e-10));
    assert_true(result.function_value == fn_cos_minus_x(result.root, NULL));
    assert_true(result.residual_norm == fabs(result.function_value));
}

static void test_newton_cubic(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    double expected_root = 2.0945514815423265;
    st = lmmc_newton_solve(fn_cubic, df_cubic, NULL, 2.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(lmmc_test_nearly_equal(result.root, expected_root, 1e-10));
    assert_true(result.function_value == fn_cubic(result.root, NULL));
    assert_true(result.residual_norm == fabs(result.function_value));
}

static void test_secant_exponential(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    double expected_root = log(3.0);
    st = lmmc_secant_solve(fn_exp_minus_3, NULL, 0.0, 2.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(lmmc_test_nearly_equal(result.root, expected_root, 1e-10));
    assert_true(result.function_value == fn_exp_minus_3(result.root, NULL));
    assert_true(result.residual_norm == fabs(result.function_value));
}

static void test_separate_sine_roots(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    double pi_val = 3.14159265358979323846;

    st = lmmc_bisection_solve(fn_sin, NULL, 2.5, 3.8, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(lmmc_test_nearly_equal(result.root, pi_val, 1e-9));
    assert_true(result.residual_norm == fabs(fn_sin(result.root, NULL)));

    st = lmmc_bisection_solve(fn_sin, NULL, 5.5, 6.8, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(lmmc_test_nearly_equal(result.root, 2.0 * pi_val, 1e-9));
    assert_true(result.residual_norm == fabs(fn_sin(result.root, NULL)));
}

static void test_endpoint_root(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_bisection_solve(fn_endpoint_zero, NULL, 1.0, 3.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(result.root == 1.0);
    assert_true(result.function_value == 0.0);
    assert_true(result.residual_norm == 0.0);
}

static void test_cubic_conditioning(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_newton_solve(fn_cbrt, df_cbrt, NULL, 0.5, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(lmmc_test_nearly_equal(result.root, 1.0, 1e-10));
    assert_true(result.function_value == fn_cbrt(result.root, NULL));
    assert_true(result.residual_norm == fabs(result.function_value));
}

static void test_invalid_bracket(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_bisection_solve(fn_sin, NULL, 0.5, 2.5, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_INVALID_BRACKET);
}

static void test_strict_bisection(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t strict_cfg = cfg;
    strict_cfg.abs_tol = 1e-15;
    strict_cfg.rel_tol = 0.0;
    strict_cfg.max_iter = 200;

    double expected_root = 0.7390851332151607;
    st = lmmc_bisection_solve(fn_cos_minus_x, NULL, 0.0, 1.0, &strict_cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(lmmc_test_nearly_equal(result.root, expected_root, 1e-14));
    assert_true(result.function_value == fn_cos_minus_x(result.root, NULL));
    assert_true(result.residual_norm == fabs(result.function_value));
}

static void test_strict_newton(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t strict_cfg = cfg;
    strict_cfg.abs_tol = 1e-15;
    strict_cfg.rel_tol = 0.0;
    strict_cfg.max_iter = 100;
    st = lmmc_newton_solve(fn_cubic, df_cubic, NULL, 2.0, &strict_cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(lmmc_test_nearly_equal(result.root, 2.0945514815423265, 1e-14));
    assert_true(result.function_value == fn_cubic(result.root, NULL));
    assert_true(result.residual_norm == fabs(result.function_value));
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cosine_fixed_point),
        cmocka_unit_test(test_newton_cubic),
        cmocka_unit_test(test_secant_exponential),
        cmocka_unit_test(test_separate_sine_roots),
        cmocka_unit_test(test_endpoint_root),
        cmocka_unit_test(test_cubic_conditioning),
        cmocka_unit_test(test_invalid_bracket),
        cmocka_unit_test(test_strict_bisection),
        cmocka_unit_test(test_strict_newton),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
