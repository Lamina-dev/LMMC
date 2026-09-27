/**
 * @file test_nonlinear.c
 * 针对 LMMC 中 nonlinear 相关接口的单元测试。
 */
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static double lmmc_test_fn_sqrt2(double x, void *user_data) {
    (void)user_data;
    return x * x - 2.0;
}

static double lmmc_test_df_sqrt2(double x, void *user_data) {
    (void)user_data;
    return 2.0 * x;
}

static double lmmc_test_fn_linear(double x, void *user_data) {
    (void)user_data;
    return x - 1.0;
}

static double lmmc_test_df_linear(double x, void *user_data) {
    (void)user_data;
    (void)x;
    return 1.0;
}
static double lmmc_test_fn_tiny_scaled_linear(double x, void *user_data) {
    (void)user_data;
    return 1.0e-20 * (x - 1.0);
}

static double lmmc_test_fn_nan(double x, void *user_data) {
    (void)user_data;
    if (x > 1.5) {
        return NAN;
    }
    return x - 1.0;
}

static double lmmc_test_fn_flat(double x, void *user_data) {
    (void)user_data;
    return x * x * x + 1.0;
}

static double lmmc_test_df_flat(double x, void *user_data) {
    (void)user_data;
    return 3.0 * x * x;
}

static double lmmc_test_fn_const(double x, void *user_data) {
    (void)x;
    (void)user_data;
    return 1.0;
}

static double lmmc_test_fn_exp_pos(double x, void *user_data) {
    (void)user_data;
    return exp(x);
}

static double lmmc_test_df_exp_pos(double x, void *user_data) {
    (void)user_data;
    return exp(x);
}

static void test_default_config_null(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_nonlinear_default_config(NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_bisection_roots(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_bisection_solve(lmmc_test_fn_sqrt2, NULL, 0.0, 2.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(lmmc_test_nearly_equal(result.root, 1.4142135623730951, 2e-10));
    assert_true(result.function_value == lmmc_test_fn_sqrt2(result.root, NULL));
    assert_true(result.residual_norm == fabs(result.function_value));

    st = lmmc_bisection_solve(lmmc_test_fn_linear, NULL, 1.0, 3.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(result.root == 1.0);
    assert_true(result.function_value == 0.0);
    assert_true(result.residual_norm == 0.0);
}

static void test_bisection_invalid_bracket(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_bisection_solve(lmmc_test_fn_linear, NULL, 2.0, 3.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_INVALID_BRACKET);
}

static void test_bisection_iteration_budget(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t cfg_hard = {0};
    cfg_hard = cfg;
    cfg_hard.abs_tol = 0.0;
    cfg_hard.rel_tol = 0.0;
    cfg_hard.max_iter = 1;

    st = lmmc_bisection_solve(lmmc_test_fn_sqrt2, NULL, 0.0, 2.0, &cfg_hard, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_false(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_MAX_ITER);
}

static void test_bisection_nonfinite_callback(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_bisection_solve(lmmc_test_fn_nan, NULL, 0.0, 2.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE);
}

static void test_bisection_arguments(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t cfg_hard = {0};
    st = lmmc_bisection_solve(NULL, NULL, 0.0, 2.0, &cfg, &result);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_bisection_solve(lmmc_test_fn_sqrt2, NULL, 2.0, 0.0, &cfg, &result);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_bisection_solve(lmmc_test_fn_sqrt2, NULL, 0.0, 2.0, &cfg, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    cfg_hard = cfg;
    cfg_hard.max_iter = 0;
    st = lmmc_bisection_solve(lmmc_test_fn_sqrt2, NULL, 0.0, 2.0, &cfg_hard, &result);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_newton_derivative_modes(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, lmmc_test_df_sqrt2, NULL, 1.5, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(lmmc_test_nearly_equal(result.root, 1.4142135623730951, 1e-10));
    assert_true(result.function_value == lmmc_test_fn_sqrt2(result.root, NULL));
    assert_true(result.residual_norm == fabs(result.function_value));

    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, NULL, NULL, 1.5, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(lmmc_test_nearly_equal(result.root, 1.4142135623730951, 1e-8));
    assert_true(result.function_value == lmmc_test_fn_sqrt2(result.root, NULL));
    assert_true(result.residual_norm == fabs(result.function_value));
}

static void test_newton_zero_derivative(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_newton_solve(lmmc_test_fn_flat, lmmc_test_df_flat, NULL, 0.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_false(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_ZERO_DERIVATIVE);
}

static void test_newton_iteration_budget(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t cfg_hard = {0};
    cfg_hard = cfg;
    cfg_hard.max_iter = 2;
    cfg_hard.abs_tol = 0.0;
    cfg_hard.rel_tol = 0.0;
    st = lmmc_newton_solve(lmmc_test_fn_exp_pos, lmmc_test_df_exp_pos, NULL, 0.0, &cfg_hard, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_false(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_MAX_ITER);
}

static void test_newton_minimum_step(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t cfg_hard = {0};
    cfg_hard = cfg;
    cfg_hard.min_step = 0.6;
    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, lmmc_test_df_sqrt2, NULL, 1.0, &cfg_hard, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_false(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_SINGULAR_STEP);
}

static void test_newton_nonfinite_callback(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_newton_solve(lmmc_test_fn_nan, NULL, NULL, 2.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE);
}

static void test_newton_arguments(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t cfg_hard = {0};
    st = lmmc_newton_solve(NULL, lmmc_test_df_sqrt2, NULL, 1.0, &cfg, &result);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, lmmc_test_df_sqrt2, NULL, INFINITY, &cfg, &result);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, lmmc_test_df_sqrt2, NULL, 1.0, &cfg, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    cfg_hard = cfg;
    cfg_hard.max_iter = 0;
    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, lmmc_test_df_sqrt2, NULL, 1.0, &cfg_hard, &result);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_secant_roots(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_secant_solve(lmmc_test_fn_sqrt2, NULL, 1.0, 2.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(lmmc_test_nearly_equal(result.root, 1.4142135623730951, 1e-10));
    assert_true(result.function_value == lmmc_test_fn_sqrt2(result.root, NULL));
    assert_true(result.residual_norm == fabs(result.function_value));

    st = lmmc_secant_solve(lmmc_test_fn_linear, NULL, 0.0, 2.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(result.root == 1.0);
    assert_true(result.function_value == 0.0);
    assert_true(result.residual_norm == 0.0);
}

static void test_secant_tiny_scale(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t cfg_hard = {0};
    cfg_hard = cfg;
    cfg_hard.abs_tol = 0.0;
    st = lmmc_secant_solve(
        lmmc_test_fn_tiny_scaled_linear, NULL, 0.0, 2.0,
        &cfg_hard, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(result.root == 1.0);
    assert_true(result.function_value == 0.0);
    assert_true(result.residual_norm == 0.0);
}

static void test_secant_extreme_bracket(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t cfg_hard = {0};
    cfg_hard = cfg;
    cfg_hard.abs_tol = 0.0;
    st = lmmc_secant_solve(
        lmmc_test_fn_linear, NULL, -DBL_MAX, DBL_MAX, &cfg_hard, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(result.root == 1.0);
    assert_true(result.function_value == 0.0);
    assert_true(result.residual_norm == 0.0);
}

static void test_secant_singular_step(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_secant_solve(lmmc_test_fn_const, NULL, 0.0, 2.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_false(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_SINGULAR_STEP);
}

static void test_secant_iteration_budget(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t cfg_hard = {0};
    cfg_hard = cfg;
    cfg_hard.max_iter = 2;
    cfg_hard.abs_tol = 0.0;
    cfg_hard.rel_tol = 0.0;
    st = lmmc_secant_solve(lmmc_test_fn_exp_pos, NULL, 0.0, 1.0, &cfg_hard, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_false(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_MAX_ITER);
}

static void test_secant_minimum_step(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t cfg_hard = {0};
    cfg_hard = cfg;
    cfg_hard.min_step = 1.0;
    st = lmmc_secant_solve(lmmc_test_fn_sqrt2, NULL, 1.0, 2.0, &cfg_hard, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_false(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_SINGULAR_STEP);
}

static void test_secant_nonfinite_callback(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_secant_solve(lmmc_test_fn_nan, NULL, 0.0, 2.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE);
}

static void test_secant_arguments(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    lmmc_nonlinear_config_t cfg_hard = {0};
    st = lmmc_secant_solve(NULL, NULL, 1.0, 2.0, &cfg, &result);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_secant_solve(lmmc_test_fn_sqrt2, NULL, 1.0, 1.0, &cfg, &result);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_secant_solve(lmmc_test_fn_sqrt2, NULL, 1.0, 2.0, &cfg, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    cfg_hard = cfg;
    cfg_hard.max_iter = 0;
    st = lmmc_secant_solve(lmmc_test_fn_sqrt2, NULL, 1.0, 2.0, &cfg_hard, &result);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_newton_linear(void **state) {
    (void)state;
    lmmc_nonlinear_config_t cfg = {0};
    assert_int_equal(lmmc_nonlinear_default_config(&cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_nonlinear_result_t result = {0};
    st = lmmc_newton_solve(lmmc_test_fn_linear, lmmc_test_df_linear, NULL, 2.0, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_int_equal(result.failure_reason, LMMC_NONLINEAR_FAILURE_NONE);
    assert_true(result.root == 1.0);
    assert_true(result.function_value == 0.0);
    assert_true(result.residual_norm == 0.0);
}

static void test_failure_strings(void **state) {
    (void)state;
    assert_non_null(lmmc_nonlinear_failure_string(LMMC_NONLINEAR_FAILURE_MAX_ITER));
    assert_non_null(lmmc_nonlinear_failure_string((lmmc_nonlinear_failure_t)9999));
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_default_config_null),
        cmocka_unit_test(test_bisection_roots),
        cmocka_unit_test(test_bisection_invalid_bracket),
        cmocka_unit_test(test_bisection_iteration_budget),
        cmocka_unit_test(test_bisection_nonfinite_callback),
        cmocka_unit_test(test_bisection_arguments),
        cmocka_unit_test(test_newton_derivative_modes),
        cmocka_unit_test(test_newton_zero_derivative),
        cmocka_unit_test(test_newton_iteration_budget),
        cmocka_unit_test(test_newton_minimum_step),
        cmocka_unit_test(test_newton_nonfinite_callback),
        cmocka_unit_test(test_newton_arguments),
        cmocka_unit_test(test_secant_roots),
        cmocka_unit_test(test_secant_tiny_scale),
        cmocka_unit_test(test_secant_extreme_bracket),
        cmocka_unit_test(test_secant_singular_step),
        cmocka_unit_test(test_secant_iteration_budget),
        cmocka_unit_test(test_secant_minimum_step),
        cmocka_unit_test(test_secant_nonfinite_callback),
        cmocka_unit_test(test_secant_arguments),
        cmocka_unit_test(test_newton_linear),
        cmocka_unit_test(test_failure_strings),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
