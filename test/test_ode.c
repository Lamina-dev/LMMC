/**
 * @file test_ode.c
 * 针对 LMMC 中 ode 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static double lmmc_pi_value(void) {
    return 3.14159265358979323846;
}

static lmmc_status_t rhs_exp(double t, const double *y, double *y_prime, size_t dim, void *user_data) {
    (void)t;
    (void)user_data;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    y_prime[0] = y[0];
    return LMMC_STATUS_OK;
}

static lmmc_status_t rhs_decay(double t, const double *y, double *y_prime, size_t dim, void *user_data) {
    (void)t;
    (void)user_data;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    y_prime[0] = -2.0 * y[0];
    return LMMC_STATUS_OK;
}

static lmmc_status_t rhs_nan(double t, const double *y, double *y_prime, size_t dim, void *user_data) {
    (void)t;
    (void)y;
    (void)user_data;
    if (dim == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    y_prime[0] = NAN;
    return LMMC_STATUS_OK;
}

static lmmc_status_t rhs_fail(double t, const double *y, double *y_prime, size_t dim, void *user_data) {
    (void)t;
    (void)y;
    (void)y_prime;
    (void)dim;
    (void)user_data;
    return LMMC_STATUS_INVALID_ARGUMENT;
}

typedef struct {
    size_t calls;
} rhs_failure_context_t;

static lmmc_status_t rhs_fail_counted(double t, const double *y, double *y_prime,
                                      size_t dim, void *user_data) {
    rhs_failure_context_t *context = (rhs_failure_context_t *)user_data;
    (void)t;
    (void)y;
    (void)y_prime;
    (void)dim;
    context->calls++;
    return LMMC_STATUS_INVALID_ARGUMENT;
}

static lmmc_status_t rhs_harmonic(double t, const double *y, double *y_prime, size_t dim, void *user_data) {
    (void)t;
    (void)user_data;
    if (dim != 2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    y_prime[0] = y[1];
    y_prime[1] = -y[0];
    return LMMC_STATUS_OK;
}

static void test_default_config_arguments(void **state) {
    (void)state;
    lmmc_ode_config_t config = {0};
    lmmc_ode_config_t *cfg = &config;
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_ode_default_config(0.0, 1.0, 0, cfg);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_ode_default_config(1.0, 1.0, 1, cfg);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_ode_default_config(0.0, 1.0, 1, cfg);
    assert_int_equal(st, LMMC_STATUS_OK);
}

static void test_euler_exponential(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(lmmc_test_nearly_equal(y[0], exp(1.0), 2e-2));
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_NONE);
}

static void test_rk4_exponential(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    lmmc_ode_config_t rk4_cfg = cfg;
    rk4_cfg.initial_step = 0.05;
    rk4_cfg.min_step = 0.05;
    rk4_cfg.max_step = 0.05;
    rk4_cfg.max_steps = 100;

    st = lmmc_ode_rk4_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &rk4_cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(lmmc_test_nearly_equal(y[0], exp(1.0), 1e-5));
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_NONE);
}

static void test_rk45_decay(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    lmmc_ode_config_t adapt_cfg = cfg;
    adapt_cfg.initial_step = 0.2;
    adapt_cfg.min_step = 1e-8;
    adapt_cfg.max_step = 0.2;
    adapt_cfg.abs_tol = 1e-12;
    adapt_cfg.rel_tol = 1e-10;
    adapt_cfg.max_steps = 10000;

    st = lmmc_ode_rk45_solve(rhs_decay, NULL, 1, 0.0, 1.0, y, &adapt_cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(lmmc_test_nearly_equal(y[0], exp(-2.0), 1e-7));
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_NONE);
}

static void test_harmonic_period(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[2] = {1.0, 0.0};
    lmmc_ode_config_t adapt_cfg = cfg;
    adapt_cfg.initial_step = 0.1;
    adapt_cfg.min_step = 1e-8;
    adapt_cfg.max_step = 0.2;
    adapt_cfg.abs_tol = 1e-10;
    adapt_cfg.rel_tol = 1e-9;
    adapt_cfg.max_steps = 200000;

    st = lmmc_ode_rk45_solve(rhs_harmonic, NULL, 2, 0.0, 2.0 * lmmc_pi_value(), y, &adapt_cfg, &result);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(lmmc_test_nearly_equal(y[0], 1.0, 1e-5));
    assert_true(lmmc_test_nearly_equal(y[1], 0.0, 1e-5));
}

static void test_invalid_dimension(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    st = lmmc_ode_euler_solve(rhs_exp, NULL, 0, 0.0, 1.0, y, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_INVALID_DIMENSION);
}

static void test_reverse_time_rejection(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 1.0, 0.0, y, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_INVALID_STEP);
}

static void test_null_callback(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    st = lmmc_ode_euler_solve(NULL, NULL, 1, 0.0, 1.0, y, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_null_state(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, NULL, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_null_result(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    double y[1] = {1.0};
    st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &cfg, NULL);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_negative_tolerance(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    lmmc_ode_config_t bad_cfg = cfg;
    bad_cfg.abs_tol = -1.0;
    st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &bad_cfg, &result);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT);
}

static void test_step_bounds(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    lmmc_ode_config_t bad_cfg = cfg;
    bad_cfg.min_step = 0.5;
    bad_cfg.max_step = 0.1;
    st = lmmc_ode_rk4_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &bad_cfg, &result);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_INVALID_STEP);
}

static void test_zero_step(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    lmmc_ode_config_t bad_cfg = cfg;
    bad_cfg.initial_step = 0.0;
    st = lmmc_ode_rk4_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &bad_cfg, &result);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_INVALID_STEP);
}

static void test_zero_step_budget(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    lmmc_ode_config_t bad_cfg = cfg;
    bad_cfg.max_steps = 0;
    st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &bad_cfg, &result);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_INVALID_STEP);
}

static void test_euler_callback_failure(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    lmmc_status_t st = lmmc_ode_euler_solve(
        rhs_fail, NULL, 1, 0.0, 1.0, y, &cfg, &result);

    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_RHS_EVAL_FAILED);
}

static void test_rk4_callback_failure(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_ode_result_t result = {0};
    rhs_failure_context_t context = {0};
    double y[1] = {1.0};
    lmmc_status_t st = lmmc_ode_rk4_solve(
        rhs_fail_counted, &context, 1, 0.0, 1.0, y, &cfg, &result);

    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_RHS_EVAL_FAILED);
    assert_int_equal(context.calls, 1);
    assert_int_equal(result.num_rhs_evals, 1);
    assert_int_equal(result.num_steps, 0);
    assert_true(result.final_t == 0.0);
    assert_true(y[0] == 1.0);
}

static void test_nonfinite_callback(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    st = lmmc_ode_rk4_solve(rhs_nan, NULL, 1, 0.0, 1.0, y, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_NUMERICAL_ISSUE);
}

static void test_nonfinite_state(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {NAN};
    st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &cfg, &result);
    assert_int_equal(st, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_NUMERICAL_ISSUE);
}

static void test_exhausted_steps(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    lmmc_ode_config_t hard_cfg = cfg;
    hard_cfg.initial_step = 1e-4;
    hard_cfg.min_step = 1e-4;
    hard_cfg.max_step = 1e-4;
    hard_cfg.max_steps = 1;
    st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &hard_cfg, &result);
    assert_int_equal(st, LMMC_STATUS_CONVERGENCE_FAILED);
    assert_false(result.converged);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_MAX_STEPS);
}

static void test_zero_tolerance(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_ode_result_t result = {0};
    double y[1] = {1.0};
    lmmc_ode_config_t hard_cfg = cfg;
    hard_cfg.initial_step = 1.0;
    hard_cfg.min_step = 1.0;
    hard_cfg.max_step = 1.0;
    hard_cfg.abs_tol = 0.0;
    hard_cfg.rel_tol = 0.0;
    hard_cfg.max_steps = 100;
    st = lmmc_ode_rk45_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &hard_cfg, &result);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(result.failure_reason, LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT);
}

static void test_failure_strings(void **state) {
    (void)state;
    assert_non_null(lmmc_ode_failure_string(LMMC_ODE_FAILURE_MAX_STEPS));
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_default_config_arguments),
        cmocka_unit_test(test_euler_exponential),
        cmocka_unit_test(test_rk4_exponential),
        cmocka_unit_test(test_rk45_decay),
        cmocka_unit_test(test_harmonic_period),
        cmocka_unit_test(test_invalid_dimension),
        cmocka_unit_test(test_reverse_time_rejection),
        cmocka_unit_test(test_null_callback),
        cmocka_unit_test(test_null_state),
        cmocka_unit_test(test_null_result),
        cmocka_unit_test(test_negative_tolerance),
        cmocka_unit_test(test_step_bounds),
        cmocka_unit_test(test_zero_step),
        cmocka_unit_test(test_zero_step_budget),
        cmocka_unit_test(test_euler_callback_failure),
        cmocka_unit_test(test_rk4_callback_failure),
        cmocka_unit_test(test_nonfinite_callback),
        cmocka_unit_test(test_nonfinite_state),
        cmocka_unit_test(test_exhausted_steps),
        cmocka_unit_test(test_zero_tolerance),
        cmocka_unit_test(test_failure_strings),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
