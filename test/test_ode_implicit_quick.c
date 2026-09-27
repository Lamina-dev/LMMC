#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static lmmc_status_t rhs_decay(double t, const double *y, double *yp,
                               size_t dim, void *ud) {
    (void)t;
    (void)ud;
    (void)dim;
    yp[0] = -10.0 * y[0];
    return LMMC_STATUS_OK;
}

static lmmc_status_t rhs_coupled_quadratic(double t, const double *y, double *yp,
                                           size_t dim, void *ud) {
    (void)t;
    if (dim != 2)
        return LMMC_STATUS_INVALID_ARGUMENT;
    yp[0] = y[1] * y[1];
    yp[1] = *(const double *)ud;
    return LMMC_STATUS_OK;
}

static lmmc_status_t jac_coupled_quadratic(double t, const double *y, double *jac,
                                           size_t dim, void *ud) {
    (void)t;
    (void)ud;
    if (dim != 2)
        return LMMC_STATUS_INVALID_ARGUMENT;
    jac[0] = jac[2] = jac[3] = 0.0;
    jac[1] = 2.0 * y[1];
    return LMMC_STATUS_OK;
}

static void configure_relative(lmmc_ode_config_t *cfg) {
    assert_int_equal(lmmc_ode_default_config(0.0, 0.1, 2, cfg), LMMC_STATUS_OK);
    cfg->abs_tol = 0.0;
    cfg->rel_tol = 1e-6;
    cfg->initial_step = cfg->min_step = cfg->max_step = 0.1;
    cfg->max_steps = 1;
    cfg->jacobian = jac_coupled_quadratic;
}

static void test_sdirk_relative_nonfinite(void **state) {
    double initial = *(double *)*state;
    lmmc_ode_config_t config;
    configure_relative(&config);
    const lmmc_ode_config_t *cfg = &config;
    lmmc_ode_result_t result;
    double rate = 1e200;
    double y[2] = {initial, 0.0};
    lmmc_status_t st = lmmc_ode_sdirk4_solve(
        rhs_coupled_quadratic, &rate, 2, 0.0, 0.1, y, cfg, &result);
    const int unchanged = isnan(initial) ? isnan(y[0]) : y[0] == initial;
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE || result.converged ||
                 result.num_steps != 0 || result.final_t != 0.0 ||
                 result.failure_reason != LMMC_ODE_FAILURE_NUMERICAL_ISSUE ||
                 !unchanged || y[1] != 0.0);
}

static void test_sdirk_relative_intermediate_scale(void **state) {
    (void)state;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t result;
    double rate = 1.0;
    double y[2] = {0.0, 0.0};
    lmmc_status_t st;
    configure_relative(&cfg);
    /**
     * @brief 首次修正仅改变 y[1]，y[0] 及其尺度仍为零。
     * 后续修正解析 y[0]' = y[1]^2，精确解为 y(t) = (t^3/3, t)。
     */
    st = lmmc_ode_sdirk4_solve(rhs_coupled_quadratic, &rate, 2,
                               0.0, 0.1, y, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || !result.converged || result.num_steps != 1 ||
                 result.final_t != 0.1 || result.failure_reason != LMMC_ODE_FAILURE_NONE ||
                 !lmmc_test_nearly_equal(y[0], 0.001 / 3.0, 1e-15) ||
                 !lmmc_test_nearly_equal(y[1], 0.1, 1e-15));
}

static lmmc_status_t rhs_cyclic_newton(double t, const double *y, double *yp,
                                       size_t dim, void *ud) {
    (void)t;
    (void)ud;
    if (dim != 2)
        return LMMC_STATUS_INVALID_ARGUMENT;
    yp[0] = 0.0;
    yp[1] = 12.0 * y[1] - 64.0 * y[1] * y[1] * y[1] - 2.0;
    return LMMC_STATUS_OK;
}

static lmmc_status_t jac_cyclic_newton(double t, const double *y, double *jac,
                                       size_t dim, void *ud) {
    (void)t;
    (void)ud;
    if (dim != 2)
        return LMMC_STATUS_INVALID_ARGUMENT;
    jac[0] = jac[1] = jac[2] = 0.0;
    jac[3] = 12.0 - 192.0 * y[1] * y[1];
    return LMMC_STATUS_OK;
}

static void test_sdirk_relative_scale_iteration_limit(void **state) {
    (void)state;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t result;
    double y[2] = {0.0, 0.0};
    lmmc_status_t st;
    assert_false(lmmc_ode_default_config(0.0, 1.0, 2, &cfg) != LMMC_STATUS_OK);
    cfg.abs_tol = 0.0;
    cfg.rel_tol = 1e-6;
    cfg.initial_step = cfg.min_step = cfg.max_step = 1.0;
    cfg.max_steps = 1;
    cfg.jacobian = jac_cyclic_newton;
    /**
     * @brief h=1 时首阶段方程为 k^3-2*k+2=0。
     * Newton 在 k=0 与 k=1 间精确循环，静止分量的尺度保持为零。
     */
    st = lmmc_ode_sdirk4_solve(rhs_cyclic_newton, NULL, 2,
                               0.0, 1.0, y, &cfg, &result);
    assert_false(st != LMMC_STATUS_CONVERGENCE_FAILED || result.converged ||
                 result.num_steps != 0 || result.final_t != 0.0 ||
                 result.failure_reason != LMMC_ODE_FAILURE_NUMERICAL_ISSUE ||
                 y[0] != 0.0 || y[1] != 0.0);
}
typedef lmmc_status_t (*implicit_solver_t)(
    lmmc_ode_rhs_t, void *, size_t, lmmc_real_t, lmmc_real_t,
    lmmc_real_t *, const lmmc_ode_config_t *, lmmc_ode_result_t *);

static void test_implicit_decay(void **state) {
    implicit_solver_t solver = *(implicit_solver_t *)*state;
    lmmc_ode_config_t cfg = {0};
    lmmc_ode_result_t result = {0};
    double y = 1.0;
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    cfg.jacobian = NULL;
    assert_int_equal(solver(rhs_decay, NULL, 1, 0.0, 1.0, &y, &cfg, &result), LMMC_STATUS_OK);
    assert_true(result.converged);
}

static void test_implicit_euler_large_step(void **state) {
    (void)state;
    lmmc_ode_config_t cfg = {0};
    lmmc_ode_result_t result = {0};
    double y = 1.0;
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    cfg.jacobian = NULL;
    cfg.initial_step = 0.5;
    cfg.min_step = 0.1;
    cfg.max_step = 1.0;
    lmmc_ode_implicit_euler_solve(rhs_decay, NULL, 1, 0.0, 1.0, &y, &cfg, &result);
    assert_false(fabs(y) > 2.0);
}

static int setup_runtime(void **state) {
    (void)state;
    assert_int_equal(lmmc_init(), LMMC_STATUS_OK);
    return 0;
}

static int teardown_runtime(void **state) {
    (void)state;
    assert_int_equal(lmmc_deinit(), LMMC_STATUS_OK);
    return 0;
}

int main(void) {
    implicit_solver_t euler = lmmc_ode_implicit_euler_solve;
    implicit_solver_t trapezoidal = lmmc_ode_trapezoidal_solve;
    implicit_solver_t sdirk = lmmc_ode_sdirk4_solve;
    implicit_solver_t rosenbrock = lmmc_ode_rosenbrock_grk4t_solve;
    double zero = 0.0;
    double nan = NAN;
    const struct CMUnitTest tests[] = {
        {"euler_decay", test_implicit_decay, NULL, NULL, &euler},
        {"trapezoidal_decay", test_implicit_decay, NULL, NULL, &trapezoidal},
        {"sdirk_decay", test_implicit_decay, NULL, NULL, &sdirk},
        {"rosenbrock_decay", test_implicit_decay, NULL, NULL, &rosenbrock},
        cmocka_unit_test(test_implicit_euler_large_step),
        cmocka_unit_test(test_sdirk_relative_intermediate_scale),
        cmocka_unit_test(test_sdirk_relative_scale_iteration_limit),
        {"relative_overflow", test_sdirk_relative_nonfinite, NULL, NULL, &zero},
        {"relative_nan", test_sdirk_relative_nonfinite, NULL, NULL, &nan},
    };
    return cmocka_run_group_tests(tests, setup_runtime, teardown_runtime);
}
