#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static lmmc_status_t rhs_constant_components(double t, const double *y, double *yp,
                                             size_t dim, void *ud) {
    const double *rates = (const double *)ud;
    size_t i;
    (void)t;
    (void)y;
    for (i = 0; i < dim; ++i)
        yp[i] = rates[i];
    return LMMC_STATUS_OK;
}

static lmmc_status_t jac_constant_components(double t, const double *y, double *jac,
                                             size_t dim, void *ud) {
    (void)t;
    (void)y;
    (void)ud;
    memset(jac, 0, dim * dim * sizeof(*jac));
    return LMMC_STATUS_OK;
}

typedef struct {
    double rate;
    int exponent;
    int analytic;
    int crossing;
} scalar_case_t;

static void test_sdirk_scaled_scalar(void **state) {
    const scalar_case_t *params = *state;
    double rate = params->rate;
    int exponent = params->exponent;
    int analytic = params->analytic;
    int crossing = params->crossing;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t result;
    double slope = scalbn(rate, exponent);
    double start = crossing ? -0.5 * slope : 0.0;
    double y = start;
    double tolerance = scalbn(fabs(rate) * 1e-5, exponent);
    lmmc_status_t st;
    assert_false(lmmc_ode_default_config(0.0, 1.0, 1, &cfg) != LMMC_STATUS_OK);
    cfg.abs_tol = tolerance;
    cfg.rel_tol = 0.0;
    cfg.initial_step = cfg.max_step = 0.25;
    cfg.min_step = 1e-8;
    cfg.max_steps = 16;
    cfg.jacobian = analytic ? jac_constant_components : NULL;
    st = lmmc_ode_sdirk4_solve(rhs_constant_components, &slope, 1,
                               0.0, 1.0, &y, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || !result.converged || result.final_t != 1.0 ||
                 !lmmc_test_nearly_equal(y, start + slope, tolerance));
}

static void test_sdirk_scaled_components(void **state) {
    int analytic = *(int *)*state;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t result;
    double slopes[2] = {1e-13, -2.0};
    double y[2] = {0.0, 1.0};
    assert_false(lmmc_ode_default_config(0.0, 1.0, 2, &cfg) != LMMC_STATUS_OK);
    cfg.abs_tol = 1e-18;
    cfg.rel_tol = 1e-12;
    cfg.initial_step = cfg.max_step = 0.25;
    cfg.min_step = 1e-8;
    cfg.max_steps = 16;
    cfg.jacobian = analytic ? jac_constant_components : NULL;
    assert_false(lmmc_ode_sdirk4_solve(rhs_constant_components, slopes, 2,
                                       0.0, 1.0, y, &cfg, &result) != LMMC_STATUS_OK ||
                 !result.converged || result.final_t != 1.0 ||
                 !lmmc_test_nearly_equal(y[0], 1e-13, 1e-18) ||
                 !lmmc_test_nearly_equal(y[1], -1.0, 1e-12));
}

static lmmc_status_t rhs_singular_stage(double t, const double *y, double *yp,
                                        size_t dim, void *ud) {
    (void)ud;
    if (dim != 1)
        return LMMC_STATUS_INVALID_ARGUMENT;
    yp[0] = t <= 0.25 ? 1.0 : 16.0 * y[0];
    return LMMC_STATUS_OK;
}

static lmmc_status_t jac_singular_stage(double t, const double *y, double *jac,
                                        size_t dim, void *ud) {
    (void)y;
    (void)ud;
    if (dim != 1)
        return LMMC_STATUS_INVALID_ARGUMENT;
    jac[0] = t <= 0.25 ? 0.0 : 16.0;
    return LMMC_STATUS_OK;
}

static void test_sdirk_nonstationary_zero(void **state) {
    (void)state;
    lmmc_ode_config_t cfg;
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 0.0;
    cfg.initial_step = cfg.min_step = cfg.max_step = 0.25;
    cfg.max_steps = 16;
    cfg.jacobian = jac_constant_components;
    cfg.rel_tol = 1e-8;
    lmmc_ode_result_t result;
    double slope = 1e-13;
    double y = 0.0;
    lmmc_status_t st;
    cfg.abs_tol = 0.0;
    st = lmmc_ode_sdirk4_solve(rhs_constant_components, &slope, 1,
                               0.0, 1.0, &y, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || !result.converged || result.final_t != 1.0 ||
                 !lmmc_test_nearly_equal(y, 1e-13, 1e-21));
}

static void test_sdirk_zero_state(void **state) {
    (void)state;
    lmmc_ode_config_t cfg;
    assert_int_equal(lmmc_ode_default_config(0.0, 1.0, 1, &cfg), LMMC_STATUS_OK);
    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 0.0;
    cfg.initial_step = cfg.min_step = cfg.max_step = 0.25;
    cfg.max_steps = 16;
    cfg.jacobian = jac_constant_components;
    lmmc_ode_result_t result;
    double y;
    lmmc_status_t st;
    {
        double slope = 0.0;
        cfg.jacobian = jac_constant_components;
        cfg.abs_tol = 0.0;
        cfg.rel_tol = 1e-8;
        y = 0.0;
        st = lmmc_ode_sdirk4_solve(rhs_constant_components, &slope, 1,
                                   0.0, 1.0, &y, &cfg, &result);
        assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE || result.converged ||
                     result.num_steps != 0 || result.final_t != 0.0 || y != 0.0);
        cfg.abs_tol = 1e-18;
        st = lmmc_ode_sdirk4_solve(rhs_constant_components, &slope, 1,
                                   0.0, 1.0, &y, &cfg, &result);
        assert_false(st != LMMC_STATUS_OK || !result.converged || result.final_t != 1.0 || y != 0.0);
    }
}

static void test_sdirk_failed_stage_preserves_state(void **state) {
    (void)state;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t result;
    double y = 1.0;
    lmmc_status_t st;
    assert_false(lmmc_ode_default_config(0.0, 1.0, 1, &cfg) != LMMC_STATUS_OK);
    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 0.0;
    cfg.initial_step = cfg.min_step = cfg.max_step = 0.25;
    cfg.max_steps = 16;
    cfg.jacobian = jac_singular_stage;
    st = lmmc_ode_sdirk4_solve(rhs_singular_stage, NULL, 1, 0.0, 1.0, &y, &cfg, &result);
    assert_false(st == LMMC_STATUS_OK || result.converged || result.num_steps != 1 ||
                 result.final_t != 0.25 || result.failure_reason != LMMC_ODE_FAILURE_NUMERICAL_ISSUE ||
                 !lmmc_test_nearly_equal(y, 1.25, 1e-14));
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

static scalar_case_t scalar_cases[] = {
    {1e-13, -200, 0, 0},
    {1e-13, -200, 0, 1},
    {1e-13, 0, 0, 0},
    {1e-13, 0, 0, 1},
    {1e-13, 200, 0, 0},
    {1e-13, 200, 0, 1},
    {-1e-13, -200, 0, 0},
    {-1e-13, -200, 0, 1},
    {-1e-13, 0, 0, 0},
    {-1e-13, 0, 0, 1},
    {-1e-13, 200, 0, 0},
    {-1e-13, 200, 0, 1},
    {1.0, -200, 0, 0},
    {1.0, -200, 0, 1},
    {1.0, 0, 0, 0},
    {1.0, 0, 0, 1},
    {1.0, 200, 0, 0},
    {1.0, 200, 0, 1},
    {-1.0, -200, 0, 0},
    {-1.0, -200, 0, 1},
    {-1.0, 0, 0, 0},
    {-1.0, 0, 0, 1},
    {-1.0, 200, 0, 0},
    {-1.0, 200, 0, 1},
    {1e-13, -200, 1, 0},
    {1e-13, -200, 1, 1},
    {1e-13, 0, 1, 0},
    {1e-13, 0, 1, 1},
    {1e-13, 200, 1, 0},
    {1e-13, 200, 1, 1},
    {-1e-13, -200, 1, 0},
    {-1e-13, -200, 1, 1},
    {-1e-13, 0, 1, 0},
    {-1e-13, 0, 1, 1},
    {-1e-13, 200, 1, 0},
    {-1e-13, 200, 1, 1},
    {1.0, -200, 1, 0},
    {1.0, -200, 1, 1},
    {1.0, 0, 1, 0},
    {1.0, 0, 1, 1},
    {1.0, 200, 1, 0},
    {1.0, 200, 1, 1},
    {-1.0, -200, 1, 0},
    {-1.0, -200, 1, 1},
    {-1.0, 0, 1, 0},
    {-1.0, 0, 1, 1},
    {-1.0, 200, 1, 0},
    {-1.0, 200, 1, 1},
};
static int analytic = 1;
static int finite_difference = 0;
static const struct CMUnitTest tests[] = {
    {"scalar_analytic_0_rate_0_exponent_-200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[0]},
    {"scalar_analytic_0_rate_0_exponent_-200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[1]},
    {"scalar_analytic_0_rate_0_exponent_0_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[2]},
    {"scalar_analytic_0_rate_0_exponent_0_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[3]},
    {"scalar_analytic_0_rate_0_exponent_200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[4]},
    {"scalar_analytic_0_rate_0_exponent_200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[5]},
    {"scalar_analytic_0_rate_1_exponent_-200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[6]},
    {"scalar_analytic_0_rate_1_exponent_-200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[7]},
    {"scalar_analytic_0_rate_1_exponent_0_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[8]},
    {"scalar_analytic_0_rate_1_exponent_0_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[9]},
    {"scalar_analytic_0_rate_1_exponent_200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[10]},
    {"scalar_analytic_0_rate_1_exponent_200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[11]},
    {"scalar_analytic_0_rate_2_exponent_-200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[12]},
    {"scalar_analytic_0_rate_2_exponent_-200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[13]},
    {"scalar_analytic_0_rate_2_exponent_0_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[14]},
    {"scalar_analytic_0_rate_2_exponent_0_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[15]},
    {"scalar_analytic_0_rate_2_exponent_200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[16]},
    {"scalar_analytic_0_rate_2_exponent_200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[17]},
    {"scalar_analytic_0_rate_3_exponent_-200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[18]},
    {"scalar_analytic_0_rate_3_exponent_-200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[19]},
    {"scalar_analytic_0_rate_3_exponent_0_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[20]},
    {"scalar_analytic_0_rate_3_exponent_0_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[21]},
    {"scalar_analytic_0_rate_3_exponent_200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[22]},
    {"scalar_analytic_0_rate_3_exponent_200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[23]},
    {"scalar_analytic_1_rate_0_exponent_-200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[24]},
    {"scalar_analytic_1_rate_0_exponent_-200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[25]},
    {"scalar_analytic_1_rate_0_exponent_0_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[26]},
    {"scalar_analytic_1_rate_0_exponent_0_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[27]},
    {"scalar_analytic_1_rate_0_exponent_200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[28]},
    {"scalar_analytic_1_rate_0_exponent_200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[29]},
    {"scalar_analytic_1_rate_1_exponent_-200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[30]},
    {"scalar_analytic_1_rate_1_exponent_-200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[31]},
    {"scalar_analytic_1_rate_1_exponent_0_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[32]},
    {"scalar_analytic_1_rate_1_exponent_0_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[33]},
    {"scalar_analytic_1_rate_1_exponent_200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[34]},
    {"scalar_analytic_1_rate_1_exponent_200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[35]},
    {"scalar_analytic_1_rate_2_exponent_-200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[36]},
    {"scalar_analytic_1_rate_2_exponent_-200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[37]},
    {"scalar_analytic_1_rate_2_exponent_0_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[38]},
    {"scalar_analytic_1_rate_2_exponent_0_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[39]},
    {"scalar_analytic_1_rate_2_exponent_200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[40]},
    {"scalar_analytic_1_rate_2_exponent_200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[41]},
    {"scalar_analytic_1_rate_3_exponent_-200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[42]},
    {"scalar_analytic_1_rate_3_exponent_-200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[43]},
    {"scalar_analytic_1_rate_3_exponent_0_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[44]},
    {"scalar_analytic_1_rate_3_exponent_0_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[45]},
    {"scalar_analytic_1_rate_3_exponent_200_crossing_0", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[46]},
    {"scalar_analytic_1_rate_3_exponent_200_crossing_1", test_sdirk_scaled_scalar, NULL, NULL, &scalar_cases[47]},
    {"components_analytic", test_sdirk_scaled_components, NULL, NULL, &analytic},
    {"components_finite_difference", test_sdirk_scaled_components, NULL, NULL, &finite_difference},
    cmocka_unit_test(test_sdirk_failed_stage_preserves_state),
    cmocka_unit_test(test_sdirk_zero_state),
    cmocka_unit_test(test_sdirk_nonstationary_zero),
};

int main(void) {
    return cmocka_run_group_tests(tests, setup_runtime, teardown_runtime);
}
