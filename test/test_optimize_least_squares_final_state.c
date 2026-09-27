/**
 * @file test_optimize_least_squares_final_state.c
 * @brief 优化模块单元测试。
 */
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "lmmc/optimize.h"
#include "test_common.h"
#include "test_optimize_common.h"

typedef struct {
    size_t n;
    lmmc_vec_t x;
} vector_fixture_t;

static int setup_vector(void **state) {
    vector_fixture_t *fixture = *state;
    assert_int_equal(lmmc_vec_create(fixture->n, &fixture->x), LMMC_STATUS_OK);
    return 0;
}

static int teardown_vector(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_destroy(&fixture->x);
    return 0;
}

static lmmc_status_t positive_square_residual(
    const lmmc_vec_t *x, lmmc_vec_t *r, void *user_data) {
    (void)user_data;
    r->data[0] = x->data[0] * x->data[0] + 1.0;
    return LMMC_STATUS_OK;
}

static lmmc_status_t scalar_square_jacobian(
    const lmmc_vec_t *x, lmmc_mat_t *J, void *user_data) {
    (void)user_data;
    J->data[0] = 2.0 * x->data[0];
    return LMMC_STATUS_OK;
}

static lmmc_status_t nonfinite_jacobian(
    const lmmc_vec_t *x, lmmc_mat_t *J, void *user_data) {
    (void)x;
    (void)user_data;
    J->data[0] = INFINITY;
    return LMMC_STATUS_OK;
}
static void test_lm_damped(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 1;
    cfg.lm_damping = 1.0;
    for (int converged = 0; converged < 2; ++converged) {
        x.data[0] = 1.0;
        cfg.abs_tol = converged ? 0.5 : 1e-12;
        st = lmmc_minimize_levenberg_marquardt(
            lmmc_test_identity_system_f, lmmc_test_identity_system_j, NULL, &x, &cfg, &result);
        assert_false(st != LMMC_STATUS_OK || result.converged != converged ||
                     result.failure_reason != (converged ? LMMC_OPT_FAILURE_NONE : LMMC_OPT_FAILURE_MAX_ITER) ||
                     result.num_iter != 1 || x.data[0] != 0.5 || result.final_residual != 0.5);
    }
}
static void test_lm_nonzero_optimum(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 1;
    cfg.lm_damping = 1.0;
    cfg.abs_tol = 1e-12;
    cfg.lm_damping = 1.5;
    for (int updates = 0; updates < 2; ++updates) {
        x.data[0] = updates ? 0.5 : 0.0;
        st = lmmc_minimize_levenberg_marquardt(
            positive_square_residual, scalar_square_jacobian, NULL, &x, &cfg, &result);
        assert_false(st != LMMC_STATUS_OK || !result.converged ||
                     result.failure_reason != LMMC_OPT_FAILURE_NONE ||
                     result.num_iter != (size_t)updates ||
                     x.data[0] != 0.0 || result.final_residual != 1.0);
    }
}
static void test_lm_relative_tolerance(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 1;
    cfg.lm_damping = 1.0;
    cfg.abs_tol = 1e-12;
    x.data[0] = 0.5;
    cfg.lm_damping = 3.5;
    cfg.abs_tol = 0.0;
    cfg.rel_tol = 0.5;
    st = lmmc_minimize_levenberg_marquardt(
        positive_square_residual, scalar_square_jacobian, NULL, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || !result.converged || result.num_iter != 1 ||
                 fabs(x.data[0] - 2.0 / 9.0) > 1e-12 ||
                 !isfinite(result.final_residual) ||
                 fabs(result.final_residual - (x.data[0] * x.data[0] + 1.0)) > 1e-12 ||
                 result.final_residual <= 0.5 * 1.25);
}
static void test_lm_nonfinite_gradient(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 1;
    cfg.lm_damping = 1.0;
    cfg.lm_damping = 3.5;
    cfg.rel_tol = 0.5;
    x.data[0] = 0.0;
    cfg.abs_tol = 2.0;
    st = lmmc_minimize_levenberg_marquardt(
        positive_square_residual, nonfinite_jacobian, NULL, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || result.converged ||
                 result.failure_reason != LMMC_OPT_FAILURE_NUMERICAL_ISSUE ||
                 result.num_iter != 0 || x.data[0] != 0.0 || result.final_residual != 1.0);
}

int main(void) {
    vector_fixture_t test_lm_damped_state = {.n = 1};
    vector_fixture_t test_lm_nonzero_optimum_state = {.n = 1};
    vector_fixture_t test_lm_relative_tolerance_state = {.n = 1};
    vector_fixture_t test_lm_nonfinite_gradient_state = {.n = 1};
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate_setup_teardown(test_lm_damped, setup_vector, teardown_vector, &test_lm_damped_state),
        cmocka_unit_test_prestate_setup_teardown(test_lm_nonzero_optimum, setup_vector, teardown_vector, &test_lm_nonzero_optimum_state),
        cmocka_unit_test_prestate_setup_teardown(test_lm_relative_tolerance, setup_vector, teardown_vector, &test_lm_relative_tolerance_state),
        cmocka_unit_test_prestate_setup_teardown(test_lm_nonfinite_gradient, setup_vector, teardown_vector, &test_lm_nonfinite_gradient_state),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
