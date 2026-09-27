/**
 * @file test_optimize_newton_boundaries.c
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

static lmmc_status_t boundary_linear_system_F(
    const lmmc_vec_t *x,
    lmmc_vec_t *F,
    void *user_data) {
    (void)user_data;
    F->data[0] = x->data[0] - DBL_MAX * 0.5;
    return LMMC_STATUS_OK;
}
static lmmc_status_t twice_identity_system_J(
    const lmmc_vec_t *x,
    lmmc_mat_t *J,
    void *user_data) {
    (void)x;
    (void)user_data;
    J->data[0] = 2.0;
    return LMMC_STATUS_OK;
}

static void test_newton_extreme_finite_residual(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;
    x.data[0] = 1e200;
    x.data[1] = -1e200;
    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 4;

    st = lmmc_nleq_newton(
        lmmc_test_identity_system_f, lmmc_test_identity_system_j, NULL, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || !result.converged ||
                 x.data[0] != 0.0 || x.data[1] != 0.0);
}

static void test_newton_finite_difference_step(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;
    x.data[0] = 1.0;
    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 1;

    st = lmmc_nleq_newton(
        lmmc_test_scalar_square_system_f, NULL, NULL, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || fabs(x.data[0] - 1.5) > 1e-6);
}
static void test_newton_finite_difference_at_upper_boundary(void **state) {
    vector_fixture_t *fixture = *state;
    const lmmc_real_t expected = DBL_MAX * 0.5;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    x.data[0] = DBL_MAX;
    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 3;
    st = lmmc_nleq_newton(
        boundary_linear_system_F, NULL, NULL, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || !result.converged || x.data[0] != expected);
}

static void test_newton_relative_tolerance(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    x.data[0] = 1.0;
    lmmc_optimize_default_config(&cfg);
    cfg.abs_tol = 0.0;
    cfg.rel_tol = 0.75;
    cfg.max_iter = 2;
    st = lmmc_nleq_newton(
        lmmc_test_identity_system_f, twice_identity_system_J, NULL, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || !result.converged ||
                 result.num_iter != 1 || x.data[0] != 0.5);
}

int main(void) {
    vector_fixture_t test_newton_extreme_finite_residual_state = {.n = 2};
    vector_fixture_t test_newton_finite_difference_step_state = {.n = 1};
    vector_fixture_t test_newton_finite_difference_at_upper_boundary_state = {.n = 1};
    vector_fixture_t test_newton_relative_tolerance_state = {.n = 1};
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate_setup_teardown(test_newton_extreme_finite_residual, setup_vector, teardown_vector, &test_newton_extreme_finite_residual_state),
        cmocka_unit_test_prestate_setup_teardown(test_newton_finite_difference_step, setup_vector, teardown_vector, &test_newton_finite_difference_step_state),
        cmocka_unit_test_prestate_setup_teardown(test_newton_finite_difference_at_upper_boundary, setup_vector, teardown_vector, &test_newton_finite_difference_at_upper_boundary_state),
        cmocka_unit_test_prestate_setup_teardown(test_newton_relative_tolerance, setup_vector, teardown_vector, &test_newton_relative_tolerance_state),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
