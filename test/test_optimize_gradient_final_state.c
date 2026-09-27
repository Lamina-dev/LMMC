/**
 * @file test_optimize_gradient_final_state.c
 * @brief 梯度法最终迭代状态与失败原因测试。
 */
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "lmmc/optimize.h"
#include "test_common.h"
static lmmc_real_t scaled_square_obj(const lmmc_vec_t *x, void *user_data) {
    const lmmc_real_t scale = *(const lmmc_real_t *)user_data;
    return 0.5 * scale * x->data[0] * x->data[0];
}

static lmmc_status_t scaled_square_grad(
    const lmmc_vec_t *x, lmmc_vec_t *grad, void *user_data) {
    const lmmc_real_t scale = *(const lmmc_real_t *)user_data;
    grad->data[0] = scale * x->data[0];
    return LMMC_STATUS_OK;
}

static lmmc_real_t absolute_value_obj(const lmmc_vec_t *x, void *user_data) {
    (void)user_data;
    return fabs(x->data[0]);
}

static lmmc_status_t absolute_value_grad(
    const lmmc_vec_t *x, lmmc_vec_t *grad, void *user_data) {
    (void)user_data;
    if (x->data[0] == 0.0)
        return LMMC_STATUS_NUMERICAL_FAILURE;
    grad->data[0] = x->data[0] > 0.0 ? 1.0 : -1.0;
    return LMMC_STATUS_OK;
}
typedef lmmc_status_t (*gradient_solver_t)(
    lmmc_opt_obj_t, lmmc_opt_grad_t, void *, lmmc_vec_t *,
    const lmmc_optimize_config_t *, lmmc_optimize_result_t *);
typedef struct {
    size_t n;
    lmmc_vec_t x;
    gradient_solver_t solver;
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

static void test_gradient_converged(void **state) {
    vector_fixture_t *fixture = *state;
    gradient_solver_t solver = fixture->solver;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};

    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 1;
    lmmc_real_t scale = 1.0;
    lmmc_status_t st;
    x.data[0] = 1.0;
    st = solver(scaled_square_obj, scaled_square_grad, &scale,
                &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || !result.converged ||
                 result.failure_reason != LMMC_OPT_FAILURE_NONE ||
                 result.num_iter != 1 || x.data[0] != 0.0 ||
                 result.final_residual != 0.0);
}
static void test_gradient_exhausted(void **state) {
    vector_fixture_t *fixture = *state;
    gradient_solver_t solver = fixture->solver;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};

    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 1;
    lmmc_real_t scale = 0.5;
    lmmc_status_t st;
    x.data[0] = 1.0;
    st = solver(scaled_square_obj, scaled_square_grad, &scale,
                &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || result.converged ||
                 result.failure_reason != LMMC_OPT_FAILURE_MAX_ITER ||
                 result.num_iter != 1 || x.data[0] != 0.5 ||
                 result.final_residual != 0.25);
}
static void test_gradient_undefined(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};

    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 1;
    x.data[0] = 1.0;
    assert_false(lmmc_minimize_gradient_descent(
                     absolute_value_obj, absolute_value_grad, NULL, &x, &cfg, &result) != LMMC_STATUS_OK ||
                 result.converged || result.failure_reason != LMMC_OPT_FAILURE_NUMERICAL_ISSUE ||
                 result.num_iter != 1 || x.data[0] != 0.0 || !isnan(result.final_residual));
}

int main(void) {
    vector_fixture_t lbfgs_converged = {.n = 1, .solver = lmmc_minimize_lbfgs};
    vector_fixture_t lbfgs_exhausted = {.n = 1, .solver = lmmc_minimize_lbfgs};
    vector_fixture_t descent_converged = {.n = 1, .solver = lmmc_minimize_gradient_descent};
    vector_fixture_t descent_exhausted = {.n = 1, .solver = lmmc_minimize_gradient_descent};
    vector_fixture_t undefined = {.n = 1};
    const struct CMUnitTest tests[] = {
        {"lbfgs_converged", test_gradient_converged, setup_vector, teardown_vector, &lbfgs_converged},
        {"lbfgs_exhausted", test_gradient_exhausted, setup_vector, teardown_vector, &lbfgs_exhausted},
        {"descent_converged", test_gradient_converged, setup_vector, teardown_vector, &descent_converged},
        {"descent_exhausted", test_gradient_exhausted, setup_vector, teardown_vector, &descent_exhausted},
        cmocka_unit_test_prestate_setup_teardown(test_gradient_undefined, setup_vector, teardown_vector, &undefined),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
