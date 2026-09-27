/**
 * @file test_optimize_gradient_descent.c
 * @brief 梯度下降法测试。
 */
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "lmmc/optimize.h"
#include "test_common.h"

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

/**
 * @brief 二次目标 f(x) = 0.5 * x^T * A * x - b^T * x。
 * A = diag(1, 2, ..., n)，b = (1, 1, ..., 1)；
 * 按从 1 起算的下标，极小点为 x_i = 1/i。
 */
typedef struct {
    size_t n;
} quadratic_data_t;

static lmmc_real_t quadratic_obj(const lmmc_vec_t *x, void *user_data) {
    quadratic_data_t *qd = (quadratic_data_t *)user_data;
    size_t n = qd->n;
    lmmc_real_t f = 0.0;
    for (size_t i = 0; i < n; i++) {
        lmmc_real_t ai = (lmmc_real_t)(i + 1);
        f += 0.5 * ai * x->data[i] * x->data[i] - x->data[i];
    }
    return f;
}

static lmmc_status_t quadratic_grad(const lmmc_vec_t *x, lmmc_vec_t *grad, void *user_data) {
    quadratic_data_t *qd = (quadratic_data_t *)user_data;
    size_t n = qd->n;
    for (size_t i = 0; i < n; i++) {
        lmmc_real_t ai = (lmmc_real_t)(i + 1);
        grad->data[i] = ai * x->data[i] - 1.0;
    }
    return LMMC_STATUS_OK;
}
/**
 * @brief 检验梯度下降求二次函数极小点。
 * f(x) = 0.5 * sum_i (i+1)*x_i^2 - sum_i x_i；
 * 极小点为 x_i = 1/(i+1)。
 */
static void test_gradient_descent_quadratic(void **state) {
    vector_fixture_t *fixture = *state;
    size_t n = 5;
    quadratic_data_t qd = {n};
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    for (size_t i = 0; i < n; i++)
        x.data[i] = 2.0;

    st = lmmc_optimize_default_config(&cfg);
    assert_false(st != LMMC_STATUS_OK);
    cfg.abs_tol = 1e-6;
    cfg.max_iter = 200000;

    st = lmmc_minimize_gradient_descent(quadratic_obj, quadratic_grad, &qd, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK);

    assert_true(result.converged);

    for (size_t i = 0; i < n; i++) {
        lmmc_real_t expected = 1.0 / (lmmc_real_t)(i + 1);
        assert_false(fabs(x.data[i] - expected) > 1e-3);
    }
}

int main(void) {
    vector_fixture_t test_gradient_descent_quadratic_state = {.n = 5};
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate_setup_teardown(test_gradient_descent_quadratic, setup_vector, teardown_vector, &test_gradient_descent_quadratic_state),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
