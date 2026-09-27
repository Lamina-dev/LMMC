/**
 * @file test_optimize_newton_linear.c
 * @brief Newton 法求解线性系统的收敛测试。
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

typedef struct {
    size_t n;
    const lmmc_real_t *A_data; /**< n x n 行主序对称正定矩阵。 */
    const lmmc_real_t *b_data; /**< 长度为 n 的向量。 */
} linear_system_data_t;

static lmmc_status_t linear_system_F(const lmmc_vec_t *x, lmmc_vec_t *F, void *user_data) {
    linear_system_data_t *sys = (linear_system_data_t *)user_data;
    size_t n = sys->n;
    for (size_t i = 0; i < n; i++) {
        lmmc_real_t sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += sys->A_data[i * n + j] * x->data[j];
        }
        F->data[i] = sum - sys->b_data[i];
    }
    return LMMC_STATUS_OK;
}
static lmmc_status_t linear_system_J(const lmmc_vec_t *x, lmmc_mat_t *J, void *user_data) {
    (void)x;
    linear_system_data_t *sys = (linear_system_data_t *)user_data;
    size_t n = sys->n;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            J->data[i * J->stride + j] = sys->A_data[i * n + j];
        }
    }
    return LMMC_STATUS_OK;
}
static lmmc_real_t check_linear_residual(const lmmc_vec_t *x, const linear_system_data_t *sys) {
    size_t n = sys->n;
    lmmc_real_t max_residual = 0.0;
    for (size_t i = 0; i < n; i++) {
        lmmc_real_t sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += sys->A_data[i * n + j] * x->data[j];
        }
        lmmc_real_t ri = fabs(sum - sys->b_data[i]);
        if (ri > max_residual)
            max_residual = ri;
    }
    return max_residual;
}
static void test_newton_quadratic(void **state) {
    vector_fixture_t *fixture = *state;
    size_t n = 3;
    lmmc_real_t A_data[] = {4.0, 1.0, 0.0,
                            1.0, 3.0, 1.0,
                            0.0, 1.0, 2.0};
    lmmc_real_t b_data[] = {1.0, 2.0, 3.0};

    linear_system_data_t sys = {n, A_data, b_data};

    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    for (size_t i = 0; i < n; i++)
        x.data[i] = 0.0;

    st = lmmc_optimize_default_config(&cfg);
    assert_false(st != LMMC_STATUS_OK);
    cfg.abs_tol = 1e-12;
    cfg.max_iter = 1;

    st = lmmc_nleq_newton(linear_system_F, linear_system_J, &sys, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK);

    assert_true(result.converged);

    assert_false(result.num_iter != 1);

    lmmc_real_t max_residual = check_linear_residual(&x, &sys);

    assert_false(max_residual > 1e-10);
}

int main(void) {
    vector_fixture_t test_newton_quadratic_state = {.n = 3};
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate_setup_teardown(test_newton_quadratic, setup_vector, teardown_vector, &test_newton_quadratic_state),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
