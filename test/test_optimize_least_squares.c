/**
 * @file test_optimize_least_squares.c
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
 * @brief 最小化 ||r(x)||^2，其中 r_i(x) = x_i - c_i。
 * 目标 c = (1, 2, 3)，解为 x = c。
 */
static const lmmc_real_t lm_target[] = {1.0, 2.0, 3.0};

static lmmc_status_t lm_residual(const lmmc_vec_t *x, lmmc_vec_t *F, void *user_data) {
    (void)user_data;
    for (size_t i = 0; i < x->size; i++) {
        F->data[i] = x->data[i] - lm_target[i];
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lm_jacobian(const lmmc_vec_t *x, lmmc_mat_t *J, void *user_data) {
    (void)x;
    (void)user_data;
    size_t n = J->rows;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            J->data[i * J->stride + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return LMMC_STATUS_OK;
}
static void test_levenberg_marquardt(void **state) {
    vector_fixture_t *fixture = *state;
    size_t n = 3;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    for (size_t i = 0; i < n; i++)
        x.data[i] = 0.0;

    st = lmmc_optimize_default_config(&cfg);
    assert_false(st != LMMC_STATUS_OK);
    cfg.abs_tol = 1e-12;
    cfg.max_iter = 1000;

    st = lmmc_minimize_levenberg_marquardt(lm_residual, lm_jacobian, NULL, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK);

    assert_true(result.converged);

    for (size_t i = 0; i < n; i++) {
        assert_false(fabs(x.data[i] - lm_target[i]) > 1e-8);
    }
}

int main(void) {
    vector_fixture_t test_levenberg_marquardt_state = {.n = 3};
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate_setup_teardown(test_levenberg_marquardt, setup_vector, teardown_vector, &test_levenberg_marquardt_state),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
