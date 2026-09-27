/**
 * @file test_optimize_broyden.c
 * @brief Broyden 非线性求解测试。
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
 * @brief 非线性系统 F_0(x) = x_0^2 + x_1 - 3，F_1(x) = x_0 + x_1^2 - 5。
 * 目标解为 (1, 2)：1 + 2 - 3 = 0，1 + 4 - 5 = 0。
 */
static lmmc_status_t broyden_nonlinear_F(const lmmc_vec_t *x, lmmc_vec_t *F, void *user_data) {
    (void)user_data;
    F->data[0] = x->data[0] * x->data[0] + x->data[1] - 3.0;
    F->data[1] = x->data[0] + x->data[1] * x->data[1] - 5.0;
    return LMMC_STATUS_OK;
}
static void test_broyden_nonlinear(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_t *x = &fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    x->data[0] = 0.5;
    x->data[1] = 1.5;

    st = lmmc_optimize_default_config(&cfg);
    assert_false(st != LMMC_STATUS_OK);
    cfg.abs_tol = 1e-10;
    cfg.max_iter = 1000;

    st = lmmc_nleq_broyden(broyden_nonlinear_F, NULL, x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK);

    assert_true(result.converged);

    assert_false(fabs(x->data[0] - 1.0) > 1e-6 || fabs(x->data[1] - 2.0) > 1e-6);
}

int main(void) {
    vector_fixture_t test_broyden_nonlinear_state = {.n = 2};
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate_setup_teardown(test_broyden_nonlinear, setup_vector, teardown_vector, &test_broyden_nonlinear_state),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
