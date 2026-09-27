/**
 * @file test_optimize_lbfgs.c
 * @brief L-BFGS 的 Rosenbrock 收敛测试。
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
    lmmc_vec_t grad;
} vector_fixture_t;

static int setup_vector(void **state) {
    vector_fixture_t *fixture = *state;
    assert_int_equal(lmmc_vec_create(fixture->n, &fixture->x), LMMC_STATUS_OK);
    lmmc_status_t status = lmmc_vec_create(fixture->n, &fixture->grad);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&fixture->x);
        fail_msg("Cannot allocate gradient vector");
    }
    return 0;
}

static int teardown_vector(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_destroy(&fixture->grad);
    lmmc_vec_destroy(&fixture->x);
    return 0;
}

/**
 * @brief Rosenbrock 目标函数。
 * f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]。
 */
static lmmc_real_t rosenbrock_obj(const lmmc_vec_t *x, void *user_data) {
    (void)user_data;
    size_t n = x->size;
    lmmc_real_t f = 0.0;
    for (size_t i = 0; i < n - 1; i++) {
        lmmc_real_t xi = x->data[i];
        lmmc_real_t xi1 = x->data[i + 1];
        lmmc_real_t t1 = xi1 - xi * xi;
        lmmc_real_t t2 = 1.0 - xi;
        f += 100.0 * t1 * t1 + t2 * t2;
    }
    return f;
}

static lmmc_status_t rosenbrock_grad(const lmmc_vec_t *x, lmmc_vec_t *grad, void *user_data) {
    (void)user_data;
    size_t n = x->size;
    memset(grad->data, 0, n * sizeof(lmmc_real_t));
    for (size_t i = 0; i < n - 1; i++) {
        lmmc_real_t xi = x->data[i];
        lmmc_real_t xi1 = x->data[i + 1];
        lmmc_real_t t1 = xi1 - xi * xi;
        grad->data[i] += -400.0 * xi * t1 + 2.0 * (xi - 1.0);
        grad->data[i + 1] += 200.0 * t1;
    }
    return LMMC_STATUS_OK;
}
static void check_rosenbrock_solution(const lmmc_vec_t *x, lmmc_vec_t *gradient, size_t n) {
    lmmc_vec_t grad = *gradient;

    rosenbrock_grad(x, &grad, NULL);
    lmmc_real_t grad_norm = 0.0;
    lmmc_vec_norm2(&grad, &grad_norm);

    assert_false(grad_norm > 1e-6);

    for (size_t i = 0; i < n; i++) {
        assert_false(fabs(x->data[i] - 1.0) > 1e-3);
    }
}
static void test_rosenbrock_dimension(void **state) {
    vector_fixture_t *fixture = *state;
    size_t n = fixture->n;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    for (size_t i = 0; i < n; i++) {
        x.data[i] = -1.0;
    }

    st = lmmc_optimize_default_config(&cfg);
    assert_false(st != LMMC_STATUS_OK);
    cfg.max_iter = 50000;
    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 1e-10;
    cfg.lbfgs_memory = 20;

    st = lmmc_minimize_lbfgs(rosenbrock_obj, rosenbrock_grad, NULL, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK);

    assert_true(result.converged);

    check_rosenbrock_solution(&x, &fixture->grad, n);
}

int main(void) {
    vector_fixture_t dimension_2 = {.n = 2};
    vector_fixture_t dimension_5 = {.n = 5};
    vector_fixture_t dimension_10 = {.n = 10};
    vector_fixture_t dimension_20 = {.n = 20};
    const struct CMUnitTest tests[] = {
        {"rosenbrock_2", test_rosenbrock_dimension, setup_vector, teardown_vector, &dimension_2},
        {"rosenbrock_5", test_rosenbrock_dimension, setup_vector, teardown_vector, &dimension_5},
        {"rosenbrock_10", test_rosenbrock_dimension, setup_vector, teardown_vector, &dimension_10},
        {"rosenbrock_20", test_rosenbrock_dimension, setup_vector, teardown_vector, &dimension_20},
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
