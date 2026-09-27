/**
 * @file test_gmres.c
 * 针对 LMMC 中 gmres 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_mat_t a_dense;
    lmmc_sparse_mat_t a_sparse;
    lmmc_vec_t b;
    lmmc_vec_t x;
    lmmc_vec_t b_bad;
    lmmc_vec_t x_nan;
    lmmc_precond_t jacobi;
    lmmc_precond_t ilu0;
    lmmc_precond_t ilut;
    lmmc_itersolve_config_t cfg;
    lmmc_itersolve_result_t result;
    lmmc_mat_t zero_dense;
    lmmc_sparse_mat_t zero_sparse;
    lmmc_vec_t zero_b;
    lmmc_vec_t zero_x;
};

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->a_dense);
    lmmc_sparse_destroy(&fixture->a_sparse);
    lmmc_vec_destroy(&fixture->b);
    lmmc_vec_destroy(&fixture->x);
    lmmc_precond_destroy(&fixture->jacobi);
    lmmc_precond_destroy(&fixture->ilu0);
    lmmc_precond_destroy(&fixture->ilut);
    lmmc_vec_destroy(&fixture->b_bad);
    lmmc_vec_destroy(&fixture->x_nan);
    lmmc_mat_destroy(&fixture->zero_dense);
    lmmc_sparse_destroy(&fixture->zero_sparse);
    lmmc_vec_destroy(&fixture->zero_b);
    lmmc_vec_destroy(&fixture->zero_x);
    free(fixture);
    *state = NULL;
    return 0;
}

static lmmc_status_t create_system_operands(struct test_fixture *fixture) {
    lmmc_mat_t *a_dense = &fixture->a_dense;
    lmmc_vec_t *b = &fixture->b;
    lmmc_status_t st = lmmc_mat_create(3, 3, a_dense);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    LMMC_REAL_SET_D(&a_dense->data[0], 4.0);
    LMMC_REAL_SET_D(&a_dense->data[1], 1.0);
    LMMC_REAL_SET_D(&a_dense->data[2], 0.0);
    LMMC_REAL_SET_D(&a_dense->data[3], 2.0);
    LMMC_REAL_SET_D(&a_dense->data[4], 3.0);
    LMMC_REAL_SET_D(&a_dense->data[5], 1.0);
    LMMC_REAL_SET_D(&a_dense->data[6], 0.0);
    LMMC_REAL_SET_D(&a_dense->data[7], 1.0);
    LMMC_REAL_SET_D(&a_dense->data[8], 2.0);
    st = lmmc_sparse_from_dense(a_dense, 1e-14, &fixture->a_sparse);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(3, b);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(3, &fixture->x);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    LMMC_REAL_SET_D(&b->data[0], 6.0);
    LMMC_REAL_SET_D(&b->data[1], 11.0);
    LMMC_REAL_SET_D(&b->data[2], 8.0);
    return LMMC_STATUS_OK;
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    lmmc_status_t st = create_system_operands(fixture);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_precond_create_jacobi(&fixture->a_sparse, &fixture->jacobi);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_precond_create_ilu0(&fixture->a_sparse, &fixture->ilu0);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_precond_create_ilut(&fixture->a_sparse, 1e-12, 4, &fixture->ilut);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_itersolve_default_config(3, &fixture->cfg);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    fixture->cfg.max_iter = 60;
    fixture->cfg.restart = 2;
    return 0;

cleanup:
    teardown(state);
    return st;
}

static void test_gmres_and_jacobi(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *a_sparse = &fixture->a_sparse;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    lmmc_precond_t *jacobi = &fixture->jacobi;
    lmmc_itersolve_config_t *cfg = &fixture->cfg;
    lmmc_itersolve_result_t *result = &fixture->result;

    lmmc_status_t st;

    st = lmmc_gmres_solve(a_sparse, b, NULL, cfg, x, result);
    assert_true((st == LMMC_STATUS_OK) && (result->converged == 1));
    assert_true(((lmmc_test_nearly_equal(x->data[0], 1.0, 1e-6)) && (lmmc_test_nearly_equal(x->data[1], 2.0, 1e-6))) && (lmmc_test_nearly_equal(x->data[2], 3.0, 1e-6)));

    st = lmmc_vec_fill(x, 0.0);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_gmres_solve(a_sparse, b, jacobi, cfg, x, result);
    assert_true((st == LMMC_STATUS_OK) && (result->converged == 1));
}

static void test_gmres_incomplete_factors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *a_sparse = &fixture->a_sparse;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    lmmc_precond_t *ilu0 = &fixture->ilu0;
    lmmc_precond_t *ilut = &fixture->ilut;
    lmmc_itersolve_config_t *cfg = &fixture->cfg;
    lmmc_itersolve_result_t *result = &fixture->result;

    lmmc_status_t st;

    st = lmmc_vec_fill(x, 0.0);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_gmres_solve(a_sparse, b, ilu0, cfg, x, result);
    assert_true((st == LMMC_STATUS_OK) && (result->converged == 1));

    st = lmmc_vec_fill(x, 0.0);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_gmres_solve(a_sparse, b, ilut, cfg, x, result);
    assert_true((st == LMMC_STATUS_OK) && (result->converged == 1));
}

static void test_gmres_iteration_limits(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *a_sparse = &fixture->a_sparse;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    lmmc_precond_t *jacobi = &fixture->jacobi;
    lmmc_itersolve_config_t *cfg = &fixture->cfg;
    lmmc_itersolve_result_t *result = &fixture->result;

    lmmc_status_t st;

    st = lmmc_vec_fill(x, 0.0);
    assert_true(st == LMMC_STATUS_OK);

    cfg->abs_tol = 1e-20;
    cfg->rel_tol = 1e-20;
    cfg->max_iter = 1;
    cfg->restart = 1;
    st = lmmc_gmres_solve(a_sparse, b, jacobi, cfg, x, result);
    assert_true(((st == LMMC_STATUS_OK) && (result->converged == 0)) && (result->num_iter == 1));

    cfg->max_iter = 0;
    st = lmmc_gmres_solve(a_sparse, b, NULL, cfg, x, result);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_gmres_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *a_sparse = &fixture->a_sparse;
    lmmc_vec_t *x = &fixture->x;
    lmmc_vec_t *b_bad = &fixture->b_bad;
    lmmc_precond_t *ilut = &fixture->ilut;
    lmmc_itersolve_config_t *cfg = &fixture->cfg;
    lmmc_itersolve_result_t *result = &fixture->result;

    lmmc_status_t st;

    st = lmmc_vec_create(2, b_bad);
    assert_true(st == LMMC_STATUS_OK);

    cfg->abs_tol = 1e-12;
    cfg->rel_tol = 1e-8;
    cfg->max_iter = 100;
    cfg->restart = 2;

    st = lmmc_precond_create_ilut(a_sparse, -1.0, 4, ilut);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
    st = lmmc_precond_create_ilut(a_sparse, 1e-12, 0, ilut);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_gmres_solve(a_sparse, b_bad, NULL, cfg, x, result);
    assert_true(st == LMMC_STATUS_DIMENSION_MISMATCH);
}

static void test_gmres_nonfinite_guess(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *a_sparse = &fixture->a_sparse;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x_nan = &fixture->x_nan;
    lmmc_precond_t *jacobi = &fixture->jacobi;
    lmmc_itersolve_config_t *cfg = &fixture->cfg;
    lmmc_itersolve_result_t *result = &fixture->result;

    lmmc_status_t st;

    st = lmmc_vec_create(3, x_nan);
    assert_true(st == LMMC_STATUS_OK);
    x_nan->data[0] = NAN;
    LMMC_REAL_SET_D(&x_nan->data[1], 0.0);
    LMMC_REAL_SET_D(&x_nan->data[2], 0.0);

    st = lmmc_gmres_solve(a_sparse, b, jacobi, cfg, x_nan, result);
    assert_true(st == LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_gmres_breakdown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_itersolve_config_t *cfg = &fixture->cfg;
    lmmc_itersolve_result_t *result = &fixture->result;
    lmmc_mat_t *zero_dense = &fixture->zero_dense;
    lmmc_sparse_mat_t *zero_sparse = &fixture->zero_sparse;
    lmmc_vec_t *zero_b = &fixture->zero_b;
    lmmc_vec_t *zero_x = &fixture->zero_x;

    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, zero_dense);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_from_dense(zero_dense, 0.0, zero_sparse);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(2, zero_b);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(2, zero_x);
    assert_true(st == LMMC_STATUS_OK);

    LMMC_REAL_SET_D(&zero_b->data[0], 1.0);
    LMMC_REAL_SET_D(&zero_b->data[1], 1.0);

    cfg->abs_tol = 1e-12;
    cfg->rel_tol = 1e-8;
    cfg->max_iter = 10;
    cfg->restart = 2;
    st = lmmc_gmres_solve(zero_sparse, zero_b, NULL, cfg, zero_x, result);
    assert_true(st == LMMC_STATUS_NUMERICAL_FAILURE);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_gmres_and_jacobi, setup, teardown),
        cmocka_unit_test_setup_teardown(test_gmres_incomplete_factors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_gmres_iteration_limits, setup, teardown),
        cmocka_unit_test_setup_teardown(test_gmres_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_gmres_nonfinite_guess, setup, teardown),
        cmocka_unit_test_setup_teardown(test_gmres_breakdown, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
