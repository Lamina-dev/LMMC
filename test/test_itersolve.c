/**
 * @file test_itersolve.c
 * 针对 LMMC 中 itersolve 相关接口的单元测试。
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
    lmmc_vec_t x_bad;
    lmmc_vec_t x_nan;
    lmmc_sparse_mat_t a_sing;
    lmmc_mat_t sing_dense;
    lmmc_precond_t jacobi;
    lmmc_precond_t none_bad;
    lmmc_itersolve_config_t cfg;
    lmmc_itersolve_result_t result;
    lmmc_precond_t zero_diagonal_preconditioner_jacobi_fail;
};

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_precond_destroy(&fixture->zero_diagonal_preconditioner_jacobi_fail);
    lmmc_mat_destroy(&fixture->a_dense);
    lmmc_sparse_destroy(&fixture->a_sparse);
    lmmc_vec_destroy(&fixture->b);
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->b_bad);
    lmmc_vec_destroy(&fixture->x_bad);
    lmmc_vec_destroy(&fixture->x_nan);
    lmmc_precond_destroy(&fixture->jacobi);
    lmmc_precond_destroy(&fixture->none_bad);
    lmmc_mat_destroy(&fixture->sing_dense);
    lmmc_sparse_destroy(&fixture->a_sing);
    free(fixture);
    *state = NULL;
    return 0;
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    lmmc_status_t st;

    {
        lmmc_mat_t *a_dense = &fixture->a_dense;
        lmmc_sparse_mat_t *a_sparse = &fixture->a_sparse;
        lmmc_vec_t *b = &fixture->b;
        lmmc_vec_t *x = &fixture->x;

        st = lmmc_mat_create(3, 3, a_dense);
        if (st != LMMC_STATUS_OK)
            goto cleanup;

        LMMC_REAL_SET_D(&a_dense->data[0], 4.0);
        LMMC_REAL_SET_D(&a_dense->data[1], 1.0);
        LMMC_REAL_SET_D(&a_dense->data[2], 0.0);
        LMMC_REAL_SET_D(&a_dense->data[3], 1.0);
        LMMC_REAL_SET_D(&a_dense->data[4], 3.0);
        LMMC_REAL_SET_D(&a_dense->data[5], 1.0);
        LMMC_REAL_SET_D(&a_dense->data[6], 0.0);
        LMMC_REAL_SET_D(&a_dense->data[7], 1.0);
        LMMC_REAL_SET_D(&a_dense->data[8], 2.0);

        st = lmmc_sparse_from_dense(a_dense, 1e-14, a_sparse);
        if (st != LMMC_STATUS_OK)
            goto cleanup;

        st = lmmc_vec_create(3, b);
        if (st != LMMC_STATUS_OK)
            goto cleanup;
        st = lmmc_vec_create(3, x);
        if (st != LMMC_STATUS_OK)
            goto cleanup;

        LMMC_REAL_SET_D(&b->data[0], 6.0);
        LMMC_REAL_SET_D(&b->data[1], 10.0);
        LMMC_REAL_SET_D(&b->data[2], 8.0);
    }
    {
        lmmc_sparse_mat_t *a_sparse = &fixture->a_sparse;

        lmmc_precond_t *jacobi = &fixture->jacobi;
        lmmc_itersolve_config_t *cfg = &fixture->cfg;

        st = lmmc_precond_create_jacobi(a_sparse, jacobi);
        if (st != LMMC_STATUS_OK)
            goto cleanup;

        st = lmmc_itersolve_default_config(3, cfg);
        if (st != LMMC_STATUS_OK)
            goto cleanup;
        cfg->max_iter = 100;
    }
    return 0;

cleanup:
    teardown(state);
    return st;
}

static void test_cg_with_and_without_jacobi(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *a_sparse = &fixture->a_sparse;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    lmmc_precond_t *jacobi = &fixture->jacobi;
    lmmc_itersolve_config_t *cfg = &fixture->cfg;
    lmmc_itersolve_result_t *result = &fixture->result;
    lmmc_status_t st;
    st = lmmc_cg_solve(a_sparse, b, jacobi, cfg, x, result);
    assert_true((st == LMMC_STATUS_OK) && (result->converged == 1));
    assert_true(((lmmc_test_nearly_equal(x->data[0], 1.0, 1e-10)) && (lmmc_test_nearly_equal(x->data[1], 2.0, 1e-10))) && (lmmc_test_nearly_equal(x->data[2], 3.0, 1e-10)));

    st = lmmc_vec_fill(x, 0.0);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_cg_solve(a_sparse, b, NULL, cfg, x, result);
    assert_true((st == LMMC_STATUS_OK) && (result->converged == 1));
}

static void test_cg_iteration_limit(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a_dense = &fixture->a_dense;
    lmmc_sparse_mat_t *a_sparse = &fixture->a_sparse;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    lmmc_itersolve_config_t *cfg = &fixture->cfg;
    lmmc_itersolve_result_t *result = &fixture->result;

    lmmc_sparse_destroy(a_sparse);
    assert_int_equal(lmmc_mat_fill(a_dense, 0.0), LMMC_STATUS_OK);
    a_dense->data[0] = 1.0;
    a_dense->data[a_dense->stride + 1] = 2.0;
    a_dense->data[2 * a_dense->stride + 2] = 3.0;
    assert_int_equal(
        lmmc_sparse_from_dense(a_dense, 0.0, a_sparse),
        LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(b, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(x, 0.0), LMMC_STATUS_OK);

    cfg->abs_tol = 1e-20;
    cfg->rel_tol = 1e-20;
    cfg->max_iter = 1;
    assert_int_equal(
        lmmc_cg_solve(a_sparse, b, NULL, cfg, x, result),
        LMMC_STATUS_OK);
    assert_false(result->converged);
    assert_int_equal(result->num_iter, 1);

    double residual_sq = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        const double residual = (double)(i + 1) * x->data[i] - 1.0;
        residual_sq += residual * residual;
    }
    const double requested_tolerance =
        cfg->abs_tol + cfg->rel_tol * sqrt(3.0);
    assert_true(sqrt(residual_sq) > requested_tolerance);
}

static void test_cg_shape_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *a_sparse = &fixture->a_sparse;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    lmmc_vec_t *b_bad = &fixture->b_bad;
    lmmc_vec_t *x_bad = &fixture->x_bad;
    lmmc_precond_t *none_bad = &fixture->none_bad;
    lmmc_itersolve_config_t *cfg = &fixture->cfg;
    lmmc_itersolve_result_t *result = &fixture->result;

    lmmc_status_t st;

    st = lmmc_vec_create(2, b_bad);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(2, x_bad);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_precond_create_none(2, none_bad);
    assert_true(st == LMMC_STATUS_OK);

    cfg->abs_tol = 1e-12;
    cfg->rel_tol = 1e-8;
    cfg->max_iter = 50;

    st = lmmc_cg_solve(a_sparse, b_bad, NULL, cfg, x, result);
    assert_true(st == LMMC_STATUS_DIMENSION_MISMATCH);

    st = lmmc_cg_solve(a_sparse, b, none_bad, cfg, x, result);
    assert_true(st == LMMC_STATUS_DIMENSION_MISMATCH);
}

static void test_cg_nonfinite_guess(void **state) {
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

    st = lmmc_cg_solve(a_sparse, b, jacobi, cfg, x_nan, result);
    assert_true(st == LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_zero_diagonal_preconditioner(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *a_sing = &fixture->a_sing;
    lmmc_mat_t *sing_dense = &fixture->sing_dense;

    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, sing_dense);
    assert_true(st == LMMC_STATUS_OK);
    LMMC_REAL_SET_D(&sing_dense->data[0], 0.0);
    LMMC_REAL_SET_D(&sing_dense->data[1], 1.0);
    LMMC_REAL_SET_D(&sing_dense->data[2], 1.0);
    LMMC_REAL_SET_D(&sing_dense->data[3], 2.0);

    st = lmmc_sparse_from_dense(sing_dense, 1e-14, a_sing);
    assert_true(st == LMMC_STATUS_OK);

    {

        st = lmmc_precond_create_jacobi(a_sing, &fixture->zero_diagonal_preconditioner_jacobi_fail);
        assert_true(st == LMMC_STATUS_SINGULAR_MATRIX);
        lmmc_precond_destroy(&fixture->zero_diagonal_preconditioner_jacobi_fail);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_cg_with_and_without_jacobi, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cg_iteration_limit, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cg_shape_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cg_nonfinite_guess, setup, teardown),
        cmocka_unit_test_setup_teardown(test_zero_diagonal_preconditioner, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
