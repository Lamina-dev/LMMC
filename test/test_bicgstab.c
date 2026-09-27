/**
 * @file test_bicgstab.c
 * 针对 LMMC 中 bicgstab 相关接口的单元测试。
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
    free(fixture);
    *state = NULL;
    return 0;
}

static lmmc_status_t create_system_operands(struct test_fixture *fixture) {
    lmmc_status_t st = lmmc_mat_create(3, 3, &fixture->a_dense);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    LMMC_REAL_SET_D(&fixture->a_dense.data[0], 4.0);
    LMMC_REAL_SET_D(&fixture->a_dense.data[1], 1.0);
    LMMC_REAL_SET_D(&fixture->a_dense.data[2], 0.0);
    LMMC_REAL_SET_D(&fixture->a_dense.data[3], 2.0);
    LMMC_REAL_SET_D(&fixture->a_dense.data[4], 3.0);
    LMMC_REAL_SET_D(&fixture->a_dense.data[5], 1.0);
    LMMC_REAL_SET_D(&fixture->a_dense.data[6], 0.0);
    LMMC_REAL_SET_D(&fixture->a_dense.data[7], 1.0);
    LMMC_REAL_SET_D(&fixture->a_dense.data[8], 2.0);

    st = lmmc_sparse_from_dense(&fixture->a_dense, 1e-14, &fixture->a_sparse);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(3, &fixture->b);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(3, &fixture->x);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    LMMC_REAL_SET_D(&fixture->b.data[0], 6.0);
    LMMC_REAL_SET_D(&fixture->b.data[1], 11.0);
    LMMC_REAL_SET_D(&fixture->b.data[2], 8.0);
    return LMMC_STATUS_OK;
}

static lmmc_status_t create_zero_system(struct test_fixture *fixture) {
    lmmc_status_t st = lmmc_mat_create(2, 2, &fixture->a_dense);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_sparse_from_dense(&fixture->a_dense, 0.0, &fixture->a_sparse);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(2, &fixture->b);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(2, &fixture->x);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    LMMC_REAL_SET_D(&fixture->b.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->b.data[1], 1.0);
    return LMMC_STATUS_OK;
}

static int setup_empty(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    return fixture == NULL ? -1 : 0;
}

static int setup_jacobi(void **state) {
    if (setup_empty(state) != 0) {
        return -1;
    }
    struct test_fixture *fixture = *state;
    lmmc_status_t st = create_system_operands(fixture);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_itersolve_default_config(3, &fixture->cfg);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    fixture->cfg.max_iter = 100;
    st = lmmc_precond_create_jacobi(&fixture->a_sparse, &fixture->jacobi);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    return 0;

cleanup:
    teardown(state);
    return st;
}

static int setup_incomplete(void **state) {
    if (setup_empty(state) != 0) {
        return -1;
    }
    struct test_fixture *fixture = *state;
    lmmc_status_t st = create_system_operands(fixture);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_itersolve_default_config(3, &fixture->cfg);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    fixture->cfg.max_iter = 100;
    st = lmmc_precond_create_ilu0(&fixture->a_sparse, &fixture->ilu0);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_precond_create_ilut(&fixture->a_sparse, 1e-12, 4, &fixture->ilut);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    return 0;

cleanup:
    teardown(state);
    return st;
}

static int setup_input_errors(void **state) {
    int status = setup_jacobi(state);
    if (status != 0) {
        return status;
    }
    struct test_fixture *fixture = *state;
    lmmc_status_t st = lmmc_vec_create(2, &fixture->b_bad);
    if (st != LMMC_STATUS_OK) {
        teardown(state);
        return st;
    }
    return 0;
}

static int setup_zero(void **state) {
    if (setup_empty(state) != 0) {
        return -1;
    }
    struct test_fixture *fixture = *state;
    lmmc_status_t st = create_zero_system(fixture);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_itersolve_default_config(3, &fixture->cfg);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    fixture->cfg.max_iter = 100;
    return 0;

cleanup:
    teardown(state);
    return st;
}

static void test_unpreconditioned_and_jacobi(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_bicgstab_solve(&fixture->a_sparse, &fixture->b, NULL, &fixture->cfg, &fixture->x, &fixture->result);
    assert_true((st == LMMC_STATUS_OK) && (fixture->result.converged == 1));
    assert_true(((lmmc_test_nearly_equal(fixture->x.data[0], 1.0, 1e-9)) && (lmmc_test_nearly_equal(fixture->x.data[1], 2.0, 1e-9))) && (lmmc_test_nearly_equal(fixture->x.data[2], 3.0, 1e-9)));

    st = lmmc_vec_fill(&fixture->x, 0.0);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_bicgstab_solve(&fixture->a_sparse, &fixture->b, &fixture->jacobi, &fixture->cfg, &fixture->x, &fixture->result);
    assert_true((st == LMMC_STATUS_OK) && (fixture->result.converged == 1));
}

static void test_incomplete_factor_solvers(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_vec_fill(&fixture->x, 0.0);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_bicgstab_solve(&fixture->a_sparse, &fixture->b, &fixture->ilu0, &fixture->cfg, &fixture->x, &fixture->result);
    assert_true((st == LMMC_STATUS_OK) && (fixture->result.converged == 1));

    st = lmmc_vec_fill(&fixture->x, 0.0);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_bicgstab_solve(&fixture->a_sparse, &fixture->b, &fixture->ilut, &fixture->cfg, &fixture->x, &fixture->result);
    assert_true((st == LMMC_STATUS_OK) && (fixture->result.converged == 1));
}

static void test_iteration_limit_and_ilut_arguments(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_vec_fill(&fixture->x, 0.0);
    assert_true(st == LMMC_STATUS_OK);

    fixture->cfg.abs_tol = 1e-20;
    fixture->cfg.rel_tol = 1e-20;
    fixture->cfg.max_iter = 1;
    st = lmmc_bicgstab_solve(&fixture->a_sparse, &fixture->b, &fixture->jacobi, &fixture->cfg, &fixture->x, &fixture->result);
    assert_true(((st == LMMC_STATUS_OK) && (fixture->result.converged == 0)) && (fixture->result.num_iter == 1));

    fixture->cfg.abs_tol = 1e-12;
    fixture->cfg.rel_tol = 1e-8;
    fixture->cfg.max_iter = 100;

    st = lmmc_precond_create_ilut(&fixture->a_sparse, -1.0, 4, &fixture->ilut);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
    st = lmmc_precond_create_ilut(&fixture->a_sparse, 1e-12, 0, &fixture->ilut);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_large_diagonal_ilut(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    const size_t n_large = 1500;
    size_t i = 0;

    st = lmmc_sparse_create_csr(n_large, n_large, n_large, &fixture->a_sparse);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < n_large; ++i) {
        fixture->a_sparse.row_ptr[i] = i;
        fixture->a_sparse.col_idx[i] = i;
        LMMC_REAL_SET_D(&fixture->a_sparse.values[i], 1.0);
    }
    fixture->a_sparse.row_ptr[n_large] = n_large;

    st = lmmc_precond_create_ilut(&fixture->a_sparse, 1e-12, 4, &fixture->ilut);
    assert_true(st == LMMC_STATUS_OK);
}

static void test_solver_input_errors(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_bicgstab_solve(&fixture->a_sparse, &fixture->b_bad, NULL, &fixture->cfg, &fixture->x, &fixture->result);
    assert_true(st == LMMC_STATUS_DIMENSION_MISMATCH);

    st = lmmc_vec_create(3, &fixture->x_nan);
    assert_true(st == LMMC_STATUS_OK);
    fixture->x_nan.data[0] = NAN;
    LMMC_REAL_SET_D(&fixture->x_nan.data[1], 0.0);
    LMMC_REAL_SET_D(&fixture->x_nan.data[2], 0.0);

    st = lmmc_bicgstab_solve(&fixture->a_sparse, &fixture->b, &fixture->jacobi, &fixture->cfg, &fixture->x_nan, &fixture->result);
    assert_true(st == LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_singular_preconditioners_and_breakdown(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    {

        st = lmmc_precond_create_ilu0(&fixture->a_sparse, &fixture->ilu0);
        assert_true(st == LMMC_STATUS_SINGULAR_MATRIX);
    }

    {

        st = lmmc_precond_create_ilut(&fixture->a_sparse, 1e-12, 2, &fixture->ilut);
        assert_true(st == LMMC_STATUS_SINGULAR_MATRIX);
    }

    st = lmmc_bicgstab_solve(&fixture->a_sparse, &fixture->b, NULL, &fixture->cfg, &fixture->x, &fixture->result);
    assert_true(st == LMMC_STATUS_NUMERICAL_FAILURE);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_unpreconditioned_and_jacobi, setup_jacobi, teardown),
        cmocka_unit_test_setup_teardown(test_incomplete_factor_solvers, setup_incomplete, teardown),
        cmocka_unit_test_setup_teardown(test_iteration_limit_and_ilut_arguments, setup_jacobi, teardown),
        cmocka_unit_test_setup_teardown(test_large_diagonal_ilut, setup_empty, teardown),
        cmocka_unit_test_setup_teardown(test_solver_input_errors, setup_input_errors, teardown),
        cmocka_unit_test_setup_teardown(test_singular_preconditioners_and_breakdown, setup_zero, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
