/**
 * @file test_minres_lsqr.c
 * MINRES / LSQR 迭代求解器单元测试.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

typedef struct {
    lmmc_sparse_builder_t *builder;
    lmmc_sparse_mat_t A;
    lmmc_vec_t b, x, x_true, x_mfree, Ax, r, ATr;
} solver_fixture_t;

static int setup(void **state) {
    solver_fixture_t *f = calloc(1, sizeof(*f));
    assert_non_null(f);
    *state = f;
    return 0;
}

static int teardown(void **state) {
    solver_fixture_t *f = *state;
    lmmc_vec_destroy(&f->ATr);
    lmmc_vec_destroy(&f->r);
    lmmc_vec_destroy(&f->Ax);
    lmmc_vec_destroy(&f->x_mfree);
    lmmc_vec_destroy(&f->x_true);
    lmmc_vec_destroy(&f->x);
    lmmc_vec_destroy(&f->b);
    lmmc_sparse_destroy(&f->A);
    lmmc_sparse_builder_destroy(f->builder);
    free(f);
    return 0;
}

static void build_symmetric_tridiag(solver_fixture_t *f, size_t n,
                                    double even_diag, double odd_diag, double off_diag) {
    assert_int_equal(lmmc_sparse_builder_create(n, n, 3 * n, &f->builder), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; ++i) {
        double diag = (i % 2 == 0) ? even_diag : odd_diag;
        assert_int_equal(lmmc_sparse_builder_add(f->builder, i, i, diag), LMMC_STATUS_OK);
        if (i + 1 < n) {
            assert_int_equal(lmmc_sparse_builder_add(f->builder, i, i + 1, off_diag), LMMC_STATUS_OK);
            assert_int_equal(lmmc_sparse_builder_add(f->builder, i + 1, i, off_diag), LMMC_STATUS_OK);
        }
    }
    assert_int_equal(lmmc_sparse_builder_build(f->builder, LMMC_SPARSE_CSR, &f->A), LMMC_STATUS_OK);
    lmmc_sparse_builder_destroy(f->builder);
    f->builder = NULL;
}

/* Each row has entries in column i % n and its neighbors. */
static void build_overdetermined(solver_fixture_t *f, size_t m, size_t n) {
    assert_int_equal(lmmc_sparse_builder_create(m, n, 3 * m, &f->builder), LMMC_STATUS_OK);
    for (size_t i = 0; i < m; ++i) {
        size_t col = i < n ? i : (i % n);
        assert_int_equal(lmmc_sparse_builder_add(f->builder, i, col, 3.0 + (double)(i % 5) * 0.5), LMMC_STATUS_OK);
        if (col > 0) {
            assert_int_equal(lmmc_sparse_builder_add(f->builder, i, col - 1, -1.0), LMMC_STATUS_OK);
        }
        if (col + 1 < n) {
            assert_int_equal(lmmc_sparse_builder_add(f->builder, i, col + 1, -0.5), LMMC_STATUS_OK);
        }
    }
    assert_int_equal(lmmc_sparse_builder_build(f->builder, LMMC_SPARSE_CSR, &f->A), LMMC_STATUS_OK);
    lmmc_sparse_builder_destroy(f->builder);
    f->builder = NULL;
}

static void compute_rhs_ones(solver_fixture_t *f) {
    assert_int_equal(lmmc_vec_create(f->A.cols, &f->x_true), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(&f->x_true, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_mat_vec_mul(&f->A, &f->x_true, &f->b), LMMC_STATUS_OK);
    lmmc_vec_destroy(&f->x_true);
}

/* Compute ||A^T(Ax - b)||_2 using a manual CSR transpose multiply. */
static double compute_normal_eq_residual(solver_fixture_t *f) {
    const lmmc_sparse_mat_t *A = &f->A;
    const size_t m = A->rows;
    const size_t n = A->cols;
    double norm_sq = 0.0;
    assert_int_equal(lmmc_vec_create(m, &f->Ax), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(m, &f->r), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->ATr), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_mat_vec_mul(A, &f->x, &f->Ax), LMMC_STATUS_OK);
    for (size_t i = 0; i < m; ++i) {
        f->r.data[i] = f->Ax.data[i] - f->b.data[i];
    }
    for (size_t i = 0; i < n; ++i) {
        f->ATr.data[i] = 0.0;
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = A->row_ptr[i]; j < A->row_ptr[i + 1]; ++j) {
            size_t col = A->col_idx[j];
            f->ATr.data[col] += A->values[j] * f->r.data[i];
        }
    }
    for (size_t i = 0; i < n; ++i) {
        norm_sq += f->ATr.data[i] * f->ATr.data[i];
    }
    return sqrt(norm_sq);
}

static lmmc_status_t matvec_callback(const lmmc_vec_t *x, lmmc_vec_t *y, void *user_data) {
    const lmmc_sparse_mat_t *A = user_data;
    return lmmc_sparse_mat_vec_mul(A, x, y);
}

typedef struct {
    size_t forward_calls;
    size_t transpose_calls;
} lsqr_operator_t;

static lmmc_status_t lsqr_forward_callback(const lmmc_vec_t *x, lmmc_vec_t *y,
                                           void *user_data) {
    lsqr_operator_t *op = user_data;
    op->forward_calls++;
    if (x->size != 2 || y->size != 3) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    y->data[0] = x->data[0];
    y->data[1] = x->data[1];
    y->data[2] = x->data[0] + x->data[1];
    return LMMC_STATUS_OK;
}

static lmmc_status_t lsqr_transpose_callback(const lmmc_vec_t *x, lmmc_vec_t *y,
                                             void *user_data) {
    lsqr_operator_t *op = user_data;
    op->transpose_calls++;
    if (x->size != 3 || y->size != 2) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    y->data[0] = x->data[0] + x->data[2];
    y->data[1] = x->data[1] + x->data[2];
    return LMMC_STATUS_OK;
}

/**
 * 正定三对角用例验证收敛性及有限数值输出。
 * @see C. C. Paige and M. A. Saunders, "Solution of Sparse Indefinite Systems
 *      of Linear Equations," SIAM J. Numer. Anal. 12(4), 1975.
 */
static void test_minres_positive_definite(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 20;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    build_symmetric_tridiag(f, n, 4.0, 4.0, -1.0);
    assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    compute_rhs_ones(f);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 500;
    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 1e-12;
    assert_int_equal(lmmc_minres_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_OK);
    assert_true(result.converged);
    for (size_t i = 0; i < n; ++i) {
        assert_true(isfinite(f->x.data[i]));
    }
}

/* The indefinite case requires an OK status and a finite solution. */
static void test_minres_indefinite(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 10;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    build_symmetric_tridiag(f, n, 4.0, -3.0, 0.1);
    assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    compute_rhs_ones(f);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 200;
    cfg.abs_tol = 1e-10;
    cfg.rel_tol = 1e-10;
    assert_int_equal(lmmc_minres_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; ++i) {
        assert_true(isfinite(f->x.data[i]));
    }
}

static void test_lsqr_overdetermined(void **state) {
    solver_fixture_t *f = *state;
    const size_t m = 30, n = 10;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    double norm_b;
    build_overdetermined(f, m, n);
    assert_int_equal(lmmc_vec_create(m, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    /* Consistent system with x_true = [1, 2, ..., n]. */
    assert_int_equal(lmmc_vec_create(n, &f->x_true), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; ++i) {
        f->x_true.data[i] = (double)(i + 1);
    }
    assert_int_equal(lmmc_sparse_mat_vec_mul(&f->A, &f->x_true, &f->b), LMMC_STATUS_OK);
    lmmc_vec_destroy(&f->x_true);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 500;
    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 1e-12;
    assert_int_equal(lmmc_lsqr_solve(&f->A, &f->b, &cfg, &f->x, &result), LMMC_STATUS_OK);
    assert_true(result.converged);
    double normal_res = compute_normal_eq_residual(f);
    assert_int_equal(lmmc_vec_norm2(&f->b, &norm_b), LMMC_STATUS_OK);
    assert_true(normal_res <= 1e-8 * norm_b);
}

static void test_lsqr_matrix_free_forward_and_transpose(void **state) {
    solver_fixture_t *f = *state;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    lsqr_operator_t op = {0};

    assert_int_equal(lmmc_vec_create(3, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(2, &f->x), LMMC_STATUS_OK);
    f->b.data[0] = 1.0;
    f->b.data[1] = 2.0;
    f->b.data[2] = 3.0;
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);

    assert_int_equal(lmmc_itersolve_default_config(2, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 100;
    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 1e-12;
    cfg.apply_op = lsqr_forward_callback;
    cfg.op_user_data = &op;
    cfg.apply_transpose_op = lsqr_transpose_callback;

    assert_int_equal(
        lmmc_lsqr_solve(NULL, &f->b, &cfg, &f->x, &result),
        LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(lmmc_test_nearly_equal(f->x.data[0], 1.0, 1e-10));
    assert_true(lmmc_test_nearly_equal(f->x.data[1], 2.0, 1e-10));
    const double r0 = f->x.data[0] - f->b.data[0];
    const double r1 = f->x.data[1] - f->b.data[1];
    const double r2 = f->x.data[0] + f->x.data[1] - f->b.data[2];
    assert_true(sqrt(r0 * r0 + r1 * r1 + r2 * r2) <= 1e-10);
    assert_true(op.forward_calls > 0);
    assert_true(op.transpose_calls > 0);
}

static void test_lsqr_matrix_free_requires_transpose_before_execution(void **state) {
    solver_fixture_t *f = *state;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    lsqr_operator_t op = {0};

    assert_int_equal(lmmc_vec_create(3, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(2, &f->x), LMMC_STATUS_OK);
    f->b.data[0] = 1.0;
    f->b.data[1] = 2.0;
    f->b.data[2] = 3.0;
    f->x.data[0] = 7.0;
    f->x.data[1] = -4.0;

    assert_int_equal(lmmc_itersolve_default_config(2, &cfg), LMMC_STATUS_OK);
    cfg.apply_op = lsqr_forward_callback;
    cfg.op_user_data = &op;
    cfg.apply_transpose_op = NULL;

    assert_int_equal(
        lmmc_lsqr_solve(NULL, &f->b, &cfg, &f->x, &result),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(op.forward_calls, 0);
    assert_int_equal(op.transpose_calls, 0);
    assert_true(f->x.data[0] == 7.0);
    assert_true(f->x.data[1] == -4.0);
}

static void test_matrix_free_agrees_with_sparse(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 15;
    lmmc_itersolve_config_t cfg_sparse = {0}, cfg_mfree = {0};
    lmmc_itersolve_result_t result_sparse = {0}, result_mfree = {0};
    double norm_b;
    double diff_norm = 0.0;
    build_symmetric_tridiag(f, n, 4.0, 4.0, -1.0);
    assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x_mfree), LMMC_STATUS_OK);
    compute_rhs_ones(f);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg_sparse), LMMC_STATUS_OK);
    cfg_sparse.max_iter = 500;
    cfg_sparse.abs_tol = 1e-14;
    cfg_sparse.rel_tol = 1e-14;
    assert_int_equal(lmmc_minres_solve(&f->A, &f->b, NULL, &cfg_sparse, &f->x, &result_sparse), LMMC_STATUS_OK);
    assert_true(result_sparse.converged);
    assert_int_equal(lmmc_vec_fill(&f->x_mfree, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg_mfree), LMMC_STATUS_OK);
    cfg_mfree.max_iter = 500;
    cfg_mfree.abs_tol = 1e-14;
    cfg_mfree.rel_tol = 1e-14;
    cfg_mfree.apply_op = matvec_callback;
    cfg_mfree.op_user_data = &f->A;
    assert_int_equal(lmmc_minres_solve(NULL, &f->b, NULL, &cfg_mfree, &f->x_mfree, &result_mfree), LMMC_STATUS_OK);
    assert_true(result_mfree.converged);
    assert_int_equal(lmmc_vec_norm2(&f->b, &norm_b), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; ++i) {
        double d = f->x.data[i] - f->x_mfree.data[i];
        diff_norm += d * d;
    }
    assert_true(sqrt(diff_norm) <= 1e-10 * norm_b);
}

static void test_error_both_a_and_apply_op(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 5;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    lsqr_operator_t transpose_op = {0};
    build_symmetric_tridiag(f, n, 5.0, -5.0, 0.5);
    assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(&f->b, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.apply_op = matvec_callback;
    cfg.op_user_data = &f->A;
    assert_int_equal(lmmc_minres_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    memset(&result, 0, sizeof(result));
    assert_int_equal(lmmc_lsqr_solve(&f->A, &f->b, &cfg, &f->x, &result), LMMC_STATUS_INVALID_ARGUMENT);
    cfg.apply_op = NULL;
    cfg.apply_transpose_op = lsqr_transpose_callback;
    cfg.op_user_data = &transpose_op;
    assert_int_equal(
        lmmc_lsqr_solve(&f->A, &f->b, &cfg, &f->x, &result),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(transpose_op.transpose_calls, 0);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_minres_positive_definite, setup, teardown),
        cmocka_unit_test_setup_teardown(test_minres_indefinite, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lsqr_overdetermined, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lsqr_matrix_free_forward_and_transpose, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lsqr_matrix_free_requires_transpose_before_execution, setup, teardown),
        cmocka_unit_test_setup_teardown(test_matrix_free_agrees_with_sparse, setup, teardown),
        cmocka_unit_test_setup_teardown(test_error_both_a_and_apply_op, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
