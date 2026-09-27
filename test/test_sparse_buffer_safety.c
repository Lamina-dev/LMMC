/**
 * @file test_sparse_buffer_safety.c
 * 稀疏 LU/Cholesky 缓冲区安全测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/*
 * Helper: Build an arrowhead SPD matrix of size n in CSC format.
 *
 * Structure:
 *   A[i][i] = n + 1  for i = 0..n-1  (diagonal dominance)
 *   A[0][j] = 1.0    for j = 1..n-1  (first row dense)
 *   A[i][0] = 1.0    for i = 1..n-1  (first column dense)
 *
 * This is symmetric positive definite and produces significant fill-in
 * during factorization because the first row/column connects to all
 * other rows/columns.
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_sparse_mat_t lu_tridiagonal_fillin_A;
    lmmc_sparse_lu_t *lu_tridiagonal_fillin_lu;
    lmmc_vec_t lu_tridiagonal_fillin_b;
    lmmc_vec_t lu_tridiagonal_fillin_x;
    lmmc_sparse_mat_t lu_large_tridiagonal_A;
    lmmc_sparse_lu_t *lu_large_tridiagonal_lu;
    lmmc_vec_t lu_large_tridiagonal_b;
    lmmc_vec_t lu_large_tridiagonal_x;
    lmmc_sparse_mat_t cholesky_dense_fillin_A;
    lmmc_sparse_chol_t *cholesky_dense_fillin_chol;
    lmmc_vec_t cholesky_dense_fillin_b;
    lmmc_vec_t cholesky_dense_fillin_x;
    lmmc_vec_t cholesky_dense_fillin_x_exact;
    lmmc_sparse_mat_t cholesky_arrowhead_fillin_A;
    lmmc_sparse_chol_t *cholesky_arrowhead_fillin_chol;
    lmmc_vec_t cholesky_arrowhead_fillin_b;
    lmmc_vec_t cholesky_arrowhead_fillin_x;
    lmmc_vec_t cholesky_arrowhead_fillin_x_exact;
    lmmc_sparse_builder_t *destroy_after_failed_cholesky_builder;
    lmmc_sparse_mat_t destroy_after_failed_cholesky_A;
    lmmc_sparse_chol_t *destroy_after_failed_cholesky_chol;
    lmmc_sparse_builder_t *destroy_after_failed_lu_builder;
    lmmc_sparse_mat_t destroy_after_failed_lu_A;
    lmmc_sparse_lu_t *destroy_after_failed_lu_lu;
    lmmc_sparse_mat_t successful_factorize_then_destroy_A;
    lmmc_sparse_lu_t *successful_factorize_then_destroy_lu;
    lmmc_sparse_chol_t *successful_factorize_then_destroy_chol;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_chol_destroy(fixture->successful_factorize_then_destroy_chol);
    lmmc_sparse_lu_destroy(fixture->successful_factorize_then_destroy_lu);
    lmmc_sparse_destroy(&fixture->successful_factorize_then_destroy_A);
    lmmc_sparse_lu_destroy(fixture->destroy_after_failed_lu_lu);
    lmmc_sparse_destroy(&fixture->destroy_after_failed_lu_A);
    lmmc_sparse_builder_destroy(fixture->destroy_after_failed_lu_builder);
    lmmc_sparse_chol_destroy(fixture->destroy_after_failed_cholesky_chol);
    lmmc_sparse_destroy(&fixture->destroy_after_failed_cholesky_A);
    lmmc_sparse_builder_destroy(fixture->destroy_after_failed_cholesky_builder);
    lmmc_vec_destroy(&fixture->cholesky_arrowhead_fillin_x_exact);
    lmmc_vec_destroy(&fixture->cholesky_arrowhead_fillin_x);
    lmmc_vec_destroy(&fixture->cholesky_arrowhead_fillin_b);
    lmmc_sparse_chol_destroy(fixture->cholesky_arrowhead_fillin_chol);
    lmmc_sparse_destroy(&fixture->cholesky_arrowhead_fillin_A);
    lmmc_vec_destroy(&fixture->cholesky_dense_fillin_x_exact);
    lmmc_vec_destroy(&fixture->cholesky_dense_fillin_x);
    lmmc_vec_destroy(&fixture->cholesky_dense_fillin_b);
    lmmc_sparse_chol_destroy(fixture->cholesky_dense_fillin_chol);
    lmmc_sparse_destroy(&fixture->cholesky_dense_fillin_A);
    lmmc_vec_destroy(&fixture->lu_large_tridiagonal_x);
    lmmc_vec_destroy(&fixture->lu_large_tridiagonal_b);
    lmmc_sparse_lu_destroy(fixture->lu_large_tridiagonal_lu);
    lmmc_sparse_destroy(&fixture->lu_large_tridiagonal_A);
    lmmc_vec_destroy(&fixture->lu_tridiagonal_fillin_x);
    lmmc_vec_destroy(&fixture->lu_tridiagonal_fillin_b);
    lmmc_sparse_lu_destroy(fixture->lu_tridiagonal_fillin_lu);
    lmmc_sparse_destroy(&fixture->lu_tridiagonal_fillin_A);
    free(fixture);
    return 0;
}

static lmmc_status_t build_arrowhead_spd(size_t n, lmmc_sparse_mat_t *out) {
    lmmc_sparse_builder_t *builder = NULL;
    lmmc_status_t st;
    lmmc_real_t diag_val = (lmmc_real_t)(n + 1);

    st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
    if (st != LMMC_STATUS_OK)
        return st;

    for (size_t i = 0; i < n; i++) {
        /* Diagonal */
        st = lmmc_sparse_builder_add(builder, i, i, diag_val);
        if (st != LMMC_STATUS_OK) {
            lmmc_sparse_builder_destroy(builder);
            return st;
        }

        if (i > 0) {
            /* First row */
            st = lmmc_sparse_builder_add(builder, 0, i, 1.0);
            if (st != LMMC_STATUS_OK) {
                lmmc_sparse_builder_destroy(builder);
                return st;
            }
            /* First column */
            st = lmmc_sparse_builder_add(builder, i, 0, 1.0);
            if (st != LMMC_STATUS_OK) {
                lmmc_sparse_builder_destroy(builder);
                return st;
            }
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

static lmmc_status_t build_dense_spd_as_sparse(size_t n, lmmc_sparse_mat_t *out) {
    lmmc_sparse_builder_t *builder = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(n, n, n * n, &builder);
    if (st != LMMC_STATUS_OK)
        return st;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t val = (lmmc_real_t)((i < j ? i : j) + 1);
            if (i == j) {
                val += (lmmc_real_t)n;
            }
            st = lmmc_sparse_builder_add(builder, i, j, val);
            if (st != LMMC_STATUS_OK) {
                lmmc_sparse_builder_destroy(builder);
                return st;
            }
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

static lmmc_status_t build_tridiagonal(size_t n, lmmc_sparse_mat_t *out) {
    lmmc_sparse_builder_t *builder = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
    if (st != LMMC_STATUS_OK)
        return st;

    for (size_t i = 0; i < n; i++) {
        st = lmmc_sparse_builder_add(builder, i, i, 4.0);
        if (st != LMMC_STATUS_OK) {
            lmmc_sparse_builder_destroy(builder);
            return st;
        }
        if (i > 0) {
            st = lmmc_sparse_builder_add(builder, i, i - 1, 1.0);
            if (st != LMMC_STATUS_OK) {
                lmmc_sparse_builder_destroy(builder);
                return st;
            }
        }
        if (i < n - 1) {
            st = lmmc_sparse_builder_add(builder, i, i + 1, 1.0);
            if (st != LMMC_STATUS_OK) {
                lmmc_sparse_builder_destroy(builder);
                return st;
            }
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

static lmmc_real_t compute_residual(const lmmc_sparse_mat_t *A,
                                    const lmmc_vec_t *x,
                                    const lmmc_vec_t *b) {
    lmmc_vec_t Ax = {0};
    lmmc_real_t norm = 0.0;

    if (lmmc_vec_create(b->size, &Ax) != LMMC_STATUS_OK)
        return 1e30;
    if (lmmc_sparse_mat_vec_mul(A, x, &Ax) != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&Ax);
        return 1e30;
    }

    for (size_t i = 0; i < b->size; i++) {
        lmmc_real_t diff = Ax.data[i] - b->data[i];
        norm += diff * diff;
    }

    lmmc_vec_destroy(&Ax);
    return sqrt(norm);
}

static void test_lu_tridiagonal_fillin(void **state) {
    struct test_fixture *fixture = *state;

    const size_t n = 20;

    lmmc_status_t st;

    st = build_tridiagonal(n, &fixture->lu_tridiagonal_fillin_A);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->lu_tridiagonal_fillin_A.nnz < n * n);

    st = lmmc_sparse_lu_symbolic(&fixture->lu_tridiagonal_fillin_A, &fixture->lu_tridiagonal_fillin_lu);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_lu_numeric(&fixture->lu_tridiagonal_fillin_A, fixture->lu_tridiagonal_fillin_lu);
    assert_true(st == LMMC_STATUS_OK);

    /* Verify solve doesn't crash on the reallocated buffers */
    st = lmmc_vec_create(n, &fixture->lu_tridiagonal_fillin_b);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(n, &fixture->lu_tridiagonal_fillin_x);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < n; i++)
        fixture->lu_tridiagonal_fillin_b.data[i] = 1.0;

    st = lmmc_sparse_lu_solve(fixture->lu_tridiagonal_fillin_lu, &fixture->lu_tridiagonal_fillin_b, &fixture->lu_tridiagonal_fillin_x);
    assert_true(st == LMMC_STATUS_OK);

    lmmc_sparse_lu_destroy(fixture->lu_tridiagonal_fillin_lu);
    fixture->lu_tridiagonal_fillin_lu = NULL;
    lmmc_vec_destroy(&fixture->lu_tridiagonal_fillin_b);
    lmmc_vec_destroy(&fixture->lu_tridiagonal_fillin_x);
    lmmc_sparse_destroy(&fixture->lu_tridiagonal_fillin_A);
}

static void test_lu_large_tridiagonal(void **state) {
    struct test_fixture *fixture = *state;

    const size_t n = 100;

    lmmc_status_t st;

    st = build_tridiagonal(n, &fixture->lu_large_tridiagonal_A);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_lu_symbolic(&fixture->lu_large_tridiagonal_A, &fixture->lu_large_tridiagonal_lu);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_lu_numeric(&fixture->lu_large_tridiagonal_A, fixture->lu_large_tridiagonal_lu);
    assert_true(st == LMMC_STATUS_OK);

    /* Verify solve doesn't crash */
    st = lmmc_vec_create(n, &fixture->lu_large_tridiagonal_b);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(n, &fixture->lu_large_tridiagonal_x);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < n; i++)
        fixture->lu_large_tridiagonal_b.data[i] = 1.0;

    st = lmmc_sparse_lu_solve(fixture->lu_large_tridiagonal_lu, &fixture->lu_large_tridiagonal_b, &fixture->lu_large_tridiagonal_x);
    assert_true(st == LMMC_STATUS_OK);

    lmmc_sparse_lu_destroy(fixture->lu_large_tridiagonal_lu);
    fixture->lu_large_tridiagonal_lu = NULL;
    lmmc_vec_destroy(&fixture->lu_large_tridiagonal_b);
    lmmc_vec_destroy(&fixture->lu_large_tridiagonal_x);
    lmmc_sparse_destroy(&fixture->lu_large_tridiagonal_A);
}

static void test_cholesky_dense_fillin(void **state) {
    struct test_fixture *fixture = *state;

    const size_t n = 15;

    lmmc_status_t st;

    st = build_dense_spd_as_sparse(n, &fixture->cholesky_dense_fillin_A);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_symbolic(&fixture->cholesky_dense_fillin_A, &fixture->cholesky_dense_fillin_chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_numeric(&fixture->cholesky_dense_fillin_A, fixture->cholesky_dense_fillin_chol);
    assert_true(st == LMMC_STATUS_OK);

    /* Solve with known solution */
    st = lmmc_vec_create(n, &fixture->cholesky_dense_fillin_b);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(n, &fixture->cholesky_dense_fillin_x);
    assert_true(st == LMMC_STATUS_OK);

    {

        st = lmmc_vec_create(n, &fixture->cholesky_dense_fillin_x_exact);
        assert_true(st == LMMC_STATUS_OK);
        for (size_t i = 0; i < n; i++) {
            fixture->cholesky_dense_fillin_x_exact.data[i] = (lmmc_real_t)(i + 1);
        }
        st = lmmc_sparse_mat_vec_mul(&fixture->cholesky_dense_fillin_A, &fixture->cholesky_dense_fillin_x_exact, &fixture->cholesky_dense_fillin_b);
        assert_true(st == LMMC_STATUS_OK);
        lmmc_vec_destroy(&fixture->cholesky_dense_fillin_x_exact);
    }

    st = lmmc_sparse_chol_solve(fixture->cholesky_dense_fillin_chol, &fixture->cholesky_dense_fillin_b, &fixture->cholesky_dense_fillin_x);
    assert_true(st == LMMC_STATUS_OK);

    /* Residual check: ||Ax - b||_2 should be small */
    {
        lmmc_real_t residual = compute_residual(&fixture->cholesky_dense_fillin_A, &fixture->cholesky_dense_fillin_x, &fixture->cholesky_dense_fillin_b);
        lmmc_real_t b_norm = 0.0;
        for (size_t i = 0; i < n; i++)
            b_norm += fixture->cholesky_dense_fillin_b.data[i] * fixture->cholesky_dense_fillin_b.data[i];
        b_norm = sqrt(b_norm);

        assert_true(isfinite(residual) && isfinite(b_norm) && residual <= 1e-8 * (1.0 + b_norm));
    }

    lmmc_sparse_chol_destroy(fixture->cholesky_dense_fillin_chol);
    fixture->cholesky_dense_fillin_chol = NULL;
    lmmc_vec_destroy(&fixture->cholesky_dense_fillin_b);
    lmmc_vec_destroy(&fixture->cholesky_dense_fillin_x);
    lmmc_sparse_destroy(&fixture->cholesky_dense_fillin_A);
}

static void test_cholesky_arrowhead_fillin(void **state) {
    struct test_fixture *fixture = *state;

    const size_t n = 30;

    lmmc_status_t st;

    st = build_arrowhead_spd(n, &fixture->cholesky_arrowhead_fillin_A);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_symbolic(&fixture->cholesky_arrowhead_fillin_A, &fixture->cholesky_arrowhead_fillin_chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_numeric(&fixture->cholesky_arrowhead_fillin_A, fixture->cholesky_arrowhead_fillin_chol);
    assert_true(st == LMMC_STATUS_OK);

    /* Solve with known solution x = [1, 1, ..., 1] */
    st = lmmc_vec_create(n, &fixture->cholesky_arrowhead_fillin_b);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(n, &fixture->cholesky_arrowhead_fillin_x);
    assert_true(st == LMMC_STATUS_OK);

    {

        st = lmmc_vec_create(n, &fixture->cholesky_arrowhead_fillin_x_exact);
        assert_true(st == LMMC_STATUS_OK);
        for (size_t i = 0; i < n; i++) {
            fixture->cholesky_arrowhead_fillin_x_exact.data[i] = 1.0;
        }
        st = lmmc_sparse_mat_vec_mul(&fixture->cholesky_arrowhead_fillin_A, &fixture->cholesky_arrowhead_fillin_x_exact, &fixture->cholesky_arrowhead_fillin_b);
        assert_true(st == LMMC_STATUS_OK);
        lmmc_vec_destroy(&fixture->cholesky_arrowhead_fillin_x_exact);
    }

    st = lmmc_sparse_chol_solve(fixture->cholesky_arrowhead_fillin_chol, &fixture->cholesky_arrowhead_fillin_b, &fixture->cholesky_arrowhead_fillin_x);
    assert_true(st == LMMC_STATUS_OK);

    /* Check solution: all entries should be 1.0 */
    {
        lmmc_real_t max_err = 0.0;
        for (size_t i = 0; i < n; i++) {
            lmmc_real_t err = fabs(fixture->cholesky_arrowhead_fillin_x.data[i] - 1.0);
            assert_true(isfinite(err));
            if (err > max_err)
                max_err = err;
        }
        assert_true(max_err < 1e-8);
    }

    lmmc_sparse_chol_destroy(fixture->cholesky_arrowhead_fillin_chol);
    fixture->cholesky_arrowhead_fillin_chol = NULL;
    lmmc_vec_destroy(&fixture->cholesky_arrowhead_fillin_b);
    lmmc_vec_destroy(&fixture->cholesky_arrowhead_fillin_x);
    lmmc_sparse_destroy(&fixture->cholesky_arrowhead_fillin_A);
}

static void test_destroy_after_failed_cholesky(void **state) {
    struct test_fixture *fixture = *state;

    /* Build a non-SPD matrix */
    lmmc_real_t data[] = {
        1.0, 2.0, 0.0,
        2.0, 1.0, 0.0,
        0.0, 0.0, 1.0};

    lmmc_status_t st;

    st = lmmc_sparse_builder_create(3, 3, 9, &fixture->destroy_after_failed_cholesky_builder);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            if (data[i * 3 + j] != 0.0) {
                st = lmmc_sparse_builder_add(fixture->destroy_after_failed_cholesky_builder, i, j, data[i * 3 + j]);
                assert_true(st == LMMC_STATUS_OK);
            }
        }
    }

    st = lmmc_sparse_builder_build(fixture->destroy_after_failed_cholesky_builder, LMMC_SPARSE_CSC, &fixture->destroy_after_failed_cholesky_A);
    assert_true(st == LMMC_STATUS_OK);
    lmmc_sparse_builder_destroy(fixture->destroy_after_failed_cholesky_builder);
    fixture->destroy_after_failed_cholesky_builder = NULL;

    st = lmmc_sparse_chol_symbolic(&fixture->destroy_after_failed_cholesky_A, &fixture->destroy_after_failed_cholesky_chol);
    assert_true(st == LMMC_STATUS_OK);

    /* Numeric should fail because matrix is not SPD */
    st = lmmc_sparse_chol_numeric(&fixture->destroy_after_failed_cholesky_A, fixture->destroy_after_failed_cholesky_chol);
    assert_true(st == LMMC_STATUS_NOT_POSITIVE_DEFINITE);

    /* Destroy should not crash even after failed factorization */
    lmmc_sparse_chol_destroy(fixture->destroy_after_failed_cholesky_chol);
    fixture->destroy_after_failed_cholesky_chol = NULL;

    lmmc_sparse_destroy(&fixture->destroy_after_failed_cholesky_A);
}

static void test_destroy_after_failed_lu(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_builder_create(3, 3, 9, &fixture->destroy_after_failed_lu_builder);
    assert_true(st == LMMC_STATUS_OK);

    /* Matrix: [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
     * Column 1 is all zeros -> singular, detected during factorization. */
    assert_int_equal(lmmc_sparse_builder_add(fixture->destroy_after_failed_lu_builder, 0, 0, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_builder_add(fixture->destroy_after_failed_lu_builder, 2, 2, 1.0), LMMC_STATUS_OK);

    st = lmmc_sparse_builder_build(fixture->destroy_after_failed_lu_builder, LMMC_SPARSE_CSC, &fixture->destroy_after_failed_lu_A);
    assert_true(st == LMMC_STATUS_OK);
    lmmc_sparse_builder_destroy(fixture->destroy_after_failed_lu_builder);
    fixture->destroy_after_failed_lu_builder = NULL;

    st = lmmc_sparse_lu_symbolic(&fixture->destroy_after_failed_lu_A, &fixture->destroy_after_failed_lu_lu);
    assert_true(st == LMMC_STATUS_OK);

    /* Numeric should fail because matrix is singular */
    st = lmmc_sparse_lu_numeric(&fixture->destroy_after_failed_lu_A, fixture->destroy_after_failed_lu_lu);
    assert_true(st == LMMC_STATUS_SINGULAR_MATRIX);

    /* Destroy should not crash even after failed factorization */
    lmmc_sparse_lu_destroy(fixture->destroy_after_failed_lu_lu);
    fixture->destroy_after_failed_lu_lu = NULL;

    lmmc_sparse_destroy(&fixture->destroy_after_failed_lu_A);
}

static void test_destroy_null(void **state) {
    (void)state;

    lmmc_sparse_lu_destroy(NULL);

    lmmc_sparse_chol_destroy(NULL);
}

static void test_successful_factorize_then_destroy(void **state) {
    struct test_fixture *fixture = *state;

    const size_t n = 10;

    lmmc_status_t st;

    st = build_tridiagonal(n, &fixture->successful_factorize_then_destroy_A);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_lu_symbolic(&fixture->successful_factorize_then_destroy_A, &fixture->successful_factorize_then_destroy_lu);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_lu_numeric(&fixture->successful_factorize_then_destroy_A, fixture->successful_factorize_then_destroy_lu);
    assert_true(st == LMMC_STATUS_OK);

    lmmc_sparse_lu_destroy(fixture->successful_factorize_then_destroy_lu);
    fixture->successful_factorize_then_destroy_lu = NULL;

    lmmc_sparse_destroy(&fixture->successful_factorize_then_destroy_A);

    st = build_arrowhead_spd(n, &fixture->successful_factorize_then_destroy_A);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_symbolic(&fixture->successful_factorize_then_destroy_A, &fixture->successful_factorize_then_destroy_chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_numeric(&fixture->successful_factorize_then_destroy_A, fixture->successful_factorize_then_destroy_chol);
    assert_true(st == LMMC_STATUS_OK);

    lmmc_sparse_chol_destroy(fixture->successful_factorize_then_destroy_chol);
    fixture->successful_factorize_then_destroy_chol = NULL;

    lmmc_sparse_destroy(&fixture->successful_factorize_then_destroy_A);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_lu_tridiagonal_fillin, setup, teardown),
        cmocka_unit_test_setup_teardown(test_lu_large_tridiagonal, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cholesky_dense_fillin, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cholesky_arrowhead_fillin, setup, teardown),
        cmocka_unit_test_setup_teardown(test_destroy_after_failed_cholesky, setup, teardown),
        cmocka_unit_test_setup_teardown(test_destroy_after_failed_lu, setup, teardown),
        cmocka_unit_test_setup_teardown(test_destroy_null, setup, teardown),
        cmocka_unit_test_setup_teardown(test_successful_factorize_then_destroy, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
