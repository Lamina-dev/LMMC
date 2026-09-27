/*
 * test_sparse_solve_residual_prop.c — 稀疏求解残差属性测试。
 *
 * 验证 ||Ax - b||_2 <= 1e-8 * (||A||_F * ||x||_2 + ||b||_2)，
 * 覆盖 LU 和 Cholesky 两条路径。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/*
 * Helper: Generate a random sparse non-singular matrix for LU testing.
 *
 * Strategy: Start with a diagonally dominant matrix to ensure non-singularity.
 * For each row, place the diagonal entry as sum of absolute off-diagonals + 1,
 * and randomly place a few off-diagonal entries.
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct property_case {
    size_t n;
    int trial;
};

struct test_fixture {
    struct property_case parameters;
    lmmc_rng_t *rng;
    lmmc_sparse_mat_t A;
    lmmc_sparse_lu_t *lu;
    lmmc_sparse_chol_t *chol;
    lmmc_vec_t b;
    lmmc_vec_t x;
};

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->b);
    lmmc_sparse_chol_destroy(fixture->chol);
    lmmc_sparse_lu_destroy(fixture->lu);
    lmmc_sparse_destroy(&fixture->A);
    lmmc_rng_destroy(fixture->rng);
    free(fixture);
    *state = NULL;
    return 0;
}

static int setup(void **state) {
    const struct property_case parameters = *(const struct property_case *)*state;
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    if (!fixture) {
        return -1;
    }
    fixture->parameters = parameters;
    lmmc_status_t st = lmmc_rng_create(&fixture->rng);
    if (st == LMMC_STATUS_OK) {
        st = lmmc_rng_seed(fixture->rng, UINT64_C(0x53505253) + fixture->parameters.n * 10 + (unsigned)fixture->parameters.trial);
    }
    if (st != LMMC_STATUS_OK) {
        teardown(state);
        return st;
    }
    return 0;
}

static lmmc_status_t generate_sparse_nonsingular(
    lmmc_rng_t *rng, size_t n, size_t nnz_per_row,
    lmmc_sparse_mat_t *out_A) {
    lmmc_sparse_builder_t *builder = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(n, n, n * (nnz_per_row + 1), &builder);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    for (size_t i = 0; i < n; i++) {
        double row_sum = 0.0;

        /* Add random off-diagonal entries */
        for (size_t k = 0; k < nnz_per_row; k++) {
            lmmc_real_t u;
            lmmc_rng_uniform(rng, 0.0, 1.0, &u);
            size_t j = (size_t)(u * (double)n);
            if (j >= n) {
                j = n - 1;
            }
            if (j == i) {
                j = (i + 1) % n;
            }

            lmmc_real_t val;
            lmmc_rng_uniform(rng, -1.0, 1.0, &val);
            row_sum += fabs(val);

            st = lmmc_sparse_builder_add(builder, i, j, val);
            if (st != LMMC_STATUS_OK) {
                lmmc_sparse_builder_destroy(builder);
                return st;
            }
        }

        /* Set diagonal to ensure diagonal dominance (non-singular) */
        double diag_val = row_sum + 1.0;
        st = lmmc_sparse_builder_add(builder, i, i, diag_val);
        if (st != LMMC_STATUS_OK) {
            lmmc_sparse_builder_destroy(builder);
            return st;
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out_A);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

static lmmc_status_t generate_sparse_spd(
    lmmc_rng_t *rng, size_t n, size_t nnz_per_row,
    lmmc_sparse_mat_t *out_A) {
    lmmc_sparse_builder_t *builder = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(n, n, n * (2 * nnz_per_row + 1), &builder);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    for (size_t i = 0; i < n; i++) {
        double off_diag_sum = 0.0;

        /* Add random symmetric off-diagonal entries */
        for (size_t k = 0; k < nnz_per_row; k++) {
            lmmc_real_t u;
            lmmc_rng_uniform(rng, 0.0, 1.0, &u);
            size_t j = (size_t)(u * (double)n);
            if (j >= n) {
                j = n - 1;
            }
            if (j == i) {
                continue;
            }

            lmmc_real_t val;
            lmmc_rng_uniform(rng, -0.5, 0.5, &val);
            off_diag_sum += fabs(val);

            /* Add both (i,j) and (j,i) for symmetry */
            st = lmmc_sparse_builder_add(builder, i, j, val);
            if (st != LMMC_STATUS_OK) {
                lmmc_sparse_builder_destroy(builder);
                return st;
            }
            st = lmmc_sparse_builder_add(builder, j, i, val);
            if (st != LMMC_STATUS_OK) {
                lmmc_sparse_builder_destroy(builder);
                return st;
            }
        }

        /* Set diagonal to ensure SPD (strictly diagonally dominant => SPD for symmetric) */
        double diag_val = off_diag_sum * 2.0 + (double)n + 1.0;
        st = lmmc_sparse_builder_add(builder, i, i, diag_val);
        if (st != LMMC_STATUS_OK) {
            lmmc_sparse_builder_destroy(builder);
            return st;
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out_A);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

static lmmc_status_t generate_random_vec(lmmc_rng_t *rng, size_t n, lmmc_vec_t *out_b) {
    lmmc_status_t st = lmmc_vec_create(n, out_b);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    for (size_t i = 0; i < n; i++) {
        lmmc_real_t val;
        lmmc_rng_uniform(rng, -1.0, 1.0, &val);
        out_b->data[i] = val;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t compute_residual_norm(
    const lmmc_sparse_mat_t *A, const lmmc_vec_t *x, const lmmc_vec_t *b,
    lmmc_real_t *out_residual_norm) {
    lmmc_vec_t Ax = {0};
    lmmc_status_t st;

    st = lmmc_vec_create(A->rows, &Ax);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    /* Compute Ax */
    st = lmmc_sparse_mat_vec_mul(A, x, &Ax);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&Ax);
        return st;
    }

    /* Compute Ax - b */
    for (size_t i = 0; i < A->rows; i++) {
        Ax.data[i] -= b->data[i];
    }

    /* Compute ||Ax - b||_2 */
    st = lmmc_vec_norm2(&Ax, out_residual_norm);
    lmmc_vec_destroy(&Ax);
    return st;
}

static void check_sparse_lu_residual_bound(lmmc_sparse_mat_t *A, lmmc_vec_t *b, lmmc_vec_t *x) {

    lmmc_status_t st;

    {
        lmmc_real_t residual_norm = 0.0;
        lmmc_real_t A_norm = 0.0;
        lmmc_real_t x_norm = 0.0;
        lmmc_real_t b_norm = 0.0;

        st = compute_residual_norm(A, x, b, &residual_norm);
        assert_true(st == LMMC_STATUS_OK);

        st = lmmc_sparse_norm_fro(A, &A_norm);
        assert_true(st == LMMC_STATUS_OK);

        st = lmmc_vec_norm2(x, &x_norm);
        assert_true(st == LMMC_STATUS_OK);

        st = lmmc_vec_norm2(b, &b_norm);
        assert_true(st == LMMC_STATUS_OK);

        double bound = 1e-8 * (A_norm * x_norm + b_norm);
        assert_true(isfinite(residual_norm) && isfinite(bound) && residual_norm <= bound);
    }
}

static void test_sparse_lu_residual(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_rng_t *rng = fixture->rng;
    size_t n = fixture->parameters.n;
    size_t nnz_per_row = 3;

    lmmc_status_t st;

    /* Generate random non-singular sparse matrix */
    st = generate_sparse_nonsingular(rng, n, nnz_per_row, &fixture->A);
    assert_true(st == LMMC_STATUS_OK);

    /* Generate random RHS vector */
    st = generate_random_vec(rng, n, &fixture->b);
    assert_true(st == LMMC_STATUS_OK);

    /* Create solution vector */
    st = lmmc_vec_create(n, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);

    /* Three-stage pipeline: symbolic -> numeric -> solve */
    st = lmmc_sparse_lu_symbolic(&fixture->A, &fixture->lu);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_lu_numeric(&fixture->A, fixture->lu);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_lu_solve(fixture->lu, &fixture->b, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);

    check_sparse_lu_residual_bound(&fixture->A, &fixture->b, &fixture->x);
}

static void check_sparse_cholesky_residual_bound(lmmc_sparse_mat_t *A, lmmc_vec_t *b, lmmc_vec_t *x) {

    lmmc_status_t st;

    /* Compute residual and check bound */
    {
        lmmc_real_t residual_norm = 0.0;
        lmmc_real_t A_norm = 0.0;
        lmmc_real_t x_norm = 0.0;
        lmmc_real_t b_norm = 0.0;

        st = compute_residual_norm(A, x, b, &residual_norm);
        assert_true(st == LMMC_STATUS_OK);

        st = lmmc_sparse_norm_fro(A, &A_norm);
        assert_true(st == LMMC_STATUS_OK);

        st = lmmc_vec_norm2(x, &x_norm);
        assert_true(st == LMMC_STATUS_OK);

        st = lmmc_vec_norm2(b, &b_norm);
        assert_true(st == LMMC_STATUS_OK);

        double bound = 1e-8 * (A_norm * x_norm + b_norm);
        assert_true(isfinite(residual_norm) && isfinite(bound) && residual_norm <= bound);
    }
}

static void test_sparse_chol_residual(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_rng_t *rng = fixture->rng;
    size_t n = fixture->parameters.n;
    size_t nnz_per_row = 3;

    lmmc_status_t st;

    /* Generate random sparse SPD matrix */
    st = generate_sparse_spd(rng, n, nnz_per_row, &fixture->A);
    assert_true(st == LMMC_STATUS_OK);

    /* Generate random RHS vector */
    st = generate_random_vec(rng, n, &fixture->b);
    assert_true(st == LMMC_STATUS_OK);

    /* Create solution vector */
    st = lmmc_vec_create(n, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);

    /* Three-stage pipeline: symbolic -> numeric -> solve */
    st = lmmc_sparse_chol_symbolic(&fixture->A, &fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_numeric(&fixture->A, fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_solve(fixture->chol, &fixture->b, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);

    check_sparse_cholesky_residual_bound(&fixture->A, &fixture->b, &fixture->x);
}

int main(void) {
    const size_t sizes[] = {5, 7, 10, 12, 15, 18, 20, 25, 30};
    struct property_case parameters[180];
    char names[180][64];
    struct CMUnitTest tests[180];
    for (size_t i = 0; i < 180; ++i) {
        parameters[i].n = sizes[(i % 90) / 10];
        parameters[i].trial = (int)(i % 10) + 1;
        snprintf(names[i], sizeof(names[i]), "%s_n%zu_trial%d", i < 90 ? "lu" : "cholesky",
                 parameters[i].n, parameters[i].trial);
        tests[i] = (struct CMUnitTest)cmocka_unit_test_prestate_setup_teardown(
            test_sparse_lu_residual, setup, teardown, &parameters[i]);
        if (i >= 90)
            tests[i].test_func = test_sparse_chol_residual;
        tests[i].name = names[i];
    }
    return cmocka_run_group_tests(tests, NULL, NULL);
}
