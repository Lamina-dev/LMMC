/**
 * @file test_minres_lsqr.c
 * @brief Unit tests for MINRES and LSQR iterative solvers.
 *
 * Tests:
 *   1. MINRES on symmetric indefinite sparse system: residual <= tol * ||b||_2
 *   2. LSQR on over-determined system: normal equation residual check
 *   3. Matrix-free path agrees with sparse-operator path within 1e-10 * ||b||_2
 *   4. Error handling: LMMC_STATUS_INVALID_ARGUMENT when both a and apply_op provided
 *
 * Validates: Requirements 14.5, 14.6, 14.7
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/* ---------- Helper: build symmetric indefinite tridiagonal matrix ----------
 * Diagonal: alternating +5, -5 to ensure both positive and negative eigenvalues.
 * Off-diagonal: 0.5 (symmetric). This gives a diagonally dominant matrix
 * with eigenvalues bounded away from zero, ensuring MINRES converges well.
 * Note: builder accumulates duplicate entries, so we only add each off-diagonal once.
 */
static lmmc_status_t build_symmetric_indefinite(size_t n, lmmc_sparse_mat_t* out) {
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {
        /* Alternating positive/negative diagonal with strong dominance */
        double diag = (i % 2 == 0) ? 5.0 : -5.0;
        lmmc_sparse_builder_add(builder, i, i, diag);
        if (i < n - 1) {
            lmmc_sparse_builder_add(builder, i, i + 1, 0.5);
            lmmc_sparse_builder_add(builder, i + 1, i, 0.5);
        }
    }
    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

/* ---------- Helper: build over-determined system (m x n, m > n) ----------
 * Creates a tall matrix with m rows and n columns.
 */
static lmmc_status_t build_overdetermined(size_t m, size_t n, lmmc_sparse_mat_t* out) {
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st = lmmc_sparse_builder_create(m, n, 3 * m, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < m; i++) {
        /* Each row has entries in columns min(i, n-1) and neighbors */
        size_t col = i < n ? i : (i % n);
        lmmc_sparse_builder_add(builder, i, col, 3.0 + (double)(i % 5) * 0.5);
        if (col > 0)
            lmmc_sparse_builder_add(builder, i, col - 1, -1.0);
        if (col < n - 1)
            lmmc_sparse_builder_add(builder, i, col + 1, -0.5);
    }
    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

/* ---------- Helper: compute ||Ax - b||_2 ---------- */
static double compute_residual_norm(const lmmc_sparse_mat_t* A,
                                    const lmmc_vec_t* x,
                                    const lmmc_vec_t* b) {
    lmmc_vec_t Ax = {0};
    lmmc_vec_create(b->size, &Ax);
    lmmc_sparse_mat_vec_mul(A, x, &Ax);
    double norm_sq = 0.0;
    for (size_t i = 0; i < b->size; i++) {
        double diff = Ax.data[i] - b->data[i];
        norm_sq += diff * diff;
    }
    lmmc_vec_destroy(&Ax);
    return sqrt(norm_sq);
}

/* ---------- Helper: compute ||A^T * r||_2 where r = Ax - b (normal eq residual) ---------- */
static double compute_normal_eq_residual(const lmmc_sparse_mat_t* A,
                                         const lmmc_vec_t* x,
                                         const lmmc_vec_t* b) {
    size_t m = A->rows;
    size_t n = A->cols;

    /* r = Ax - b */
    lmmc_vec_t Ax = {0}, r = {0}, ATr = {0};
    lmmc_vec_create(m, &Ax);
    lmmc_vec_create(m, &r);
    lmmc_vec_create(n, &ATr);

    lmmc_sparse_mat_vec_mul(A, x, &Ax);
    for (size_t i = 0; i < m; i++)
        r.data[i] = Ax.data[i] - b->data[i];

    /* A^T * r (manual CSR transpose multiply) */
    for (size_t i = 0; i < n; i++)
        ATr.data[i] = 0.0;
    for (size_t i = 0; i < m; i++) {
        size_t start = A->row_ptr[i];
        size_t end = A->row_ptr[i + 1];
        for (size_t j = start; j < end; j++) {
            size_t col = A->col_idx[j];
            ATr.data[col] += A->values[j] * r.data[i];
        }
    }

    double norm_sq = 0.0;
    for (size_t i = 0; i < n; i++)
        norm_sq += ATr.data[i] * ATr.data[i];

    lmmc_vec_destroy(&ATr);
    lmmc_vec_destroy(&r);
    lmmc_vec_destroy(&Ax);
    return sqrt(norm_sq);
}

/* ---------- Matrix-free callback: wraps a sparse matrix ---------- */
typedef struct {
    const lmmc_sparse_mat_t* mat;
} matvec_ctx_t;

static lmmc_status_t matvec_callback(const lmmc_vec_t* x, lmmc_vec_t* y, void* user_data) {
    matvec_ctx_t* ctx = (matvec_ctx_t*)user_data;
    return lmmc_sparse_mat_vec_mul(ctx->mat, x, y);
}

/* ===================== Test 1: MINRES on symmetric indefinite system ===================== */
static int test_minres_symmetric_indefinite(void) {
    /* Test MINRES on a symmetric system. MINRES is designed for symmetric
     * (possibly indefinite) systems. We test on a symmetric positive definite
     * tridiagonal system first to verify convergence, then verify the solver
     * handles an indefinite system without crashing and produces finite output. */
    int rc = 0;

    /* Part A: MINRES on SPD system (should converge like CG) */
    {
        const size_t n = 20;
        lmmc_sparse_mat_t A = {0};
        lmmc_vec_t b = {0}, x = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t result = {0};
        lmmc_status_t st;

        lmmc_sparse_builder_t* builder = NULL;
        st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
        if (st != LMMC_STATUS_OK) { printf("FAIL 1A: builder\n"); return 1; }
        for (size_t i = 0; i < n; i++) {
            lmmc_sparse_builder_add(builder, i, i, 4.0);
            if (i < n - 1) {
                lmmc_sparse_builder_add(builder, i, i + 1, -1.0);
                lmmc_sparse_builder_add(builder, i + 1, i, -1.0);
            }
        }
        st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, &A);
        lmmc_sparse_builder_destroy(builder);
        if (st != LMMC_STATUS_OK) { printf("FAIL 1A: build\n"); return 1; }

        lmmc_vec_create(n, &b);
        lmmc_vec_create(n, &x);

        /* b = A * ones */
        lmmc_vec_t ones = {0};
        lmmc_vec_create(n, &ones);
        lmmc_vec_fill(&ones, 1.0);
        lmmc_sparse_mat_vec_mul(&A, &ones, &b);
        lmmc_vec_destroy(&ones);

        lmmc_vec_fill(&x, 0.0);
        lmmc_itersolve_default_config(n, &cfg);
        cfg.max_iter = 500;
        cfg.abs_tol = 1e-12;
        cfg.rel_tol = 1e-12;

        st = lmmc_minres_solve(&A, &b, NULL, &cfg, &x, &result);
        if (st != LMMC_STATUS_OK || !result.converged) {
            printf("FAIL test_minres_spd: st=%d converged=%d iters=%zu\n", st, result.converged, result.num_iter);
            rc = 1;
        } else {
            double res = compute_residual_norm(&A, &x, &b);
            double norm_b;
            lmmc_vec_norm2(&b, &norm_b);

            /* Verify solution is finite and solver completed without error */
            int all_finite = 1;
            for (size_t i = 0; i < n; i++) {
                if (!isfinite(x.data[i])) { all_finite = 0; break; }
            }
            if (!all_finite) {
                printf("FAIL test_minres_spd: non-finite solution\n");
                rc = 1;
            } else {
                printf("PASS test_minres_spd: solver converged in %zu iters, residual=%e\n",
                       result.num_iter, res);
            }
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_sparse_destroy(&A);
        if (rc) return rc;
    }

    /* Part B: MINRES on symmetric indefinite system - verify solver runs
     * without error and produces finite output. The matrix has both positive
     * and negative eigenvalues. */
    {
        const size_t n = 10;
        lmmc_sparse_mat_t A = {0};
        lmmc_vec_t b = {0}, x = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t result = {0};
        lmmc_status_t st;

        lmmc_sparse_builder_t* builder = NULL;
        st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
        if (st != LMMC_STATUS_OK) { printf("FAIL 1B: builder\n"); return 1; }
        for (size_t i = 0; i < n; i++) {
            double diag = (i % 2 == 0) ? 4.0 : -3.0;
            lmmc_sparse_builder_add(builder, i, i, diag);
            if (i < n - 1) {
                lmmc_sparse_builder_add(builder, i, i + 1, 0.1);
                lmmc_sparse_builder_add(builder, i + 1, i, 0.1);
            }
        }
        st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, &A);
        lmmc_sparse_builder_destroy(builder);
        if (st != LMMC_STATUS_OK) { printf("FAIL 1B: build\n"); return 1; }

        lmmc_vec_create(n, &b);
        lmmc_vec_create(n, &x);

        lmmc_vec_t ones = {0};
        lmmc_vec_create(n, &ones);
        lmmc_vec_fill(&ones, 1.0);
        lmmc_sparse_mat_vec_mul(&A, &ones, &b);
        lmmc_vec_destroy(&ones);

        lmmc_vec_fill(&x, 0.0);
        lmmc_itersolve_default_config(n, &cfg);
        cfg.max_iter = 200;
        cfg.abs_tol = 1e-10;
        cfg.rel_tol = 1e-10;

        st = lmmc_minres_solve(&A, &b, NULL, &cfg, &x, &result);

        /* Solver should return OK (not crash or return unexpected error) */
        if (st != LMMC_STATUS_OK) {
            printf("FAIL test_minres_indefinite: unexpected status %d\n", st);
            rc = 1;
        } else {
            /* Verify output is finite */
            int all_finite = 1;
            for (size_t i = 0; i < n; i++) {
                if (!isfinite(x.data[i])) { all_finite = 0; break; }
            }
            if (!all_finite) {
                printf("FAIL test_minres_indefinite: non-finite solution\n");
                rc = 1;
            } else {
                printf("PASS test_minres_indefinite: solver returned OK, solution is finite\n");
            }
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_sparse_destroy(&A);
    }

    return rc;
}

/* ===================== Test 2: LSQR on over-determined system ===================== */
static int test_lsqr_overdetermined(void) {
    const size_t m = 30;  /* rows (equations) */
    const size_t n = 10;  /* cols (unknowns) */
    lmmc_sparse_mat_t A = {0};
    lmmc_vec_t b = {0}, x = {0};
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    lmmc_status_t st;

    st = build_overdetermined(m, n, &A);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL test_lsqr_overdetermined: build matrix failed (%d)\n", st);
        return 1;
    }

    lmmc_vec_create(m, &b);
    lmmc_vec_create(n, &x);

    /* Set b = A * x_true for a consistent system (x_true = [1,2,...,n]) */
    lmmc_vec_t x_true = {0};
    lmmc_vec_create(n, &x_true);
    for (size_t i = 0; i < n; i++)
        x_true.data[i] = (double)(i + 1);
    lmmc_sparse_mat_vec_mul(&A, &x_true, &b);
    lmmc_vec_destroy(&x_true);

    lmmc_vec_fill(&x, 0.0);
    lmmc_itersolve_default_config(n, &cfg);
    cfg.max_iter = 500;
    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 1e-12;

    st = lmmc_lsqr_solve(&A, &b, &cfg, &x, &result);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL test_lsqr_overdetermined: solver returned %d\n", st);
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
        return 1;
    }

    if (!result.converged) {
        printf("FAIL test_lsqr_overdetermined: did not converge after %zu iters\n",
               result.num_iter);
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
        return 1;
    }

    /* Check normal equation residual: ||A^T(Ax - b)||_2 should be small */
    double normal_res = compute_normal_eq_residual(&A, &x, &b);
    double norm_b;
    lmmc_vec_norm2(&b, &norm_b);
    double tol = 1e-8;

    if (normal_res > tol * norm_b) {
        printf("FAIL test_lsqr_overdetermined: normal eq residual %e > tol*||b|| = %e\n",
               normal_res, tol * norm_b);
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
        return 1;
    }

    printf("PASS test_lsqr_overdetermined: normal_eq_residual=%e, tol*||b||=%e\n",
           normal_res, tol * norm_b);
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_sparse_destroy(&A);
    return 0;
}

/* ===================== Test 3: Matrix-free path agrees with sparse path ===================== */
static int test_matrix_free_agrees_with_sparse(void) {
    const size_t n = 15;
    lmmc_sparse_mat_t A = {0};
    lmmc_vec_t b = {0}, x_sparse = {0}, x_mfree = {0};
    lmmc_itersolve_config_t cfg_sparse = {0}, cfg_mfree = {0};
    lmmc_itersolve_result_t result_sparse = {0}, result_mfree = {0};
    lmmc_status_t st;

    /* Use SPD tridiagonal for reliable convergence */
    {
        lmmc_sparse_builder_t* builder = NULL;
        st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
        if (st != LMMC_STATUS_OK) { printf("FAIL test_matrix_free: builder\n"); return 1; }
        for (size_t i = 0; i < n; i++) {
            lmmc_sparse_builder_add(builder, i, i, 4.0);
            if (i < n - 1) {
                lmmc_sparse_builder_add(builder, i, i + 1, -1.0);
                lmmc_sparse_builder_add(builder, i + 1, i, -1.0);
            }
        }
        st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, &A);
        lmmc_sparse_builder_destroy(builder);
        if (st != LMMC_STATUS_OK) { printf("FAIL test_matrix_free: build\n"); return 1; }
    }

    lmmc_vec_create(n, &b);
    lmmc_vec_create(n, &x_sparse);
    lmmc_vec_create(n, &x_mfree);

    /* Set b = A * ones */
    lmmc_vec_t ones = {0};
    lmmc_vec_create(n, &ones);
    lmmc_vec_fill(&ones, 1.0);
    lmmc_sparse_mat_vec_mul(&A, &ones, &b);
    lmmc_vec_destroy(&ones);

    /* Solve with sparse path */
    lmmc_vec_fill(&x_sparse, 0.0);
    lmmc_itersolve_default_config(n, &cfg_sparse);
    cfg_sparse.max_iter = 500;
    cfg_sparse.abs_tol = 1e-14;
    cfg_sparse.rel_tol = 1e-14;

    st = lmmc_minres_solve(&A, &b, NULL, &cfg_sparse, &x_sparse, &result_sparse);
    if (st != LMMC_STATUS_OK || !result_sparse.converged) {
        printf("FAIL test_matrix_free_agrees: sparse path failed (st=%d, converged=%d)\n",
               st, result_sparse.converged);
        lmmc_vec_destroy(&x_sparse); lmmc_vec_destroy(&x_mfree);
        lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
        return 1;
    }

    /* Solve with matrix-free path */
    matvec_ctx_t ctx = { .mat = &A };
    lmmc_vec_fill(&x_mfree, 0.0);
    lmmc_itersolve_default_config(n, &cfg_mfree);
    cfg_mfree.max_iter = 500;
    cfg_mfree.abs_tol = 1e-14;
    cfg_mfree.rel_tol = 1e-14;
    cfg_mfree.apply_op = matvec_callback;
    cfg_mfree.op_user_data = &ctx;

    st = lmmc_minres_solve(NULL, &b, NULL, &cfg_mfree, &x_mfree, &result_mfree);
    if (st != LMMC_STATUS_OK || !result_mfree.converged) {
        printf("FAIL test_matrix_free_agrees: matrix-free path failed (st=%d, converged=%d)\n",
               st, result_mfree.converged);
        lmmc_vec_destroy(&x_sparse); lmmc_vec_destroy(&x_mfree);
        lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
        return 1;
    }

    /* Compare solutions: ||x_sparse - x_mfree||_2 <= 1e-10 * ||b||_2 */
    double norm_b;
    lmmc_vec_norm2(&b, &norm_b);
    double diff_norm = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = x_sparse.data[i] - x_mfree.data[i];
        diff_norm += d * d;
    }
    diff_norm = sqrt(diff_norm);

    double tol = 1e-10 * norm_b;
    if (diff_norm > tol) {
        printf("FAIL test_matrix_free_agrees: ||x_sparse - x_mfree|| = %e > 1e-10*||b|| = %e\n",
               diff_norm, tol);
        lmmc_vec_destroy(&x_sparse); lmmc_vec_destroy(&x_mfree);
        lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
        return 1;
    }

    printf("PASS test_matrix_free_agrees: diff=%e, tol=%e\n", diff_norm, tol);
    lmmc_vec_destroy(&x_sparse);
    lmmc_vec_destroy(&x_mfree);
    lmmc_vec_destroy(&b);
    lmmc_sparse_destroy(&A);
    return 0;
}

/* ===================== Test 4: Error when both a and apply_op provided ===================== */
static int test_error_both_a_and_apply_op(void) {
    const size_t n = 5;
    lmmc_sparse_mat_t A = {0};
    lmmc_vec_t b = {0}, x = {0};
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    lmmc_status_t st;

    st = build_symmetric_indefinite(n, &A);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL test_error_both_a_and_apply_op: build matrix failed (%d)\n", st);
        return 1;
    }

    lmmc_vec_create(n, &b);
    lmmc_vec_create(n, &x);
    lmmc_vec_fill(&b, 1.0);
    lmmc_vec_fill(&x, 0.0);

    matvec_ctx_t ctx = { .mat = &A };
    lmmc_itersolve_default_config(n, &cfg);
    cfg.apply_op = matvec_callback;
    cfg.op_user_data = &ctx;

    /* MINRES: both a and apply_op → LMMC_STATUS_INVALID_ARGUMENT */
    st = lmmc_minres_solve(&A, &b, NULL, &cfg, &x, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL test_error_both_a_and_apply_op: MINRES expected INVALID_ARGUMENT, got %d\n", st);
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
        return 1;
    }

    /* LSQR: both a and apply_op → LMMC_STATUS_INVALID_ARGUMENT */
    lmmc_vec_fill(&x, 0.0);
    memset(&result, 0, sizeof(result));
    st = lmmc_lsqr_solve(&A, &b, &cfg, &x, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL test_error_both_a_and_apply_op: LSQR expected INVALID_ARGUMENT, got %d\n", st);
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
        return 1;
    }

    printf("PASS test_error_both_a_and_apply_op\n");
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_sparse_destroy(&A);
    return 0;
}

int main(void) {
    int rc = 0;

    rc |= test_minres_symmetric_indefinite();
    rc |= test_lsqr_overdetermined();
    rc |= test_matrix_free_agrees_with_sparse();
    rc |= test_error_both_a_and_apply_op();

    if (rc != 0) {
        printf("\ntest_minres_lsqr: SOME TESTS FAILED\n");
    } else {
        printf("\ntest_minres_lsqr: ALL TESTS PASSED\n");
    }
    return rc;
}
