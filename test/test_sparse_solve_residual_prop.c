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

/* --------------------------------------------------------------------------
 * Helper: Generate a random sparse non-singular matrix for LU testing.
 *
 * Strategy: Start with a diagonally dominant matrix to ensure non-singularity.
 * For each row, place the diagonal entry as sum of absolute off-diagonals + 1,
 * and randomly place a few off-diagonal entries.
 * -------------------------------------------------------------------------- */
static lmmc_status_t generate_sparse_nonsingular(
    lmmc_rng_t* rng, size_t n, size_t nnz_per_row,
    lmmc_sparse_mat_t* out_A)
{
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(n, n, n * (nnz_per_row + 1), &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {
        double row_sum = 0.0;

        /* Add random off-diagonal entries */
        for (size_t k = 0; k < nnz_per_row; k++) {
            lmmc_real_t u;
            lmmc_rng_uniform(rng, 0.0, 1.0, &u);
            size_t j = (size_t)(u * (double)n);
            if (j >= n) j = n - 1;
            if (j == i) j = (i + 1) % n; /* avoid diagonal */

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

/* --------------------------------------------------------------------------
 * Helper: Generate a random sparse SPD matrix for Cholesky testing.
 *
 * Strategy: Build a sparse lower-triangular L with positive diagonal,
 * then compute A = L * L^T via the builder (explicit construction).
 * For simplicity, we use a diagonally dominant approach:
 *   A = B + B^T + (n + 1) * I
 * where B is a sparse matrix with random entries.
 * -------------------------------------------------------------------------- */
static lmmc_status_t generate_sparse_spd(
    lmmc_rng_t* rng, size_t n, size_t nnz_per_row,
    lmmc_sparse_mat_t* out_A)
{
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(n, n, n * (2 * nnz_per_row + 1), &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {
        double off_diag_sum = 0.0;

        /* Add random symmetric off-diagonal entries */
        for (size_t k = 0; k < nnz_per_row; k++) {
            lmmc_real_t u;
            lmmc_rng_uniform(rng, 0.0, 1.0, &u);
            size_t j = (size_t)(u * (double)n);
            if (j >= n) j = n - 1;
            if (j == i) continue; /* skip diagonal */

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

/* --------------------------------------------------------------------------
 * Helper: Generate a random vector b of size n with entries in [-1, 1].
 * -------------------------------------------------------------------------- */
static lmmc_status_t generate_random_vec(lmmc_rng_t* rng, size_t n, lmmc_vec_t* out_b)
{
    lmmc_status_t st = lmmc_vec_create(n, out_b);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {
        lmmc_real_t val;
        lmmc_rng_uniform(rng, -1.0, 1.0, &val);
        out_b->data[i] = val;
    }
    return LMMC_STATUS_OK;
}

/* --------------------------------------------------------------------------
 * Helper: Compute residual ||Ax - b||_2 using sparse mat-vec multiply.
 * -------------------------------------------------------------------------- */
static lmmc_status_t compute_residual_norm(
    const lmmc_sparse_mat_t* A, const lmmc_vec_t* x, const lmmc_vec_t* b,
    lmmc_real_t* out_residual_norm)
{
    lmmc_vec_t Ax = {0};
    lmmc_status_t st;

    st = lmmc_vec_create(A->rows, &Ax);
    if (st != LMMC_STATUS_OK) return st;

    /* Compute Ax */
    st = lmmc_sparse_mat_vec_mul(A, x, &Ax);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&Ax); return st; }

    /* Compute Ax - b */
    for (size_t i = 0; i < A->rows; i++) {
        Ax.data[i] -= b->data[i];
    }

    /* Compute ||Ax - b||_2 */
    st = lmmc_vec_norm2(&Ax, out_residual_norm);
    lmmc_vec_destroy(&Ax);
    return st;
}

/* --------------------------------------------------------------------------
 * Test: Sparse LU solve residual property.
 *
 * For a random non-singular sparse matrix A and random vector b:
 *   1. Perform symbolic factorization
 *   2. Perform numeric factorization
 *   3. Solve Ax = b
 *   4. Verify ||Ax - b||_2 <= 1e-8 * (||A||_F * ||x||_2 + ||b||_2)
 * -------------------------------------------------------------------------- */
static int test_sparse_lu_residual(lmmc_rng_t* rng, size_t n, size_t nnz_per_row, int trial)
{
    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_lu_t* lu = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;
    int rc = 0;

    /* Generate random non-singular sparse matrix */
    st = generate_sparse_nonsingular(rng, n, nnz_per_row, &A);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] LU trial %d (n=%zu): generate matrix failed: %d\n", trial, n, (int)st);
        return 1;
    }

    /* Generate random RHS vector */
    st = generate_random_vec(rng, n, &b);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] LU trial %d (n=%zu): generate b failed\n", trial, n);
        lmmc_sparse_destroy(&A);
        return 1;
    }

    /* Create solution vector */
    st = lmmc_vec_create(n, &x);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] LU trial %d (n=%zu): create x failed\n", trial, n);
        lmmc_vec_destroy(&b);
        lmmc_sparse_destroy(&A);
        return 1;
    }

    /* Three-stage pipeline: symbolic -> numeric -> solve */
    st = lmmc_sparse_lu_symbolic(&A, &lu);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] LU trial %d (n=%zu): symbolic failed: %d\n", trial, n, (int)st);
        rc = 1; goto cleanup;
    }

    st = lmmc_sparse_lu_numeric(&A, lu);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] LU trial %d (n=%zu): numeric failed: %d\n", trial, n, (int)st);
        rc = 1; goto cleanup;
    }

    st = lmmc_sparse_lu_solve(lu, &b, &x);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] LU trial %d (n=%zu): solve failed: %d\n", trial, n, (int)st);
        rc = 1; goto cleanup;
    }

    /* Compute residual and check bound */
    {
        lmmc_real_t residual_norm = 0.0;
        lmmc_real_t A_norm = 0.0;
        lmmc_real_t x_norm = 0.0;
        lmmc_real_t b_norm = 0.0;

        st = compute_residual_norm(&A, &x, &b, &residual_norm);
        if (st != LMMC_STATUS_OK) {
            printf("  [FAIL] LU trial %d (n=%zu): residual computation failed\n", trial, n);
            rc = 1; goto cleanup;
        }

        st = lmmc_sparse_norm_fro(&A, &A_norm);
        if (st != LMMC_STATUS_OK) {
            printf("  [FAIL] LU trial %d (n=%zu): A norm computation failed\n", trial, n);
            rc = 1; goto cleanup;
        }

        st = lmmc_vec_norm2(&x, &x_norm);
        if (st != LMMC_STATUS_OK) {
            printf("  [FAIL] LU trial %d (n=%zu): x norm computation failed\n", trial, n);
            rc = 1; goto cleanup;
        }

        st = lmmc_vec_norm2(&b, &b_norm);
        if (st != LMMC_STATUS_OK) {
            printf("  [FAIL] LU trial %d (n=%zu): b norm computation failed\n", trial, n);
            rc = 1; goto cleanup;
        }

        double bound = 1e-8 * (A_norm * x_norm + b_norm);
        if (residual_norm > bound) {
            printf("  [FAIL] LU trial %d (n=%zu): ||Ax-b||_2 = %.6e > 1e-8*(||A||_F*||x||_2 + ||b||_2) = %.6e\n",
                   trial, n, residual_norm, bound);
            rc = 1; goto cleanup;
        }
    }

cleanup:
    lmmc_sparse_lu_destroy(lu);
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_sparse_destroy(&A);
    return rc;
}

/* --------------------------------------------------------------------------
 * Test: Sparse Cholesky solve residual property.
 *
 * For a random sparse SPD matrix A and random vector b:
 *   1. Perform symbolic factorization
 *   2. Perform numeric factorization
 *   3. Solve Ax = b
 *   4. Verify ||Ax - b||_2 <= 1e-8 * (||A||_F * ||x||_2 + ||b||_2)
 * -------------------------------------------------------------------------- */
static int test_sparse_chol_residual(lmmc_rng_t* rng, size_t n, size_t nnz_per_row, int trial)
{
    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;
    int rc = 0;

    /* Generate random sparse SPD matrix */
    st = generate_sparse_spd(rng, n, nnz_per_row, &A);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] Chol trial %d (n=%zu): generate SPD matrix failed: %d\n", trial, n, (int)st);
        return 1;
    }

    /* Generate random RHS vector */
    st = generate_random_vec(rng, n, &b);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] Chol trial %d (n=%zu): generate b failed\n", trial, n);
        lmmc_sparse_destroy(&A);
        return 1;
    }

    /* Create solution vector */
    st = lmmc_vec_create(n, &x);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] Chol trial %d (n=%zu): create x failed\n", trial, n);
        lmmc_vec_destroy(&b);
        lmmc_sparse_destroy(&A);
        return 1;
    }

    /* Three-stage pipeline: symbolic -> numeric -> solve */
    st = lmmc_sparse_chol_symbolic(&A, &chol);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] Chol trial %d (n=%zu): symbolic failed: %d\n", trial, n, (int)st);
        rc = 1; goto cleanup;
    }

    st = lmmc_sparse_chol_numeric(&A, chol);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] Chol trial %d (n=%zu): numeric failed: %d\n", trial, n, (int)st);
        rc = 1; goto cleanup;
    }

    st = lmmc_sparse_chol_solve(chol, &b, &x);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] Chol trial %d (n=%zu): solve failed: %d\n", trial, n, (int)st);
        rc = 1; goto cleanup;
    }

    /* Compute residual and check bound */
    {
        lmmc_real_t residual_norm = 0.0;
        lmmc_real_t A_norm = 0.0;
        lmmc_real_t x_norm = 0.0;
        lmmc_real_t b_norm = 0.0;

        st = compute_residual_norm(&A, &x, &b, &residual_norm);
        if (st != LMMC_STATUS_OK) {
            printf("  [FAIL] Chol trial %d (n=%zu): residual computation failed\n", trial, n);
            rc = 1; goto cleanup;
        }

        st = lmmc_sparse_norm_fro(&A, &A_norm);
        if (st != LMMC_STATUS_OK) {
            printf("  [FAIL] Chol trial %d (n=%zu): A norm computation failed\n", trial, n);
            rc = 1; goto cleanup;
        }

        st = lmmc_vec_norm2(&x, &x_norm);
        if (st != LMMC_STATUS_OK) {
            printf("  [FAIL] Chol trial %d (n=%zu): x norm computation failed\n", trial, n);
            rc = 1; goto cleanup;
        }

        st = lmmc_vec_norm2(&b, &b_norm);
        if (st != LMMC_STATUS_OK) {
            printf("  [FAIL] Chol trial %d (n=%zu): b norm computation failed\n", trial, n);
            rc = 1; goto cleanup;
        }

        double bound = 1e-8 * (A_norm * x_norm + b_norm);
        if (residual_norm > bound) {
            printf("  [FAIL] Chol trial %d (n=%zu): ||Ax-b||_2 = %.6e > 1e-8*(||A||_F*||x||_2 + ||b||_2) = %.6e\n",
                   trial, n, residual_norm, bound);
            rc = 1; goto cleanup;
        }
    }

cleanup:
    lmmc_sparse_chol_destroy(chol);
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_sparse_destroy(&A);
    return rc;
}

int main(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    int failures = 0;
    int total_trials = 0;

    printf("=== Property Test: Sparse Solve Residual ===\n");
    printf("Property: ||Ax - b||_2 <= 1e-8 * (||A||_F * ||x||_2 + ||b||_2)\n");

    /* 固定种子保证属性测试可复现。 */
    const uint64_t seed = UINT64_C(0x53505253);
    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) {
        printf("FATAL: Failed to create RNG\n");
        return 1;
    }
    lmmc_rng_seed(rng, seed);

    /* Test sizes from 5x5 to 30x30 */
    size_t sizes[] = {5, 7, 10, 12, 15, 18, 20, 25, 30};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int trials_per_size = 10;
    size_t nnz_per_row = 3; /* Average off-diagonal entries per row */

    /* ---- Sparse LU path ---- */
    printf("--- Testing Sparse LU Path ---\n");
    for (size_t si = 0; si < num_sizes; si++) {
        size_t n = sizes[si];
        printf("  LU: n=%zu (%d trials)...\n", n, trials_per_size);
        for (int t = 0; t < trials_per_size; t++) {
            total_trials++;
            if (test_sparse_lu_residual(rng, n, nnz_per_row, t + 1) != 0) {
                failures++;
            }
        }
    }

    /* ---- Sparse Cholesky path ---- */
    printf("\n--- Testing Sparse Cholesky Path ---\n");
    for (size_t si = 0; si < num_sizes; si++) {
        size_t n = sizes[si];
        printf("  Chol: n=%zu (%d trials)...\n", n, trials_per_size);
        for (int t = 0; t < trials_per_size; t++) {
            total_trials++;
            if (test_sparse_chol_residual(rng, n, nnz_per_row, t + 1) != 0) {
                failures++;
            }
        }
    }

    printf("\n=== Results ===\n");
    printf("Total trials: %d\n", total_trials);
    printf("Passed: %d\n", total_trials - failures);
    printf("Failed: %d\n", failures);

    if (failures == 0) {
        printf("\nProperty test PASSED: Sparse solve residual bound satisfied.\n");
    } else {
        printf("\nProperty test FAILED: %d/%d trials violated the property.\n", failures, total_trials);
    }

    lmmc_rng_destroy(rng);
    return (failures == 0) ? 0 : 1;
}
