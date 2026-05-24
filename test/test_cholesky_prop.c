/**
 * @file test_cholesky_prop.c
 * @brief Property-based test for Cholesky reconstruction accuracy.
 *
 * **Validates: Requirements 3.4**
 *
 * Property 3: Cholesky reconstruction
 * Verifies that for random SPD matrices A, the Cholesky factor L satisfies:
 *   ||L * L^T - A||_F <= 1e-10 * ||A||_F
 *
 * SPD matrices are generated as A = B * B^T + eps * I for random B.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/**
 * @brief Generate a random SPD matrix of size n x n.
 *
 * Constructs A = B * B^T + eps * I where B is n x n with entries in [-1, 1].
 * This guarantees A is symmetric positive definite.
 *
 * @param rng   Random number generator.
 * @param n     Matrix dimension.
 * @param eps   Diagonal regularization (e.g. 0.01).
 * @param out_a Output matrix (must be pre-created as n x n).
 * @return LMMC_STATUS_OK on success.
 */
static lmmc_status_t generate_random_spd(lmmc_rng_t* rng, size_t n, double eps, lmmc_mat_t* out_a) {
    lmmc_mat_t b = {0};
    lmmc_mat_t bt = {0};
    lmmc_status_t st;

    st = lmmc_mat_create(n, n, &b);
    if (st != LMMC_STATUS_OK) return st;

    st = lmmc_mat_create(n, n, &bt);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&b); return st; }

    /* Fill B with random values in [-1, 1] */
    for (size_t i = 0; i < n * n; i++) {
        lmmc_real_t val;
        lmmc_rng_uniform(rng, -1.0, 1.0, &val);
        b.data[i] = val;
    }

    /* Compute B^T */
    st = lmmc_mat_transpose_to(&b, &bt);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&bt); lmmc_mat_destroy(&b); return st; }

    /* Compute A = B * B^T */
    st = lmmc_mat_mul(&b, &bt, out_a);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&bt); lmmc_mat_destroy(&b); return st; }

    /* Add eps * I to ensure strict positive definiteness */
    for (size_t i = 0; i < n; i++) {
        out_a->data[i * out_a->stride + i] += eps;
    }

    lmmc_mat_destroy(&bt);
    lmmc_mat_destroy(&b);
    return LMMC_STATUS_OK;
}

/**
 * @brief Compute the Frobenius norm of the difference between two matrices.
 *
 * @param a First matrix.
 * @param b Second matrix (same dimensions as a).
 * @return The Frobenius norm ||a - b||_F.
 */
static double frobenius_norm_diff(const lmmc_mat_t* a, const lmmc_mat_t* b) {
    double sum = 0.0;
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            double diff = a->data[i * a->stride + j] - b->data[i * b->stride + j];
            sum += diff * diff;
        }
    }
    return sqrt(sum);
}

/**
 * @brief Reconstruct L * L^T from the lower-triangular Cholesky factor.
 *
 * After lmmc_cholesky_decompose_inplace, the lower triangle of the matrix
 * contains L. This function computes L * L^T into out_reconstructed.
 *
 * @param l_mat            Matrix containing L in its lower triangle.
 * @param n                Dimension.
 * @param out_reconstructed Output matrix (pre-created n x n).
 */
static void reconstruct_from_cholesky(const lmmc_mat_t* l_mat, size_t n, lmmc_mat_t* out_reconstructed) {
    /* Zero out the result */
    for (size_t i = 0; i < n * n; i++) {
        out_reconstructed->data[i] = 0.0;
    }

    /* Compute L * L^T manually using only the lower triangle of l_mat */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            /* (L * L^T)[i][j] = sum_k L[i][k] * L[j][k] for k = 0..min(i,j) */
            double sum = 0.0;
            for (size_t k = 0; k <= j; k++) {
                double l_ik = l_mat->data[i * l_mat->stride + k];
                double l_jk = l_mat->data[j * l_mat->stride + k];
                sum += l_ik * l_jk;
            }
            out_reconstructed->data[i * out_reconstructed->stride + j] = sum;
            out_reconstructed->data[j * out_reconstructed->stride + i] = sum;
        }
    }
}

/**
 * @brief Test the Cholesky reconstruction property for a single random SPD matrix.
 *
 * @param rng  Random number generator.
 * @param n    Matrix dimension.
 * @param trial Trial number (for reporting).
 * @return 0 on success, 1 on failure.
 */
static int test_cholesky_reconstruction(lmmc_rng_t* rng, size_t n, int trial) {
    lmmc_mat_t a = {0};
    lmmc_mat_t a_copy = {0};
    lmmc_mat_t reconstructed = {0};
    lmmc_status_t st;
    int rc = 0;

    /* Create matrices */
    st = lmmc_mat_create(n, n, &a);
    if (st != LMMC_STATUS_OK) { printf("  [FAIL] trial %d (n=%zu): mat_create failed\n", trial, n); return 1; }

    st = lmmc_mat_create(n, n, &a_copy);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); printf("  [FAIL] trial %d (n=%zu): mat_create failed\n", trial, n); return 1; }

    st = lmmc_mat_create(n, n, &reconstructed);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a_copy); lmmc_mat_destroy(&a); printf("  [FAIL] trial %d (n=%zu): mat_create failed\n", trial, n); return 1; }

    /* Generate random SPD matrix */
    st = generate_random_spd(rng, n, 0.01, &a);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] trial %d (n=%zu): generate_random_spd failed\n", trial, n);
        rc = 1; goto cleanup;
    }

    /* Save a copy of A before decomposition (since it's done in-place) */
    st = lmmc_mat_copy(&a, &a_copy);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] trial %d (n=%zu): mat_copy failed\n", trial, n);
        rc = 1; goto cleanup;
    }

    /* Perform Cholesky decomposition */
    st = lmmc_cholesky_decompose_inplace(&a);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] trial %d (n=%zu): cholesky_decompose_inplace failed: %s\n",
               trial, n, lmmc_status_string(st));
        rc = 1; goto cleanup;
    }

    /* Reconstruct L * L^T */
    reconstruct_from_cholesky(&a, n, &reconstructed);

    /* Compute ||L*L^T - A||_F */
    double diff_norm = frobenius_norm_diff(&reconstructed, &a_copy);

    /* Compute ||A||_F */
    lmmc_real_t a_norm;
    st = lmmc_mat_norm_fro(&a_copy, &a_norm);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] trial %d (n=%zu): mat_norm_fro failed\n", trial, n);
        rc = 1; goto cleanup;
    }

    /* Check property: ||L*L^T - A||_F <= 1e-10 * ||A||_F */
    double tolerance = 1e-10 * a_norm;
    if (diff_norm > tolerance) {
        printf("  [FAIL] trial %d (n=%zu): ||L*L^T - A||_F = %.6e > 1e-10 * ||A||_F = %.6e\n",
               trial, n, diff_norm, tolerance);
        rc = 1; goto cleanup;
    }

cleanup:
    lmmc_mat_destroy(&reconstructed);
    lmmc_mat_destroy(&a_copy);
    lmmc_mat_destroy(&a);
    return rc;
}

int main(void) {
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    int failures = 0;
    int total_trials = 0;

    printf("=== Property Test: Cholesky Reconstruction ===\n");
    printf("Property: ||L*L^T - A||_F <= 1e-10 * ||A||_F for random SPD inputs\n");
    printf("Validates: Requirements 3.4\n\n");

    /* Create RNG with a seed based on time for randomness */
    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) {
        printf("FATAL: Failed to create RNG\n");
        return 1;
    }
    lmmc_rng_seed(rng, (uint64_t)time(NULL));

    /* Test various matrix sizes from 2x2 to 20x20 */
    size_t sizes[] = {2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int trials_per_size = 10;

    for (size_t si = 0; si < num_sizes; si++) {
        size_t n = sizes[si];
        printf("Testing n=%zu (%d trials)...\n", n, trials_per_size);
        for (int t = 0; t < trials_per_size; t++) {
            total_trials++;
            if (test_cholesky_reconstruction(rng, n, t + 1) != 0) {
                failures++;
            }
        }
    }

    printf("\n=== Results ===\n");
    printf("Total trials: %d\n", total_trials);
    printf("Passed: %d\n", total_trials - failures);
    printf("Failed: %d\n", failures);

    if (failures == 0) {
        printf("\nProperty test PASSED: Cholesky reconstruction is accurate.\n");
    } else {
        printf("\nProperty test FAILED: %d/%d trials violated the property.\n", failures, total_trials);
    }

    lmmc_rng_destroy(rng);
    return (failures == 0) ? 0 : 1;
}
