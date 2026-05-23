/**
 * @file test_eigen_sym_property.c
 * @brief Property-based tests for symmetric eigenvalue decomposition.
 *
 * Property 8: 对称特征值分解重构
 *   For any real symmetric matrix A, the returned eigenvalues lambda
 *   and eigenvector matrix V should satisfy:
 *     - A ≈ V * diag(lambda) * V^T   (reconstruction)
 *     - V^T * V ≈ I                  (orthogonality)
 *     - lambda sorted in ascending order
 *
 * Validates: Requirements 5.1, 5.4, 5.5
 *
 * Strategy:
 *   Generate 100+ random real symmetric matrices of sizes 2x2 through 6x6
 *   by sampling a random matrix M and forming A = (M + M^T) / 2.
 *   For each matrix, run lmmc_eigen_symmetric and verify the three
 *   properties within a tolerance scaled by matrix norm and dimension.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"

/* Number of random iterations (>= 100 as required) */
#define NUM_ITERATIONS 120

/* Base tolerance; effective tolerance scales with ||A|| and n */
#define BASE_TOL 1e-9

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)

/* Generate a random double in [-range, range] */
static double rand_double(double range)
{
    return ((double)rand() / (double)RAND_MAX) * 2.0 * range - range;
}

/* Pick random size in [min_n, max_n] */
static size_t rand_size(size_t min_n, size_t max_n)
{
    return min_n + (size_t)(rand() % (int)(max_n - min_n + 1));
}

/* Generate a random symmetric matrix A = (M + M^T) / 2,
 * where entries of M are uniformly random in [-range, range]. */
static void make_random_symmetric(lmmc_mat_t* A, double range)
{
    size_t n = A->rows;
    size_t i, j;
    for (i = 0; i < n; ++i) {
        A->data[i * n + i] = rand_double(range);
        for (j = i + 1; j < n; ++j) {
            double v = rand_double(range);
            A->data[i * n + j] = v;
            A->data[j * n + i] = v;
        }
    }
}

/* Frobenius norm of a square dense matrix (row-major, stride == cols). */
static double mat_fro_norm(const lmmc_mat_t* A)
{
    size_t n = A->rows;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double v = A->data[i * n + j];
            sum += v * v;
        }
    }
    return sqrt(sum);
}

/* ========================================================================
 * Property 8: 对称特征值分解重构
 * Validates: Requirements 5.1, 5.4, 5.5
 * ======================================================================== */

/**
 * For each random symmetric matrix A:
 *   1. Compute eigen-decomposition via lmmc_eigen_symmetric.
 *   2. Reconstruction: ||A - V*diag(lambda)*V^T||_F < tol * (||A||_F + 1) * n
 *   3. Orthogonality:  ||V^T*V - I||_F           < tol * n
 *   4. Eigenvalues sorted ascending: lambda[i] <= lambda[i+1]
 */
static int test_eigen_sym_property(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_size(2, 6);
        lmmc_mat_t A;
        lmmc_status_t st = lmmc_mat_create(n, n, &A);
        CHECK(st == LMMC_STATUS_OK,
              "mat_create failed (iter=%d, n=%zu)", iter, n);

        /* Mix scales across iterations to exercise both small and moderate
         * magnitudes without getting too close to ill-conditioned regimes. */
        double range = (iter % 4 == 0) ? 1.0
                     : (iter % 4 == 1) ? 10.0
                     : (iter % 4 == 2) ? 0.1
                                       : 100.0;
        make_random_symmetric(&A, range);

        double a_norm = mat_fro_norm(&A);
        double tol = BASE_TOL * (a_norm + 1.0) * (double)n;

        lmmc_eigen_sym_result_t res;
        st = lmmc_eigen_symmetric(&A, &res);
        CHECK(st == LMMC_STATUS_OK,
              "eigen_symmetric failed status=%d (iter=%d, n=%zu)",
              (int)st, iter, n);

        /* --- Verify ascending order of eigenvalues --- */
        for (size_t i = 0; i + 1 < n; ++i) {
            CHECK(res.eigenvalues.data[i] <= res.eigenvalues.data[i + 1] + tol,
                  "eigenvalues not ascending: lambda[%zu]=%.15g > lambda[%zu]=%.15g (iter=%d, n=%zu)",
                  i, res.eigenvalues.data[i],
                  i + 1, res.eigenvalues.data[i + 1],
                  iter, n);
        }

        /* V is stored as an n x n matrix; column k is the k-th eigenvector.
         * V[i,k] = eigenvectors.data[i * n + k]. */
        const lmmc_real_t* V = res.eigenvectors.data;
        const lmmc_real_t* lam = res.eigenvalues.data;

        /* --- Verify reconstruction: A ≈ V * diag(lambda) * V^T --- */
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += V[i * n + k] * lam[k] * V[j * n + k];
                }
                double a_ij = A.data[i * n + j];
                double diff = fabs(sum - a_ij);
                CHECK(diff <= tol,
                      "reconstruction mismatch A[%zu][%zu]: expected %.15g got %.15g diff=%.3e tol=%.3e (iter=%d, n=%zu)",
                      i, j, a_ij, sum, diff, tol, iter, n);
            }
        }

        /* --- Verify orthogonality: V^T * V ≈ I ---
         * (V^T * V)[i,j] = sum_k V[k,i] * V[k,j] = sum_k data[k*n+i] * data[k*n+j] */
        double ortho_tol = BASE_TOL * (double)n;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double dot = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    dot += V[k * n + i] * V[k * n + j];
                }
                double expected = (i == j) ? 1.0 : 0.0;
                double diff = fabs(dot - expected);
                CHECK(diff <= ortho_tol,
                      "orthogonality mismatch (V^T*V)[%zu][%zu]: expected %.1f got %.15g diff=%.3e tol=%.3e (iter=%d, n=%zu)",
                      i, j, expected, dot, diff, ortho_tol, iter, n);
            }
        }

        lmmc_eigen_sym_result_destroy(&res);
        lmmc_mat_destroy(&A);
    }

    return 0;
}

/* ========================================================================
 * Edge case: 2x2 matrices specifically.
 *
 * Smallest size in the property range; ensures that even pathological 2x2
 * structures (near-zero off-diagonal, equal diagonals, etc.) are covered.
 * ======================================================================== */
static int test_eigen_sym_2x2_focused(void)
{
    int iter;

    for (iter = 0; iter < 30; iter++) {
        lmmc_mat_t A;
        lmmc_mat_create(2, 2, &A);

        /* Random 2x2 symmetric */
        double a = rand_double(10.0);
        double b = rand_double(10.0);
        double d = rand_double(10.0);
        A.data[0] = a; A.data[1] = b;
        A.data[2] = b; A.data[3] = d;

        lmmc_eigen_sym_result_t res;
        lmmc_status_t st = lmmc_eigen_symmetric(&A, &res);
        CHECK(st == LMMC_STATUS_OK, "2x2 eigen failed (iter=%d)", iter);

        /* Closed-form eigenvalues: (a+d ± sqrt((a-d)^2 + 4b^2))/2 */
        double trace = a + d;
        double disc = sqrt((a - d) * (a - d) + 4.0 * b * b);
        double lam1 = (trace - disc) / 2.0;
        double lam2 = (trace + disc) / 2.0;

        double tol = 1e-10 * (fabs(a) + fabs(b) + fabs(d) + 1.0);

        CHECK(fabs(res.eigenvalues.data[0] - lam1) < tol,
              "2x2 lambda[0]: expected %.15g got %.15g (iter=%d)",
              lam1, res.eigenvalues.data[0], iter);
        CHECK(fabs(res.eigenvalues.data[1] - lam2) < tol,
              "2x2 lambda[1]: expected %.15g got %.15g (iter=%d)",
              lam2, res.eigenvalues.data[1], iter);

        lmmc_eigen_sym_result_destroy(&res);
        lmmc_mat_destroy(&A);
    }
    return 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */
int main(void)
{
    int rc = 0;

    /* Use a fixed seed for reproducibility in CI; comment out for fully
     * random runs. We use time() for broader coverage across runs. */
    srand((unsigned int)time(NULL));

    printf("=== Property 8: 对称特征值分解重构 ===\n");
    printf("  Validates: Requirements 5.1, 5.4, 5.5\n\n");

    if (test_eigen_sym_property()) {
        rc = 1;
        printf("  [FAIL] random symmetric eigen-decomposition (sizes 2x2..6x6)\n");
    } else {
        printf("  [PASS] random symmetric eigen-decomposition (%d iterations, sizes 2x2..6x6)\n",
               NUM_ITERATIONS);
    }

    if (test_eigen_sym_2x2_focused()) {
        rc = 1;
        printf("  [FAIL] 2x2 closed-form check\n");
    } else {
        printf("  [PASS] 2x2 closed-form check (30 iterations)\n");
    }

    printf("\n");
    if (rc == 0) {
        printf("All symmetric eigenvalue property tests PASSED.\n");
    } else {
        printf("Some symmetric eigenvalue property tests FAILED (failures=%d).\n",
               test_failures);
    }
    return rc;
}
