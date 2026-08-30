/**
 * @file test_gemm_accuracy_prop.c
 * GEMM 精度属性测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/**
 * @brief Naive triple-loop GEMM reference implementation.
 *
 * Computes C_ref = alpha * op(A) * op(B) + beta * C0
 * where op(X) = X if trans == 0, X^T if trans != 0.
 *
 * @param alpha   Scalar multiplier for A*B product.
 * @param A       Input matrix A.
 * @param transA  Non-zero means transpose A.
 * @param B       Input matrix B.
 * @param transB  Non-zero means transpose B.
 * @param beta    Scalar multiplier for C0.
 * @param C0      Original C matrix (before GEMM).
 * @param C_ref   Output reference result (pre-created, same dims as C).
 */
static void naive_gemm(lmmc_real_t alpha, const lmmc_mat_t* A, int transA,
                        const lmmc_mat_t* B, int transB, lmmc_real_t beta,
                        const lmmc_mat_t* C0, lmmc_mat_t* C_ref) {
    size_t M = transA ? A->cols : A->rows;
    size_t K = transA ? A->rows : A->cols;
    size_t N = transB ? B->rows : B->cols;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                double a_ik;
                if (transA) {
                    a_ik = A->data[k * A->stride + i];
                } else {
                    a_ik = A->data[i * A->stride + k];
                }
                double b_kj;
                if (transB) {
                    b_kj = B->data[j * B->stride + k];
                } else {
                    b_kj = B->data[k * B->stride + j];
                }
                sum += a_ik * b_kj;
            }
            C_ref->data[i * C_ref->stride + j] =
                alpha * sum + beta * C0->data[i * C0->stride + j];
        }
    }
}

/**
 * @brief Generate a random scalar value from a set of interesting values.
 *
 * Returns 0, 1, -1, or a random value in [-5, 5] to cover edge cases.
 */
static lmmc_real_t random_scalar(lmmc_rng_t* rng) {
    lmmc_real_t u;
    lmmc_rng_uniform(rng, 0.0, 1.0, &u);
    if (u < 0.15) return 0.0;
    if (u < 0.30) return 1.0;
    if (u < 0.40) return -1.0;
    lmmc_real_t val;
    lmmc_rng_uniform(rng, -5.0, 5.0, &val);
    return val;
}

/**
 * @brief Fill a matrix with random values in [-1, 1].
 */
static void fill_random_matrix(lmmc_rng_t* rng, lmmc_mat_t* mat) {
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            lmmc_real_t val;
            lmmc_rng_uniform(rng, -1.0, 1.0, &val);
            mat->data[i * mat->stride + j] = val;
        }
    }
}

/**
 * @brief Test GEMM accuracy for a single random configuration.
 *
 * @param rng    Random number generator.
 * @param M      Rows of op(A) and C.
 * @param K      Cols of op(A) / rows of op(B).
 * @param N      Cols of op(B) and C.
 * @param trial  Trial number for reporting.
 * @return 0 on success, 1 on failure.
 */
static int test_gemm_accuracy(lmmc_rng_t* rng, size_t M, size_t K, size_t N, int trial) {
    lmmc_mat_t A = {0}, B = {0}, C = {0}, C0 = {0}, C_ref = {0};
    lmmc_status_t st;
    int rc = 0;

    /* Generate random transA, transB */
    lmmc_real_t u;
    lmmc_rng_uniform(rng, 0.0, 1.0, &u);
    int transA = (u > 0.5) ? 1 : 0;
    lmmc_rng_uniform(rng, 0.0, 1.0, &u);
    int transB = (u > 0.5) ? 1 : 0;

    /* Determine physical dimensions of A and B based on transpose flags */
    size_t A_rows = transA ? K : M;
    size_t A_cols = transA ? M : K;
    size_t B_rows = transB ? N : K;
    size_t B_cols = transB ? K : N;

    /* Generate random alpha and beta */
    lmmc_real_t alpha = random_scalar(rng);
    lmmc_real_t beta = random_scalar(rng);

    /* Create matrices */
    st = lmmc_mat_create(A_rows, A_cols, &A);
    if (st != LMMC_STATUS_OK) { printf("  [FAIL] trial %d: A create failed\n", trial); return 1; }

    st = lmmc_mat_create(B_rows, B_cols, &B);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); printf("  [FAIL] trial %d: B create failed\n", trial); return 1; }

    st = lmmc_mat_create(M, N, &C);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&B); lmmc_mat_destroy(&A); printf("  [FAIL] trial %d: C create failed\n", trial); return 1; }

    st = lmmc_mat_create(M, N, &C0);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&C); lmmc_mat_destroy(&B); lmmc_mat_destroy(&A); printf("  [FAIL] trial %d: C0 create failed\n", trial); return 1; }

    st = lmmc_mat_create(M, N, &C_ref);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&C0); lmmc_mat_destroy(&C); lmmc_mat_destroy(&B); lmmc_mat_destroy(&A); printf("  [FAIL] trial %d: C_ref create failed\n", trial); return 1; }

    /* Fill matrices with random values */
    fill_random_matrix(rng, &A);
    fill_random_matrix(rng, &B);
    fill_random_matrix(rng, &C0);

    /* Copy C0 into C (GEMM will modify C in-place) */
    st = lmmc_mat_copy(&C0, &C);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] trial %d: mat_copy failed\n", trial);
        rc = 1; goto cleanup;
    }

    /* Compute reference result using naive triple-loop */
    naive_gemm(alpha, &A, transA, &B, transB, beta, &C0, &C_ref);

    /* Compute result using lmmc_mat_gemm */
    st = lmmc_mat_gemm(alpha, &A, transA, &B, transB, beta, &C);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] trial %d: lmmc_mat_gemm returned %s\n", trial, lmmc_status_string(st));
        rc = 1; goto cleanup;
    }

    /* Compute norms for tolerance bound */
    lmmc_real_t norm_A, norm_B, norm_C0;
    st = lmmc_mat_norm_fro(&A, &norm_A);
    if (st != LMMC_STATUS_OK) { printf("  [FAIL] trial %d: norm_fro(A) failed\n", trial); rc = 1; goto cleanup; }
    st = lmmc_mat_norm_fro(&B, &norm_B);
    if (st != LMMC_STATUS_OK) { printf("  [FAIL] trial %d: norm_fro(B) failed\n", trial); rc = 1; goto cleanup; }
    st = lmmc_mat_norm_fro(&C0, &norm_C0);
    if (st != LMMC_STATUS_OK) { printf("  [FAIL] trial %d: norm_fro(C0) failed\n", trial); rc = 1; goto cleanup; }

    /* Tolerance: 1e-10 * (1 + |alpha| * ||A||_F * ||B||_F + |beta| * ||C0||_F) */
    double tolerance = 1e-10 * (1.0 + fabs(alpha) * norm_A * norm_B + fabs(beta) * norm_C0);

    /* Check element-wise accuracy */
    double max_err = 0.0;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double diff = fabs(C.data[i * C.stride + j] - C_ref.data[i * C_ref.stride + j]);
            if (diff > max_err) max_err = diff;
        }
    }

    if (max_err > tolerance) {
        printf("  [FAIL] trial %d (M=%zu, K=%zu, N=%zu, transA=%d, transB=%d, alpha=%.3f, beta=%.3f):\n"
               "    max_err = %.6e > tolerance = %.6e\n",
               trial, M, K, N, transA, transB, alpha, beta, max_err, tolerance);
        rc = 1; goto cleanup;
    }

cleanup:
    lmmc_mat_destroy(&C_ref);
    lmmc_mat_destroy(&C0);
    lmmc_mat_destroy(&C);
    lmmc_mat_destroy(&B);
    lmmc_mat_destroy(&A);
    return rc;
}

int main(void) {
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    int failures = 0;
    int total_trials = 0;

    printf("=== Property Test: GEMM Accuracy ===\n");
    printf("GEMM result matches naive triple-loop within tolerance\n");
    printf("Tolerance: 1e-10 * (1 + |alpha| * ||A||_F * ||B||_F + |beta| * ||C0||_F)\n");

    /* 固定种子保证属性测试可复现。 */
    const uint64_t seed = UINT64_C(0x47454D4D);
    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) {
        printf("FATAL: Failed to create RNG\n");
        return 1;
    }
    lmmc_rng_seed(rng, seed);

    /* Test various matrix sizes */
    typedef struct { size_t M; size_t K; size_t N; } dim_triple_t;
    dim_triple_t sizes[] = {
        {1, 1, 1},    /* Minimal */
        {2, 2, 2},    /* Small square */
        {3, 3, 3},
        {4, 4, 4},
        {5, 5, 5},
        {8, 8, 8},
        {10, 10, 10},
        {16, 16, 16},
        {20, 20, 20},
        {2, 5, 3},    /* Non-square */
        {5, 2, 7},
        {1, 10, 1},   /* Thin */
        {10, 1, 10},  /* Wide inner */
        {7, 4, 9},
        {15, 8, 12},
    };
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int trials_per_size = 8;

    for (size_t si = 0; si < num_sizes; si++) {
        size_t M = sizes[si].M;
        size_t K = sizes[si].K;
        size_t N = sizes[si].N;
        printf("Testing M=%zu, K=%zu, N=%zu (%d trials)...\n", M, K, N, trials_per_size);
        for (int t = 0; t < trials_per_size; t++) {
            total_trials++;
            if (test_gemm_accuracy(rng, M, K, N, t + 1) != 0) {
                failures++;
            }
        }
    }

    printf("\n=== Results ===\n");
    printf("Total trials: %d\n", total_trials);
    printf("Passed: %d\n", total_trials - failures);
    printf("Failed: %d\n", failures);

    if (failures == 0) {
        printf("\nProperty test PASSED: GEMM accuracy matches naive reference.\n");
    } else {
        printf("\nProperty test FAILED: %d/%d trials violated the property.\n", failures, total_trials);
    }

    lmmc_rng_destroy(rng);
    return (failures == 0) ? 0 : 1;
}
