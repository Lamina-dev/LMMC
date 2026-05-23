/**
 * @file test_dense_mat.c
 * @brief Property-based tests for dense matrix operations.
 *
 * Property 6: 矩阵加减法互逆性
 *   For any two same-dimension matrices A and B, mat_add(A, B, C) then
 *   mat_sub(C, B, D) gives D ≈ A within machine precision.
 *
 * Property 7: 单位矩阵性质
 *   For any positive integer n, identity(n) has all 1s on diagonal,
 *   0s elsewhere, trace == n, det == 1.
 *
 * Validates: Requirements 3.1, 3.2, 3.4, 3.5, 3.6
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"

/* Number of random iterations for property tests */
#define NUM_ITERATIONS 100

/* Floating point tolerance for comparisons */
#define TOLERANCE 1e-12

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)

/* Generate a random double in [-range, range] */
static double rand_double(double range)
{
    return ((double)rand() / (double)RAND_MAX) * 2.0 * range - range;
}

/* Generate a random matrix size between min_size and max_size (inclusive) */
static size_t rand_size(size_t min_size, size_t max_size)
{
    return min_size + (size_t)(rand() % (int)(max_size - min_size + 1));
}

/* Fill a matrix with random values in [-range, range] */
static void fill_random_matrix(lmmc_mat_t* mat, double range)
{
    for (size_t i = 0; i < mat->rows; ++i) {
        for (size_t j = 0; j < mat->cols; ++j) {
            lmmc_real_t val = rand_double(range);
            LMMC_REAL_SET(&mat->data[i * mat->stride + j], &val);
        }
    }
}

/* ========================================================================
 * Property 6: 矩阵加减法互逆性
 * For any two same-dimension matrices A and B, mat_add(A, B, C) then
 * mat_sub(C, B, D) gives D ≈ A within machine precision.
 * Validates: Requirements 3.1, 3.2
 * ======================================================================== */

/**
 * Test: mat_add then mat_sub recovers original matrix (random sizes 2x2 to 8x8).
 */
static int test_mat_add_sub_inverse(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t rows = rand_size(2, 8);
        size_t cols = rand_size(2, 8);

        lmmc_mat_t A, B, C, D;
        lmmc_status_t status;

        /* Create matrices */
        status = lmmc_mat_create(rows, cols, &A);
        CHECK(status == LMMC_STATUS_OK, "iter %d: failed to create A (%zux%zu)", iter, rows, cols);

        status = lmmc_mat_create(rows, cols, &B);
        CHECK(status == LMMC_STATUS_OK, "iter %d: failed to create B (%zux%zu)", iter, rows, cols);

        status = lmmc_mat_create(rows, cols, &C);
        CHECK(status == LMMC_STATUS_OK, "iter %d: failed to create C (%zux%zu)", iter, rows, cols);

        status = lmmc_mat_create(rows, cols, &D);
        CHECK(status == LMMC_STATUS_OK, "iter %d: failed to create D (%zux%zu)", iter, rows, cols);

        /* Fill A and B with random values */
        fill_random_matrix(&A, 100.0);
        fill_random_matrix(&B, 100.0);

        /* C = A + B */
        status = lmmc_mat_add(&A, &B, &C);
        CHECK(status == LMMC_STATUS_OK, "iter %d: mat_add failed", iter);

        /* D = C - B, should equal A */
        status = lmmc_mat_sub(&C, &B, &D);
        CHECK(status == LMMC_STATUS_OK, "iter %d: mat_sub failed", iter);

        /* Verify D ≈ A within tolerance */
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                lmmc_real_t d_val = D.data[i * D.stride + j];
                lmmc_real_t a_val = A.data[i * A.stride + j];
                double diff = fabs(d_val - a_val);
                CHECK(diff <= TOLERANCE,
                      "iter %d: D[%zu][%zu] = %.15g != A[%zu][%zu] = %.15g (diff = %.2e)",
                      iter, i, j, d_val, i, j, a_val, diff);
            }
        }

        /* Cleanup */
        lmmc_mat_destroy(&A);
        lmmc_mat_destroy(&B);
        lmmc_mat_destroy(&C);
        lmmc_mat_destroy(&D);
    }

    return 0;
}

/**
 * Test: mat_sub then mat_add recovers original matrix (reverse direction).
 * This tests the inverse property from the other direction: (A - B) + B = A.
 */
static int test_mat_sub_add_inverse(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t rows = rand_size(2, 8);
        size_t cols = rand_size(2, 8);

        lmmc_mat_t A, B, C, D;
        lmmc_status_t status;

        /* Create matrices */
        status = lmmc_mat_create(rows, cols, &A);
        CHECK(status == LMMC_STATUS_OK, "iter %d: failed to create A", iter);

        status = lmmc_mat_create(rows, cols, &B);
        CHECK(status == LMMC_STATUS_OK, "iter %d: failed to create B", iter);

        status = lmmc_mat_create(rows, cols, &C);
        CHECK(status == LMMC_STATUS_OK, "iter %d: failed to create C", iter);

        status = lmmc_mat_create(rows, cols, &D);
        CHECK(status == LMMC_STATUS_OK, "iter %d: failed to create D", iter);

        /* Fill A and B with random values */
        fill_random_matrix(&A, 100.0);
        fill_random_matrix(&B, 100.0);

        /* C = A - B */
        status = lmmc_mat_sub(&A, &B, &C);
        CHECK(status == LMMC_STATUS_OK, "iter %d: mat_sub failed", iter);

        /* D = C + B, should equal A */
        status = lmmc_mat_add(&C, &B, &D);
        CHECK(status == LMMC_STATUS_OK, "iter %d: mat_add failed", iter);

        /* Verify D ≈ A within tolerance */
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                lmmc_real_t d_val = D.data[i * D.stride + j];
                lmmc_real_t a_val = A.data[i * A.stride + j];
                double diff = fabs(d_val - a_val);
                CHECK(diff <= TOLERANCE,
                      "iter %d: D[%zu][%zu] = %.15g != A[%zu][%zu] = %.15g (diff = %.2e)",
                      iter, i, j, d_val, i, j, a_val, diff);
            }
        }

        /* Cleanup */
        lmmc_mat_destroy(&A);
        lmmc_mat_destroy(&B);
        lmmc_mat_destroy(&C);
        lmmc_mat_destroy(&D);
    }

    return 0;
}

/* ========================================================================
 * Property 7: 单位矩阵性质
 * For any positive integer n, identity(n) has all 1s on diagonal,
 * 0s elsewhere, trace == n, det == 1.
 * Validates: Requirements 3.4, 3.5, 3.6
 * ======================================================================== */

/**
 * Test: identity matrix has correct structure (1s on diagonal, 0s elsewhere).
 */
static int test_identity_structure(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_size(2, 8);

        lmmc_mat_t I;
        lmmc_status_t status;

        status = lmmc_mat_identity(n, &I);
        CHECK(status == LMMC_STATUS_OK, "iter %d: mat_identity(%zu) failed", iter, n);

        /* Verify dimensions */
        CHECK(I.rows == n, "iter %d: identity rows = %zu, expected %zu", iter, I.rows, n);
        CHECK(I.cols == n, "iter %d: identity cols = %zu, expected %zu", iter, I.cols, n);

        /* Verify structure: diagonal = 1, off-diagonal = 0 */
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                lmmc_real_t val = I.data[i * I.stride + j];
                if (i == j) {
                    CHECK(fabs(val - 1.0) <= TOLERANCE,
                          "iter %d: I[%zu][%zu] = %.15g, expected 1.0",
                          iter, i, j, val);
                } else {
                    CHECK(fabs(val) <= TOLERANCE,
                          "iter %d: I[%zu][%zu] = %.15g, expected 0.0",
                          iter, i, j, val);
                }
            }
        }

        lmmc_mat_destroy(&I);
    }

    return 0;
}

/**
 * Test: trace of identity matrix equals n.
 */
static int test_identity_trace(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_size(2, 8);

        lmmc_mat_t I;
        lmmc_real_t trace;
        lmmc_status_t status;

        status = lmmc_mat_identity(n, &I);
        CHECK(status == LMMC_STATUS_OK, "iter %d: mat_identity(%zu) failed", iter, n);

        status = lmmc_mat_trace(&I, &trace);
        CHECK(status == LMMC_STATUS_OK, "iter %d: mat_trace failed", iter);

        CHECK(fabs(trace - (double)n) <= TOLERANCE,
              "iter %d: trace(I_%zu) = %.15g, expected %zu",
              iter, n, trace, n);

        lmmc_mat_destroy(&I);
    }

    return 0;
}

/**
 * Test: determinant of identity matrix equals 1.
 */
static int test_identity_det(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_size(2, 8);

        lmmc_mat_t I;
        lmmc_real_t det;
        lmmc_status_t status;

        status = lmmc_mat_identity(n, &I);
        CHECK(status == LMMC_STATUS_OK, "iter %d: mat_identity(%zu) failed", iter, n);

        status = lmmc_mat_det(&I, &det);
        CHECK(status == LMMC_STATUS_OK, "iter %d: mat_det failed", iter);

        CHECK(fabs(det - 1.0) <= TOLERANCE,
              "iter %d: det(I_%zu) = %.15g, expected 1.0",
              iter, n, det);

        lmmc_mat_destroy(&I);
    }

    return 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(void)
{
    int rc = 0;

    srand((unsigned int)time(NULL));

    printf("=== Property 6: 矩阵加减法互逆性 ===\n");
    printf("  Validates: Requirements 3.1, 3.2\n\n");

    if (test_mat_add_sub_inverse()) { rc = 1; printf("  [FAIL] mat_add then mat_sub inverse\n"); }
    else { printf("  [PASS] mat_add then mat_sub inverse\n"); }

    if (test_mat_sub_add_inverse()) { rc = 1; printf("  [FAIL] mat_sub then mat_add inverse\n"); }
    else { printf("  [PASS] mat_sub then mat_add inverse\n"); }

    printf("\n=== Property 7: 单位矩阵性质 ===\n");
    printf("  Validates: Requirements 3.4, 3.5, 3.6\n\n");

    if (test_identity_structure()) { rc = 1; printf("  [FAIL] identity structure\n"); }
    else { printf("  [PASS] identity structure\n"); }

    if (test_identity_trace()) { rc = 1; printf("  [FAIL] identity trace\n"); }
    else { printf("  [PASS] identity trace\n"); }

    if (test_identity_det()) { rc = 1; printf("  [FAIL] identity det\n"); }
    else { printf("  [PASS] identity det\n"); }

    printf("\n");
    if (rc == 0) {
        printf("All dense matrix property tests PASSED.\n");
    } else {
        printf("Some dense matrix property tests FAILED.\n");
    }

    return rc;
}
