/**
 * @file test_sparse_cholesky.c
 * @brief Unit tests for sparse Cholesky factorization (Requirements 9.1-9.7).
 */

#include <stdio.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static int test_count = 0;
static int pass_count = 0;

#define TEST_ASSERT(cond, msg) do { \
    test_count++; \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
    } else { \
        pass_count++; \
    } \
} while(0)

/* Helper: build a sparse SPD matrix from dense row-major data */
static lmmc_status_t build_sparse_spd(const lmmc_real_t* data, size_t n,
                                       lmmc_sparse_mat_t* out)
{
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(n, n, n * n, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t val = data[i * n + j];
            if (val != 0.0) {
                st = lmmc_sparse_builder_add(builder, i, j, val);
                if (st != LMMC_STATUS_OK) {
                    lmmc_sparse_builder_destroy(builder);
                    return st;
                }
            }
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

/* Test 1: Basic 3x3 SPD matrix solve */
static void test_basic_3x3(void)
{
    printf("Test: Basic 3x3 SPD matrix solve\n");

    /* A = [ 4  12 -16 ]
     *     [ 12 37 -43 ]
     *     [-16 -43 98 ]
     * This is SPD (eigenvalues are all positive).
     * Known solution: x = [1, 2, 3] for b = A*x = [-20, -43, 192]
     */
    lmmc_real_t A_data[] = {
         4.0,  12.0, -16.0,
        12.0,  37.0, -43.0,
       -16.0, -43.0,  98.0
    };

    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_sparse_spd(A_data, 3, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build sparse SPD matrix");

    /* Symbolic analysis (Req 9.1) */
    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "symbolic analysis succeeds");

    /* Numeric factorization (Req 9.2) */
    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "numeric factorization succeeds");

    /* Solve (Req 9.4) */
    st = lmmc_vec_create(3, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b vector");
    st = lmmc_vec_create(3, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x vector");

    b.data[0] = -20.0;
    b.data[1] = -43.0;
    b.data[2] = 192.0;

    st = lmmc_sparse_chol_solve(chol, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "solve succeeds");

    TEST_ASSERT(lmmc_test_nearly_equal(x.data[0], 1.0, 1e-10), "x[0] == 1.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[1], 2.0, 1e-10), "x[1] == 2.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[2], 3.0, 1e-10), "x[2] == 3.0");

    /* Cleanup (Req 9.5) */
    lmmc_sparse_chol_destroy(chol);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}

/* Test 2: 2x2 identity matrix */
static void test_identity_2x2(void)
{
    printf("Test: 2x2 identity matrix\n");

    lmmc_real_t A_data[] = {
        1.0, 0.0,
        0.0, 1.0
    };

    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_sparse_spd(A_data, 2, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build identity matrix");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "symbolic analysis");

    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "numeric factorization");

    st = lmmc_vec_create(2, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b");
    st = lmmc_vec_create(2, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x");

    b.data[0] = 3.0;
    b.data[1] = 7.0;

    st = lmmc_sparse_chol_solve(chol, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "solve succeeds");

    TEST_ASSERT(lmmc_test_nearly_equal(x.data[0], 3.0, 1e-10), "x[0] == 3.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[1], 7.0, 1e-10), "x[1] == 7.0");

    lmmc_sparse_chol_destroy(chol);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}

/* Test 3: Non-positive definite matrix detection (Req 9.6) */
static void test_not_positive_definite(void)
{
    printf("Test: Non-positive definite matrix detection\n");

    /* A = [1  2]
     *     [2  1]  -- eigenvalues: 3, -1 (not SPD)
     */
    lmmc_real_t A_data[] = {
        1.0, 2.0,
        2.0, 1.0
    };

    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_status_t st;

    st = build_sparse_spd(A_data, 2, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build non-SPD matrix");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "symbolic analysis succeeds for non-SPD");

    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_NOT_POSITIVE_DEFINITE,
                "numeric returns NOT_POSITIVE_DEFINITE for non-SPD matrix");

    lmmc_sparse_chol_destroy(chol);
    lmmc_sparse_destroy(&A);
}

/* Test 4: Non-square matrix detection (Req 9.7) */
static void test_non_square_matrix(void)
{
    printf("Test: Non-square matrix detection\n");

    /* Create a 2x3 sparse matrix */
    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_create_csc(2, 3, 0, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create non-square matrix");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_INVALID_ARGUMENT,
                "symbolic returns INVALID_ARGUMENT for non-square matrix");

    lmmc_sparse_destroy(&A);
}

/* Test 5: NULL pointer validation */
static void test_null_pointers(void)
{
    printf("Test: NULL pointer validation\n");

    lmmc_sparse_chol_t* chol = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_chol_symbolic(NULL, &chol);
    TEST_ASSERT(st == LMMC_STATUS_INVALID_ARGUMENT, "symbolic with NULL matrix");

    lmmc_sparse_mat_t A = {0};
    A.rows = 2; A.cols = 2;
    st = lmmc_sparse_chol_symbolic(&A, NULL);
    TEST_ASSERT(st == LMMC_STATUS_INVALID_ARGUMENT, "symbolic with NULL output");

    /* Solve with NULL chol */
    lmmc_vec_t b = {0}, x = {0};
    st = lmmc_sparse_chol_solve(NULL, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_INVALID_ARGUMENT, "solve with NULL chol");

    /* Destroy NULL is safe */
    lmmc_sparse_chol_destroy(NULL);
    TEST_ASSERT(1, "destroy NULL does not crash");
}

/* Test 6: Dimension mismatch in solve */
static void test_dimension_mismatch(void)
{
    printf("Test: Dimension mismatch in solve\n");

    lmmc_real_t A_data[] = {
        2.0, 1.0,
        1.0, 2.0
    };

    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_sparse_spd(A_data, 2, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build matrix");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "symbolic");

    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "numeric");

    /* Create vectors with wrong size */
    st = lmmc_vec_create(3, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b (wrong size)");
    st = lmmc_vec_create(2, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x");

    st = lmmc_sparse_chol_solve(chol, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_DIMENSION_MISMATCH,
                "solve returns DIMENSION_MISMATCH for wrong-size b");

    lmmc_sparse_chol_destroy(chol);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}

/* Test 7: Larger 4x4 SPD matrix */
static void test_4x4_spd(void)
{
    printf("Test: 4x4 SPD matrix solve\n");

    /* A = diag(2,3,4,5) + ones(4,4) -- diagonally dominant, hence SPD
     * A = [3 1 1 1]
     *     [1 4 1 1]
     *     [1 1 5 1]
     *     [1 1 1 6]
     */
    lmmc_real_t A_data[] = {
        3.0, 1.0, 1.0, 1.0,
        1.0, 4.0, 1.0, 1.0,
        1.0, 1.0, 5.0, 1.0,
        1.0, 1.0, 1.0, 6.0
    };

    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_sparse_spd(A_data, 4, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build 4x4 SPD matrix");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "symbolic");

    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "numeric");

    /* x_true = [1, 2, 3, 4], b = A * x_true */
    /* b[0] = 3*1 + 1*2 + 1*3 + 1*4 = 12 */
    /* b[1] = 1*1 + 4*2 + 1*3 + 1*4 = 16 */
    /* b[2] = 1*1 + 1*2 + 5*3 + 1*4 = 22 */
    /* b[3] = 1*1 + 1*2 + 1*3 + 6*4 = 30 */
    st = lmmc_vec_create(4, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b");
    st = lmmc_vec_create(4, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x");

    b.data[0] = 12.0;
    b.data[1] = 16.0;
    b.data[2] = 22.0;
    b.data[3] = 30.0;

    st = lmmc_sparse_chol_solve(chol, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "solve succeeds");

    TEST_ASSERT(lmmc_test_nearly_equal(x.data[0], 1.0, 1e-10), "x[0] == 1.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[1], 2.0, 1e-10), "x[1] == 2.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[2], 3.0, 1e-10), "x[2] == 3.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[3], 4.0, 1e-10), "x[3] == 4.0");

    lmmc_sparse_chol_destroy(chol);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}

int main(void)
{
    printf("=== Sparse Cholesky Factorization Tests ===\n\n");

    test_basic_3x3();
    test_identity_2x2();
    test_not_positive_definite();
    test_non_square_matrix();
    test_null_pointers();
    test_dimension_mismatch();
    test_4x4_spd();

    printf("\n=== Results: %d/%d tests passed ===\n", pass_count, test_count);

    return (pass_count == test_count) ? 0 : 1;
}
