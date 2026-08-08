/**
 * @file test_sparse_buffer_safety.c
 * 稀疏 LU/Cholesky 缓冲区安全测试。
 */
#include <stdio.h>
#include <stdlib.h>
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

/* --------------------------------------------------------------------------
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
 * -------------------------------------------------------------------------- */
static lmmc_status_t build_arrowhead_spd(size_t n, lmmc_sparse_mat_t* out)
{
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st;
    lmmc_real_t diag_val = (lmmc_real_t)(n + 1);

    st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {
        /* Diagonal */
        st = lmmc_sparse_builder_add(builder, i, i, diag_val);
        if (st != LMMC_STATUS_OK) { lmmc_sparse_builder_destroy(builder); return st; }

        if (i > 0) {
            /* First row */
            st = lmmc_sparse_builder_add(builder, 0, i, 1.0);
            if (st != LMMC_STATUS_OK) { lmmc_sparse_builder_destroy(builder); return st; }
            /* First column */
            st = lmmc_sparse_builder_add(builder, i, 0, 1.0);
            if (st != LMMC_STATUS_OK) { lmmc_sparse_builder_destroy(builder); return st; }
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

/* --------------------------------------------------------------------------
 * Helper: Build a dense SPD matrix stored in sparse format.
 *
 * A[i][j] = min(i,j) + 1 + (i==j)*n
 * This is diagonally dominant and SPD, and fully dense.
 * -------------------------------------------------------------------------- */
static lmmc_status_t build_dense_spd_as_sparse(size_t n, lmmc_sparse_mat_t* out)
{
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(n, n, n * n, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t val = (lmmc_real_t)((i < j ? i : j) + 1);
            if (i == j) {
                val += (lmmc_real_t)n;
            }
            st = lmmc_sparse_builder_add(builder, i, j, val);
            if (st != LMMC_STATUS_OK) { lmmc_sparse_builder_destroy(builder); return st; }
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

/* --------------------------------------------------------------------------
 * Helper: Build a tridiagonal diagonally-dominant matrix for LU testing.
 * -------------------------------------------------------------------------- */
static lmmc_status_t build_tridiagonal(size_t n, lmmc_sparse_mat_t* out)
{
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {
        st = lmmc_sparse_builder_add(builder, i, i, 4.0);
        if (st != LMMC_STATUS_OK) { lmmc_sparse_builder_destroy(builder); return st; }
        if (i > 0) {
            st = lmmc_sparse_builder_add(builder, i, i - 1, 1.0);
            if (st != LMMC_STATUS_OK) { lmmc_sparse_builder_destroy(builder); return st; }
        }
        if (i < n - 1) {
            st = lmmc_sparse_builder_add(builder, i, i + 1, 1.0);
            if (st != LMMC_STATUS_OK) { lmmc_sparse_builder_destroy(builder); return st; }
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

/* --------------------------------------------------------------------------
 * Helper: Compute residual norm ||Ax - b||_2 for sparse A.
 * -------------------------------------------------------------------------- */
static lmmc_real_t compute_residual(const lmmc_sparse_mat_t* A,
                                    const lmmc_vec_t* x,
                                    const lmmc_vec_t* b)
{
    lmmc_vec_t Ax = {0};
    lmmc_real_t norm = 0.0;

    if (lmmc_vec_create(b->size, &Ax) != LMMC_STATUS_OK) return 1e30;
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

/* ==========================================================================
 * TEST: Sparse LU factorization with tridiagonal matrix (exercises capacity)
 *
 * The tridiagonal matrix has ~3n-2 nonzeros. The LU factors will have
 * entries that grow beyond the initial capacity, exercising reallocation.
 * ========================================================================== */
static void test_lu_tridiagonal_fillin(void)
{
    printf("Test: Sparse LU factorization with tridiagonal (n=20)\n");

    const size_t n = 20;
    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_lu_t* lu = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_tridiagonal(n, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build tridiagonal matrix");
    TEST_ASSERT(A.nnz < n * n, "matrix is sparse");

    st = lmmc_sparse_lu_symbolic(&A, &lu);
    TEST_ASSERT(st == LMMC_STATUS_OK, "LU symbolic succeeds");

    st = lmmc_sparse_lu_numeric(&A, lu);
    TEST_ASSERT(st == LMMC_STATUS_OK, "LU numeric succeeds (capacity tracking works)");

    /* Verify solve doesn't crash on the reallocated buffers */
    st = lmmc_vec_create(n, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b");
    st = lmmc_vec_create(n, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x");

    for (size_t i = 0; i < n; i++) b.data[i] = 1.0;

    st = lmmc_sparse_lu_solve(lu, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "LU solve completes without error");

    lmmc_sparse_lu_destroy(lu);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}

/* ==========================================================================
 * TEST: Sparse LU with larger matrix (n=100) to stress reallocation
 * ========================================================================== */
static void test_lu_large_tridiagonal(void)
{
    printf("Test: Sparse LU with tridiagonal (n=100, stress reallocation)\n");

    const size_t n = 100;
    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_lu_t* lu = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_tridiagonal(n, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build n=100 tridiagonal");

    st = lmmc_sparse_lu_symbolic(&A, &lu);
    TEST_ASSERT(st == LMMC_STATUS_OK, "LU symbolic");

    st = lmmc_sparse_lu_numeric(&A, lu);
    TEST_ASSERT(st == LMMC_STATUS_OK, "LU numeric succeeds (reallocation exercised)");

    /* Verify solve doesn't crash */
    st = lmmc_vec_create(n, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b");
    st = lmmc_vec_create(n, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x");

    for (size_t i = 0; i < n; i++) b.data[i] = 1.0;

    st = lmmc_sparse_lu_solve(lu, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "LU solve completes without error");

    lmmc_sparse_lu_destroy(lu);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}

/* ==========================================================================
 * TEST: Sparse Cholesky with dense SPD stored as sparse (maximum fill-in)
 *
 * The dense SPD matrix (n=15) has n^2 = 225 entries. The Cholesky factor
 * will be fully lower-triangular (n*(n+1)/2 = 120 entries), which exceeds
 * the initial capacity of max(nnz_A, 64) = 225, but the factor growth
 * during computation exercises the reallocation path.
 * ========================================================================== */
static void test_cholesky_dense_fillin(void)
{
    printf("Test: Sparse Cholesky with dense SPD matrix (n=15, max fill-in)\n");

    const size_t n = 15;
    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_dense_spd_as_sparse(n, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build dense SPD as sparse");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "Cholesky symbolic");

    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "Cholesky numeric succeeds (reallocation exercised)");

    /* Solve with known solution */
    st = lmmc_vec_create(n, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b");
    st = lmmc_vec_create(n, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x");

    {
        lmmc_vec_t x_exact = {0};
        st = lmmc_vec_create(n, &x_exact);
        TEST_ASSERT(st == LMMC_STATUS_OK, "create x_exact");
        for (size_t i = 0; i < n; i++) {
            x_exact.data[i] = (lmmc_real_t)(i + 1);
        }
        st = lmmc_sparse_mat_vec_mul(&A, &x_exact, &b);
        TEST_ASSERT(st == LMMC_STATUS_OK, "compute b = A * x_exact");
        lmmc_vec_destroy(&x_exact);
    }

    st = lmmc_sparse_chol_solve(chol, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "Cholesky solve succeeds");

    /* Residual check: ||Ax - b||_2 should be small */
    {
        lmmc_real_t residual = compute_residual(&A, &x, &b);
        lmmc_real_t b_norm = 0.0;
        for (size_t i = 0; i < n; i++) b_norm += b.data[i] * b.data[i];
        b_norm = sqrt(b_norm);

        TEST_ASSERT(residual <= 1e-8 * (1.0 + b_norm),
                    "Cholesky solve residual within tolerance");
    }

    lmmc_sparse_chol_destroy(chol);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}

/* ==========================================================================
 * TEST: Sparse Cholesky with arrowhead SPD (moderate fill-in)
 *
 * The arrowhead matrix has ~3n-2 nonzeros but produces fill-in in the
 * Cholesky factor due to the dense first row/column.
 * ========================================================================== */
static void test_cholesky_arrowhead_fillin(void)
{
    printf("Test: Sparse Cholesky with arrowhead SPD (n=30)\n");

    const size_t n = 30;
    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_arrowhead_spd(n, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build arrowhead SPD");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "Cholesky symbolic");

    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "Cholesky numeric succeeds");

    /* Solve with known solution x = [1, 1, ..., 1] */
    st = lmmc_vec_create(n, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b");
    st = lmmc_vec_create(n, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x");

    {
        lmmc_vec_t x_exact = {0};
        st = lmmc_vec_create(n, &x_exact);
        TEST_ASSERT(st == LMMC_STATUS_OK, "create x_exact");
        for (size_t i = 0; i < n; i++) {
            x_exact.data[i] = 1.0;
        }
        st = lmmc_sparse_mat_vec_mul(&A, &x_exact, &b);
        TEST_ASSERT(st == LMMC_STATUS_OK, "compute b");
        lmmc_vec_destroy(&x_exact);
    }

    st = lmmc_sparse_chol_solve(chol, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "solve succeeds");

    /* Check solution: all entries should be 1.0 */
    {
        lmmc_real_t max_err = 0.0;
        for (size_t i = 0; i < n; i++) {
            lmmc_real_t err = fabs(x.data[i] - 1.0);
            if (err > max_err) max_err = err;
        }
        TEST_ASSERT(max_err < 1e-8, "Cholesky arrowhead solution correct");
    }

    lmmc_sparse_chol_destroy(chol);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}

static void test_destroy_after_failed_cholesky(void)
{
    printf("Test: Destroy after failed Cholesky (non-SPD matrix)\n");

    /* Build a non-SPD matrix */
    lmmc_real_t data[] = {
        1.0, 2.0, 0.0,
        2.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };

    lmmc_sparse_builder_t* builder = NULL;
    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(3, 3, 9, &builder);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create builder");

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            if (data[i * 3 + j] != 0.0) {
                st = lmmc_sparse_builder_add(builder, i, j, data[i * 3 + j]);
                TEST_ASSERT(st == LMMC_STATUS_OK, "add entry");
            }
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build matrix");
    lmmc_sparse_builder_destroy(builder);

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "symbolic succeeds");

    /* Numeric should fail because matrix is not SPD */
    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_NOT_POSITIVE_DEFINITE,
                "numeric returns NOT_POSITIVE_DEFINITE");

    /* Destroy should not crash even after failed factorization */
    lmmc_sparse_chol_destroy(chol);
    TEST_ASSERT(1, "destroy after failed Cholesky does not crash");

    lmmc_sparse_destroy(&A);
}

static void test_destroy_after_failed_lu(void)
{
    printf("Test: Destroy after failed LU (singular matrix)\n");

    lmmc_sparse_builder_t* builder = NULL;
    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_lu_t* lu = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(3, 3, 9, &builder);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create builder");

    /* Matrix: [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
     * Column 1 is all zeros -> singular, detected during factorization. */
    lmmc_sparse_builder_add(builder, 0, 0, 1.0);
    lmmc_sparse_builder_add(builder, 2, 2, 1.0);

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build singular matrix");
    lmmc_sparse_builder_destroy(builder);

    st = lmmc_sparse_lu_symbolic(&A, &lu);
    TEST_ASSERT(st == LMMC_STATUS_OK, "LU symbolic succeeds");

    /* Numeric should fail because matrix is singular */
    st = lmmc_sparse_lu_numeric(&A, lu);
    TEST_ASSERT(st == LMMC_STATUS_SINGULAR_MATRIX,
                "LU numeric returns SINGULAR_MATRIX");

    /* Destroy should not crash even after failed factorization */
    lmmc_sparse_lu_destroy(lu);
    TEST_ASSERT(1, "destroy after failed LU does not crash");

    lmmc_sparse_destroy(&A);
}

/* ==========================================================================
 * TEST: Destroy NULL contexts (safety check)
 * ========================================================================== */
static void test_destroy_null(void)
{
    printf("Test: Destroy NULL contexts\n");

    lmmc_sparse_lu_destroy(NULL);
    TEST_ASSERT(1, "LU destroy NULL does not crash");

    lmmc_sparse_chol_destroy(NULL);
    TEST_ASSERT(1, "Cholesky destroy NULL does not crash");
}

/* ==========================================================================
 * TEST: Successful factorization followed by destroy (basic lifecycle)
 * ========================================================================== */
static void test_successful_factorize_then_destroy(void)
{
    printf("Test: Successful factorization then destroy (lifecycle)\n");

    const size_t n = 10;
    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_lu_t* lu = NULL;
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_status_t st;

    /* Test LU lifecycle */
    st = build_tridiagonal(n, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build tridiagonal matrix");

    st = lmmc_sparse_lu_symbolic(&A, &lu);
    TEST_ASSERT(st == LMMC_STATUS_OK, "LU symbolic");

    st = lmmc_sparse_lu_numeric(&A, lu);
    TEST_ASSERT(st == LMMC_STATUS_OK, "LU numeric");

    lmmc_sparse_lu_destroy(lu);
    TEST_ASSERT(1, "LU destroy after success does not crash");
    lmmc_sparse_destroy(&A);

    /* Test Cholesky lifecycle */
    st = build_arrowhead_spd(n, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build SPD matrix");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "Cholesky symbolic");

    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "Cholesky numeric");

    lmmc_sparse_chol_destroy(chol);
    TEST_ASSERT(1, "Cholesky destroy after success does not crash");
    lmmc_sparse_destroy(&A);
}

/* ========================================================================== */

int main(void)
{
    printf("=== Sparse Buffer Safety Tests ===\n\n");

    test_lu_tridiagonal_fillin();
    test_lu_large_tridiagonal();
    test_cholesky_dense_fillin();
    test_cholesky_arrowhead_fillin();
    test_destroy_after_failed_cholesky();
    test_destroy_after_failed_lu();
    test_destroy_null();
    test_successful_factorize_then_destroy();

    printf("\n=== Results: %d/%d tests passed ===\n", pass_count, test_count);

    return (pass_count == test_count) ? 0 : 1;
}
