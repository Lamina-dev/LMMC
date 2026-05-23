/**
 * @file test_svd.c
 * @brief Unit tests for SVD decomposition, pseudo-inverse, and condition number.
 *
 * Tests:
 * 1. Input validation (NULL, zero dimensions)
 * 2. 1x1 matrix SVD
 * 3. 2x2 diagonal matrix SVD
 * 4. 3x3 matrix SVD reconstruction (A = U * diag(sigma) * Vt)
 * 5. Orthogonality of U and Vt
 * 6. Singular values in descending order
 * 7. Wide matrix (m < n) SVD
 * 8. Pseudo-inverse (A * A+ * A = A)
 * 9. Condition number
 * 10. lmmc_svd_result_destroy safety
 *
 * Validates: Requirements 7.1-7.7
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)

#define TOL 1e-10
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])

/* ========================================================================
 * Test: NULL input returns INVALID_ARGUMENT
 * ======================================================================== */
static int test_null_input(void)
{
    lmmc_svd_result_t result;
    lmmc_status_t s;

    s = lmmc_svd(NULL, &result);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "svd(NULL, ...) should return INVALID_ARGUMENT, got %d", (int)s);

    lmmc_mat_t mat;
    lmmc_mat_create(3, 2, &mat);
    s = lmmc_svd(&mat, NULL);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "svd(..., NULL) should return INVALID_ARGUMENT, got %d", (int)s);
    lmmc_mat_destroy(&mat);

    return 0;
}

/* ========================================================================
 * Test: 1x1 matrix SVD
 * ======================================================================== */
static int test_1x1(void)
{
    lmmc_mat_t mat;
    lmmc_svd_result_t result;
    lmmc_mat_create(1, 1, &mat);
    mat.data[0] = 5.0;

    lmmc_status_t s = lmmc_svd(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "1x1 SVD should succeed, got %d", (int)s);
    CHECK(fabs(result.sigma.data[0] - 5.0) < TOL,
          "1x1 sigma should be 5.0, got %f", result.sigma.data[0]);

    lmmc_svd_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test: 2x2 diagonal matrix SVD
 * ======================================================================== */
static int test_2x2_diagonal(void)
{
    lmmc_mat_t mat;
    lmmc_svd_result_t result;
    lmmc_mat_create(2, 2, &mat);
    mat.data[0] = 3.0; mat.data[1] = 0.0;
    mat.data[2] = 0.0; mat.data[3] = 7.0;

    lmmc_status_t s = lmmc_svd(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "2x2 diagonal SVD should succeed, got %d", (int)s);

    /* Singular values should be 7 and 3, descending */
    CHECK(fabs(result.sigma.data[0] - 7.0) < TOL,
          "sigma[0] should be 7.0, got %f", result.sigma.data[0]);
    CHECK(fabs(result.sigma.data[1] - 3.0) < TOL,
          "sigma[1] should be 3.0, got %f", result.sigma.data[1]);

    lmmc_svd_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test: 3x3 matrix SVD reconstruction (A = U * diag(sigma) * Vt)
 * ======================================================================== */
static int test_3x3_reconstruction(void)
{
    lmmc_mat_t mat;
    lmmc_svd_result_t result;
    size_t m = 3, n = 3;
    lmmc_mat_create(m, n, &mat);

    /* A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]] */
    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0
    };
    for (size_t i = 0; i < m * n; i++) mat.data[i] = data[i];

    lmmc_status_t s = lmmc_svd(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "3x3 SVD should succeed, got %d", (int)s);

    /* Reconstruct: A_recon[i][j] = sum_k U[i][k] * sigma[k] * Vt[k][j] */
    size_t p = (m < n) ? m : n;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t sum = 0.0;
            for (size_t k = 0; k < p; k++) {
                sum += MAT_ELEM(&result.U, i, k) *
                       result.sigma.data[k] *
                       MAT_ELEM(&result.Vt, k, j);
            }
            CHECK(fabs(sum - data[i * n + j]) < 1e-8,
                  "Reconstruction A[%zu][%zu]: expected %f, got %f",
                  i, j, data[i * n + j], sum);
        }
    }

    lmmc_svd_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test: Orthogonality of U (U^T * U = I) and Vt (Vt * Vt^T = I)
 * ======================================================================== */
static int test_orthogonality(void)
{
    lmmc_mat_t mat;
    lmmc_svd_result_t result;
    size_t m = 4, n = 3;
    lmmc_mat_create(m, n, &mat);

    /* A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 13]] */
    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 13.0
    };
    for (size_t i = 0; i < m * n; i++) mat.data[i] = data[i];

    lmmc_status_t s = lmmc_svd(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "4x3 SVD should succeed, got %d", (int)s);

    /* Check U^T * U = I (m x m) */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            lmmc_real_t dot = 0.0;
            for (size_t k = 0; k < m; k++) {
                dot += MAT_ELEM(&result.U, k, i) * MAT_ELEM(&result.U, k, j);
            }
            lmmc_real_t expected = (i == j) ? 1.0 : 0.0;
            CHECK(fabs(dot - expected) < TOL,
                  "U^T*U[%zu][%zu] should be %f, got %f", i, j, expected, dot);
        }
    }

    /* Check Vt * Vt^T = I (n x n) */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t dot = 0.0;
            for (size_t k = 0; k < n; k++) {
                dot += MAT_ELEM(&result.Vt, i, k) * MAT_ELEM(&result.Vt, j, k);
            }
            lmmc_real_t expected = (i == j) ? 1.0 : 0.0;
            CHECK(fabs(dot - expected) < TOL,
                  "Vt*Vt^T[%zu][%zu] should be %f, got %f", i, j, expected, dot);
        }
    }

    lmmc_svd_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test: Singular values in descending order
 * ======================================================================== */
static int test_descending_order(void)
{
    lmmc_mat_t mat;
    lmmc_svd_result_t result;
    size_t m = 4, n = 3;
    lmmc_mat_create(m, n, &mat);

    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
        2.0, 1.0, 4.0
    };
    for (size_t i = 0; i < m * n; i++) mat.data[i] = data[i];

    lmmc_status_t s = lmmc_svd(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "4x3 SVD should succeed, got %d", (int)s);

    size_t p = (m < n) ? m : n;
    for (size_t i = 0; i + 1 < p; i++) {
        CHECK(result.sigma.data[i] >= result.sigma.data[i + 1],
              "sigma not descending: [%zu]=%f < [%zu]=%f",
              i, result.sigma.data[i], i + 1, result.sigma.data[i + 1]);
    }

    /* All singular values should be non-negative */
    for (size_t i = 0; i < p; i++) {
        CHECK(result.sigma.data[i] >= 0.0,
              "sigma[%zu] should be >= 0, got %f", i, result.sigma.data[i]);
    }

    lmmc_svd_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test: Wide matrix (m < n) SVD
 * ======================================================================== */
static int test_wide_matrix(void)
{
    lmmc_mat_t mat;
    lmmc_svd_result_t result;
    size_t m = 2, n = 4;
    lmmc_mat_create(m, n, &mat);

    /* A = [[1, 2, 3, 4], [5, 6, 7, 8]] */
    lmmc_real_t data[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    };
    for (size_t i = 0; i < m * n; i++) mat.data[i] = data[i];

    lmmc_status_t s = lmmc_svd(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "2x4 SVD should succeed, got %d", (int)s);

    /* Check dimensions */
    CHECK(result.U.rows == m && result.U.cols == m,
          "U should be %zux%zu, got %zux%zu", m, m, result.U.rows, result.U.cols);
    CHECK(result.sigma.size == m,
          "sigma size should be %zu, got %zu", m, result.sigma.size);
    CHECK(result.Vt.rows == n && result.Vt.cols == n,
          "Vt should be %zux%zu, got %zux%zu", n, n, result.Vt.rows, result.Vt.cols);

    /* Reconstruct: A_recon[i][j] = sum_k U[i][k] * sigma[k] * Vt[k][j] */
    size_t p = m; /* min(m,n) = m */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t sum = 0.0;
            for (size_t k = 0; k < p; k++) {
                sum += MAT_ELEM(&result.U, i, k) *
                       result.sigma.data[k] *
                       MAT_ELEM(&result.Vt, k, j);
            }
            CHECK(fabs(sum - data[i * n + j]) < 1e-8,
                  "Wide reconstruction A[%zu][%zu]: expected %f, got %f",
                  i, j, data[i * n + j], sum);
        }
    }

    lmmc_svd_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test: Pseudo-inverse (A * A+ * A = A)
 * ======================================================================== */
static int test_pinv(void)
{
    lmmc_mat_t mat;
    size_t m = 3, n = 2;
    lmmc_mat_create(m, n, &mat);

    /* A = [[1, 2], [3, 4], [5, 6]] */
    lmmc_real_t data[] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };
    for (size_t i = 0; i < m * n; i++) mat.data[i] = data[i];

    /* Allocate output: pinv is n x m */
    lmmc_mat_t pinv;
    lmmc_mat_create(n, m, &pinv);

    lmmc_status_t s = lmmc_pinv(&mat, 0.0, &pinv);
    CHECK(s == LMMC_STATUS_OK, "pinv should succeed, got %d", (int)s);

    /* Verify Moore-Penrose condition: A * A+ * A = A */
    /* First compute T = A+ * A (n x n) */
    lmmc_mat_t temp_nn;
    lmmc_mat_create(n, n, &temp_nn);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t sum = 0.0;
            for (size_t k = 0; k < m; k++) {
                sum += MAT_ELEM(&pinv, i, k) * MAT_ELEM(&mat, k, j);
            }
            MAT_ELEM(&temp_nn, i, j) = sum;
        }
    }
    /* Then compute A * T (m x n) and compare with A */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += MAT_ELEM(&mat, i, k) * MAT_ELEM(&temp_nn, k, j);
            }
            CHECK(fabs(sum - data[i * n + j]) < 1e-8,
                  "A*A+*A[%zu][%zu]: expected %f, got %f",
                  i, j, data[i * n + j], sum);
        }
    }

    lmmc_mat_destroy(&temp_nn);
    lmmc_mat_destroy(&pinv);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test: Condition number
 * ======================================================================== */
static int test_cond(void)
{
    lmmc_mat_t mat;
    lmmc_mat_create(2, 2, &mat);
    /* A = [[3, 0], [0, 1]] => cond = 3/1 = 3 */
    mat.data[0] = 3.0; mat.data[1] = 0.0;
    mat.data[2] = 0.0; mat.data[3] = 1.0;

    lmmc_real_t cond_val;
    lmmc_status_t s = lmmc_cond(&mat, &cond_val);
    CHECK(s == LMMC_STATUS_OK, "cond should succeed, got %d", (int)s);
    CHECK(fabs(cond_val - 3.0) < TOL,
          "cond should be 3.0, got %f", cond_val);

    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test: Condition number for singular matrix returns INFINITY
 * ======================================================================== */
static int test_cond_singular(void)
{
    lmmc_mat_t mat;
    lmmc_mat_create(2, 2, &mat);
    /* A = [[1, 0], [0, 0]] => singular, cond = inf */
    mat.data[0] = 1.0; mat.data[1] = 0.0;
    mat.data[2] = 0.0; mat.data[3] = 0.0;

    lmmc_real_t cond_val;
    lmmc_status_t s = lmmc_cond(&mat, &cond_val);
    CHECK(s == LMMC_STATUS_OK, "cond of singular should succeed, got %d", (int)s);
    CHECK(isinf(cond_val), "cond of singular should be INFINITY, got %f", cond_val);

    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test: pinv NULL input validation
 * ======================================================================== */
static int test_pinv_null(void)
{
    lmmc_mat_t mat, pinv;
    lmmc_mat_create(2, 2, &mat);
    lmmc_mat_create(2, 2, &pinv);

    lmmc_status_t s = lmmc_pinv(NULL, 0.0, &pinv);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "pinv(NULL, ...) should return INVALID_ARGUMENT, got %d", (int)s);

    s = lmmc_pinv(&mat, 0.0, NULL);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "pinv(..., NULL) should return INVALID_ARGUMENT, got %d", (int)s);

    lmmc_mat_destroy(&pinv);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test: lmmc_svd_result_destroy safety (NULL input)
 * ======================================================================== */
static int test_destroy_null(void)
{
    /* Should not crash */
    lmmc_svd_result_destroy(NULL);
    return 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */
typedef int (*test_func_t)(void);

typedef struct {
    const char* name;
    test_func_t func;
} test_entry_t;

int main(void)
{
    test_entry_t tests[] = {
        {"null_input", test_null_input},
        {"1x1", test_1x1},
        {"2x2_diagonal", test_2x2_diagonal},
        {"3x3_reconstruction", test_3x3_reconstruction},
        {"orthogonality", test_orthogonality},
        {"descending_order", test_descending_order},
        {"wide_matrix", test_wide_matrix},
        {"pinv", test_pinv},
        {"cond", test_cond},
        {"cond_singular", test_cond_singular},
        {"pinv_null", test_pinv_null},
        {"destroy_null", test_destroy_null},
    };

    size_t n_tests = sizeof(tests) / sizeof(tests[0]);
    int n_passed = 0;
    int n_failed = 0;

    printf("=== SVD Decomposition Tests ===\n\n");

    for (size_t i = 0; i < n_tests; i++) {
        printf("[%zu/%zu] %s ... ", i + 1, n_tests, tests[i].name);
        if (tests[i].func() == 0) {
            printf("PASS\n");
            n_passed++;
        } else {
            printf("FAIL\n");
            n_failed++;
        }
    }

    printf("\n=== Results: %d passed, %d failed ===\n", n_passed, n_failed);
    return (n_failed > 0) ? 1 : 0;
}
