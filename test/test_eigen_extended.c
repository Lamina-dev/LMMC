/**
 * @file test_eigen_extended.c
 * @brief Extended tests for eigenvalue decomposition, SVD, pseudo-inverse, and condition number.
 *
 * Tests:
 * 1. Diagonal matrix eigenvalues = diagonal elements
 * 2. A*V = V*D verification (2x2, 3x3, 5x5)
 * 3. Eigenvector orthogonality V^T*V = I
 * 4. Repeated eigenvalue handling
 * 5. Non-symmetric matrix eigenvalues
 * 6. SVD: U*Sigma*V^T = A
 * 7. Pseudo-inverse Moore-Penrose condition (A*pinv(A)*A = A)
 * 8. Condition number verification
 * 9. 1x1 matrix boundary
 * 10. NULL/non-square error handling
 *
 * Validates: Requirements 6.1-6.12
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"
#include "test_common.h"

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)

#define TOL       1e-10
#define TOL_LOOSE 1e-8
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])

/* ========================================================================
 * Test 1: Diagonal matrix eigenvalues = diagonal elements (Req 6.1)
 * ======================================================================== */
static int test_diagonal_eigenvalues(void)
{
    /* 4x4 diagonal matrix with known eigenvalues */
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    lmmc_mat_create(4, 4, &mat);
    lmmc_mat_fill(&mat, 0.0);
    mat.data[0]  = 5.0;
    mat.data[5]  = 2.0;
    mat.data[10] = 8.0;
    mat.data[15] = 1.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "diagonal eigen should succeed, got %d", (int)s);

    /* Eigenvalues should be sorted ascending: 1, 2, 5, 8 */
    double expected[] = {1.0, 2.0, 5.0, 8.0};
    for (int i = 0; i < 4; i++) {
        CHECK(fabs(result.eigenvalues.data[i] - expected[i]) < TOL,
              "eigenvalue[%d]: expected %f, got %f", i, expected[i], result.eigenvalues.data[i]);
    }

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test 2: A*V = V*D verification for multiple sizes (Req 6.2)
 * ======================================================================== */
static int test_av_equals_vd_2x2(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    size_t n = 2;
    lmmc_mat_create(n, n, &mat);
    /* A = [[4, 1], [1, 3]] */
    mat.data[0] = 4.0; mat.data[1] = 1.0;
    mat.data[2] = 1.0; mat.data[3] = 3.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "2x2 eigen should succeed, got %d", (int)s);

    /* Verify A*V = V*D: for each column k of V, A*v_k = lambda_k * v_k */
    for (size_t k = 0; k < n; k++) {
        for (size_t i = 0; i < n; i++) {
            double av_ik = 0.0;
            for (size_t j = 0; j < n; j++) {
                av_ik += mat.data[i * n + j] * result.eigenvectors.data[j * n + k];
            }
            double vd_ik = result.eigenvalues.data[k] * result.eigenvectors.data[i * n + k];
            CHECK(fabs(av_ik - vd_ik) < TOL,
                  "A*V != V*D at [%zu][%zu]: %f vs %f", i, k, av_ik, vd_ik);
        }
    }

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_av_equals_vd_3x3(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    size_t n = 3;
    lmmc_mat_create(n, n, &mat);
    /* A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] tridiagonal */
    lmmc_real_t data[] = {2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0};
    memcpy(mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "3x3 eigen should succeed, got %d", (int)s);

    for (size_t k = 0; k < n; k++) {
        for (size_t i = 0; i < n; i++) {
            double av_ik = 0.0;
            for (size_t j = 0; j < n; j++) {
                av_ik += data[i * n + j] * result.eigenvectors.data[j * n + k];
            }
            double vd_ik = result.eigenvalues.data[k] * result.eigenvectors.data[i * n + k];
            CHECK(fabs(av_ik - vd_ik) < TOL,
                  "3x3 A*V != V*D at [%zu][%zu]: %f vs %f", i, k, av_ik, vd_ik);
        }
    }

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_av_equals_vd_5x5(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    size_t n = 5;
    lmmc_mat_create(n, n, &mat);
    /* 5x5 symmetric positive definite matrix */
    lmmc_real_t data[] = {
        6.0, 2.0, 1.0, 0.0, 0.0,
        2.0, 5.0, 2.0, 1.0, 0.0,
        1.0, 2.0, 6.0, 2.0, 1.0,
        0.0, 1.0, 2.0, 5.0, 2.0,
        0.0, 0.0, 1.0, 2.0, 6.0
    };
    memcpy(mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "5x5 eigen should succeed, got %d", (int)s);

    for (size_t k = 0; k < n; k++) {
        for (size_t i = 0; i < n; i++) {
            double av_ik = 0.0;
            for (size_t j = 0; j < n; j++) {
                av_ik += data[i * n + j] * result.eigenvectors.data[j * n + k];
            }
            double vd_ik = result.eigenvalues.data[k] * result.eigenvectors.data[i * n + k];
            CHECK(fabs(av_ik - vd_ik) < TOL,
                  "5x5 A*V != V*D at [%zu][%zu]: %f vs %f", i, k, av_ik, vd_ik);
        }
    }

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test 3: Eigenvector orthogonality V^T*V = I (Req 6.3)
 * ======================================================================== */
static int test_eigenvector_orthogonality(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    size_t n = 4;
    lmmc_mat_create(n, n, &mat);
    /* Symmetric matrix */
    lmmc_real_t data[] = {
        5.0, 1.0, 2.0, 0.0,
        1.0, 4.0, 1.0, 1.0,
        2.0, 1.0, 6.0, 2.0,
        0.0, 1.0, 2.0, 3.0
    };
    memcpy(mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "4x4 eigen should succeed, got %d", (int)s);

    /* Check V^T * V = I */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double dot = 0.0;
            for (size_t k = 0; k < n; k++) {
                dot += result.eigenvectors.data[k * n + i] *
                       result.eigenvectors.data[k * n + j];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(fabs(dot - expected) < TOL,
                  "V^T*V[%zu][%zu] should be %f, got %f", i, j, expected, dot);
        }
    }

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test 4: Repeated eigenvalue handling (Req 6.4)
 * ======================================================================== */
static int test_repeated_eigenvalues(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    size_t n = 3;
    lmmc_mat_create(n, n, &mat);
    /* A = [[2, 0, 0], [0, 2, 0], [0, 0, 5]] => eigenvalues: 2, 2, 5 */
    lmmc_mat_fill(&mat, 0.0);
    mat.data[0] = 2.0;
    mat.data[4] = 2.0;
    mat.data[8] = 5.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "repeated eigen should succeed, got %d", (int)s);

    CHECK(fabs(result.eigenvalues.data[0] - 2.0) < TOL,
          "eigenvalue[0] should be 2.0, got %f", result.eigenvalues.data[0]);
    CHECK(fabs(result.eigenvalues.data[1] - 2.0) < TOL,
          "eigenvalue[1] should be 2.0, got %f", result.eigenvalues.data[1]);
    CHECK(fabs(result.eigenvalues.data[2] - 5.0) < TOL,
          "eigenvalue[2] should be 5.0, got %f", result.eigenvalues.data[2]);

    /* Even with repeated eigenvalues, V^T*V should still be I */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double dot = 0.0;
            for (size_t k = 0; k < n; k++) {
                dot += result.eigenvectors.data[k * n + i] *
                       result.eigenvectors.data[k * n + j];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            CHECK(fabs(dot - expected) < TOL,
                  "Repeated: V^T*V[%zu][%zu] should be %f, got %f", i, j, expected, dot);
        }
    }

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test 5: Non-symmetric matrix eigenvalues (Req 6.5)
 * ======================================================================== */
static int test_nonsymmetric_eigenvalues(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_gen_result_t result;
    size_t n = 3;
    lmmc_mat_create(n, n, &mat);
    /* A = [[0, 1, 0], [0, 0, 1], [1, 0, 0]] - companion matrix for x^3 - 1 = 0
     * Eigenvalues: 1, -0.5+i*sqrt(3)/2, -0.5-i*sqrt(3)/2 */
    lmmc_real_t data[] = {
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0
    };
    memcpy(mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_eigen_general(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "general eigen should succeed, got %d", (int)s);

    /* Verify we got 3 eigenvalues */
    CHECK(result.real_parts.size == 3, "should have 3 eigenvalues");

    /* Find the real eigenvalue (should be 1.0) */
    int found_real = 0;
    for (size_t i = 0; i < n; i++) {
        if (fabs(result.imag_parts.data[i]) < TOL) {
            CHECK(fabs(result.real_parts.data[i] - 1.0) < TOL,
                  "real eigenvalue should be 1.0, got %f", result.real_parts.data[i]);
            found_real = 1;
        }
    }
    CHECK(found_real, "should find real eigenvalue 1.0");

    /* Verify complex eigenvalues come in conjugate pairs */
    int found_pos_imag = 0, found_neg_imag = 0;
    for (size_t i = 0; i < n; i++) {
        if (result.imag_parts.data[i] > TOL) found_pos_imag = 1;
        if (result.imag_parts.data[i] < -TOL) found_neg_imag = 1;
    }
    CHECK(found_pos_imag && found_neg_imag, "complex eigenvalues should come in conjugate pairs");

    lmmc_eigen_gen_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test 6: SVD: U*Sigma*V^T = A (Req 6.6)
 * ======================================================================== */
static int test_svd_reconstruction(void)
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
    memcpy(mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_svd(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "SVD should succeed, got %d", (int)s);

    /* Reconstruct: A_recon[i][j] = sum_k U[i][k] * sigma[k] * Vt[k][j] */
    size_t p = (m < n) ? m : n;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < p; k++) {
                sum += MAT_ELEM(&result.U, i, k) *
                       result.sigma.data[k] *
                       MAT_ELEM(&result.Vt, k, j);
            }
            CHECK(fabs(sum - data[i * n + j]) < TOL_LOOSE,
                  "SVD recon A[%zu][%zu]: expected %f, got %f", i, j, data[i * n + j], sum);
        }
    }

    lmmc_svd_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_svd_reconstruction_4x3(void)
{
    lmmc_mat_t mat;
    lmmc_svd_result_t result;
    size_t m = 4, n = 3;
    lmmc_mat_create(m, n, &mat);
    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 13.0
    };
    memcpy(mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_svd(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "4x3 SVD should succeed, got %d", (int)s);

    size_t p = (m < n) ? m : n;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < p; k++) {
                sum += MAT_ELEM(&result.U, i, k) *
                       result.sigma.data[k] *
                       MAT_ELEM(&result.Vt, k, j);
            }
            CHECK(fabs(sum - data[i * n + j]) < TOL_LOOSE,
                  "4x3 SVD recon A[%zu][%zu]: expected %f, got %f", i, j, data[i * n + j], sum);
        }
    }

    lmmc_svd_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test 7: Pseudo-inverse Moore-Penrose condition A*pinv(A)*A = A (Req 6.7, 6.8)
 * ======================================================================== */
static int test_pinv_moore_penrose(void)
{
    lmmc_mat_t mat;
    size_t m = 3, n = 3;
    lmmc_mat_create(m, n, &mat);
    /* Full rank 3x3 matrix */
    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        5.0, 6.0, 0.0
    };
    memcpy(mat.data, data, sizeof(data));

    lmmc_mat_t pinv;
    lmmc_mat_create(n, m, &pinv);

    lmmc_status_t s = lmmc_pinv(&mat, 0.0, &pinv);
    CHECK(s == LMMC_STATUS_OK, "pinv should succeed, got %d", (int)s);

    /* Verify A * pinv(A) * A = A */
    /* Step 1: compute T = pinv(A) * A (n x n) */
    double T[9];
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < m; k++) {
                sum += MAT_ELEM(&pinv, i, k) * MAT_ELEM(&mat, k, j);
            }
            T[i * n + j] = sum;
        }
    }
    /* Step 2: compute A * T and compare with A */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += MAT_ELEM(&mat, i, k) * T[k * n + j];
            }
            CHECK(fabs(sum - data[i * n + j]) < TOL_LOOSE,
                  "A*pinv(A)*A[%zu][%zu]: expected %f, got %f", i, j, data[i * n + j], sum);
        }
    }

    lmmc_mat_destroy(&pinv);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_pinv_rank_deficient(void)
{
    /* Rank-deficient matrix: row 3 = row 1 + row 2 */
    lmmc_mat_t mat;
    size_t m = 3, n = 3;
    lmmc_mat_create(m, n, &mat);
    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        5.0, 7.0, 9.0  /* = row1 + row2 */
    };
    memcpy(mat.data, data, sizeof(data));

    lmmc_mat_t pinv;
    lmmc_mat_create(n, m, &pinv);

    lmmc_status_t s = lmmc_pinv(&mat, 1e-10, &pinv);
    CHECK(s == LMMC_STATUS_OK, "pinv rank-deficient should succeed, got %d", (int)s);

    /* Verify Moore-Penrose condition: A * pinv(A) * A = A */
    double T[9];
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < m; k++) {
                sum += MAT_ELEM(&pinv, i, k) * MAT_ELEM(&mat, k, j);
            }
            T[i * n + j] = sum;
        }
    }
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += MAT_ELEM(&mat, i, k) * T[k * n + j];
            }
            CHECK(fabs(sum - data[i * n + j]) < 1e-6,
                  "Rank-def A*pinv(A)*A[%zu][%zu]: expected %f, got %f",
                  i, j, data[i * n + j], sum);
        }
    }

    lmmc_mat_destroy(&pinv);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test 8: Condition number verification (Req 6.9, 6.10)
 * ======================================================================== */
static int test_cond_well_conditioned(void)
{
    /* Identity matrix: cond = 1 */
    lmmc_mat_t mat;
    lmmc_mat_create(3, 3, &mat);
    lmmc_mat_fill(&mat, 0.0);
    mat.data[0] = 1.0;
    mat.data[4] = 1.0;
    mat.data[8] = 1.0;

    lmmc_real_t cond_val;
    lmmc_status_t s = lmmc_cond(&mat, &cond_val);
    CHECK(s == LMMC_STATUS_OK, "cond(I) should succeed, got %d", (int)s);
    CHECK(fabs(cond_val - 1.0) < TOL,
          "cond(I) should be 1.0, got %f", cond_val);

    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_cond_near_singular(void)
{
    /* Near-singular matrix: large condition number */
    lmmc_mat_t mat;
    lmmc_mat_create(2, 2, &mat);
    mat.data[0] = 1.0;    mat.data[1] = 0.0;
    mat.data[2] = 0.0;    mat.data[3] = 1e-12;

    lmmc_real_t cond_val;
    lmmc_status_t s = lmmc_cond(&mat, &cond_val);
    CHECK(s == LMMC_STATUS_OK, "cond near-singular should succeed, got %d", (int)s);
    CHECK(cond_val > 1e10,
          "cond of near-singular should be large, got %f", cond_val);

    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Test 9: 1x1 matrix boundary (Req 6.12)
 * ======================================================================== */
static int test_1x1_boundary(void)
{
    /* 1x1 symmetric eigen */
    {
        lmmc_mat_t mat;
        lmmc_eigen_sym_result_t result;
        lmmc_mat_create(1, 1, &mat);
        mat.data[0] = 7.5;

        lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
        CHECK(s == LMMC_STATUS_OK, "1x1 sym eigen should succeed");
        CHECK(fabs(result.eigenvalues.data[0] - 7.5) < TOL,
              "1x1 eigenvalue should be 7.5, got %f", result.eigenvalues.data[0]);
        CHECK(fabs(result.eigenvectors.data[0] - 1.0) < TOL,
              "1x1 eigenvector should be 1.0, got %f", result.eigenvectors.data[0]);

        lmmc_eigen_sym_result_destroy(&result);
        lmmc_mat_destroy(&mat);
    }

    /* 1x1 general eigen */
    {
        lmmc_mat_t mat;
        lmmc_eigen_gen_result_t result;
        lmmc_mat_create(1, 1, &mat);
        mat.data[0] = -3.0;

        lmmc_status_t s = lmmc_eigen_general(&mat, &result);
        CHECK(s == LMMC_STATUS_OK, "1x1 gen eigen should succeed");
        CHECK(fabs(result.real_parts.data[0] - (-3.0)) < TOL,
              "1x1 gen eigenvalue real should be -3.0, got %f", result.real_parts.data[0]);
        CHECK(fabs(result.imag_parts.data[0]) < TOL,
              "1x1 gen eigenvalue imag should be 0, got %f", result.imag_parts.data[0]);

        lmmc_eigen_gen_result_destroy(&result);
        lmmc_mat_destroy(&mat);
    }

    /* 1x1 SVD */
    {
        lmmc_mat_t mat;
        lmmc_svd_result_t result;
        lmmc_mat_create(1, 1, &mat);
        mat.data[0] = -4.0;

        lmmc_status_t s = lmmc_svd(&mat, &result);
        CHECK(s == LMMC_STATUS_OK, "1x1 SVD should succeed");
        CHECK(fabs(result.sigma.data[0] - 4.0) < TOL,
              "1x1 SVD sigma should be 4.0, got %f", result.sigma.data[0]);

        lmmc_svd_result_destroy(&result);
        lmmc_mat_destroy(&mat);
    }

    /* 1x1 pinv */
    {
        lmmc_mat_t mat, pinv;
        lmmc_mat_create(1, 1, &mat);
        lmmc_mat_create(1, 1, &pinv);
        mat.data[0] = 2.0;

        lmmc_status_t s = lmmc_pinv(&mat, 0.0, &pinv);
        CHECK(s == LMMC_STATUS_OK, "1x1 pinv should succeed");
        CHECK(fabs(pinv.data[0] - 0.5) < TOL,
              "1x1 pinv should be 0.5, got %f", pinv.data[0]);

        lmmc_mat_destroy(&pinv);
        lmmc_mat_destroy(&mat);
    }

    /* 1x1 cond */
    {
        lmmc_mat_t mat;
        lmmc_mat_create(1, 1, &mat);
        mat.data[0] = 5.0;

        lmmc_real_t cond_val;
        lmmc_status_t s = lmmc_cond(&mat, &cond_val);
        CHECK(s == LMMC_STATUS_OK, "1x1 cond should succeed");
        CHECK(fabs(cond_val - 1.0) < TOL,
              "1x1 cond should be 1.0, got %f", cond_val);

        lmmc_mat_destroy(&mat);
    }

    return 0;
}

/* ========================================================================
 * Test 10: NULL/non-square error handling (Req 6.11)
 * ======================================================================== */
static int test_error_handling(void)
{
    lmmc_eigen_sym_result_t sym_result;
    lmmc_eigen_gen_result_t gen_result;
    lmmc_svd_result_t svd_result;
    lmmc_status_t s;

    /* NULL input for symmetric */
    s = lmmc_eigen_symmetric(NULL, &sym_result);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "eigen_symmetric(NULL) should return INVALID_ARGUMENT, got %d", (int)s);

    /* NULL output for symmetric */
    {
        lmmc_mat_t mat;
        lmmc_mat_create(2, 2, &mat);
        s = lmmc_eigen_symmetric(&mat, NULL);
        CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
              "eigen_symmetric(mat, NULL) should return INVALID_ARGUMENT, got %d", (int)s);
        lmmc_mat_destroy(&mat);
    }

    /* Non-square for symmetric */
    {
        lmmc_mat_t mat;
        lmmc_mat_create(2, 3, &mat);
        s = lmmc_eigen_symmetric(&mat, &sym_result);
        CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
              "eigen_symmetric(2x3) should return INVALID_ARGUMENT, got %d", (int)s);
        lmmc_mat_destroy(&mat);
    }

    /* NULL input for general */
    s = lmmc_eigen_general(NULL, &gen_result);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "eigen_general(NULL) should return INVALID_ARGUMENT, got %d", (int)s);

    /* Non-square for general */
    {
        lmmc_mat_t mat;
        lmmc_mat_create(3, 2, &mat);
        s = lmmc_eigen_general(&mat, &gen_result);
        CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
              "eigen_general(3x2) should return INVALID_ARGUMENT, got %d", (int)s);
        lmmc_mat_destroy(&mat);
    }

    /* NULL input for SVD */
    s = lmmc_svd(NULL, &svd_result);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "svd(NULL) should return INVALID_ARGUMENT, got %d", (int)s);

    /* NULL input for pinv */
    {
        lmmc_mat_t pinv;
        lmmc_mat_create(2, 2, &pinv);
        s = lmmc_pinv(NULL, 0.0, &pinv);
        CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
              "pinv(NULL) should return INVALID_ARGUMENT, got %d", (int)s);
        lmmc_mat_destroy(&pinv);
    }

    /* NULL input for cond */
    {
        lmmc_real_t cond_val;
        s = lmmc_cond(NULL, &cond_val);
        CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
              "cond(NULL) should return INVALID_ARGUMENT, got %d", (int)s);
    }

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
        {"diagonal_eigenvalues",       test_diagonal_eigenvalues},
        {"av_equals_vd_2x2",          test_av_equals_vd_2x2},
        {"av_equals_vd_3x3",          test_av_equals_vd_3x3},
        {"av_equals_vd_5x5",          test_av_equals_vd_5x5},
        {"eigenvector_orthogonality",  test_eigenvector_orthogonality},
        {"repeated_eigenvalues",       test_repeated_eigenvalues},
        {"nonsymmetric_eigenvalues",   test_nonsymmetric_eigenvalues},
        {"svd_reconstruction",         test_svd_reconstruction},
        {"svd_reconstruction_4x3",     test_svd_reconstruction_4x3},
        {"pinv_moore_penrose",         test_pinv_moore_penrose},
        {"pinv_rank_deficient",        test_pinv_rank_deficient},
        {"cond_well_conditioned",      test_cond_well_conditioned},
        {"cond_near_singular",         test_cond_near_singular},
        {"1x1_boundary",               test_1x1_boundary},
        {"error_handling",             test_error_handling},
    };

    size_t n_tests = sizeof(tests) / sizeof(tests[0]);
    int n_passed = 0;
    int n_failed = 0;

    printf("=== Eigen Extended Tests ===\n\n");

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

    if (n_failed > 0) {
        printf("eigen_extended test failed\n");
    }
    return (n_failed > 0) ? 1 : 0;
}
