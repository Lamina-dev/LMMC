/**
 * @file test_eigen_extended.c
 * 针对 LMMC 中 eigen extended 相关接口的单元测试。
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


static int test_diagonal_eigenvalues(void)
{

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


    double expected[] = {1.0, 2.0, 5.0, 8.0};
    for (int i = 0; i < 4; i++) {
        CHECK(fabs(result.eigenvalues.data[i] - expected[i]) < TOL,
              "eigenvalue[%d]: expected %f, got %f", i, expected[i], result.eigenvalues.data[i]);
    }

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}


static int test_av_equals_vd_2x2(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    size_t n = 2;
    lmmc_mat_create(n, n, &mat);

    mat.data[0] = 4.0; mat.data[1] = 1.0;
    mat.data[2] = 1.0; mat.data[3] = 3.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "2x2 eigen should succeed, got %d", (int)s);


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


static int test_eigenvector_orthogonality(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    size_t n = 4;
    lmmc_mat_create(n, n, &mat);

    lmmc_real_t data[] = {
        5.0, 1.0, 2.0, 0.0,
        1.0, 4.0, 1.0, 1.0,
        2.0, 1.0, 6.0, 2.0,
        0.0, 1.0, 2.0, 3.0
    };
    memcpy(mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "4x4 eigen should succeed, got %d", (int)s);


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


static int test_repeated_eigenvalues(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    size_t n = 3;
    lmmc_mat_create(n, n, &mat);

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


static int test_nonsymmetric_eigenvalues(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_gen_result_t result;
    size_t n = 3;
    lmmc_mat_create(n, n, &mat);

    lmmc_real_t data[] = {
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0
    };
    memcpy(mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_eigen_general(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "general eigen should succeed, got %d", (int)s);


    CHECK(result.real_parts.size == 3, "should have 3 eigenvalues");


    int found_real = 0;
    for (size_t i = 0; i < n; i++) {
        if (fabs(result.imag_parts.data[i]) < TOL) {
            CHECK(fabs(result.real_parts.data[i] - 1.0) < TOL,
                  "real eigenvalue should be 1.0, got %f", result.real_parts.data[i]);
            found_real = 1;
        }
    }
    CHECK(found_real, "should find real eigenvalue 1.0");


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


static int test_svd_reconstruction(void)
{
    lmmc_mat_t mat;
    lmmc_svd_result_t result;
    size_t m = 3, n = 3;
    lmmc_mat_create(m, n, &mat);

    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0
    };
    memcpy(mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_svd(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "SVD should succeed, got %d", (int)s);


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


static int test_pinv_moore_penrose(void)
{
    lmmc_mat_t mat;
    size_t m = 3, n = 3;
    lmmc_mat_create(m, n, &mat);

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

    lmmc_mat_t mat;
    size_t m = 3, n = 3;
    lmmc_mat_create(m, n, &mat);
    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        5.0, 7.0, 9.0
    };
    memcpy(mat.data, data, sizeof(data));

    lmmc_mat_t pinv;
    lmmc_mat_create(n, m, &pinv);

    lmmc_status_t s = lmmc_pinv(&mat, 1e-10, &pinv);
    CHECK(s == LMMC_STATUS_OK, "pinv rank-deficient should succeed, got %d", (int)s);


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


static int test_cond_well_conditioned(void)
{

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


static int test_1x1_boundary(void)
{

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


static int test_error_handling(void)
{
    lmmc_eigen_sym_result_t sym_result;
    lmmc_eigen_gen_result_t gen_result;
    lmmc_svd_result_t svd_result;
    lmmc_status_t s;


    s = lmmc_eigen_symmetric(NULL, &sym_result);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "eigen_symmetric(NULL) should return INVALID_ARGUMENT, got %d", (int)s);


    {
        lmmc_mat_t mat;
        lmmc_mat_create(2, 2, &mat);
        s = lmmc_eigen_symmetric(&mat, NULL);
        CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
              "eigen_symmetric(mat, NULL) should return INVALID_ARGUMENT, got %d", (int)s);
        lmmc_mat_destroy(&mat);
    }


    {
        lmmc_mat_t mat;
        lmmc_mat_create(2, 3, &mat);
        s = lmmc_eigen_symmetric(&mat, &sym_result);
        CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
              "eigen_symmetric(2x3) should return INVALID_ARGUMENT, got %d", (int)s);
        lmmc_mat_destroy(&mat);
    }


    s = lmmc_eigen_general(NULL, &gen_result);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "eigen_general(NULL) should return INVALID_ARGUMENT, got %d", (int)s);


    {
        lmmc_mat_t mat;
        lmmc_mat_create(3, 2, &mat);
        s = lmmc_eigen_general(&mat, &gen_result);
        CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
              "eigen_general(3x2) should return INVALID_ARGUMENT, got %d", (int)s);
        lmmc_mat_destroy(&mat);
    }


    s = lmmc_svd(NULL, &svd_result);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "svd(NULL) should return INVALID_ARGUMENT, got %d", (int)s);


    {
        lmmc_mat_t pinv;
        lmmc_mat_create(2, 2, &pinv);
        s = lmmc_pinv(NULL, 0.0, &pinv);
        CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
              "pinv(NULL) should return INVALID_ARGUMENT, got %d", (int)s);
        lmmc_mat_destroy(&pinv);
    }


    {
        lmmc_real_t cond_val;
        s = lmmc_cond(NULL, &cond_val);
        CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
              "cond(NULL) should return INVALID_ARGUMENT, got %d", (int)s);
    }

    return 0;
}


static int test_symmetric_eigen_extreme_householder_scale(void)
{
    lmmc_mat_t mat = {0};
    lmmc_eigen_sym_result_t result = {0};
    const double scale = 1.0e200;
    const double expected = sqrt(2.0);

    lmmc_status_t s = lmmc_mat_create(3, 3, &mat);
    CHECK(s == LMMC_STATUS_OK,
          "extreme-scale symmetric matrix creation failed");
    lmmc_mat_fill(&mat, 0.0);
    MAT_ELEM(&mat, 0, 1) = scale;
    MAT_ELEM(&mat, 1, 0) = scale;
    MAT_ELEM(&mat, 0, 2) = scale;
    MAT_ELEM(&mat, 2, 0) = scale;

    s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK,
          "symmetric eigen should survive representable Householder scale, got %d",
          (int)s);
    CHECK(isfinite(result.eigenvalues.data[0]) &&
              isfinite(result.eigenvalues.data[1]) &&
              isfinite(result.eigenvalues.data[2]),
          "extreme-scale symmetric eigenvalues must be finite");
    CHECK(fabs(result.eigenvalues.data[0] / scale + expected) < 1.0e-10 &&
              fabs(result.eigenvalues.data[1] / scale) < 1.0e-10 &&
              fabs(result.eigenvalues.data[2] / scale - expected) < 1.0e-10,
          "extreme-scale symmetric matrix preserves analytic eigenvalues");

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_general_eigen_extreme_householder_scale(void)
{
    lmmc_mat_t mat = {0};
    lmmc_eigen_gen_result_t result = {0};
    const double scale = 1.0e200;
    int found_one = 0, found_two = 0, found_three = 0;

    lmmc_status_t s = lmmc_mat_create(3, 3, &mat);
    CHECK(s == LMMC_STATUS_OK, "extreme-scale matrix creation failed");
    lmmc_mat_fill(&mat, 0.0);
    MAT_ELEM(&mat, 0, 0) = scale;
    MAT_ELEM(&mat, 1, 0) = scale;
    MAT_ELEM(&mat, 1, 1) = 2.0 * scale;
    MAT_ELEM(&mat, 2, 0) = scale;
    MAT_ELEM(&mat, 2, 2) = 3.0 * scale;

    s = lmmc_eigen_general(&mat, &result);
    CHECK(s == LMMC_STATUS_OK,
          "general eigen should survive representable Householder scale, got %d",
          (int)s);
    for (size_t i = 0; i < 3; ++i) {
        CHECK(isfinite(result.real_parts.data[i]) &&
                  isfinite(result.imag_parts.data[i]),
              "extreme-scale general eigenvalue must be finite");
        CHECK(fabs(result.imag_parts.data[i]) < 1.0e-12 * scale,
              "lower-triangular matrix eigenvalues must be real");
        found_one |= fabs(result.real_parts.data[i] / scale - 1.0) < 1.0e-10;
        found_two |= fabs(result.real_parts.data[i] / scale - 2.0) < 1.0e-10;
        found_three |= fabs(result.real_parts.data[i] / scale - 3.0) < 1.0e-10;
    }
    CHECK(found_one && found_two && found_three,
          "lower-triangular extreme-scale matrix preserves diagonal eigenvalues");

    lmmc_eigen_gen_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

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
        {"general_eigen_extreme_scale", test_general_eigen_extreme_householder_scale},
        {"symmetric_eigen_extreme_scale", test_symmetric_eigen_extreme_householder_scale},
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
