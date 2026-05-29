/**
 * @file test_eigen_sym.c
 * 针对 LMMC 中 eigen sym 相关接口的单元测试。
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


static int test_null_input(void)
{
    lmmc_eigen_sym_result_t result;
    lmmc_status_t s;

    s = lmmc_eigen_symmetric(NULL, &result);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "eigen_symmetric(NULL, ...) should return INVALID_ARGUMENT, got %d", (int)s);

    lmmc_mat_t mat;
    lmmc_mat_create(3, 3, &mat);
    s = lmmc_eigen_symmetric(&mat, NULL);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "eigen_symmetric(..., NULL) should return INVALID_ARGUMENT, got %d", (int)s);
    lmmc_mat_destroy(&mat);

    return 0;
}


static int test_non_square(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    lmmc_mat_create(2, 3, &mat);

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "eigen_symmetric on 2x3 should return INVALID_ARGUMENT, got %d", (int)s);

    lmmc_mat_destroy(&mat);
    return 0;
}


static int test_1x1(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    lmmc_mat_create(1, 1, &mat);
    mat.data[0] = 5.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "1x1 should succeed, got %d", (int)s);
    CHECK(fabs(result.eigenvalues.data[0] - 5.0) < TOL,
          "1x1 eigenvalue should be 5.0, got %f", result.eigenvalues.data[0]);
    CHECK(fabs(result.eigenvectors.data[0] - 1.0) < TOL,
          "1x1 eigenvector should be 1.0, got %f", result.eigenvectors.data[0]);

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}


static int test_2x2_diagonal(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    lmmc_mat_create(2, 2, &mat);
    mat.data[0] = 3.0; mat.data[1] = 0.0;
    mat.data[2] = 0.0; mat.data[3] = 7.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "2x2 diagonal should succeed, got %d", (int)s);


    CHECK(fabs(result.eigenvalues.data[0] - 3.0) < TOL,
          "eigenvalue[0] should be 3.0, got %f", result.eigenvalues.data[0]);
    CHECK(fabs(result.eigenvalues.data[1] - 7.0) < TOL,
          "eigenvalue[1] should be 7.0, got %f", result.eigenvalues.data[1]);

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}


static int test_2x2_symmetric(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    lmmc_mat_create(2, 2, &mat);
    mat.data[0] = 2.0; mat.data[1] = 1.0;
    mat.data[2] = 1.0; mat.data[3] = 2.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "2x2 symmetric should succeed, got %d", (int)s);


    CHECK(fabs(result.eigenvalues.data[0] - 1.0) < TOL,
          "eigenvalue[0] should be 1.0, got %f", result.eigenvalues.data[0]);
    CHECK(fabs(result.eigenvalues.data[1] - 3.0) < TOL,
          "eigenvalue[1] should be 3.0, got %f", result.eigenvalues.data[1]);

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}


static int test_3x3_identity(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    lmmc_mat_create(3, 3, &mat);
    mat.data[0] = 1.0; mat.data[1] = 0.0; mat.data[2] = 0.0;
    mat.data[3] = 0.0; mat.data[4] = 1.0; mat.data[5] = 0.0;
    mat.data[6] = 0.0; mat.data[7] = 0.0; mat.data[8] = 1.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "3x3 identity should succeed, got %d", (int)s);


    for (int i = 0; i < 3; i++) {
        CHECK(fabs(result.eigenvalues.data[i] - 1.0) < TOL,
              "eigenvalue[%d] should be 1.0, got %f", i, result.eigenvalues.data[i]);
    }

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}


static int test_3x3_symmetric(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    lmmc_mat_create(3, 3, &mat);

    mat.data[0] = 4.0; mat.data[1] = 1.0; mat.data[2] = 1.0;
    mat.data[3] = 1.0; mat.data[4] = 4.0; mat.data[5] = 1.0;
    mat.data[6] = 1.0; mat.data[7] = 1.0; mat.data[8] = 4.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "3x3 symmetric should succeed, got %d", (int)s);


    CHECK(fabs(result.eigenvalues.data[0] - 3.0) < TOL,
          "eigenvalue[0] should be 3.0, got %f", result.eigenvalues.data[0]);
    CHECK(fabs(result.eigenvalues.data[1] - 3.0) < TOL,
          "eigenvalue[1] should be 3.0, got %f", result.eigenvalues.data[1]);
    CHECK(fabs(result.eigenvalues.data[2] - 6.0) < TOL,
          "eigenvalue[2] should be 6.0, got %f", result.eigenvalues.data[2]);

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}


static int test_orthogonality(void)
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
    for (size_t i = 0; i < n * n; i++) {
        mat.data[i] = data[i];
    }

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "4x4 symmetric should succeed, got %d", (int)s);


    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t dot = 0.0;
            for (size_t k = 0; k < n; k++) {
                dot += result.eigenvectors.data[k * n + i] *
                       result.eigenvectors.data[k * n + j];
            }
            lmmc_real_t expected = (i == j) ? 1.0 : 0.0;
            CHECK(fabs(dot - expected) < TOL,
                  "V^T*V[%zu][%zu] should be %f, got %f", i, j, expected, dot);
        }
    }

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}


static int test_reconstruction(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    size_t n = 3;
    lmmc_mat_create(n, n, &mat);


    lmmc_real_t data[] = {
        2.0, -1.0, 0.0,
       -1.0,  2.0, -1.0,
        0.0, -1.0,  2.0
    };
    for (size_t i = 0; i < n * n; i++) {
        mat.data[i] = data[i];
    }

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "3x3 tridiag should succeed, got %d", (int)s);


    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t sum = 0.0;
            for (size_t k = 0; k < n; k++) {

                sum += result.eigenvectors.data[i * n + k] *
                       result.eigenvalues.data[k] *
                       result.eigenvectors.data[j * n + k];
            }
            CHECK(fabs(sum - data[i * n + j]) < 1e-8,
                  "Reconstruction A[%zu][%zu]: expected %f, got %f",
                  i, j, data[i * n + j], sum);
        }
    }

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}


static int test_ascending_order(void)
{
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
    size_t n = 4;
    lmmc_mat_create(n, n, &mat);


    lmmc_real_t data[] = {
        10.0, 1.0, 2.0, 3.0,
         1.0, 5.0, 1.0, 2.0,
         2.0, 1.0, 3.0, 1.0,
         3.0, 2.0, 1.0, 1.0
    };
    for (size_t i = 0; i < n * n; i++) {
        mat.data[i] = data[i];
    }

    lmmc_status_t s = lmmc_eigen_symmetric(&mat, &result);
    CHECK(s == LMMC_STATUS_OK, "4x4 should succeed, got %d", (int)s);


    for (size_t i = 0; i < n - 1; i++) {
        CHECK(result.eigenvalues.data[i] <= result.eigenvalues.data[i + 1],
              "eigenvalues not ascending: [%zu]=%f > [%zu]=%f",
              i, result.eigenvalues.data[i], i + 1, result.eigenvalues.data[i + 1]);
    }

    lmmc_eigen_sym_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}


static int test_destroy_null(void)
{

    lmmc_eigen_sym_result_destroy(NULL);
    lmmc_eigen_gen_result_destroy(NULL);
    lmmc_svd_result_destroy(NULL);
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
        {"null_input", test_null_input},
        {"non_square", test_non_square},
        {"1x1", test_1x1},
        {"2x2_diagonal", test_2x2_diagonal},
        {"2x2_symmetric", test_2x2_symmetric},
        {"3x3_identity", test_3x3_identity},
        {"3x3_symmetric", test_3x3_symmetric},
        {"orthogonality", test_orthogonality},
        {"reconstruction", test_reconstruction},
        {"ascending_order", test_ascending_order},
        {"destroy_null", test_destroy_null},
    };

    size_t n_tests = sizeof(tests) / sizeof(tests[0]);
    int n_passed = 0;
    int n_failed = 0;

    printf("=== Symmetric Eigenvalue Decomposition Tests ===\n\n");

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
