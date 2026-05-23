/**
 * @file test_eigen_sym_prop.c
 * @brief 针对 LMMC 中 eigen sym prop 相关接口的单元测试。
 *
 * @internal
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"


#define NUM_ITERATIONS 80


#define BASE_TOL 1e-9

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)


static double rand_double(double range)
{
    return ((double)rand() / (double)RAND_MAX) * 2.0 * range - range;
}


static size_t rand_size(size_t min_n, size_t max_n)
{
    return min_n + (size_t)(rand() % (int)(max_n - min_n + 1));
}


static void make_random_symmetric_BplusBT(lmmc_mat_t* A, double range)
{
    size_t n = A->rows;
    double* B = (double*)malloc(n * n * sizeof(double));
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = rand_double(range);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A->data[i * n + j] = B[i * n + j] + B[j * n + i];
        }
    }
    free(B);
}


static void make_random_symmetric_BBT(lmmc_mat_t* A, double range)
{
    size_t n = A->rows;
    double* B = (double*)malloc(n * n * sizeof(double));
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = rand_double(range);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) {
                s += B[i * n + k] * B[j * n + k];
            }
            A->data[i * n + j] = s;
        }
    }
    free(B);
}


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


static int test_eigen_sym_property(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_size(2, 6);
        lmmc_mat_t A;
        lmmc_status_t st = lmmc_mat_create(n, n, &A);
        CHECK(st == LMMC_STATUS_OK,
              "mat_create failed (iter=%d, n=%zu)", iter, n);


        double range = (iter % 3 == 0) ? 1.0
                     : (iter % 3 == 1) ? 10.0
                                       : 0.1;


        if (iter % 2 == 0) {
            make_random_symmetric_BplusBT(&A, range);
        } else {
            make_random_symmetric_BBT(&A, range);
        }

        double a_norm = mat_fro_norm(&A);

        double tol = BASE_TOL * (a_norm + 1.0) * (double)n * (double)n;
        double tol_ortho = BASE_TOL * (double)n * (double)n;

        lmmc_eigen_sym_result_t res;
        st = lmmc_eigen_symmetric(&A, &res);
        CHECK(st == LMMC_STATUS_OK,
              "eigen_symmetric failed status=%d (iter=%d, n=%zu)",
              (int)st, iter, n);

        const lmmc_real_t* V = res.eigenvectors.data;
        const lmmc_real_t* lam = res.eigenvalues.data;


        for (size_t i = 0; i + 1 < n; ++i) {
            CHECK(lam[i] <= lam[i + 1] + tol,
                  "eigenvalues not ascending: lambda[%zu]=%.15g > lambda[%zu]=%.15g (iter=%d, n=%zu)",
                  i, lam[i], i + 1, lam[i + 1], iter, n);
        }


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


        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double dot = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    dot += V[k * n + i] * V[k * n + j];
                }
                double expected = (i == j) ? 1.0 : 0.0;
                double diff = fabs(dot - expected);
                CHECK(diff <= tol_ortho,
                      "orthogonality mismatch (V^T*V)[%zu][%zu]: expected %.1f got %.15g diff=%.3e tol=%.3e (iter=%d, n=%zu)",
                      i, j, expected, dot, diff, tol_ortho, iter, n);
            }
        }

        lmmc_eigen_sym_result_destroy(&res);
        lmmc_mat_destroy(&A);
    }

    return 0;
}


int main(void)
{
    int rc = 0;


    srand(0xC0FFEEu);

    printf("=== Property 8: 对称特征值分解重构 ===\n");
    printf("  Validates: Requirements 5.1, 5.4, 5.5\n");
    printf("  Iterations: %d, sizes 2x2..6x6\n\n", NUM_ITERATIONS);

    if (test_eigen_sym_property()) {
        rc = 1;
        printf("  [FAIL] random symmetric eigen-decomposition\n");
    } else {
        printf("  [PASS] random symmetric eigen-decomposition (%d iterations)\n",
               NUM_ITERATIONS);
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
