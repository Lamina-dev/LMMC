/**
 * @file test_eigen_full.c
 * Unit tests for lmmc_eigen_general_full (eigenvalues + eigenvectors).
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

#define TOL 1e-8

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        return 1; \
    } \
} while (0)

#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])

/**
 * @brief Verify eigenpair residual: ||A*v - lambda*v||_2 <= tol * (1 + ||A||_F) * ||v||_2
 * For complex eigenvalue lambda = re + i*im with eigenvector v = vr + i*vi:
 *   A*(vr + i*vi) = (re + i*im)*(vr + i*vi)
 *   Real part: A*vr = re*vr - im*vi
 *   Imag part: A*vi = re*vi + im*vr
 */
static int verify_eigenpair_residual(const lmmc_mat_t *A, size_t n,
                                     lmmc_real_t re, lmmc_real_t im,
                                     const lmmc_real_t *vr, const lmmc_real_t *vi,
                                     double tol)
{
    /* Compute ||A||_F */
    double norm_A = 0.0;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            norm_A += MAT_ELEM(A, i, j) * MAT_ELEM(A, i, j);
    norm_A = sqrt(norm_A);

    /* Compute ||v||_2 = sqrt(||vr||^2 + ||vi||^2) */
    double norm_v = 0.0;
    for (size_t i = 0; i < n; i++)
        norm_v += vr[i] * vr[i] + vi[i] * vi[i];
    norm_v = sqrt(norm_v);

    if (norm_v < 1e-15) {
        printf("  FAIL: eigenvector has zero norm\n");
        return 1;
    }

    /* Compute residual: A*v - lambda*v */
    double residual_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        /* (A*vr)_i */
        double Avr_i = 0.0;
        double Avi_i = 0.0;
        for (size_t j = 0; j < n; j++) {
            Avr_i += MAT_ELEM(A, i, j) * vr[j];
            Avi_i += MAT_ELEM(A, i, j) * vi[j];
        }
        /* lambda*v: (re + i*im)*(vr + i*vi) = (re*vr - im*vi) + i*(re*vi + im*vr) */
        double lv_real = re * vr[i] - im * vi[i];
        double lv_imag = re * vi[i] + im * vr[i];

        double dr = Avr_i - lv_real;
        double di = Avi_i - lv_imag;
        residual_sq += dr * dr + di * di;
    }
    double residual = sqrt(residual_sq);
    double bound = tol * (1.0 + norm_A) * norm_v;

    if (residual > bound) {
        printf("  FAIL: residual %.3e > bound %.3e (lambda = %.6f + %.6fi)\n",
               residual, bound, re, im);
        return 1;
    }
    return 0;
}

/* Simple LCG for reproducible random matrices */
static unsigned long lcg_state = 54321UL;
static double next_uniform(double lo, double hi)
{
    lcg_state = lcg_state * 1664525UL + 1013904223UL;
    double u = (double)(lcg_state & 0xFFFFFFFFUL) / 4294967296.0;
    return lo + (hi - lo) * u;
}

/* ===== Test Cases ===== */

static int test_invalid_inputs(void)
{
    lmmc_eigen_gen_full_result_t result;
    memset(&result, 0, sizeof(result));
    lmmc_status_t st;

    st = lmmc_eigen_general_full(NULL, &result);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "NULL input matrix should return INVALID_ARGUMENT, got %d", (int)st);

    lmmc_mat_t mat;
    st = lmmc_mat_create(3, 3, &mat);
    CHECK(st == LMMC_STATUS_OK, "mat_create failed");
    st = lmmc_eigen_general_full(&mat, NULL);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "NULL out_result should return INVALID_ARGUMENT, got %d", (int)st);
    lmmc_mat_destroy(&mat);

    /* Non-square */
    st = lmmc_mat_create(2, 3, &mat);
    CHECK(st == LMMC_STATUS_OK, "mat_create non-square failed");
    st = lmmc_eigen_general_full(&mat, &result);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "non-square input should return INVALID_ARGUMENT, got %d", (int)st);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_1x1(void)
{
    lmmc_mat_t mat;
    lmmc_status_t st = lmmc_mat_create(1, 1, &mat);
    CHECK(st == LMMC_STATUS_OK, "mat_create failed");
    MAT_ELEM(&mat, 0, 0) = 5.0;

    lmmc_eigen_gen_full_result_t result;
    st = lmmc_eigen_general_full(&mat, &result);
    CHECK(st == LMMC_STATUS_OK, "lmmc_eigen_general_full failed: %d", (int)st);

    CHECK(fabs(result.real_parts.data[0] - 5.0) < TOL,
          "eigenvalue should be 5.0, got %.6f", result.real_parts.data[0]);
    CHECK(fabs(result.imag_parts.data[0]) < TOL,
          "imaginary part should be 0, got %.6f", result.imag_parts.data[0]);
    CHECK(fabs(MAT_ELEM(&result.vectors_real, 0, 0) - 1.0) < TOL,
          "eigenvector should be [1], got %.6f", MAT_ELEM(&result.vectors_real, 0, 0));

    lmmc_eigen_gen_full_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_diagonal_2x2(void)
{
    lmmc_mat_t mat;
    lmmc_status_t st = lmmc_mat_create(2, 2, &mat);
    CHECK(st == LMMC_STATUS_OK, "mat_create failed");
    MAT_ELEM(&mat, 0, 0) = 3.0; MAT_ELEM(&mat, 0, 1) = 0.0;
    MAT_ELEM(&mat, 1, 0) = 0.0; MAT_ELEM(&mat, 1, 1) = 7.0;

    lmmc_eigen_gen_full_result_t result;
    st = lmmc_eigen_general_full(&mat, &result);
    CHECK(st == LMMC_STATUS_OK, "lmmc_eigen_general_full failed: %d", (int)st);

    /* Verify eigenpair residuals */
    for (size_t i = 0; i < 2; i++) {
        lmmc_real_t vr[2] = { MAT_ELEM(&result.vectors_real, 0, i),
                              MAT_ELEM(&result.vectors_real, 1, i) };
        lmmc_real_t vi[2] = { MAT_ELEM(&result.vectors_imag, 0, i),
                              MAT_ELEM(&result.vectors_imag, 1, i) };
        int rc = verify_eigenpair_residual(&mat, 2,
                    result.real_parts.data[i], result.imag_parts.data[i],
                    vr, vi, TOL);
        CHECK(rc == 0, "eigenpair %zu residual check failed", i);
    }

    /* For real eigenvalues, imag part of vectors should be zero */
    for (size_t i = 0; i < 2; i++) {
        if (fabs(result.imag_parts.data[i]) < TOL) {
            for (size_t j = 0; j < 2; j++) {
                CHECK(fabs(MAT_ELEM(&result.vectors_imag, j, i)) < TOL,
                      "vectors_imag[%zu,%zu] should be 0 for real eigenvalue", j, i);
            }
        }
    }

    lmmc_eigen_gen_full_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_rotation_2x2(void)
{
    /* Rotation matrix has complex eigenvalues */
    double theta = 0.7;
    double c = cos(theta), s = sin(theta);

    lmmc_mat_t mat;
    lmmc_status_t st = lmmc_mat_create(2, 2, &mat);
    CHECK(st == LMMC_STATUS_OK, "mat_create failed");
    MAT_ELEM(&mat, 0, 0) = c;  MAT_ELEM(&mat, 0, 1) = -s;
    MAT_ELEM(&mat, 1, 0) = s;  MAT_ELEM(&mat, 1, 1) = c;

    lmmc_eigen_gen_full_result_t result;
    st = lmmc_eigen_general_full(&mat, &result);
    CHECK(st == LMMC_STATUS_OK, "lmmc_eigen_general_full failed: %d", (int)st);

    /* Should have complex-conjugate pair */
    CHECK(fabs(result.imag_parts.data[0]) > 0.1,
          "rotation should have complex eigenvalues");
    CHECK(fabs(result.imag_parts.data[0] + result.imag_parts.data[1]) < TOL,
          "eigenvalues should be conjugate pair");

    /* Verify eigenpair residuals */
    for (size_t i = 0; i < 2; i++) {
        lmmc_real_t vr[2] = { MAT_ELEM(&result.vectors_real, 0, i),
                              MAT_ELEM(&result.vectors_real, 1, i) };
        lmmc_real_t vi[2] = { MAT_ELEM(&result.vectors_imag, 0, i),
                              MAT_ELEM(&result.vectors_imag, 1, i) };
        int rc = verify_eigenpair_residual(&mat, 2,
                    result.real_parts.data[i], result.imag_parts.data[i],
                    vr, vi, TOL);
        CHECK(rc == 0, "eigenpair %zu residual check failed", i);
    }

    /* Complex-conjugate pair: adjacent columns should share real part,
     * negate imaginary part */
    for (size_t j = 0; j < 2; j++) {
        CHECK(fabs(MAT_ELEM(&result.vectors_real, j, 0) -
                   MAT_ELEM(&result.vectors_real, j, 1)) < TOL,
              "real parts of conjugate eigenvectors should match at row %zu", j);
        CHECK(fabs(MAT_ELEM(&result.vectors_imag, j, 0) +
                   MAT_ELEM(&result.vectors_imag, j, 1)) < TOL,
              "imag parts of conjugate eigenvectors should negate at row %zu", j);
    }

    lmmc_eigen_gen_full_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_diagonal_3x3(void)
{
    lmmc_mat_t mat;
    lmmc_status_t st = lmmc_mat_create(3, 3, &mat);
    CHECK(st == LMMC_STATUS_OK, "mat_create failed");
    lmmc_mat_fill(&mat, 0.0);
    MAT_ELEM(&mat, 0, 0) = 1.0;
    MAT_ELEM(&mat, 1, 1) = 2.0;
    MAT_ELEM(&mat, 2, 2) = 3.0;

    lmmc_eigen_gen_full_result_t result;
    st = lmmc_eigen_general_full(&mat, &result);
    CHECK(st == LMMC_STATUS_OK, "lmmc_eigen_general_full failed: %d", (int)st);

    /* Verify eigenpair residuals */
    for (size_t i = 0; i < 3; i++) {
        lmmc_real_t vr[3], vi[3];
        for (size_t j = 0; j < 3; j++) {
            vr[j] = MAT_ELEM(&result.vectors_real, j, i);
            vi[j] = MAT_ELEM(&result.vectors_imag, j, i);
        }
        int rc = verify_eigenpair_residual(&mat, 3,
                    result.real_parts.data[i], result.imag_parts.data[i],
                    vr, vi, TOL);
        CHECK(rc == 0, "eigenpair %zu residual check failed", i);
    }

    lmmc_eigen_gen_full_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_random_nxn(size_t n, const char *label)
{
    lmmc_mat_t mat;
    lmmc_status_t st = lmmc_mat_create(n, n, &mat);
    CHECK(st == LMMC_STATUS_OK, "[%s] mat_create failed", label);

    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            MAT_ELEM(&mat, i, j) = (lmmc_real_t)next_uniform(-2.0, 2.0);

    lmmc_eigen_gen_full_result_t result;
    st = lmmc_eigen_general_full(&mat, &result);
    CHECK(st == LMMC_STATUS_OK, "[%s] lmmc_eigen_general_full failed: %d", label, (int)st);

    /* Verify all eigenpair residuals */
    lmmc_real_t *vr = (lmmc_real_t *)malloc(n * sizeof(lmmc_real_t));
    lmmc_real_t *vi = (lmmc_real_t *)malloc(n * sizeof(lmmc_real_t));
    CHECK(vr && vi, "[%s] allocation failed", label);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            vr[j] = MAT_ELEM(&result.vectors_real, j, i);
            vi[j] = MAT_ELEM(&result.vectors_imag, j, i);
        }
        int rc = verify_eigenpair_residual(&mat, n,
                    result.real_parts.data[i], result.imag_parts.data[i],
                    vr, vi, TOL);
        if (rc != 0) {
            printf("    [%s] eigenpair %zu failed\n", label, i);
            free(vr); free(vi);
            lmmc_eigen_gen_full_result_destroy(&result);
            lmmc_mat_destroy(&mat);
            return 1;
        }
    }

    /* Verify: for real eigenvalues, imag vectors should be zero */
    for (size_t i = 0; i < n; i++) {
        if (fabs(result.imag_parts.data[i]) < TOL) {
            for (size_t j = 0; j < n; j++) {
                if (fabs(MAT_ELEM(&result.vectors_imag, j, i)) > TOL) {
                    printf("  FAIL: [%s] vectors_imag[%zu,%zu] = %.3e for real eigenvalue\n",
                           label, j, i, MAT_ELEM(&result.vectors_imag, j, i));
                    free(vr); free(vi);
                    lmmc_eigen_gen_full_result_destroy(&result);
                    lmmc_mat_destroy(&mat);
                    return 1;
                }
            }
        }
    }

    free(vr); free(vi);
    lmmc_eigen_gen_full_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_random_5x5(void)
{
    lcg_state = 11111UL;
    char label[32];
    for (int trial = 0; trial < 5; trial++) {
        snprintf(label, sizeof(label), "random_5x5_#%d", trial);
        if (test_random_nxn(5, label) != 0) return 1;
    }
    return 0;
}

static int test_random_10x10(void)
{
    lcg_state = 22222UL;
    char label[32];
    for (int trial = 0; trial < 3; trial++) {
        snprintf(label, sizeof(label), "random_10x10_#%d", trial);
        if (test_random_nxn(10, label) != 0) return 1;
    }
    return 0;
}

static int test_random_20x20(void)
{
    lcg_state = 33333UL;
    char label[32];
    for (int trial = 0; trial < 2; trial++) {
        snprintf(label, sizeof(label), "random_20x20_#%d", trial);
        if (test_random_nxn(20, label) != 0) return 1;
    }
    return 0;
}

static int test_destroy_null(void)
{
    /* Should not crash */
    lmmc_eigen_gen_full_result_destroy(NULL);
    return 0;
}

/* ===== Main ===== */

typedef int (*test_func_t)(void);
typedef struct { const char *name; test_func_t func; } test_entry_t;

int main(void)
{
    test_entry_t tests[] = {
        {"invalid_inputs",     test_invalid_inputs},
        {"destroy_null",       test_destroy_null},
        {"1x1",                test_1x1},
        {"diagonal_2x2",      test_diagonal_2x2},
        {"rotation_2x2",      test_rotation_2x2},
        {"diagonal_3x3",      test_diagonal_3x3},
        {"random_5x5",        test_random_5x5},
        {"random_10x10",      test_random_10x10},
        {"random_20x20",      test_random_20x20},
    };

    size_t n_tests = sizeof(tests) / sizeof(tests[0]);
    size_t n_passed = 0;
    size_t n_failed = 0;

    printf("=== Eigen General Full (Eigenvalues + Eigenvectors) Tests ===\n");

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

    printf("\n=== Results: %zu passed, %zu failed ===\n", n_passed, n_failed);
    return (n_failed > 0) ? 1 : 0;
}
