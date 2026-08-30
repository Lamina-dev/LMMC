/**
 * @file test_eigen_residual_prop.c
 * 特征对残差精度属性测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define RESIDUAL_TOL 1e-8
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])

/**
 * @brief Compute the Frobenius norm of a matrix.
 */
static double frobenius_norm(const lmmc_mat_t *A) {
    double sum = 0.0;
    for (size_t i = 0; i < A->rows; i++)
        for (size_t j = 0; j < A->cols; j++) {
            double v = MAT_ELEM(A, i, j);
            sum += v * v;
        }
    return sqrt(sum);
}

/**
 * @brief Verify eigenpair residual for a single eigenpair.
 *
 * For complex eigenvalue lambda = re + i*im with eigenvector v = vr + i*vi:
 *   A*(vr + i*vi) should equal (re + i*im)*(vr + i*vi)
 *   Real part residual: A*vr - (re*vr - im*vi)
 *   Imag part residual: A*vi - (re*vi + im*vr)
 *
 * @return 0 if residual is within tolerance, 1 otherwise.
 */
static int verify_eigenpair(const lmmc_mat_t *A, size_t n,
                            lmmc_real_t re, lmmc_real_t im,
                            const lmmc_real_t *vr, const lmmc_real_t *vi,
                            double norm_A, double tol)
{
    /* Compute ||v||_2 = sqrt(||vr||^2 + ||vi||^2) */
    double norm_v_sq = 0.0;
    for (size_t i = 0; i < n; i++)
        norm_v_sq += vr[i] * vr[i] + vi[i] * vi[i];
    double norm_v = sqrt(norm_v_sq);

    if (norm_v < 1e-15) {
        /* Zero eigenvector is invalid */
        return 1;
    }

    /* Compute residual: ||A*v - lambda*v||_2 */
    double residual_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        /* (A*vr)_i and (A*vi)_i */
        double Avr_i = 0.0;
        double Avi_i = 0.0;
        for (size_t j = 0; j < n; j++) {
            double aij = MAT_ELEM(A, i, j);
            Avr_i += aij * vr[j];
            Avi_i += aij * vi[j];
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
        return 1;
    }
    return 0;
}

/**
 * @brief Test the eigenpair residual property for a single random matrix.
 *
 * @param rng   Random number generator.
 * @param n     Matrix dimension.
 * @param trial Trial number (for reporting).
 * @return 0 on success, 1 on failure.
 */
static int test_eigenpair_residual(lmmc_rng_t *rng, size_t n, int trial) {
    lmmc_mat_t A = {0};
    lmmc_eigen_gen_full_result_t result;
    lmmc_status_t st;
    int rc = 0;

    /* Create and fill random matrix */
    st = lmmc_mat_create(n, n, &A);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] trial %d (n=%zu): mat_create failed\n", trial, n);
        return 1;
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t val;
            lmmc_rng_uniform(rng, -2.0, 2.0, &val);
            MAT_ELEM(&A, i, j) = val;
        }
    }

    /* Compute eigenvalues and eigenvectors */
    st = lmmc_eigen_general_full(&A, &result);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] trial %d (n=%zu): lmmc_eigen_general_full returned %d\n",
               trial, n, (int)st);
        lmmc_mat_destroy(&A);
        return 1;
    }

    /* Compute ||A||_F */
    double norm_A = frobenius_norm(&A);

    /* Allocate temporary buffers for eigenvector extraction */
    lmmc_real_t *vr = (lmmc_real_t *)malloc(n * sizeof(lmmc_real_t));
    lmmc_real_t *vi = (lmmc_real_t *)malloc(n * sizeof(lmmc_real_t));
    if (!vr || !vi) {
        printf("  [FAIL] trial %d (n=%zu): allocation failed\n", trial, n);
        rc = 1;
        goto cleanup;
    }

    /* Verify residual for each eigenpair */
    for (size_t k = 0; k < n; k++) {
        for (size_t j = 0; j < n; j++) {
            vr[j] = MAT_ELEM(&result.vectors_real, j, k);
            vi[j] = MAT_ELEM(&result.vectors_imag, j, k);
        }

        lmmc_real_t re = result.real_parts.data[k];
        lmmc_real_t im = result.imag_parts.data[k];

        int err = verify_eigenpair(&A, n, re, im, vr, vi, norm_A, RESIDUAL_TOL);
        if (err != 0) {
            /* Compute actual residual for reporting */
            double norm_v_sq = 0.0;
            for (size_t j = 0; j < n; j++)
                norm_v_sq += vr[j] * vr[j] + vi[j] * vi[j];
            double norm_v = sqrt(norm_v_sq);

            double residual_sq = 0.0;
            for (size_t i = 0; i < n; i++) {
                double Avr_i = 0.0, Avi_i = 0.0;
                for (size_t j = 0; j < n; j++) {
                    double aij = MAT_ELEM(&A, i, j);
                    Avr_i += aij * vr[j];
                    Avi_i += aij * vi[j];
                }
                double lv_real = re * vr[i] - im * vi[i];
                double lv_imag = re * vi[i] + im * vr[i];
                double dr = Avr_i - lv_real;
                double di = Avi_i - lv_imag;
                residual_sq += dr * dr + di * di;
            }
            double residual = sqrt(residual_sq);
            double bound = RESIDUAL_TOL * (1.0 + norm_A) * norm_v;

            printf("  [FAIL] trial %d (n=%zu): eigenpair %zu residual %.3e > bound %.3e "
                   "(lambda = %.6f + %.6fi, ||v||=%.3e)\n",
                   trial, n, k, residual, bound, re, im, norm_v);
            rc = 1;
            goto cleanup;
        }
    }

cleanup:
    free(vr);
    free(vi);
    lmmc_eigen_gen_full_result_destroy(&result);
    lmmc_mat_destroy(&A);
    return rc;
}

int main(void) {
    lmmc_rng_t *rng = NULL;
    lmmc_status_t st;
    int failures = 0;
    int total_trials = 0;

    printf("=== Property Test: Eigenpair Residual ===\n");
    printf("||A*v - lambda*v||_2 <= 1e-8 * (1 + ||A||_F) * ||v||_2\n");

    /* 固定种子保证属性测试可复现。 */
    const uint64_t seed = UINT64_C(0x45524553);
    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) {
        printf("FATAL: Failed to create RNG\n");
        return 1;
    }
    lmmc_rng_seed(rng, seed);

    /* Test various matrix sizes from 3x3 to 20x20 for speed */
    size_t sizes[] = {3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int trials_per_size = 8;

    for (size_t si = 0; si < num_sizes; si++) {
        size_t n = sizes[si];
        printf("Testing n=%zu (%d trials)...\n", n, trials_per_size);
        for (int t = 0; t < trials_per_size; t++) {
            total_trials++;
            if (test_eigenpair_residual(rng, n, t + 1) != 0) {
                failures++;
                /* Print counterexample info and stop early for this size */
                break;
            }
        }
        if (failures > 0) break;
    }

    printf("\n=== Results ===\n");
    printf("Total trials: %d\n", total_trials);
    printf("Passed: %d\n", total_trials - failures);
    printf("Failed: %d\n", failures);

    if (failures == 0) {
        printf("\nProperty test PASSED: All eigenpair residuals within tolerance.\n");
    } else {
        printf("\nProperty test FAILED: %d/%d trials violated the property.\n",
               failures, total_trials);
    }

    lmmc_rng_destroy(rng);
    return (failures == 0) ? 0 : 1;
}
