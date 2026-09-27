/**
 * @file test_common.h
 * @brief Shared numerical predicates for cmocka tests.
 *
 * @internal
 */
#ifndef LMMC_TEST_COMMON_H
#define LMMC_TEST_COMMON_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "lmmc/config.h"
#include "lmmc/dense_storage.h"
#include "lmmc/sparse_storage.h"

static inline int lmmc_test_nearly_equal(lmmc_real_t a, lmmc_real_t b, lmmc_real_t eps) {
    return isfinite(a) && isfinite(b) && isfinite(eps) && eps >= 0 && fabs(a - b) <= eps;
}

static inline double lmmc_test_frobenius_norm(const lmmc_mat_t *matrix) {
    double norm = 0.0;
    for (size_t i = 0; i < matrix->rows; ++i) {
        for (size_t j = 0; j < matrix->cols; ++j) {
            const double value = matrix->data[i * matrix->stride + j];
            if (!isfinite(value)) {
                return NAN;
            }
            norm = hypot(norm, value);
        }
    }
    return norm;
}

static inline lmmc_status_t lmmc_test_build_sparse(const double *data, size_t rows, size_t cols,
                                                 lmmc_sparse_mat_t *out) {
    lmmc_mat_t dense = {0};
    lmmc_status_t st = lmmc_mat_create(rows, cols, &dense);
    if (st != LMMC_STATUS_OK)
        return st;
    for (size_t i = 0; i < rows * cols; i++) {
        LMMC_REAL_SET_D(&dense.data[i], data[i]);
    }
    st = lmmc_sparse_from_dense(&dense, 1e-14, out);
    lmmc_mat_destroy(&dense);
    return st;
}

static inline int lmmc_test_check_eigen_reconstruction(
    int iter, size_t n, const lmmc_real_t *a, double tol,
    const lmmc_real_t *V, const lmmc_real_t *lam) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += V[i * n + k] * lam[k] * V[j * n + k];
            }
            double a_ij = a[i * n + j];
            double diff = fabs(sum - a_ij);
            if (!(diff <= tol)) {
                print_error("reconstruction mismatch A[%" PRIuMAX "][%" PRIuMAX "]: expected %.15g got %.15g diff=%.3e tol=%.3e (iter=%d, n=%" PRIuMAX ")\n", (uintmax_t)(i), (uintmax_t)(j), a_ij, sum, diff, tol, iter, (uintmax_t)(n));
                return 1;
            }
        }
    }
    return 0;
}

#endif
