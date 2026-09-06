/**
 * @file eigen_symmetric.c
 * @brief 对称矩阵特征值分解（Householder 三对角化 + QL 迭代）实现。
 */

#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "eigen_internal.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/linear_algebra.h"

static lmmc_status_t householder_tridiag(lmmc_mat_t *a, lmmc_real_t *diag, lmmc_real_t *offdiag, lmmc_real_t *taus) {
    size_t n = a->rows;
    size_t k, i, j;
    if (taus) { for (i = 0; i + 1 < n; i++) taus[i] = 0.0; }
    if (n == 1) { diag[0] = MAT_ELEM(a, 0, 0); return LMMC_STATUS_OK; }
    for (k = 0; k < n - 1; k++) {
        lmmc_scaled_sumsq_t acc;
        lmmc_scaled_sumsq_init(&acc);
        for (i = k + 2; i < n; i++) {
            lmmc_scaled_sumsq_add(&acc, MAT_ELEM(a, i, k));
        }
        lmmc_real_t tail_norm = lmmc_scaled_sumsq_norm(&acc);
        lmmc_real_t alpha_val = MAT_ELEM(a, k + 1, k);
        if (tail_norm == 0.0) {
            offdiag[k] = alpha_val; diag[k] = MAT_ELEM(a, k, k);
            if (taus) taus[k] = 0.0;

            for (i = k + 2; i < n; i++) MAT_ELEM(a, i, k) = 0.0;
            continue;
        }
        lmmc_real_t norm_x = hypot(alpha_val, tail_norm);
        lmmc_real_t beta = -copysign(norm_x, alpha_val);
        lmmc_real_t tau = 1.0 - alpha_val / beta;
        lmmc_real_t denominator_scaled =
            alpha_val / norm_x - beta / norm_x;
        for (i = k + 2; i < n; i++) {
            MAT_ELEM(a, i, k) =
                (MAT_ELEM(a, i, k) / norm_x) / denominator_scaled;
        }
        offdiag[k] = beta;
        if (taus) taus[k] = tau;
        lmmc_real_t *w = (lmmc_real_t *)lmmc_alloc_array(
            n - k - 1, sizeof(lmmc_real_t));
        if (!w) return LMMC_STATUS_ALLOCATION_FAILED;
        for (i = 0; i < n - k - 1; i++) {
            lmmc_real_t sum = 0.0;
            for (j = 0; j < n - k - 1; j++) {
                lmmc_real_t vj = (j == 0) ? 1.0 : MAT_ELEM(a, k + 1 + j, k);
                sum += MAT_ELEM(a, k + 1 + i, k + 1 + j) * vj;
            }
            w[i] = tau * sum;
        }
        lmmc_real_t gamma = 0.0;
        for (i = 0; i < n - k - 1; i++) {
            lmmc_real_t vi = (i == 0) ? 1.0 : MAT_ELEM(a, k + 1 + i, k);
            gamma += w[i] * vi;
        }
        gamma *= 0.5 * tau;
        for (i = 0; i < n - k - 1; i++) {
            lmmc_real_t vi = (i == 0) ? 1.0 : MAT_ELEM(a, k + 1 + i, k);
            w[i] -= gamma * vi;
        }
        for (i = 0; i < n - k - 1; i++) {
            lmmc_real_t vi = (i == 0) ? 1.0 : MAT_ELEM(a, k + 1 + i, k);
            for (j = i; j < n - k - 1; j++) {
                lmmc_real_t vj = (j == 0) ? 1.0 : MAT_ELEM(a, k + 1 + j, k);
                lmmc_real_t update = vi * w[j] + w[i] * vj;
                MAT_ELEM(a, k + 1 + i, k + 1 + j) -= update;
                if (i != j) MAT_ELEM(a, k + 1 + j, k + 1 + i) -= update;
            }
        }
        lmmc_free(w);
    }
    for (i = 0; i < n; i++) diag[i] = MAT_ELEM(a, i, i);
    return LMMC_STATUS_OK;
}

static lmmc_status_t accumulate_Q(const lmmc_mat_t *a, const lmmc_real_t *taus, lmmc_mat_t *Q) {
    size_t n = a->rows; size_t i, j, k;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) MAT_ELEM(Q, i, j) = (i == j) ? 1.0 : 0.0;
    if (n <= 2) return LMMC_STATUS_OK;

    for (k = n - 3; ; k--) {
        lmmc_real_t tau = taus ? taus[k] : 0.0;
        if (tau != 0.0) {

            for (j = 0; j < n; j++) {
                lmmc_real_t s = MAT_ELEM(Q, k + 1, j);
                for (i = k + 2; i < n; i++) s += MAT_ELEM(a, i, k) * MAT_ELEM(Q, i, j);
                s *= tau;
                MAT_ELEM(Q, k + 1, j) -= s;
                for (i = k + 2; i < n; i++) MAT_ELEM(Q, i, j) -= s * MAT_ELEM(a, i, k);
            }
        }
        if (k == 0) break;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t tridiag_ql(lmmc_real_t *diag, lmmc_real_t *offdiag, size_t n, lmmc_mat_t *Q) {
    size_t i, k, l, m, iter;
    lmmc_real_t c, g, p, r, s, f, b, tst1;
    lmmc_real_t eps = 2.2204460492503131e-16;
    if (n <= 1) return LMMC_STATUS_OK;
    for (l = 0; l < n; l++) {
        iter = 0;
        while (1) {
            for (m = l; m < n - 1; m++) {
                tst1 = lmmc_abs(diag[m]) + lmmc_abs(diag[m + 1]);
                if (tst1 == 0.0) tst1 = 1.0;
                if (lmmc_abs(offdiag[m]) <= eps * tst1) break;
            }
            if (m == l) break;
            if (iter >= (size_t)EIGEN_MAX_ITER) return LMMC_STATUS_CONVERGENCE_FAILED;
            iter++;
            g = (diag[l + 1] - diag[l]) / (2.0 * offdiag[l]);
            r = hypot(g, 1.0);
            if (g >= 0.0) g = diag[m] - diag[l] + offdiag[l] / (g + r);
            else g = diag[m] - diag[l] + offdiag[l] / (g - r);
            s = 1.0; c = 1.0; p = 0.0;
            for (i = m; i > l; i--) {
                f = s * offdiag[i - 1]; b = c * offdiag[i - 1];
                if (lmmc_abs(f) >= lmmc_abs(g)) {
                    c = g / f; r = sqrt(c * c + 1.0); offdiag[i] = f * r; s = 1.0 / r; c *= s;
                } else {
                    s = f / g; r = sqrt(s * s + 1.0); offdiag[i] = g * r; c = 1.0 / r; s *= c;
                }
                g = diag[i] - p; r = (diag[i - 1] - g) * s + 2.0 * c * b;
                p = s * r; diag[i] = g + p; g = c * r - b;
                if (Q) {
                    for (k = 0; k < n; k++) {
                        lmmc_real_t fv = MAT_ELEM(Q, k, i);
                        MAT_ELEM(Q, k, i) = s * MAT_ELEM(Q, k, i - 1) + c * fv;
                        MAT_ELEM(Q, k, i - 1) = c * MAT_ELEM(Q, k, i - 1) - s * fv;
                    }
                }
            }
            diag[l] -= p; offdiag[l] = g; offdiag[m] = 0.0;
        }
    }
    return LMMC_STATUS_OK;
}

static void sort_eigen_ascending(lmmc_real_t *eigenvalues, lmmc_mat_t *Q, size_t n) {
    size_t i, j, min_idx;
    for (i = 0; i < n - 1; i++) {
        min_idx = i;
        for (j = i + 1; j < n; j++) if (eigenvalues[j] < eigenvalues[min_idx]) min_idx = j;
        if (min_idx != i) {
            lmmc_swap(&eigenvalues[i], &eigenvalues[min_idx]);
            if (Q) for (j = 0; j < n; j++) lmmc_swap(&MAT_ELEM(Q, j, i), &MAT_ELEM(Q, j, min_idx));
        }
    }
}

lmmc_status_t lmmc_eigen_symmetric(const lmmc_mat_t *a, lmmc_eigen_sym_result_t *out_result) {
    lmmc_status_t status;
    lmmc_real_t matrix_scale = 0.0;
    size_t n, i, j;
    if (!lmmc_mat_descriptor_is_valid(a) || !out_result) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    n = a->rows;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            lmmc_real_t magnitude = fabs(MAT_ELEM(a, i, j));
            if (!isfinite(magnitude)) {
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            if (magnitude > matrix_scale) matrix_scale = magnitude;
        }
    }
    if (matrix_scale == 0.0) matrix_scale = 1.0;
    status = lmmc_vec_create(n, &out_result->eigenvalues);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_create(n, n, &out_result->eigenvectors);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&out_result->eigenvalues); return status; }
    if (n == 1) {
        out_result->eigenvalues.data[0] = MAT_ELEM(a, 0, 0);
        MAT_ELEM(&out_result->eigenvectors, 0, 0) = 1.0;
        return LMMC_STATUS_OK;
    }
    lmmc_mat_t work;
    status = lmmc_mat_create(n, n, &work);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&out_result->eigenvalues); lmmc_mat_destroy(&out_result->eigenvectors); return status; }
    status = lmmc_mat_copy(a, &work);
    if (status != LMMC_STATUS_OK) { lmmc_mat_destroy(&work); lmmc_vec_destroy(&out_result->eigenvalues); lmmc_mat_destroy(&out_result->eigenvectors); return status; }
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            MAT_ELEM(&work, i, j) /= matrix_scale;
        }
    }
    lmmc_real_t *offdiag = (lmmc_real_t *)lmmc_alloc_array(
        n, sizeof(lmmc_real_t));
    if (!offdiag) { lmmc_mat_destroy(&work); lmmc_vec_destroy(&out_result->eigenvalues); lmmc_mat_destroy(&out_result->eigenvectors); return LMMC_STATUS_ALLOCATION_FAILED; }
    memset(offdiag, 0, n * sizeof(lmmc_real_t));
    lmmc_real_t *taus = (lmmc_real_t *)lmmc_alloc_array(
        n > 1 ? n - 1 : 1, sizeof(lmmc_real_t));
    if (!taus) { lmmc_free(offdiag); lmmc_mat_destroy(&work); lmmc_vec_destroy(&out_result->eigenvalues); lmmc_mat_destroy(&out_result->eigenvectors); return LMMC_STATUS_ALLOCATION_FAILED; }
    status = householder_tridiag(&work, out_result->eigenvalues.data, offdiag, taus);
    if (status != LMMC_STATUS_OK) { lmmc_free(taus); lmmc_free(offdiag); lmmc_mat_destroy(&work); lmmc_vec_destroy(&out_result->eigenvalues); lmmc_mat_destroy(&out_result->eigenvectors); return status; }
    status = accumulate_Q(&work, taus, &out_result->eigenvectors);
    lmmc_free(taus);
    lmmc_mat_destroy(&work);
    if (status != LMMC_STATUS_OK) { lmmc_free(offdiag); lmmc_vec_destroy(&out_result->eigenvalues); lmmc_mat_destroy(&out_result->eigenvectors); return status; }

    {
        size_t ii, jj;
        lmmc_real_t s = 1.0;
        for (ii = 0; ii < n - 1; ii++) {
            lmmc_real_t off = offdiag[ii];
            lmmc_real_t sign_next = (off >= 0.0) ? s : -s;
            offdiag[ii] = (off >= 0.0) ? off : -off;
            s = sign_next;
            if (sign_next < 0.0) {

                for (jj = 0; jj < n; jj++) {
                    MAT_ELEM(&out_result->eigenvectors, jj, ii + 1) =
                        -MAT_ELEM(&out_result->eigenvectors, jj, ii + 1);
                }
            }
        }
    }
    status = tridiag_ql(out_result->eigenvalues.data, offdiag, n, &out_result->eigenvectors);
    lmmc_free(offdiag);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&out_result->eigenvalues); lmmc_mat_destroy(&out_result->eigenvectors); return status; }
    sort_eigen_ascending(out_result->eigenvalues.data, &out_result->eigenvectors, n);
    for (i = 0; i < n; ++i) {
        out_result->eigenvalues.data[i] *= matrix_scale;
        if (!isfinite(out_result->eigenvalues.data[i])) {
            lmmc_vec_destroy(&out_result->eigenvalues);
            lmmc_mat_destroy(&out_result->eigenvectors);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}

void lmmc_eigen_sym_result_destroy(lmmc_eigen_sym_result_t *result) {
    if (!result) return;
    lmmc_vec_destroy(&result->eigenvalues);
    lmmc_mat_destroy(&result->eigenvectors);
}

