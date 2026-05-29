/**
 * @file eigen.c
 * @brief 特征值、奇异值、伪逆与条件数算法实现。
 */
#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/linear_algebra.h"

#define EIGEN_MAX_ITER 30
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])

/* Forward declarations for helpers defined later in this file */
static void householder_make(lmmc_real_t *x, size_t len, lmmc_real_t *tau_out,
                             lmmc_real_t *beta_out);
static void householder_apply_left(lmmc_mat_t *M, size_t i0, size_t len,
                                   size_t j0, size_t j1,
                                   const lmmc_real_t *v, lmmc_real_t tau);
static void householder_apply_right(lmmc_mat_t *M, size_t i0, size_t i1,
                                    size_t j0, size_t len,
                                    const lmmc_real_t *v, lmmc_real_t tau);
static void quad_solve(lmmc_real_t p, lmmc_real_t q,
                       lmmc_real_t *re1, lmmc_real_t *im1,
                       lmmc_real_t *re2, lmmc_real_t *im2);

static lmmc_status_t householder_tridiag(lmmc_mat_t *a, lmmc_real_t *diag, lmmc_real_t *offdiag, lmmc_real_t *taus) {
    size_t n = a->rows;
    size_t k, i, j;
    if (taus) { for (i = 0; i + 1 < n; i++) taus[i] = 0.0; }
    if (n == 1) { diag[0] = MAT_ELEM(a, 0, 0); return LMMC_STATUS_OK; }
    for (k = 0; k < n - 1; k++) {
        lmmc_real_t sigma = 0.0;
        for (i = k + 2; i < n; i++) sigma += MAT_ELEM(a, i, k) * MAT_ELEM(a, i, k);
        lmmc_real_t alpha_val = MAT_ELEM(a, k + 1, k);
        if (sigma == 0.0) {
            offdiag[k] = alpha_val; diag[k] = MAT_ELEM(a, k, k);
            if (taus) taus[k] = 0.0;

            for (i = k + 2; i < n; i++) MAT_ELEM(a, i, k) = 0.0;
            continue;
        }
        lmmc_real_t norm_x = sqrt(alpha_val * alpha_val + sigma);
        lmmc_real_t beta = (alpha_val >= 0.0) ? -norm_x : norm_x;
        lmmc_real_t v_first = alpha_val - beta;
        lmmc_real_t tau = -v_first / beta;
        lmmc_real_t inv_v_first = 1.0 / v_first;
        for (i = k + 2; i < n; i++) MAT_ELEM(a, i, k) *= inv_v_first;
        offdiag[k] = beta;
        if (taus) taus[k] = tau;
        lmmc_real_t *w = (lmmc_real_t *)lmmc_alloc((n - k - 1) * sizeof(lmmc_real_t));
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
            r = sqrt(g * g + 1.0);
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
    lmmc_status_t status; size_t n;
    if (!a || !out_result || !a->data) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols || a->rows == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    n = a->rows;
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
    lmmc_real_t *offdiag = (lmmc_real_t *)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (!offdiag) { lmmc_mat_destroy(&work); lmmc_vec_destroy(&out_result->eigenvalues); lmmc_mat_destroy(&out_result->eigenvectors); return LMMC_STATUS_ALLOCATION_FAILED; }
    memset(offdiag, 0, n * sizeof(lmmc_real_t));
    lmmc_real_t *taus = (lmmc_real_t *)lmmc_alloc((n > 1 ? n - 1 : 1) * sizeof(lmmc_real_t));
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
    return LMMC_STATUS_OK;
}

void lmmc_eigen_sym_result_destroy(lmmc_eigen_sym_result_t *result) {
    if (!result) return;
    lmmc_vec_destroy(&result->eigenvalues);
    lmmc_mat_destroy(&result->eigenvectors);
}


/* Forward declarations for Householder helpers (defined later in SVD section) */
static void householder_make(lmmc_real_t *x, size_t len, lmmc_real_t *tau_out,
                             lmmc_real_t *beta_out);
static void householder_apply_left(lmmc_mat_t *M, size_t i0, size_t len,
                                   size_t j0, size_t j1,
                                   const lmmc_real_t *v, lmmc_real_t tau);
static void householder_apply_right(lmmc_mat_t *M, size_t i0, size_t i1,
                                    size_t j0, size_t len,
                                    const lmmc_real_t *v, lmmc_real_t tau);

/* ===== Hessenberg Reduction and Francis Double-Shift QR ===== */

/**
 * @brief Reduce a general square matrix to upper Hessenberg form via
 *        Householder reflectors (reuses householder_make, householder_apply_left,
 *        householder_apply_right from the SVD code path).
 *
 * On entry, H contains the matrix A. On exit, H is overwritten with the
 * upper Hessenberg form and Q accumulates the orthogonal similarity
 * transformation such that A = Q * H * Q^T.
 */
static lmmc_status_t hessenberg_reduce(lmmc_mat_t *H, lmmc_mat_t *Q) {
    size_t n = H->rows;
    size_t k, i, j;
    lmmc_real_t *vbuf;

    /* Initialize Q = I */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(Q, i, j) = (i == j) ? 1.0 : 0.0;

    if (n <= 2) return LMMC_STATUS_OK;

    vbuf = (lmmc_real_t *)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (!vbuf) return LMMC_STATUS_ALLOCATION_FAILED;

    for (k = 0; k < n - 2; k++) {
        size_t len = n - k - 1;
        lmmc_real_t tau, beta;

        /* Extract column below diagonal */
        for (i = 0; i < len; i++)
            vbuf[i] = MAT_ELEM(H, k + 1 + i, k);

        householder_make(vbuf, len, &tau, &beta);

        /* Set subdiagonal entry */
        MAT_ELEM(H, k + 1, k) = beta;
        for (i = 1; i < len; i++)
            MAT_ELEM(H, k + 1 + i, k) = 0.0;

        /* Apply from left: H <- (I - tau*v*v^T) * H */
        householder_apply_left(H, k + 1, len, k + 1, n, vbuf, tau);

        /* Apply from right: H <- H * (I - tau*v*v^T) */
        householder_apply_right(H, 0, n, k + 1, len, vbuf, tau);

        /* Accumulate Q: Q <- Q * (I - tau*v*v^T) */
        householder_apply_right(Q, 0, n, k + 1, len, vbuf, tau);
    }

    lmmc_free(vbuf);
    return LMMC_STATUS_OK;
}


/**
 * @brief Implicit Francis double-shift QR iteration on an upper Hessenberg matrix.
 *
 * Operates on the matrix H which must be in upper Hessenberg form.
 * On exit, H is in real Schur form (quasi-upper-triangular with 1x1 and 2x2
 * diagonal blocks). Q is updated to accumulate the similarity transformations.
 *
 * @param H   Upper Hessenberg matrix (modified in place to real Schur form).
 * @param Q   Orthogonal matrix accumulating transformations.
 * @param nn  Dimension of the matrix.
 * @return LMMC_STATUS_OK on success, LMMC_STATUS_CONVERGENCE_FAILED if
 *         maximum iterations exceeded.
 */
static lmmc_status_t francis_qr_iteration(lmmc_mat_t *H, lmmc_mat_t *Q, size_t nn) {
    /*
     * Implicit double-shift QR (Francis) based on EISPACK hqr2.
     * Reduces upper Hessenberg H to real Schur form.
     */
    const lmmc_real_t eps = 2.2204460492503131e-16;
    const size_t max_iter = 30 * nn;
    size_t total_iter = 0;
    int n = (int)nn;
    int i, j, k, l, its, en;
    lmmc_real_t p, q, r, s, t, w, x, y, z, norm;
    int notlast;
    lmmc_real_t v[3];

    /* Compute norm for convergence testing */
    norm = 0.0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            norm += lmmc_abs(MAT_ELEM(H, i, j));
    if (norm == 0.0) return LMMC_STATUS_OK;

    en = n - 1;

    while (en >= 2) {
        its = 0;

        for (;;) {
            /* Look for single small sub-diagonal element */
            for (l = en; l > 0; l--) {
                s = lmmc_abs(MAT_ELEM(H, l - 1, l - 1)) +
                    lmmc_abs(MAT_ELEM(H, l, l));
                if (s == 0.0) s = norm;
                if (lmmc_abs(MAT_ELEM(H, l, l - 1)) <= eps * s) {
                    MAT_ELEM(H, l, l - 1) = 0.0;
                    break;
                }
            }

            x = MAT_ELEM(H, en, en);
            if (l == en) { en--; break; }

            y = MAT_ELEM(H, en - 1, en - 1);
            w = MAT_ELEM(H, en, en - 1) * MAT_ELEM(H, en - 1, en);
            if (l == en - 1) { en -= 2; break; }

            if (total_iter >= max_iter)
                return LMMC_STATUS_CONVERGENCE_FAILED;

            if (its == 10 || its == 20) {
                /* Exceptional shift */
                t = lmmc_abs(MAT_ELEM(H, en, en - 1)) +
                    lmmc_abs(MAT_ELEM(H, en - 1, en - 2));
                x = MAT_ELEM(H, en, en) + 1.5 * t;
                y = x;
                w = -0.4375 * t * t;
            }

            its++;
            total_iter++;

            /* Form shift from trailing 2x2 */
            s = x + y;
            t = x * y - w;

            /* First column of (H - s1*I)(H - s2*I) */
            p = MAT_ELEM(H, l, l) * (MAT_ELEM(H, l, l) - s) +
                MAT_ELEM(H, l, l + 1) * MAT_ELEM(H, l + 1, l) + t;
            q = MAT_ELEM(H, l + 1, l) *
                (MAT_ELEM(H, l, l) + MAT_ELEM(H, l + 1, l + 1) - s);
            r = MAT_ELEM(H, l + 1, l) * MAT_ELEM(H, l + 2, l + 1);

            /* Chase the bulge using Householder reflectors */
            for (k = l; k <= en - 1; k++) {
                lmmc_real_t tau, beta_val;
                notlast = (k != en - 1);
                size_t len = notlast ? 3 : 2;

                if (k != l) {
                    v[0] = MAT_ELEM(H, k, k - 1);
                    v[1] = MAT_ELEM(H, k + 1, k - 1);
                    if (notlast) v[2] = MAT_ELEM(H, k + 2, k - 1);
                } else {
                    v[0] = p; v[1] = q;
                    if (notlast) v[2] = r;
                }

                householder_make(v, len, &tau, &beta_val);

                if (k != l) {
                    MAT_ELEM(H, k, k - 1) = beta_val;
                    MAT_ELEM(H, k + 1, k - 1) = 0.0;
                    if (notlast) MAT_ELEM(H, k + 2, k - 1) = 0.0;
                } else if (l != 0) {
                    /* Flip sign of existing subdiagonal for implicit shift */
                    MAT_ELEM(H, k, k - 1) = -MAT_ELEM(H, k, k - 1);
                }

                /* Apply from left: H[k:k+len, k:n] */
                householder_apply_left(H, (size_t)k, len, (size_t)k, (size_t)n, v, tau);

                /* Apply from right: H[0:min(k+len+1,en+1), k:k+len] */
                {
                    size_t nr = (size_t)((int)(k + len + 1) <= en + 1 ?
                                         (int)(k + len + 1) : en + 1);
                    householder_apply_right(H, 0, nr, (size_t)k, len, v, tau);
                }

                /* Accumulate in Q */
                householder_apply_right(Q, 0, (size_t)n, (size_t)k, len, v, tau);
            }
        }
    }

    return LMMC_STATUS_OK;
}


/**
 * @brief Extract eigenvalues from the real Schur form.
 *
 * 1x1 diagonal blocks give real eigenvalues.
 * 2x2 diagonal blocks give complex-conjugate pairs.
 */
static void extract_eigenvalues_from_schur(const lmmc_mat_t *H, size_t n,
                                           lmmc_real_t *re, lmmc_real_t *im) {
    size_t i = 0;
    while (i < n) {
        if (i + 1 == n || MAT_ELEM(H, i + 1, i) == 0.0) {
            /* 1x1 block: real eigenvalue */
            re[i] = MAT_ELEM(H, i, i);
            im[i] = 0.0;
            i++;
        } else {
            /* 2x2 block: complex-conjugate pair */
            lmmc_real_t a11 = MAT_ELEM(H, i, i);
            lmmc_real_t a12 = MAT_ELEM(H, i, i + 1);
            lmmc_real_t a21 = MAT_ELEM(H, i + 1, i);
            lmmc_real_t a22 = MAT_ELEM(H, i + 1, i + 1);
            lmmc_real_t tr = a11 + a22;
            lmmc_real_t det = a11 * a22 - a12 * a21;
            lmmc_real_t disc = tr * tr - 4.0 * det;
            if (disc >= 0.0) {
                lmmc_real_t sq = sqrt(disc);
                lmmc_real_t lam1 = (tr + sq) * 0.5;
                lmmc_real_t lam2 = (tr - sq) * 0.5;
                re[i] = lam1; im[i] = 0.0;
                re[i + 1] = lam2; im[i + 1] = 0.0;
            } else {
                lmmc_real_t sq = sqrt(-disc) * 0.5;
                re[i] = tr * 0.5;     im[i] = sq;
                re[i + 1] = tr * 0.5; im[i + 1] = -sq;
            }
            i += 2;
        }
    }
}

lmmc_status_t lmmc_eigen_general(const lmmc_mat_t *a, lmmc_eigen_gen_result_t *out_result)
{
    lmmc_status_t status;
    if (!a || !out_result || !a->data) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols || a->rows == 0) return LMMC_STATUS_INVALID_ARGUMENT;

    size_t n = a->rows;
    size_t i, j;

    /* Allocate output vectors */
    status = lmmc_vec_create(n, &out_result->real_parts);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(n, &out_result->imag_parts);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&out_result->real_parts);
        return status;
    }

    if (n == 1) {
        out_result->real_parts.data[0] = MAT_ELEM(a, 0, 0);
        out_result->imag_parts.data[0] = 0.0;
        return LMMC_STATUS_OK;
    }

    if (n == 2) {
        lmmc_real_t a00 = MAT_ELEM(a, 0, 0), a01 = MAT_ELEM(a, 0, 1);
        lmmc_real_t a10 = MAT_ELEM(a, 1, 0), a11 = MAT_ELEM(a, 1, 1);
        lmmc_real_t tr = a00 + a11;
        lmmc_real_t det = a00 * a11 - a01 * a10;
        lmmc_real_t disc = tr * tr - 4.0 * det;
        if (disc >= 0.0) {
            lmmc_real_t sq = sqrt(disc);
            out_result->real_parts.data[0] = (tr + sq) * 0.5;
            out_result->imag_parts.data[0] = 0.0;
            out_result->real_parts.data[1] = (tr - sq) * 0.5;
            out_result->imag_parts.data[1] = 0.0;
        } else {
            lmmc_real_t sq = sqrt(-disc) * 0.5;
            out_result->real_parts.data[0] = tr * 0.5;
            out_result->imag_parts.data[0] = sq;
            out_result->real_parts.data[1] = tr * 0.5;
            out_result->imag_parts.data[1] = -sq;
        }
        return LMMC_STATUS_OK;
    }

    /* Allocate workspace: H (copy of A) and Q (orthogonal accumulator) */
    lmmc_mat_t H, Q_mat;
    status = lmmc_mat_create(n, n, &H);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&out_result->real_parts);
        lmmc_vec_destroy(&out_result->imag_parts);
        return status;
    }
    status = lmmc_mat_create(n, n, &Q_mat);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&H);
        lmmc_vec_destroy(&out_result->real_parts);
        lmmc_vec_destroy(&out_result->imag_parts);
        return status;
    }

    /* Copy A into H */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&H, i, j) = MAT_ELEM(a, i, j);

    /* Step 1: Reduce to upper Hessenberg form */
    status = hessenberg_reduce(&H, &Q_mat);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&H);
        lmmc_mat_destroy(&Q_mat);
        lmmc_vec_destroy(&out_result->real_parts);
        lmmc_vec_destroy(&out_result->imag_parts);
        return status;
    }

    /* Step 2: Francis double-shift QR iteration */
    status = francis_qr_iteration(&H, &Q_mat, n);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&H);
        lmmc_mat_destroy(&Q_mat);
        lmmc_vec_destroy(&out_result->real_parts);
        lmmc_vec_destroy(&out_result->imag_parts);
        return status;
    }

    /* Step 3: Extract eigenvalues from real Schur form */
    extract_eigenvalues_from_schur(&H, n,
                                   out_result->real_parts.data,
                                   out_result->imag_parts.data);

    lmmc_mat_destroy(&H);
    lmmc_mat_destroy(&Q_mat);
    return LMMC_STATUS_OK;
}


static void hqr_eig2(lmmc_real_t a, lmmc_real_t b,
                     lmmc_real_t c, lmmc_real_t d,
                     lmmc_real_t *re_p, lmmc_real_t *im_p,
                     lmmc_real_t *re_q, lmmc_real_t *im_q)
{
    lmmc_real_t tr = a + d;
    lmmc_real_t det = a * d - b * c;
    lmmc_real_t disc = tr * tr - 4.0 * det;
    if (disc >= 0.0) {
        lmmc_real_t s = sqrt(disc);

        lmmc_real_t lam1 = (tr >= 0.0) ? (tr + s) * 0.5 : (tr - s) * 0.5;
        lmmc_real_t lam2 = (lam1 != 0.0) ? (det / lam1) : ((tr - s) * 0.5);
        *re_p = lam1; *im_p = 0.0;
        *re_q = lam2; *im_q = 0.0;
    } else {
        lmmc_real_t s = sqrt(-disc) * 0.5;
        *re_p = tr * 0.5; *im_p =  s;
        *re_q = tr * 0.5; *im_q = -s;
    }
}

void lmmc_eigen_gen_result_destroy(lmmc_eigen_gen_result_t *result) {
    if (!result) return; lmmc_vec_destroy(&result->real_parts); lmmc_vec_destroy(&result->imag_parts);
}

/* ===== Eigenvector Computation via Inverse Iteration ===== */

/**
 * @brief Compute eigenvectors via inverse iteration with Wilkinson shift.
 *
 * For each eigenvalue mu, solve (A - mu*I) * x = x_prev iteratively.
 * For complex-conjugate pairs, use real arithmetic with the 2x2 block structure.
 */

void lmmc_eigen_gen_full_result_destroy(lmmc_eigen_gen_full_result_t *result) {
    if (!result) return;
    lmmc_vec_destroy(&result->real_parts);
    lmmc_vec_destroy(&result->imag_parts);
    lmmc_mat_destroy(&result->vectors_real);
    lmmc_mat_destroy(&result->vectors_imag);
}

/**
 * @brief Perform inverse iteration for a real eigenvalue.
 *
 * Solves (A - mu*I) x = b repeatedly to converge to the eigenvector.
 * Uses LU decomposition with partial pivoting.
 * Applies a small perturbation to the shift to avoid exact singularity.
 */
static lmmc_status_t inverse_iteration_real(
    const lmmc_mat_t *A, size_t n, lmmc_real_t mu,
    lmmc_real_t *vec_out)
{
    lmmc_status_t status;
    size_t i, j, iter;
    const size_t max_iter = 20;
    const lmmc_real_t eps = 2.2204460492503131e-16;

    /* Allocate shifted matrix and workspace */
    lmmc_mat_t shifted;
    status = lmmc_mat_create(n, n, &shifted);
    if (status != LMMC_STATUS_OK) return status;

    size_t *pivots = (size_t *)lmmc_alloc(n * sizeof(size_t));
    if (!pivots) { lmmc_mat_destroy(&shifted); return LMMC_STATUS_ALLOCATION_FAILED; }

    lmmc_vec_t b_vec, x_vec;
    status = lmmc_vec_create(n, &b_vec);
    if (status != LMMC_STATUS_OK) { lmmc_free(pivots); lmmc_mat_destroy(&shifted); return status; }
    status = lmmc_vec_create(n, &x_vec);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&b_vec); lmmc_free(pivots); lmmc_mat_destroy(&shifted); return status; }

    /* Compute ||A||_F for scaling the perturbation */
    lmmc_real_t norm_A = 0.0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            norm_A += MAT_ELEM(A, i, j) * MAT_ELEM(A, i, j);
    norm_A = sqrt(norm_A);
    if (norm_A < 1.0) norm_A = 1.0;

    /* Apply Wilkinson-style perturbation to avoid exact singularity.
     * The perturbation is small enough not to affect convergence. */
    lmmc_real_t perturb = eps * norm_A * (lmmc_real_t)n;
    lmmc_real_t shift = mu + perturb;

    /* Form (A - shift*I) */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&shifted, i, j) = MAT_ELEM(A, i, j) - ((i == j) ? shift : 0.0);

    /* LU decompose the shifted matrix */
    status = lmmc_lu_decompose_inplace(&shifted, pivots, NULL);
    if (status == LMMC_STATUS_SINGULAR_MATRIX) {
        /* Try a larger perturbation */
        perturb = sqrt(eps) * norm_A;
        shift = mu + perturb;
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                MAT_ELEM(&shifted, i, j) = MAT_ELEM(A, i, j) - ((i == j) ? shift : 0.0);
        status = lmmc_lu_decompose_inplace(&shifted, pivots, NULL);
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x_vec);
        lmmc_vec_destroy(&b_vec);
        lmmc_free(pivots);
        lmmc_mat_destroy(&shifted);
        return status;
    }

    /* Initialize starting vector with entries that avoid null spaces */
    for (i = 0; i < n; i++)
        b_vec.data[i] = ((i % 2 == 0) ? 1.0 : -1.0) / (lmmc_real_t)(i + 1);

    /* Normalize initial vector */
    {
        lmmc_real_t nrm = 0.0;
        for (i = 0; i < n; i++) nrm += b_vec.data[i] * b_vec.data[i];
        nrm = sqrt(nrm);
        if (nrm > 0.0) for (i = 0; i < n; i++) b_vec.data[i] /= nrm;
    }

    /* Inverse iteration loop */
    for (iter = 0; iter < max_iter; iter++) {
        status = lmmc_lu_solve(&shifted, pivots, &b_vec, &x_vec);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(&x_vec);
            lmmc_vec_destroy(&b_vec);
            lmmc_free(pivots);
            lmmc_mat_destroy(&shifted);
            return status;
        }

        /* Normalize */
        lmmc_real_t nrm = 0.0;
        for (i = 0; i < n; i++) nrm += x_vec.data[i] * x_vec.data[i];
        nrm = sqrt(nrm);
        if (nrm == 0.0) nrm = 1.0;
        for (i = 0; i < n; i++) x_vec.data[i] /= nrm;

        /* Copy x to b for next iteration */
        for (i = 0; i < n; i++) b_vec.data[i] = x_vec.data[i];
    }

    /* Copy result */
    for (i = 0; i < n; i++) vec_out[i] = x_vec.data[i];

    lmmc_vec_destroy(&x_vec);
    lmmc_vec_destroy(&b_vec);
    lmmc_free(pivots);
    lmmc_mat_destroy(&shifted);
    return LMMC_STATUS_OK;
}

/**
 * @brief Perform inverse iteration for a complex-conjugate eigenvalue pair.
 *
 * For eigenvalue mu = alpha + i*beta, solves the real 2n x 2n system:
 *   [A - alpha*I,  beta*I ] [x_re]   [b_re]
 *   [-beta*I,  A - alpha*I] [x_im] = [b_im]
 * to find the real and imaginary parts of the eigenvector.
 */
static lmmc_status_t inverse_iteration_complex(
    const lmmc_mat_t *A, size_t n, lmmc_real_t alpha, lmmc_real_t beta,
    lmmc_real_t *vec_real_out, lmmc_real_t *vec_imag_out)
{
    lmmc_status_t status;
    size_t i, j, k, iter;
    const size_t max_iter = 20;
    const lmmc_real_t eps = 2.2204460492503131e-16;

    /* Compute norm of A */
    lmmc_real_t norm_A = 0.0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            norm_A += MAT_ELEM(A, i, j) * MAT_ELEM(A, i, j);
    norm_A = sqrt(norm_A);
    if (norm_A < 1.0) norm_A = 1.0;

    /*
     * For complex eigenvalue mu = alpha + i*beta, we solve the 2n x 2n real system:
     *   [A - alpha*I,  beta*I ] [x_re]   [b_re]
     *   [-beta*I,  A - alpha*I] [x_im] = [b_im]
     *
     * Apply a small perturbation to alpha to avoid singularity.
     */
    lmmc_real_t perturb = eps * norm_A * (lmmc_real_t)n;
    lmmc_real_t alpha_p = alpha + perturb;

    lmmc_mat_t big;
    size_t n2 = 2 * n;
    status = lmmc_mat_create(n2, n2, &big);
    if (status != LMMC_STATUS_OK) return status;

    size_t *pivots = (size_t *)lmmc_alloc(n2 * sizeof(size_t));
    if (!pivots) { lmmc_mat_destroy(&big); return LMMC_STATUS_ALLOCATION_FAILED; }

    lmmc_vec_t b_vec, x_vec;
    status = lmmc_vec_create(n2, &b_vec);
    if (status != LMMC_STATUS_OK) { lmmc_free(pivots); lmmc_mat_destroy(&big); return status; }
    status = lmmc_vec_create(n2, &x_vec);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&b_vec); lmmc_free(pivots); lmmc_mat_destroy(&big); return status; }

    /* Fill the 2n x 2n block matrix */
    for (i = 0; i < n2; i++)
        for (j = 0; j < n2; j++)
            MAT_ELEM(&big, i, j) = 0.0;

    /* Top-left block: A - alpha_p*I */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&big, i, j) = MAT_ELEM(A, i, j) - ((i == j) ? alpha_p : 0.0);

    /* Top-right block: beta*I */
    for (i = 0; i < n; i++)
        MAT_ELEM(&big, i, n + i) = beta;

    /* Bottom-left block: -beta*I */
    for (i = 0; i < n; i++)
        MAT_ELEM(&big, n + i, i) = -beta;

    /* Bottom-right block: A - alpha_p*I */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&big, n + i, n + j) = MAT_ELEM(A, i, j) - ((i == j) ? alpha_p : 0.0);

    /* LU decompose */
    status = lmmc_lu_decompose_inplace(&big, pivots, NULL);
    if (status == LMMC_STATUS_SINGULAR_MATRIX) {
        /* Try a larger perturbation */
        alpha_p = alpha + sqrt(eps) * norm_A;
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++) {
                lmmc_real_t aij = MAT_ELEM(A, i, j) - ((i == j) ? alpha_p : 0.0);
                MAT_ELEM(&big, i, j) = aij;
                MAT_ELEM(&big, n + i, n + j) = aij;
            }
        for (i = 0; i < n; i++) {
            MAT_ELEM(&big, i, n + i) = beta;
            MAT_ELEM(&big, n + i, i) = -beta;
        }
        status = lmmc_lu_decompose_inplace(&big, pivots, NULL);
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x_vec);
        lmmc_vec_destroy(&b_vec);
        lmmc_free(pivots);
        lmmc_mat_destroy(&big);
        return status;
    }

    /* Initialize starting vector */
    for (i = 0; i < n; i++) {
        b_vec.data[i] = ((i % 2 == 0) ? 1.0 : -1.0) / (lmmc_real_t)(i + 1);
        b_vec.data[n + i] = 1.0 / (lmmc_real_t)(n + i + 1);
    }

    /* Normalize */
    {
        lmmc_real_t nrm = 0.0;
        for (i = 0; i < n2; i++) nrm += b_vec.data[i] * b_vec.data[i];
        nrm = sqrt(nrm);
        if (nrm > 0.0) for (i = 0; i < n2; i++) b_vec.data[i] /= nrm;
    }

    /* Inverse iteration loop */
    for (iter = 0; iter < max_iter; iter++) {
        status = lmmc_lu_solve(&big, pivots, &b_vec, &x_vec);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(&x_vec);
            lmmc_vec_destroy(&b_vec);
            lmmc_free(pivots);
            lmmc_mat_destroy(&big);
            return status;
        }

        /* Normalize */
        lmmc_real_t nrm = 0.0;
        for (i = 0; i < n2; i++) nrm += x_vec.data[i] * x_vec.data[i];
        nrm = sqrt(nrm);
        if (nrm == 0.0) nrm = 1.0;
        for (i = 0; i < n2; i++) x_vec.data[i] /= nrm;

        /* Copy x to b for next iteration */
        for (i = 0; i < n2; i++) b_vec.data[i] = x_vec.data[i];
    }

    /* Extract real and imaginary parts */
    for (i = 0; i < n; i++) {
        vec_real_out[i] = x_vec.data[i];
        vec_imag_out[i] = x_vec.data[n + i];
    }

    lmmc_vec_destroy(&x_vec);
    lmmc_vec_destroy(&b_vec);
    lmmc_free(pivots);
    lmmc_mat_destroy(&big);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_eigen_general_full(
    const lmmc_mat_t *a,
    lmmc_eigen_gen_full_result_t *out_result)
{
    lmmc_status_t status;
    if (!a || !out_result || !a->data) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols || a->rows == 0) return LMMC_STATUS_INVALID_ARGUMENT;

    size_t n = a->rows;
    size_t i, j;

    /* Initialize output to zeros so cleanup is safe on partial failure */
    memset(out_result, 0, sizeof(*out_result));

    /* Allocate output vectors and matrices */
    status = lmmc_vec_create(n, &out_result->real_parts);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(n, &out_result->imag_parts);
    if (status != LMMC_STATUS_OK) goto fail;
    status = lmmc_mat_create(n, n, &out_result->vectors_real);
    if (status != LMMC_STATUS_OK) goto fail;
    status = lmmc_mat_create(n, n, &out_result->vectors_imag);
    if (status != LMMC_STATUS_OK) goto fail;

    /* Handle trivial cases */
    if (n == 1) {
        out_result->real_parts.data[0] = MAT_ELEM(a, 0, 0);
        out_result->imag_parts.data[0] = 0.0;
        MAT_ELEM(&out_result->vectors_real, 0, 0) = 1.0;
        MAT_ELEM(&out_result->vectors_imag, 0, 0) = 0.0;
        return LMMC_STATUS_OK;
    }

    if (n == 2) {
        lmmc_real_t a00 = MAT_ELEM(a, 0, 0), a01 = MAT_ELEM(a, 0, 1);
        lmmc_real_t a10 = MAT_ELEM(a, 1, 0), a11 = MAT_ELEM(a, 1, 1);
        lmmc_real_t tr = a00 + a11;
        lmmc_real_t det = a00 * a11 - a01 * a10;
        lmmc_real_t disc = tr * tr - 4.0 * det;

        if (disc >= 0.0) {
            lmmc_real_t sq = sqrt(disc);
            lmmc_real_t lam1 = (tr + sq) * 0.5;
            lmmc_real_t lam2 = (tr - sq) * 0.5;
            out_result->real_parts.data[0] = lam1;
            out_result->imag_parts.data[0] = 0.0;
            out_result->real_parts.data[1] = lam2;
            out_result->imag_parts.data[1] = 0.0;

            /* Compute eigenvectors for 2x2 real case.
             * For eigenvalue lam, solve (A - lam*I)*v = 0. */
            for (i = 0; i < 2; i++) {
                lmmc_real_t lam = out_result->real_parts.data[i];
                lmmc_real_t v0, v1, nrm;
                /* Row 0: (a00 - lam)*v0 + a01*v1 = 0 */
                /* Row 1: a10*v0 + (a11 - lam)*v1 = 0 */
                lmmc_real_t r0_diag = lmmc_abs(a00 - lam);
                lmmc_real_t r1_diag = lmmc_abs(a11 - lam);
                lmmc_real_t r0_off = lmmc_abs(a01);
                lmmc_real_t r1_off = lmmc_abs(a10);

                if (r0_off > 1e-15 || r0_diag > 1e-15) {
                    /* Use row 0: (a00 - lam)*v0 + a01*v1 = 0 */
                    if (r0_off >= r0_diag) {
                        /* v0 = -a01, v1 = a00 - lam */
                        v0 = -a01;
                        v1 = a00 - lam;
                    } else {
                        /* v0 = a01, v1 = lam - a00 ... but if a01=0, use row 1 */
                        if (r0_off < 1e-15) {
                            /* a01 = 0, so row 0 gives (a00-lam)*v0 = 0 */
                            /* If a00 != lam, then v0 = 0 */
                            if (r0_diag > 1e-15) {
                                v0 = 0.0; v1 = 1.0;
                            } else {
                                v0 = 1.0; v1 = 0.0;
                            }
                        } else {
                            v0 = -a01;
                            v1 = a00 - lam;
                        }
                    }
                } else if (r1_off > 1e-15 || r1_diag > 1e-15) {
                    /* Use row 1: a10*v0 + (a11 - lam)*v1 = 0 */
                    if (r1_diag > 1e-15) {
                        v0 = 1.0; v1 = -a10 / (a11 - lam);
                    } else {
                        v0 = 0.0; v1 = 1.0;
                    }
                } else {
                    /* Both rows are zero - any vector is an eigenvector */
                    v0 = 1.0; v1 = 0.0;
                }
                nrm = sqrt(v0 * v0 + v1 * v1);
                if (nrm < 1e-15) { v0 = 1.0; v1 = 0.0; nrm = 1.0; }
                MAT_ELEM(&out_result->vectors_real, 0, i) = v0 / nrm;
                MAT_ELEM(&out_result->vectors_real, 1, i) = v1 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 0, i) = 0.0;
                MAT_ELEM(&out_result->vectors_imag, 1, i) = 0.0;
            }
        } else {
            lmmc_real_t sq = sqrt(-disc) * 0.5;
            out_result->real_parts.data[0] = tr * 0.5;
            out_result->imag_parts.data[0] = sq;
            out_result->real_parts.data[1] = tr * 0.5;
            out_result->imag_parts.data[1] = -sq;

            /* Complex-conjugate pair: eigenvector for lam = (tr/2) + i*sq
             * From row 0: (a00 - tr/2 - i*sq)*v0 + a01*v1 = 0
             * From row 1: a10*v0 + (a11 - tr/2 - i*sq)*v1 = 0 */
            lmmc_real_t re_part = tr * 0.5 - a00;
            if (lmmc_abs(a01) > lmmc_abs(a10)) {
                /* v = (a01, re_part + i*sq) */
                lmmc_real_t vr0 = a01;
                lmmc_real_t vr1 = re_part;
                lmmc_real_t vi0 = 0.0;
                lmmc_real_t vi1 = sq;
                lmmc_real_t nrm = sqrt(vr0*vr0 + vr1*vr1 + vi0*vi0 + vi1*vi1);
                if (nrm == 0.0) nrm = 1.0;
                MAT_ELEM(&out_result->vectors_real, 0, 0) = vr0 / nrm;
                MAT_ELEM(&out_result->vectors_real, 1, 0) = vr1 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 0, 0) = vi0 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 1, 0) = vi1 / nrm;
            } else {
                /* v = (tr/2 - a11 + i*sq, a10) */
                lmmc_real_t vr0 = tr * 0.5 - a11;
                lmmc_real_t vr1 = a10;
                lmmc_real_t vi0 = sq;
                lmmc_real_t vi1 = 0.0;
                lmmc_real_t nrm = sqrt(vr0*vr0 + vr1*vr1 + vi0*vi0 + vi1*vi1);
                if (nrm == 0.0) nrm = 1.0;
                MAT_ELEM(&out_result->vectors_real, 0, 0) = vr0 / nrm;
                MAT_ELEM(&out_result->vectors_real, 1, 0) = vr1 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 0, 0) = vi0 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 1, 0) = vi1 / nrm;
            }
            /* Conjugate column */
            MAT_ELEM(&out_result->vectors_real, 0, 1) = MAT_ELEM(&out_result->vectors_real, 0, 0);
            MAT_ELEM(&out_result->vectors_real, 1, 1) = MAT_ELEM(&out_result->vectors_real, 1, 0);
            MAT_ELEM(&out_result->vectors_imag, 0, 1) = -MAT_ELEM(&out_result->vectors_imag, 0, 0);
            MAT_ELEM(&out_result->vectors_imag, 1, 1) = -MAT_ELEM(&out_result->vectors_imag, 1, 0);
        }
        return LMMC_STATUS_OK;
    }

    /* General case: use Hessenberg + Francis QR to get eigenvalues,
     * then inverse iteration for eigenvectors */
    {
        lmmc_mat_t H, Q_mat;
        status = lmmc_mat_create(n, n, &H);
        if (status != LMMC_STATUS_OK) goto fail;
        status = lmmc_mat_create(n, n, &Q_mat);
        if (status != LMMC_STATUS_OK) { lmmc_mat_destroy(&H); goto fail; }

        /* Copy A into H */
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                MAT_ELEM(&H, i, j) = MAT_ELEM(a, i, j);

        /* Step 1: Reduce to upper Hessenberg form */
        status = hessenberg_reduce(&H, &Q_mat);
        if (status != LMMC_STATUS_OK) {
            lmmc_mat_destroy(&H);
            lmmc_mat_destroy(&Q_mat);
            goto fail;
        }

        /* Step 2: Francis double-shift QR iteration */
        status = francis_qr_iteration(&H, &Q_mat, n);
        if (status != LMMC_STATUS_OK) {
            lmmc_mat_destroy(&H);
            lmmc_mat_destroy(&Q_mat);
            goto fail;
        }

        /* Step 3: Extract eigenvalues from real Schur form */
        extract_eigenvalues_from_schur(&H, n,
                                       out_result->real_parts.data,
                                       out_result->imag_parts.data);

        lmmc_mat_destroy(&H);
        lmmc_mat_destroy(&Q_mat);
    }

    /* Step 4: Compute eigenvectors via inverse iteration */
    /* Initialize vectors_imag to zero */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&out_result->vectors_imag, i, j) = 0.0;

    i = 0;
    while (i < n) {
        if (out_result->imag_parts.data[i] == 0.0) {
            /* Real eigenvalue: inverse iteration for real eigenvector */
            lmmc_real_t *col = (lmmc_real_t *)lmmc_alloc(n * sizeof(lmmc_real_t));
            if (!col) { status = LMMC_STATUS_ALLOCATION_FAILED; goto fail; }

            status = inverse_iteration_real(a, n, out_result->real_parts.data[i], col);
            if (status != LMMC_STATUS_OK) {
                lmmc_free(col);
                goto fail;
            }

            /* Store in vectors_real column i, vectors_imag column i = 0 */
            for (j = 0; j < n; j++) {
                MAT_ELEM(&out_result->vectors_real, j, i) = col[j];
                MAT_ELEM(&out_result->vectors_imag, j, i) = 0.0;
            }
            lmmc_free(col);
            i++;
        } else {
            /* Complex-conjugate pair: eigenvalues at i and i+1 */
            lmmc_real_t alpha_val = out_result->real_parts.data[i];
            lmmc_real_t beta_val = out_result->imag_parts.data[i];

            lmmc_real_t *vr = (lmmc_real_t *)lmmc_alloc(n * sizeof(lmmc_real_t));
            lmmc_real_t *vi = (lmmc_real_t *)lmmc_alloc(n * sizeof(lmmc_real_t));
            if (!vr || !vi) {
                if (vr) lmmc_free(vr);
                if (vi) lmmc_free(vi);
                status = LMMC_STATUS_ALLOCATION_FAILED;
                goto fail;
            }

            status = inverse_iteration_complex(a, n, alpha_val, beta_val, vr, vi);
            if (status != LMMC_STATUS_OK) {
                lmmc_free(vr);
                lmmc_free(vi);
                goto fail;
            }

            /* Store: column i gets (vr, vi), column i+1 gets (vr, -vi) */
            for (j = 0; j < n; j++) {
                MAT_ELEM(&out_result->vectors_real, j, i) = vr[j];
                MAT_ELEM(&out_result->vectors_imag, j, i) = vi[j];
                MAT_ELEM(&out_result->vectors_real, j, i + 1) = vr[j];
                MAT_ELEM(&out_result->vectors_imag, j, i + 1) = -vi[j];
            }
            lmmc_free(vr);
            lmmc_free(vi);
            i += 2;
        }
    }

    return LMMC_STATUS_OK;

fail:
    lmmc_eigen_gen_full_result_destroy(out_result);
    memset(out_result, 0, sizeof(*out_result));
    return status;
}


static void givens_right(lmmc_mat_t *M, size_t rows, size_t p, size_t q,
                         lmmc_real_t c, lmmc_real_t s) {
    size_t i;
    for (i = 0; i < rows; i++) {
        lmmc_real_t mp = MAT_ELEM(M, i, p);
        lmmc_real_t mq = MAT_ELEM(M, i, q);
        MAT_ELEM(M, i, p) =  c * mp + s * mq;
        MAT_ELEM(M, i, q) = -s * mp + c * mq;
    }
}


static void householder_make(lmmc_real_t *x, size_t len, lmmc_real_t *tau_out,
                             lmmc_real_t *beta_out) {
    size_t i;
    if (len == 0) { *tau_out = 0.0; *beta_out = 0.0; return; }
    if (len == 1) { *tau_out = 0.0; *beta_out = x[0]; return; }
    lmmc_real_t sigma = 0.0;
    for (i = 1; i < len; i++) sigma += x[i] * x[i];
    lmmc_real_t alpha = x[0];
    if (sigma == 0.0) { *tau_out = 0.0; *beta_out = alpha; return; }
    lmmc_real_t mu = sqrt(alpha * alpha + sigma);

    lmmc_real_t beta = (alpha >= 0.0) ? -mu : mu;
    lmmc_real_t v0 = alpha - beta;
    *beta_out = beta;

    lmmc_real_t v0_sq = v0 * v0;
    *tau_out = 2.0 * v0_sq / (sigma + v0_sq);
    lmmc_real_t inv = 1.0 / v0;
    for (i = 1; i < len; i++) x[i] *= inv;
    x[0] = 1.0;
}


static void householder_apply_left(lmmc_mat_t *M, size_t i0, size_t len,
                                   size_t j0, size_t j1,
                                   const lmmc_real_t *v, lmmc_real_t tau) {
    size_t i, j;
    if (tau == 0.0) return;
    for (j = j0; j < j1; j++) {
        lmmc_real_t s = MAT_ELEM(M, i0, j);
        for (i = 1; i < len; i++) s += v[i] * MAT_ELEM(M, i0 + i, j);
        s *= tau;
        MAT_ELEM(M, i0, j) -= s;
        for (i = 1; i < len; i++) MAT_ELEM(M, i0 + i, j) -= s * v[i];
    }
}


static void householder_apply_right(lmmc_mat_t *M, size_t i0, size_t i1,
                                    size_t j0, size_t len,
                                    const lmmc_real_t *v, lmmc_real_t tau) {
    size_t i, j;
    if (tau == 0.0) return;
    for (i = i0; i < i1; i++) {
        lmmc_real_t s = MAT_ELEM(M, i, j0);
        for (j = 1; j < len; j++) s += v[j] * MAT_ELEM(M, i, j0 + j);
        s *= tau;
        MAT_ELEM(M, i, j0) -= s;
        for (j = 1; j < len; j++) MAT_ELEM(M, i, j0 + j) -= s * v[j];
    }
}


/* Solve x^2 + p*x + q = 0, returning roots as (re1,im1) and (re2,im2). */
static void quad_solve(lmmc_real_t p, lmmc_real_t q,
                       lmmc_real_t *re1, lmmc_real_t *im1,
                       lmmc_real_t *re2, lmmc_real_t *im2) {
    lmmc_real_t disc = p * p - 4.0 * q;
    if (disc >= 0.0) {
        lmmc_real_t sq = sqrt(disc);
        *re1 = (-p + sq) * 0.5;
        *im1 = 0.0;
        *re2 = (-p - sq) * 0.5;
        *im2 = 0.0;
    } else {
        lmmc_real_t sq = sqrt(-disc) * 0.5;
        *re1 = -p * 0.5;
        *im1 = sq;
        *re2 = -p * 0.5;
        *im2 = -sq;
    }
}


static void givens_compute(lmmc_real_t a, lmmc_real_t b,
                           lmmc_real_t *c, lmmc_real_t *s, lmmc_real_t *r) {
    if (b == 0.0) { *c = (a >= 0.0) ? 1.0 : -1.0; *s = 0.0; *r = lmmc_abs(a); }
    else if (a == 0.0) { *c = 0.0; *s = (b >= 0.0) ? 1.0 : -1.0; *r = lmmc_abs(b); }
    else {
        lmmc_real_t h = sqrt(a * a + b * b);
        *r = h;
        *c = a / h;
        *s = b / h;
    }
}


static void bidiag_qr_step(lmmc_real_t *d, lmmc_real_t *e,
                           size_t lo, size_t hi,
                           lmmc_mat_t *U, lmmc_mat_t *V) {

    lmmc_real_t f, g;
    {
        lmmc_real_t emm = (hi >= lo + 2) ? e[hi - 2] : 0.0;
        lmmc_real_t tnm = d[hi - 1] * d[hi - 1] + emm * emm;
        lmmc_real_t tnn = d[hi] * d[hi] + e[hi - 1] * e[hi - 1];
        lmmc_real_t tmn = d[hi - 1] * e[hi - 1];
        lmmc_real_t bb = (tnm - tnn) * 0.5;
        lmmc_real_t cc = tmn * tmn;
        lmmc_real_t shift;
        if (bb == 0.0 && cc == 0.0) shift = tnn;
        else {
            lmmc_real_t disc = sqrt(bb * bb + cc);
            lmmc_real_t denom = (bb >= 0.0) ? (bb + disc) : (bb - disc);
            shift = tnn - cc / denom;
        }
        f = d[lo] * d[lo] - shift;
        g = d[lo] * e[lo];
    }

    size_t i;
    for (i = lo; i < hi; i++) {
        lmmc_real_t c, s, r;

        givens_compute(f, g, &c, &s, &r);
        if (i > lo) e[i - 1] = r;
        f = c * d[i] + s * e[i];
        e[i] = c * e[i] - s * d[i];
        g = s * d[i + 1];
        d[i + 1] = c * d[i + 1];
        if (V) givens_right(V, V->rows, i, i + 1, c, s);


        givens_compute(f, g, &c, &s, &r);
        d[i] = r;
        f = c * e[i] + s * d[i + 1];
        d[i + 1] = c * d[i + 1] - s * e[i];
        if (i + 1 < hi) {
            g = s * e[i + 1];
            e[i + 1] = c * e[i + 1];
        }
        if (U) givens_right(U, U->rows, i, i + 1, c, s);
    }
    e[hi - 1] = f;
}


static lmmc_status_t bidiagonalize(const lmmc_mat_t *a,
                                   lmmc_real_t *d, lmmc_real_t *e,
                                   lmmc_mat_t *U, lmmc_mat_t *V) {
    size_t m = a->rows;
    size_t n = a->cols;
    size_t i, j, k;
    lmmc_status_t status;


    lmmc_mat_t W;
    status = lmmc_mat_create(m, n, &W);
    if (status != LMMC_STATUS_OK) return status;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&W, i, j) = MAT_ELEM(a, i, j);


    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++) MAT_ELEM(U, i, j) = (i == j) ? 1.0 : 0.0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) MAT_ELEM(V, i, j) = (i == j) ? 1.0 : 0.0;

    lmmc_real_t *vbuf = (lmmc_real_t *)lmmc_alloc(((m > n) ? m : n) * sizeof(lmmc_real_t));
    if (!vbuf) { lmmc_mat_destroy(&W); return LMMC_STATUS_ALLOCATION_FAILED; }

    for (k = 0; k < n; k++) {

        size_t len = m - k;
        for (i = 0; i < len; i++) vbuf[i] = MAT_ELEM(&W, k + i, k);
        lmmc_real_t tau, beta;
        householder_make(vbuf, len, &tau, &beta);
        d[k] = beta;

        if (k + 1 < n) {
            householder_apply_left(&W, k, len, k + 1, n, vbuf, tau);
        }

        if (tau != 0.0) {
            size_t r;
            for (r = 0; r < m; r++) {
                lmmc_real_t s = MAT_ELEM(U, r, k);
                for (i = 1; i < len; i++) s += vbuf[i] * MAT_ELEM(U, r, k + i);
                s *= tau;
                MAT_ELEM(U, r, k) -= s;
                for (i = 1; i < len; i++) MAT_ELEM(U, r, k + i) -= s * vbuf[i];
            }
        }


        if (k + 1 < n) {
            size_t rlen = n - k - 1;
            for (j = 0; j < rlen; j++) vbuf[j] = MAT_ELEM(&W, k, k + 1 + j);
            householder_make(vbuf, rlen, &tau, &beta);
            e[k] = beta;

            if (k + 1 < m) {
                householder_apply_right(&W, k + 1, m, k + 1, rlen, vbuf, tau);
            }

            if (tau != 0.0) {
                size_t r;
                for (r = 0; r < n; r++) {
                    lmmc_real_t s = MAT_ELEM(V, r, k + 1);
                    for (j = 1; j < rlen; j++) s += vbuf[j] * MAT_ELEM(V, r, k + 1 + j);
                    s *= tau;
                    MAT_ELEM(V, r, k + 1) -= s;
                    for (j = 1; j < rlen; j++) MAT_ELEM(V, r, k + 1 + j) -= s * vbuf[j];
                }
            }
        }
    }

    lmmc_free(vbuf);
    lmmc_mat_destroy(&W);
    return LMMC_STATUS_OK;
}


static lmmc_status_t svd_tall(const lmmc_mat_t *a,
                              lmmc_mat_t *out_U,
                              lmmc_vec_t *out_sigma,
                              lmmc_mat_t *out_Vt) {
    size_t m = a->rows;
    size_t n = a->cols;
    size_t i, j, k;
    lmmc_status_t status;
    int U_init = 0, sigma_init = 0, Vt_init = 0;
    lmmc_mat_t V; int V_init = 0;
    lmmc_real_t *d = NULL;
    lmmc_real_t *e = NULL;
    size_t *order = NULL;

    status = lmmc_mat_create(m, m, out_U);
    if (status != LMMC_STATUS_OK) goto fail;
    U_init = 1;
    status = lmmc_vec_create(n, out_sigma);
    if (status != LMMC_STATUS_OK) goto fail;
    sigma_init = 1;
    status = lmmc_mat_create(n, n, out_Vt);
    if (status != LMMC_STATUS_OK) goto fail;
    Vt_init = 1;
    status = lmmc_mat_create(n, n, &V);
    if (status != LMMC_STATUS_OK) goto fail;
    V_init = 1;

    d = (lmmc_real_t *)lmmc_alloc(n * sizeof(lmmc_real_t));
    e = (lmmc_real_t *)lmmc_alloc(((n > 0) ? n : 1) * sizeof(lmmc_real_t));
    if (!d || !e) { status = LMMC_STATUS_ALLOCATION_FAILED; goto fail; }
    for (i = 0; i < n; i++) { d[i] = 0.0; e[i] = 0.0; }


    status = bidiagonalize(a, d, e, out_U, &V);
    if (status != LMMC_STATUS_OK) goto fail;

    if (n >= 1) e[n - 1] = 0.0;


    {
        const size_t max_iter_total = 30 * n + 30;
        size_t iter = 0;
        size_t hi = (n == 0) ? 0 : n - 1;
        while (n >= 1) {

            while (hi > 0) {
                lmmc_real_t thr = LMMC_REAL_EPSILON *
                                  (lmmc_abs(d[hi - 1]) + lmmc_abs(d[hi]));
                if (lmmc_abs(e[hi - 1]) <= thr) {
                    e[hi - 1] = 0.0;
                    hi--;
                } else {
                    break;
                }
            }
            if (hi == 0) break;


            size_t lo = hi;
            while (lo > 0) {
                lmmc_real_t thr = LMMC_REAL_EPSILON *
                                  (lmmc_abs(d[lo - 1]) + lmmc_abs(d[lo]));
                if (lmmc_abs(e[lo - 1]) <= thr) {
                    e[lo - 1] = 0.0;
                    break;
                }
                lo--;
            }

            int handled = 0;
            for (k = lo; k < hi; k++) {
                if (d[k] == 0.0) {

                    lmmc_real_t f = e[k];
                    e[k] = 0.0;
                    for (i = k + 1; i <= hi; i++) {
                        lmmc_real_t c, s, r;
                        givens_compute(d[i], f, &c, &s, &r);
                        d[i] = r;
                        if (i < hi) {
                            f = -s * e[i];
                            e[i] = c * e[i];
                        }
                        givens_right(out_U, out_U->rows, k, i, c, s);
                    }
                    handled = 1;
                    break;
                }
            }
            if (handled) continue;

            if (iter++ > max_iter_total) {
                status = LMMC_STATUS_CONVERGENCE_FAILED;
                goto fail;
            }
            bidiag_qr_step(d, e, lo, hi, out_U, &V);
        }
    }


    for (i = 0; i < n; i++) {
        if (d[i] < 0.0) {
            d[i] = -d[i];
            for (k = 0; k < n; k++) MAT_ELEM(&V, k, i) = -MAT_ELEM(&V, k, i);
        }
    }


    order = (size_t *)lmmc_alloc(n * sizeof(size_t));
    if (!order) { status = LMMC_STATUS_ALLOCATION_FAILED; goto fail; }
    for (i = 0; i < n; i++) order[i] = i;

    for (i = 0; i + 1 < n; i++) {
        size_t mx = i;
        for (j = i + 1; j < n; j++) {
            if (d[order[j]] > d[order[mx]]) mx = j;
        }
        if (mx != i) { size_t tmp = order[i]; order[i] = order[mx]; order[mx] = tmp; }
    }


    for (i = 0; i < n; i++) out_sigma->data[i] = d[order[i]];


    for (i = 0; i < n; i++)
        for (k = 0; k < n; k++)
            MAT_ELEM(out_Vt, i, k) = MAT_ELEM(&V, k, order[i]);


    {
        lmmc_real_t *col_buf = (lmmc_real_t *)lmmc_alloc(m * n * sizeof(lmmc_real_t));
        if (!col_buf) { status = LMMC_STATUS_ALLOCATION_FAILED; goto fail; }
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++)
                col_buf[j * m + i] = MAT_ELEM(out_U, i, j);
        for (j = 0; j < n; j++) {
            size_t src = order[j];
            for (i = 0; i < m; i++)
                MAT_ELEM(out_U, i, j) = col_buf[src * m + i];
        }
        lmmc_free(col_buf);
    }

    lmmc_free(d);
    lmmc_free(e);
    lmmc_free(order);
    lmmc_mat_destroy(&V);
    return LMMC_STATUS_OK;

fail:
    if (order) lmmc_free(order);
    if (d) lmmc_free(d);
    if (e) lmmc_free(e);
    if (V_init) lmmc_mat_destroy(&V);
    if (Vt_init) lmmc_mat_destroy(out_Vt);
    if (sigma_init) lmmc_vec_destroy(out_sigma);
    if (U_init) lmmc_mat_destroy(out_U);
    return status;
}

lmmc_status_t lmmc_svd(const lmmc_mat_t *a, lmmc_svd_result_t *out_result) {
    if (!a || !out_result || !a->data) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows == 0 || a->cols == 0) return LMMC_STATUS_INVALID_ARGUMENT;

    size_t m = a->rows;
    size_t n = a->cols;
    lmmc_status_t status;

    if (m >= n) {
        return svd_tall(a, &out_result->U, &out_result->sigma, &out_result->Vt);
    }


    lmmc_mat_t A_T;
    status = lmmc_mat_create(n, m, &A_T);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_transpose_to(a, &A_T);
    if (status != LMMC_STATUS_OK) { lmmc_mat_destroy(&A_T); return status; }

    lmmc_mat_t Up;
    lmmc_vec_t sp;
    lmmc_mat_t Vtp;
    status = svd_tall(&A_T, &Up, &sp, &Vtp);
    lmmc_mat_destroy(&A_T);
    if (status != LMMC_STATUS_OK) return status;


    status = lmmc_mat_create(m, m, &out_result->U);
    if (status != LMMC_STATUS_OK) goto cleanup_tmp;
    status = lmmc_vec_create(m, &out_result->sigma);
    if (status != LMMC_STATUS_OK) { lmmc_mat_destroy(&out_result->U); goto cleanup_tmp; }
    status = lmmc_mat_create(n, n, &out_result->Vt);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&out_result->U);
        lmmc_vec_destroy(&out_result->sigma);
        goto cleanup_tmp;
    }


    {
        size_t i, j;
        for (i = 0; i < m; i++)
            for (j = 0; j < m; j++)
                MAT_ELEM(&out_result->U, i, j) = MAT_ELEM(&Vtp, j, i);
    }

    {
        size_t i;
        for (i = 0; i < m; i++) out_result->sigma.data[i] = sp.data[i];
    }

    {
        size_t i, j;
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                MAT_ELEM(&out_result->Vt, i, j) = MAT_ELEM(&Up, j, i);
    }

    lmmc_mat_destroy(&Up);
    lmmc_vec_destroy(&sp);
    lmmc_mat_destroy(&Vtp);
    return LMMC_STATUS_OK;

cleanup_tmp:
    lmmc_mat_destroy(&Up);
    lmmc_vec_destroy(&sp);
    lmmc_mat_destroy(&Vtp);
    return status;
}

lmmc_status_t lmmc_pinv(const lmmc_mat_t *a, lmmc_real_t tol, lmmc_mat_t *out_pinv) {
    if (!a || !out_pinv || !a->data || !out_pinv->data)
        return LMMC_STATUS_INVALID_ARGUMENT;
    size_t m = a->rows;
    size_t n = a->cols;
    if (m == 0 || n == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (out_pinv->rows != n || out_pinv->cols != m)
        return LMMC_STATUS_DIMENSION_MISMATCH;

    lmmc_svd_result_t svd;
    lmmc_status_t status = lmmc_svd(a, &svd);
    if (status != LMMC_STATUS_OK) return status;

    size_t p = (m < n) ? m : n;

    lmmc_real_t sigma_max = (p > 0) ? svd.sigma.data[0] : 0.0;
    lmmc_real_t use_tol = tol;
    if (use_tol <= 0.0) {
        lmmc_real_t mn = (m > n) ? (lmmc_real_t)m : (lmmc_real_t)n;
        use_tol = LMMC_REAL_EPSILON * mn * sigma_max;
    }


    size_t i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            lmmc_real_t sum = 0.0;
            for (k = 0; k < p; k++) {
                lmmc_real_t s = svd.sigma.data[k];
                if (s > use_tol) {
                    sum += MAT_ELEM(&svd.Vt, k, i) * MAT_ELEM(&svd.U, j, k) / s;
                }
            }
            MAT_ELEM(out_pinv, i, j) = sum;
        }
    }

    lmmc_svd_result_destroy(&svd);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_cond(const lmmc_mat_t *a, lmmc_real_t *out_cond) {
    if (!a || !out_cond || !a->data) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows == 0 || a->cols == 0) return LMMC_STATUS_INVALID_ARGUMENT;

    lmmc_svd_result_t svd;
    lmmc_status_t status = lmmc_svd(a, &svd);
    if (status != LMMC_STATUS_OK) return status;

    size_t p = svd.sigma.size;
    if (p == 0) {
        lmmc_svd_result_destroy(&svd);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_real_t s_max = svd.sigma.data[0];
    lmmc_real_t s_min = svd.sigma.data[p - 1];
    if (s_min <= 0.0) {
        *out_cond = INFINITY;
    } else {
        *out_cond = s_max / s_min;
    }
    lmmc_svd_result_destroy(&svd);
    return LMMC_STATUS_OK;
}

void lmmc_svd_result_destroy(lmmc_svd_result_t *result) {
    if (!result) return;
    lmmc_mat_destroy(&result->U);
    lmmc_vec_destroy(&result->sigma);
    lmmc_mat_destroy(&result->Vt);
}
