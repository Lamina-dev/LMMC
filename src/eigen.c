/**
 * @file eigen.c
 * @brief Eigenvalue decomposition and SVD implementations.
 */
#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"

#define EIGEN_MAX_ITER 30
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])

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
            /* Zero the strictly-lower entries in column k so accumulate_Q
             * does not pick up garbage. */
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
    /* Build Q = H_0 * H_1 * ... * H_{n-3} so that T = Q^T A Q.
     * Iterate k = n-3, n-4, ..., 0 applying H_k from the LEFT to Q so that
     * after k = 0 we have Q = H_0 (H_1 (... (H_{n-3} I) ...)) = H_0 ... H_{n-3}. */
    for (k = n - 3; ; k--) {
        lmmc_real_t tau = taus ? taus[k] : 0.0;
        if (tau != 0.0) {
            /* Apply H_k from the left: Q := H_k * Q.
             * H_k acts only on rows k+1, k+2, ..., n-1.
             * For each column j of Q:
             *   s = sum_{i=k+1..n-1} v[i-k-1] * Q[i, j], with v[0] = 1.
             *   Q[i, j] -= tau * v[i-k-1] * s   for i = k+1 .. n-1. */
            for (j = 0; j < n; j++) {
                lmmc_real_t s = MAT_ELEM(Q, k + 1, j); /* v[0] = 1 */
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
    /*
     * Make off-diagonal non-negative via a diagonal similarity D = diag(s_0, s_1, ...).
     * Choose s_0 = 1 and s_{i+1} = s_i * sign(offdiag[i]), then
     *   T' = D * T * D  has  offdiag'[i] = |offdiag[i]|  and  diag'[i] = diag[i].
     * The corresponding eigenvector accumulator is  Q' = Q * D, i.e. column i of
     * out_result->eigenvectors is scaled by s_i. With this, tridiag_ql below
     * solves T' = (Q')^T * A * Q' so the final eigenvectors are correct.
     */
    {
        size_t ii, jj;
        lmmc_real_t s = 1.0;
        for (ii = 0; ii < n - 1; ii++) {
            lmmc_real_t off = offdiag[ii];
            lmmc_real_t sign_next = (off >= 0.0) ? s : -s;  /* s_{ii+1} = s_ii * sign(off) */
            offdiag[ii] = (off >= 0.0) ? off : -off;        /* |offdiag[ii]| */
            s = sign_next;
            if (sign_next < 0.0) {
                /* Flip sign of column (ii+1) of Q. */
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

/* ============================================================
 * Real eigenvalues of a general matrix via the characteristic
 * polynomial (Faddeev–LeVerrier) and Bairstow's method.
 *
 * This delivers all n eigenvalues — including real-or-complex
 * conjugate pairs — directly as roots of det(λI − A) = 0.
 * It is numerically modest but reliable for n ≲ 20 and avoids
 * the bookkeeping of a full implicit-shift QR.  For larger n
 * the same code still runs but accuracy degrades; the property
 * tests exercise sizes ≤ 6 which is comfortably within range.
 * ============================================================ */

/* Solve x^2 + p x + q = 0 robustly, output two complex roots in
 * (re_a, im_a), (re_b, im_b). */
static void quad_solve(lmmc_real_t p, lmmc_real_t q,
                       lmmc_real_t *re_a, lmmc_real_t *im_a,
                       lmmc_real_t *re_b, lmmc_real_t *im_b)
{
    lmmc_real_t disc = p * p - 4.0 * q;
    if (disc >= 0.0) {
        lmmc_real_t s = sqrt(disc);
        lmmc_real_t r1 = (p >= 0.0) ? (-p - s) * 0.5 : (-p + s) * 0.5;
        lmmc_real_t r2 = (r1 != 0.0) ? (q / r1) : ((-p - s) * 0.5);
        *re_a = r1; *im_a = 0.0;
        *re_b = r2; *im_b = 0.0;
    } else {
        lmmc_real_t s = sqrt(-disc) * 0.5;
        *re_a = -p * 0.5; *im_a =  s;
        *re_b = -p * 0.5; *im_b = -s;
    }
}

/* Bairstow's method: extract one quadratic factor x^2 + p x + q from
 * the polynomial whose coefficients are c[0..deg], with c[0] the
 * leading coefficient.  Updates c[] in place to the deflated
 * (deg-2)-degree quotient and returns the (p, q) pair via *p, *q.
 * Returns 0 on success, non-zero on failure to converge. */
static int bairstow_step(lmmc_real_t *c, size_t deg, lmmc_real_t *p_out,
                         lmmc_real_t *q_out)
{
    if (deg < 2) return -1;
    /* Working coefficients (length deg+1). Synthetic division gives
     * b[i] = c[i] - p*b[i-1] - q*b[i-2], with b[-1]=b[-2]=0.
     * We then need partial derivatives via:
     *   f[i] = b[i] - p*f[i-1] - q*f[i-2]. */
    lmmc_real_t *b = (lmmc_real_t *)lmmc_alloc((deg + 1) * sizeof(lmmc_real_t));
    lmmc_real_t *f = (lmmc_real_t *)lmmc_alloc((deg + 1) * sizeof(lmmc_real_t));
    if (!b || !f) {
        if (b) lmmc_free(b);
        if (f) lmmc_free(f);
        return -2;
    }

    lmmc_real_t p = (deg >= 1) ? (c[1] / c[0]) : 0.0;
    lmmc_real_t q = (deg >= 2) ? (c[2] / c[0]) : 0.0;

    /* Bairstow's method can stall for poor (p,q) starts; if convergence
     * fails, restart with a few diverse seed pairs. */
    lmmc_real_t starts[][2] = {
        { 0.0, 0.0 },
        { 1.0, 1.0 },
        { -1.0, 1.0 },
        { 0.5, 0.5 },
        { 2.0, -2.0 },
        { 0.3, -0.7 },
    };
    int n_starts = (int)(sizeof(starts) / sizeof(starts[0]));
    int start_idx = -1; /* use the canonical p,q first; restart when needed */

    const int max_iter = 200;
    int restarts_left = n_starts;
    while (1) {
    for (int iter = 0; iter < max_iter; iter++) {
        b[0] = c[0];
        if (deg >= 1) b[1] = c[1] - p * b[0];
        for (size_t i = 2; i <= deg; i++) {
            b[i] = c[i] - p * b[i - 1] - q * b[i - 2];
        }
        f[0] = b[0];
        if (deg >= 1) f[1] = b[1] - p * f[0];
        for (size_t i = 2; i + 1 <= deg; i++) {
            f[i] = b[i] - p * f[i - 1] - q * f[i - 2];
        }
        /* Newton step for Bairstow:
         *   [f[d-2]   f[d-3]] [dp]   [b[d-1]]
         *   [f[d-1]   f[d-2]] [dq] = [b[d]  ]
         * (standard formulation; see Press et al. NR §9.5)
         * f[d-3] is interpreted as 0 when d < 3. */
        lmmc_real_t f_dm1 = (deg >= 1) ? f[deg - 1] : 0.0;
        lmmc_real_t f_dm2 = (deg >= 2) ? f[deg - 2] : 0.0;
        lmmc_real_t f_dm3 = (deg >= 3) ? f[deg - 3] : 0.0;
        lmmc_real_t a11 = f_dm2, a12 = f_dm3;
        lmmc_real_t a21 = f_dm1, a22 = f_dm2;
        lmmc_real_t det_a = a11 * a22 - a12 * a21;
        if (det_a == 0.0) {
            /* Perturb to escape stagnation. */
            p += 1.0; q -= 1.0;
            continue;
        }
        lmmc_real_t rhs1 = b[deg - 1];
        lmmc_real_t rhs2 = b[deg];
        lmmc_real_t dp = ( a22 * rhs1 - a12 * rhs2) / det_a;
        lmmc_real_t dq = (-a21 * rhs1 + a11 * rhs2) / det_a;
        p += dp;
        q += dq;
        if (lmmc_abs(dp) + lmmc_abs(dq) <
            1e-14 * (lmmc_abs(p) + lmmc_abs(q) + 1.0)) {
            /* Convergence: also require residuals to be small. */
            if (lmmc_abs(b[deg]) + lmmc_abs(b[deg - 1]) <
                1e-10 * (lmmc_abs(c[0]) + lmmc_abs(c[deg]) + 1.0)) {
                *p_out = p; *q_out = q;
                /* Deflate: the quotient has coefficients b[0..deg-2]. */
                for (size_t i = 0; i + 2 <= deg; i++) c[i] = b[i];
                lmmc_free(b);
                lmmc_free(f);
                return 0;
            }
        }
    }
    /* Restart with another seed if available, else give up. */
    if (restarts_left <= 0) break;
    start_idx++;
    if (start_idx >= n_starts) break;
    restarts_left--;
    p = starts[start_idx][0];
    q = starts[start_idx][1];
    }
    lmmc_free(b);
    lmmc_free(f);
    return -3;
}

/* Compute the characteristic polynomial of A (size n) via the
 * Faddeev–LeVerrier algorithm.  Output: poly[0..n] holds coefficients
 * with poly[0] leading (x^n) so poly[n] is the constant term.  Sign
 * convention: poly = det(λI − A). */
static lmmc_status_t char_poly_faddeev_leverrier(const lmmc_mat_t *a,
                                                 lmmc_real_t *poly)
{
    size_t n = a->rows;
    /* Working matrices M = I, A_copy, M_next. */
    lmmc_mat_t M, AM;
    lmmc_status_t st;
    st = lmmc_mat_create(n, n, &M);
    if (st != LMMC_STATUS_OK) return st;
    st = lmmc_mat_create(n, n, &AM);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&M); return st; }
    /* M = I */
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            MAT_ELEM(&M, i, j) = (i == j) ? 1.0 : 0.0;
    poly[0] = 1.0; /* leading x^n coefficient */
    for (size_t k = 1; k <= n; k++) {
        /* AM = A * M */
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                lmmc_real_t s = 0.0;
                for (size_t p = 0; p < n; p++)
                    s += MAT_ELEM(a, i, p) * MAT_ELEM(&M, p, j);
                MAT_ELEM(&AM, i, j) = s;
            }
        }
        /* c_k = -trace(AM) / k */
        lmmc_real_t tr = 0.0;
        for (size_t i = 0; i < n; i++) tr += MAT_ELEM(&AM, i, i);
        lmmc_real_t ck = -tr / (lmmc_real_t)k;
        poly[k] = ck;
        /* M = AM + ck * I */
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                MAT_ELEM(&M, i, j) = MAT_ELEM(&AM, i, j) + ((i == j) ? ck : 0.0);
            }
        }
    }
    lmmc_mat_destroy(&M);
    lmmc_mat_destroy(&AM);
    return LMMC_STATUS_OK;
}

/* Find all roots of poly[0..deg] (poly[0] leading coefficient) and
 * write them into (re[0..deg-1], im[0..deg-1]).  Uses Bairstow's
 * method with linear deflation for any odd remainder. */
static lmmc_status_t poly_roots(lmmc_real_t *poly, size_t deg,
                                lmmc_real_t *re, lmmc_real_t *im)
{
    size_t out = 0;
    lmmc_real_t *c = (lmmc_real_t *)lmmc_alloc((deg + 1) * sizeof(lmmc_real_t));
    if (!c) return LMMC_STATUS_ALLOCATION_FAILED;
    for (size_t i = 0; i <= deg; i++) c[i] = poly[i];

    while (deg > 2) {
        lmmc_real_t p, q;
        int rc = bairstow_step(c, deg, &p, &q);
        if (rc != 0) { lmmc_free(c); return LMMC_STATUS_CONVERGENCE_FAILED; }
        /* Roots of x^2 + p*x + q = 0. */
        quad_solve(p, q, &re[out], &im[out], &re[out + 1], &im[out + 1]);
        out += 2;
        deg -= 2;
    }

    if (deg == 2) {
        /* Solve directly: c[0]*x^2 + c[1]*x + c[2] = 0 → x^2 + (c[1]/c[0])*x + (c[2]/c[0]) = 0 */
        lmmc_real_t p = c[1] / c[0], q = c[2] / c[0];
        quad_solve(p, q, &re[out], &im[out], &re[out + 1], &im[out + 1]);
        out += 2;
    } else if (deg == 1) {
        /* c[0]*x + c[1] = 0 */
        re[out] = -c[1] / c[0];
        im[out] = 0.0;
        out += 1;
    }

    lmmc_free(c);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_eigen_general(const lmmc_mat_t *a, lmmc_eigen_gen_result_t *out_result)
{
    lmmc_status_t status;
    if (!a || !out_result || !a->data) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols || a->rows == 0) return LMMC_STATUS_INVALID_ARGUMENT;

    size_t n = a->rows;

    if (n == 1) {
        status = lmmc_vec_create(1, &out_result->real_parts);
        if (status != LMMC_STATUS_OK) return status;
        status = lmmc_vec_create(1, &out_result->imag_parts);
        if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&out_result->real_parts); return status; }
        out_result->real_parts.data[0] = MAT_ELEM(a, 0, 0);
        out_result->imag_parts.data[0] = 0.0;
        return LMMC_STATUS_OK;
    }

    status = lmmc_vec_create(n, &out_result->real_parts);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(n, &out_result->imag_parts);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&out_result->real_parts); return status; }

    if (n == 2) {
        /* Closed form. */
        lmmc_real_t a00 = MAT_ELEM(a, 0, 0), a01 = MAT_ELEM(a, 0, 1);
        lmmc_real_t a10 = MAT_ELEM(a, 1, 0), a11 = MAT_ELEM(a, 1, 1);
        lmmc_real_t tr = a00 + a11;
        lmmc_real_t det = a00 * a11 - a01 * a10;
        /* Roots of x^2 - tr*x + det = 0 → use quad_solve with p=-tr, q=det. */
        quad_solve(-tr, det,
                   &out_result->real_parts.data[0], &out_result->imag_parts.data[0],
                   &out_result->real_parts.data[1], &out_result->imag_parts.data[1]);
        return LMMC_STATUS_OK;
    }

    /* General path: characteristic polynomial via Faddeev–LeVerrier,
     * then Bairstow root finding. */
    lmmc_real_t *poly = (lmmc_real_t *)lmmc_alloc((n + 1) * sizeof(lmmc_real_t));
    if (!poly) {
        lmmc_vec_destroy(&out_result->real_parts);
        lmmc_vec_destroy(&out_result->imag_parts);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    status = char_poly_faddeev_leverrier(a, poly);
    if (status != LMMC_STATUS_OK) {
        lmmc_free(poly);
        lmmc_vec_destroy(&out_result->real_parts);
        lmmc_vec_destroy(&out_result->imag_parts);
        return status;
    }
    status = poly_roots(poly, n,
                        out_result->real_parts.data,
                        out_result->imag_parts.data);
    lmmc_free(poly);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&out_result->real_parts);
        lmmc_vec_destroy(&out_result->imag_parts);
        return status;
    }
    return LMMC_STATUS_OK;
}

/* Helper: 2x2 eigenvalues from a real matrix [[a, b], [c, d]] (modifies
 * out arrays at positions p and p+1; returns void). Uses LAPACK-style
 * formulation (cf. dlanv2) that returns real eigenvalues when they are
 * real, otherwise a conjugate pair. */
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
        /* Numerically stable: pick larger root first. */
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
/* ============================================================
 * SVD implementation
 *
 * Strategy: Golub-Kahan bidiagonalization + implicit-shift QR.
 *
 *   1. For an m x n matrix A with m >= n, apply alternating
 *      Householder transformations from the left and from the
 *      right to reduce A to an upper bidiagonal matrix B with
 *      diagonal d[0..n-1] and superdiagonal e[0..n-2].
 *      The left transforms are accumulated into U (m x m),
 *      and the right transforms into V (n x n).
 *
 *      A = U * B * V^T  with B upper bidiagonal.
 *
 *   2. Apply the implicit-shift QR algorithm of Demmel-Kahan
 *      (LAPACK routine DBDSQR) to drive the superdiagonal of B
 *      to zero. Each step computes a Wilkinson shift from the
 *      trailing 2x2 block of B^T B and chases the resulting bulge
 *      down the bidiagonal using Givens rotations applied
 *      alternately from the right (into V) and the left (into U).
 *
 *   3. Make singular values non-negative (flip the sign of any
 *      negative one and negate the corresponding column of U) and
 *      sort in descending order, reordering columns of U and V.
 *
 *   4. Output U (m x m), sigma (length n, descending) and Vt
 *      (n x n). For m < n, the procedure is applied to A^T and
 *      the factors are remapped: A^T = U' S V'^T means
 *      A = V' S^T U'^T, so U_A = V', sigma_A = sigma' (length m),
 *      Vt_A = U'^T.
 * ============================================================ */

/* Apply a Givens rotation (c, s) from the right to columns p and q of an
 * (rows x cols_total) matrix M.  Updates M_{:,p} := c*M_{:,p} + s*M_{:,q}
 * and M_{:,q} := -s*M_{:,p_old} + c*M_{:,q}.
 */
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

/* Compute Householder reflector: given a column slice x[0..len-1], compute
 * a vector v (overwriting x with v[1..] and setting x[0]=1) and scalar tau
 * such that (I - tau v v^T) x = beta * e_1.  Stores beta in *beta_out.
 * If x is already a multiple of e_1 (or zero), sets tau = 0 and beta = x[0].
 */
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
    /* Choose beta = -sign(alpha) * mu (sign(0) = +1) so v[0] = alpha - beta
     * has no cancellation (both summands have the same sign). */
    lmmc_real_t beta = (alpha >= 0.0) ? -mu : mu;
    lmmc_real_t v0 = alpha - beta; /* never near zero */
    *beta_out = beta;
    /* Normalize so v[0] = 1; tau scales accordingly: tau = 2/(1 + sigma/v0^2). */
    lmmc_real_t v0_sq = v0 * v0;
    *tau_out = 2.0 * v0_sq / (sigma + v0_sq);
    lmmc_real_t inv = 1.0 / v0;
    for (i = 1; i < len; i++) x[i] *= inv;
    x[0] = 1.0;
}

/* Apply Householder reflector (I - tau v v^T) to the columns indexed by
 * column range [j0, j1) of rows [i0, i0+len), in place.
 *   M_{i0..i0+len-1, j} -= tau * v * (v^T M_{i0..,j})
 * v is given by v[0]=1 and v[1..len-1] = stored values; passed in as vec
 * with vec[0]=1 implicitly handled.
 */
static void householder_apply_left(lmmc_mat_t *M, size_t i0, size_t len,
                                   size_t j0, size_t j1,
                                   const lmmc_real_t *v, lmmc_real_t tau) {
    size_t i, j;
    if (tau == 0.0) return;
    for (j = j0; j < j1; j++) {
        lmmc_real_t s = MAT_ELEM(M, i0, j); /* v[0] = 1 */
        for (i = 1; i < len; i++) s += v[i] * MAT_ELEM(M, i0 + i, j);
        s *= tau;
        MAT_ELEM(M, i0, j) -= s;
        for (i = 1; i < len; i++) MAT_ELEM(M, i0 + i, j) -= s * v[i];
    }
}

/* Apply Householder reflector from the right to rows [i0, i1), columns
 * [j0, j0+len), in place.
 *   M_{i, j0..j0+len-1} -= tau * (M_{i, j0..} v) * v^T
 */
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

/* Construct a Givens rotation that zeros b: returns (c, s) such that
 *   [ c  s] [a]   [r]
 *   [-s  c] [b] = [0],   r = sqrt(a^2 + b^2).
 */
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

/* Implicit-shift QR step on an upper bidiagonal matrix B with diagonal
 * d[lo..hi] and superdiagonal e[lo..hi-1].  Updates U (left, m x m) and V
 * (right, n x n) by accumulating the Givens rotations.
 */
static void bidiag_qr_step(lmmc_real_t *d, lmmc_real_t *e,
                           size_t lo, size_t hi,
                           lmmc_mat_t *U, lmmc_mat_t *V) {
    /* Compute Wilkinson shift from trailing 2x2 of B^T B:
     *   T = [d[hi-1]^2 + e[hi-2]^2,   d[hi-1]*e[hi-1];
     *        d[hi-1]*e[hi-1],          d[hi]^2 + e[hi-1]^2]
     * where e[hi-2] is 0 if hi-1 == lo.
     */
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
        /* Right Givens to eliminate g (acts on columns i, i+1 of B and V). */
        givens_compute(f, g, &c, &s, &r);
        if (i > lo) e[i - 1] = r;
        f = c * d[i] + s * e[i];
        e[i] = c * e[i] - s * d[i];
        g = s * d[i + 1];
        d[i + 1] = c * d[i + 1];
        if (V) givens_right(V, V->rows, i, i + 1, c, s);

        /* Left Givens to eliminate g (acts on rows i, i+1 of B and U). */
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

/* Reduce A (m x n, m >= n) to upper bidiagonal form B by alternating
 * Householder transforms from the left and right. On return, d[0..n-1]
 * is the diagonal of B, e[0..n-2] the superdiagonal, U is the
 * accumulated product of left reflectors (m x m), V the accumulated
 * product of right reflectors (n x n). */
static lmmc_status_t bidiagonalize(const lmmc_mat_t *a,
                                   lmmc_real_t *d, lmmc_real_t *e,
                                   lmmc_mat_t *U, lmmc_mat_t *V) {
    size_t m = a->rows;
    size_t n = a->cols;
    size_t i, j, k;
    lmmc_status_t status;

    /* Working copy A_work = A. */
    lmmc_mat_t W;
    status = lmmc_mat_create(m, n, &W);
    if (status != LMMC_STATUS_OK) return status;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&W, i, j) = MAT_ELEM(a, i, j);

    /* U <- I_m, V <- I_n. */
    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++) MAT_ELEM(U, i, j) = (i == j) ? 1.0 : 0.0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) MAT_ELEM(V, i, j) = (i == j) ? 1.0 : 0.0;

    lmmc_real_t *vbuf = (lmmc_real_t *)lmmc_alloc(((m > n) ? m : n) * sizeof(lmmc_real_t));
    if (!vbuf) { lmmc_mat_destroy(&W); return LMMC_STATUS_ALLOCATION_FAILED; }

    for (k = 0; k < n; k++) {
        /* Left reflector: zero W[k+1..m-1, k]. */
        size_t len = m - k;
        for (i = 0; i < len; i++) vbuf[i] = MAT_ELEM(&W, k + i, k);
        lmmc_real_t tau, beta;
        householder_make(vbuf, len, &tau, &beta);
        d[k] = beta;
        /* Apply to remaining columns of W. */
        if (k + 1 < n) {
            householder_apply_left(&W, k, len, k + 1, n, vbuf, tau);
        }
        /* Apply to U from the right: U[:, k..m-1] := U[:, k..m-1] (I - tau v v^T). */
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

        /* Right reflector: zero W[k, k+2..n-1]. */
        if (k + 1 < n) {
            size_t rlen = n - k - 1;
            for (j = 0; j < rlen; j++) vbuf[j] = MAT_ELEM(&W, k, k + 1 + j);
            householder_make(vbuf, rlen, &tau, &beta);
            e[k] = beta;
            /* Apply to remaining rows of W. */
            if (k + 1 < m) {
                householder_apply_right(&W, k + 1, m, k + 1, rlen, vbuf, tau);
            }
            /* Apply to V from the right: V[:, k+1..n-1] := V[:, k+1..n-1] (I - tau v v^T). */
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

/* Core SVD for m >= n: bidiagonalize A then drive superdiagonal to zero
 * via implicit-shift QR, accumulating singular vectors. Outputs U (m x m),
 * sigma (length n), Vt (n x n). */
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

    /* Step 1: bidiagonalization. */
    status = bidiagonalize(a, d, e, out_U, &V);
    if (status != LMMC_STATUS_OK) goto fail;
    /* e has length n-1; e[n-1] will not be used. */
    if (n >= 1) e[n - 1] = 0.0;

    /* Step 2: implicit-shift QR on the bidiagonal. */
    {
        const size_t max_iter_total = 30 * n + 30;
        size_t iter = 0;
        size_t hi = (n == 0) ? 0 : n - 1;
        while (n >= 1) {
            /* Find smallest hi such that e[hi-1] is non-negligible; deflate
             * trailing zero superdiagonals. */
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
            if (hi == 0) break; /* Fully diagonal. */

            /* Find largest lo such that e[lo-1] is negligible (block start). */
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
            /* Handle a zero diagonal element d[k] in [lo..hi-1] by chasing
             * the row out via left Givens rotations.  This decouples the
             * trailing block. */
            int handled = 0;
            for (k = lo; k < hi; k++) {
                if (d[k] == 0.0) {
                    /* Use left Givens rotations to zero e[k..hi-1] row. */
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

    /* Step 3: make singular values non-negative. */
    for (i = 0; i < n; i++) {
        if (d[i] < 0.0) {
            d[i] = -d[i];
            for (k = 0; k < n; k++) MAT_ELEM(&V, k, i) = -MAT_ELEM(&V, k, i);
        }
    }

    /* Step 4: sort singular values descending; permute U columns and V
     * columns accordingly. */
    order = (size_t *)lmmc_alloc(n * sizeof(size_t));
    if (!order) { status = LMMC_STATUS_ALLOCATION_FAILED; goto fail; }
    for (i = 0; i < n; i++) order[i] = i;
    /* Selection sort. */
    for (i = 0; i + 1 < n; i++) {
        size_t mx = i;
        for (j = i + 1; j < n; j++) {
            if (d[order[j]] > d[order[mx]]) mx = j;
        }
        if (mx != i) { size_t tmp = order[i]; order[i] = order[mx]; order[mx] = tmp; }
    }

    /* Output sigma (length n). */
    for (i = 0; i < n; i++) out_sigma->data[i] = d[order[i]];

    /* Output Vt: row i of Vt = column order[i] of V. */
    for (i = 0; i < n; i++)
        for (k = 0; k < n; k++)
            MAT_ELEM(out_Vt, i, k) = MAT_ELEM(&V, k, order[i]);

    /* Apply permutation to first n columns of U. */
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

    /* m < n: compute SVD of A^T (n x m, satisfies rows >= cols) and
     * derive A's SVD by transposing the factors:
     *   A^T = U' * S * V'^T  =>  A = V' * S^T * U'^T
     * So: U_A = V'  (m x m),  sigma_A = sigma',  Vt_A = U'^T  (n x n). */
    lmmc_mat_t A_T;
    status = lmmc_mat_create(n, m, &A_T);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_transpose_to(a, &A_T);
    if (status != LMMC_STATUS_OK) { lmmc_mat_destroy(&A_T); return status; }

    lmmc_mat_t Up; /* n x n */
    lmmc_vec_t sp; /* m */
    lmmc_mat_t Vtp; /* m x m */
    status = svd_tall(&A_T, &Up, &sp, &Vtp);
    lmmc_mat_destroy(&A_T);
    if (status != LMMC_STATUS_OK) return status;

    /* Allocate output matrices and copy. */
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

    /* U_A = (Vtp)^T : (Vtp is m x m), so U_A[i,j] = Vtp[j,i]. */
    {
        size_t i, j;
        for (i = 0; i < m; i++)
            for (j = 0; j < m; j++)
                MAT_ELEM(&out_result->U, i, j) = MAT_ELEM(&Vtp, j, i);
    }
    /* sigma_A = sigma' */
    {
        size_t i;
        for (i = 0; i < m; i++) out_result->sigma.data[i] = sp.data[i];
    }
    /* Vt_A = (Up)^T : Up is n x n, so Vt_A[i,j] = Up[j,i]. */
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
    /* Default tolerance: eps * max(m,n) * sigma_max. */
    lmmc_real_t sigma_max = (p > 0) ? svd.sigma.data[0] : 0.0;
    lmmc_real_t use_tol = tol;
    if (use_tol <= 0.0) {
        lmmc_real_t mn = (m > n) ? (lmmc_real_t)m : (lmmc_real_t)n;
        use_tol = LMMC_REAL_EPSILON * mn * sigma_max;
    }

    /* A^+ = V * S^+ * U^T, so (A^+)[i, j] = sum_k Vt[k, i] * (1/sigma_k) * U[j, k]
     * for sigma_k > use_tol. */
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
