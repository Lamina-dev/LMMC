/**
 * @file singular_value_decomposition.c
 * @brief 奇异值分解、伪逆与条件数实现。
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

