/** @file eigen_svd_iteration.c
 * @brief Golub-Kahan 鼓包追赶、消去与零对角元消除。
 */
#include <math.h>
#include "internal.h"
#include "internal/eigen_svd_internal.h"
#include "eigen_internal.h"
#include "lmmc/linear_algebra.h"

static lmmc_real_t trailing_shift(const lmmc_real_t *d,
                                  const lmmc_real_t *e, size_t lo, size_t hi)
{
    lmmc_real_t emm = (hi >= lo + 2) ? e[hi - 2] : 0.0;
    lmmc_real_t tnm = d[hi - 1] * d[hi - 1] + emm * emm;
    lmmc_real_t tnn = d[hi] * d[hi] + e[hi - 1] * e[hi - 1];
    lmmc_real_t tmn = d[hi - 1] * e[hi - 1];
    lmmc_real_t bb = (tnm - tnn) * 0.5;
    lmmc_real_t cc = tmn * tmn;
    if (bb == 0.0 && cc == 0.0) {
        return tnn;
    }
    lmmc_real_t disc = sqrt(bb * bb + cc);
    lmmc_real_t denom = (bb >= 0.0) ? (bb + disc) : (bb - disc);
    return tnn - cc / denom;
}

static void bidiag_qr_step(lmmc_real_t *d, lmmc_real_t *e,
                           size_t lo, size_t hi, lmmc_mat_t *u, lmmc_mat_t *v)
{
    lmmc_real_t f = d[lo] * d[lo] - trailing_shift(d, e, lo, hi);
    lmmc_real_t g = d[lo] * e[lo];
    for (size_t i = lo; i < hi; ++i) {
        lmmc_real_t c, s, r;
        givens_compute(f, g, &c, &s, &r);
        if (i > lo) {
            e[i - 1] = r;
        }
        f = c * d[i] + s * e[i];
        e[i] = c * e[i] - s * d[i];
        g = s * d[i + 1];
        d[i + 1] = c * d[i + 1];
        givens_right(v, v->rows, i, i + 1, c, s);

        givens_compute(f, g, &c, &s, &r);
        d[i] = r;
        f = c * e[i] + s * d[i + 1];
        d[i + 1] = c * d[i + 1] - s * e[i];
        if (i + 1 < hi) {
            g = s * e[i + 1];
            e[i + 1] = c * e[i + 1];
        }
        givens_right(u, u->rows, i, i + 1, c, s);
    }
    e[hi - 1] = f;
}

static int deflate_edge(const lmmc_real_t *d, lmmc_real_t *e, size_t index)
{
    lmmc_real_t threshold = LMMC_REAL_EPSILON *
        (lmmc_abs(d[index - 1]) + lmmc_abs(d[index]));
    if (lmmc_abs(e[index - 1]) <= threshold) {
        e[index - 1] = 0.0;
        return 1;
    }
    return 0;
}

static void cancel_zero_diagonal(lmmc_real_t *d, lmmc_real_t *e,
                                  size_t k, size_t hi, lmmc_mat_t *u)
{
    lmmc_real_t f = e[k];
    e[k] = 0.0;
    for (size_t i = k + 1; i <= hi; ++i) {
        lmmc_real_t c, s, r;
        givens_compute(d[i], f, &c, &s, &r);
        d[i] = r;
        if (i < hi) {
            f = -s * e[i];
            e[i] = c * e[i];
        }
        givens_right(u, u->rows, k, i, c, s);
    }
}

static int cancel_block_zero(lmmc_real_t *d, lmmc_real_t *e,
                              size_t lo, size_t hi, lmmc_mat_t *u)
{
    for (size_t k = lo; k < hi; ++k) {
        if (d[k] == 0.0) {
            cancel_zero_diagonal(d, e, k, hi, u);
            return 1;
        }
    }
    return 0;
}

lmmc_status_t lmmc_svd_diagonalize(lmmc_real_t *d, lmmc_real_t *e,
                                 size_t n, lmmc_mat_t *u, lmmc_mat_t *v)
{
    const size_t max_iter_total = 30 * n + 30;
    size_t iter = 0;
    size_t hi = n - 1;
    while (hi > 0) {
        while (hi > 0 && deflate_edge(d, e, hi)) {
            --hi;
        }
        if (hi == 0) {
            break;
        }
        size_t lo = hi;
        while (lo > 0 && !deflate_edge(d, e, lo)) {
            --lo;
        }
        if (cancel_block_zero(d, e, lo, hi, u)) {
            continue;
        }
        if (iter++ > max_iter_total) {
            return LMMC_STATUS_CONVERGENCE_FAILED;
        }
        bidiag_qr_step(d, e, lo, hi, u, v);
    }
    return LMMC_STATUS_OK;
}
