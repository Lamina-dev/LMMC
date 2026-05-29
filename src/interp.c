/**
 * @file interp.c
 * @brief 一维插值算法实现：线性、三次样条（多种边界条件）、Lagrange、PCHIP、Akima、二维插值。
 */
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/interp.h"

#include <stddef.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * 内部辅助
 * ======================================================================== */

static size_t interp_find_interval(const lmmc_real_t* xs, size_t n,
                                   lmmc_real_t query_x)
{
    size_t lo = 0, hi = n - 1, mid;
    while (hi - lo > 1) {
        mid = lo + (hi - lo) / 2;
        if (xs[mid] <= query_x) lo = mid;
        else hi = mid;
    }
    return lo;
}

static int interp_check_strictly_increasing(const lmmc_real_t* xs, size_t n)
{
    size_t i;
    for (i = 0; i < n - 1; i++) {
        if (xs[i + 1] <= xs[i]) return 0;
    }
    return 1;
}

/* ========================================================================
 * 一维线性插值
 * ======================================================================== */

lmmc_status_t lmmc_interp_linear(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n,
    lmmc_real_t query_x, lmmc_real_t* out_y)
{
    size_t lo;
    lmmc_real_t t;
    if (!xs || !ys || !out_y) return LMMC_STATUS_INVALID_ARGUMENT;
    if (n < 2) return LMMC_STATUS_INVALID_ARGUMENT;
    if (query_x < xs[0] || query_x > xs[n - 1]) return LMMC_STATUS_OUT_OF_RANGE;
    if (query_x == xs[n - 1]) { *out_y = ys[n - 1]; return LMMC_STATUS_OK; }
    lo = interp_find_interval(xs, n, query_x);
    t = (query_x - xs[lo]) / (xs[lo + 1] - xs[lo]);
    *out_y = ys[lo] + (ys[lo + 1] - ys[lo]) * t;
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * 三次样条插值
 * ======================================================================== */

struct lmmc_interp_cspline_t {
    size_t n;
    lmmc_real_t* xs;
    lmmc_real_t* ys;
    lmmc_real_t* coeffs; /* 4*(n-1): a,b,c,d per segment */
};

void lmmc_interp_cspline_destroy(lmmc_interp_cspline_t* spline)
{
    if (!spline) return;
    lmmc_free(spline->xs);
    lmmc_free(spline->ys);
    lmmc_free(spline->coeffs);
    lmmc_free(spline);
}

lmmc_status_t lmmc_interp_cspline_eval(
    const lmmc_interp_cspline_t* spline,
    lmmc_real_t query_x, lmmc_real_t* out_y)
{
    size_t seg;
    lmmc_real_t dx, a, b, c, d;
    if (!spline || !out_y) return LMMC_STATUS_INVALID_ARGUMENT;
    if (query_x < spline->xs[0] || query_x > spline->xs[spline->n - 1])
        return LMMC_STATUS_OUT_OF_RANGE;
    if (query_x == spline->xs[spline->n - 1]) {
        *out_y = spline->ys[spline->n - 1];
        return LMMC_STATUS_OK;
    }
    seg = interp_find_interval(spline->xs, spline->n, query_x);
    dx = query_x - spline->xs[seg];
    a = spline->coeffs[4 * seg + 0];
    b = spline->coeffs[4 * seg + 1];
    c = spline->coeffs[4 * seg + 2];
    d = spline->coeffs[4 * seg + 3];
    *out_y = a + dx * (b + dx * (c + dx * d));
    return LMMC_STATUS_OK;
}

/**
 * @internal
 * @brief Solve a tridiagonal system using Thomas algorithm.
 * @param n   System size.
 * @param sub Sub-diagonal (length n, sub[0] unused).
 * @param dia Main diagonal (length n).
 * @param sup Super-diagonal (length n, sup[n-1] unused).
 * @param rhs Right-hand side (length n), overwritten with solution.
 *
 * All arrays are modified in-place.
 */
static void tridiag_solve(size_t n, lmmc_real_t* sub, lmmc_real_t* dia,
                           lmmc_real_t* sup, lmmc_real_t* rhs)
{
    size_t i;
    lmmc_real_t w;
    for (i = 1; i < n; i++) {
        w = sub[i] / dia[i - 1];
        dia[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    rhs[n - 1] /= dia[n - 1];
    for (i = n - 1; i > 0; i--) {
        rhs[i - 1] = (rhs[i - 1] - sup[i - 1] * rhs[i]) / dia[i - 1];
    }
}

/**
 * @internal
 * @brief Compute spline coefficients from second derivatives (moments).
 */
static void cspline_compute_coeffs(lmmc_interp_cspline_t* spline,
                                    const lmmc_real_t* h,
                                    const lmmc_real_t* M)
{
    size_t i, nm1 = spline->n - 1;
    for (i = 0; i < nm1; i++) {
        lmmc_real_t ai = spline->ys[i];
        lmmc_real_t ci = M[i] / 2.0;
        lmmc_real_t di = (M[i + 1] - M[i]) / (6.0 * h[i]);
        lmmc_real_t bi = (spline->ys[i + 1] - spline->ys[i]) / h[i]
                       - h[i] * (2.0 * M[i] + M[i + 1]) / 6.0;
        spline->coeffs[4 * i + 0] = ai;
        spline->coeffs[4 * i + 1] = bi;
        spline->coeffs[4 * i + 2] = ci;
        spline->coeffs[4 * i + 3] = di;
    }
}

lmmc_status_t lmmc_interp_cspline_create_ex(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n,
    lmmc_spline_bc_t bc, lmmc_real_t deriv_left, lmmc_real_t deriv_right,
    lmmc_interp_cspline_t** out_spline)
{
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_real_t* h = NULL;
    lmmc_real_t* M = NULL;  /* second derivatives (moments) */
    lmmc_real_t* sub = NULL;
    lmmc_real_t* dia = NULL;
    lmmc_real_t* sup = NULL;
    lmmc_real_t* rhs_arr = NULL;
    size_t i, nm1, sys_n;

    if (!xs || !ys || !out_spline) return LMMC_STATUS_INVALID_ARGUMENT;
    if (n < 3) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!interp_check_strictly_increasing(xs, n))
        return LMMC_STATUS_INVALID_ARGUMENT;

    /* Periodic BC validation */
    if (bc == LMMC_SPLINE_PERIODIC) {
        if (fabs(ys[0] - ys[n - 1]) > 1e-12)
            return LMMC_STATUS_INVALID_ARGUMENT;
    }

    nm1 = n - 1;

    /* Allocate spline */
    spline = (lmmc_interp_cspline_t*)lmmc_alloc(sizeof(*spline));
    if (!spline) return LMMC_STATUS_ALLOCATION_FAILED;
    spline->n = n;
    spline->xs = NULL; spline->ys = NULL; spline->coeffs = NULL;

    spline->xs = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    spline->ys = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    spline->coeffs = (lmmc_real_t*)lmmc_alloc(4 * nm1 * sizeof(lmmc_real_t));
    if (!spline->xs || !spline->ys || !spline->coeffs) {
        lmmc_interp_cspline_destroy(spline);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(spline->xs, xs, n * sizeof(lmmc_real_t));
    memcpy(spline->ys, ys, n * sizeof(lmmc_real_t));

    /* Compute h[i] = xs[i+1] - xs[i] */
    h = (lmmc_real_t*)lmmc_alloc(nm1 * sizeof(lmmc_real_t));
    M = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (!h || !M) {
        lmmc_free(h); lmmc_free(M);
        lmmc_interp_cspline_destroy(spline);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    for (i = 0; i < nm1; i++) h[i] = xs[i + 1] - xs[i];
    for (i = 0; i < n; i++) M[i] = 0.0;

    if (bc == LMMC_SPLINE_NATURAL || bc == LMMC_SPLINE_CLAMPED) {
        /* System size = n. Natural: M[0]=M[n-1]=0. Clamped: boundary eqs. */
        sys_n = n;
        sub = (lmmc_real_t*)lmmc_alloc(sys_n * sizeof(lmmc_real_t));
        dia = (lmmc_real_t*)lmmc_alloc(sys_n * sizeof(lmmc_real_t));
        sup = (lmmc_real_t*)lmmc_alloc(sys_n * sizeof(lmmc_real_t));
        rhs_arr = (lmmc_real_t*)lmmc_alloc(sys_n * sizeof(lmmc_real_t));
        if (!sub || !dia || !sup || !rhs_arr) {
            lmmc_free(sub); lmmc_free(dia); lmmc_free(sup); lmmc_free(rhs_arr);
            lmmc_free(h); lmmc_free(M);
            lmmc_interp_cspline_destroy(spline);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        for (i = 0; i < sys_n; i++) {
            sub[i] = dia[i] = sup[i] = rhs_arr[i] = 0.0;
        }

        /* Interior rows */
        for (i = 1; i < nm1; i++) {
            sub[i] = h[i - 1];
            dia[i] = 2.0 * (h[i - 1] + h[i]);
            sup[i] = h[i];
            rhs_arr[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i]
                               - (ys[i] - ys[i - 1]) / h[i - 1]);
        }

        if (bc == LMMC_SPLINE_NATURAL) {
            dia[0] = 1.0; rhs_arr[0] = 0.0;
            dia[nm1] = 1.0; rhs_arr[nm1] = 0.0;
        } else { /* CLAMPED */
            dia[0] = 2.0 * h[0];
            sup[0] = h[0];
            rhs_arr[0] = 6.0 * ((ys[1] - ys[0]) / h[0] - deriv_left);
            sub[nm1] = h[nm1 - 1];
            dia[nm1] = 2.0 * h[nm1 - 1];
            rhs_arr[nm1] = 6.0 * (deriv_right - (ys[nm1] - ys[nm1 - 1]) / h[nm1 - 1]);
        }

        tridiag_solve(sys_n, sub, dia, sup, rhs_arr);
        for (i = 0; i < n; i++) M[i] = rhs_arr[i];

        lmmc_free(sub); lmmc_free(dia); lmmc_free(sup); lmmc_free(rhs_arr);
    }
    else if (bc == LMMC_SPLINE_NOT_A_KNOT) {
        /* Not-a-knot: third derivative continuous at x[1] and x[n-2].
         * This means d[0] = d[1] and d[n-3] = d[n-2] where d[i] = (M[i+1]-M[i])/(6*h[i]).
         * Left:  M[0]/h[0] - M[1]*(1/h[0]+1/h[1]) + M[2]/h[1] = 0
         *   => h[1]*M[0] - (h[0]+h[1])*M[1] + h[0]*M[2] = 0
         * Right: h[n-2]*M[n-3] - (h[n-3]+h[n-2])*M[n-2] + h[n-3]*M[n-1] = 0
         *
         * We solve for M[0..n-1] with n equations:
         * Row 0: not-a-knot left condition
         * Rows 1..n-2: interior equations
         * Row n-1: not-a-knot right condition
         */
        sys_n = n;
        sub = (lmmc_real_t*)lmmc_alloc(sys_n * sizeof(lmmc_real_t));
        dia = (lmmc_real_t*)lmmc_alloc(sys_n * sizeof(lmmc_real_t));
        sup = (lmmc_real_t*)lmmc_alloc(sys_n * sizeof(lmmc_real_t));
        rhs_arr = (lmmc_real_t*)lmmc_alloc(sys_n * sizeof(lmmc_real_t));
        if (!sub || !dia || !sup || !rhs_arr) {
            lmmc_free(sub); lmmc_free(dia); lmmc_free(sup); lmmc_free(rhs_arr);
            lmmc_free(h); lmmc_free(M);
            lmmc_interp_cspline_destroy(spline);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        for (i = 0; i < sys_n; i++) {
            sub[i] = dia[i] = sup[i] = rhs_arr[i] = 0.0;
        }

        /* Row 0: h[1]*M[0] - (h[0]+h[1])*M[1] + h[0]*M[2] = 0
         * This is a 3-term equation involving M[0], M[1], M[2].
         * For tridiagonal, row 0 only has dia[0] and sup[0].
         * We need to eliminate M[2] from row 0 using row 1.
         * Alternative: use the condition to express M[0] in terms of M[1],M[2]
         * and substitute into row 1.
         *
         * Better approach: build the full n x n system and solve with
         * a modified tridiagonal that handles the extra coupling.
         * Actually, the not-a-knot condition IS tridiagonal if we note:
         * Row 0: h[1]*M[0] - (h[0]+h[1])*M[1] + h[0]*M[2] = 0
         * This couples M[0], M[1], M[2] -- not tridiagonal.
         *
         * Standard approach: eliminate M[0] and M[n-1] using the not-a-knot
         * conditions, solve the (n-2) interior system, then recover endpoints.
         */
        lmmc_free(sub); lmmc_free(dia); lmmc_free(sup); lmmc_free(rhs_arr);

        /* Solve for M[1..n-2] with modified first and last equations */
        if (n == 3) {
            /* Special case: single interior unknown M[1].
             * Not-a-knot left:  h[1]*M[0] - (h[0]+h[1])*M[1] + h[0]*M[2] = 0
             * Not-a-knot right: same condition for n=3.
             * Interior: h[0]*M[0] + 2*(h[0]+h[1])*M[1] + h[1]*M[2] = rhs1
             * From not-a-knot: M[0] = M[2] = ((h[0]+h[1])*M[1]) / h[1]
             *   Wait: h[1]*M[0] - (h[0]+h[1])*M[1] + h[0]*M[2] = 0
             *   For n=3, both not-a-knot conditions are the same.
             *   We have M[0] = ((h[0]+h[1])*M[1] - h[0]*M[2]) / h[1]
             *   and M[2] = ((h[0]+h[1])*M[1] - h[1]*M[0]) / h[0]
             *   These together imply M[0] = M[1] = M[2].
             */
            lmmc_real_t r1 = 6.0 * ((ys[2] - ys[1]) / h[1] - (ys[1] - ys[0]) / h[0]);
            /* M[0]=M[1]=M[2]=val: h[0]*val + 2*(h[0]+h[1])*val + h[1]*val = r1
             * => val*(3*h[0] + 3*h[1]) = r1 => val = r1/(3*(h[0]+h[1])) */
            lmmc_real_t val = r1 / (3.0 * (h[0] + h[1]));
            M[0] = M[1] = M[2] = val;
        } else {
            /* n >= 4. Solve (n-2) system for M[1]..M[n-2].
             * From not-a-knot left: M[0] = ((h[0]+h[1])*M[1] - h[0]*M[2]) / h[1]
             * From not-a-knot right: M[n-1] = ((h[n-3]+h[n-2])*M[n-2] - h[n-2]*M[n-3]) / h[n-3]
             */
            size_t sn = n - 2;
            sub = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
            dia = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
            sup = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
            rhs_arr = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
            if (!sub || !dia || !sup || !rhs_arr) {
                lmmc_free(sub); lmmc_free(dia); lmmc_free(sup); lmmc_free(rhs_arr);
                lmmc_free(h); lmmc_free(M);
                lmmc_interp_cspline_destroy(spline);
                return LMMC_STATUS_ALLOCATION_FAILED;
            }

            /* Fill interior equations. Local index j maps to global M[j+1]. */
            for (i = 0; i < sn; i++) {
                size_t gi = i + 1;
                sub[i] = h[gi - 1];
                dia[i] = 2.0 * (h[gi - 1] + h[gi]);
                sup[i] = h[gi];
                rhs_arr[i] = 6.0 * ((ys[gi + 1] - ys[gi]) / h[gi]
                                   - (ys[gi] - ys[gi - 1]) / h[gi - 1]);
            }

            /* Modify first equation (i=0, gi=1):
             * Original: h[0]*M[0] + 2*(h[0]+h[1])*M[1] + h[1]*M[2] = rhs
             * Substitute M[0] = ((h[0]+h[1])*M[1] - h[0]*M[2]) / h[1]:
             * h[0]*((h[0]+h[1])*M[1] - h[0]*M[2])/h[1] + 2*(h[0]+h[1])*M[1] + h[1]*M[2] = rhs
             * => M[1]*(h[0]*(h[0]+h[1])/h[1] + 2*(h[0]+h[1]))
             *  + M[2]*(h[1] - h[0]*h[0]/h[1]) = rhs
             */
            dia[0] = h[0] * (h[0] + h[1]) / h[1] + 2.0 * (h[0] + h[1]);
            sup[0] = h[1] - h[0] * h[0] / h[1];
            sub[0] = 0.0; /* no sub-diagonal for first row */

            /* Modify last equation (i=sn-1, gi=n-2):
             * Original: h[n-3]*M[n-3] + 2*(h[n-3]+h[n-2])*M[n-2] + h[n-2]*M[n-1] = rhs
             * Substitute M[n-1] = ((h[n-3]+h[n-2])*M[n-2] - h[n-2]*M[n-3]) / h[n-3]:
             * h[n-3]*M[n-3] + 2*(h[n-3]+h[n-2])*M[n-2]
             *   + h[n-2]*((h[n-3]+h[n-2])*M[n-2] - h[n-2]*M[n-3])/h[n-3] = rhs
             * => M[n-3]*(h[n-3] - h[n-2]*h[n-2]/h[n-3])
             *  + M[n-2]*(2*(h[n-3]+h[n-2]) + h[n-2]*(h[n-3]+h[n-2])/h[n-3]) = rhs
             */
            {
                size_t last = sn - 1;
                size_t gi_last = last + 1; /* n-2 */
                lmmc_real_t ha = h[gi_last - 1]; /* h[n-3] */
                lmmc_real_t hb = h[gi_last];     /* h[n-2] */
                sub[last] = ha - hb * hb / ha;
                dia[last] = 2.0 * (ha + hb) + hb * (ha + hb) / ha;
                sup[last] = 0.0; /* no super-diagonal for last row */
            }

            tridiag_solve(sn, sub, dia, sup, rhs_arr);
            for (i = 0; i < sn; i++) M[i + 1] = rhs_arr[i];

            /* Recover M[0] and M[n-1] */
            M[0] = ((h[0] + h[1]) * M[1] - h[0] * M[2]) / h[1];
            M[nm1] = ((h[nm1 - 2] + h[nm1 - 1]) * M[nm1 - 1]
                    - h[nm1 - 1] * M[nm1 - 2]) / h[nm1 - 2];

            lmmc_free(sub); lmmc_free(dia); lmmc_free(sup); lmmc_free(rhs_arr);
        }
    }
    else if (bc == LMMC_SPLINE_PERIODIC) {
        /* Periodic: M[0] = M[n-1]. Solve cyclic tridiagonal for M[0..n-2].
         * Use Sherman-Morrison formula.
         * System rows for i=0..n-2:
         *   Row i (1<=i<=n-3): h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = rhs_i
         *   Row 0: h[n-2]*M[n-2] + 2*(h[n-2]+h[0])*M[0] + h[0]*M[1] = rhs_0
         *     where rhs_0 = 6*((y1-y0)/h0 - (y0-y[n-2])/h[n-2])
         *     (using y[n-1]=y[0])
         *   Row n-2: h[n-3]*M[n-3] + 2*(h[n-3]+h[n-2])*M[n-2] + h[n-2]*M[0] = rhs_{n-2}
         */
        size_t sn = nm1; /* n-1 unknowns: M[0]..M[n-2] */
        lmmc_real_t* a_sub = NULL;
        lmmc_real_t* a_dia = NULL;
        lmmc_real_t* a_sup = NULL;
        lmmc_real_t* b_vec = NULL;
        lmmc_real_t* u_vec = NULL;
        lmmc_real_t* y_sol = NULL;
        lmmc_real_t* z_sol = NULL;
        lmmc_real_t gamma_val, alpha_corner, beta_corner;
        lmmc_real_t vy, vz;
        size_t j;

        a_sub = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
        a_dia = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
        a_sup = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
        b_vec = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
        u_vec = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
        y_sol = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
        z_sol = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
        if (!a_sub || !a_dia || !a_sup || !b_vec || !u_vec || !y_sol || !z_sol) {
            lmmc_free(a_sub); lmmc_free(a_dia); lmmc_free(a_sup);
            lmmc_free(b_vec); lmmc_free(u_vec); lmmc_free(y_sol); lmmc_free(z_sol);
            lmmc_free(h); lmmc_free(M);
            lmmc_interp_cspline_destroy(spline);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }

        /* Build the cyclic system */
        /* Row 0 */
        a_dia[0] = 2.0 * (h[nm1 - 1] + h[0]);
        a_sup[0] = h[0];
        a_sub[0] = h[nm1 - 1]; /* cyclic corner: connects to M[n-2] */
        b_vec[0] = 6.0 * ((ys[1] - ys[0]) / h[0]
                         - (ys[nm1] - ys[nm1 - 1]) / h[nm1 - 1]);
        /* Note: ys[nm1] = ys[n-1] = ys[0] for periodic */

        /* Interior rows 1..sn-2 */
        for (j = 1; j < sn - 1; j++) {
            a_sub[j] = h[j - 1];
            a_dia[j] = 2.0 * (h[j - 1] + h[j]);
            a_sup[j] = h[j];
            b_vec[j] = 6.0 * ((ys[j + 1] - ys[j]) / h[j]
                             - (ys[j] - ys[j - 1]) / h[j - 1]);
        }

        /* Last row (j = sn-1 = n-2) */
        j = sn - 1;
        a_sub[j] = h[j - 1];
        a_dia[j] = 2.0 * (h[j - 1] + h[j]);
        a_sup[j] = h[j]; /* cyclic corner: connects to M[0] */
        b_vec[j] = 6.0 * ((ys[j + 1] - ys[j]) / h[j]
                         - (ys[j] - ys[j - 1]) / h[j - 1]);

        /* Sherman-Morrison: A = T + u*v^T where
         * alpha_corner = a_sub[0] (top-right corner)
         * beta_corner = a_sup[sn-1] (bottom-left corner)
         * gamma = -a_dia[0]
         * u = [gamma, 0, ..., 0, beta_corner]^T
         * v = [1, 0, ..., 0, alpha_corner/gamma]^T
         * Modified T: T[0][0] -= gamma, T[sn-1][sn-1] -= beta_corner*alpha_corner/gamma
         */
        alpha_corner = a_sub[0];  /* h[n-2] */
        beta_corner = a_sup[sn - 1]; /* h[n-2] */
        gamma_val = -a_dia[0];

        /* Modify diagonal to form T */
        a_dia[0] -= gamma_val;  /* a_dia[0] = 2*original */
        a_dia[sn - 1] -= beta_corner * alpha_corner / gamma_val;

        /* Zero out corners in T */
        a_sub[0] = 0.0;
        a_sup[sn - 1] = 0.0;

        /* u vector */
        for (j = 0; j < sn; j++) u_vec[j] = 0.0;
        u_vec[0] = gamma_val;
        u_vec[sn - 1] = beta_corner;

        /* Solve T*y = b */
        memcpy(y_sol, b_vec, sn * sizeof(lmmc_real_t));
        {
            /* Need copies of a_sub, a_dia, a_sup for Thomas algorithm */
            lmmc_real_t* ts = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
            lmmc_real_t* td = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
            lmmc_real_t* tu = (lmmc_real_t*)lmmc_alloc(sn * sizeof(lmmc_real_t));
            if (!ts || !td || !tu) {
                lmmc_free(ts); lmmc_free(td); lmmc_free(tu);
                lmmc_free(a_sub); lmmc_free(a_dia); lmmc_free(a_sup);
                lmmc_free(b_vec); lmmc_free(u_vec); lmmc_free(y_sol); lmmc_free(z_sol);
                lmmc_free(h); lmmc_free(M);
                lmmc_interp_cspline_destroy(spline);
                return LMMC_STATUS_ALLOCATION_FAILED;
            }
            memcpy(ts, a_sub, sn * sizeof(lmmc_real_t));
            memcpy(td, a_dia, sn * sizeof(lmmc_real_t));
            memcpy(tu, a_sup, sn * sizeof(lmmc_real_t));
            tridiag_solve(sn, ts, td, tu, y_sol);

            /* Solve T*z = u */
            memcpy(ts, a_sub, sn * sizeof(lmmc_real_t));
            memcpy(td, a_dia, sn * sizeof(lmmc_real_t));
            memcpy(tu, a_sup, sn * sizeof(lmmc_real_t));
            memcpy(z_sol, u_vec, sn * sizeof(lmmc_real_t));
            tridiag_solve(sn, ts, td, tu, z_sol);

            lmmc_free(ts); lmmc_free(td); lmmc_free(tu);
        }

        /* x = y - z * (v^T * y) / (1 + v^T * z)
         * v = [1, 0, ..., 0, alpha_corner/gamma]^T
         */
        vy = y_sol[0] + (alpha_corner / gamma_val) * y_sol[sn - 1];
        vz = z_sol[0] + (alpha_corner / gamma_val) * z_sol[sn - 1];

        for (j = 0; j < sn; j++) {
            M[j] = y_sol[j] - z_sol[j] * vy / (1.0 + vz);
        }
        M[nm1] = M[0]; /* periodic: M[n-1] = M[0] */

        lmmc_free(a_sub); lmmc_free(a_dia); lmmc_free(a_sup);
        lmmc_free(b_vec); lmmc_free(u_vec); lmmc_free(y_sol); lmmc_free(z_sol);
    }

    /* Compute polynomial coefficients from moments */
    cspline_compute_coeffs(spline, h, M);

    lmmc_free(h);
    lmmc_free(M);

    *out_spline = spline;
    return LMMC_STATUS_OK;
}

/* Original create function delegates to create_ex with natural BC */
lmmc_status_t lmmc_interp_cspline_create(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n,
    lmmc_interp_cspline_t** out_spline)
{
    return lmmc_interp_cspline_create_ex(xs, ys, n, LMMC_SPLINE_NATURAL, 0.0, 0.0, out_spline);
}

/* ========================================================================
 * PCHIP (Fritsch-Carlson) 单调三次插值
 * ======================================================================== */

struct lmmc_interp_pchip_t {
    size_t n;
    lmmc_real_t* xs;
    lmmc_real_t* ys;
    lmmc_real_t* d; /* derivatives at each node */
};

void lmmc_interp_pchip_destroy(lmmc_interp_pchip_t* p)
{
    if (!p) return;
    lmmc_free(p->xs);
    lmmc_free(p->ys);
    lmmc_free(p->d);
    lmmc_free(p);
}

lmmc_status_t lmmc_interp_pchip_create(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n,
    lmmc_interp_pchip_t** out)
{
    lmmc_interp_pchip_t* p = NULL;
    lmmc_real_t* delta = NULL; /* slopes of each segment */
    size_t i;

    if (!xs || !ys || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (n < 2) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!interp_check_strictly_increasing(xs, n))
        return LMMC_STATUS_INVALID_ARGUMENT;

    p = (lmmc_interp_pchip_t*)lmmc_alloc(sizeof(*p));
    if (!p) return LMMC_STATUS_ALLOCATION_FAILED;
    p->n = n; p->xs = NULL; p->ys = NULL; p->d = NULL;

    p->xs = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    p->ys = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    p->d = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    delta = (lmmc_real_t*)lmmc_alloc((n - 1) * sizeof(lmmc_real_t));
    if (!p->xs || !p->ys || !p->d || !delta) {
        lmmc_free(delta);
        lmmc_interp_pchip_destroy(p);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(p->xs, xs, n * sizeof(lmmc_real_t));
    memcpy(p->ys, ys, n * sizeof(lmmc_real_t));

    /* Compute segment slopes */
    for (i = 0; i < n - 1; i++) {
        delta[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    }

    if (n == 2) {
        /* Only one segment: use the segment slope */
        p->d[0] = delta[0];
        p->d[1] = delta[0];
    } else {
        /* Fritsch-Carlson method for interior points */
        for (i = 1; i < n - 1; i++) {
            if (delta[i - 1] * delta[i] <= 0.0) {
                /* Sign change or zero: set derivative to zero for monotonicity */
                p->d[i] = 0.0;
            } else {
                /* Harmonic mean weighted by interval widths */
                lmmc_real_t h1 = xs[i] - xs[i - 1];
                lmmc_real_t h2 = xs[i + 1] - xs[i];
                lmmc_real_t w1 = 2.0 * h2 + h1;
                lmmc_real_t w2 = h2 + 2.0 * h1;
                p->d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            }
        }

        /* Endpoint derivatives: one-sided shape-preserving */
        /* Left endpoint */
        p->d[0] = ((2.0 * (xs[1] - xs[0]) + (xs[2] - xs[1])) * delta[0]
                  - (xs[1] - xs[0]) * delta[1]) / (xs[2] - xs[0]);
        if (p->d[0] * delta[0] <= 0.0) {
            p->d[0] = 0.0;
        } else if (delta[0] * delta[1] <= 0.0 &&
                   fabs(p->d[0]) > 3.0 * fabs(delta[0])) {
            p->d[0] = 3.0 * delta[0];
        }

        /* Right endpoint */
        p->d[n - 1] = ((2.0 * (xs[n - 1] - xs[n - 2]) + (xs[n - 2] - xs[n - 3])) * delta[n - 2]
                      - (xs[n - 1] - xs[n - 2]) * delta[n - 3]) / (xs[n - 1] - xs[n - 3]);
        if (p->d[n - 1] * delta[n - 2] <= 0.0) {
            p->d[n - 1] = 0.0;
        } else if (delta[n - 2] * delta[n - 3] <= 0.0 &&
                   fabs(p->d[n - 1]) > 3.0 * fabs(delta[n - 2])) {
            p->d[n - 1] = 3.0 * delta[n - 2];
        }
    }

    lmmc_free(delta);
    *out = p;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_interp_pchip_eval(
    const lmmc_interp_pchip_t* p, lmmc_real_t x, lmmc_real_t* out_y)
{
    size_t seg;
    lmmc_real_t h, t, a, b;
    if (!p || !out_y) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < p->xs[0] || x > p->xs[p->n - 1]) return LMMC_STATUS_OUT_OF_RANGE;
    if (x == p->xs[p->n - 1]) { *out_y = p->ys[p->n - 1]; return LMMC_STATUS_OK; }

    seg = interp_find_interval(p->xs, p->n, x);
    h = p->xs[seg + 1] - p->xs[seg];
    t = (x - p->xs[seg]) / h;

    /* Hermite basis evaluation:
     * p(t) = (1-t)^2*(1+2t)*y0 + t^2*(3-2t)*y1
     *       + t*(1-t)^2*h*d0 - t^2*(1-t)*h*d1
     */
    a = (1.0 - t);
    b = t;
    *out_y = a * a * (1.0 + 2.0 * t) * p->ys[seg]
           + b * b * (3.0 - 2.0 * t) * p->ys[seg + 1]
           + t * a * a * h * p->d[seg]
           - b * b * a * h * p->d[seg + 1];
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Akima 局部三次插值
 * ======================================================================== */

struct lmmc_interp_akima_t {
    size_t n;
    lmmc_real_t* xs;
    lmmc_real_t* ys;
    lmmc_real_t* d; /* derivatives at each node */
};

void lmmc_interp_akima_destroy(lmmc_interp_akima_t* a)
{
    if (!a) return;
    lmmc_free(a->xs);
    lmmc_free(a->ys);
    lmmc_free(a->d);
    lmmc_free(a);
}

lmmc_status_t lmmc_interp_akima_create(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n,
    lmmc_interp_akima_t** out)
{
    lmmc_interp_akima_t* a = NULL;
    lmmc_real_t* m = NULL; /* extended slopes: m[-2]..m[n] stored as m[0..n+3] */
    size_t i, nm1;

    if (!xs || !ys || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (n < 5) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!interp_check_strictly_increasing(xs, n))
        return LMMC_STATUS_INVALID_ARGUMENT;

    nm1 = n - 1;

    a = (lmmc_interp_akima_t*)lmmc_alloc(sizeof(*a));
    if (!a) return LMMC_STATUS_ALLOCATION_FAILED;
    a->n = n; a->xs = NULL; a->ys = NULL; a->d = NULL;

    a->xs = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    a->ys = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    a->d = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    /* Extended slopes: indices 0..n+2 map to m[-2]..m[n] */
    m = (lmmc_real_t*)lmmc_alloc((n + 3) * sizeof(lmmc_real_t));
    if (!a->xs || !a->ys || !a->d || !m) {
        lmmc_free(m);
        lmmc_interp_akima_destroy(a);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(a->xs, xs, n * sizeof(lmmc_real_t));
    memcpy(a->ys, ys, n * sizeof(lmmc_real_t));

    /* Compute interior slopes m[i] = (y[i+1]-y[i])/(x[i+1]-x[i]) for i=0..n-2
     * Stored at index i+2 in the extended array (offset by 2). */
    for (i = 0; i < nm1; i++) {
        m[i + 2] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    }

    /* Extend slopes at boundaries using Akima's extrapolation:
     * m[-1] = 2*m[0] - m[1]
     * m[-2] = 2*m[-1] - m[0]
     * m[n-1] = 2*m[n-2] - m[n-3]
     * m[n]   = 2*m[n-1] - m[n-2]
     */
    m[1] = 2.0 * m[2] - m[3];       /* m[-1] at index 1 */
    m[0] = 2.0 * m[1] - m[2];       /* m[-2] at index 0 */
    m[nm1 + 2] = 2.0 * m[nm1 + 1] - m[nm1]; /* m[n-1] at index n+1 */
    m[nm1 + 3] = 2.0 * m[nm1 + 2] - m[nm1 + 1]; /* m[n] at index n+2 */

    /* Compute Akima derivatives:
     * d[i] = (w1*m[i] + w2*m[i-1]) / (w1 + w2)
     * where w1 = |m[i+1] - m[i]|, w2 = |m[i-1] - m[i-2]|
     * Using extended array: m_ext[i+2] = m[i]
     * So for node i: use m_ext[i], m_ext[i+1], m_ext[i+2], m_ext[i+3]
     *   w1 = |m_ext[i+3] - m_ext[i+2]|
     *   w2 = |m_ext[i+1] - m_ext[i]|
     *   d[i] = (w1*m_ext[i+1] + w2*m_ext[i+2]) / (w1+w2)
     */
    for (i = 0; i < n; i++) {
        lmmc_real_t w1 = fabs(m[i + 3] - m[i + 2]);
        lmmc_real_t w2 = fabs(m[i + 1] - m[i]);
        if (w1 + w2 < 1e-30) {
            /* All slopes equal: use average */
            a->d[i] = 0.5 * (m[i + 1] + m[i + 2]);
        } else {
            a->d[i] = (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2);
        }
    }

    lmmc_free(m);
    *out = a;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_interp_akima_eval(
    const lmmc_interp_akima_t* a, lmmc_real_t x, lmmc_real_t* out_y)
{
    size_t seg;
    lmmc_real_t h, t, omt;
    if (!a || !out_y) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < a->xs[0] || x > a->xs[a->n - 1]) return LMMC_STATUS_OUT_OF_RANGE;
    if (x == a->xs[a->n - 1]) { *out_y = a->ys[a->n - 1]; return LMMC_STATUS_OK; }

    seg = interp_find_interval(a->xs, a->n, x);
    h = a->xs[seg + 1] - a->xs[seg];
    t = (x - a->xs[seg]) / h;
    omt = 1.0 - t;

    /* Hermite basis */
    *out_y = omt * omt * (1.0 + 2.0 * t) * a->ys[seg]
           + t * t * (3.0 - 2.0 * t) * a->ys[seg + 1]
           + t * omt * omt * h * a->d[seg]
           - t * t * omt * h * a->d[seg + 1];
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Lagrange 多项式插值
 * ======================================================================== */

struct lmmc_interp_lagrange_t {
    size_t n;
    lmmc_real_t* xs;
    lmmc_real_t* ys;
    lmmc_real_t* weights;
};

lmmc_status_t lmmc_interp_lagrange_create(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n,
    lmmc_interp_lagrange_t** out_lagrange)
{
    lmmc_interp_lagrange_t* lag = NULL;
    size_t i, j, alloc_size;

    if (!xs || !ys || !out_lagrange) return LMMC_STATUS_INVALID_ARGUMENT;
    if (n < 1) return LMMC_STATUS_INVALID_ARGUMENT;

    lag = (lmmc_interp_lagrange_t*)lmmc_alloc(sizeof(*lag));
    if (!lag) return LMMC_STATUS_ALLOCATION_FAILED;
    lag->n = n; lag->xs = NULL; lag->ys = NULL; lag->weights = NULL;

    if (!lmmc_safe_mul_size(n, sizeof(lmmc_real_t), &alloc_size)) {
        lmmc_free(lag);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    lag->xs = (lmmc_real_t*)lmmc_alloc(alloc_size);
    lag->ys = (lmmc_real_t*)lmmc_alloc(alloc_size);
    lag->weights = (lmmc_real_t*)lmmc_alloc(alloc_size);
    if (!lag->xs || !lag->ys || !lag->weights) {
        lmmc_interp_lagrange_destroy(lag);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(lag->xs, xs, alloc_size);
    memcpy(lag->ys, ys, alloc_size);

    /* Barycentric weights */
    for (j = 0; j < n; j++) {
        lmmc_real_t prod = 1.0;
        for (i = 0; i < n; i++) {
            if (i != j) prod *= (xs[j] - xs[i]);
        }
        lag->weights[j] = 1.0 / prod;
    }

    *out_lagrange = lag;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_interp_lagrange_eval(
    const lmmc_interp_lagrange_t* lagrange,
    lmmc_real_t query_x, lmmc_real_t* out_y)
{
    size_t j;
    lmmc_real_t numer, denom, diff, term;
    if (!lagrange || !out_y) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Check if query_x is exactly a node */
    for (j = 0; j < lagrange->n; j++) {
        if (query_x == lagrange->xs[j]) {
            *out_y = lagrange->ys[j];
            return LMMC_STATUS_OK;
        }
    }

    numer = 0.0;
    denom = 0.0;
    for (j = 0; j < lagrange->n; j++) {
        diff = query_x - lagrange->xs[j];
        term = lagrange->weights[j] / diff;
        numer += term * lagrange->ys[j];
        denom += term;
    }
    *out_y = numer / denom;
    return LMMC_STATUS_OK;
}

void lmmc_interp_lagrange_destroy(lmmc_interp_lagrange_t* lagrange)
{
    if (!lagrange) return;
    lmmc_free(lagrange->weights);
    lmmc_free(lagrange->ys);
    lmmc_free(lagrange->xs);
    lmmc_free(lagrange);
}

/* ========================================================================
 * 二维插值（矩形网格）
 * ======================================================================== */

lmmc_status_t lmmc_interp_bilinear(
    const lmmc_real_t* xs, size_t nx,
    const lmmc_real_t* ys, size_t ny,
    const lmmc_real_t* zs,
    lmmc_real_t qx, lmmc_real_t qy,
    lmmc_real_t* out_z)
{
    size_t ix, iy;
    lmmc_real_t tx, ty;
    lmmc_real_t z00, z01, z10, z11;

    if (!xs || !ys || !zs || !out_z) return LMMC_STATUS_INVALID_ARGUMENT;
    if (nx < 2 || ny < 2) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!interp_check_strictly_increasing(xs, nx))
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (!interp_check_strictly_increasing(ys, ny))
        return LMMC_STATUS_INVALID_ARGUMENT;

    if (qx < xs[0] || qx > xs[nx - 1]) return LMMC_STATUS_OUT_OF_RANGE;
    if (qy < ys[0] || qy > ys[ny - 1]) return LMMC_STATUS_OUT_OF_RANGE;

    /* Find intervals */
    ix = interp_find_interval(xs, nx, qx);
    iy = interp_find_interval(ys, ny, qy);

    /* Clamp to last valid interval */
    if (ix >= nx - 1) ix = nx - 2;
    if (iy >= ny - 1) iy = ny - 2;

    tx = (qx - xs[ix]) / (xs[ix + 1] - xs[ix]);
    ty = (qy - ys[iy]) / (ys[iy + 1] - ys[iy]);

    /* Grid values: zs[i*ny + j] = f(xs[i], ys[j]) */
    z00 = zs[ix * ny + iy];
    z01 = zs[ix * ny + (iy + 1)];
    z10 = zs[(ix + 1) * ny + iy];
    z11 = zs[(ix + 1) * ny + (iy + 1)];

    *out_z = (1.0 - tx) * (1.0 - ty) * z00
           + (1.0 - tx) * ty * z01
           + tx * (1.0 - ty) * z10
           + tx * ty * z11;
    return LMMC_STATUS_OK;
}

/**
 * @internal
 * @brief Cubic interpolation kernel using Catmull-Rom weights.
 * Given 4 values p[0..3] at positions -1, 0, 1, 2 (normalized),
 * interpolate at position t in [0,1].
 */
static lmmc_real_t cubic_interp_1d(lmmc_real_t p0, lmmc_real_t p1,
                                    lmmc_real_t p2, lmmc_real_t p3,
                                    lmmc_real_t t)
{
    /* Catmull-Rom spline */
    lmmc_real_t t2 = t * t;
    lmmc_real_t t3 = t2 * t;
    return 0.5 * ((-p0 + 3.0*p1 - 3.0*p2 + p3) * t3
                + (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * t2
                + (-p0 + p2) * t
                + 2.0*p1);
}

lmmc_status_t lmmc_interp_bicubic(
    const lmmc_real_t* xs, size_t nx,
    const lmmc_real_t* ys, size_t ny,
    const lmmc_real_t* zs,
    lmmc_real_t qx, lmmc_real_t qy,
    lmmc_real_t* out_z)
{
    size_t ix, iy;
    lmmc_real_t tx, ty;
    int i, j;
    size_t xi, yj;
    lmmc_real_t col_vals[4];
    lmmc_real_t row_vals[4];

    if (!xs || !ys || !zs || !out_z) return LMMC_STATUS_INVALID_ARGUMENT;
    if (nx < 4 || ny < 4) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!interp_check_strictly_increasing(xs, nx))
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (!interp_check_strictly_increasing(ys, ny))
        return LMMC_STATUS_INVALID_ARGUMENT;

    if (qx < xs[0] || qx > xs[nx - 1]) return LMMC_STATUS_OUT_OF_RANGE;
    if (qy < ys[0] || qy > ys[ny - 1]) return LMMC_STATUS_OUT_OF_RANGE;

    /* Find intervals */
    ix = interp_find_interval(xs, nx, qx);
    iy = interp_find_interval(ys, ny, qy);
    if (ix >= nx - 1) ix = nx - 2;
    if (iy >= ny - 1) iy = ny - 2;

    tx = (qx - xs[ix]) / (xs[ix + 1] - xs[ix]);
    ty = (qy - ys[iy]) / (ys[iy + 1] - ys[iy]);

    /* For bicubic, we need a 4x4 neighborhood.
     * Center the stencil: use indices ix-1, ix, ix+1, ix+2 (clamped).
     * Similarly for iy.
     */
    /* Interpolate along y for each of the 4 x-rows */
    for (i = -1; i <= 2; i++) {
        /* Clamp x index */
        xi = ix + (size_t)i;
        if (i == -1) {
            xi = (ix > 0) ? ix - 1 : 0;
        } else {
            xi = ix + (size_t)i;
            if (xi >= nx) xi = nx - 1;
        }

        /* Get 4 y-values for this x-row */
        for (j = -1; j <= 2; j++) {
            yj = iy + (size_t)j;
            if (j == -1) {
                yj = (iy > 0) ? iy - 1 : 0;
            } else {
                yj = iy + (size_t)j;
                if (yj >= ny) yj = ny - 1;
            }
            row_vals[j + 1] = zs[xi * ny + yj];
        }

        /* Cubic interpolation along y */
        col_vals[i + 1] = cubic_interp_1d(row_vals[0], row_vals[1],
                                           row_vals[2], row_vals[3], ty);
    }

    /* Cubic interpolation along x */
    *out_z = cubic_interp_1d(col_vals[0], col_vals[1], col_vals[2], col_vals[3], tx);
    return LMMC_STATUS_OK;
}
