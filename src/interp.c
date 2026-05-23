/**
 * @file interp.c
 * @brief Interpolation module implementation.
 *
 * Provides linear interpolation, cubic spline interpolation, and
 * Lagrange interpolation (barycentric form).
 */

#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/interp.h"

#include <stddef.h>
#include <string.h>

/* ========================================================================
 * Linear Interpolation
 * ======================================================================== */

lmmc_status_t lmmc_interp_linear(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_real_t query_x,
    lmmc_real_t* out_y)
{
    size_t lo, hi, mid;
    lmmc_real_t x0, x1, y0, y1, t;

    /* Parameter validation */
    if (xs == NULL || ys == NULL || out_y == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n < 2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Range check: query_x must be within [xs[0], xs[n-1]] */
    if (query_x < xs[0] || query_x > xs[n - 1]) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }

    /* Handle exact match at the last point */
    if (query_x == xs[n - 1]) {
        *out_y = ys[n - 1];
        return LMMC_STATUS_OK;
    }

    /* Binary search to find interval [xs[lo], xs[lo+1]] containing query_x */
    lo = 0;
    hi = n - 1;
    while (hi - lo > 1) {
        mid = lo + (hi - lo) / 2;
        if (xs[mid] <= query_x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    /* Linear interpolation: y = y0 + (y1 - y0) * (query_x - x0) / (x1 - x0) */
    x0 = xs[lo];
    x1 = xs[lo + 1];
    y0 = ys[lo];
    y1 = ys[lo + 1];

    t = (query_x - x0) / (x1 - x0);
    *out_y = y0 + (y1 - y0) * t;

    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Cubic Spline Interpolation (Natural Boundary Conditions)
 * ======================================================================== */

struct lmmc_interp_cspline_t {
    size_t n;
    lmmc_real_t* xs;
    lmmc_real_t* ys;
    lmmc_real_t* coeffs;  /* 4*(n-1) coefficients: a,b,c,d per segment */
};

lmmc_status_t lmmc_interp_cspline_create(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_interp_cspline_t** out_spline)
{
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_real_t* h = NULL;    /* h[i] = xs[i+1] - xs[i] */
    lmmc_real_t* mu = NULL;   /* sub-diagonal ratio for Thomas algorithm */
    lmmc_real_t* z = NULL;    /* solution vector (second derivatives M[i]) */
    lmmc_real_t* l = NULL;    /* diagonal for Thomas algorithm */
    size_t i;
    size_t nm1; /* n - 1 */

    /* Parameter validation */
    if (xs == NULL || ys == NULL || out_spline == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n < 3) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Verify xs is strictly increasing */
    for (i = 0; i < n - 1; i++) {
        if (xs[i + 1] <= xs[i]) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }

    nm1 = n - 1;

    /* Allocate the spline structure */
    spline = (lmmc_interp_cspline_t*)lmmc_alloc(sizeof(lmmc_interp_cspline_t));
    if (spline == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    spline->n = n;
    spline->xs = NULL;
    spline->ys = NULL;
    spline->coeffs = NULL;

    /* Allocate and copy xs */
    spline->xs = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (spline->xs == NULL) {
        lmmc_interp_cspline_destroy(spline);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(spline->xs, xs, n * sizeof(lmmc_real_t));

    /* Allocate and copy ys */
    spline->ys = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (spline->ys == NULL) {
        lmmc_interp_cspline_destroy(spline);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(spline->ys, ys, n * sizeof(lmmc_real_t));

    /* Allocate coefficients: 4 per segment */
    spline->coeffs = (lmmc_real_t*)lmmc_alloc(4 * nm1 * sizeof(lmmc_real_t));
    if (spline->coeffs == NULL) {
        lmmc_interp_cspline_destroy(spline);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* Allocate temporary arrays for tridiagonal solve */
    h = (lmmc_real_t*)lmmc_alloc(nm1 * sizeof(lmmc_real_t));
    l = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    mu = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    z = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));

    if (h == NULL || l == NULL || mu == NULL || z == NULL) {
        lmmc_free(h);
        lmmc_free(l);
        lmmc_free(mu);
        lmmc_free(z);
        lmmc_interp_cspline_destroy(spline);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* Compute step sizes h[i] = xs[i+1] - xs[i] */
    for (i = 0; i < nm1; i++) {
        h[i] = xs[i + 1] - xs[i];
    }

    /*
     * Solve the tridiagonal system for natural cubic spline.
     * Natural boundary conditions: M[0] = 0, M[n-1] = 0
     * where M[i] are the second derivatives at each knot.
     *
     * The tridiagonal system (for i = 1, ..., n-2):
     *   h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1]
     *       = 6*((ys[i+1]-ys[i])/h[i] - (ys[i]-ys[i-1])/h[i-1])
     *
     * With M[0] = M[n-1] = 0, we solve for M[1]...M[n-2].
     * We use the Thomas algorithm (forward elimination + back substitution).
     */

    /* Forward sweep (Thomas algorithm) */
    l[0] = 1.0;
    mu[0] = 0.0;
    z[0] = 0.0;

    for (i = 1; i < nm1; i++) {
        lmmc_real_t alpha;
        alpha = 6.0 * ((ys[i + 1] - ys[i]) / h[i]
                     - (ys[i] - ys[i - 1]) / h[i - 1]);
        l[i] = 2.0 * (h[i - 1] + h[i]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha - h[i - 1] * z[i - 1]) / l[i];
    }

    l[nm1] = 1.0;
    z[nm1] = 0.0;

    /* Back substitution: z[i] now holds M[i] (second derivatives) */
    for (i = nm1 - 1; i >= 1; i--) {
        z[i] = z[i] - mu[i] * z[i + 1];
    }
    /* z[0] = 0 (natural BC, already set) */

    /*
     * Compute cubic polynomial coefficients for each segment [xs[i], xs[i+1]].
     * The cubic polynomial in each segment is:
     *   S_i(x) = a_i + b_i*(x - xs[i]) + c_i*(x - xs[i])^2 + d_i*(x - xs[i])^3
     *
     * Where:
     *   a_i = ys[i]
     *   c_i = M[i] / 2
     *   d_i = (M[i+1] - M[i]) / (6 * h[i])
     *   b_i = (ys[i+1] - ys[i]) / h[i] - h[i] * (2*M[i] + M[i+1]) / 6
     */
    for (i = 0; i < nm1; i++) {
        lmmc_real_t ai, bi, ci, di;
        ai = ys[i];
        ci = z[i] / 2.0;
        di = (z[i + 1] - z[i]) / (6.0 * h[i]);
        bi = (ys[i + 1] - ys[i]) / h[i]
           - h[i] * (2.0 * z[i] + z[i + 1]) / 6.0;

        spline->coeffs[4 * i + 0] = ai;
        spline->coeffs[4 * i + 1] = bi;
        spline->coeffs[4 * i + 2] = ci;
        spline->coeffs[4 * i + 3] = di;
    }

    /* Free temporary arrays */
    lmmc_free(h);
    lmmc_free(l);
    lmmc_free(mu);
    lmmc_free(z);

    *out_spline = spline;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_interp_cspline_eval(
    const lmmc_interp_cspline_t* spline,
    lmmc_real_t query_x,
    lmmc_real_t* out_y)
{
    size_t lo, hi, mid;
    size_t seg;
    lmmc_real_t dx, a, b, c, d;

    /* Parameter validation */
    if (spline == NULL || out_y == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Range check */
    if (query_x < spline->xs[0] || query_x > spline->xs[spline->n - 1]) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }

    /* Handle exact match at the last point */
    if (query_x == spline->xs[spline->n - 1]) {
        *out_y = spline->ys[spline->n - 1];
        return LMMC_STATUS_OK;
    }

    /* Binary search to find segment index: xs[seg] <= query_x < xs[seg+1] */
    lo = 0;
    hi = spline->n - 1;
    while (hi - lo > 1) {
        mid = lo + (hi - lo) / 2;
        if (spline->xs[mid] <= query_x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    seg = lo;

    /* Evaluate cubic polynomial: S(x) = a + b*dx + c*dx^2 + d*dx^3 */
    dx = query_x - spline->xs[seg];
    a = spline->coeffs[4 * seg + 0];
    b = spline->coeffs[4 * seg + 1];
    c = spline->coeffs[4 * seg + 2];
    d = spline->coeffs[4 * seg + 3];

    /* Horner's method: a + dx*(b + dx*(c + dx*d)) */
    *out_y = a + dx * (b + dx * (c + dx * d));

    return LMMC_STATUS_OK;
}

void lmmc_interp_cspline_destroy(lmmc_interp_cspline_t* spline)
{
    if (spline == NULL) {
        return;
    }
    if (spline->xs != NULL) {
        lmmc_free(spline->xs);
    }
    if (spline->ys != NULL) {
        lmmc_free(spline->ys);
    }
    if (spline->coeffs != NULL) {
        lmmc_free(spline->coeffs);
    }
    lmmc_free(spline);
}

/* ========================================================================
 * Lagrange Interpolation - Barycentric Form
 * ======================================================================== */

struct lmmc_interp_lagrange_t {
    size_t n;
    lmmc_real_t* xs;
    lmmc_real_t* ys;
    lmmc_real_t* weights;  /* barycentric weights */
};

lmmc_status_t lmmc_interp_lagrange_create(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_interp_lagrange_t** out_lagrange)
{
    lmmc_interp_lagrange_t* lag = NULL;
    size_t i, j;
    size_t alloc_size;

    /* Parameter validation */
    if (xs == NULL || ys == NULL || out_lagrange == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n < 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Allocate the context struct */
    lag = (lmmc_interp_lagrange_t*)lmmc_alloc(sizeof(lmmc_interp_lagrange_t));
    if (lag == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    lag->n = n;
    lag->xs = NULL;
    lag->ys = NULL;
    lag->weights = NULL;

    /* Allocate and copy xs */
    if (!lmmc_safe_mul_size(n, sizeof(lmmc_real_t), &alloc_size)) {
        lmmc_free(lag);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    lag->xs = (lmmc_real_t*)lmmc_alloc(alloc_size);
    if (lag->xs == NULL) {
        lmmc_free(lag);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(lag->xs, xs, alloc_size);

    /* Allocate and copy ys */
    lag->ys = (lmmc_real_t*)lmmc_alloc(alloc_size);
    if (lag->ys == NULL) {
        lmmc_free(lag->xs);
        lmmc_free(lag);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(lag->ys, ys, alloc_size);

    /* Allocate weights */
    lag->weights = (lmmc_real_t*)lmmc_alloc(alloc_size);
    if (lag->weights == NULL) {
        lmmc_free(lag->ys);
        lmmc_free(lag->xs);
        lmmc_free(lag);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* Compute barycentric weights: w[j] = 1 / prod_{k != j} (xs[j] - xs[k]) */
    for (j = 0; j < n; j++) {
        lmmc_real_t prod = 1.0;
        for (i = 0; i < n; i++) {
            if (i != j) {
                prod *= (xs[j] - xs[i]);
            }
        }
        lag->weights[j] = 1.0 / prod;
    }

    *out_lagrange = lag;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_interp_lagrange_eval(
    const lmmc_interp_lagrange_t* lagrange,
    lmmc_real_t query_x,
    lmmc_real_t* out_y)
{
    size_t j;
    lmmc_real_t numer, denom, diff, term;

    /* Parameter validation */
    if (lagrange == NULL || out_y == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Check if query_x exactly matches a data node */
    for (j = 0; j < lagrange->n; j++) {
        if (query_x == lagrange->xs[j]) {
            *out_y = lagrange->ys[j];
            return LMMC_STATUS_OK;
        }
    }

    /* Second-form barycentric formula:
     * p(x) = sum_j( w[j]*y[j] / (x - x[j]) ) / sum_j( w[j] / (x - x[j]) )
     */
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
    if (lagrange == NULL) {
        return;
    }
    if (lagrange->weights != NULL) {
        lmmc_free(lagrange->weights);
    }
    if (lagrange->ys != NULL) {
        lmmc_free(lagrange->ys);
    }
    if (lagrange->xs != NULL) {
        lmmc_free(lagrange->xs);
    }
    lmmc_free(lagrange);
}
