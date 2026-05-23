/**
 * @file interp.c
 * @brief 一维插值算法实现：线性、三次样条、Lagrange。
 */
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/interp.h"

#include <stddef.h>
#include <string.h>


lmmc_status_t lmmc_interp_linear(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_real_t query_x,
    lmmc_real_t* out_y)
{
    size_t lo, hi, mid;
    lmmc_real_t x0, x1, y0, y1, t;


    if (xs == NULL || ys == NULL || out_y == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n < 2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    if (query_x < xs[0] || query_x > xs[n - 1]) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }


    if (query_x == xs[n - 1]) {
        *out_y = ys[n - 1];
        return LMMC_STATUS_OK;
    }


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


    x0 = xs[lo];
    x1 = xs[lo + 1];
    y0 = ys[lo];
    y1 = ys[lo + 1];

    t = (query_x - x0) / (x1 - x0);
    *out_y = y0 + (y1 - y0) * t;

    return LMMC_STATUS_OK;
}


struct lmmc_interp_cspline_t {
    size_t n;
    lmmc_real_t* xs;
    lmmc_real_t* ys;
    lmmc_real_t* coeffs;
};

lmmc_status_t lmmc_interp_cspline_create(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_interp_cspline_t** out_spline)
{
    lmmc_interp_cspline_t* spline = NULL;
    lmmc_real_t* h = NULL;
    lmmc_real_t* mu = NULL;
    lmmc_real_t* z = NULL;
    lmmc_real_t* l = NULL;
    size_t i;
    size_t nm1;


    if (xs == NULL || ys == NULL || out_spline == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n < 3) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    for (i = 0; i < n - 1; i++) {
        if (xs[i + 1] <= xs[i]) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }

    nm1 = n - 1;


    spline = (lmmc_interp_cspline_t*)lmmc_alloc(sizeof(lmmc_interp_cspline_t));
    if (spline == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    spline->n = n;
    spline->xs = NULL;
    spline->ys = NULL;
    spline->coeffs = NULL;


    spline->xs = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (spline->xs == NULL) {
        lmmc_interp_cspline_destroy(spline);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(spline->xs, xs, n * sizeof(lmmc_real_t));


    spline->ys = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (spline->ys == NULL) {
        lmmc_interp_cspline_destroy(spline);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(spline->ys, ys, n * sizeof(lmmc_real_t));


    spline->coeffs = (lmmc_real_t*)lmmc_alloc(4 * nm1 * sizeof(lmmc_real_t));
    if (spline->coeffs == NULL) {
        lmmc_interp_cspline_destroy(spline);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }


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


    for (i = 0; i < nm1; i++) {
        h[i] = xs[i + 1] - xs[i];
    }


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


    for (i = nm1 - 1; i >= 1; i--) {
        z[i] = z[i] - mu[i] * z[i + 1];
    }


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


    if (spline == NULL || out_y == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    if (query_x < spline->xs[0] || query_x > spline->xs[spline->n - 1]) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }


    if (query_x == spline->xs[spline->n - 1]) {
        *out_y = spline->ys[spline->n - 1];
        return LMMC_STATUS_OK;
    }


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


    dx = query_x - spline->xs[seg];
    a = spline->coeffs[4 * seg + 0];
    b = spline->coeffs[4 * seg + 1];
    c = spline->coeffs[4 * seg + 2];
    d = spline->coeffs[4 * seg + 3];


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


struct lmmc_interp_lagrange_t {
    size_t n;
    lmmc_real_t* xs;
    lmmc_real_t* ys;
    lmmc_real_t* weights;
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


    if (xs == NULL || ys == NULL || out_lagrange == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n < 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    lag = (lmmc_interp_lagrange_t*)lmmc_alloc(sizeof(lmmc_interp_lagrange_t));
    if (lag == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    lag->n = n;
    lag->xs = NULL;
    lag->ys = NULL;
    lag->weights = NULL;


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


    lag->ys = (lmmc_real_t*)lmmc_alloc(alloc_size);
    if (lag->ys == NULL) {
        lmmc_free(lag->xs);
        lmmc_free(lag);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(lag->ys, ys, alloc_size);


    lag->weights = (lmmc_real_t*)lmmc_alloc(alloc_size);
    if (lag->weights == NULL) {
        lmmc_free(lag->ys);
        lmmc_free(lag->xs);
        lmmc_free(lag);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }


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


    if (lagrange == NULL || out_y == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


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
