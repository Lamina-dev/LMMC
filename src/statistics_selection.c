#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"

#include "statistics_internal.h"

/** @brief 比较 lmmc_real_t 元素，供 qsort 排序。 */
static int lmmc_real_compare(const void* a, const void* b) {
    lmmc_real_t va = *(const lmmc_real_t*)a;
    lmmc_real_t vb = *(const lmmc_real_t*)b;
    if (va < vb) { return -1; }
    if (va > vb) { return 1; }
    return 0;
}


static int lmmc_vec_values_are_finite(const lmmc_vec_t* x) {
    return lmmc_real_range_is_finite(x->data, x->size);
}

lmmc_status_t lmmc_vec_median(const lmmc_vec_t* x, lmmc_real_t* out) {
    lmmc_real_t* sorted = NULL;
    size_t n;

    if (out == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x == NULL || x->data == NULL || x->size == 0) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_vec_values_are_finite(x)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    n = x->size;
    sorted = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    if (sorted == NULL) { return LMMC_STATUS_ALLOCATION_FAILED; }

    for (size_t i = 0; i < n; ++i) {
        sorted[i] = x->data[i];
    }
    qsort(sorted, n, sizeof(lmmc_real_t), lmmc_real_compare);

    if (n % 2 == 1) {
        *out = sorted[n / 2];
    } else {
        *out = lmmc_interval_midpoint(
            sorted[n / 2 - 1], sorted[n / 2]);
    }

    lmmc_memory_free(sorted);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_quantile(const lmmc_vec_t* x, lmmc_real_t p, lmmc_real_t* out) {
    lmmc_real_t* sorted = NULL;
    size_t n;
    double h, frac;
    size_t lo, hi;

    if (out == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x == NULL || x->data == NULL || x->size == 0) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_is_finite(&p) || p < 0.0 || p > 1.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_vec_values_are_finite(x)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    n = x->size;
    sorted = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    if (sorted == NULL) { return LMMC_STATUS_ALLOCATION_FAILED; }

    for (size_t i = 0; i < n; ++i) {
        sorted[i] = x->data[i];
    }
    qsort(sorted, n, sizeof(lmmc_real_t), lmmc_real_compare);

    /** @brief 使用与 NumPy 默认方式相同的线性插值。 */
    h = p * (double)(n - 1);
    lo = (size_t)h;
    hi = lo + 1;
    frac = h - (double)lo;

    if (hi >= n) {
        *out = sorted[n - 1];
    } else {
        *out = sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }

    lmmc_memory_free(sorted);
    return LMMC_STATUS_OK;
}

static void lmmc_histogram_bounds(
    const lmmc_vec_t* x, lmmc_real_t* xmin, lmmc_real_t* xmax) {
    size_t i;
    (*xmin) = x->data[0];
    (*xmax) = x->data[0];
    for (i = 1; i < x->size; ++i) {
        if (x->data[i] < (*xmin)) {
            (*xmin) = x->data[i];
        }
        if (x->data[i] > (*xmax)) {
            (*xmax) = x->data[i];
        }
    }

    if ((*xmax) == (*xmin)) {
        const lmmc_real_t value = (*xmin);
        const lmmc_real_t lower = value - 0.5;
        const lmmc_real_t upper = value + 0.5;
        if (lower < value && upper > value &&
            lmmc_is_finite(&lower) && lmmc_is_finite(&upper)) {
            (*xmin) = lower;
            (*xmax) = upper;
        } else if (value > 0.0) {
            (*xmin) = value * 0.5;
            (*xmax) = value <= DBL_MAX / 1.5 ? value * 1.5 : value;
        } else if (value < 0.0) {
            (*xmin) = value >= -DBL_MAX / 1.5 ? value * 1.5 : value;
            (*xmax) = value * 0.5;
        } else {
            (*xmin) = -0.5;
            (*xmax) = 0.5;
        }
    }
}

static void lmmc_histogram_edges(
    lmmc_real_t xmin, lmmc_real_t xmax, size_t nbins, lmmc_real_t* edges) {
    size_t i;
    edges[0] = xmin;
    for (i = 1; i < nbins; ++i) {
        const lmmc_real_t t =
            (lmmc_real_t)i / (lmmc_real_t)nbins;
        if ((xmin < 0.0 && xmax > 0.0) ||
            (xmin > 0.0 && xmax < 0.0)) {
            edges[i] = (1.0 - t) * xmin + t * xmax;
        } else {
            edges[i] = xmin + t * (xmax - xmin);
        }
    }
    edges[nbins] = xmax;

}

static size_t lmmc_histogram_bin(
    lmmc_real_t value, size_t nbins, lmmc_real_t xmin, lmmc_real_t xmax,
    int crosses_zero, lmmc_real_t range_scale, lmmc_real_t scaled_min,
    lmmc_real_t scaled_span) {
    size_t bin;
    if (value <= xmin) {
        bin = 0;
    } else if (value >= xmax) {
        bin = nbins - 1;
    } else {
        lmmc_real_t position;
        lmmc_real_t bin_position;
        if (crosses_zero) {
            position =
                (value / range_scale - scaled_min) /
                scaled_span;
        } else {
            position =
                (value - xmin) / (xmax - xmin);
        }
        bin_position = position * (lmmc_real_t)nbins;
        if (position >= 1.0 ||
            bin_position >= (lmmc_real_t)(nbins - 1)) {
            bin = nbins - 1;
        } else if (position <= 0.0) {
            bin = 0;
        } else {
            bin = (size_t)bin_position;
        }
    }
    return bin;
}

lmmc_status_t lmmc_vec_histogram(const lmmc_vec_t* x, size_t nbins, lmmc_real_t* edges, size_t* counts) {
    size_t n, i;
    lmmc_real_t xmin, xmax, range_scale = 1.0;
    lmmc_real_t scaled_min = 0.0, scaled_span = 0.0;
    int crosses_zero;

    if (x == NULL || x->data == NULL || x->size == 0) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (nbins == 0 || edges == NULL || counts == NULL) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_vec_values_are_finite(x)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    n = x->size;

    lmmc_histogram_bounds(x, &xmin, &xmax);
    crosses_zero = xmin < 0.0 && xmax > 0.0;
    if (crosses_zero) {
        range_scale = fmax(-xmin, xmax);
        scaled_min = xmin / range_scale;
        scaled_span = xmax / range_scale - scaled_min;
    }


    lmmc_histogram_edges(xmin, xmax, nbins, edges);

    for (i = 0; i < nbins; ++i) {
        counts[i] = 0;
    }

    for (i = 0; i < n; ++i) {
        const size_t bin = lmmc_histogram_bin(x->data[i], nbins, xmin, xmax,
            crosses_zero, range_scale, scaled_min, scaled_span);
        counts[bin]++;
    }

    return LMMC_STATUS_OK;
}
