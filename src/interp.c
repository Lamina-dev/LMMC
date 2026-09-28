#include "internal.h"
#include "internal/interp_internal.h"

#include <math.h>

size_t interp_find_interval(const lmmc_real_t* xs, size_t n,
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

int interp_check_strictly_increasing(
    const lmmc_real_t* xs, size_t n)
{
    size_t i;
    if (!xs || n == 0 || !isfinite(xs[0])) {
        return 0;
    }
    for (i = 1; i < n; ++i) {
        if (!isfinite(xs[i]) || !(xs[i] > xs[i - 1])) {
            return 0;
        }
    }
    return 1;
}

int interp_check_finite_values(
    const lmmc_real_t* values, size_t n)
{
    size_t i;
    if (!values && n != 0) {
        return 0;
    }
    for (i = 0; i < n; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

lmmc_status_t interp_barycentric_weights(const lmmc_real_t* nodes, size_t n,
                                        lmmc_real_t* weights)
{
    size_t i, j;
    lmmc_real_t largest = -INFINITY;
    for (j = 0; j < n; j++) {
        lmmc_real_t log_weight = 0.0;
        for (i = 0; i < n; i++) {
            if (i != j) {
                lmmc_real_t difference = nodes[j] - nodes[i];
                if (difference == 0.0) {
                    return LMMC_STATUS_INVALID_ARGUMENT;
                }
                if (!isfinite(difference)) {
                    log_weight -= log(fabs(nodes[j] / 2.0 - nodes[i] / 2.0)) + log(2.0);
                } else {
                    log_weight -= log(fabs(difference));
                }
            }
        }
        weights[j] = log_weight;
        if (log_weight > largest) largest = log_weight;
    }
    for (j = 0; j < n; j++) {
        weights[j] = exp(weights[j] - largest);
        if (weights[j] == 0.0 || !isfinite(weights[j])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        for (i = 0; i < n; i++) {
            if (nodes[j] < nodes[i]) weights[j] = -weights[j];
        }
    }
    return LMMC_STATUS_OK;
}

int interp_check_samples(const lmmc_real_t* xs, const lmmc_real_t* ys,
                         size_t n, size_t minimum)
{
    return xs && ys && n >= minimum &&
        interp_check_strictly_increasing(xs, n) &&
        interp_check_finite_values(ys, n);
}

static int interp_grid_coordinates_valid(const interp_grid_t* grid)
{
    size_t grid_size;
    if (!isfinite(grid->qx) || !isfinite(grid->qy)) {
        return 0;
    }
    if (!interp_check_strictly_increasing(grid->xs, grid->nx)) {
        return 0;
    }
    if (!interp_check_strictly_increasing(grid->ys, grid->ny)) {
        return 0;
    }
    return lmmc_safe_mul_size(grid->nx, grid->ny, &grid_size);
}

lmmc_status_t interp_check_grid(const interp_grid_t* grid, size_t minimum,
                               const lmmc_real_t* out)
{
    if (!grid->xs || !grid->ys || !grid->zs || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (grid->nx < minimum || grid->ny < minimum) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!interp_grid_coordinates_valid(grid)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (grid->qx < grid->xs[0] || grid->qx > grid->xs[grid->nx - 1] ||
        grid->qy < grid->ys[0] || grid->qy > grid->ys[grid->ny - 1]) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    return LMMC_STATUS_OK;
}

lmmc_real_t interp_interval_fraction(lmmc_real_t left,
                                     lmmc_real_t right,
                                     lmmc_real_t query)
{
    const lmmc_real_t span = right - left;
    lmmc_real_t scale, scaled_left, scaled_right;
    if (isfinite(span)) {
        return (query - left) / span;
    }
    scale = fmax(fabs(left), fabs(right));
    scaled_left = left / scale;
    scaled_right = right / scale;
    return (query / scale - scaled_left) / (scaled_right - scaled_left);
}

static lmmc_real_t interp_linear_value(lmmc_real_t left,
                                      lmmc_real_t right, lmmc_real_t t)
{
    if (t == 0.0) {
        return left;
    }
    if (t == 1.0) {
        return right;
    }
    if ((left <= 0.0 && right >= 0.0) || (left >= 0.0 && right <= 0.0)) {
        return (1.0 - t) * left + t * right;
    }
    return left + t * (right - left);
}

lmmc_status_t lmmc_interp_linear(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n,
    lmmc_real_t query_x, lmmc_real_t* out_y)
{
    size_t lo;
    lmmc_real_t t, result;
    if (!out_y || !isfinite(query_x) || !interp_check_samples(xs, ys, n, 2)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (query_x < xs[0] || query_x > xs[n - 1]) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    if (query_x == xs[n - 1]) {
        *out_y = ys[n - 1];
        return LMMC_STATUS_OK;
    }
    lo = interp_find_interval(xs, n, query_x);
    t = interp_interval_fraction(xs[lo], xs[lo + 1], query_x);
    result = interp_linear_value(ys[lo], ys[lo + 1], t);
    if (!isfinite(t) || !isfinite(result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out_y = result;
    return LMMC_STATUS_OK;
}
