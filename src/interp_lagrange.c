#include "memory_bridge.h"
#include "internal.h"
#include "internal/interp_internal.h"

#include <math.h>
#include <string.h>

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
    size_t alloc_size;
    lmmc_status_t status;

    if (!out_lagrange) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out_lagrange = NULL;
    if (!xs || !ys || n < 1 ||
        !interp_check_finite_values(xs, n) ||
        !interp_check_finite_values(ys, n)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lag = (lmmc_interp_lagrange_t*)lmmc_memory_alloc(sizeof(*lag));
    if (!lag) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    lag->n = n; lag->xs = NULL; lag->ys = NULL; lag->weights = NULL;

    if (!lmmc_safe_mul_size(n, sizeof(lmmc_real_t), &alloc_size)) {
        lmmc_memory_free(lag);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    lag->xs = (lmmc_real_t*)lmmc_memory_alloc(alloc_size);
    lag->ys = (lmmc_real_t*)lmmc_memory_alloc(alloc_size);
    lag->weights = (lmmc_real_t*)lmmc_memory_alloc(alloc_size);
    if (!lag->xs || !lag->ys || !lag->weights) {
        lmmc_interp_lagrange_destroy(lag);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(lag->xs, xs, alloc_size);
    memcpy(lag->ys, ys, alloc_size);

    status = interp_barycentric_weights(lag->xs, lag->n, lag->weights);
    if (status != LMMC_STATUS_OK) {
        lmmc_interp_lagrange_destroy(lag);
        return status;
    }

    *out_lagrange = lag;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_interp_lagrange_eval(
    const lmmc_interp_lagrange_t* lagrange,
    lmmc_real_t query_x, lmmc_real_t* out_y)
{
    size_t j;
    lmmc_real_t numer, denom, diff, term, result;
    if (!lagrange || !out_y || !isfinite(query_x)) {
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
    result = numer / denom;
    if (!isfinite(result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out_y = result;
    return LMMC_STATUS_OK;
}

void lmmc_interp_lagrange_destroy(lmmc_interp_lagrange_t* lagrange)
{
    if (!lagrange) {
        return;
    }
    lmmc_memory_free(lagrange->weights);
    lmmc_memory_free(lagrange->ys);
    lmmc_memory_free(lagrange->xs);
    lmmc_memory_free(lagrange);
}
