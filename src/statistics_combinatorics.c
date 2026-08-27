#include <math.h>

#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"

#include "statistics_internal.h"

static int lmmc_real_mul_overflows(lmmc_real_t value, lmmc_real_t factor) {
    return factor != 0.0 && value > (lmmc_real_t)(DBL_MAX / factor);
}

void lmmc_stats_factorial(lmmc_real_t* out_val, uint32_t n) {
    if (out_val == NULL) {
        return;
    }
    lmmc_real_t result = 1.0;
    for (uint32_t k = 2; k <= n; ++k) {
        lmmc_real_t factor = (lmmc_real_t)k;
        if (lmmc_real_mul_overflows(result, factor)) {
            LMMC_REAL_SET_D(out_val, INFINITY);
            return;
        }
        result *= factor;
    }
    LMMC_REAL_SET(out_val, &result);
}

void lmmc_stats_nPr(lmmc_real_t* out_val, uint32_t n, uint32_t r) {
    if (out_val == NULL) {
        return;
    }
    if (r > n) {
        LMMC_REAL_SET_D(out_val, 0.0);
        return;
    }
    lmmc_real_t result = 1.0;
    for (uint32_t k = 0; k < r; ++k) {
        lmmc_real_t factor = (lmmc_real_t)(n - k);
        if (lmmc_real_mul_overflows(result, factor)) {
            LMMC_REAL_SET_D(out_val, INFINITY);
            return;
        }
        result *= factor;
    }
    LMMC_REAL_SET(out_val, &result);
}

void lmmc_stats_nCr(lmmc_real_t* out_val, uint32_t n, uint32_t r) {
    if (out_val == NULL) {
        return;
    }
    if (r > n) {
        LMMC_REAL_SET_D(out_val, 0.0);
        return;
    }
    if (r > n - r) {
        r = n - r;
    }
    lmmc_real_t result = 1.0;
    for (uint32_t k = 1; k <= r; ++k) {
        lmmc_real_t numerator = (lmmc_real_t)(n - r + k);
        lmmc_real_t denominator = (lmmc_real_t)k;
        if (lmmc_real_mul_overflows(result, numerator)) {
            LMMC_REAL_SET_D(out_val, INFINITY);
            return;
        }
        result = (result * numerator) / denominator;
    }
    LMMC_REAL_SET(out_val, &result);
}

