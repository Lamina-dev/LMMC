#ifndef LMMC_PROBABILITY_INTERNAL_H
#define LMMC_PROBABILITY_INTERNAL_H

#include <math.h>
#include <float.h>
#include "statistics_internal.h"

static inline int valid_positive(lmmc_real_t value) {
    return isfinite(value) && value > 0.0;
}

static inline int valid_probability(lmmc_real_t value) {
    return isfinite(value) && value > 0.0 && value < 1.0;
}

static inline int student_t_uses_normal_limit(lmmc_real_t df) {
    return df >= 1.0 / DBL_EPSILON;
}

static inline lmmc_status_t store_density_from_log(
    lmmc_real_t log_density, lmmc_real_t* out) {
    lmmc_real_t result;
    if (isnan(log_density) || log_density > log(DBL_MAX)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    result = exp(log_density);
    if (!isfinite(result)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *out = result;
    return LMMC_STATUS_OK;
}

static inline lmmc_status_t store_probability_result(
    lmmc_real_t probability, lmmc_real_t* out) {
    if (!isfinite(probability) || probability < 0.0 || probability > 1.0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *out = probability;
    return LMMC_STATUS_OK;
}

#endif
