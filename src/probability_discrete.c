#include <math.h>

#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"

#include "statistics_internal.h"

lmmc_status_t lmmc_dist_binomial_pmf(size_t k, size_t n, lmmc_real_t p,
                                      lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p < 0.0 || p > 1.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (k > n) { *out = 0.0; return LMMC_STATUS_OK; }

    /* Use log to avoid overflow: log(C(n,k)) + k*log(p) + (n-k)*log(1-p) */
    if (p == 0.0) {
        *out = (k == 0) ? 1.0 : 0.0;
        return LMMC_STATUS_OK;
    }
    if (p == 1.0) {
        *out = (k == n) ? 1.0 : 0.0;
        return LMMC_STATUS_OK;
    }

    {
        lmmc_real_t log_pmf = lgamma((double)(n + 1)) - lgamma((double)(k + 1)) -
                              lgamma((double)(n - k + 1)) +
                              (double)k * log(p) + (double)(n - k) * log(1.0 - p);
        *out = exp(log_pmf);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_binomial_cdf(size_t k, size_t n, lmmc_real_t p_param,
                                      lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p_param < 0.0 || p_param > 1.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (k >= n) { *out = 1.0; return LMMC_STATUS_OK; }

    /* Use regularized incomplete beta: CDF = I_{1-p}(n-k, k+1) */
    *out = regularized_beta(1.0 - p_param, (lmmc_real_t)(n - k), (lmmc_real_t)(k + 1));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_poisson_pmf(size_t k, lmmc_real_t lambda,
                                     lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lambda < 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lambda == 0.0) {
        *out = (k == 0) ? 1.0 : 0.0;
        return LMMC_STATUS_OK;
    }

    {
        lmmc_real_t log_pmf = (double)k * log(lambda) - lambda -
                              lgamma((double)(k + 1));
        *out = exp(log_pmf);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_poisson_cdf(size_t k, lmmc_real_t lambda,
                                     lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lambda < 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lambda == 0.0) { *out = 1.0; return LMMC_STATUS_OK; }

    /* CDF = Q(k+1, lambda) = 1 - P(k+1, lambda) = upper regularized gamma */
    *out = 1.0 - regularized_gamma_lower((lmmc_real_t)(k + 1), lambda);
    return LMMC_STATUS_OK;
}
