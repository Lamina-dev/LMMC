#include <math.h>

#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"

#include "statistics_internal.h"


lmmc_status_t lmmc_dist_binomial_pmf(size_t k, size_t n, lmmc_real_t p,
                                      lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(p) || p < 0.0 || p > 1.0)
        return LMMC_STATUS_INVALID_ARGUMENT;
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
        const lmmc_real_t trials = (lmmc_real_t)n;
        const lmmc_real_t successes = (lmmc_real_t)k;
        const lmmc_real_t failures = (lmmc_real_t)(n - k);
        const lmmc_real_t q = 1.0 - p;
        lmmc_real_t log_pmf;
        lmmc_real_t result;

        if (k == 0) {
            log_pmf = trials * log1p(-p);
        } else if (k == n) {
            log_pmf = trials * log(p);
        } else {
            const lmmc_real_t correction =
                lmmc_stirling_error(trials) -
                lmmc_stirling_error(successes) -
                lmmc_stirling_error(failures);
            const lmmc_real_t deviance =
                lmmc_deviance_part(successes, trials * p) +
                lmmc_deviance_part(failures, trials * q);
            const lmmc_real_t log_scale =
                2.0 * LMMC_LOG_SQRT_2PI + log(successes) +
                log1p(-successes / trials);
            log_pmf = correction - deviance - 0.5 * log_scale;
        }

        if (!isfinite(log_pmf)) {
            if (log_pmf == -INFINITY) {
                *out = 0.0;
                return LMMC_STATUS_OK;
            }
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        result = exp(log_pmf);
        if (!isfinite(result) || result < 0.0 || result > 1.0)
            return LMMC_STATUS_NUMERICAL_FAILURE;
        *out = result;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_binomial_cdf(size_t k, size_t n, lmmc_real_t p_param,
                                      lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(p_param) || p_param < 0.0 || p_param > 1.0)
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (k >= n) { *out = 1.0; return LMMC_STATUS_OK; }

    /* Use regularized incomplete beta: CDF = I_{1-p}(n-k, k+1) */
    return regularized_beta(
        1.0 - p_param,
        (lmmc_real_t)(n - k),
        (lmmc_real_t)(k + 1),
        out);
}

lmmc_status_t lmmc_dist_poisson_pmf(size_t k, lmmc_real_t lambda,
                                     lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(lambda) || lambda < 0.0)
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (lambda == 0.0) {
        *out = (k == 0) ? 1.0 : 0.0;
        return LMMC_STATUS_OK;
    }

    {
        const lmmc_real_t count = (lmmc_real_t)k;
        lmmc_real_t log_pmf;
        lmmc_real_t result;

        if (k == 0) {
            log_pmf = -lambda;
        } else {
            log_pmf =
                -lmmc_stirling_error(count) -
                lmmc_deviance_part(count, lambda) -
                0.5 * (2.0 * LMMC_LOG_SQRT_2PI + log(count));
        }
        if (!isfinite(log_pmf)) {
            if (log_pmf == -INFINITY) {
                *out = 0.0;
                return LMMC_STATUS_OK;
            }
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        result = exp(log_pmf);
        if (!isfinite(result) || result < 0.0 || result > 1.0)
            return LMMC_STATUS_NUMERICAL_FAILURE;
        *out = result;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_poisson_cdf(size_t k, lmmc_real_t lambda,
                                     lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(lambda) || lambda < 0.0)
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (lambda == 0.0) { *out = 1.0; return LMMC_STATUS_OK; }

    /* CDF = Q(k+1, lambda); evaluate the small upper-gamma tail directly. */
    return regularized_gamma_upper((lmmc_real_t)k + 1.0, lambda, out);
}
