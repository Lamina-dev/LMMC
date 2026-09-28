#include <math.h>
#include <float.h>

#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"

#include "probability_internal.h"

static lmmc_real_t standardized_normal_value(
    lmmc_real_t x, lmmc_real_t mu, lmmc_real_t sigma) {
    const lmmc_real_t centered = x - mu;
    if (isfinite(centered)) { return centered / sigma; }
    return x / sigma - mu / sigma;
}

lmmc_status_t lmmc_dist_normal_pdf(lmmc_real_t x, lmmc_real_t mu,
                                    lmmc_real_t sigma, lmmc_real_t* out) {
    lmmc_real_t z, log_density;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !isfinite(mu) || !valid_positive(sigma)) { return LMMC_STATUS_INVALID_ARGUMENT; }

    z = standardized_normal_value(x, mu, sigma);
    log_density =
        -0.5 * z * z - log(sigma) - log(LMMC_SQRT_2PI);
    return store_density_from_log(log_density, out);
}

lmmc_status_t lmmc_dist_normal_cdf(lmmc_real_t x, lmmc_real_t mu,
                                    lmmc_real_t sigma, lmmc_real_t* out) {
    lmmc_real_t z;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !isfinite(mu) || !valid_positive(sigma)) { return LMMC_STATUS_INVALID_ARGUMENT; }

    z = standardized_normal_value(x, mu, sigma);
    return store_probability_result(
        0.5 * erfc(-z / sqrt(2.0)), out);
}

static lmmc_real_t student_t_log_kernel(
    lmmc_real_t x, lmmc_real_t df) {
    const lmmc_real_t abs_x = fabs(x);
    const lmmc_real_t sqrt_df = sqrt(df);
    if (abs_x <= sqrt_df) {
        const lmmc_real_t ratio = abs_x / sqrt_df;
        return log1p(ratio * ratio);
    }
    {
        const lmmc_real_t ratio = sqrt_df / abs_x;
        return 2.0 * log(abs_x) - log(df) +
               log1p(ratio * ratio);
    }
}

static lmmc_real_t student_t_beta_argument(
    lmmc_real_t x, lmmc_real_t df) {
    const lmmc_real_t abs_x = fabs(x);
    const lmmc_real_t sqrt_df = sqrt(df);
    if (abs_x <= sqrt_df) {
        const lmmc_real_t ratio = abs_x / sqrt_df;
        const lmmc_real_t ratio_sq = ratio * ratio;
        return 1.0 / (1.0 + ratio_sq);
    }
    {
        const lmmc_real_t ratio = sqrt_df / abs_x;
        const lmmc_real_t ratio_sq = ratio * ratio;
        return ratio_sq / (1.0 + ratio_sq);
    }
}

lmmc_status_t lmmc_dist_t_pdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    lmmc_real_t log_density;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !valid_positive(df)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (student_t_uses_normal_limit(df)) {
        return store_density_from_log(
            -0.5 * x * x - LMMC_LOG_SQRT_2PI, out);
    }
    if (df / 2.0 == 0.0) {
        /* B(df/2, 1/2) ~ 2/df as df tends to zero. */
        log_density = 0.5 * log(df) - log(2.0) -
            0.5 * student_t_log_kernel(x, df);
        return store_density_from_log(log_density, out);
    }

    {
        double log_beta;
        const lmmc_status_t status = lmmc_log_beta(df / 2.0, 0.5, &log_beta);
        if (status != LMMC_STATUS_OK) { return status; }
        log_density = -log_beta - 0.5 * log(df) -
            ((df + 1.0) / 2.0) * student_t_log_kernel(x, df);
    }
    return store_density_from_log(log_density, out);
}

lmmc_status_t lmmc_dist_t_cdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    lmmc_real_t t_val, ib, result;
    lmmc_status_t status;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !valid_positive(df)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (student_t_uses_normal_limit(df)) {
        return store_probability_result(
            0.5 * erfc(-x / sqrt(2.0)), out);
    }
    if (df / 2.0 == 0.0) {
        *out = 0.5;
        return LMMC_STATUS_OK;
    }

    t_val = student_t_beta_argument(x, df);
    if (t_val > 0.0) {
        status = regularized_beta(t_val, df / 2.0, 0.5, &ib);
        if (status != LMMC_STATUS_OK) { return status; }
    } else {
        const lmmc_real_t a = df / 2.0;
        const lmmc_real_t log_t =
            log(df) - 2.0 * log(fabs(x));
        lmmc_real_t log_beta;
        status = lmmc_log_beta(a, 0.5, &log_beta);
        if (status != LMMC_STATUS_OK) { return status; }
        ib = exp(a * log_t - log(a) - log_beta);
    }
    result = x >= 0.0 ? 1.0 - 0.5 * ib : 0.5 * ib;
    return store_probability_result(result, out);
}

lmmc_status_t lmmc_dist_chi2_pdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    lmmc_real_t k2;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !valid_positive(df)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (df < 2.0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        if (df == 2.0) { *out = 0.5; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    if (df / 2.0 == 0.0) {
        /* Gamma(df/2) ~ 2/df as df tends to zero. */
        return store_density_from_log(
            log(df) - log(x) - log(2.0) - x / 2.0, out);
    }

    k2 = df / 2.0;
    if (k2 >= 16.0) {
        return lmmc_dist_gamma_pdf(x, k2, 2.0, out);
    }
    return store_density_from_log(
        (k2 - 1.0) * log(x) - x / 2.0 -
            k2 * log(2.0) - lgamma(k2),
        out);
}

lmmc_status_t lmmc_dist_chi2_cdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !valid_positive(df)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    /* For a = df/2 below the smallest double, P(a, x/2) rounds to one. */
    if (df / 2.0 == 0.0) { *out = 1.0; return LMMC_STATUS_OK; }

    return lmmc_dist_gamma_cdf(x, df / 2.0, 2.0, out);
}

static int f_beta_density(
    lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2,
    lmmc_real_t first_shape, lmmc_real_t second_shape,
    lmmc_real_t* out, lmmc_status_t* status) {
    const lmmc_real_t scale = df2 / df1;
    if (isfinite(scale) && scale > 0.0) {
        lmmc_real_t beta_x;
        lmmc_real_t beta_complement;
        if (x > scale) {
            const lmmc_real_t ratio = scale / x;
            beta_x = 1.0 / (1.0 + ratio);
            beta_complement = ratio / (1.0 + ratio);
        } else {
            const lmmc_real_t ratio = x / scale;
            beta_x = ratio / (1.0 + ratio);
            beta_complement = 1.0 / (1.0 + ratio);
        }
        if (beta_x > 0.0 && beta_complement > 0.0) {
            double log_kernel;
            *status = lmmc_beta_log_kernel(
                beta_x, beta_complement, first_shape, second_shape, &log_kernel);
            if (*status == LMMC_STATUS_OK) {
                *status = store_density_from_log(log_kernel - log(x), out);
            }
            return 1;
        }
    }
    return 0;
}

lmmc_status_t lmmc_dist_f_pdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2,
                               lmmc_real_t* out) {
    lmmc_real_t first_shape, second_shape, total_shape;
    lmmc_real_t num, den;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !valid_positive(df1) || !valid_positive(df2)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (df1 < 2.0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        if (df1 == 2.0) { *out = 1.0; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    first_shape = df1 / 2.0;
    second_shape = df2 / 2.0;
    total_shape = first_shape + second_shape;
    {
        lmmc_status_t status;
        if (f_beta_density(x, df1, df2, first_shape, second_shape, out, &status)) {
            return status;
        }
    }

    num = first_shape * log(df1 / df2) +
          (first_shape - 1.0) * log(x);
    den = total_shape * log1p(df1 * x / df2);
    {
        double log_beta;
        const lmmc_status_t status =
            lmmc_log_beta(first_shape, second_shape, &log_beta);
        if (status != LMMC_STATUS_OK) { return status; }
        return store_density_from_log(num - den - log_beta, out);
    }
}

lmmc_status_t lmmc_dist_f_cdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2,
                               lmmc_real_t* out) {
    lmmc_real_t scale, result;
    lmmc_status_t status;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !valid_positive(df1) || !valid_positive(df2)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }

    if (df1 / 2.0 == 0.0) {
        /* Beta(df1/2, df2/2) concentrates df2/(df1+df2) at zero. */
        *out = df2 / (df1 + df2);
        return LMMC_STATUS_OK;
    }

    scale = df2 / df1;
    if (x > scale) {
        lmmc_real_t ratio = scale / x;
        lmmc_real_t tail_x = ratio / (1.0 + ratio);
        status = regularized_beta_upper(
            tail_x, df2 / 2.0, df1 / 2.0, &result);
        if (status != LMMC_STATUS_OK) { return status; }
    } else {
        lmmc_real_t ratio = (df1 / df2) * x;
        lmmc_real_t beta_x = ratio / (1.0 + ratio);
        status = regularized_beta(
            beta_x, df1 / 2.0, df2 / 2.0, &result);
        if (status != LMMC_STATUS_OK) { return status; }
    }
    return store_probability_result(result, out);
}

lmmc_status_t lmmc_dist_gamma_pdf(lmmc_real_t x, lmmc_real_t shape,
                                   lmmc_real_t scale, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !valid_positive(shape) || !valid_positive(scale)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (shape < 1.0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        if (shape == 1.0) { return store_density_from_log(-log(scale), out); }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    if (shape >= 16.0) {
        const lmmc_real_t standardized = x / scale;
        if (isfinite(standardized) && standardized > 0.0) {
            const lmmc_real_t log_density =
                -lmmc_deviance_part(shape, standardized) -
                lmmc_log_gamma_stirling_error(shape) -
                log(standardized) + 0.5 * log(shape) -
                LMMC_LOG_SQRT_2PI - log(scale);
            return store_density_from_log(log_density, out);
        }
    }

    return store_density_from_log(
        (shape - 1.0) * log(x) - x / scale -
            shape * log(scale) - lgamma(shape),
        out);
}

lmmc_status_t lmmc_dist_gamma_cdf(lmmc_real_t x, lmmc_real_t shape,
                                   lmmc_real_t scale, lmmc_real_t* out) {
    lmmc_real_t standardized;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !valid_positive(shape) || !valid_positive(scale)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }

    standardized = x / scale;
    if (standardized == 0.0) {
        const lmmc_real_t log_probability =
            shape * (log(x) - log(scale)) - lgamma(shape + 1.0);
        return store_probability_result(exp(log_probability), out);
    }
    return regularized_gamma_lower(shape, standardized, out);
}

static lmmc_status_t beta_interior_density(
    lmmc_real_t x, lmmc_real_t alpha, lmmc_real_t beta_param, lmmc_real_t* out) {
    double logarithm;
    const lmmc_status_t status =
        lmmc_beta_log_kernel(x, 1.0 - x, alpha, beta_param, &logarithm);
    if (status != LMMC_STATUS_OK) { return status; }
    return store_density_from_log(logarithm - log(x) - log1p(-x), out);
}

lmmc_status_t lmmc_dist_beta_pdf(lmmc_real_t x, lmmc_real_t alpha,
                                  lmmc_real_t beta_param, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !valid_positive(alpha) || !valid_positive(beta_param)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x < 0.0 || x > 1.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (alpha < 1.0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        if (alpha == 1.0) { *out = beta_param; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }
    if (x == 1.0) {
        if (beta_param < 1.0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        if (beta_param == 1.0) { *out = alpha; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    return beta_interior_density(x, alpha, beta_param, out);
}

lmmc_status_t lmmc_dist_beta_cdf(lmmc_real_t x, lmmc_real_t alpha,
                                  lmmc_real_t beta_param, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(x) || !valid_positive(alpha) || !valid_positive(beta_param)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x >= 1.0) { *out = 1.0; return LMMC_STATUS_OK; }

    return regularized_beta(x, alpha, beta_param, out);
}
