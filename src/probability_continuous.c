#include <math.h>
#include <float.h>

#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"

#include "statistics_internal.h"

static int valid_positive(lmmc_real_t value) {
    return isfinite(value) && value > 0.0;
}

static int valid_probability(lmmc_real_t value) {
    return isfinite(value) && value > 0.0 && value < 1.0;
}

static int student_t_uses_normal_limit(lmmc_real_t df) {
    return df >= 1.0 / DBL_EPSILON;
}

static lmmc_status_t store_density_from_log(
    lmmc_real_t log_density, lmmc_real_t* out) {
    lmmc_real_t result;
    if (isnan(log_density) || log_density > log(DBL_MAX))
        return LMMC_STATUS_NUMERICAL_FAILURE;
    result = exp(log_density);
    if (!isfinite(result)) return LMMC_STATUS_NUMERICAL_FAILURE;
    *out = result;
    return LMMC_STATUS_OK;
}

static lmmc_status_t store_probability_result(
    lmmc_real_t probability, lmmc_real_t* out) {
    if (!isfinite(probability) || probability < 0.0 || probability > 1.0)
        return LMMC_STATUS_NUMERICAL_FAILURE;
    *out = probability;
    return LMMC_STATUS_OK;
}

static lmmc_real_t standardized_normal_value(
    lmmc_real_t x, lmmc_real_t mu, lmmc_real_t sigma) {
    const lmmc_real_t centered = x - mu;
    if (isfinite(centered)) return centered / sigma;
    return x / sigma - mu / sigma;
}

lmmc_status_t lmmc_dist_normal_pdf(lmmc_real_t x, lmmc_real_t mu,
                                    lmmc_real_t sigma, lmmc_real_t* out) {
    lmmc_real_t z, log_density;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !isfinite(mu) || !valid_positive(sigma))
        return LMMC_STATUS_INVALID_ARGUMENT;

    z = standardized_normal_value(x, mu, sigma);
    log_density =
        -0.5 * z * z - log(sigma) - log(LMMC_SQRT_2PI);
    return store_density_from_log(log_density, out);
}

lmmc_status_t lmmc_dist_normal_cdf(lmmc_real_t x, lmmc_real_t mu,
                                    lmmc_real_t sigma, lmmc_real_t* out) {
    lmmc_real_t z;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !isfinite(mu) || !valid_positive(sigma))
        return LMMC_STATUS_INVALID_ARGUMENT;

    z = standardized_normal_value(x, mu, sigma);
    return store_probability_result(
        0.5 * erfc(-z / sqrt(2.0)), out);
}

/**
 * @brief Rational approximation for the normal quantile (Beasley-Springer-Moro).
 */
static lmmc_real_t normal_quantile_rational(lmmc_real_t p) {
    /* Coefficients for rational approximation */
    static const double a[] = {
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00
    };

    double q, r;
    double plow = 0.02425;
    double phigh = 1.0 - plow;

    if (p < plow) {
        /* Lower tail */
        q = sqrt(-2.0 * log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= phigh) {
        /* Central region */
        q = p - 0.5;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        /* Upper tail */
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

lmmc_status_t lmmc_dist_normal_quantile(lmmc_real_t p, lmmc_real_t mu,
                                         lmmc_real_t sigma, lmmc_real_t* out) {
    lmmc_real_t result;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(mu) || !valid_positive(sigma) || !valid_probability(p))
        return LMMC_STATUS_INVALID_ARGUMENT;

    result = fma(sigma, normal_quantile_rational(p), mu);
    if (!isfinite(result)) return LMMC_STATUS_NUMERICAL_FAILURE;
    *out = result;
    return LMMC_STATUS_OK;
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
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !valid_positive(df)) return LMMC_STATUS_INVALID_ARGUMENT;
    if (student_t_uses_normal_limit(df)) {
        return store_density_from_log(
            -0.5 * x * x - LMMC_LOG_SQRT_2PI, out);
    }

    log_density = lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0) -
                  0.5 * (log(df) + log(LMMC_PI)) -
                  ((df + 1.0) / 2.0) * student_t_log_kernel(x, df);
    return store_density_from_log(log_density, out);
}

lmmc_status_t lmmc_dist_t_cdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    lmmc_real_t t_val, ib, result;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !valid_positive(df)) return LMMC_STATUS_INVALID_ARGUMENT;
    if (student_t_uses_normal_limit(df)) {
        return store_probability_result(
            0.5 * erfc(-x / sqrt(2.0)), out);
    }

    t_val = student_t_beta_argument(x, df);
    if (t_val > 0.0) {
        status = regularized_beta(t_val, df / 2.0, 0.5, &ib);
        if (status != LMMC_STATUS_OK) return status;
    } else {
        const lmmc_real_t a = df / 2.0;
        const lmmc_real_t log_t =
            log(df) - 2.0 * log(fabs(x));
        const lmmc_real_t log_beta =
            lgamma(a) + lgamma(0.5) - lgamma(a + 0.5);
        ib = exp(a * log_t - log(a) - log_beta);
    }
    result = x >= 0.0 ? 1.0 - 0.5 * ib : 0.5 * ib;
    return store_probability_result(result, out);
}

typedef lmmc_status_t (*distribution_eval_fn)(
    lmmc_real_t, const void*, lmmc_real_t*);

typedef struct {
    lmmc_real_t first;
    lmmc_real_t second;
} distribution_parameters_t;

static lmmc_status_t eval_t_cdf(
    lmmc_real_t x, const void* context, lmmc_real_t* out) {
    const distribution_parameters_t* parameters =
        (const distribution_parameters_t*)context;
    return lmmc_dist_t_cdf(x, parameters->first, out);
}

static lmmc_status_t eval_t_pdf(
    lmmc_real_t x, const void* context, lmmc_real_t* out) {
    const distribution_parameters_t* parameters =
        (const distribution_parameters_t*)context;
    return lmmc_dist_t_pdf(x, parameters->first, out);
}

static lmmc_status_t eval_chi2_cdf(
    lmmc_real_t x, const void* context, lmmc_real_t* out) {
    const distribution_parameters_t* parameters =
        (const distribution_parameters_t*)context;
    return lmmc_dist_chi2_cdf(x, parameters->first, out);
}

static lmmc_status_t eval_chi2_pdf(
    lmmc_real_t x, const void* context, lmmc_real_t* out) {
    const distribution_parameters_t* parameters =
        (const distribution_parameters_t*)context;
    return lmmc_dist_chi2_pdf(x, parameters->first, out);
}

static lmmc_status_t eval_f_cdf(
    lmmc_real_t x, const void* context, lmmc_real_t* out) {
    const distribution_parameters_t* parameters =
        (const distribution_parameters_t*)context;
    return lmmc_dist_f_cdf(
        x, parameters->first, parameters->second, out);
}

static lmmc_status_t eval_f_pdf(
    lmmc_real_t x, const void* context, lmmc_real_t* out) {
    const distribution_parameters_t* parameters =
        (const distribution_parameters_t*)context;
    return lmmc_dist_f_pdf(
        x, parameters->first, parameters->second, out);
}

static lmmc_status_t eval_gamma_cdf(
    lmmc_real_t x, const void* context, lmmc_real_t* out) {
    const distribution_parameters_t* parameters =
        (const distribution_parameters_t*)context;
    return lmmc_dist_gamma_cdf(
        x, parameters->first, parameters->second, out);
}

static lmmc_status_t eval_gamma_pdf(
    lmmc_real_t x, const void* context, lmmc_real_t* out) {
    const distribution_parameters_t* parameters =
        (const distribution_parameters_t*)context;
    return lmmc_dist_gamma_pdf(
        x, parameters->first, parameters->second, out);
}

static lmmc_status_t eval_beta_cdf(
    lmmc_real_t x, const void* context, lmmc_real_t* out) {
    const distribution_parameters_t* parameters =
        (const distribution_parameters_t*)context;
    return lmmc_dist_beta_cdf(
        x, parameters->first, parameters->second, out);
}

static lmmc_status_t eval_beta_pdf(
    lmmc_real_t x, const void* context, lmmc_real_t* out) {
    const distribution_parameters_t* parameters =
        (const distribution_parameters_t*)context;
    return lmmc_dist_beta_pdf(
        x, parameters->first, parameters->second, out);
}

static lmmc_status_t evaluate_distribution(
    distribution_eval_fn function,
    lmmc_real_t x,
    const void* context,
    lmmc_real_t* value) {
    lmmc_status_t status = function(x, context, value);
    if (status != LMMC_STATUS_OK) return status;
    if (!isfinite(*value)) return LMMC_STATUS_NUMERICAL_FAILURE;
    return LMMC_STATUS_OK;
}

static int quantile_probability_close(lmmc_real_t value, lmmc_real_t target) {
    const lmmc_real_t tail = fmin(target, 1.0 - target);
    return fabs(value - target) <= 1e-12 * tail;
}

static lmmc_status_t finish_quantile_bracket(
    lmmc_real_t probability,
    lmmc_real_t lower,
    lmmc_real_t lower_cdf,
    lmmc_real_t upper,
    lmmc_real_t upper_cdf,
    lmmc_real_t* out) {
    if (quantile_probability_close(lower_cdf, probability)) {
        *out = lower;
        return LMMC_STATUS_OK;
    }
    if (quantile_probability_close(upper_cdf, probability)) {
        *out = upper;
        return LMMC_STATUS_OK;
    }
    return LMMC_STATUS_CONVERGENCE_FAILED;
}

static lmmc_status_t bracketed_quantile(
    lmmc_real_t probability,
    lmmc_real_t initial,
    lmmc_real_t lower,
    lmmc_real_t upper,
    int expand_lower,
    int expand_upper,
    distribution_eval_fn cdf,
    distribution_eval_fn pdf,
    const void* context,
    lmmc_real_t* out) {
    lmmc_real_t lower_cdf, upper_cdf, x;
    lmmc_status_t status;
    int iteration;

    status = evaluate_distribution(cdf, lower, context, &lower_cdf);
    if (status != LMMC_STATUS_OK) return status;
    status = evaluate_distribution(cdf, upper, context, &upper_cdf);
    if (status != LMMC_STATUS_OK) return status;

    for (iteration = 0; lower_cdf > probability && expand_lower;
         ++iteration) {
        if (iteration >= 1024 || lower <= -DBL_MAX) {
            return LMMC_STATUS_OUT_OF_RANGE;
        }
        lower = lower <= -DBL_MAX / 2.0 ? -DBL_MAX : lower * 2.0;
        status = evaluate_distribution(cdf, lower, context, &lower_cdf);
        if (status != LMMC_STATUS_OK) return status;
    }
    for (iteration = 0; upper_cdf < probability && expand_upper;
         ++iteration) {
        if (iteration >= 1024 || upper >= DBL_MAX) {
            return LMMC_STATUS_OUT_OF_RANGE;
        }
        upper = upper >= DBL_MAX / 2.0 ? DBL_MAX : upper * 2.0;
        status = evaluate_distribution(cdf, upper, context, &upper_cdf);
        if (status != LMMC_STATUS_OK) return status;
    }
    if (lower_cdf > probability || upper_cdf < probability) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }

    x = initial;
    if (!isfinite(x) || x <= lower || x >= upper) {
        x = lower * 0.5 + upper * 0.5;
    }
    for (iteration = 0; iteration < 2048; ++iteration) {
        lmmc_real_t cdf_value, pdf_value, candidate;
        status = evaluate_distribution(cdf, x, context, &cdf_value);
        if (status != LMMC_STATUS_OK) return status;
        if (quantile_probability_close(cdf_value, probability)) {
            *out = x;
            return LMMC_STATUS_OK;
        }

        if (cdf_value < probability) {
            lower = x;
            lower_cdf = cdf_value;
        } else {
            upper = x;
            upper_cdf = cdf_value;
        }
        if (nextafter(lower, upper) == upper) {
            return finish_quantile_bracket(
                probability, lower, lower_cdf, upper, upper_cdf, out);
        }

        status = pdf(x, context, &pdf_value);
        candidate = NAN;
        if (status == LMMC_STATUS_OK && isfinite(pdf_value) &&
            pdf_value > DBL_MIN) {
            candidate = x - (cdf_value - probability) / pdf_value;
        }
        if (!isfinite(candidate) || candidate <= lower ||
            candidate >= upper) {
            candidate = lower * 0.5 + upper * 0.5;
        }
        if (candidate == lower || candidate == upper) {
            return finish_quantile_bracket(
                probability, lower, lower_cdf, upper, upper_cdf, out);
        }
        x = candidate;
    }
    return LMMC_STATUS_CONVERGENCE_FAILED;
}


lmmc_status_t lmmc_dist_t_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out) {
    distribution_parameters_t parameters;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!valid_positive(df) || !valid_probability(p))
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (student_t_uses_normal_limit(df)) {
        const lmmc_real_t result = normal_quantile_rational(p);
        if (!isfinite(result)) return LMMC_STATUS_NUMERICAL_FAILURE;
        *out = result;
        return LMMC_STATUS_OK;
    }

    parameters.first = df;
    parameters.second = 0.0;
    return bracketed_quantile(
        p, normal_quantile_rational(p), -1.0, 1.0, 1, 1,
        eval_t_cdf, eval_t_pdf, &parameters, out);
}

lmmc_status_t lmmc_dist_chi2_pdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    lmmc_real_t k2;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !valid_positive(df)) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (df < 2.0) return LMMC_STATUS_NUMERICAL_FAILURE;
        if (df == 2.0) { *out = 0.5; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
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
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !valid_positive(df)) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }

    return lmmc_dist_gamma_cdf(x, df / 2.0, 2.0, out);
}

lmmc_status_t lmmc_dist_chi2_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out) {
    distribution_parameters_t parameters;
    lmmc_real_t z, term, initial;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!valid_positive(df) || !valid_probability(p))
        return LMMC_STATUS_INVALID_ARGUMENT;

    z = normal_quantile_rational(p);
    term = 1.0 - 2.0 / (9.0 * df) + z * sqrt(2.0 / (9.0 * df));
    initial = df * term * term * term;
    if (!isfinite(initial) || initial <= 0.0) initial = 1.0;
    parameters.first = df;
    parameters.second = 0.0;
    return bracketed_quantile(
        p, initial, 0.0, fmax(initial, 1.0), 0, 1,
        eval_chi2_cdf, eval_chi2_pdf, &parameters, out);
}

lmmc_status_t lmmc_dist_f_pdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2,
                               lmmc_real_t* out) {
    lmmc_real_t first_shape, second_shape, total_shape;
    lmmc_real_t num, den;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !valid_positive(df1) || !valid_positive(df2))
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (df1 < 2.0) return LMMC_STATUS_NUMERICAL_FAILURE;
        if (df1 == 2.0) { *out = 1.0; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    first_shape = df1 / 2.0;
    second_shape = df2 / 2.0;
    total_shape = first_shape + second_shape;
    if (first_shape >= 16.0 && second_shape >= 16.0 &&
        isfinite(total_shape)) {
        const lmmc_real_t scale = df2 / df1;
        if (isfinite(scale) && scale > 0.0) {
            lmmc_real_t beta_x;
            lmmc_real_t beta_complement;
            if (x > scale) {
                const lmmc_real_t ratio = scale / x;
                beta_x = 1.0 / (1.0 + ratio);
                beta_complement = ratio / (1.0 + ratio);
            } else {
                const lmmc_real_t ratio = (df1 / df2) * x;
                beta_x = ratio / (1.0 + ratio);
                beta_complement = 1.0 / (1.0 + ratio);
            }
            if (beta_x > 0.0 && beta_complement > 0.0) {
                const lmmc_real_t center = first_shape / total_shape;
                const lmmc_real_t complement_center =
                    second_shape / total_shape;
                const lmmc_real_t correction =
                    lmmc_log_gamma_stirling_error(total_shape) -
                    lmmc_log_gamma_stirling_error(first_shape) -
                    lmmc_log_gamma_stirling_error(second_shape);
                const lmmc_real_t deviance =
                    lmmc_deviance_part(
                        first_shape, total_shape * beta_x) +
                    lmmc_deviance_part(
                        second_shape, total_shape * beta_complement);
                const lmmc_real_t log_density =
                    correction - deviance +
                    0.5 * (log(total_shape) + log(center) +
                           log(complement_center) -
                           2.0 * LMMC_LOG_SQRT_2PI) -
                    log(x);
                return store_density_from_log(log_density, out);
            }
        }
    }

    num = first_shape * log(df1 / df2) +
          (first_shape - 1.0) * log(x);
    den = total_shape * log1p(df1 * x / df2);
    return store_density_from_log(
        num - den - lgamma(first_shape) - lgamma(second_shape) +
            lgamma(total_shape),
        out);
}

lmmc_status_t lmmc_dist_f_cdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2,
                               lmmc_real_t* out) {
    lmmc_real_t scale, result, beta_value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !valid_positive(df1) || !valid_positive(df2))
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }

    scale = df2 / df1;
    if (x > scale) {
        lmmc_real_t ratio = scale / x;
        lmmc_real_t tail_x = ratio / (1.0 + ratio);
        status = regularized_beta(
            tail_x, df2 / 2.0, df1 / 2.0, &beta_value);
        if (status != LMMC_STATUS_OK) return status;
        result = 1.0 - beta_value;
    } else {
        lmmc_real_t ratio = (df1 / df2) * x;
        lmmc_real_t beta_x = ratio / (1.0 + ratio);
        status = regularized_beta(
            beta_x, df1 / 2.0, df2 / 2.0, &result);
        if (status != LMMC_STATUS_OK) return status;
    }
    return store_probability_result(result, out);
}

lmmc_status_t lmmc_dist_f_quantile(lmmc_real_t p, lmmc_real_t df1, lmmc_real_t df2,
                                    lmmc_real_t* out) {
    distribution_parameters_t parameters;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!valid_positive(df1) || !valid_positive(df2) || !valid_probability(p))
        return LMMC_STATUS_INVALID_ARGUMENT;

    parameters.first = df1;
    parameters.second = df2;
    return bracketed_quantile(
        p, 1.0, 0.0, 1.0, 0, 1,
        eval_f_cdf, eval_f_pdf, &parameters, out);
}

lmmc_status_t lmmc_dist_gamma_pdf(lmmc_real_t x, lmmc_real_t shape,
                                   lmmc_real_t scale, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !valid_positive(shape) || !valid_positive(scale))
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (shape < 1.0) return LMMC_STATUS_NUMERICAL_FAILURE;
        if (shape == 1.0)
            return store_density_from_log(-log(scale), out);
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
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !valid_positive(shape) || !valid_positive(scale))
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }

    standardized = x / scale;
    if (standardized == 0.0) {
        const lmmc_real_t log_probability =
            shape * (log(x) - log(scale)) - lgamma(shape + 1.0);
        return store_probability_result(exp(log_probability), out);
    }
    return regularized_gamma_lower(shape, standardized, out);
}

lmmc_status_t lmmc_dist_gamma_quantile(lmmc_real_t p, lmmc_real_t shape,
                                        lmmc_real_t scale, lmmc_real_t* out) {
    distribution_parameters_t parameters;
    lmmc_real_t initial;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!valid_positive(shape) || !valid_positive(scale) || !valid_probability(p))
        return LMMC_STATUS_INVALID_ARGUMENT;

    initial = shape * scale;
    if (!isfinite(initial) || initial <= 0.0) initial = 1.0;
    parameters.first = shape;
    parameters.second = scale;
    return bracketed_quantile(
        p, initial, 0.0, fmax(initial, 1.0), 0, 1,
        eval_gamma_cdf, eval_gamma_pdf, &parameters, out);
}

lmmc_status_t lmmc_dist_beta_pdf(lmmc_real_t x, lmmc_real_t alpha,
                                  lmmc_real_t beta_param, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !valid_positive(alpha) || !valid_positive(beta_param))
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0 || x > 1.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (alpha < 1.0) return LMMC_STATUS_NUMERICAL_FAILURE;
        if (alpha == 1.0) { *out = beta_param; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }
    if (x == 1.0) {
        if (beta_param < 1.0) return LMMC_STATUS_NUMERICAL_FAILURE;
        if (beta_param == 1.0) { *out = alpha; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    if (x == 0.5 && alpha == beta_param &&
        !isfinite(alpha + beta_param)) {
        const lmmc_real_t log_density =
            log(2.0) + 0.5 * (log(alpha) - log(LMMC_PI)) -
            1.0 / (8.0 * alpha);
        return store_density_from_log(log_density, out);
    }

    if (alpha >= 16.0 && beta_param >= 16.0 &&
        isfinite(alpha + beta_param)) {
        const lmmc_real_t total = alpha + beta_param;
        const lmmc_real_t center = alpha / total;
        const lmmc_real_t complement_center = beta_param / total;
        const lmmc_real_t complement_x = 1.0 - x;
        const lmmc_real_t correction =
            lmmc_log_gamma_stirling_error(total) -
            lmmc_log_gamma_stirling_error(alpha) -
            lmmc_log_gamma_stirling_error(beta_param);
        const lmmc_real_t deviance =
            lmmc_deviance_part(alpha, total * x) +
            lmmc_deviance_part(beta_param, total * complement_x);
        const lmmc_real_t log_density =
            correction - deviance +
            0.5 * (log(total) + log(center) +
                   log(complement_center) -
                   2.0 * LMMC_LOG_SQRT_2PI) -
            log(x) - log1p(-x);
        return store_density_from_log(log_density, out);
    }

    return store_density_from_log(
        lgamma(alpha + beta_param) - lgamma(alpha) - lgamma(beta_param) +
            (alpha - 1.0) * log(x) +
            (beta_param - 1.0) * log1p(-x),
        out);
}

lmmc_status_t lmmc_dist_beta_cdf(lmmc_real_t x, lmmc_real_t alpha,
                                  lmmc_real_t beta_param, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(x) || !valid_positive(alpha) || !valid_positive(beta_param))
        return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x >= 1.0) { *out = 1.0; return LMMC_STATUS_OK; }

    return regularized_beta(x, alpha, beta_param, out);
}

lmmc_status_t lmmc_dist_beta_quantile(lmmc_real_t p, lmmc_real_t alpha,
                                       lmmc_real_t beta_param, lmmc_real_t* out) {
    distribution_parameters_t parameters;
    lmmc_real_t ratio, initial;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!valid_positive(alpha) || !valid_positive(beta_param) ||
        !valid_probability(p))
        return LMMC_STATUS_INVALID_ARGUMENT;

    if (alpha >= beta_param) {
        initial = 1.0 / (1.0 + beta_param / alpha);
    } else {
        ratio = alpha / beta_param;
        initial = ratio / (1.0 + ratio);
    }
    parameters.first = alpha;
    parameters.second = beta_param;
    return bracketed_quantile(
        p, initial, 0.0, 1.0, 0, 0,
        eval_beta_cdf, eval_beta_pdf, &parameters, out);
}

