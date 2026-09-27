#include "probability_internal.h"

/** @brief 用有理近似计算正态分位数。 */
static lmmc_real_t normal_quantile_rational(lmmc_real_t p) {
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
        /** @brief 下尾近似。 */
        q = sqrt(-2.0 * log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= phigh) {
        /** @brief 中心区间近似。 */
        q = p - 0.5;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        /** @brief 上尾近似。 */
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
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
    if (status != LMMC_STATUS_OK) { return status; }
    if (!isfinite(*value)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
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

static lmmc_status_t expand_quantile_lower(
    lmmc_real_t probability, lmmc_real_t* bound, lmmc_real_t* value,
    distribution_eval_fn cdf, const void* context) {
    int iteration;
    lmmc_status_t status;
    for (iteration = 0; *value > probability;
         ++iteration) {
        if (iteration >= 1024 || *bound <= -DBL_MAX) {
            return LMMC_STATUS_OUT_OF_RANGE;
        }
        *bound = *bound <= -DBL_MAX / 2.0 ? -DBL_MAX : *bound * 2.0;
        status = evaluate_distribution(cdf, *bound, context, value);
        if (status != LMMC_STATUS_OK) { return status; }
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t expand_quantile_upper(
    lmmc_real_t probability, lmmc_real_t* bound, lmmc_real_t* value,
    distribution_eval_fn cdf, const void* context) {
    int iteration;
    lmmc_status_t status;
    for (iteration = 0; *value < probability;
         ++iteration) {
        if (iteration >= 1024 || *bound >= DBL_MAX) {
            return LMMC_STATUS_OUT_OF_RANGE;
        }
        *bound = *bound >= DBL_MAX / 2.0 ? DBL_MAX : *bound * 2.0;
        status = evaluate_distribution(cdf, *bound, context, value);
        if (status != LMMC_STATUS_OK) { return status; }
    }
    return LMMC_STATUS_OK;
}

static lmmc_real_t quantile_newton_candidate(
    lmmc_real_t probability, lmmc_real_t x, lmmc_real_t cdf_value,
    lmmc_real_t lower, lmmc_real_t upper, distribution_eval_fn pdf,
    const void* context) {
    lmmc_real_t pdf_value;
    lmmc_status_t status = pdf(x, context, &pdf_value);
    lmmc_real_t candidate = NAN;
    if (status == LMMC_STATUS_OK && isfinite(pdf_value) && pdf_value > DBL_MIN) {
        candidate = x - (cdf_value - probability) / pdf_value;
    }
    if (!isfinite(candidate) || candidate <= lower || candidate >= upper) {
        candidate = lower * 0.5 + upper * 0.5;
    }
    return candidate;
}

static lmmc_status_t refine_quantile_bracket(
    lmmc_real_t probability, lmmc_real_t initial, lmmc_real_t lower, lmmc_real_t upper,
    lmmc_real_t lower_cdf, lmmc_real_t upper_cdf, distribution_eval_fn cdf,
    distribution_eval_fn pdf, const void* context, lmmc_real_t* out) {
    int iteration;
    lmmc_real_t x;
    lmmc_status_t status;
    x = initial;
    if (!isfinite(x) || x <= lower || x >= upper) {
        x = lower * 0.5 + upper * 0.5;
    }
    for (iteration = 0; iteration < 2048; ++iteration) {
        lmmc_real_t cdf_value, candidate;
        status = evaluate_distribution(cdf, x, context, &cdf_value);
        if (status != LMMC_STATUS_OK) { return status; }
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

        candidate = quantile_newton_candidate(
            probability, x, cdf_value, lower, upper, pdf, context);
        if (candidate == lower || candidate == upper) {
            return finish_quantile_bracket(
                probability, lower, lower_cdf, upper, upper_cdf, out);
        }
        x = candidate;
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
    lmmc_real_t lower_cdf, upper_cdf;
    lmmc_status_t status;

    status = evaluate_distribution(cdf, lower, context, &lower_cdf);
    if (status != LMMC_STATUS_OK) { return status; }
    status = evaluate_distribution(cdf, upper, context, &upper_cdf);
    if (status != LMMC_STATUS_OK) { return status; }

    if (expand_lower) {
        status = expand_quantile_lower(probability, &lower, &lower_cdf, cdf, context);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
    }
    if (expand_upper) {
        status = expand_quantile_upper(probability, &upper, &upper_cdf, cdf, context);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
    }
    if (lower_cdf > probability || upper_cdf < probability) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    return refine_quantile_bracket(probability, initial, lower, upper,
        lower_cdf, upper_cdf, cdf, pdf, context, out);
}


lmmc_status_t lmmc_dist_normal_quantile(lmmc_real_t p, lmmc_real_t mu,
                                         lmmc_real_t sigma, lmmc_real_t* out) {
    lmmc_real_t result;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(mu) || !valid_positive(sigma) || !valid_probability(p)) { return LMMC_STATUS_INVALID_ARGUMENT; }

    result = fma(sigma, normal_quantile_rational(p), mu);
    if (!isfinite(result)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_t_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out) {
    distribution_parameters_t parameters;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!valid_positive(df) || !valid_probability(p)) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (student_t_uses_normal_limit(df)) {
        const lmmc_real_t result = normal_quantile_rational(p);
        if (!isfinite(result)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        *out = result;
        return LMMC_STATUS_OK;
    }

    parameters.first = df;
    parameters.second = 0.0;
    return bracketed_quantile(
        p, normal_quantile_rational(p), -1.0, 1.0, 1, 1,
        eval_t_cdf, eval_t_pdf, &parameters, out);
}

lmmc_status_t lmmc_dist_chi2_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out) {
    distribution_parameters_t parameters;
    lmmc_real_t z, term, initial;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!valid_positive(df) || !valid_probability(p)) { return LMMC_STATUS_INVALID_ARGUMENT; }

    z = normal_quantile_rational(p);
    term = 1.0 - 2.0 / (9.0 * df) + z * sqrt(2.0 / (9.0 * df));
    initial = df * term * term * term;
    if (!isfinite(initial) || initial <= 0.0) {
        initial = 1.0;
    }
    parameters.first = df;
    parameters.second = 0.0;
    return bracketed_quantile(
        p, initial, 0.0, fmax(initial, 1.0), 0, 1,
        eval_chi2_cdf, eval_chi2_pdf, &parameters, out);
}

lmmc_status_t lmmc_dist_f_quantile(lmmc_real_t p, lmmc_real_t df1, lmmc_real_t df2,
                                    lmmc_real_t* out) {
    distribution_parameters_t parameters;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!valid_positive(df1) || !valid_positive(df2) || !valid_probability(p)) { return LMMC_STATUS_INVALID_ARGUMENT; }

    parameters.first = df1;
    parameters.second = df2;
    return bracketed_quantile(
        p, 1.0, 0.0, 1.0, 0, 1,
        eval_f_cdf, eval_f_pdf, &parameters, out);
}

lmmc_status_t lmmc_dist_gamma_quantile(lmmc_real_t p, lmmc_real_t shape,
                                        lmmc_real_t scale, lmmc_real_t* out) {
    distribution_parameters_t parameters;
    lmmc_real_t initial;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!valid_positive(shape) || !valid_positive(scale) || !valid_probability(p)) { return LMMC_STATUS_INVALID_ARGUMENT; }

    initial = shape * scale;
    if (!isfinite(initial) || initial <= 0.0) {
        initial = 1.0;
    }
    parameters.first = shape;
    parameters.second = scale;
    return bracketed_quantile(
        p, initial, 0.0, fmax(initial, 1.0), 0, 1,
        eval_gamma_cdf, eval_gamma_pdf, &parameters, out);
}

lmmc_status_t lmmc_dist_beta_quantile(lmmc_real_t p, lmmc_real_t alpha,
                                       lmmc_real_t beta_param, lmmc_real_t* out) {
    distribution_parameters_t parameters;
    lmmc_real_t ratio, initial;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!valid_positive(alpha) || !valid_positive(beta_param) ||
        !valid_probability(p)) { return LMMC_STATUS_INVALID_ARGUMENT; }

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
