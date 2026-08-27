#include <math.h>

#include "lmmc/config.h"
#include "lmmc/numeric.h"
#include "lmmc/stats.h"

#include "statistics_internal.h"

lmmc_status_t lmmc_dist_normal_pdf(lmmc_real_t x, lmmc_real_t mu,
                                    lmmc_real_t sigma, lmmc_real_t* out) {
    lmmc_real_t z, exponent;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (sigma <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    z = (x - mu) / sigma;
    exponent = -0.5 * z * z;
    *out = exp(exponent) / (sigma * LMMC_SQRT_2PI);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_normal_cdf(lmmc_real_t x, lmmc_real_t mu,
                                    lmmc_real_t sigma, lmmc_real_t* out) {
    lmmc_real_t z;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (sigma <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    z = (x - mu) / sigma;
    *out = 0.5 * (1.0 + erf(z / sqrt(2.0)));
    return LMMC_STATUS_OK;
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
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (sigma <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    *out = mu + sigma * normal_quantile_rational(p);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_t_pdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    lmmc_real_t coeff, base;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    coeff = exp(lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0)) /
            sqrt(df * LMMC_PI);
    base = 1.0 + x * x / df;
    *out = coeff * pow(base, -(df + 1.0) / 2.0);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_t_cdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    lmmc_real_t t_val, ib;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;

    t_val = df / (df + x * x);
    ib = regularized_beta(t_val, df / 2.0, 0.5);

    if (x >= 0.0) {
        *out = 1.0 - 0.5 * ib;
    } else {
        *out = 0.5 * ib;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_t_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out) {
    /* Newton's method using t CDF and PDF */
    lmmc_real_t x, cdf_val, pdf_val;
    int iter;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Initial guess from normal quantile */
    x = normal_quantile_rational(p);

    for (iter = 0; iter < 100; ++iter) {
        lmmc_real_t dx;
        lmmc_dist_t_cdf(x, df, &cdf_val);
        lmmc_dist_t_pdf(x, df, &pdf_val);
        if (pdf_val < 1e-300) break;
        dx = (cdf_val - p) / pdf_val;
        x -= dx;
        if (fabs(dx) < 1e-12 * (1.0 + fabs(x))) break;
    }

    *out = x;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_chi2_pdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    lmmc_real_t k2;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (df < 2.0) { *out = 1.0 / 0.0; return LMMC_STATUS_OK; }
        if (df == 2.0) { *out = 0.5; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    k2 = df / 2.0;
    *out = exp((k2 - 1.0) * log(x) - x / 2.0 - k2 * log(2.0) - lgamma(k2));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_chi2_cdf(lmmc_real_t x, lmmc_real_t df, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }

    *out = regularized_gamma_lower(df / 2.0, x / 2.0);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_chi2_quantile(lmmc_real_t p, lmmc_real_t df, lmmc_real_t* out) {
    /* Newton's method on chi2 CDF */
    lmmc_real_t x, cdf_val, pdf_val, dx;
    int iter;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Initial guess: Wilson-Hilferty approximation */
    {
        lmmc_real_t z = normal_quantile_rational(p);
        lmmc_real_t tmp = 1.0 - 2.0 / (9.0 * df) + z * sqrt(2.0 / (9.0 * df));
        x = df * tmp * tmp * tmp;
        if (x <= 0.0) x = 0.01;
    }

    for (iter = 0; iter < 100; ++iter) {
        lmmc_dist_chi2_cdf(x, df, &cdf_val);
        lmmc_dist_chi2_pdf(x, df, &pdf_val);
        if (pdf_val < 1e-300) break;
        dx = (cdf_val - p) / pdf_val;
        x -= dx;
        if (x <= 0.0) x = 1e-10;
        if (fabs(dx) < 1e-12 * (1.0 + fabs(x))) break;
    }

    *out = x;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_f_pdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2,
                               lmmc_real_t* out) {
    lmmc_real_t num, den;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df1 <= 0.0 || df2 <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (df1 < 2.0) { *out = 1.0 / 0.0; return LMMC_STATUS_OK; }
        if (df1 == 2.0) { *out = 1.0; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    num = (df1 / 2.0) * log(df1 / df2) + (df1 / 2.0 - 1.0) * log(x);
    den = ((df1 + df2) / 2.0) * log(1.0 + df1 * x / df2);
    *out = exp(num - den - lgamma(df1 / 2.0) - lgamma(df2 / 2.0) +
               lgamma((df1 + df2) / 2.0));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_f_cdf(lmmc_real_t x, lmmc_real_t df1, lmmc_real_t df2,
                               lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df1 <= 0.0 || df2 <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }

    *out = regularized_beta(df1 * x / (df1 * x + df2), df1 / 2.0, df2 / 2.0);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_f_quantile(lmmc_real_t p, lmmc_real_t df1, lmmc_real_t df2,
                                    lmmc_real_t* out) {
    /* Newton's method on F CDF */
    lmmc_real_t x, cdf_val, pdf_val, dx;
    int iter;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (df1 <= 0.0 || df2 <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Initial guess */
    x = df2 / (df2 - 2.0 > 0.1 ? df2 - 2.0 : 0.1);
    if (x <= 0.0) x = 1.0;

    for (iter = 0; iter < 100; ++iter) {
        lmmc_dist_f_cdf(x, df1, df2, &cdf_val);
        lmmc_dist_f_pdf(x, df1, df2, &pdf_val);
        if (pdf_val < 1e-300) break;
        dx = (cdf_val - p) / pdf_val;
        x -= dx;
        if (x <= 0.0) x = 1e-10;
        if (fabs(dx) < 1e-12 * (1.0 + fabs(x))) break;
    }

    *out = x;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_gamma_pdf(lmmc_real_t x, lmmc_real_t shape,
                                   lmmc_real_t scale, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (shape <= 0.0 || scale <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (shape < 1.0) { *out = 1.0 / 0.0; return LMMC_STATUS_OK; }
        if (shape == 1.0) { *out = 1.0 / scale; return LMMC_STATUS_OK; }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    *out = exp((shape - 1.0) * log(x) - x / scale -
               shape * log(scale) - lgamma(shape));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_gamma_cdf(lmmc_real_t x, lmmc_real_t shape,
                                   lmmc_real_t scale, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (shape <= 0.0 || scale <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }

    *out = regularized_gamma_lower(shape, x / scale);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_gamma_quantile(lmmc_real_t p, lmmc_real_t shape,
                                        lmmc_real_t scale, lmmc_real_t* out) {
    /* Newton's method on gamma CDF */
    lmmc_real_t x, cdf_val, pdf_val, dx;
    int iter;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (shape <= 0.0 || scale <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Initial guess: use chi2 approximation */
    x = shape * scale;
    if (x <= 0.0) x = 1.0;

    for (iter = 0; iter < 100; ++iter) {
        lmmc_dist_gamma_cdf(x, shape, scale, &cdf_val);
        lmmc_dist_gamma_pdf(x, shape, scale, &pdf_val);
        if (pdf_val < 1e-300) break;
        dx = (cdf_val - p) / pdf_val;
        x -= dx;
        if (x <= 0.0) x = 1e-10;
        if (fabs(dx) < 1e-12 * (1.0 + fabs(x))) break;
    }

    *out = x;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_beta_pdf(lmmc_real_t x, lmmc_real_t alpha,
                                  lmmc_real_t beta_param, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (alpha <= 0.0 || beta_param <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < 0.0 || x > 1.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x == 0.0) {
        if (alpha < 1.0) { *out = 1.0 / 0.0; return LMMC_STATUS_OK; }
        if (alpha == 1.0) {
            *out = exp(lgamma(alpha + beta_param) - lgamma(alpha) - lgamma(beta_param));
            return LMMC_STATUS_OK;
        }
        *out = 0.0; return LMMC_STATUS_OK;
    }
    if (x == 1.0) {
        if (beta_param < 1.0) { *out = 1.0 / 0.0; return LMMC_STATUS_OK; }
        if (beta_param == 1.0) {
            *out = exp(lgamma(alpha + beta_param) - lgamma(alpha) - lgamma(beta_param));
            return LMMC_STATUS_OK;
        }
        *out = 0.0; return LMMC_STATUS_OK;
    }

    *out = exp(lgamma(alpha + beta_param) - lgamma(alpha) - lgamma(beta_param) +
               (alpha - 1.0) * log(x) + (beta_param - 1.0) * log(1.0 - x));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_beta_cdf(lmmc_real_t x, lmmc_real_t alpha,
                                  lmmc_real_t beta_param, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (alpha <= 0.0 || beta_param <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) { *out = 0.0; return LMMC_STATUS_OK; }
    if (x >= 1.0) { *out = 1.0; return LMMC_STATUS_OK; }

    *out = regularized_beta(x, alpha, beta_param);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_dist_beta_quantile(lmmc_real_t p, lmmc_real_t alpha,
                                       lmmc_real_t beta_param, lmmc_real_t* out) {
    /* Newton's method on beta CDF */
    lmmc_real_t x, cdf_val, pdf_val, dx;
    int iter;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (alpha <= 0.0 || beta_param <= 0.0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (p <= 0.0 || p >= 1.0) return LMMC_STATUS_INVALID_ARGUMENT;

    /* Initial guess: mean of beta distribution */
    x = alpha / (alpha + beta_param);

    for (iter = 0; iter < 100; ++iter) {
        lmmc_dist_beta_cdf(x, alpha, beta_param, &cdf_val);
        lmmc_dist_beta_pdf(x, alpha, beta_param, &pdf_val);
        if (pdf_val < 1e-300) break;
        dx = (cdf_val - p) / pdf_val;
        x -= dx;
        if (x <= 0.0) x = 1e-10;
        if (x >= 1.0) x = 1.0 - 1e-10;
        if (fabs(dx) < 1e-12 * (1.0 + fabs(x))) break;
    }

    *out = x;
    return LMMC_STATUS_OK;
}

