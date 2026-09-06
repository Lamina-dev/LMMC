#include "lmmc/stdlib.h"

#include <math.h>
#include <stdlib.h>

#include "lmmc/dense.h"
#include "lmmc/random.h"
#include "lmmc/stats.h"
#include "memory_bridge.h"
#include "internal.h"

#include "stdlib_internal.h"

static lmmc_status_t lmmc_std_wrap_const_vec(const lmmc_real_t* values,
                                             size_t count,
                                             lmmc_vec_t* out)
{
    if (!values || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (count == 0) return LMMC_STATUS_EMPTY_INPUT;
    if (!lmmc_std_real_array_is_finite(values, count)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    out->size = count;
    out->data = (lmmc_real_t*)values;
    out->owns_data = 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_std_stats_mean(const lmmc_real_t* values, size_t count,
                                  lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_mean(&view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_stats_median(const lmmc_real_t* values, size_t count,
                                    lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_median(&view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_stats_var(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_variance_sample(&view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_stats_std(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_stddev_sample(&view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_stats_quantile(const lmmc_real_t* values, size_t count,
                                      lmmc_real_t q, lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(q)) return LMMC_STATUS_NUMERICAL_FAILURE;
    if (q < (lmmc_real_t)0 || q > (lmmc_real_t)1) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    status = lmmc_std_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_quantile(&view, q, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_stats_cov(const lmmc_real_t* x,
                                 const lmmc_real_t* y,
                                 size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t x_view;
    lmmc_vec_t y_view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_wrap_const_vec(x, count, &x_view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_std_wrap_const_vec(y, count, &y_view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_covariance_sample(&x_view, &y_view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_stats_corr(const lmmc_real_t* x,
                                  const lmmc_real_t* y,
                                  size_t count,
                                  lmmc_real_t* out)
{
    lmmc_vec_t x_view;
    lmmc_vec_t y_view;
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_std_wrap_const_vec(x, count, &x_view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_std_wrap_const_vec(y, count, &y_view);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_correlation_sample(&x_view, &y_view, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}
typedef lmmc_status_t (*lmmc_std_dist2_fn)(lmmc_real_t, lmmc_real_t,
                                           lmmc_real_t*);
typedef lmmc_status_t (*lmmc_std_dist3_fn)(lmmc_real_t, lmmc_real_t,
                                           lmmc_real_t, lmmc_real_t*);

static lmmc_status_t lmmc_std_stats_call_dist2(lmmc_std_dist2_fn fn,
                                               lmmc_real_t a,
                                               lmmc_real_t b,
                                               lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!fn || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(a) || !lmmc_std_real_is_finite(b)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = fn(a, b, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

static lmmc_status_t lmmc_std_stats_call_dist3(lmmc_std_dist3_fn fn,
                                               lmmc_real_t a,
                                               lmmc_real_t b,
                                               lmmc_real_t c,
                                               lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!fn || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(a) || !lmmc_std_real_is_finite(b) ||
        !lmmc_std_real_is_finite(c)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = fn(a, b, c, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_stats_normal_pdf(lmmc_real_t x, lmmc_real_t mean,
                                        lmmc_real_t stddev,
                                        lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_normal_pdf, x, mean, stddev,
                                     out);
}

lmmc_status_t lmmc_std_stats_normal_cdf(lmmc_real_t x, lmmc_real_t mean,
                                        lmmc_real_t stddev,
                                        lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_normal_cdf, x, mean, stddev,
                                     out);
}

lmmc_status_t lmmc_std_stats_normal_quantile(lmmc_real_t p,
                                             lmmc_real_t mean,
                                             lmmc_real_t stddev,
                                             lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_normal_quantile, p, mean,
                                     stddev, out);
}

lmmc_status_t lmmc_std_stats_t_pdf(lmmc_real_t x, lmmc_real_t df,
                                   lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist2(lmmc_dist_t_pdf, x, df, out);
}

lmmc_status_t lmmc_std_stats_t_cdf(lmmc_real_t x, lmmc_real_t df,
                                   lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist2(lmmc_dist_t_cdf, x, df, out);
}

lmmc_status_t lmmc_std_stats_t_quantile(lmmc_real_t p, lmmc_real_t df,
                                        lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist2(lmmc_dist_t_quantile, p, df, out);
}

lmmc_status_t lmmc_std_stats_chi2_pdf(lmmc_real_t x, lmmc_real_t df,
                                      lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist2(lmmc_dist_chi2_pdf, x, df, out);
}

lmmc_status_t lmmc_std_stats_chi2_cdf(lmmc_real_t x, lmmc_real_t df,
                                      lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist2(lmmc_dist_chi2_cdf, x, df, out);
}

lmmc_status_t lmmc_std_stats_chi2_quantile(lmmc_real_t p, lmmc_real_t df,
                                           lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist2(lmmc_dist_chi2_quantile, p, df, out);
}

lmmc_status_t lmmc_std_stats_f_pdf(lmmc_real_t x, lmmc_real_t df1,
                                   lmmc_real_t df2, lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_f_pdf, x, df1, df2, out);
}

lmmc_status_t lmmc_std_stats_f_cdf(lmmc_real_t x, lmmc_real_t df1,
                                   lmmc_real_t df2, lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_f_cdf, x, df1, df2, out);
}

lmmc_status_t lmmc_std_stats_f_quantile(lmmc_real_t p, lmmc_real_t df1,
                                        lmmc_real_t df2,
                                        lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_f_quantile, p, df1, df2,
                                     out);
}

lmmc_status_t lmmc_std_stats_gamma_pdf(lmmc_real_t x, lmmc_real_t shape,
                                       lmmc_real_t scale,
                                       lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_gamma_pdf, x, shape, scale,
                                     out);
}

lmmc_status_t lmmc_std_stats_gamma_cdf(lmmc_real_t x, lmmc_real_t shape,
                                       lmmc_real_t scale,
                                       lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_gamma_cdf, x, shape, scale,
                                     out);
}

lmmc_status_t lmmc_std_stats_gamma_quantile(lmmc_real_t p,
                                            lmmc_real_t shape,
                                            lmmc_real_t scale,
                                            lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_gamma_quantile, p, shape,
                                     scale, out);
}

lmmc_status_t lmmc_std_stats_beta_pdf(lmmc_real_t x, lmmc_real_t alpha,
                                      lmmc_real_t beta,
                                      lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_beta_pdf, x, alpha, beta,
                                     out);
}

lmmc_status_t lmmc_std_stats_beta_cdf(lmmc_real_t x, lmmc_real_t alpha,
                                      lmmc_real_t beta,
                                      lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_beta_cdf, x, alpha, beta,
                                     out);
}

lmmc_status_t lmmc_std_stats_beta_quantile(lmmc_real_t p,
                                           lmmc_real_t alpha,
                                           lmmc_real_t beta,
                                           lmmc_real_t* out)
{
    return lmmc_std_stats_call_dist3(lmmc_dist_beta_quantile, p, alpha,
                                     beta, out);
}

lmmc_status_t lmmc_std_stats_binomial_pmf(size_t k, size_t n,
                                          lmmc_real_t p,
                                          lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(p)) return LMMC_STATUS_NUMERICAL_FAILURE;
    status = lmmc_dist_binomial_pmf(k, n, p, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_stats_binomial_cdf(size_t k, size_t n,
                                          lmmc_real_t p,
                                          lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(p)) return LMMC_STATUS_NUMERICAL_FAILURE;
    status = lmmc_dist_binomial_cdf(k, n, p, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_stats_poisson_pmf(size_t k, lmmc_real_t lambda,
                                         lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(lambda)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_dist_poisson_pmf(k, lambda, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_stats_poisson_cdf(size_t k, lmmc_real_t lambda,
                                         lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(lambda)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_dist_poisson_cdf(k, lambda, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}
lmmc_status_t lmmc_std_random_seed(lmmc_rng_t* rng, uint64_t seed)
{
    if (!rng) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_rng_seed(rng, seed);
}

lmmc_status_t lmmc_std_random_rand(lmmc_rng_t* rng, lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!rng || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_rng_uniform(rng, (lmmc_real_t)0, (lmmc_real_t)1, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_random_randint(lmmc_rng_t* rng, int64_t lo,
                                      int64_t hi, int64_t* out)
{
    if (!rng || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_rng_int_uniform(rng, lo, hi, out);
}

lmmc_status_t lmmc_std_random_normal(lmmc_rng_t* rng, lmmc_real_t mean,
                                     lmmc_real_t stddev, lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!rng || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_std_real_is_finite(mean) ||
        !lmmc_std_real_is_finite(stddev)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_rng_normal(rng, mean, stddev, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_random_choice(lmmc_rng_t* rng,
                                     const lmmc_real_t* values,
                                     size_t count,
                                     lmmc_real_t* out)
{
    int64_t index = 0;
    lmmc_status_t status;
    if (!rng || !values || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (count == 0) return LMMC_STATUS_EMPTY_INPUT;
    if (count > (size_t)INT64_MAX) return LMMC_STATUS_OUT_OF_RANGE;
    if (!lmmc_std_real_array_is_finite(values, count)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_rng_int_uniform(rng, 0, (int64_t)count - 1, &index);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_store_finite_real(values[index], out);
}

static lmmc_status_t lmmc_std_default_rng_get(lmmc_rng_t** out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = lmmc_rng_default_get();
    return LMMC_STATUS_OK;
}

void lmmc_std_random_default_deinit(void)
{
    lmmc_rng_default_reset();
}

lmmc_status_t lmmc_std_random_default_seed(uint64_t seed)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status = lmmc_std_default_rng_get(&rng);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_random_seed(rng, seed);
}

lmmc_status_t lmmc_std_random_default_rand(lmmc_real_t* out)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status = lmmc_std_default_rng_get(&rng);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_random_rand(rng, out);
}

lmmc_status_t lmmc_std_random_default_randint(int64_t lo,
                                              int64_t hi,
                                              int64_t* out)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status = lmmc_std_default_rng_get(&rng);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_random_randint(rng, lo, hi, out);
}

lmmc_status_t lmmc_std_random_default_normal(lmmc_real_t mean,
                                             lmmc_real_t stddev,
                                             lmmc_real_t* out)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status = lmmc_std_default_rng_get(&rng);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_random_normal(rng, mean, stddev, out);
}

lmmc_status_t lmmc_std_random_default_choice(const lmmc_real_t* values,
                                             size_t count,
                                             lmmc_real_t* out)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status = lmmc_std_default_rng_get(&rng);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_std_random_choice(rng, values, count, out);
}
