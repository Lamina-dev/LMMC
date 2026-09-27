/** @file stdlib_stats.h */
#ifndef LMMC_STDLIB_STATS_H
#define LMMC_STDLIB_STATS_H

#include "lmmc/random.h"
#include "lmmc/status.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

lmmc_status_t lmmc_std_stats_mean(const lmmc_real_t* values, size_t count,
                                  lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_median(const lmmc_real_t* values, size_t count,
                                    lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_var(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_std(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_quantile(const lmmc_real_t* values, size_t count,
                                      lmmc_real_t q, lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_cov(const lmmc_real_t* x,
                                 const lmmc_real_t* y,
                                 size_t count,
                                 lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_corr(const lmmc_real_t* x,
                                  const lmmc_real_t* y,
                                  size_t count,
                                  lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_normal_pdf(lmmc_real_t x, lmmc_real_t mean,
                                        lmmc_real_t stddev,
                                        lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_normal_cdf(lmmc_real_t x, lmmc_real_t mean,
                                        lmmc_real_t stddev,
                                        lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_normal_quantile(lmmc_real_t p,
                                             lmmc_real_t mean,
                                             lmmc_real_t stddev,
                                             lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_t_pdf(lmmc_real_t x, lmmc_real_t df,
                                   lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_t_cdf(lmmc_real_t x, lmmc_real_t df,
                                   lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_t_quantile(lmmc_real_t p, lmmc_real_t df,
                                        lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_chi2_pdf(lmmc_real_t x, lmmc_real_t df,
                                      lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_chi2_cdf(lmmc_real_t x, lmmc_real_t df,
                                      lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_chi2_quantile(lmmc_real_t p, lmmc_real_t df,
                                           lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_f_pdf(lmmc_real_t x, lmmc_real_t df1,
                                   lmmc_real_t df2, lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_f_cdf(lmmc_real_t x, lmmc_real_t df1,
                                   lmmc_real_t df2, lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_f_quantile(lmmc_real_t p, lmmc_real_t df1,
                                        lmmc_real_t df2,
                                        lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_gamma_pdf(lmmc_real_t x, lmmc_real_t shape,
                                       lmmc_real_t scale,
                                       lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_gamma_cdf(lmmc_real_t x, lmmc_real_t shape,
                                       lmmc_real_t scale,
                                       lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_gamma_quantile(lmmc_real_t p,
                                            lmmc_real_t shape,
                                            lmmc_real_t scale,
                                            lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_beta_pdf(lmmc_real_t x, lmmc_real_t alpha,
                                      lmmc_real_t beta,
                                      lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_beta_cdf(lmmc_real_t x, lmmc_real_t alpha,
                                      lmmc_real_t beta,
                                      lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_beta_quantile(lmmc_real_t p,
                                           lmmc_real_t alpha,
                                           lmmc_real_t beta,
                                           lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_binomial_pmf(size_t k, size_t n,
                                          lmmc_real_t p,
                                          lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_binomial_cdf(size_t k, size_t n,
                                          lmmc_real_t p,
                                          lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_poisson_pmf(size_t k, lmmc_real_t lambda,
                                         lmmc_real_t* out);
lmmc_status_t lmmc_std_stats_poisson_cdf(size_t k, lmmc_real_t lambda,
                                         lmmc_real_t* out);

lmmc_status_t lmmc_std_random_seed(lmmc_rng_t* rng, uint64_t seed);
lmmc_status_t lmmc_std_random_rand(lmmc_rng_t* rng, lmmc_real_t* out);
lmmc_status_t lmmc_std_random_randint(lmmc_rng_t* rng, int64_t lo,
                                      int64_t hi, int64_t* out);
lmmc_status_t lmmc_std_random_normal(lmmc_rng_t* rng, lmmc_real_t mean,
                                     lmmc_real_t stddev, lmmc_real_t* out);
lmmc_status_t lmmc_std_random_choice(lmmc_rng_t* rng,
                                     const lmmc_real_t* values,
                                     size_t count,
                                     lmmc_real_t* out);
/**
 * @brief 各线程独立持有默认随机流，采用零分配实现。
 * 反初始化仅重置调用线程的流。
 */
void lmmc_std_random_default_deinit(void);
lmmc_status_t lmmc_std_random_default_seed(uint64_t seed);
lmmc_status_t lmmc_std_random_default_rand(lmmc_real_t* out);
lmmc_status_t lmmc_std_random_default_randint(int64_t lo,
                                              int64_t hi,
                                              int64_t* out);
lmmc_status_t lmmc_std_random_default_normal(lmmc_real_t mean,
                                             lmmc_real_t stddev,
                                             lmmc_real_t* out);
lmmc_status_t lmmc_std_random_default_choice(const lmmc_real_t* values,
                                             size_t count,
                                             lmmc_real_t* out);

#ifdef __cplusplus
}
#endif

#endif
