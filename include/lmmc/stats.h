#ifndef LMMC_STATS_H
#define LMMC_STATS_H

#include "lmmc/dense.h"

#ifdef __cplusplus
extern "C" {
#endif

lmmc_status_t lmmc_vec_mean(const lmmc_vec_t* x, double* out_mean);

lmmc_status_t lmmc_vec_variance_population(const lmmc_vec_t* x, double* out_variance);
lmmc_status_t lmmc_vec_variance_sample(const lmmc_vec_t* x, double* out_variance);

lmmc_status_t lmmc_vec_stddev_population(const lmmc_vec_t* x, double* out_stddev);
lmmc_status_t lmmc_vec_stddev_sample(const lmmc_vec_t* x, double* out_stddev);

lmmc_status_t lmmc_vec_covariance_population(const lmmc_vec_t* x, const lmmc_vec_t* y, double* out_covariance);
lmmc_status_t lmmc_vec_covariance_sample(const lmmc_vec_t* x, const lmmc_vec_t* y, double* out_covariance);

lmmc_status_t lmmc_vec_correlation_population(const lmmc_vec_t* x, const lmmc_vec_t* y, double* out_correlation);
lmmc_status_t lmmc_vec_correlation_sample(const lmmc_vec_t* x, const lmmc_vec_t* y, double* out_correlation);

lmmc_status_t lmmc_mat_column_mean(const lmmc_mat_t* x, lmmc_vec_t* out_means);

lmmc_status_t lmmc_mat_covariance_population(const lmmc_mat_t* x, lmmc_mat_t* out_covariance);
lmmc_status_t lmmc_mat_covariance_sample(const lmmc_mat_t* x, lmmc_mat_t* out_covariance);

lmmc_status_t lmmc_mat_correlation_population(const lmmc_mat_t* x, lmmc_mat_t* out_correlation);
lmmc_status_t lmmc_mat_correlation_sample(const lmmc_mat_t* x, lmmc_mat_t* out_correlation);

#ifdef __cplusplus
}
#endif

#endif
