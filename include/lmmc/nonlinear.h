#ifndef LMMC_NONLINEAR_H
#define LMMC_NONLINEAR_H

#include <stddef.h>
#include "lmmc/numeric.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*lmmc_scalar_func_t)(double x, void* user_data);
typedef double (*lmmc_scalar_dfunc_t)(double x, void* user_data);

typedef enum {
    LMMC_NONLINEAR_FAILURE_NONE = 0,
    LMMC_NONLINEAR_FAILURE_INVALID_BRACKET = 1,
    LMMC_NONLINEAR_FAILURE_MAX_ITER = 2,
    LMMC_NONLINEAR_FAILURE_ZERO_DERIVATIVE = 3,
    LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE = 4,
    LMMC_NONLINEAR_FAILURE_SINGULAR_STEP = 5
} lmmc_nonlinear_failure_t;

typedef struct {
    double abs_tol;
    double rel_tol;
    size_t max_iter;
    double derivative_step;
    double min_derivative;
    double min_step;
    int verbose;
} lmmc_nonlinear_config_t;

typedef struct {
    int converged;
    size_t num_iter;
    double root;
    double function_value;
    double residual_norm;
    lmmc_nonlinear_failure_t failure_reason;
} lmmc_nonlinear_result_t;

const char* lmmc_nonlinear_failure_string(lmmc_nonlinear_failure_t reason);

lmmc_status_t lmmc_nonlinear_default_config(lmmc_nonlinear_config_t* out_cfg);

lmmc_status_t lmmc_bisection_solve(
    lmmc_scalar_func_t func,
    void* user_data,
    double left,
    double right,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
);

lmmc_status_t lmmc_newton_solve(
    lmmc_scalar_func_t func,
    lmmc_scalar_dfunc_t dfunc,
    void* user_data,
    double x0,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
);

lmmc_status_t lmmc_secant_solve(
    lmmc_scalar_func_t func,
    void* user_data,
    double x0,
    double x1,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
);

#ifdef __cplusplus
}
#endif

#endif
