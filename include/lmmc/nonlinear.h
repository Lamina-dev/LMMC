#ifndef LMMC_NONLINEAR_H
#define LMMC_NONLINEAR_H

#include <stddef.h>
#include "lmmc/numeric.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef lmmc_real_t (*lmmc_scalar_func_t)(lmmc_real_t x, void* user_data);
typedef lmmc_real_t (*lmmc_scalar_dfunc_t)(lmmc_real_t x, void* user_data);

typedef enum {
    LMMC_NONLINEAR_FAILURE_NONE = 0,
    LMMC_NONLINEAR_FAILURE_INVALID_BRACKET = 1,
    LMMC_NONLINEAR_FAILURE_MAX_ITER = 2,
    LMMC_NONLINEAR_FAILURE_ZERO_DERIVATIVE = 3,
    LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE = 4,
    LMMC_NONLINEAR_FAILURE_SINGULAR_STEP = 5
} lmmc_nonlinear_failure_t;

typedef struct {
    int converged;
    size_t num_iter;
    lmmc_real_t root;
    lmmc_real_t function_value;
    lmmc_real_t residual_norm;
    lmmc_nonlinear_failure_t failure_reason;
} lmmc_nonlinear_result_t;

typedef void (*lmmc_nonlinear_log_callback_t)(
    size_t iter,
    lmmc_real_t x,
    lmmc_real_t f_x,
    void* user_data
);

typedef struct {
    lmmc_real_t abs_tol;
    lmmc_real_t rel_tol;
    size_t max_iter;
    lmmc_real_t derivative_step;
    lmmc_real_t min_derivative;
    lmmc_real_t min_step;
    int verbose;
    lmmc_nonlinear_log_callback_t log_cb;
    void* log_user_data;
} lmmc_nonlinear_config_t;

const char* lmmc_nonlinear_failure_string(lmmc_nonlinear_failure_t reason);

lmmc_status_t lmmc_nonlinear_default_config(lmmc_nonlinear_config_t* out_cfg);

lmmc_status_t lmmc_bisection_solve(
    lmmc_scalar_func_t func,
    void* user_data,
    lmmc_real_t left,
    lmmc_real_t right,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
);

lmmc_status_t lmmc_newton_solve(
    lmmc_scalar_func_t func,
    lmmc_scalar_dfunc_t dfunc,
    void* user_data,
    lmmc_real_t x0,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
);

lmmc_status_t lmmc_secant_solve(
    lmmc_scalar_func_t func,
    void* user_data,
    lmmc_real_t x0,
    lmmc_real_t x1,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
);

#ifdef __cplusplus
}
#endif

#endif
