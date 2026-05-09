#ifndef LMMC_ODE_H
#define LMMC_ODE_H

#include <stddef.h>
#include "lmmc/numeric.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef lmmc_status_t (*lmmc_ode_rhs_t)(
    lmmc_real_t t,
    const lmmc_real_t* y,
    lmmc_real_t* y_prime,
    size_t dim,
    void* user_data
);

typedef enum {
    LMMC_ODE_FAILURE_NONE = 0,
    LMMC_ODE_FAILURE_INVALID_DIMENSION = 1,
    LMMC_ODE_FAILURE_INVALID_STEP = 2,
    LMMC_ODE_FAILURE_MAX_STEPS = 3,
    LMMC_ODE_FAILURE_NUMERICAL_ISSUE = 4,
    LMMC_ODE_FAILURE_RHS_EVAL_FAILED = 5,
    LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT = 6
} lmmc_ode_failure_t;

typedef struct {
    int converged;
    size_t num_steps;
    size_t num_rhs_evals;
    lmmc_real_t final_t;
    lmmc_ode_failure_t failure_reason;
} lmmc_ode_result_t;

typedef void (*lmmc_ode_log_callback_t)(
    size_t step,
    lmmc_real_t t,
    const lmmc_real_t* y,
    size_t dim,
    void* user_data
);

typedef struct {
    lmmc_real_t initial_step;
    lmmc_real_t min_step;
    lmmc_real_t max_step;
    lmmc_real_t abs_tol;
    lmmc_real_t rel_tol;
    size_t max_steps;
    lmmc_real_t adaptive_step_beta;
    int verbose;
    lmmc_ode_log_callback_t log_cb;
    void* log_user_data;
} lmmc_ode_config_t;

const char* lmmc_ode_failure_string(lmmc_ode_failure_t reason);

lmmc_status_t lmmc_ode_default_config(
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    size_t problem_dim,
    lmmc_ode_config_t* out_cfg
);

lmmc_status_t lmmc_ode_euler_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    lmmc_real_t* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
);

lmmc_status_t lmmc_ode_rk4_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    lmmc_real_t* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
);

lmmc_status_t lmmc_ode_rk45_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    lmmc_real_t* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
);

#ifdef __cplusplus
}
#endif

#endif
