/**
 * @file ode_internal.h
 * @brief ODE 求解器共享的内部辅助函数。
 *
 * 提供日志、结果统计、配置加载与校验、RHS 求值和 Jacobian 计算。
 * @internal
 */
#ifndef LMMC_ODE_INTERNAL_H
#define LMMC_ODE_INTERNAL_H

#include <math.h>
#include <string.h>

#include "internal.h"
#include "lmmc/ode.h"
#include "lmmc/diagnostic.h"

typedef struct {
    lmmc_ode_rhs_t rhs;
    lmmc_ode_jac_t jacobian;
    void* user_data;
    lmmc_real_t t;
    const lmmc_real_t* y;
    const lmmc_real_t* base_rhs;
    size_t dim;
    lmmc_real_t* jacobian_data;
    lmmc_real_t* base_rhs_work;
    lmmc_real_t* y_perturbed;
    lmmc_real_t* rhs_perturbed;
    size_t* io_rhs_evals;
    int* out_callback_failed;
} lmmc_ode_jacobian_request_t;

void lmmc_ode_do_log(const lmmc_ode_config_t* cfg, size_t step, lmmc_real_t t, const lmmc_real_t* y, size_t dim);

void lmmc_ode_reset_result(lmmc_ode_result_t* out_result, lmmc_real_t t_start);

lmmc_status_t lmmc_ode_load_and_validate_config(
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_config_t* out_cfg,
    lmmc_ode_result_t* out_result
);

lmmc_status_t lmmc_ode_rhs_eval(
    lmmc_ode_rhs_t rhs,
    lmmc_real_t t,
    const lmmc_real_t* y,
    lmmc_real_t* y_prime,
    size_t dim,
    void* user_data,
    size_t* io_eval_count,
    int* out_callback_failed
);

int lmmc_ode_state_is_finite(const lmmc_real_t* y, size_t dim);

lmmc_status_t validate_and_init_ode_config(
    lmmc_ode_rhs_t rhs,
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    const lmmc_real_t* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_config_t* out_cfg,
    lmmc_ode_result_t* out_result,
    size_t* out_work_bytes
);

lmmc_status_t lmmc_ode_weighted_rms(
    const lmmc_real_t* error,
    const lmmc_real_t* y_old,
    const lmmc_real_t* y_new,
    size_t dim,
    lmmc_real_t abs_tol,
    lmmc_real_t rel_tol,
    lmmc_real_t* out_norm
);

lmmc_real_t lmmc_ode_next_step(
    lmmc_real_t h,
    lmmc_real_t error_norm,
    const lmmc_ode_config_t* cfg
);

lmmc_status_t lmmc_ode_values_are_finite(
    const lmmc_real_t* values,
    size_t count
);

lmmc_status_t lmmc_ode_jacobian_eval(const lmmc_ode_jacobian_request_t* request);

#endif
