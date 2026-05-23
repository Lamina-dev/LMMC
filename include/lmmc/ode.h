/**
 * @file ode.h
 * @brief 常微分方程初值问题求解器：Euler、RK4、RK45（自适应）。
 */
#ifndef LMMC_ODE_H
#define LMMC_ODE_H

#include <stddef.h>
#include "lmmc/numeric.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ODE 右端函数签名 @f$y' = f(t, y)@f$ 。
 *
 * @param[in]  t          当前时间。
 * @param[in]  y          当前状态向量，长度 @p dim 。
 * @param[out] y_prime    输出导数向量，长度 @p dim 。
 * @param[in]  dim        状态维度。
 * @param[in]  user_data  用户上下文。
 * @return ::LMMC_STATUS_OK 表示求值成功，其它状态会终止积分。
 */
typedef lmmc_status_t (*lmmc_ode_rhs_t)(
    lmmc_real_t t,
    const lmmc_real_t* y,
    lmmc_real_t* y_prime,
    size_t dim,
    void* user_data
);

/** @brief ODE 求解失败原因。 */
typedef enum {
    LMMC_ODE_FAILURE_NONE = 0,                       /**< 成功。 */
    LMMC_ODE_FAILURE_INVALID_DIMENSION = 1,          /**< 维度参数非法。 */
    LMMC_ODE_FAILURE_INVALID_STEP = 2,               /**< 步长参数非法（如负值）。 */
    LMMC_ODE_FAILURE_MAX_STEPS = 3,                  /**< 达到最大步数仍未抵达终点。 */
    LMMC_ODE_FAILURE_NUMERICAL_ISSUE = 4,            /**< 状态变量出现 NaN/Inf 等。 */
    LMMC_ODE_FAILURE_RHS_EVAL_FAILED = 5,            /**< RHS 回调返回非 ::LMMC_STATUS_OK 。 */
    LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT = 6      /**< abs_tol 与 rel_tol 配置非法。 */
} lmmc_ode_failure_t;

/** @brief ODE 求解结果统计。 */
typedef struct {
    int converged;                       /**< 是否成功抵达终点。 */
    size_t num_steps;                    /**< 实际接受的积分步数。 */
    size_t num_rhs_evals;                /**< RHS 求值次数。 */
    lmmc_real_t final_t;                 /**< 终止时刻。 */
    lmmc_ode_failure_t failure_reason;   /**< 失败原因。 */
} lmmc_ode_result_t;

/**
 * @brief ODE 求解日志回调。
 *
 * @param step      当前步号。
 * @param t         当前时间。
 * @param y         当前状态向量。
 * @param dim       状态维度。
 * @param user_data 用户上下文。
 */
typedef void (*lmmc_ode_log_callback_t)(
    size_t step,
    lmmc_real_t t,
    const lmmc_real_t* y,
    size_t dim,
    void* user_data
);

/** @brief ODE 求解配置。 */
typedef struct {
    lmmc_real_t initial_step;                  /**< 初始步长建议。 */
    lmmc_real_t min_step;                      /**< 最小允许步长。 */
    lmmc_real_t max_step;                      /**< 最大允许步长。 */
    lmmc_real_t abs_tol;                       /**< 局部误差绝对容差。 */
    lmmc_real_t rel_tol;                       /**< 局部误差相对容差。 */
    size_t max_steps;                          /**< 最大允许步数。 */
    lmmc_real_t adaptive_step_beta;            /**< 自适应步长安全系数（典型 0.8~0.9）。 */
    int verbose;                               /**< 非 0 时打印日志。 */
    lmmc_ode_log_callback_t log_cb;            /**< 自定义日志回调。 */
    void* log_user_data;                       /**< 回调上下文。 */
} lmmc_ode_config_t;

/** @brief 获取失败原因对应的可读字符串。 */
const char* lmmc_ode_failure_string(lmmc_ode_failure_t reason);

/**
 * @brief 根据积分区间与维度生成默认配置。
 */
lmmc_status_t lmmc_ode_default_config(
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    size_t problem_dim,
    lmmc_ode_config_t* out_cfg
);

/**
 * @brief 显式 Euler 法求解 @f$y' = f(t,y)@f$ 。
 *
 * @param[in,out] y          初始条件 / 输出末态，长度 @p dim 。
 */
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

/** @brief 经典四阶 Runge-Kutta 求解器（固定步长）。 */
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

/** @brief 自适应 Runge-Kutta-Fehlberg (RK45) 求解器。 */
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
