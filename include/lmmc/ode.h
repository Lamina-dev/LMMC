/**
 * @file ode.h
 * @brief 常微分方程初值问题求解器：Euler、RK4、RK45（自适应）。
 */
#ifndef LMMC_ODE_H
#define LMMC_ODE_H

#include <stddef.h>
#include "lmmc/numeric.h"
#include "lmmc/diagnostic.h"

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
 * @brief ODE Jacobian 回调签名 @f$J = \partial f / \partial y@f$ 。
 *
 * @param[in]  t          当前时间。
 * @param[in]  y          当前状态向量，长度 @p dim 。
 * @param[out] J          输出 Jacobian 矩阵（行优先，dim×dim）。
 * @param[in]  dim        状态维度。
 * @param[in]  user_data  用户上下文。
 * @return ::LMMC_STATUS_OK 表示求值成功。
 */
typedef lmmc_status_t (*lmmc_ode_jac_t)(
    lmmc_real_t t,
    const lmmc_real_t* y,
    lmmc_real_t* J,
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
    lmmc_diagnostic_sink_t diagnostics;        /**< 统一诊断出口。 */
    lmmc_ode_jac_t jacobian;                   /**< 可选 Jacobian 回调（隐式方法使用，NULL 时用有限差分）。 */
} lmmc_ode_config_t;

/** @brief 获取失败原因对应的可读字符串。 */
const char* lmmc_ode_failure_string(lmmc_ode_failure_t reason);

/**
 * @brief 根据积分区间与维度生成合理的默认求解配置。
 *
 * 根据 @p t_start 与 @p t_end 的跨度自动设置初始步长、最小/最大步长、
 * 容差及最大步数等参数，适合大多数非刚性问题。
 *
 * @param[in]  t_start     积分起始时间。
 * @param[in]  t_end       积分终止时间。
 * @param[in]  problem_dim 状态向量维度（>= 1）。
 * @param[out] out_cfg     输出配置结构体，所有字段将被覆盖写入。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 problem_dim == 0 或 out_cfg 为 NULL。
 *
 * @par 副作用
 * - 将 @p out_cfg 指向的结构体全部字段覆盖写入默认值。
 * - 不分配堆内存。
 */
lmmc_status_t lmmc_ode_default_config(
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    size_t problem_dim,
    lmmc_ode_config_t* out_cfg
);

/**
 * @brief 显式 Euler 法求解常微分方程初值问题 @f$y' = f(t,y)@f$ 。
 *
 * 使用固定步长（由 cfg->initial_step 决定）逐步推进，一阶精度。
 * 适用于快速原型验证，不建议用于高精度需求。
 *
 * @param[in]     rhs        右端函数回调。
 * @param[in]     user_data  传递给 @p rhs 的用户上下文指针。
 * @param[in]     dim        状态向量维度（>= 1）。
 * @param[in]     t_start    积分起始时间。
 * @param[in]     t_end      积分终止时间。
 * @param[in,out] y          输入初始条件，输出终态向量，长度 @p dim 。
 * @param[in]     cfg        求解配置（步长、容差等）。
 * @param[out]    out_result 求解统计信息（步数、RHS 求值次数、收敛状态等）。
 *
 * @return ::LMMC_STATUS_OK 成功抵达终点；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若参数非法（dim==0、NULL 指针等）；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若积分过程中出现 NaN/Inf。
 *
 * @par 副作用
 * - 就地修改 @p y 数组为终态值。
 * - 内部分配临时工作数组（1 个长度为 dim 的缓冲区），函数返回前释放。
 * - 回调 @p rhs 被调用 num_steps 次（每步一次）。
 * - 若配置了诊断 sink，则每步发送一个诊断事件。
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

/**
 * @brief 经典四阶 Runge-Kutta 法求解 @f$y' = f(t,y)@f$ （固定步长）。
 *
 * 使用固定步长（由 cfg->initial_step 决定），四阶精度。
 * 每步需要 4 次 RHS 求值。
 *
 * @param[in]     rhs        右端函数回调。
 * @param[in]     user_data  传递给 @p rhs 的用户上下文指针。
 * @param[in]     dim        状态向量维度（>= 1）。
 * @param[in]     t_start    积分起始时间。
 * @param[in]     t_end      积分终止时间。
 * @param[in,out] y          输入初始条件，输出终态向量，长度 @p dim 。
 * @param[in]     cfg        求解配置（步长、容差等）。
 * @param[out]    out_result 求解统计信息。
 *
 * @return ::LMMC_STATUS_OK 成功抵达终点；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若参数非法；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若积分过程中出现 NaN/Inf。
 *
 * @par 副作用
 * - 就地修改 @p y 数组为终态值。
 * - 内部分配临时工作数组（4 个长度为 dim 的缓冲区用于 k1~k4），函数返回前释放。
 * - 回调 @p rhs 被调用 4 × num_steps 次。
 * - 若配置了诊断 sink，则每步发送一个诊断事件。
 */
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

/**
 * @brief 自适应 Runge-Kutta-Fehlberg (RK45) 求解器。
 *
 * 使用嵌入式 4(5) 阶公式进行局部误差估计，自动调整步长以满足
 * cfg->abs_tol 和 cfg->rel_tol 指定的精度要求。每步需要 6 次 RHS 求值。
 * 适用于大多数非刚性问题。
 *
 * @param[in]     rhs        右端函数回调。
 * @param[in]     user_data  传递给 @p rhs 的用户上下文指针。
 * @param[in]     dim        状态向量维度（>= 1）。
 * @param[in]     t_start    积分起始时间。
 * @param[in]     t_end      积分终止时间。
 * @param[in,out] y          输入初始条件，输出终态向量，长度 @p dim 。
 * @param[in]     cfg        求解配置（步长范围、容差、最大步数等）。
 * @param[out]    out_result 求解统计信息。
 *
 * @return ::LMMC_STATUS_OK 成功抵达终点；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若参数非法；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 若积分过程中出现 NaN/Inf；
 *         ::LMMC_STATUS_MAX_ITERATIONS 若达到 cfg->max_steps 仍未抵达终点。
 *
 * @par 副作用
 * - 就地修改 @p y 数组为终态值。
 * - 内部分配临时工作数组（6 个长度为 dim 的缓冲区用于 k1~k6 及误差估计），函数返回前释放。
 * - 回调 @p rhs 被调用约 6 × num_steps 次（被拒绝的步也会消耗求值次数）。
 * - 若配置了诊断 sink，则每个被接受的步发送一个诊断事件。
 */
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

/** @brief 隐式 Euler 法求解器（A-稳定，适用于刚性问题）。 */
lmmc_status_t lmmc_ode_implicit_euler_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    lmmc_real_t* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
);

/** @brief 梯形法求解器（A-稳定，二阶精度）。 */
lmmc_status_t lmmc_ode_trapezoidal_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    lmmc_real_t* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
);

/** @brief 4 阶 SDIRK 求解器（含嵌入误差估计，自适应步长）。 */
lmmc_status_t lmmc_ode_sdirk4_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    lmmc_real_t t_start,
    lmmc_real_t t_end,
    lmmc_real_t* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
);

/** @brief Rosenbrock-Wanner GRK4T 求解器（线性隐式，四阶，每步一次 LU）。 */
lmmc_status_t lmmc_ode_rosenbrock_grk4t_solve(
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
