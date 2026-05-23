/**
 * @file nonlinear.h
 * @brief 标量非线性方程求根：二分法、Newton、割线法。
 */
#ifndef LMMC_NONLINEAR_H
#define LMMC_NONLINEAR_H

#include <stddef.h>
#include "lmmc/numeric.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 标量函数签名：@f$f(x)@f$ 。
 *
 * @param x         自变量。
 * @param user_data 用户上下文指针。
 * @return @f$f(x)@f$ 的值。
 */
typedef lmmc_real_t (*lmmc_scalar_func_t)(lmmc_real_t x, void* user_data);

/**
 * @brief 标量函数导数签名：@f$f'(x)@f$ ，与 ::lmmc_scalar_func_t 同形参。
 */
typedef lmmc_real_t (*lmmc_scalar_dfunc_t)(lmmc_real_t x, void* user_data);

/**
 * @brief 求根失败原因。
 */
typedef enum {
    LMMC_NONLINEAR_FAILURE_NONE = 0,              /**< 成功，无失败。 */
    LMMC_NONLINEAR_FAILURE_INVALID_BRACKET = 1,   /**< 二分法初始区间不变号。 */
    LMMC_NONLINEAR_FAILURE_MAX_ITER = 2,          /**< 达到最大迭代步数仍未收敛。 */
    LMMC_NONLINEAR_FAILURE_ZERO_DERIVATIVE = 3,   /**< Newton 法导数过小。 */
    LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE = 4,   /**< 出现 NaN / Inf 等数值问题。 */
    LMMC_NONLINEAR_FAILURE_SINGULAR_STEP = 5      /**< 步长退化为零或反向震荡。 */
} lmmc_nonlinear_failure_t;

/**
 * @brief 求根结果统计。
 */
typedef struct {
    int converged;                              /**< 是否收敛。 */
    size_t num_iter;                            /**< 实际迭代次数。 */
    lmmc_real_t root;                           /**< 估计的根。 */
    lmmc_real_t function_value;                 /**< 在 @c root 处的 f(x) 。 */
    lmmc_real_t residual_norm;                  /**< 残差范数 |f(root)| 。 */
    lmmc_nonlinear_failure_t failure_reason;    /**< 失败原因，未失败为 ::LMMC_NONLINEAR_FAILURE_NONE 。 */
} lmmc_nonlinear_result_t;

/**
 * @brief 求根日志回调签名。
 */
typedef void (*lmmc_nonlinear_log_callback_t)(
    size_t iter,
    lmmc_real_t x,
    lmmc_real_t f_x,
    void* user_data
);

/**
 * @brief 求根算法配置。
 */
typedef struct {
    lmmc_real_t abs_tol;                          /**< 残差绝对容差。 */
    lmmc_real_t rel_tol;                          /**< 步长相对容差。 */
    size_t max_iter;                              /**< 最大迭代次数。 */
    lmmc_real_t derivative_step;                  /**< 数值导数差分步长（仅割线 / 自动差分使用）。 */
    lmmc_real_t min_derivative;                   /**< Newton 法允许的最小导数绝对值。 */
    lmmc_real_t min_step;                         /**< 步长下限（防止震荡）。 */
    int verbose;                                  /**< 非 0 时打印日志。 */
    lmmc_nonlinear_log_callback_t log_cb;         /**< 自定义日志回调，可为 NULL 。 */
    void* log_user_data;                          /**< 传入回调的上下文。 */
} lmmc_nonlinear_config_t;

/** @brief 获取失败原因对应的可读字符串。 */
const char* lmmc_nonlinear_failure_string(lmmc_nonlinear_failure_t reason);

/** @brief 生成默认配置。 */
lmmc_status_t lmmc_nonlinear_default_config(lmmc_nonlinear_config_t* out_cfg);

/**
 * @brief 二分法求根（要求 @c f(left)*f(right) < 0 ）。
 */
lmmc_status_t lmmc_bisection_solve(
    lmmc_scalar_func_t func,
    void* user_data,
    lmmc_real_t left,
    lmmc_real_t right,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
);

/**
 * @brief Newton 法求根，需要解析导数 @p dfunc 。
 */
lmmc_status_t lmmc_newton_solve(
    lmmc_scalar_func_t func,
    lmmc_scalar_dfunc_t dfunc,
    void* user_data,
    lmmc_real_t x0,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_result_t* out_result
);

/**
 * @brief 割线法求根，仅需函数值，初始两点 @p x0 ≠ @p x1 。
 */
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
