/**
 * @file nonlinear.h
 * @brief 标量非线性方程求根:二分法,Newton,割线法.
 */
#ifndef LMMC_NONLINEAR_H
#define LMMC_NONLINEAR_H

#include <stddef.h>
#include "lmmc/numeric.h"
#include "lmmc/diagnostic.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 标量函数签名:@f$f(x)@f$ .
 *
 * @param x         自变量.
 * @param user_data 用户上下文指针.
 * @return @f$f(x)@f$ 的值.
 */
typedef lmmc_real_t (*lmmc_scalar_func_t)(lmmc_real_t x, void* user_data);

/**
 * @brief 标量函数导数签名:@f$f'(x)@f$ ,与 ::lmmc_scalar_func_t 同形参.
 */
typedef lmmc_real_t (*lmmc_scalar_dfunc_t)(lmmc_real_t x, void* user_data);

/**
 * @brief 求根失败原因.
 */
typedef enum {
    LMMC_NONLINEAR_FAILURE_NONE = 0,              /**< 成功,无失败. */
    LMMC_NONLINEAR_FAILURE_INVALID_BRACKET = 1,   /**< 二分法初始区间不变号. */
    LMMC_NONLINEAR_FAILURE_MAX_ITER = 2,          /**< 达到最大迭代步数仍未收敛. */
    LMMC_NONLINEAR_FAILURE_ZERO_DERIVATIVE = 3,   /**< Newton 法导数过小. */
    LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE = 4,   /**< 出现 NaN / Inf 等数值问题. */
    LMMC_NONLINEAR_FAILURE_SINGULAR_STEP = 5      /**< 步长退化为零或反向震荡. */
} lmmc_nonlinear_failure_t;

/**
 * @brief 求根结果统计.
 */
typedef struct {
    int converged;                              /**< 是否收敛. */
    size_t num_iter;                            /**< 实际迭代次数. */
    lmmc_real_t root;                           /**< 估计的根. */
    lmmc_real_t function_value;                 /**< 在 @c root 处的 f(x) . */
    lmmc_real_t residual_norm;                  /**< 残差范数 |f(root)| . */
    lmmc_nonlinear_failure_t failure_reason;    /**< 失败原因,未失败为 ::LMMC_NONLINEAR_FAILURE_NONE . */
} lmmc_nonlinear_result_t;

/**
 * @brief 求根算法配置.
 */
typedef struct {
    lmmc_real_t abs_tol;                          /**< 残差绝对容差. */
    lmmc_real_t rel_tol;                          /**< 步长相对容差. */
    size_t max_iter;                              /**< 最大迭代次数. */
    lmmc_real_t derivative_step;                  /**< Newton 自动差分的相对扰动步长. */
    lmmc_real_t min_derivative;                   /**< Newton 法允许的最小导数绝对值. */
    lmmc_real_t min_step;                         /**< 维持迭代稳定性的步长下限. */
    lmmc_diagnostic_sink_t diagnostics;           /**< 统一诊断出口. */
} lmmc_nonlinear_config_t;

/** @brief 获取失败原因对应的可读字符串. */
const char* lmmc_nonlinear_failure_string(lmmc_nonlinear_failure_t reason);

/** @brief 生成默认配置. */
lmmc_status_t lmmc_nonlinear_default_config(lmmc_nonlinear_config_t* out_cfg);

/**
 * @brief 二分法求根.
 *
 * 要求 f(left)*f(right) < 0（或端点本身为根）。中点与半区间宽度按
 * 端点符号选择等价公式，不形成可能溢出的 `left+right` 或
 * `right-left`。收敛判据为 |f(mid)| <= abs_tol 或半区间宽度 <= x_tol。
 * @param[in]  func       标量函数 f(x).
 * @param[in]  user_data  传递给 func 的用户上下文.
 * @param[in]  left       区间左端点,要求 left < right.
 * @param[in]  right      区间右端点.
 * @param[in]  cfg        配置参数,可为 NULL(使用默认值).
 * @param[out] out_result 求根结果统计.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功(含收敛情况).
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 参数非法或区间不变号.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 出现 NaN/Inf 或达到最大步数.
 *
 * @par 副作用
 * - 无内存分配.多次调用 func.
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
 * @brief Newton 法求根.
 *
 * 若 dfunc 为 NULL，使用中心差分自动计算导数。解析导数与数值导数均按
 * @c min_derivative 判定是否可用，该阈值不随 x 的绝对量级隐式变化。
 * 收敛判据为 |f(x)| <= abs_tol 或 |x_{k+1}-x_k| <= x_tol。
 *
 * @param[in]  func       标量函数 f(x).
 * @param[in]  dfunc      解析导数 f'(x),可为 NULL(自动差分).
 * @param[in]  user_data  传递给 func/dfunc 的用户上下文.
 * @param[in]  x0         初始猜测.
 * @param[in]  cfg        配置参数,可为 NULL.
 * @param[out] out_result 求根结果统计.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 参数非法.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 数值问题或未收敛.
 *
 * @par 副作用
 * - 无内存分配.多次调用 func(和 dfunc).
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
 * @brief 割线法求根.
 *
 * 仅需函数值，不需要导数。使用两个初始点 x0 != x1 逼近；函数值先按
 * 相邻值的最大绝对值缩放，异号函数值使用凸组合计算截距，因此函数值差、
 * 初始点差均不必在原始量级上可表示。非零常数缩放 f 不改变判定。
 * 收敛判据同 Newton 法。
 * @param[in]  func       标量函数 f(x).
 * @param[in]  user_data  传递给 func 的用户上下文.
 * @param[in]  x0         第一个初始点.
 * @param[in]  x1         第二个初始点,须 x0 != x1.
 * @param[in]  cfg        配置参数,可为 NULL.
 * @param[out] out_result 求根结果统计.
 *
 * @return 同 ::lmmc_newton_solve.
 *
 * @par 副作用
 * - 无内存分配.多次调用 func.
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
