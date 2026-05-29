/**
 * @file optimize.h
 * @brief 优化与非线性方程组求解模块。
 *
 * 提供 Newton、Broyden 非线性方程组求解器，以及 L-BFGS、
 * Levenberg-Marquardt、梯度下降等无约束优化算法。
 */
#ifndef LMMC_OPTIMIZE_H
#define LMMC_OPTIMIZE_H

#include <stddef.h>
#include "lmmc/status.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================== 回调类型 ===================== */

/**
 * @brief 非线性函数回调：计算 F(x)。
 *
 * @param[in]  x         当前迭代点。
 * @param[out] F         函数值输出向量（与 x 同维）。
 * @param[in]  user_data 用户自定义数据指针。
 * @return ::LMMC_STATUS_OK 成功。
 */
typedef lmmc_status_t (*lmmc_opt_func_t)(const lmmc_vec_t* x, lmmc_vec_t* F, void* user_data);

/**
 * @brief Jacobian 回调：计算 J(x)。
 *
 * @param[in]  x         当前迭代点。
 * @param[out] J         Jacobian 矩阵输出（n×n）。
 * @param[in]  user_data 用户自定义数据指针。
 * @return ::LMMC_STATUS_OK 成功。
 */
typedef lmmc_status_t (*lmmc_opt_jac_t)(const lmmc_vec_t* x, lmmc_mat_t* J, void* user_data);

/**
 * @brief 梯度回调：计算 ∇f(x)。
 *
 * @param[in]  x         当前迭代点。
 * @param[out] grad      梯度输出向量（与 x 同维）。
 * @param[in]  user_data 用户自定义数据指针。
 * @return ::LMMC_STATUS_OK 成功。
 */
typedef lmmc_status_t (*lmmc_opt_grad_t)(const lmmc_vec_t* x, lmmc_vec_t* grad, void* user_data);

/**
 * @brief 目标函数回调：计算标量 f(x)。
 *
 * @param[in]  x         当前迭代点。
 * @param[in]  user_data 用户自定义数据指针。
 * @return 目标函数值。
 */
typedef lmmc_real_t (*lmmc_opt_obj_t)(const lmmc_vec_t* x, void* user_data);

/* ===================== 配置与结果类型 ===================== */

/**
 * @brief 优化算法配置参数。
 */
typedef struct {
    lmmc_real_t abs_tol;       /**< 绝对收敛容差。 */
    lmmc_real_t rel_tol;       /**< 相对收敛容差。 */
    size_t max_iter;           /**< 最大迭代次数。 */
    size_t lbfgs_memory;       /**< L-BFGS 存储的 (s,y) 对数，默认 10。 */
    lmmc_real_t lm_damping;    /**< Levenberg-Marquardt 初始阻尼参数 λ。 */
    int verbose;               /**< 非零时输出迭代信息。 */
} lmmc_optimize_config_t;

/**
 * @brief 优化失败原因枚举。
 */
typedef enum {
    LMMC_OPT_FAILURE_NONE = 0,                /**< 无失败（成功收敛）。 */
    LMMC_OPT_FAILURE_MAX_ITER,                /**< 达到最大迭代次数。 */
    LMMC_OPT_FAILURE_NUMERICAL_ISSUE,         /**< 数值问题（NaN/Inf）。 */
    LMMC_OPT_FAILURE_LINE_SEARCH_FAILED,      /**< 线搜索失败。 */
    LMMC_OPT_FAILURE_SINGULAR_JACOBIAN        /**< Jacobian 奇异。 */
} lmmc_optimize_failure_t;

/**
 * @brief 优化结果。
 */
typedef struct {
    int converged;                     /**< 非零表示已收敛。 */
    size_t num_iter;                   /**< 实际迭代次数。 */
    lmmc_real_t final_residual;        /**< 最终残差范数。 */
    lmmc_optimize_failure_t failure_reason; /**< 失败原因（converged=0 时有效）。 */
} lmmc_optimize_result_t;

/* ===================== 公共函数 ===================== */

/**
 * @brief 初始化默认优化配置。
 *
 * 默认值：abs_tol=1e-12, rel_tol=1e-10, max_iter=1000,
 * lbfgs_memory=10, lm_damping=1e-3, verbose=0。
 */
lmmc_status_t lmmc_optimize_default_config(lmmc_optimize_config_t* cfg);

/**
 * @brief Newton 法求解非线性方程组 F(x) = 0。
 *
 * 每步求解 J(x)δ = -F(x)，通过 LU 分解。若 J 为 NULL，
 * 使用前向有限差分近似 Jacobian。
 *
 * @param[in]     F         非线性函数回调。
 * @param[in]     J         Jacobian 回调（可为 NULL，使用有限差分）。
 * @param[in]     user_data 传递给回调的用户数据。
 * @param[in,out] x         初始猜测 / 输出解。
 * @param[in]     cfg       算法配置。
 * @param[out]    out       迭代结果。
 *
 * @return ::LMMC_STATUS_OK（通过 out->converged 判断是否收敛）。
 *
 * @par 副作用
 * - 就地修改 x->data。
 * - 分配并释放临时向量（Fx, delta）和矩阵（Jmat, Jlu）+ pivots 数组。
 * - 多次调用 F 和 J 回调。
 */
lmmc_status_t lmmc_nleq_newton(
    lmmc_opt_func_t F,
    lmmc_opt_jac_t J,
    void* user_data,
    lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out);

/**
 * @brief Broyden 法求解非线性方程组 F(x) = 0。
 *
 * 维护 B ≈ J 的秩一更新：B += (ΔF - B·Δx)·Δxᵀ / (Δxᵀ·Δx)。
 * 初始 B 通过有限差分近似。
 *
 * @param[in]     F         非线性函数回调。
 * @param[in]     user_data 传递给回调的用户数据。
 * @param[in,out] x         初始猜测 / 输出解。
 * @param[in]     cfg       算法配置。
 * @param[out]    out       迭代结果。
 *
 * @return ::LMMC_STATUS_OK（通过 out->converged 判断是否收敛）。
 *
 * @par 副作用
 * - 就地修改 x->data。
 * - 分配并释放多个临时向量和 n×n 矩阵。
 * - 多次调用 F 回调。
 */
lmmc_status_t lmmc_nleq_broyden(
    lmmc_opt_func_t F,
    void* user_data,
    lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out);

/**
 * @brief L-BFGS 无约束最小化。
 *
 * 使用两循环递归与 m 个存储的 (s,y) 对。含 Armijo 回溯线搜索。
 *
 * @param[in]     obj       目标函数回调。
 * @param[in]     grad      梯度回调。
 * @param[in]     user_data 传递给回调的用户数据。
 * @param[in,out] x         初始猜测 / 输出解。
 * @param[in]     cfg       算法配置（lbfgs_memory 指定 m）。
 * @param[out]    out       迭代结果。
 *
 * @return ::LMMC_STATUS_OK（通过 out->converged 判断是否收敛）。
 *
 * @par 副作用
 * - 就地修改 x->data。
 * - 分配 m*n*2 的 s/y 存储 + 多个临时向量，返回前释放。
 * - 多次调用 obj 和 grad 回调。
 */
lmmc_status_t lmmc_minimize_lbfgs(
    lmmc_opt_obj_t obj,
    lmmc_opt_grad_t grad,
    void* user_data,
    lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out);

/**
 * @brief Levenberg-Marquardt 非线性最小二乘。
 *
 * 求解 (JᵀJ + λI)δ = -Jᵀr，自适应调整 λ（接受步减小 λ，拒绝步增大 λ）。
 *
 * @param[in]     residual  残差函数回调 r(x)。
 * @param[in]     J         Jacobian 回调（可为 NULL，使用有限差分）。
 * @param[in]     user_data 传递给回调的用户数据。
 * @param[in,out] x         初始猜测 / 输出解。
 * @param[in]     cfg       算法配置（lm_damping 指定初始 λ）。
 * @param[out]    out       迭代结果。
 *
 * @return ::LMMC_STATUS_OK（通过 out->converged 判断是否收敛）。
 *
 * @par 副作用
 * - 就地修改 x->data。
 * - 分配并释放多个临时向量和 n×n 矩阵。
 * - 多次调用 residual 和 J 回调。
 */
lmmc_status_t lmmc_minimize_levenberg_marquardt(
    lmmc_opt_func_t residual,
    lmmc_opt_jac_t J,
    void* user_data,
    lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out);

/**
 * @brief 梯度下降法（带 Armijo 回溯线搜索）。
 *
 * @param[in]     obj       目标函数回调。
 * @param[in]     grad      梯度回调。
 * @param[in]     user_data 传递给回调的用户数据。
 * @param[in,out] x         初始猜测 / 输出解。
 * @param[in]     cfg       算法配置。
 * @param[out]    out       迭代结果。
 *
 * @return ::LMMC_STATUS_OK（通过 out->converged 判断是否收敛）。
 *
 * @par 副作用
 * - 就地修改 x->data。
 * - 分配并释放临时向量。多次调用 obj 和 grad 回调。
 */
lmmc_status_t lmmc_minimize_gradient_descent(
    lmmc_opt_obj_t obj,
    lmmc_opt_grad_t grad,
    void* user_data,
    lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out);

#ifdef __cplusplus
}
#endif

#endif
