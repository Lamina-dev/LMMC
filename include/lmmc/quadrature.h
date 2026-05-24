/**
 * @file quadrature.h
 * @brief 一元数值积分：梯形、Simpson、Gauss-Legendre、自适应 Simpson。
 */
#ifndef LMMC_QUADRATURE_H
#define LMMC_QUADRATURE_H

#include "lmmc/config.h"
#include "lmmc/status.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 被积函数签名 @f$f(x)@f$ 。
 */
typedef lmmc_real_t (*lmmc_quad_func_t)(lmmc_real_t x, void* user_data);

/**
 * @brief 自适应积分输出结果。
 */
typedef struct {
    lmmc_real_t value;     /**< 积分估计值。 */
    lmmc_real_t error;     /**< 误差估计。 */
    size_t num_evals;      /**< 函数求值次数。 */
} lmmc_quad_result_t;

/**
 * @brief 复合梯形公式：将 @f$[a,b]@f$ 等分为 @p n 段。
 *
 * @param[in]  n          子区间数（>= 1）。
 */
lmmc_status_t lmmc_quad_trapezoid(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t n,
    lmmc_real_t* out_result
);

/**
 * @brief 复合 Simpson 公式：将 @f$[a,b]@f$ 等分为 @p n 段（要求偶数）。
 */
lmmc_status_t lmmc_quad_simpson(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t n,
    lmmc_real_t* out_result
);

/**
 * @brief Gauss-Legendre 求积：将 @f$[a,b]@f$ 仿射映射到 @f$[-1,1]@f$ 。
 *
 * @param[in]  order  正交多项式阶数（节点数）。
 */
lmmc_status_t lmmc_quad_gauss_legendre(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t order,
    lmmc_real_t* out_result
);

/**
 * @brief 自适应 Simpson 积分。
 *
 * @param[in]  abs_tol    绝对容差。
 * @param[in]  rel_tol    相对容差。
 * @param[in]  max_depth  最大递归深度，超出返回 ::LMMC_STATUS_WARNING_MAX_DEPTH 。
 * @param[out] out_result 包含积分值、误差估计与求值次数。
 */
lmmc_status_t lmmc_quad_adaptive(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t abs_tol,
    lmmc_real_t rel_tol,
    size_t max_depth,
    lmmc_quad_result_t* out_result
);

/**
 * @brief Romberg 积分（Richardson 外推加速梯形法则）。
 *
 * @param[in]  f         被积函数。
 * @param[in]  ud        用户数据指针，传递给 @p f 。
 * @param[in]  a         积分下限。
 * @param[in]  b         积分上限。
 * @param[in]  abs_tol   绝对容差，范围 [1e-15, 1e-1]。
 * @param[in]  max_iter  最大迭代次数（Romberg 表行数），范围 [1, 1000000]。
 * @param[out] out       积分结果（值、误差估计、求值次数）。
 */
lmmc_status_t lmmc_quad_romberg(
    lmmc_quad_func_t f,
    void* ud,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t abs_tol,
    size_t max_iter,
    lmmc_quad_result_t* out
);

/**
 * @brief Tanh-Sinh（双指数）积分，适用于端点奇异性。
 *
 * @param[in]  f         被积函数。
 * @param[in]  ud        用户数据指针，传递给 @p f 。
 * @param[in]  a         积分下限。
 * @param[in]  b         积分上限。
 * @param[in]  abs_tol   绝对容差，范围 [1e-15, 1e-1]。
 * @param[in]  max_nodes 最大节点数，范围 [1, 1000000]。
 * @param[out] out       积分结果（值、误差估计、求值次数）。
 */
lmmc_status_t lmmc_quad_tanh_sinh(
    lmmc_quad_func_t f,
    void* ud,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t abs_tol,
    size_t max_nodes,
    lmmc_quad_result_t* out
);

/**
 * @brief Gauss-Hermite 求积：权函数 @f$\exp(-x^2)@f$ ，积分域 @f$(-\infty, +\infty)@f$ 。
 *
 * 计算 @f$\int_{-\infty}^{+\infty} f(x) \exp(-x^2) dx@f$ 。
 *
 * @param[in]  f      被积函数（不含权函数部分）。
 * @param[in]  ud     用户数据指针。
 * @param[in]  order  节点数（阶数），范围 [1, 20]。
 * @param[out] out    积分结果值。
 */
lmmc_status_t lmmc_quad_gauss_hermite(
    lmmc_quad_func_t f,
    void* ud,
    size_t order,
    lmmc_real_t* out
);

/**
 * @brief Gauss-Laguerre 求积：权函数 @f$\exp(-x)@f$ ，积分域 @f$[0, +\infty)@f$ 。
 *
 * 计算 @f$\int_0^{+\infty} f(x) \exp(-x) dx@f$ 。
 *
 * @param[in]  f      被积函数（不含权函数部分）。
 * @param[in]  ud     用户数据指针。
 * @param[in]  order  节点数（阶数），范围 [1, 20]。
 * @param[out] out    积分结果值。
 */
lmmc_status_t lmmc_quad_gauss_laguerre(
    lmmc_quad_func_t f,
    void* ud,
    size_t order,
    lmmc_real_t* out
);

#ifdef __cplusplus
}
#endif

#endif
