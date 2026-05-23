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

#ifdef __cplusplus
}
#endif

#endif
