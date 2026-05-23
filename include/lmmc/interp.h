/**
 * @file interp.h
 * @brief 一维插值算法：线性、三次样条、Lagrange 多项式。
 *
 * 节点数组 @c xs 必须严格升序排列。
 */
#ifndef LMMC_INTERP_H
#define LMMC_INTERP_H

#include "lmmc/config.h"
#include "lmmc/status.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 在节点 @c (xs[i], ys[i]) 间做分段线性插值。
 *
 * @param[in]  xs       严格升序的节点 x 数组。
 * @param[in]  ys       对应的 y 值数组。
 * @param[in]  n        节点数，至少 2 。
 * @param[in]  query_x  查询点。
 * @param[out] out_y    输出插值结果。超出 @c [xs[0], xs[n-1]] 范围返回 ::LMMC_STATUS_OUT_OF_RANGE 。
 */
lmmc_status_t lmmc_interp_linear(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_real_t query_x,
    lmmc_real_t* out_y
);

/** @brief 三次样条插值上下文（不透明类型，自然边界条件）。 */
typedef struct lmmc_interp_cspline_t lmmc_interp_cspline_t;

/**
 * @brief 由节点构造自然三次样条。
 *
 * @param[in]  xs          严格升序节点 x 数组。
 * @param[in]  ys          对应 y 值数组。
 * @param[in]  n           节点数，至少 3 。
 * @param[out] out_spline  返回的样条句柄。
 */
lmmc_status_t lmmc_interp_cspline_create(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_interp_cspline_t** out_spline
);

/** @brief 在样条上求值。 */
lmmc_status_t lmmc_interp_cspline_eval(
    const lmmc_interp_cspline_t* spline,
    lmmc_real_t query_x,
    lmmc_real_t* out_y
);

/** @brief 销毁三次样条上下文。 */
void lmmc_interp_cspline_destroy(lmmc_interp_cspline_t* spline);

/** @brief Lagrange 多项式插值上下文（不透明类型）。 */
typedef struct lmmc_interp_lagrange_t lmmc_interp_lagrange_t;

/**
 * @brief 由节点构造 Lagrange 插值多项式（基于重心权重）。
 */
lmmc_status_t lmmc_interp_lagrange_create(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_interp_lagrange_t** out_lagrange
);

/** @brief 在 Lagrange 插值多项式上求值。 */
lmmc_status_t lmmc_interp_lagrange_eval(
    const lmmc_interp_lagrange_t* lagrange,
    lmmc_real_t query_x,
    lmmc_real_t* out_y
);

/** @brief 销毁 Lagrange 插值上下文。 */
void lmmc_interp_lagrange_destroy(lmmc_interp_lagrange_t* lagrange);

#ifdef __cplusplus
}
#endif

#endif
