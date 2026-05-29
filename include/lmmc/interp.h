/**
 * @file interp.h
 * @brief 一维插值算法：线性、三次样条、Lagrange 多项式、PCHIP、Akima、二维插值。
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

/* ========================================================================
 * 一维线性插值
 * ======================================================================== */

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

/* ========================================================================
 * 三次样条插值
 * ======================================================================== */

/** @brief 三次样条插值上下文（不透明类型，自然边界条件）。 */
typedef struct lmmc_interp_cspline_t lmmc_interp_cspline_t;

/**
 * @brief 样条边界条件类型。
 */
typedef enum {
    LMMC_SPLINE_NATURAL = 0,   /**< 自然边界：二阶导数为零。 */
    LMMC_SPLINE_CLAMPED,       /**< 固定边界：指定端点一阶导数。 */
    LMMC_SPLINE_NOT_A_KNOT,    /**< Not-a-knot：第三阶导数在第二和倒数第二节点连续。 */
    LMMC_SPLINE_PERIODIC       /**< 周期边界：首尾值和导数匹配。 */
} lmmc_spline_bc_t;

/**
 * @brief 由节点构造自然三次样条。
 *
 * @param[in]  xs          严格升序节点 x 数组。
 * @param[in]  ys          对应 y 值数组。
 * @param[in]  n           节点数，至少 3 。
 * @param[out] out_spline  返回的样条句柄。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 n < 3 或指针为 NULL；
 *         ::LMMC_STATUS_ALLOC_FAILED 若内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存存储样条系数，调用方必须调用 ::lmmc_interp_cspline_destroy 释放。
 * - 内部复制节点数据，调用后可安全释放 xs/ys。
 */
lmmc_status_t lmmc_interp_cspline_create(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_interp_cspline_t** out_spline
);

/**
 * @brief 由节点构造带指定边界条件的三次样条。
 *
 * 支持自然、固定（clamped）、not-a-knot 和周期四种边界条件。
 * 对于 LMMC_SPLINE_CLAMPED，需通过 deriv_left/deriv_right 指定端点导数；
 * 其他边界条件下这两个参数被忽略。
 *
 * @param[in]  xs          严格升序节点 x 数组。
 * @param[in]  ys          对应 y 值数组。
 * @param[in]  n           节点数，至少 3 。
 * @param[in]  bc          边界条件类型。
 * @param[in]  deriv_left  左端点一阶导数（仅 LMMC_SPLINE_CLAMPED 时使用）。
 * @param[in]  deriv_right 右端点一阶导数（仅 LMMC_SPLINE_CLAMPED 时使用）。
 * @param[out] out_spline  返回的样条句柄。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 n < 3、xs 非严格升序、
 *         PERIODIC 模式下首尾 y 值不匹配、或指针为 NULL；
 *         ::LMMC_STATUS_ALLOC_FAILED 若内存分配失败。
 *
 * @note 对于 LMMC_SPLINE_PERIODIC，要求 |ys[0] - ys[n-1]| <= 1e-12，否则返回
 *       LMMC_STATUS_INVALID_ARGUMENT。
 *
 * @par 副作用
 * - 分配堆内存存储样条系数，调用方必须调用 ::lmmc_interp_cspline_destroy 释放。
 * - 内部复制节点数据，调用后可安全释放 xs/ys。
 */
lmmc_status_t lmmc_interp_cspline_create_ex(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_spline_bc_t bc,
    lmmc_real_t deriv_left,
    lmmc_real_t deriv_right,
    lmmc_interp_cspline_t** out_spline
);

/** @brief 在样条上求值。 */
lmmc_status_t lmmc_interp_cspline_eval(
    const lmmc_interp_cspline_t* spline,
    lmmc_real_t query_x,
    lmmc_real_t* out_y
);

/** @brief 销毁三次样条上下文，释放内部分配的所有内存。 */
void lmmc_interp_cspline_destroy(lmmc_interp_cspline_t* spline);

/* ========================================================================
 * PCHIP 单调插值 (Fritsch-Carlson)
 * ======================================================================== */

/** @brief PCHIP 插值上下文（不透明类型）。 */
typedef struct lmmc_interp_pchip_t lmmc_interp_pchip_t;

/**
 * @brief 由节点构造 PCHIP 单调三次插值。
 *
 * PCHIP（Piecewise Cubic Hermite Interpolating Polynomial）保证在单调数据上
 * 插值结果也保持单调性，不会产生虚假振荡。使用 Fritsch-Carlson 方法计算导数。
 *
 * @param[in]  xs   严格升序节点 x 数组。
 * @param[in]  ys   对应 y 值数组。
 * @param[in]  n    节点数，至少 2 。
 * @param[out] out  返回的 PCHIP 句柄。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 n < 2、xs 非严格升序或指针为 NULL；
 *         ::LMMC_STATUS_ALLOC_FAILED 若内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存存储插值系数，调用方必须调用 ::lmmc_interp_pchip_destroy 释放。
 * - 内部复制节点数据，调用后可安全释放 xs/ys。
 */
lmmc_status_t lmmc_interp_pchip_create(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_interp_pchip_t** out
);

/** @brief 在 PCHIP 插值上求值。 */
lmmc_status_t lmmc_interp_pchip_eval(
    const lmmc_interp_pchip_t* p,
    lmmc_real_t x,
    lmmc_real_t* out_y
);

/** @brief 销毁 PCHIP 插值上下文，释放内部分配的所有内存。 */
void lmmc_interp_pchip_destroy(lmmc_interp_pchip_t* p);

/* ========================================================================
 * Akima 局部三次插值
 * ======================================================================== */

/** @brief Akima 插值上下文（不透明类型）。 */
typedef struct lmmc_interp_akima_t lmmc_interp_akima_t;

/**
 * @brief 由节点构造 Akima 局部三次插值。
 *
 * Akima 插值使用局部加权平均计算节点导数，相比全局样条对离群点更鲁棒，
 * 不会产生全局振荡。需要至少 5 个节点以计算边界处的导数。
 *
 * @param[in]  xs   严格升序节点 x 数组。
 * @param[in]  ys   对应 y 值数组。
 * @param[in]  n    节点数，至少 5（Akima 需要至少 5 个点）。
 * @param[out] out  返回的 Akima 句柄。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 n < 5、xs 非严格升序或指针为 NULL；
 *         ::LMMC_STATUS_ALLOC_FAILED 若内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存存储插值系数，调用方必须调用 ::lmmc_interp_akima_destroy 释放。
 * - 内部复制节点数据，调用后可安全释放 xs/ys。
 */
lmmc_status_t lmmc_interp_akima_create(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_interp_akima_t** out
);

/** @brief 在 Akima 插值上求值。 */
lmmc_status_t lmmc_interp_akima_eval(
    const lmmc_interp_akima_t* a,
    lmmc_real_t x,
    lmmc_real_t* out_y
);

/** @brief 销毁 Akima 插值上下文，释放内部分配的所有内存。 */
void lmmc_interp_akima_destroy(lmmc_interp_akima_t* a);

/* ========================================================================
 * Lagrange 多项式插值
 * ======================================================================== */

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

/* ========================================================================
 * 二维插值（矩形网格）
 * ======================================================================== */

/**
 * @brief 双线性插值（矩形网格）。
 *
 * @param[in]  xs    x 方向严格升序节点数组，长度 nx >= 2。
 * @param[in]  nx    x 方向节点数。
 * @param[in]  ys    y 方向严格升序节点数组，长度 ny >= 2。
 * @param[in]  ny    y 方向节点数。
 * @param[in]  zs    网格值数组，行优先存储 (nx * ny)，zs[i*ny + j] = f(xs[i], ys[j])。
 * @param[in]  qx    查询点 x 坐标。
 * @param[in]  qy    查询点 y 坐标。
 * @param[out] out_z 输出插值结果。
 */
lmmc_status_t lmmc_interp_bilinear(
    const lmmc_real_t* xs, size_t nx,
    const lmmc_real_t* ys, size_t ny,
    const lmmc_real_t* zs,
    lmmc_real_t qx, lmmc_real_t qy,
    lmmc_real_t* out_z
);

/**
 * @brief 双三次插值（矩形网格）。
 *
 * @param[in]  xs    x 方向严格升序节点数组，长度 nx >= 4。
 * @param[in]  nx    x 方向节点数。
 * @param[in]  ys    y 方向严格升序节点数组，长度 ny >= 4。
 * @param[in]  ny    y 方向节点数。
 * @param[in]  zs    网格值数组，行优先存储 (nx * ny)，zs[i*ny + j] = f(xs[i], ys[j])。
 * @param[in]  qx    查询点 x 坐标。
 * @param[in]  qy    查询点 y 坐标。
 * @param[out] out_z 输出插值结果。
 */
lmmc_status_t lmmc_interp_bicubic(
    const lmmc_real_t* xs, size_t nx,
    const lmmc_real_t* ys, size_t ny,
    const lmmc_real_t* zs,
    lmmc_real_t qx, lmmc_real_t qy,
    lmmc_real_t* out_z
);

#ifdef __cplusplus
}
#endif

#endif
