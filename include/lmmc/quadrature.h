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
 * 采样点在 [-1,1] 的归一化坐标中生成，并通过溢出安全的中心和半长度
 * 映射到 [a,b]；不会直接形成可能溢出的 `b-a`。
 *
 * @param[in]  func       被积函数回调。
 * @param[in]  user_data  传递给 @p func 的用户上下文。
 * @param[in]  a          积分下限。
 * @param[in]  b          积分上限。
 * @param[in]  n          子区间数（>= 1）。
 * @param[out] out_result 输出积分近似值。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若端点非有限、顺序无效、
 *         n == 0 或指针为 NULL。
 *
 * @par 副作用
 * - 回调 @p func 被调用 n+1 次。
 * - 不分配堆内存。
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
 *
 * 采样点先在 [-1,1] 的归一化坐标中生成，再通过溢出安全的中心和半长度
 * 映射到 [a,b]；不会直接形成可能溢出的 `b-a`。
 *
 * @param[in]  func       被积函数回调。
 * @param[in]  user_data  传递给 @p func 的用户上下文。
 * @param[in]  a          积分下限。
 * @param[in]  b          积分上限。
 * @param[in]  n          子区间数（必须为偶数且 >= 2）。
 * @param[out] out_result 输出积分近似值。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 n 为奇数、n == 0 或指针为 NULL。
 *
 * @par 副作用
 * - 回调 @p func 被调用 n+1 次。
 * - 不分配堆内存。
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
 * @brief Gauss-Legendre 求积：将 @f$[a,b]@f$ 仿射映射到 @f$[-1,1]@f$ 后使用正交节点。
 *
 * 中心与半长度按端点符号选择等价公式，不直接形成可能溢出的 `a+b`
 * 或 `b-a`。对 2*order-1 次以下的多项式精确积分；节点与权重在编译时
 * 预计算（order <= 20）。
 *
 * @param[in]  func       被积函数回调。
 * @param[in]  user_data  传递给 @p func 的用户上下文。
 * @param[in]  a          积分下限。
 * @param[in]  b          积分上限。
 * @param[in]  order      正交多项式阶数（节点数），范围 [1, 20]。
 * @param[out] out_result 输出积分近似值。
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 order 超出范围或指针为 NULL。
 *
 * @par 副作用
 * - 回调 @p func 被调用恰好 @p order 次。
 * - 不分配堆内存。
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
 * @brief 自适应 Simpson 积分，通过递归二分自动控制精度。
 *
 * 递归地将区间二分，直到局部 Simpson 估计满足容差条件或达到最大递归深度。
 * 中点和各 Simpson 区段通过溢出安全的中心与半长度计算，不直接形成
 * 可能溢出的 `a+b` 或 `b-a`。适用于被积函数局部变化剧烈的情况。
 *
 * @param[in]  func       被积函数回调。
 * @param[in]  user_data  传递给 @p func 的用户上下文。
 * @param[in]  a          积分下限。
 * @param[in]  b          积分上限。
 * @param[in]  abs_tol    绝对容差（>= 0）。
 * @param[in]  rel_tol    相对容差（>= 0）。
 * @param[in]  max_depth  最大递归深度，超出返回 ::LMMC_STATUS_WARNING_MAX_DEPTH 。
 * @param[out] out_result 包含积分值、误差估计与求值次数。
 *
 * @return ::LMMC_STATUS_OK 在容差内收敛；
 *         ::LMMC_STATUS_WARNING_MAX_DEPTH 达到最大深度但结果仍可用；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若指针为 NULL 或容差为负。
 *
 * @par 副作用
 * - 回调 @p func 被调用次数取决于递归深度，最坏情况约 @f$O(2^{max\_depth})@f$ 次。
 * - 使用递归栈，深度受 @p max_depth 限制。
 * - 不分配堆内存（仅使用栈空间）。
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
 * 通过逐次加密梯形法则并进行 Richardson 外推，快速提升收敛阶。
 * 外推使用增量形式，避免先将有限梯形估计乘以 @f$4^j@f$；
 * 采样点在 [-1,1] 生成后通过溢出安全的中心和半长度映射到 [a,b]，
 * 不直接形成可能溢出的 `b-a`。对光滑被积函数效率极高。
 *
 * @param[in]  f         被积函数回调。
 * @param[in]  ud        用户数据指针，传递给 @p f 。
 * @param[in]  a         积分下限。
 * @param[in]  b         积分上限。
 * @param[in]  abs_tol   绝对容差，范围 [1e-15, 1e-1]。
 * @param[in]  max_iter  最大迭代次数（Romberg 表行数），范围 [1, 30]。
 * @param[out] out       积分结果（值、误差估计、求值次数）。
 *
 * @return ::LMMC_STATUS_OK 在容差内收敛；
 *         ::LMMC_STATUS_CONVERGENCE_FAILED 达到最大迭代次数；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 回调或中间结果为非有限值；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若参数超出有效范围或指针为 NULL。
 *
 * @par 副作用
 * - 回调 @p f 被调用次数记录在 out->num_evals 中，最多为 @f$2^{max\_iter-1}+1@f$ 次。
 * - 不分配堆内存。
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
 * 利用 @f$\tanh(\frac{\pi}{2}\sinh t)@f$ 变换将端点奇异性映射为指数衰减。
 * 因浮点舍入落到精确端点的节点不调用 @p f，其极限权重贡献按零处理；
 * 非端点回调的 NaN 或无穷值仍返回数值失败。
 * 对端点处有代数或对数奇异性的被积函数特别有效。
 *
 * @param[in]  f         被积函数回调。
 * @param[in]  ud        用户数据指针，传递给 @p f 。
 * @param[in]  a         积分下限。
 * @param[in]  b         积分上限。
 * @param[in]  abs_tol   绝对容差，范围 [1e-15, 1e-1]。
 * @param[in]  max_nodes 最大节点数，范围 [1, 1000000]。
 * @param[out] out       积分结果（值、误差估计、求值次数）。
 *
 * @return ::LMMC_STATUS_OK 在容差内收敛；
 *         ::LMMC_STATUS_CONVERGENCE_FAILED 达到最大节点数；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 回调或中间结果为非有限值；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若参数超出有效范围或指针为 NULL。
 *
 * @par 副作用
 * - 仅对可表示的内部节点调用 @p f。
 * - 实际回调次数记录在 out->num_evals 中，不超过 @p max_nodes 次。
 * - 不分配堆内存。
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
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 回调或累加结果为非有限值；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 order 超出范围或指针为 NULL。
 *
 * @par 副作用
 * - 回调 @p f 被调用恰好 @p order 次。
 * - 不分配堆内存。
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
 *
 * @return ::LMMC_STATUS_OK 成功；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 回调或累加结果为非有限值；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 order 超出范围或指针为 NULL。
 *
 * @par 副作用
 * - 回调 @p f 被调用恰好 @p order 次。
 * - 不分配堆内存。
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
