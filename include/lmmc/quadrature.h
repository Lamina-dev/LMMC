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
 * @brief 自适应 Simpson 积分，使用最大误差堆控制全局误差。
 *
 * 每次二分误差最大的可分叶区间，并复用已经求得的回调样本。
 * 所有叶区间（包括达到深度限制的叶）的校正积分和非负误差估计
 * 分别以补偿求和累加。仅在全局条件
 * @f$sum\_error \le \max(abs\_tol, rel\_tol |sum\_value|)@f$
 * 满足时返回成功。误差为 Simpson 外推估计，对不光滑或欠采样函数
 * 不保证严格误差界。
 * 中点和区段半长度使用溢出安全公式。无法继续表示内部采样点的叶
 * 不再细分，但仍保留在全局积分和误差中。
 *
 * @param[in]  func       被积函数回调。
 * @param[in]  user_data  传递给 @p func 的用户上下文。
 * @param[in]  a          积分下限。
 * @param[in]  b          积分上限。
 * @param[in]  abs_tol    绝对容差（>= 0）。
 * @param[in]  rel_tol    相对容差（>= 0）。
 * @param[in]  max_depth  最大二分深度（根为 0）；工作区按需增长。
 * @param[out] out_result 包含积分值、误差估计与求值次数。
 *
 * @return ::LMMC_STATUS_OK 全局误差估计满足容差；
 *         ::LMMC_STATUS_WARNING_MAX_DEPTH 容差未满足且所有叶因深度或
 *         浮点表示限制不能继续细分，输出仍为有效的有限估计；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 指针为 NULL、端点非有限或
 *         不满足 a < b、容差非有限或为负；
 *         ::LMMC_STATUS_ALLOCATION_FAILED 堆分配失败或容量/字节计数溢出；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 回调或中间量非有限、求值计数
 *         溢出，或初始区间不能表示五个不同的 Simpson 采样点。
 *
 * @note
 * - 使用可增长的堆工作区（LMMC 可恢复分配器），无递归栈。
 * - 已有样本不重复调用；通常初始 5 次，每次二分新增 4 次求值。
 * - 只有 OK 或 WARNING_MAX_DEPTH 一次性写入完整 out_result；
 *   其余错误完全保留调用前的输出。所有退出路径释放工作区。
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
 * @param[in]  max_nodes 实际回调次数的总预算，范围 [1, 1000000]。
 * @param[out] out       积分结果（值、误差估计、求值次数）。
 *
 * @return ::LMMC_STATUS_OK 在容差内收敛；
 *         ::LMMC_STATUS_CONVERGENCE_FAILED 预算不足以完成求积层，或十层后仍未收敛；
 *         ::LMMC_STATUS_NUMERICAL_FAILURE 回调或中间结果为非有限值；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若参数超出有效范围或指针为 NULL。
 *
 * @note
 * - 仅对可表示的内部节点调用 @p f。
 * - 实际回调次数记录在 out->num_evals 中，不超过 @p max_nodes 次。
 * - 仅比较并提交完整层的估计。
 * - 预算/数值失败保留最后完整层的 value（尚无完整层则为零），error 为正无穷。
 * - 完成十层仍不收敛时，保留最后完整层值和最后两个完整层的误差估计。
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
