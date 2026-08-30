/**
 * @file complex.h
 * @brief 复数类型与运算模块。
 *
 * 本模块提供复数类型 ::lmmc_complex_t 及其基本算术运算（加、减、乘、除、
 * 共轭、模、辐角）、极坐标构造、超越函数（exp、log、sqrt、sin、cos、pow），
 * 以及复数向量 ::lmmc_cvec_t 和复数矩阵 ::lmmc_cmat_t 的生命周期管理。
 */
#ifndef LMMC_COMPLEX_H
#define LMMC_COMPLEX_H

#include <stddef.h>
#include "lmmc/config.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 复数类型，包含实部与虚部。
 *
 * 以笛卡尔坐标形式存储复数 @f$ z = \text{real} + i \cdot \text{imag} @f$。
 */
typedef struct {
    lmmc_real_t real;  /**< 实部。 */
    lmmc_real_t imag;  /**< 虚部。 */
} lmmc_complex_t;

/**
 * @brief 复数稠密向量。
 *
 * 当 ::lmmc_cvec_t::owns_data 非零时，::lmmc_cvec_destroy 会释放 @c data。
 */
typedef struct {
    size_t size;            /**< 元素个数。 */
    lmmc_complex_t* data;  /**< 数据缓冲区起始地址。 */
    int owns_data;          /**< 是否拥有缓冲区所有权（非 0 表示销毁时释放）。 */
} lmmc_cvec_t;

/**
 * @brief 复数稠密矩阵（行优先存储）。
 *
 * 元素 @c (i,j) 位于 @c data[i * stride + j]，其中 @c stride >= cols。
 * 当 ::lmmc_cmat_t::owns_data 非零时，::lmmc_cmat_destroy 会释放 @c data。
 */
typedef struct {
    size_t rows;            /**< 行数。 */
    size_t cols;            /**< 列数。 */
    size_t stride;          /**< 行步距（每行实际占用的元素数，>= @c cols）。 */
    lmmc_complex_t* data;  /**< 数据缓冲区起始地址。 */
    int owns_data;          /**< 是否拥有缓冲区所有权（非 0 表示销毁时释放）。 */
} lmmc_cmat_t;

/**
 * @brief 从笛卡尔坐标构造复数。
 *
 * 将 @p real 和 @p imag 写入输出参数 @p out。
 *
 * @param[in]  real 实部。
 * @param[in]  imag 虚部。
 * @param[out] out  输出复数。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — out 为 NULL。
 */
lmmc_status_t lmmc_complex_create(lmmc_real_t real, lmmc_real_t imag, lmmc_complex_t* out);

/**
 * @brief 从极坐标构造复数。
 *
 * 计算 @f$ z = r \cdot (\cos\theta + i \cdot \sin\theta) @f$。
 * 当 @p r 为 0 时，无论 @p theta 取何值，结果均为 0+0i。
 *
 * @param[in]  r     模（非负）。
 * @param[in]  theta 辐角（弧度）。
 * @param[out] out   输出复数。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — out 为 NULL。
 */
lmmc_status_t lmmc_complex_from_polar(lmmc_real_t r, lmmc_real_t theta, lmmc_complex_t* out);

/**
 * @brief 复数加法：@f$ \text{out} = a + b @f$。
 *
 * @param[in]  a   加数。
 * @param[in]  b   加数。
 * @param[out] out 输出和。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 */
lmmc_status_t lmmc_complex_add(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out);

/**
 * @brief 复数减法：@f$ \text{out} = a - b @f$。
 *
 * @param[in]  a   被减数。
 * @param[in]  b   减数。
 * @param[out] out 输出差。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 */
lmmc_status_t lmmc_complex_sub(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out);

/**
 * @brief 复数乘法：@f$ \text{out} = a \cdot b @f$。
 *
 * 使用公式 @f$ (ac - bd) + i(ad + bc) @f$，其中 @f$ a = a_r + i \cdot a_i @f$，
 * @f$ b = b_r + i \cdot b_i @f$。
 *
 * @param[in]  a   乘数。
 * @param[in]  b   乘数。
 * @param[out] out 输出积。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 */
lmmc_status_t lmmc_complex_mul(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out);

/**
 * @brief 复数除法：@f$ \text{out} = a / b @f$。
 *
 * 使用 Smith 缩放除法控制中间结果幅值。除数 @p b 的模为零时返回错误。
 *
 * @see Robert L. Smith, “Algorithm 116: Complex Division,”
 *      Communications of the ACM 5(8), 1962.
 *
 * @param[in]  a   被除数。
 * @param[in]  b   除数（模不可为零）。
 * @param[out] out 输出商。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 * - ::LMMC_STATUS_NUMERICAL_FAILURE — 除数模为零（除零错误）。
 */
lmmc_status_t lmmc_complex_div(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out);

/**
 * @brief 复数共轭：@f$ \text{out} = \overline{z} = (\text{real}, -\text{imag}) @f$。
 *
 * @param[in]  z   输入复数。
 * @param[out] out 输出共轭。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 */
lmmc_status_t lmmc_complex_conj(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 复数模：@f$ |z| = \sqrt{\text{real}^2 + \text{imag}^2} @f$。
 *
 * @param[in]  z   输入复数。
 * @param[out] out 输出模值（非负实数）。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 */
lmmc_status_t lmmc_complex_modulus(const lmmc_complex_t* z, lmmc_real_t* out);

/**
 * @brief 复数辐角：@f$ \arg(z) = \text{atan2}(\text{imag}, \text{real}) @f$。
 *
 * 返回值范围为 @f$ (-\pi, \pi] @f$。
 *
 * @param[in]  z   输入复数。
 * @param[out] out 输出辐角（弧度）。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 */
lmmc_status_t lmmc_complex_arg(const lmmc_complex_t* z, lmmc_real_t* out);

/**
 * @brief 复数指数函数：@f$ e^z = e^x (\cos y + i \sin y) @f$，其中 @f$ z = x + iy @f$。
 *
 * @param[in]  z   输入复数。
 * @param[out] out 输出 @f$ e^z @f$。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 */
lmmc_status_t lmmc_complex_exp(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 复数主值对数：@f$ \ln z = \ln|z| + i \arg(z) @f$。
 *
 * 当 @p z 的模为零时（即 z = 0+0i），返回域错误。
 *
 * @param[in]  z   输入复数（模不可为零）。
 * @param[out] out 输出主值对数。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 * - ::LMMC_STATUS_OUT_OF_RANGE — z 的模为零（对数无定义）。
 */
lmmc_status_t lmmc_complex_log(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 复数主值平方根：@f$ \sqrt{z} = \sqrt{|z|} \cdot (\cos(\arg(z)/2) + i \sin(\arg(z)/2)) @f$。
 *
 * @param[in]  z   输入复数。
 * @param[out] out 输出主值平方根。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 */
lmmc_status_t lmmc_complex_sqrt(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 复数正弦：@f$ \sin(z) = \sin(x)\cosh(y) + i\cos(x)\sinh(y) @f$。
 *
 * @param[in]  z   输入复数。
 * @param[out] out 输出复数正弦值。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 */
lmmc_status_t lmmc_complex_sin(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 复数余弦：@f$ \cos(z) = \cos(x)\cosh(y) - i\sin(x)\sinh(y) @f$。
 *
 * @param[in]  z   输入复数。
 * @param[out] out 输出复数余弦值。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 */
lmmc_status_t lmmc_complex_cos(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 复数幂：@f$ \text{base}^{\text{exp}} = e^{\text{exp} \cdot \ln(\text{base})} @f$。
 *
 * 当 base 的模为零且 exp 的实部为负时，返回域错误。
 * 当 base = 0+0i 且 exp = 0+0i 时，按 cpow 惯例返回 1+0i。
 * 当 base = 0+0i 且 exp 实部 > 0 时，返回 0+0i。
 *
 * @param[in]  base 底数（当模为零且指数实部为负时返回错误）。
 * @param[in]  exp  指数。
 * @param[out] out  输出幂值。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针为 NULL。
 * - ::LMMC_STATUS_OUT_OF_RANGE — base 模为零且 exp 实部为负。
 */
lmmc_status_t lmmc_complex_pow(const lmmc_complex_t* base, const lmmc_complex_t* exp, lmmc_complex_t* out);

/**
 * @brief 分配并初始化一个长度为 @p size 的复数向量，所有元素置零。
 *
 * 内部通过 lmmc_alloc 分配 size 个 lmmc_complex_t 的连续缓冲区，
 * 所有实部和虚部初始化为 0，owns_data 置 1。
 *
 * @param[in]  size    元素个数，必须 > 0。
 * @param[out] out     输出向量结构体。调用方需配对调用 ::lmmc_cvec_destroy 释放。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — size 为 0 或 out 为 NULL。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存（owns_data=1）。out 的所有字段被覆写。
 */
lmmc_status_t lmmc_cvec_create(size_t size, lmmc_cvec_t* out);

/**
 * @brief 分配并初始化一个 @p rows × @p cols 的复数矩阵，所有元素置零。
 *
 * 内部通过 lmmc_alloc 分配 rows*cols 个 lmmc_complex_t 的连续缓冲区，
 * stride 设为 cols（紧凑存储），所有实部和虚部初始化为 0，owns_data 置 1。
 *
 * @param[in]  rows    行数，必须 > 0。
 * @param[in]  cols    列数，必须 > 0。
 * @param[out] out     输出矩阵结构体。调用方需配对调用 ::lmmc_cmat_destroy 释放。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — rows 或 cols 为 0，或 out 为 NULL。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存（owns_data=1）。out 的所有字段被覆写。
 */
lmmc_status_t lmmc_cmat_create(size_t rows, size_t cols, lmmc_cmat_t* out);

/**
 * @brief 销毁复数向量，释放其拥有的底层缓冲区。
 *
 * 若 vec->owns_data 非零，调用 lmmc_free 释放 vec->data；
 * 否则仅将结构体字段清零。对 NULL 指针安全。
 *
 * @param[in,out] vec 待销毁的复数向量，可为 NULL。
 *
 * @par 副作用
 * - 若 owns_data：释放堆内存，vec->data 置 NULL，size 清零。
 * - 若 !owns_data：仅清零结构体字段，不释放外部缓冲区。
 * - 销毁后该向量不可再用于任何运算。
 */
void lmmc_cvec_destroy(lmmc_cvec_t* vec);

/**
 * @brief 销毁复数矩阵，释放其拥有的底层缓冲区。
 *
 * 若 mat->owns_data 非零，调用 lmmc_free 释放 mat->data；
 * 否则仅将结构体字段清零。对 NULL 指针安全。
 *
 * @param[in,out] mat 待销毁的复数矩阵，可为 NULL。
 *
 * @par 副作用
 * - 若 owns_data：释放堆内存，mat->data 置 NULL，rows/cols/stride 清零。
 * - 若 !owns_data：仅清零结构体字段，不释放外部缓冲区。
 * - 销毁后该矩阵不可再用于任何运算。
 */
void lmmc_cmat_destroy(lmmc_cmat_t* mat);

#ifdef __cplusplus
}
#endif

#endif
