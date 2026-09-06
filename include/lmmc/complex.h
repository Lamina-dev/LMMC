/**
 * @file complex.h
 * @brief 复数类型与运算模块.
 *
 * 本模块提供复数类型 ::lmmc_complex_t 及其基本算术运算(加,减,乘,除,
 * 共轭,模,辐角),极坐标构造,超越函数(exp,log,sqrt,sin,cos,pow),
 * 以及复数向量 ::lmmc_cvec_t 和复数矩阵 ::lmmc_cmat_t 的生命周期管理.
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
 * @brief 复数类型,包含实部与虚部.
 *
 * 以笛卡尔坐标形式存储复数 @f$ z = \text{real} + i \cdot \text{imag} @f$.
 */
typedef struct {
    lmmc_real_t real;  /**< 实部. */
    lmmc_real_t imag;  /**< 虚部. */
} lmmc_complex_t;

/**
 * @brief 复数稠密向量.
 *
 * 当 ::lmmc_cvec_t::owns_data 非零时,::lmmc_cvec_destroy 会释放 @c data.
 */
typedef struct {
    size_t size;            /**< 元素个数. */
    lmmc_complex_t* data;  /**< 数据缓冲区起始地址. */
    int owns_data;          /**< 是否拥有缓冲区所有权(非 0 表示销毁时释放). */
} lmmc_cvec_t;

/**
 * @brief 复数稠密矩阵(行优先存储).
 *
 * 元素 @c (i,j) 位于 @c data[i * stride + j],其中 @c stride >= cols.
 * 当 ::lmmc_cmat_t::owns_data 非零时,::lmmc_cmat_destroy 会释放 @c data.
 */
typedef struct {
    size_t rows;            /**< 行数. */
    size_t cols;            /**< 列数. */
    size_t stride;          /**< 行步距(每行实际占用的元素数,>= @c cols). */
    lmmc_complex_t* data;  /**< 数据缓冲区起始地址. */
    int owns_data;          /**< 是否拥有缓冲区所有权(非 0 表示销毁时释放). */
} lmmc_cmat_t;

/**
 * @brief 从笛卡尔坐标构造复数.
 *
 * 将 @p real 和 @p imag 写入输出参数 @p out.
 *
 * @param[in]  real 实部.
 * @param[in]  imag 虚部.
 * @param[out] out  输出复数.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 输入非有限或 out 为 NULL.
 */
lmmc_status_t lmmc_complex_create(lmmc_real_t real, lmmc_real_t imag, lmmc_complex_t* out);

/**
 * @brief 从极坐标构造复数.
 *
 * 计算 @f$ z = r \cdot (\cos\theta + i \cdot \sin\theta) @f$.
 * @p r 必须为有限非负数,@p theta 必须有限.
 * 当 @p r 为 0 时,结果为 0+0i.
 *
 * @param[in]  r     有限非负的模.
 * @param[in]  theta 有限辐角(弧度).
 * @param[out] out   输出复数.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 输入非有限、r 为负数或 out 为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 结果无法表示为有限值.
 */
lmmc_status_t lmmc_complex_from_polar(lmmc_real_t r, lmmc_real_t theta, lmmc_complex_t* out);

/**
 * @brief 复数加法:@f$ \text{out} = a + b @f$.
 *
 * @param[in]  a   加数.
 * @param[in]  b   加数.
 * @param[out] out 输出和.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入或结果非有限.
 */
lmmc_status_t lmmc_complex_add(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out);

/**
 * @brief 复数减法:@f$ \text{out} = a - b @f$.
 *
 * @param[in]  a   被减数.
 * @param[in]  b   减数.
 * @param[out] out 输出差.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入或结果非有限.
 */
lmmc_status_t lmmc_complex_sub(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out);

/**
 * @brief 复数乘法:@f$ \text{out} = a \cdot b @f$.
 *
 * 实部和虚部分别计算二项乘积和，普通幅值使用 FMA 与乘积残差保留消减余量；
 * 乘积或结果位于指数边界时，使用尾数与公共二进制指数完成运算。
 * 两个结果均为有限值时写入 @p out；输出可以与任一输入共用存储。
 *
 * @param[in]  a   乘数.
 * @param[in]  b   乘数.
 * @param[out] out 输出积.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入或结果非有限.
 */
lmmc_status_t lmmc_complex_mul(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out);

/**
 * @brief 复数除法:@f$ \text{out} = a / b @f$.
 *
 * 分子实部、虚部及分母平方模分别按二进制指数缩放，在尾数除法后恢复尺度。
 * 除数 @p b 的两个分量均为零时返回错误。
 * 支持 @p out 与任一输入重合；失败时保持输出不变。
 *
 * @param[in]  a   有限被除数.
 * @param[in]  b   有限除数(模不可为零).
 * @param[out] out 输出商.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入非有限、除数为零或有限结果不可表示.
 */
lmmc_status_t lmmc_complex_div(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out);

/**
 * @brief 复数共轭:@f$ \text{out} = \overline{z} = (\text{real}, -\text{imag}) @f$.
 *
 * @param[in]  z   输入复数.
 * @param[out] out 输出共轭.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入或结果非有限.
 */
lmmc_status_t lmmc_complex_conj(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 采用缩放计算复数模:@f$ |z| = \sqrt{\text{real}^2 + \text{imag}^2} @f$.
 *
 * @param[in]  z   有限复数.
 * @param[out] out 输出模值(非负实数).
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入或结果非有限.
 */
lmmc_status_t lmmc_complex_modulus(const lmmc_complex_t* z, lmmc_real_t* out);

/**
 * @brief 复数辐角:@f$ \arg(z) = \text{atan2}(\text{imag}, \text{real}) @f$.
 *
 * 返回值范围为 @f$ (-\pi, \pi] @f$.
 *
 * @param[in]  z   输入复数.
 * @param[out] out 输出辐角(弧度).
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入或结果非有限.
 */
lmmc_status_t lmmc_complex_arg(const lmmc_complex_t* z, lmmc_real_t* out);

/**
 * @brief 复数指数函数:@f$ e^z = e^x (\cos y + i \sin y) @f$,其中 @f$ z = x + iy @f$.
 *
 * 正常幅值直接计算；当实指数落在溢出或次正规数范围时，使用两个
 * @f$e^{x/2}@f$ 因子，将三角因子纳入最终乘积后形成结果分量。
 * 输出可以与输入共用存储；成功时写入完整结果，失败时保持输出不变。
 *
 * @see ISO C11 committee draft N1570, 7.3.7.1 (cexp).
 * https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
 *
 * @param[in]  z   输入复数.
 * @param[out] out 输出 @f$ e^z @f$.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入或结果非有限.
 */
lmmc_status_t lmmc_complex_exp(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 复数主值对数:@f$ \ln z = \ln(\operatorname{hypot}(x,y)) + i \arg(z) @f$.
 *
 * 最大分量绝对值位于 [0.5, 1] 时，实部通过补偿平方和计算
 * `log1p(x*x + y*y - 1)/2`，保留单位模附近的小量。
 * 其他幅值使用 `log(max(|x|,|y|)) + log1p(ratio^2)/2`；
 * 有限输入即使数学模超出 `double` 范围仍可返回有限对数。
 *
 * 当 @p z 的模为零时(即 z = 0+0i),返回域错误.
 * 负实轴两侧由虚部的有符号零区分，主辐角分别为 @f$+\pi@f$ 和 @f$-\pi@f$。
 * 输出可以与输入共用存储；成功时写入完整结果，失败时保持输出不变。
 *
 * @see ISO C11 committee draft N1570, 7.3.7.2 (clog).
 * https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
 *
 * @param[in]  z   有限复数(模不可为零).
 * @param[out] out 输出主值对数.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_OUT_OF_RANGE - z 的模为零(对数无定义).
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入或结果非有限.
 */
lmmc_status_t lmmc_complex_log(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 采用缩放算法计算复数主值平方根.
 *
 * 即使有限实部与虚部对应的数学模超过可表示范围，只要平方根的两个
 * 分量仍可表示，算法也通过输入尺度分解返回有限结果；同时避免
 * @f$x^2+y^2@f$ 和 @f$|z| \pm x@f$ 的中间溢出与相消。
 * 虚部非零时，以其幅值除以大分量幅值的两倍得到小分量幅值。
 * 该分母保持有限且非零；缩放因子在除法前合并，使次正规小分量只经历一次除法舍入。
 * 结果实部为非负值（零取正零），虚部保留输入虚部的符号，包括有符号零。
 * 输出可以与输入共用存储；成功时写入完整结果，失败时保持输出不变。
 *
 * @see ISO C11 committee draft N1570, 7.3.8.3 (csqrt).
 * https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
 *
 * @param[in]  z   有限复数.
 * @param[out] out 输出主值平方根.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入或结果非有限.
 */
lmmc_status_t lmmc_complex_sqrt(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 复数正弦:@f$ \sin(z) = \sin(x)\cosh(y) + i\cos(x)\sinh(y) @f$.
 *
 * 有限双曲因子使用直接乘积；双曲因子溢出时，将主导指数项
 * 分解为两个 @f$e^{|y|/2}@f$ 因子，与三角因子共同形成结果分量。
 * 输出可以与输入共用存储；成功时写入完整结果，失败时保持输出不变。
 *
 * @see NIST DLMF, 4.28.1 and 4.28.2, hyperbolic exponential identities.
 * https://dlmf.nist.gov/4.28
 *
 * @param[in]  z   输入复数.
 * @param[out] out 输出复数正弦值.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入或结果非有限.
 */
lmmc_status_t lmmc_complex_sin(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 复数余弦:@f$ \cos(z) = \cos(x)\cosh(y) - i\sin(x)\sinh(y) @f$.
 *
 * 与复数正弦共用双曲乘积缩放算法，在最终分量的幅值范围内完成计算。
 * 输出可以与输入共用存储；成功时写入完整结果，失败时保持输出不变。
 *
 * @see lmmc_complex_sin
 *
 * @param[in]  z   输入复数.
 * @param[out] out 输出复数余弦值.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入或结果非有限.
 */
lmmc_status_t lmmc_complex_cos(const lmmc_complex_t* z, lmmc_complex_t* out);

/**
 * @brief 复数幂:@f$ \text{base}^{\text{exp}} = e^{\text{exp} \cdot \ln(\text{base})} @f$.
 *
 * 先验证两个输入均为有限复数。非零底数的实整数指数使用二进制平方求幂，
 * 负整数指数先取倒数；其他指数使用主值对数计算。
 * 对零底数，本接口定义零次幂为 1+0i；指数实部为正时返回 0+0i；
 * 指数实部为负或指数为非零纯虚数时返回域错误。
 * 输出可以与任一输入共用存储；成功时写入完整结果，失败时保持输出不变。
 *
 * @see ISO C11 committee draft N1570, 7.3.8.2 (cpow), principal branch.
 * https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
 *
 * @param[in]  base 有限底数.
 * @param[in]  exp  有限指数.
 * @param[out] out  输出幂值.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针为 NULL.
 * - ::LMMC_STATUS_OUT_OF_RANGE - 零底数的指数实部为负，或指数为非零纯虚数.
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - 输入非有限，或数值运算失败（包括中间结果或输出溢出）.
 */
lmmc_status_t lmmc_complex_pow(const lmmc_complex_t* base, const lmmc_complex_t* exp, lmmc_complex_t* out);

/**
 * @brief 分配并初始化一个长度为 @p size 的复数向量,所有元素置零.
 *
 * 内部通过 lmmc_alloc 分配 size 个 lmmc_complex_t 的连续缓冲区,
 * 所有实部和虚部初始化为 0,owns_data 置 1.
 *
 * @param[in]  size    元素个数,必须 > 0.
 * @param[out] out     输出向量结构体.调用方需配对调用 ::lmmc_cvec_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - size 为 0 或 out 为 NULL.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存(owns_data=1).out 的所有字段被覆写.
 */
lmmc_status_t lmmc_cvec_create(size_t size, lmmc_cvec_t* out);

/**
 * @brief 分配并初始化一个 @p rows x @p cols 的复数矩阵,所有元素置零.
 *
 * 内部通过 lmmc_alloc 分配 rows*cols 个 lmmc_complex_t 的连续缓冲区,
 * stride 设为 cols(紧凑存储),所有实部和虚部初始化为 0,owns_data 置 1.
 *
 * @param[in]  rows    行数,必须 > 0.
 * @param[in]  cols    列数,必须 > 0.
 * @param[out] out     输出矩阵结构体.调用方需配对调用 ::lmmc_cmat_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - rows 或 cols 为 0,或 out 为 NULL.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存(owns_data=1).out 的所有字段被覆写.
 */
lmmc_status_t lmmc_cmat_create(size_t rows, size_t cols, lmmc_cmat_t* out);

/**
 * @brief 销毁复数向量,释放其拥有的底层缓冲区.
 *
 * 若 vec->owns_data 非零,调用 lmmc_free 释放 vec->data;
 * 否则仅将结构体字段清零.对 NULL 指针安全.
 *
 * @param[in,out] vec 待销毁的复数向量,可为 NULL.
 *
 * @par 副作用
 * - 若 owns_data:释放堆内存,vec->data 置 NULL,size 清零.
 * - 若 !owns_data:仅清零结构体字段,不释放外部缓冲区.
 * - 销毁后该向量不可再用于任何运算.
 */
void lmmc_cvec_destroy(lmmc_cvec_t* vec);

/**
 * @brief 销毁复数矩阵,释放其拥有的底层缓冲区.
 *
 * 若 mat->owns_data 非零,调用 lmmc_free 释放 mat->data;
 * 否则仅将结构体字段清零.对 NULL 指针安全.
 *
 * @param[in,out] mat 待销毁的复数矩阵,可为 NULL.
 *
 * @par 副作用
 * - 若 owns_data:释放堆内存,mat->data 置 NULL,rows/cols/stride 清零.
 * - 若 !owns_data:仅清零结构体字段,不释放外部缓冲区.
 * - 销毁后该矩阵不可再用于任何运算.
 */
void lmmc_cmat_destroy(lmmc_cmat_t* mat);

#ifdef __cplusplus
}
#endif

#endif
