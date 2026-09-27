/** @file dense_arithmetic.h */
#ifndef LMMC_DENSE_ARITHMETIC_H
#define LMMC_DENSE_ARITHMETIC_H

#include "lmmc/dense_types.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 通用矩阵-矩阵乘法(GEMM):@f$C \leftarrow \alpha \cdot \mathrm{op}(A) \cdot \mathrm{op}(B) + \beta \cdot C@f$。
 *
 * op(X) = X(transX==0)或 X^T(transX!=0)。
 * 设 op(A) 为 MxK,op(B) 为 KxN,则 C 必须为 MxN。
 * 使用内置分块三重循环实现，块大小为 64。
 *
 * @param[in]     alpha  标量乘子 alpha。
 * @param[in]     A      输入矩阵 A(不被修改)。
 * @param[in]     transA 非零表示对 A 取转置。
 * @param[in]     B      输入矩阵 B(不被修改)。
 * @param[in]     transB 非零表示对 B 取转置。
 * @param[in]     beta   标量乘子 beta.beta==0 时 C 的旧值被忽略(可含 NaN)。
 * @param[in,out] C      输出矩阵,维度须为 MxN.C 不可与 A 或 B 别名。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL、存储包络溢出，或 C 与 A/B 重叠。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - op(A) 的列数 != op(B) 的行数,或 C 维度不匹配；
 *   维度错误优先于重叠检查。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_mat_gemm(lmmc_real_t alpha, const lmmc_mat_t* A, int transA,
    const lmmc_mat_t* B, int transB, lmmc_real_t beta, lmmc_mat_t* C);

/**
 * @brief 通用矩阵-向量乘法(GEMV):@f$y \leftarrow \alpha \cdot \mathrm{op}(A) \cdot x + \beta \cdot y@f$。
 *
 * op(A) 为 MxN 时,要求 x->size==N,y->size==M。
 * y 的半开存储包络不得与 A 或 x 的存储包络重叠。
 *
 * @param[in]     alpha  标量乘子 alpha。
 * @param[in]     A      输入矩阵 A(不被修改)。
 * @param[in]     transA 非零表示对 A 取转置。
 * @param[in]     x      输入向量 x(不被修改)。
 * @param[in]     beta   标量乘子 beta.beta==0 时 y 的旧值被忽略。
 * @param[in,out] y      输出向量,长度须为 op(A) 的行数，且不得与 A 或 x 重叠。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL、存储包络溢出或输出重叠。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 向量长度与矩阵维度不匹配（优先于重叠检查）。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_mat_gemv(lmmc_real_t alpha, const lmmc_mat_t* A, int transA,
    const lmmc_vec_t* x, lmmc_real_t beta, lmmc_vec_t* y);

/**
 * @brief 计算矩阵乘积：c = a * b。
 *
 * 等价于 lmmc_mat_gemm(1.0, a, 0, b, 0, 0.0, c)。
 * a 为 mxk,b 为 kxn,c 必须已创建为 mxn.c 不可与 a 或 b 别名。
 *
 * @param[in]  a 左矩阵(不被修改)。
 * @param[in]  b 右矩阵(不被修改)。
 * @param[out] c 结果矩阵,旧内容被完全覆写。
 *
 * @return 同 ::lmmc_mat_gemm。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_mat_mul(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 计算矩阵的 Frobenius 范数:@f$\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}@f$。
 *
 * 采用缩放平方和累加；最终结果超出 binary64 范围时可为无穷。
 *
 * @param[in]  a        输入矩阵(不被修改)。
 * @param[out] out_norm 输出范数值。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 */
lmmc_status_t lmmc_mat_norm_fro(const lmmc_mat_t* a, lmmc_real_t* out_norm);

/**
 * @brief 计算两个等长向量的内积:@f$\text{out\_dot} = \sum_i a_i \cdot b_i@f$。
 *
 * @param[in]  a       第一个向量(不被修改)。
 * @param[in]  b       第二个向量(不被修改),长度须与 a 相同。
 * @param[out] out_dot 输出内积值。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - a->size != b->size。
 */
lmmc_status_t lmmc_vec_dot(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t* out_dot);

/**
 * @brief 计算矩阵与向量的乘积：y = A * x。
 *
 * 等价于 lmmc_mat_gemv(1.0, a, 0, x, 0.0, y)。
 * a 为 mxn,x->size==n,y->size==m.y 的旧内容被完全覆写，且 y 不得与 a 或 x 重叠。
 *
 * @param[in]  a 输入矩阵(不被修改)。
 * @param[in]  x 输入向量(不被修改)。
 * @param[out] y 输出向量,旧内容被覆写。
 *
 * @return 同 ::lmmc_mat_gemv。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_mat_vec_mul(const lmmc_mat_t* a, const lmmc_vec_t* x, lmmc_vec_t* y);

/**
 * @brief 计算向量的欧几里得(L2)范数:@f$\|x\|_2 = \sqrt{\sum_i x_i^2}@f$。
 * 采用缩放平方和累加；最终结果超出 binary64 范围时可为无穷。
 *
 * @param[in]  x        输入向量(不被修改),size 必须 > 0。
 * @param[out] out_norm 输出范数值(非负)。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或 size==0。
 */
lmmc_status_t lmmc_vec_norm2(const lmmc_vec_t* x, lmmc_real_t* out_norm);

/**
 * @brief 计算向量的无穷范数:@f$\|x\|_\infty = \max_i |x_i|@f$。
 *
 * @param[in]  x        输入向量(不被修改),size 必须 > 0。
 * @param[out] out_norm 输出范数值(非负)。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或 size==0。
 */
lmmc_status_t lmmc_vec_norm_inf(const lmmc_vec_t* x, lmmc_real_t* out_norm);

/**
 * @brief 就地向量缩放:@f$x \leftarrow \alpha \cdot x@f$。
 *
 * @param[in,out] x     待缩放向量,size 必须 > 0。
 * @param[in]     alpha 缩放因子。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - x 或 x->data 为 NULL 或 size==0。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_vec_scale(lmmc_vec_t* x, lmmc_real_t alpha);

/**
 * @brief 向量 AXPY 运算:@f$y \leftarrow \alpha \cdot x + y@f$。
 *
 * x 与 y 长度须相同。
 *
 * @param[in]     alpha 缩放因子。
 * @param[in]     x     输入向量(不被修改)。
 * @param[in,out] y     累加目标向量,就地修改。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或 size==0。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - x->size != y->size。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_vec_axpy(lmmc_real_t alpha, const lmmc_vec_t* x, lmmc_vec_t* y);

/**
 * @brief 计算向量元素绝对值之和(L1 范数):@f$\sum_i |x_i|@f$。
 *
 * @param[in]  x        输入向量(不被修改),size 必须 > 0。
 * @param[out] out_asum 输出绝对值和(非负)。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或 size==0。
 */
lmmc_status_t lmmc_vec_asum(const lmmc_vec_t* x, lmmc_real_t* out_asum);

/**
 * @brief 返回绝对值最大元素的下标。
 *
 * 若有多个相同最大值,返回最小下标。
 *
 * @param[in]  x       输入向量(不被修改),size 必须 > 0。
 * @param[out] out_idx 输出下标（从 0 开始）。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或 size==0。
 */
lmmc_status_t lmmc_vec_iamax(const lmmc_vec_t* x, size_t* out_idx);

/**
 * @brief 同维矩阵逐元素加法:c = a + b。
 *
 * a,b,c 维度须完全相同.c 可与 a 或 b 别名(就地加法)。
 *
 * @param[in]  a 输入矩阵(不被修改)。
 * @param[in]  b 输入矩阵(不被修改)。
 * @param[out] c 输出矩阵,内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_mat_add(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 同维矩阵逐元素减法:c = a - b。
 *
 * a,b,c 维度须完全相同.c 可与 a 或 b 别名。
 *
 * @param[in]  a 输入矩阵(不被修改)。
 * @param[in]  b 输入矩阵(不被修改)。
 * @param[out] c 输出矩阵,内容被覆写。
 *
 * @return 同 ::lmmc_mat_add。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_mat_sub(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 就地矩阵标量乘:@f$A \leftarrow \alpha \cdot A@f$。
 *
 * @param[in,out] a     待缩放矩阵。
 * @param[in]     alpha 缩放因子。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - a 或 a->data 为 NULL。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_mat_scale(lmmc_mat_t* a, lmmc_real_t alpha);

/**
 * @brief 计算方阵的迹:@f$\mathrm{tr}(A) = \sum_i A_{ii}@f$。
 *
 * 要求 a->rows == a->cols。
 *
 * @param[in]  a         输入方阵(不被修改)。
 * @param[out] out_trace 输出迹值。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或矩阵非方阵。
 */
lmmc_status_t lmmc_mat_trace(const lmmc_mat_t* a, lmmc_real_t* out_trace);

/**
 * @brief 计算矩阵的 1-范数(最大绝对列和):@f$\|A\|_1 = \max_j \sum_i |A_{ij}|@f$。
 *
 * 无临时内存分配。
 *
 * @param[in]  a        输入矩阵(不被修改)。
 * @param[out] out_norm 输出范数值。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 */
lmmc_status_t lmmc_mat_norm1(const lmmc_mat_t* a, lmmc_real_t* out_norm);

/**
 * @brief 计算矩阵的无穷范数(最大绝对行和):@f$\|A\|_\infty = \max_i \sum_j |A_{ij}|@f$。
 *
 * 无临时内存分配。
 *
 * @param[in]  a        输入矩阵(不被修改)。
 * @param[out] out_norm 输出范数值。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 */
lmmc_status_t lmmc_mat_norm_inf(const lmmc_mat_t* a, lmmc_real_t* out_norm);

/**
 * @brief 计算三维向量叉积:@f$c = a \times b@f$。
 *
 * 公式:
 * - @f$c_0 = a_1 b_2 - a_2 b_1@f$
 * - @f$c_1 = a_2 b_0 - a_0 b_2@f$
 * - @f$c_2 = a_0 b_1 - a_1 b_0@f$
 *
 * 支持别名(c 可与 a 或 b 为同一向量),内部使用临时变量计算。
 *
 * @param[in]  a 输入向量(不被修改),长度必须为 3。
 * @param[in]  b 输入向量(不被修改),长度必须为 3。
 * @param[out] c 输出向量,长度必须为 3,内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 任一向量长度不为 3。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_vec_cross(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c);

#ifdef __cplusplus
}
#endif

#endif
