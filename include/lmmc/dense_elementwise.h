/** @file dense_elementwise.h */
#ifndef LMMC_DENSE_ELEMENTWISE_H
#define LMMC_DENSE_ELEMENTWISE_H

#include "lmmc/dense_types.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 逐元素向量 Hadamard 乘积：@f$c_i = a_i \cdot b_i@f$。
 *
 * a,b,c 长度须相同；c 可与 a 或 b 别名(就地运算)。
 *
 * @param[in]  a 输入向量。
 * @param[in]  b 输入向量。
 * @param[out] c 输出向量,覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_vec_hadamard(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c);

/**
 * @brief 逐元素向量除法：@f$c_i = a_i / b_i@f$。
 *
 * 写入输出前扫描 b 的全部元素，检测零值。
 * a,b,c 长度须相同；c 可与 a 或 b 别名。
 *
 * @param[in]  a 被除向量。
 * @param[in]  b 除数向量。
 * @param[out] c 输出向量,覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - b 中存在零元素。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_vec_elementwise_div(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c);

/**
 * @brief 逐元素向量幂运算：@f$c_i = a_i^{b_i}@f$。
 *
 * 使用标准 pow() 函数；a,b,c 长度须相同；c 可与 a 或 b 别名。
 *
 * @param[in]  a 底数向量。
 * @param[in]  b 指数向量。
 * @param[out] c 输出向量,覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_vec_elementwise_pow(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c);

/**
 * @brief 逐元素矩阵 Hadamard 乘积：@f$C_{ij} = A_{ij} \cdot B_{ij}@f$。
 *
 * a,b,c 维度须完全相同；c 可与 a 或 b 别名(就地运算)。
 *
 * @param[in]  a 输入矩阵。
 * @param[in]  b 输入矩阵。
 * @param[out] c 输出矩阵,覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_mat_hadamard(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 逐元素矩阵除法：@f$C_{ij} = A_{ij} / B_{ij}@f$。
 *
 * 写入输出前扫描 b 的全部元素，检测零值。
 * a,b,c 维度须完全相同；c 可与 a 或 b 别名。
 *
 * @param[in]  a 被除矩阵。
 * @param[in]  b 除数矩阵。
 * @param[out] c 输出矩阵,覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 * - ::LMMC_STATUS_NUMERICAL_FAILURE - b 中存在零元素。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_mat_elementwise_div(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 逐元素矩阵幂运算：@f$C_{ij} = A_{ij}^{B_{ij}}@f$。
 *
 * 使用标准 pow() 函数；a,b,c 维度须完全相同；c 可与 a 或 b 别名。
 *
 * @param[in]  a 底数矩阵。
 * @param[in]  b 指数矩阵。
 * @param[out] c 输出矩阵,覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_mat_elementwise_pow(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 逐元素向量大于比较：@f$\text{out}[i] = (a_i > b_i) \;?\; 1 : 0@f$。
 *
 * a,b 长度须相同；out 数组由调用方预分配,长度至少为 a->size。
 *
 * @param[in]  a   输入向量。
 * @param[in]  b   输入向量。
 * @param[out] out 输出整数数组，由调用方预分配。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 *
 * @note 覆写 out 数组，不分配内存。
 */
lmmc_status_t lmmc_vec_cmp_gt(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out);

/**
 * @brief 逐元素向量小于比较：@f$\text{out}[i] = (a_i < b_i) \;?\; 1 : 0@f$。
 *
 * a,b 长度须相同；out 数组由调用方预分配。
 *
 * @param[in]  a   输入向量。
 * @param[in]  b   输入向量。
 * @param[out] out 输出整数数组，由调用方预分配。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 *
 * @note 覆写 out 数组，不分配内存。
 */
lmmc_status_t lmmc_vec_cmp_lt(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out);

/**
 * @brief 逐元素向量大于等于比较：@f$\text{out}[i] = (a_i \geq b_i) \;?\; 1 : 0@f$。
 *
 * a,b 长度须相同；out 数组由调用方预分配。
 *
 * @param[in]  a   输入向量。
 * @param[in]  b   输入向量。
 * @param[out] out 输出整数数组，由调用方预分配。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 *
 * @note 覆写 out 数组，不分配内存。
 */
lmmc_status_t lmmc_vec_cmp_ge(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out);

/**
 * @brief 逐元素向量小于等于比较：@f$\text{out}[i] = (a_i \leq b_i) \;?\; 1 : 0@f$。
 *
 * a,b 长度须相同；out 数组由调用方预分配。
 *
 * @param[in]  a   输入向量。
 * @param[in]  b   输入向量。
 * @param[out] out 输出整数数组，由调用方预分配。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 *
 * @note 覆写 out 数组，不分配内存。
 */
lmmc_status_t lmmc_vec_cmp_le(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out);

/**
 * @brief 逐元素向量近似相等比较：@f$\text{out}[i] = (|a_i - b_i| \leq \text{tol}) \;?\; 1 : 0@f$。
 *
 * a,b 长度须相同；out 数组由调用方预分配。
 *
 * @param[in]  a   输入向量。
 * @param[in]  b   输入向量。
 * @param[in]  tol 容差阈值(非负)。
 * @param[out] out 输出整数数组，由调用方预分配。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不一致。
 *
 * @note 覆写 out 数组，不分配内存。
 */
lmmc_status_t lmmc_vec_cmp_eq(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t tol, int* out);

/**
 * @brief 对向量逐元素应用标量函数：@f$\text{out}[i] = \text{func}(\text{in}[i])@f$。
 *
 * 支持就地操作( @p in 与 @p out 可指向同一向量)。
 *
 * @param[in]  in   输入向量（in == out 时就地更新）。
 * @param[in]  func 标量函数指针,接受一个 lmmc_real_t 参数并返回 lmmc_real_t。须为有效指针。
 * @param[out] out  输出向量,长度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out,in->data,out->data 或 func 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in->size != out->size。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_vec_apply(const lmmc_vec_t* in, lmmc_real_t (*func)(lmmc_real_t), lmmc_vec_t* out);

/**
 * @brief 对矩阵逐元素应用标量函数：@f$\text{out}[i][j] = \text{func}(\text{in}[i][j])@f$。
 *
 * 支持就地操作( @p in 与 @p out 可指向同一矩阵)。按各自的 stride 遍历。
 *
 * @param[in]  in   输入矩阵（in == out 时就地更新）。
 * @param[in]  func 标量函数指针,接受一个 lmmc_real_t 参数并返回 lmmc_real_t。须为有效指针。
 * @param[out] out  输出矩阵,维度须与 @p in 相同(rows 和 cols 均相等)。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out,in->data,out->data 或 func 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in 与 out 的 rows 或 cols 不同。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_mat_apply(const lmmc_mat_t* in, lmmc_real_t (*func)(lmmc_real_t), lmmc_mat_t* out);

/**
 * @brief 对向量逐元素求正弦:@f$\text{out}[i] = \sin(\text{in}[i])@f$。
 *
 * 等价于 lmmc_vec_apply(in, sin, out)。
 *
 * @param[in]  in  输入向量（in == out 时就地更新）。
 * @param[out] out 输出向量,长度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_sin(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对向量逐元素求余弦:@f$\text{out}[i] = \cos(\text{in}[i])@f$。
 *
 * 等价于 lmmc_vec_apply(in, cos, out)。
 *
 * @param[in]  in  输入向量（in == out 时就地更新）。
 * @param[out] out 输出向量,长度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_cos(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对向量逐元素求自然指数:@f$\text{out}[i] = \exp(\text{in}[i])@f$。
 *
 * 等价于 lmmc_vec_apply(in, exp, out)。
 *
 * @param[in]  in  输入向量（in == out 时就地更新）。
 * @param[out] out 输出向量,长度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_exp(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对向量逐元素求自然对数:@f$\text{out}[i] = \ln(\text{in}[i])@f$。
 *
 * 等价于 lmmc_vec_apply(in, log, out)。
 *
 * @param[in]  in  输入向量（in == out 时就地更新）。
 * @param[out] out 输出向量,长度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_log(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对向量逐元素求平方根:@f$\text{out}[i] = \sqrt{\text{in}[i]}@f$。
 *
 * 等价于 lmmc_vec_apply(in, sqrt, out)。
 *
 * @param[in]  in  输入向量（in == out 时就地更新）。
 * @param[out] out 输出向量,长度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_sqrt(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对向量逐元素求绝对值:@f$\text{out}[i] = |\text{in}[i]|@f$。
 *
 * 等价于 lmmc_vec_apply(in, fabs, out)。
 *
 * @param[in]  in  输入向量（in == out 时就地更新）。
 * @param[out] out 输出向量,长度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_abs(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对矩阵逐元素求正弦:@f$\text{out}[i][j] = \sin(\text{in}[i][j])@f$。
 *
 * 等价于 lmmc_mat_apply(in, sin, out)。
 *
 * @param[in]  in  输入矩阵（in == out 时就地更新）。
 * @param[out] out 输出矩阵,维度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_sin(const lmmc_mat_t* in, lmmc_mat_t* out);

/**
 * @brief 对矩阵逐元素求余弦:@f$\text{out}[i][j] = \cos(\text{in}[i][j])@f$。
 *
 * 等价于 lmmc_mat_apply(in, cos, out)。
 *
 * @param[in]  in  输入矩阵（in == out 时就地更新）。
 * @param[out] out 输出矩阵,维度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_cos(const lmmc_mat_t* in, lmmc_mat_t* out);

/**
 * @brief 对矩阵逐元素求自然指数:@f$\text{out}[i][j] = \exp(\text{in}[i][j])@f$。
 *
 * 等价于 lmmc_mat_apply(in, exp, out)。
 *
 * @param[in]  in  输入矩阵（in == out 时就地更新）。
 * @param[out] out 输出矩阵,维度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_exp(const lmmc_mat_t* in, lmmc_mat_t* out);

/**
 * @brief 对矩阵逐元素求自然对数:@f$\text{out}[i][j] = \ln(\text{in}[i][j])@f$。
 *
 * 等价于 lmmc_mat_apply(in, log, out)。
 *
 * @param[in]  in  输入矩阵（in == out 时就地更新）。
 * @param[out] out 输出矩阵,维度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_log(const lmmc_mat_t* in, lmmc_mat_t* out);

/**
 * @brief 对矩阵逐元素求平方根:@f$\text{out}[i][j] = \sqrt{\text{in}[i][j]}@f$。
 *
 * 等价于 lmmc_mat_apply(in, sqrt, out)。
 *
 * @param[in]  in  输入矩阵（in == out 时就地更新）。
 * @param[out] out 输出矩阵,维度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_sqrt(const lmmc_mat_t* in, lmmc_mat_t* out);

/**
 * @brief 对矩阵逐元素求绝对值:@f$\text{out}[i][j] = |\text{in}[i][j]|@f$。
 *
 * 等价于 lmmc_mat_apply(in, fabs, out)。
 *
 * @param[in]  in  输入矩阵（in == out 时就地更新）。
 * @param[out] out 输出矩阵,维度须与 @p in 相同。覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - in,out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_abs(const lmmc_mat_t* in, lmmc_mat_t* out);

#ifdef __cplusplus
}
#endif

#endif
