/** @file dense_storage.h */
#ifndef LMMC_DENSE_STORAGE_H
#define LMMC_DENSE_STORAGE_H

#include "lmmc/dense_types.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建 @p rows x @p cols 的零矩阵。
 *
 * 用 lmmc_alloc 分配 rows*cols 个 lmmc_real_t 的连续缓冲区，
 * stride=cols，owns_data=1。
 *
 * @param[in]  rows    行数，须 > 0。
 * @param[in]  cols    列数，须 > 0。
 * @param[out] out_mat 调用方提供的矩阵结构体，全部字段被覆写。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - rows 或 cols 为 0、out_mat 为 NULL 或 rows*cols 溢出。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 * @note 调用方须用 ::lmmc_mat_destroy 释放缓冲区。
 * @note 函数无共享状态；lmmc_alloc 线程安全时支持并发调用。
 */
lmmc_status_t lmmc_mat_create(size_t rows, size_t cols, lmmc_mat_t* out_mat);

/**
 * @brief 创建引用调用方连续缓冲区的矩阵视图。
 *
 * 所有权保留给调用方，data 的生命周期须覆盖视图使用期。
 * ::lmmc_mat_destroy 仅销毁视图，写入视图会直接修改 data。
 *
 * @param[in]  rows    行数，须 > 0。
 * @param[in]  cols    列数，须 > 0。
 * @param[in]  stride  相邻行首元素间距，须 >= @p cols。
 * @param[in]  data    非 NULL 的外部缓冲区，至少容纳 rows*stride 个 lmmc_real_t。
 * @param[out] out_mat 输出矩阵视图，全部字段被覆写。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 参数非法。
 * @note 直接使用外部缓冲区，无内存分配。
 */
lmmc_status_t lmmc_mat_wrap(size_t rows, size_t cols, size_t stride, lmmc_real_t* data, lmmc_mat_t* out_mat);

/**
 * @brief 销毁矩阵，释放其拥有的缓冲区并清零结构体字段。
 *
 * owns_data 非零时用 lmmc_free 释放 data；外部缓冲区保留。
 * 销毁后 data 为 NULL，rows、cols、stride 为零；重新 create/wrap 后可用于运算。
 *
 * @param[in,out] mat 待销毁矩阵，接受 NULL 或已销毁（data==NULL）的矩阵。
 */
void lmmc_mat_destroy(lmmc_mat_t* mat);

/**
 * @brief 就地将矩阵所有元素设为 @p value。
 *
 * @param[in,out] mat   已创建且 data 非 NULL 的矩阵。
 * @param[in]     value 填充值。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - mat 或 mat->data 为 NULL。
 * @note 按 stride 写入 data[i*stride+j]，无内存分配或释放。
 */
lmmc_status_t lmmc_mat_fill(lmmc_mat_t* mat, lmmc_real_t value);

/**
 * @brief 逐元素复制矩阵内容：dst = src。
 *
 * @param[in]  src 源矩阵，保持不变。
 * @param[out] dst 已通过 create/wrap 初始化的目标矩阵，行列数须与 src 相同，
 *                 stride 可不同，全部元素被覆写。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 任一指针或 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - src 与 dst 维度不同。
 * @note 使用目标缓冲区，无内存分配。
 */
lmmc_status_t lmmc_mat_copy(const lmmc_mat_t* src, lmmc_mat_t* dst);

/**
 * @brief 计算矩阵转置：dst = src^T。
 *
 * src 与 dst 的半开存储包络须互不重叠。
 *
 * @param[in]  src 源矩阵（mxn），保持不变。
 * @param[out] dst 已创建为 (src->cols x src->rows) 的目标矩阵，内容被覆写。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL、存储包络溢出或 src/dst 重叠。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - dst 维度不匹配，优先于重叠检查。
 * @note 使用目标缓冲区，无内存分配。
 */
lmmc_status_t lmmc_mat_transpose_to(const lmmc_mat_t* src, lmmc_mat_t* dst);

/**
 * @brief 创建长度为 @p size 的零向量。
 *
 * @param[in]  size    元素个数，须 > 0。
 * @param[out] out_vec 输出向量，全部字段被覆写，owns_data=1。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - size==0、out_vec==NULL 或 size*sizeof 溢出。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 * @note 分配堆内存，调用方须用 ::lmmc_vec_destroy 释放。
 */
lmmc_status_t lmmc_vec_create(size_t size, lmmc_vec_t* out_vec);

/**
 * @brief 创建引用调用方连续缓冲区的向量视图。
 *
 * 所有权保留给调用方，data 的生命周期须覆盖视图使用期。
 * ::lmmc_vec_destroy 仅销毁视图，写入视图会直接修改 data。
 *
 * @param[in]  size    元素个数，须 > 0。
 * @param[in]  data    非 NULL 的外部缓冲区，至少容纳 size 个 lmmc_real_t。
 * @param[out] out_vec 输出向量视图。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 参数非法。
 * @note 直接使用外部缓冲区，无内存分配。
 */
lmmc_status_t lmmc_vec_wrap(size_t size, lmmc_real_t* data, lmmc_vec_t* out_vec);

/**
 * @brief 销毁向量，释放其拥有的缓冲区并清零字段。
 *
 * owns_data 非零时释放 data；外部缓冲区保留。
 * 销毁后 data 为 NULL，size 为零，vec 须重新初始化后用于运算。
 *
 * @param[in,out] vec 待销毁向量，接受 NULL。
 */
void lmmc_vec_destroy(lmmc_vec_t* vec);

/**
 * @brief 就地将向量所有元素设为 @p value。
 *
 * @param[in,out] vec   已创建的向量。
 * @param[in]     value 填充值。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - vec 或 vec->data 为 NULL。
 * @note 使用原缓冲区，无内存分配。
 */
lmmc_status_t lmmc_vec_fill(lmmc_vec_t* vec, lmmc_real_t value);

/**
 * @brief 逐元素复制向量内容：dst = src。
 *
 * @param[in]  src 源向量，保持不变。
 * @param[out] dst 与 src 等长的目标向量，全部元素被覆写。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或 size==0。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - src->size != dst->size。
 * @note 使用目标缓冲区，无内存分配。
 */
lmmc_status_t lmmc_vec_copy(const lmmc_vec_t* src, lmmc_vec_t* dst);

/**
 * @brief 就地交换两个等长向量的全部元素。
 *
 * @param[in,out] x 第一个向量。
 * @param[in,out] y 与 x 等长的第二个向量。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或 size==0。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - x->size != y->size。
 * @note 使用原缓冲区，无内存分配。
 */
lmmc_status_t lmmc_vec_swap(lmmc_vec_t* x, lmmc_vec_t* y);

/**
 * @brief 创建 nxn 单位矩阵，对角线为 1，其余元素为 0。
 *
 * @param[in]  n       矩阵阶数，须 > 0。
 * @param[out] out_mat 输出矩阵，owns_data=1。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - n==0 或 out_mat==NULL。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 * @note 通过 ::lmmc_mat_create 分配内存，调用方须用 ::lmmc_mat_destroy 释放。
 */
lmmc_status_t lmmc_mat_identity(size_t n, lmmc_mat_t* out_mat);

#ifdef __cplusplus
}
#endif

#endif
