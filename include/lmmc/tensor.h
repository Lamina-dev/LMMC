/**
 * @file tensor.h
 * @brief 三阶稠密张量数据结构与基本运算。
 *
 * 元素 (i,j,k) 位于
 * @c data[i*stride0 + j*stride1 + k*stride2] ，
 * 默认行优先布局；亦支持任意视图。
 */
#ifndef LMMC_TENSOR_H
#define LMMC_TENSOR_H

#include <stddef.h>
#include "lmmc/dense.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 三阶张量结构。
 *
 * 通过 ::lmmc_tensor_reshape_view / ::lmmc_tensor_slice_view 得到的
 * 视图共享底层缓冲区且 @c owns_data == 0 。
 */
typedef struct {
    size_t dim0;          /**< 第 0 维大小。 */
    size_t dim1;          /**< 第 1 维大小。 */
    size_t dim2;          /**< 第 2 维大小。 */
    size_t stride0;       /**< 第 0 维步距。 */
    size_t stride1;       /**< 第 1 维步距。 */
    size_t stride2;       /**< 第 2 维步距。 */
    lmmc_real_t* data;    /**< 数据缓冲区。 */
    int owns_data;        /**< 是否拥有缓冲区所有权。 */
} lmmc_tensor_t;

/** @brief 创建 @p dim0 × @p dim1 × @p dim2 的张量，元素未初始化。 */
lmmc_status_t lmmc_tensor3_create(size_t dim0, size_t dim1, size_t dim2, lmmc_tensor_t* out_tensor);

/** @brief 用外部缓冲区构造张量视图。 */
lmmc_status_t lmmc_tensor3_wrap(
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t stride0,
    size_t stride1,
    size_t stride2,
    lmmc_real_t* data,
    lmmc_tensor_t* out_tensor
);

/** @brief 销毁张量。 */
void lmmc_tensor_destroy(lmmc_tensor_t* tensor);

/** @brief 用 @p value 填充张量所有元素。 */
lmmc_status_t lmmc_tensor_fill(lmmc_tensor_t* tensor, lmmc_real_t value);

/** @brief 写入元素 (i,j,k) 。 */
lmmc_status_t lmmc_tensor_set(lmmc_tensor_t* tensor, size_t i, size_t j, size_t k, lmmc_real_t value);

/** @brief 读取元素 (i,j,k) 。 */
lmmc_status_t lmmc_tensor_get(const lmmc_tensor_t* tensor, size_t i, size_t j, size_t k, lmmc_real_t* out_value);

/** @brief 计算张量 Frobenius 范数。 */
lmmc_status_t lmmc_tensor_norm_fro(const lmmc_tensor_t* tensor, lmmc_real_t* out_norm);

/** @brief 同形状张量逐元素加法。 */
lmmc_status_t lmmc_tensor_add(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
/** @brief 同形状张量逐元素减法。 */
lmmc_status_t lmmc_tensor_sub(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
/** @brief 同形状张量逐元素乘法（Hadamard）。 */
lmmc_status_t lmmc_tensor_mul(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
/** @brief 同形状张量逐元素除法（@c b 中元素须非零）。 */
lmmc_status_t lmmc_tensor_div(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
/** @brief 标量乘：@c out = alpha * tensor 。 */
lmmc_status_t lmmc_tensor_scale(const lmmc_tensor_t* tensor, lmmc_real_t alpha, lmmc_tensor_t* out_tensor);

/** @brief 累加全部元素。 */
lmmc_status_t lmmc_tensor_sum(const lmmc_tensor_t* tensor, lmmc_real_t* out_sum);
/** @brief 求最大元素。 */
lmmc_status_t lmmc_tensor_max(const lmmc_tensor_t* tensor, lmmc_real_t* out_max);
/** @brief 求最小元素。 */
lmmc_status_t lmmc_tensor_min(const lmmc_tensor_t* tensor, lmmc_real_t* out_min);

/**
 * @brief 沿指定轴求和，结果为二维矩阵。
 *
 * @param[in]  axis        归约轴，取值 0、1 或 2 。
 * @param[out] out_matrix  输出矩阵，形状由其余两维决定。
 */
lmmc_status_t lmmc_tensor_sum_axis(const lmmc_tensor_t* tensor, size_t axis, lmmc_mat_t* out_matrix);

/**
 * @brief 在元素总数与连续布局兼容时，构造重塑后的视图（不拷贝）。
 */
lmmc_status_t lmmc_tensor_reshape_view(
    const lmmc_tensor_t* tensor,
    size_t new_dim0,
    size_t new_dim1,
    size_t new_dim2,
    lmmc_tensor_t* out_view
);

/**
 * @brief 取张量子区间视图：每维使用半开区间 [begin, end) 。
 */
lmmc_status_t lmmc_tensor_slice_view(
    const lmmc_tensor_t* tensor,
    size_t begin0,
    size_t end0,
    size_t begin1,
    size_t end1,
    size_t begin2,
    size_t end2,
    lmmc_tensor_t* out_view
);

#ifdef __cplusplus
}
#endif

#endif
