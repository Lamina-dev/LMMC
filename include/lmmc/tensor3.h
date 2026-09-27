/**
 * @file tensor3.h
 * @brief 三阶稠密张量数据结构、运算与视图。
 */
#ifndef LMMC_TENSOR3_H
#define LMMC_TENSOR3_H

#include <stddef.h>
#include "lmmc/dense.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 三阶张量结构。
 *
 * 通过 ::lmmc_tensor3_reshape_view / ::lmmc_tensor3_slice_view 得到的
 * 视图共享底层缓冲区且 @c owns_data == 0。
 */
typedef struct {
    size_t dim0;          /**< 第 0 维大小。 */
    size_t dim1;          /**< 第 1 维大小。 */
    size_t dim2;          /**< 第 2 维大小。 */
    size_t stride0;       /**< 第 0 维步距。 */
    size_t stride1;       /**< 第 1 维步距。 */
    size_t stride2;       /**< 第 2 维步距。 */
    lmmc_real_t* data;    /**< 数据缓冲区。 */
    int owns_data;        /**< 缓冲区所有权标志。 */
} lmmc_tensor3_t;

/** @brief 创建 @p dim0 x @p dim1 x @p dim2 的张量，元素未初始化。 */
lmmc_status_t lmmc_tensor3_create(size_t dim0, size_t dim1, size_t dim2, lmmc_tensor3_t* out_tensor);

/** @brief 用外部缓冲区构造张量视图。 */
lmmc_status_t lmmc_tensor3_wrap(
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t stride0,
    size_t stride1,
    size_t stride2,
    lmmc_real_t* data,
    lmmc_tensor3_t* out_tensor
);

/** @brief 销毁张量。 */
void lmmc_tensor3_destroy(lmmc_tensor3_t* tensor);

/** @brief 用 @p value 填充张量所有元素。 */
lmmc_status_t lmmc_tensor3_fill(lmmc_tensor3_t* tensor, lmmc_real_t value);

/** @brief 写入元素 (i,j,k)。 */
lmmc_status_t lmmc_tensor3_set(lmmc_tensor3_t* tensor, size_t i, size_t j, size_t k, lmmc_real_t value);

/** @brief 读取元素 (i,j,k)。 */
lmmc_status_t lmmc_tensor3_get(const lmmc_tensor3_t* tensor, size_t i, size_t j, size_t k, lmmc_real_t* out_value);

/** @brief 用缩放平方和计算张量 Frobenius 范数。 */
lmmc_status_t lmmc_tensor3_norm_fro(const lmmc_tensor3_t* tensor, lmmc_real_t* out_norm);

/** @brief 同形状张量逐元素加法。 */
lmmc_status_t lmmc_tensor3_add(const lmmc_tensor3_t* a, const lmmc_tensor3_t* b, lmmc_tensor3_t* out_tensor);
/** @brief 同形状张量逐元素减法。 */
lmmc_status_t lmmc_tensor3_sub(const lmmc_tensor3_t* a, const lmmc_tensor3_t* b, lmmc_tensor3_t* out_tensor);
/** @brief 同形状张量逐元素乘法（Hadamard）。 */
lmmc_status_t lmmc_tensor3_mul(const lmmc_tensor3_t* a, const lmmc_tensor3_t* b, lmmc_tensor3_t* out_tensor);
/** @brief 同形状张量逐元素除法，@c b 中元素须非零。 */
lmmc_status_t lmmc_tensor3_div(const lmmc_tensor3_t* a, const lmmc_tensor3_t* b, lmmc_tensor3_t* out_tensor);
/** @brief 标量乘：@c out = alpha * tensor。 */
lmmc_status_t lmmc_tensor3_scale(const lmmc_tensor3_t* tensor, lmmc_real_t alpha, lmmc_tensor3_t* out_tensor);

/** @brief 累加全部元素。 */
lmmc_status_t lmmc_tensor3_sum(const lmmc_tensor3_t* tensor, lmmc_real_t* out_sum);
/** @brief 求最大元素。 */
lmmc_status_t lmmc_tensor3_max(const lmmc_tensor3_t* tensor, lmmc_real_t* out_max);
/** @brief 求最小元素。 */
lmmc_status_t lmmc_tensor3_min(const lmmc_tensor3_t* tensor, lmmc_real_t* out_min);

/**
 * @brief 沿指定轴求和，结果为二维矩阵。
 *
 * @param[in]  tensor     输入张量。
 * @param[in]  axis       归约轴，取值 0、1 或 2。
 * @param[out] out_matrix 输出矩阵，形状由其余两维决定。
 */
lmmc_status_t lmmc_tensor3_sum_axis(const lmmc_tensor3_t* tensor, size_t axis, lmmc_mat_t* out_matrix);

/**
 * @brief 在元素总数与连续布局兼容时，构造共享缓冲区的重塑视图。
 *
 * 非拥有型视图支持原地重塑；拥有数据的 tensor 须使用独立的 out_view。
 * 元素总数溢出返回 ::LMMC_STATUS_INVALID_ARGUMENT；
 * 可表示但不相等的元素总数返回 ::LMMC_STATUS_DIMENSION_MISMATCH。
 */
lmmc_status_t lmmc_tensor3_reshape_view(
    const lmmc_tensor3_t* tensor,
    size_t new_dim0,
    size_t new_dim1,
    size_t new_dim2,
    lmmc_tensor3_t* out_view
);

/** @brief 取张量子区间视图，每维使用半开区间 [begin, end)。 */
lmmc_status_t lmmc_tensor3_slice_view(
    const lmmc_tensor3_t* tensor,
    size_t begin0,
    size_t end0,
    size_t begin1,
    size_t end1,
    size_t begin2,
    size_t end2,
    lmmc_tensor3_t* out_view
);

#ifdef __cplusplus
}
#endif

#endif
