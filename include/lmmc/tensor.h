/**
 * @file tensor.h
 * @brief N-D 稠密张量数据结构与基本运算.
 *
 * 支持任意阶(1 至 LMMC_TENSOR_MAX_NDIM)的张量,
 * 以及向后兼容的三阶张量接口.
 */
#ifndef LMMC_TENSOR_H
#define LMMC_TENSOR_H

#include <stddef.h>
#include "lmmc/dense.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 支持的最大张量维度. */
#define LMMC_TENSOR_MAX_NDIM 8

/**
 * @brief N-D 张量结构.
 *
 * 支持 1 至 LMMC_TENSOR_MAX_NDIM 维的张量.
 * 通过 reshape_view 等得到的视图共享底层缓冲区且 @c owns_data == 0 .
 */
typedef struct {
    size_t ndim;                            /**< 维度数(1..LMMC_TENSOR_MAX_NDIM). */
    size_t dims[LMMC_TENSOR_MAX_NDIM];      /**< 各维大小(仅前 ndim 个有效). */
    size_t strides[LMMC_TENSOR_MAX_NDIM];   /**< 各维步距(仅前 ndim 个有效). */
    lmmc_real_t* data;                      /**< 数据缓冲区. */
    int owns_data;                          /**< 是否拥有缓冲区所有权. */
} lmmc_tensor_nd_t;

/**
 * @brief 创建 N-D 张量,行优先连续存储,元素初始化为零.
 *
 * 分配 dims[0] x dims[1] x ... x dims[ndim-1] 个元素的连续缓冲区,
 * 并设置行优先步距.输出张量的 owns_data 为 1.
 *
 * @param[in]  ndim 维度数,范围 [1, LMMC_TENSOR_MAX_NDIM].
 * @param[in]  dims 各维大小数组,长度 @p ndim ,每个元素须 >= 1.
 * @param[out] out  输出张量结构体(调用方提供存储).
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 ndim 超出范围,dims 含零或指针为 NULL;
 *         ::LMMC_STATUS_ALLOC_FAILED 若内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存用于数据缓冲区(out->data),调用方必须调用 ::lmmc_tensor_nd_destroy 释放.
 * - out 结构体的所有字段被覆盖写入.
 */
lmmc_status_t lmmc_tensor_create(size_t ndim, const size_t* dims, lmmc_tensor_nd_t* out);

/** @brief 读取 N-D 张量元素. */
lmmc_status_t lmmc_tensor_get_nd(const lmmc_tensor_nd_t* t, const size_t* idx, lmmc_real_t* out);

/** @brief 写入 N-D 张量元素. */
lmmc_status_t lmmc_tensor_set_nd(lmmc_tensor_nd_t* t, const size_t* idx, lmmc_real_t value);

/**
 * @brief 对张量维度进行置换(转置的推广),产生新的拥有数据的张量.
 *
 * 按照 perm 指定的轴顺序重新排列数据.例如 perm={1,0,2} 交换前两个轴.
 * 输出张量拥有独立的数据副本(owns_data == 1).
 *
 * @param[in]  in   输入张量.
 * @param[in]  perm 置换数组,长度 in->ndim,为 [0, ndim) 的排列.
 * @param[out] out  输出张量结构体(调用方提供存储).
 *
 * @return ::LMMC_STATUS_OK 表示成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 表示 perm 位于排列集合之外或指针为 NULL;
 *         ::LMMC_STATUS_ALLOC_FAILED 表示内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存用于输出数据缓冲区,调用方必须调用 ::lmmc_tensor_nd_destroy 释放.
 * - 不修改输入张量 @p in .
 */
lmmc_status_t lmmc_tensor_permute(const lmmc_tensor_nd_t* in, const size_t* perm, lmmc_tensor_nd_t* out);

/** @brief 张量收缩:沿匹配轴对求和. */
lmmc_status_t lmmc_tensor_contract(const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    const size_t* axes_a, const size_t* axes_b, size_t naxes, lmmc_tensor_nd_t* out);

/** @brief 模-n 乘积:沿指定模与矩阵收缩. */
lmmc_status_t lmmc_tensor_mode_n_product(const lmmc_tensor_nd_t* t, const lmmc_mat_t* mat,
    size_t mode, lmmc_tensor_nd_t* out);

/**
 * @brief N-D 张量重塑视图(零拷贝),返回共享底层缓冲区的非拥有视图.
 *
 * 仅当源张量为连续存储(行优先步距)时可成功.新形状的元素总数必须与源相同.
 * 输出视图的 owns_data == 0,生命周期不得超过源张量.
 *
 * @param[in]  src      源张量(必须为连续存储).
 * @param[in]  new_ndim 新维度数,范围 [1, LMMC_TENSOR_MAX_NDIM].
 * @param[in]  new_dims 新各维大小数组,长度 @p new_ndim .
 * @param[out] out_view 输出视图结构体.
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若元素总数不匹配,源非连续或指针为 NULL.
 *
 * @par 副作用
 * - 不分配堆内存.输出视图共享源张量的 data 指针.
 * - 修改源张量数据会同时影响视图,反之亦然(别名关系).
 * - 源张量销毁后视图将悬空,调用方须保证生命周期正确.
 */
lmmc_status_t lmmc_tensor_nd_reshape_view(const lmmc_tensor_nd_t* src,
    size_t new_ndim, const size_t* new_dims, lmmc_tensor_nd_t* out_view);

/** @brief 销毁 N-D 张量(仅当 owns_data 为 1 时释放缓冲区). */
void lmmc_tensor_nd_destroy(lmmc_tensor_nd_t* t);

/**
 * @brief 三阶张量结构(向后兼容).
 *
 * 通过 ::lmmc_tensor_reshape_view / ::lmmc_tensor_slice_view 得到的
 * 视图共享底层缓冲区且 @c owns_data == 0 .
 */
typedef struct {
    size_t dim0;          /**< 第 0 维大小. */
    size_t dim1;          /**< 第 1 维大小. */
    size_t dim2;          /**< 第 2 维大小. */
    size_t stride0;       /**< 第 0 维步距. */
    size_t stride1;       /**< 第 1 维步距. */
    size_t stride2;       /**< 第 2 维步距. */
    lmmc_real_t* data;    /**< 数据缓冲区. */
    int owns_data;        /**< 是否拥有缓冲区所有权. */
} lmmc_tensor_t;

/** @brief 创建 @p dim0 x @p dim1 x @p dim2 的张量,元素未初始化. */
lmmc_status_t lmmc_tensor3_create(size_t dim0, size_t dim1, size_t dim2, lmmc_tensor_t* out_tensor);

/** @brief 用外部缓冲区构造张量视图. */
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

/** @brief 销毁张量. */
void lmmc_tensor_destroy(lmmc_tensor_t* tensor);

/** @brief 用 @p value 填充张量所有元素. */
lmmc_status_t lmmc_tensor_fill(lmmc_tensor_t* tensor, lmmc_real_t value);

/** @brief 写入元素 (i,j,k) . */
lmmc_status_t lmmc_tensor_set(lmmc_tensor_t* tensor, size_t i, size_t j, size_t k, lmmc_real_t value);

/** @brief 读取元素 (i,j,k) . */
lmmc_status_t lmmc_tensor_get(const lmmc_tensor_t* tensor, size_t i, size_t j, size_t k, lmmc_real_t* out_value);

/** @brief 计算张量 Frobenius 范数. */
lmmc_status_t lmmc_tensor_norm_fro(const lmmc_tensor_t* tensor, lmmc_real_t* out_norm);

/** @brief 同形状张量逐元素加法. */
lmmc_status_t lmmc_tensor_add(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
/** @brief 同形状张量逐元素减法. */
lmmc_status_t lmmc_tensor_sub(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
/** @brief 同形状张量逐元素乘法(Hadamard). */
lmmc_status_t lmmc_tensor_mul(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
/** @brief 同形状张量逐元素除法(@c b 中元素须非零). */
lmmc_status_t lmmc_tensor_div(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
/** @brief 标量乘:@c out = alpha * tensor . */
lmmc_status_t lmmc_tensor_scale(const lmmc_tensor_t* tensor, lmmc_real_t alpha, lmmc_tensor_t* out_tensor);

/** @brief 累加全部元素. */
lmmc_status_t lmmc_tensor_sum(const lmmc_tensor_t* tensor, lmmc_real_t* out_sum);
/** @brief 求最大元素. */
lmmc_status_t lmmc_tensor_max(const lmmc_tensor_t* tensor, lmmc_real_t* out_max);
/** @brief 求最小元素. */
lmmc_status_t lmmc_tensor_min(const lmmc_tensor_t* tensor, lmmc_real_t* out_min);

/**
 * @brief 沿指定轴求和,结果为二维矩阵.
 *
 * @param[in]  tensor     输入张量.
 * @param[in]  axis       归约轴,取值 0,1 或 2 .
 * @param[out] out_matrix 输出矩阵,形状由其余两维决定.
 */
lmmc_status_t lmmc_tensor_sum_axis(const lmmc_tensor_t* tensor, size_t axis, lmmc_mat_t* out_matrix);

/**
 * @brief 在元素总数与连续布局兼容时,构造重塑后的视图(不拷贝).
 */
lmmc_status_t lmmc_tensor_reshape_view(
    const lmmc_tensor_t* tensor,
    size_t new_dim0,
    size_t new_dim1,
    size_t new_dim2,
    lmmc_tensor_t* out_view
);

/**
 * @brief 取张量子区间视图:每维使用半开区间 [begin, end) .
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
