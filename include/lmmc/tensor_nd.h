/**
 * @file tensor_nd.h
 * @brief N-D 稠密张量数据结构、收缩与视图.
 */
#ifndef LMMC_TENSOR_ND_H
#define LMMC_TENSOR_ND_H

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
 * @param[in]  dims 各维大小数组,长度 @p ndim ,每个元素须 >= 1;可指向 out->dims.
 * @param[out] out  输出张量结构体(调用方提供存储).
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 ndim 超出范围,dims 含零,元素数或字节数溢出,或指针为 NULL;
 *         ::LMMC_STATUS_ALLOCATION_FAILED 若内存分配失败.
 *
 * @note 覆写 out 的全部字段；堆分配的数据缓冲区由调用方
 *       调用 ::lmmc_tensor_nd_destroy 释放。
 */
lmmc_status_t lmmc_tensor_nd_create(size_t ndim, const size_t* dims, lmmc_tensor_nd_t* out);

/** @brief 读取 N-D 张量元素. */
lmmc_status_t lmmc_tensor_nd_get(const lmmc_tensor_nd_t* t, const size_t* idx, lmmc_real_t* out);

/** @brief 写入 N-D 张量元素. */
lmmc_status_t lmmc_tensor_nd_set(lmmc_tensor_nd_t* t, const size_t* idx, lmmc_real_t value);

/**
 * @brief 按指定轴顺序置换张量，生成独立数据副本（转置的推广）。
 *
 * 例如 perm={1,0,2} 交换前两个轴；输出 owns_data == 1。
 *
 * @param[in]  in   输入张量.
 * @param[in]  perm 置换数组,长度 in->ndim,为 [0, ndim) 的排列.
 * @param[out] out  输出张量结构体(调用方提供存储).
 *
 * @return ::LMMC_STATUS_OK 表示成功；
 *         ::LMMC_STATUS_INVALID_ARGUMENT 表示 perm 无效或指针为 NULL；
 *         ::LMMC_STATUS_ALLOCATION_FAILED 表示内存分配失败。
 *
 * @note 输入张量保持原值；堆分配的输出缓冲区由调用方
 *       调用 ::lmmc_tensor_nd_destroy 释放。
 */
lmmc_status_t lmmc_tensor_nd_permute(const lmmc_tensor_nd_t* in, const size_t* perm, lmmc_tensor_nd_t* out);

/** @brief 张量收缩:沿匹配轴对求和. */
lmmc_status_t lmmc_tensor_nd_contract(const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    const size_t* axes_a, const size_t* axes_b, size_t naxes, lmmc_tensor_nd_t* out);

/** @brief 模-n 乘积:沿指定模与矩阵收缩. */
lmmc_status_t lmmc_tensor_nd_mode_n_product(const lmmc_tensor_nd_t* t, const lmmc_mat_t* mat,
    size_t mode, lmmc_tensor_nd_t* out);

/**
 * @brief 零拷贝重塑 N-D 张量，返回共享缓冲区的非拥有视图。
 *
 * 源张量须为行优先连续存储，新旧形状的元素总数须相同。
 * 输出 owns_data == 0，视图须在源张量销毁前使用。
 *
 * @param[in]  src      源张量(必须为连续存储).
 * @param[in]  new_ndim 新维度数,范围 [1, LMMC_TENSOR_MAX_NDIM].
 * @param[in]  new_dims 新各维大小数组,长度 @p new_ndim;可指向 out_view->dims.
 * @param[out] out_view 输出视图结构体;不可与拥有数据的 src 为同一对象.
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若维度无效,元素总数溢出或不匹配,源非连续,
 *         拥有数据的源与输出为同一对象,或指针为 NULL.
 *
 * @note 不分配堆内存；输出与源共享 data，双方数据修改相互可见。
 */
lmmc_status_t lmmc_tensor_nd_reshape_view(const lmmc_tensor_nd_t* src,
    size_t new_ndim, const size_t* new_dims, lmmc_tensor_nd_t* out_view);

/** @brief 销毁 N-D 张量(仅当 owns_data 为 1 时释放缓冲区). */
void lmmc_tensor_nd_destroy(lmmc_tensor_nd_t* t);

#ifdef __cplusplus
}
#endif

#endif
