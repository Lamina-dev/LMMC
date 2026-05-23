/**
 * @file dense.h
 * @brief 稠密矩阵 / 向量数据结构与基本运算（含 BLAS Level 1 / 2 接口）。
 *
 * 本模块提供按行优先存储的二维稠密矩阵 ::lmmc_mat_t 与一维稠密向量
 * ::lmmc_vec_t ，并涵盖创建、包装、销毁、复制、转置、矩阵乘法、
 * 范数、AXPY 等常用线性代数操作。
 */
#ifndef LMMC_DENSE_H
#define LMMC_DENSE_H

#include <stddef.h>
#include "lmmc/status.h"
#include "lmmc/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 行优先存储的稠密矩阵。
 *
 * 元素 @c (i,j) 位于 @c data[i * stride + j] ，其中 @c stride>=cols 。
 * 当 ::lmmc_mat_t::owns_data 非零时， ::lmmc_mat_destroy 会释放 @c data 。
 */
typedef struct {
    size_t rows;          /**< 行数。 */
    size_t cols;          /**< 列数。 */
    size_t stride;        /**< 行步距（每行实际占用的元素数，>= @c cols ）。 */
    lmmc_real_t* data;    /**< 数据缓冲区起始地址。 */
    int owns_data;        /**< 是否拥有缓冲区所有权（非 0 表示销毁时释放）。 */
} lmmc_mat_t;

/**
 * @brief 一维稠密向量。
 *
 * 当 ::lmmc_vec_t::owns_data 非零时， ::lmmc_vec_destroy 会释放 @c data 。
 */
typedef struct {
    size_t size;          /**< 元素个数。 */
    lmmc_real_t* data;    /**< 数据缓冲区起始地址。 */
    int owns_data;        /**< 是否拥有缓冲区所有权。 */
} lmmc_vec_t;

/**
 * @brief 创建一个 @p rows × @p cols 的稠密矩阵，元素未初始化。
 *
 * @param[in]  rows    行数，必须 > 0。
 * @param[in]  cols    列数，必须 > 0。
 * @param[out] out_mat 输出矩阵，调用方需通过 ::lmmc_mat_destroy 释放。
 * @return ::LMMC_STATUS_OK 成功；分配失败返回 ::LMMC_STATUS_ALLOCATION_FAILED 。
 */
lmmc_status_t lmmc_mat_create(size_t rows, size_t cols, lmmc_mat_t* out_mat);

/**
 * @brief 用外部缓冲区构造矩阵视图（不拥有内存）。
 *
 * @param[in]  rows    行数。
 * @param[in]  cols    列数。
 * @param[in]  stride  行步距，要求 >= @p cols 。
 * @param[in]  data    指向已分配缓冲区，至少 @p rows*@p stride 个元素。
 * @param[out] out_mat 输出矩阵视图， @c owns_data 置 0 。
 * @return ::LMMC_STATUS_OK 成功。
 */
lmmc_status_t lmmc_mat_wrap(size_t rows, size_t cols, size_t stride, lmmc_real_t* data, lmmc_mat_t* out_mat);

/**
 * @brief 销毁矩阵，必要时释放底层缓冲区。
 *
 * 对空指针或已销毁矩阵安全。销毁后将 @c data 置空、维度清零。
 */
void lmmc_mat_destroy(lmmc_mat_t* mat);

/**
 * @brief 用 @p value 填充矩阵的全部元素。
 *
 * @param[in,out] mat   待填充矩阵。
 * @param[in]     value 填充值。
 */
lmmc_status_t lmmc_mat_fill(lmmc_mat_t* mat, lmmc_real_t value);

/**
 * @brief 将矩阵 @p src 复制到 @p dst 。
 *
 * @p dst 必须已被创建且与 @p src 同维。
 */
lmmc_status_t lmmc_mat_copy(const lmmc_mat_t* src, lmmc_mat_t* dst);

/**
 * @brief 将 @p src 的转置写入 @p dst （ @p dst 必须为 @c src->cols × @c src->rows ）。
 */
lmmc_status_t lmmc_mat_transpose_to(const lmmc_mat_t* src, lmmc_mat_t* dst);

/**
 * @brief 计算矩阵乘法 @c c = a * b 。
 *
 * @p a 为 @c m×k ， @p b 为 @c k×n ， @p c 为 @c m×n 且应与 @p a, @p b 不别名。
 */
lmmc_status_t lmmc_mat_mul(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 计算矩阵的 Frobenius 范数 @f$\|A\|_F@f$ 。
 */
lmmc_status_t lmmc_mat_norm_fro(const lmmc_mat_t* a, lmmc_real_t* out_norm);

/**
 * @brief 创建长度为 @p size 的稠密向量，元素未初始化。
 */
lmmc_status_t lmmc_vec_create(size_t size, lmmc_vec_t* out_vec);

/**
 * @brief 用外部缓冲区构造向量视图（不拥有内存）。
 */
lmmc_status_t lmmc_vec_wrap(size_t size, lmmc_real_t* data, lmmc_vec_t* out_vec);

/**
 * @brief 销毁向量，必要时释放底层缓冲区。
 */
void lmmc_vec_destroy(lmmc_vec_t* vec);

/**
 * @brief 用 @p value 填充向量的全部元素。
 */
lmmc_status_t lmmc_vec_fill(lmmc_vec_t* vec, lmmc_real_t value);

/**
 * @brief 计算向量内积 @f$a \cdot b@f$ 。
 */
lmmc_status_t lmmc_vec_dot(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t* out_dot);

/**
 * @brief 计算 @c y = a * x （矩阵-向量乘法）。
 */
lmmc_status_t lmmc_mat_vec_mul(const lmmc_mat_t* a, const lmmc_vec_t* x, lmmc_vec_t* y);

/* ===================== BLAS Level 1 向量运算 ===================== */

/** @brief 计算向量的欧几里得范数 @f$\|x\|_2@f$ 。 */
lmmc_status_t lmmc_vec_norm2(const lmmc_vec_t* x, lmmc_real_t* out_norm);

/** @brief 计算向量的无穷范数 @f$\|x\|_\infty = \max_i |x_i|@f$ 。 */
lmmc_status_t lmmc_vec_norm_inf(const lmmc_vec_t* x, lmmc_real_t* out_norm);

/** @brief 就地缩放：@f$x \leftarrow \alpha x@f$ 。 */
lmmc_status_t lmmc_vec_scale(lmmc_vec_t* x, lmmc_real_t alpha);

/** @brief 计算 @f$y \leftarrow \alpha x + y@f$ 。 */
lmmc_status_t lmmc_vec_axpy(lmmc_real_t alpha, const lmmc_vec_t* x, lmmc_vec_t* y);

/** @brief 将 @p src 内容复制到 @p dst （要求长度相同）。 */
lmmc_status_t lmmc_vec_copy(const lmmc_vec_t* src, lmmc_vec_t* dst);

/** @brief 交换两向量内容。 */
lmmc_status_t lmmc_vec_swap(lmmc_vec_t* x, lmmc_vec_t* y);

/** @brief 计算元素绝对值之和 @f$\sum_i |x_i|@f$ 。 */
lmmc_status_t lmmc_vec_asum(const lmmc_vec_t* x, lmmc_real_t* out_asum);

/** @brief 返回绝对值最大元素的索引。 */
lmmc_status_t lmmc_vec_iamax(const lmmc_vec_t* x, size_t* out_idx);

/* ===================== 矩阵基本运算 ===================== */

/** @brief 同维矩阵逐元素加法：@c c = a + b 。 */
lmmc_status_t lmmc_mat_add(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/** @brief 同维矩阵逐元素减法：@c c = a - b 。 */
lmmc_status_t lmmc_mat_sub(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/** @brief 就地标量乘：@f$A \leftarrow \alpha A@f$ 。 */
lmmc_status_t lmmc_mat_scale(lmmc_mat_t* a, lmmc_real_t alpha);

/** @brief 生成 @p n × @p n 单位矩阵。 */
lmmc_status_t lmmc_mat_identity(size_t n, lmmc_mat_t* out_mat);

/** @brief 计算方阵的迹 @f$\mathrm{tr}(A)=\sum_i A_{ii}@f$ 。 */
lmmc_status_t lmmc_mat_trace(const lmmc_mat_t* a, lmmc_real_t* out_trace);

/**
 * @brief 计算方阵行列式（基于 LU 分解）。
 *
 * @return ::LMMC_STATUS_OK 成功；矩阵奇异时返回 ::LMMC_STATUS_SINGULAR_MATRIX 。
 */
lmmc_status_t lmmc_mat_det(const lmmc_mat_t* a, lmmc_real_t* out_det);

#ifdef __cplusplus
}
#endif

#endif
