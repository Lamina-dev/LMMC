#ifndef LMMC_DENSE_H
#define LMMC_DENSE_H

#include <stddef.h>
#include "lmmc/status.h"
#include "lmmc/config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    lmmc_real_t* data;
    int owns_data;
} lmmc_mat_t;

typedef struct {
    size_t size;
    lmmc_real_t* data;
    int owns_data;
} lmmc_vec_t;

lmmc_status_t lmmc_mat_create(size_t rows, size_t cols, lmmc_mat_t* out_mat);
lmmc_status_t lmmc_mat_wrap(size_t rows, size_t cols, size_t stride, lmmc_real_t* data, lmmc_mat_t* out_mat);
void lmmc_mat_destroy(lmmc_mat_t* mat);
lmmc_status_t lmmc_mat_fill(lmmc_mat_t* mat, lmmc_real_t value);
lmmc_status_t lmmc_mat_copy(const lmmc_mat_t* src, lmmc_mat_t* dst);
lmmc_status_t lmmc_mat_transpose_to(const lmmc_mat_t* src, lmmc_mat_t* dst);
lmmc_status_t lmmc_mat_mul(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);
lmmc_status_t lmmc_mat_norm_fro(const lmmc_mat_t* a, lmmc_real_t* out_norm);

lmmc_status_t lmmc_vec_create(size_t size, lmmc_vec_t* out_vec);
lmmc_status_t lmmc_vec_wrap(size_t size, lmmc_real_t* data, lmmc_vec_t* out_vec);
void lmmc_vec_destroy(lmmc_vec_t* vec);
lmmc_status_t lmmc_vec_fill(lmmc_vec_t* vec, lmmc_real_t value);
lmmc_status_t lmmc_vec_dot(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t* out_dot);
lmmc_status_t lmmc_mat_vec_mul(const lmmc_mat_t* a, const lmmc_vec_t* x, lmmc_vec_t* y);

/* === BLAS Level 1 向量运算 === */

/** 计算向量的欧几里得范数（L2 范数） */
lmmc_status_t lmmc_vec_norm2(const lmmc_vec_t* x, lmmc_real_t* out_norm);

/** 计算向量的无穷范数（最大绝对值） */
lmmc_status_t lmmc_vec_norm_inf(const lmmc_vec_t* x, lmmc_real_t* out_norm);

/** 将向量所有元素乘以标量（就地修改） */
lmmc_status_t lmmc_vec_scale(lmmc_vec_t* x, lmmc_real_t alpha);

/** 执行 y := alpha * x + y 线性组合运算 */
lmmc_status_t lmmc_vec_axpy(lmmc_real_t alpha, const lmmc_vec_t* x, lmmc_vec_t* y);

/** 将源向量内容复制到目标向量 */
lmmc_status_t lmmc_vec_copy(const lmmc_vec_t* src, lmmc_vec_t* dst);

/** 交换两个向量的内容 */
lmmc_status_t lmmc_vec_swap(lmmc_vec_t* x, lmmc_vec_t* y);

/** 计算向量元素绝对值之和 */
lmmc_status_t lmmc_vec_asum(const lmmc_vec_t* x, lmmc_real_t* out_asum);

/** 返回绝对值最大元素的索引 */
lmmc_status_t lmmc_vec_iamax(const lmmc_vec_t* x, size_t* out_idx);

/* === 矩阵基本运算 === */

/** 计算两个同维矩阵的逐元素加法：c = a + b */
lmmc_status_t lmmc_mat_add(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/** 计算两个同维矩阵的逐元素减法：c = a - b */
lmmc_status_t lmmc_mat_sub(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/** 将矩阵所有元素乘以标量（就地修改） */
lmmc_status_t lmmc_mat_scale(lmmc_mat_t* a, lmmc_real_t alpha);

/** 生成 n×n 单位矩阵 */
lmmc_status_t lmmc_mat_identity(size_t n, lmmc_mat_t* out_mat);

/** 计算方阵的迹（对角元素之和） */
lmmc_status_t lmmc_mat_trace(const lmmc_mat_t* a, lmmc_real_t* out_trace);

/** 计算方阵的行列式（内部使用 LU 分解，O(n³)） */
lmmc_status_t lmmc_mat_det(const lmmc_mat_t* a, lmmc_real_t* out_det);

#ifdef __cplusplus
}
#endif

#endif
