/** @file sparse_arithmetic.h */
#ifndef LMMC_SPARSE_ARITHMETIC_H
#define LMMC_SPARSE_ARITHMETIC_H

#include "lmmc/sparse_types.h"
#include "lmmc/dense_types.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 稀疏矩阵-稠密向量乘法:y = A * x。
 *
 * 支持 CSR 和 CSC 格式.CSR 时使用 4-展开内循环优化。
 * x->size 须等于 sparse->cols,y->size 须等于 sparse->rows。
 *
 * @param[in]  sparse 输入稀疏矩阵(不被修改)。
 * @param[in]  x      输入向量(不被修改)。
 * @param[out] y      输出向量,内容被完全覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或结构非法。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 向量长度与矩阵维度不匹配。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_sparse_mat_vec_mul(const lmmc_sparse_mat_t* sparse, const lmmc_vec_t* x, lmmc_vec_t* y);

/**
 * @brief 稀疏矩阵 x 稠密矩阵:C = A * B(结果为稠密矩阵)。
 *
 * sparse 为 mxk,b 为 kxn,c 必须已创建为 mxn。
 * 支持 CSR 和 CSC 格式,使用 4-展开优化。
 *
 * @param[in]  sparse 输入稀疏矩阵(不被修改)。
 * @param[in]  b      输入稠密矩阵(不被修改)。
 * @param[out] c      输出稠密矩阵,内容被完全覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不匹配。
 *
 * @note 不分配内存。
 */
lmmc_status_t lmmc_sparse_mat_mat_mul_dense(const lmmc_sparse_mat_t* sparse, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 稀疏矩阵-稀疏矩阵乘法(SpGEMM):C = A * B。
 *
 * 输出为 CSR 格式.内部使用行累加器(marker + accumulator)两遍算法:
 * 第一遍计算 nnz 结构,第二遍填充数值。
 * 若输入为 CSC 格式会先内部转换为 CSR。
 *
 * @param[in]  a 左稀疏矩阵(不被修改)。
 * @param[in]  b 右稀疏矩阵(不被修改),a->cols 须等于 b->rows。
 * @param[out] c 输出稀疏矩阵(CSR),调用方需配对调用 ::lmmc_sparse_destroy 释放。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - a->cols != b->rows。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 *
 * @note 分配堆内存；CSC 输入的临时 CSR 副本在返回前释放。
 */
lmmc_status_t lmmc_sparse_mat_mat_mul_sparse(const lmmc_sparse_mat_t* a, const lmmc_sparse_mat_t* b, lmmc_sparse_mat_t* c);

/** @brief 计算 @c C = alpha * A + beta * B .要求 @p a 与 @p b 同维同格式。 */
lmmc_status_t lmmc_sparse_add(
    lmmc_real_t alpha, const lmmc_sparse_mat_t* a,
    lmmc_real_t beta, const lmmc_sparse_mat_t* b,
    lmmc_sparse_mat_t* out_c
);

/** @brief 就地标量缩放:@f$A \leftarrow \alpha A@f$ 。 */
lmmc_status_t lmmc_sparse_scale(
    lmmc_sparse_mat_t* a,
    lmmc_real_t alpha
);

/** @brief 用缩放平方和计算稀疏矩阵的 Frobenius 范数。 */
lmmc_status_t lmmc_sparse_norm_fro(
    const lmmc_sparse_mat_t* a,
    lmmc_real_t* out_norm
);

/** @brief 提取主对角线为稠密向量.要求 @c a->rows == a->cols 。 */
lmmc_status_t lmmc_sparse_diag(
    const lmmc_sparse_mat_t* a,
    lmmc_vec_t* out_diag
);

/**
 * @brief 对称半存储 SpMV:@c y = A * x 。
 *
 * 利用对称性,仅存储半三角但计算完整矩阵-向量乘积。
 *
 * @param[in]  A 对称半存储 CSR 矩阵。
 * @param[in]  x 输入向量,长度为 A->n 。
 * @param[out] y 输出向量,长度为 A->n 。
 */
lmmc_status_t lmmc_sparse_sym_spmv(const lmmc_sparse_sym_csr_t* A,
    const lmmc_vec_t* x, lmmc_vec_t* y);

#ifdef __cplusplus
}
#endif

#endif
