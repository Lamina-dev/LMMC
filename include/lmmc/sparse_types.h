/** @file sparse_types.h */
#ifndef LMMC_SPARSE_TYPES_H
#define LMMC_SPARSE_TYPES_H

#include <stddef.h>
#include "lmmc/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 稀疏矩阵存储格式。 */
typedef enum {
    LMMC_SPARSE_CSR = 0, /**< 压缩行存储 (Compressed Sparse Row)。 */
    LMMC_SPARSE_CSC = 1  /**< 压缩列存储 (Compressed Sparse Column)。 */
} lmmc_sparse_format_t;

/**
 * @brief 通用稀疏矩阵(CSR / CSC 共用同一结构)。
 *
 * 当 @c format = ::LMMC_SPARSE_CSR :
 *  - @c row_ptr 长度为 @c rows+1 ,
 *  - @c col_idx 与 @c values 长度均为 @c nnz 。
 *
 * 当 @c format = ::LMMC_SPARSE_CSC :
 *  - @c row_ptr 在此别名为 col_ptr,长度为 @c cols+1 ,
 *  - @c col_idx 在此别名为 row_idx,长度为 @c nnz 。
 *
 * 有效矩阵的 format 只能为 CSR 或 CSC, rows/cols 必须非零。
 * 外指针从 0 单调递增到 nnz, 各段内索引必须在范围内且严格递增:
 * 同一坐标仅存一项. 显式零保留, nnz 表示存储条目数而非数学非零数。
 * 乱序或重复输入须通过 builder/COO 转换规范化。
 */
typedef struct {
    size_t rows;                       /**< 行数。 */
    size_t cols;                       /**< 列数。 */
    size_t nnz;                        /**< 存储条目数,包括显式零。 */
    size_t* row_ptr;                   /**< CSR 行指针 / CSC 列指针数组。 */
    size_t* col_idx;                   /**< CSR 列索引 / CSC 行索引数组。 */
    lmmc_real_t* values;               /**< 存储条目的数值数组，长度 @c nnz 。 */
    lmmc_sparse_format_t format;       /**< 存储格式。 */
    int owns_data;                     /**< 是否拥有底层缓冲区。 */
} lmmc_sparse_mat_t;

/**
 * @brief 增量构造稀疏矩阵的辅助对象(不透明类型)。
 *
 * 适合 nnz 未知或乱序插入的场景.最终调用 ::lmmc_sparse_builder_build
 * 得到 ::lmmc_sparse_mat_t 。
 */
typedef struct lmmc_sparse_builder_t lmmc_sparse_builder_t;

/**
 * @brief 稀疏 LU 分解上下文(不透明类型)。
 *
 * 符号阶段在 A+A^T 的去重对称简单图上执行确定性的贪心残余度排序。
 * 排序仅删除已消元顶点的关联边，不建模消元填充（区别于 AMD）。
 * 分解按符号分析、数值分解与求解三个阶段执行。
 */
typedef struct lmmc_sparse_lu_t lmmc_sparse_lu_t;

/**
 * @brief 稀疏 Cholesky 分解上下文(不透明类型,要求 @c A 对称正定)。
 *
 * 符号阶段在去重对称简单图上执行确定性的贪心残余度排序；
 * 数值阶段使用稠密列工作区的左看 Cholesky。排序不建模填充（区别于 AMD）。
 * 分解按符号分析、数值分解与求解三个阶段执行。
 */
typedef struct lmmc_sparse_chol_t lmmc_sparse_chol_t;

/**
 * @brief 三元组 (COO) 形式的稀疏矩阵临时表示。
 *
 * 拥有自身缓冲区,主要用于增量插入后转换为 CSR / CSC 。
 */
typedef struct {
    size_t rows;                /**< 行数。 */
    size_t cols;                /**< 列数。 */
    size_t nnz;                 /**< 当前已插入的条目数。 */
    size_t capacity;            /**< 当前容量。 */
    size_t* row_idx;            /**< 行下标数组,长度 @c capacity 。 */
    size_t* col_idx;            /**< 列下标数组,长度 @c capacity 。 */
    lmmc_real_t* values;        /**< 数值数组,长度 @c capacity 。 */
} lmmc_sparse_coo_t;

/**
 * @brief 块稀疏行 (BSR) 格式矩阵。
 *
 * 每个非零块为 @c block_size x @c block_size 的稠密子矩阵,
 * 按行优先存储在 @c values 中。
 */
typedef struct {
    size_t rows;          /**< 块行数。 */
    size_t cols;          /**< 块列数。 */
    size_t block_size;    /**< rxr 块维度。 */
    size_t nnz_blocks;    /**< 非零块数量。 */
    size_t* row_ptr;      /**< 块行指针数组,长度 rows+1 。 */
    size_t* col_idx;      /**< 块列索引数组,长度 nnz_blocks 。 */
    lmmc_real_t* values;  /**< 块数值数组,长度 nnz_blocks * block_size^2 。 */
    int owns_data;        /**< 是否拥有底层缓冲区。 */
} lmmc_sparse_bsr_t;

/** @brief 对称半存储选择:上三角或下三角。 */
typedef enum {
    LMMC_SPARSE_SYM_UPPER = 0, /**< 存储上三角(含对角线)。 */
    LMMC_SPARSE_SYM_LOWER = 1  /**< 存储下三角(含对角线)。 */
} lmmc_sparse_sym_half_t;

/**
 * @brief 对称半存储 CSR 格式矩阵。
 *
 * 仅存储上三角或下三角(含对角线),用于对称矩阵的紧凑表示。
 */
typedef struct {
    size_t n;                        /**< 矩阵阶数(方阵)。 */
    size_t nnz;                      /**< 存储的非零元个数(仅半三角)。 */
    size_t* row_ptr;                 /**< 行指针数组,长度 n+1 。 */
    size_t* col_idx;                 /**< 列索引数组,长度 nnz 。 */
    lmmc_real_t* values;             /**< 数值数组,长度 nnz 。 */
    lmmc_sparse_sym_half_t half;     /**< 存储的半三角类型。 */
    int owns_data;                   /**< 是否拥有底层缓冲区。 */
} lmmc_sparse_sym_csr_t;

#ifdef __cplusplus
}
#endif

#endif
