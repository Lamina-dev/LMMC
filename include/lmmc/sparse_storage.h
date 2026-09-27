/** @file sparse_storage.h */
#ifndef LMMC_SPARSE_STORAGE_H
#define LMMC_SPARSE_STORAGE_H

#include "lmmc/sparse_types.h"
#include "lmmc/dense_types.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建稀疏矩阵增量构造器。
 *
 * 支持未知条目数和乱序插入，动态数组容量不足时翻倍扩容。
 * 用 ::lmmc_sparse_builder_build 生成压缩格式矩阵。
 *
 * @param[in]  rows             矩阵行数，须 > 0。
 * @param[in]  cols             矩阵列数，须 > 0。
 * @param[in]  initial_capacity 初始容量，0 时采用默认值 16。
 * @param[out] out_builder      输出构造器句柄。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - rows/cols 为 0 或 out_builder 为 NULL。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 * @note 分配堆内存，调用方须用 ::lmmc_sparse_builder_destroy 释放。
 */
lmmc_status_t lmmc_sparse_builder_create(size_t rows, size_t cols, size_t initial_capacity, lmmc_sparse_builder_t** out_builder);

/**
 * @brief 向构造器追加一项非零元。
 *
 * 重复坐标在 build 时累加合并；nnz 递增，容量不足时通过 realloc 翻倍扩容。
 *
 * @param[in,out] builder 构造器句柄。
 * @param[in]     row     行下标，须 < rows。
 * @param[in]     col     列下标，须 < cols。
 * @param[in]     value   非零元数值。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - builder 为 NULL 或下标越界。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 扩容时内存分配失败。
 */
lmmc_status_t lmmc_sparse_builder_add(lmmc_sparse_builder_t* builder, size_t row, size_t col, lmmc_real_t value);

/**
 * @brief 将构造器条目转换为指定压缩格式的稀疏矩阵。
 *
 * 按坐标排序，重复 (row,col) 按插入顺序累加，保留相消后的显式零。
 * 每段索引严格递增；构造器保持不变，可继续追加或再次 build。
 *
 * @param[in]  builder    构造器句柄。
 * @param[in]  format     目标格式，CSR 或 CSC。
 * @param[out] out_sparse 输出稀疏矩阵，调用方须用 ::lmmc_sparse_destroy 释放。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 参数为 NULL 或 format 非 CSR/CSC。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 * @note 为输出数组分配堆内存。
 */
lmmc_status_t lmmc_sparse_builder_build(lmmc_sparse_builder_t* builder, lmmc_sparse_format_t format, lmmc_sparse_mat_t* out_sparse);

/**
 * @brief 销毁稀疏构造器，释放动态数组与句柄。
 * @param[in] builder 构造器句柄，接受 NULL，销毁后失效。
 */
void lmmc_sparse_builder_destroy(lmmc_sparse_builder_t* builder);

/**
 * @brief 为指定 nnz 的 CSR 矩阵分配数组。
 *
 * owns_data=1，row_ptr 清零，col_idx/values 未初始化。
 * 调用方须填充三个数组，满足 sparse_types.h 中索引严格有序、
 * 坐标唯一的压缩结构契约后使用。
 *
 * @param[in]  rows       行数，须 > 0。
 * @param[in]  cols       列数，须 > 0。
 * @param[in]  nnz        非零元个数，可为 0。
 * @param[out] out_sparse 输出矩阵，调用方须用 ::lmmc_sparse_destroy 释放。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - rows/cols 为 0 或 out_sparse 为 NULL。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 */
lmmc_status_t lmmc_sparse_create_csr(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse);
/**
 * @brief 为指定 nnz 的 CSC 矩阵分配数组。
 *
 * 初始化、所有权及使用前提同 ::lmmc_sparse_create_csr，format 为 CSC。
 * row_ptr 表示 col_ptr，col_idx 表示 row_idx。
 *
 * @param[in]  rows       行数，须 > 0。
 * @param[in]  cols       列数，须 > 0。
 * @param[in]  nnz        非零元个数，可为 0。
 * @param[out] out_sparse 输出矩阵。
 * @return 同 ::lmmc_sparse_create_csr。
 */
lmmc_status_t lmmc_sparse_create_csc(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse);

/**
 * @brief 将调用方缓冲区包装为 CSR 视图。
 *
 * 校验行指针边界、单调性及每行索引范围和严格递增性，拒绝重复或乱序。
 * 输入及其顺序保持不变，失败时 out_sparse 保持不变。
 * 所有权保留给调用方，::lmmc_sparse_destroy 仅销毁视图。
 * 乱序或重复输入应使用 builder/COO。
 *
 * @param[in]  rows       行数。
 * @param[in]  cols       列数。
 * @param[in]  nnz        非零元个数。
 * @param[in]  row_ptr    行指针数组，长度 rows+1，row_ptr[0]==0，row_ptr[rows]==nnz。
 * @param[in]  col_idx    列索引数组，长度 nnz，每行严格递增且各值 < cols。
 * @param[in]  values     数值数组，长度 nnz。
 * @param[out] out_sparse 输出矩阵视图。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 参数非法或结构校验失败。
 * @note 无内存分配，写入视图会直接修改外部缓冲区。
 */
lmmc_status_t lmmc_sparse_wrap_csr(
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t* row_ptr,
    size_t* col_idx,
    lmmc_real_t* values,
    lmmc_sparse_mat_t* out_sparse
);

/**
 * @brief 将调用方缓冲区包装为 CSC 视图。
 *
 * 校验、所有权及写入语义同 ::lmmc_sparse_wrap_csr，format 为 CSC。
 * col_ptr 对应 row_ptr 字段，row_idx 对应 col_idx 字段。
 *
 * @param[in]  rows       行数。
 * @param[in]  cols       列数。
 * @param[in]  nnz        非零元个数。
 * @param[in]  col_ptr    列指针数组，长度 cols+1。
 * @param[in]  row_idx    行索引数组，长度 nnz，每列严格递增且各值 < rows。
 * @param[in]  values     数值数组，长度 nnz。
 * @param[out] out_sparse 输出矩阵视图。
 * @return 同 ::lmmc_sparse_wrap_csr。
 * @note 直接使用外部缓冲区，无内存分配。
 */
lmmc_status_t lmmc_sparse_wrap_csc(
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t* col_ptr,
    size_t* row_idx,
    lmmc_real_t* values,
    lmmc_sparse_mat_t* out_sparse
);

/**
 * @brief 销毁稀疏矩阵，释放其拥有的缓冲区并清零全部字段。
 *
 * owns_data 非零时释放 row_ptr、col_idx、values；外部缓冲区保留。
 * 销毁后矩阵须重新初始化才能用于运算。
 *
 * @param[in,out] sparse 待销毁矩阵，接受 NULL。
 */
void lmmc_sparse_destroy(lmmc_sparse_mat_t* sparse);

/**
 * @brief 由稠密矩阵生成 CSR，丢弃绝对值 <= eps 的元素。
 *
 * @param[in]  dense      输入稠密矩阵，保持不变。
 * @param[in]  eps        非负丢弃阈值。
 * @param[out] out_sparse 输出 CSR，调用方须用 ::lmmc_sparse_destroy 释放。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或 eps < 0。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 * @note 两遍扫描分别计数 nnz 和填充新分配的数组。
 */
lmmc_status_t lmmc_sparse_from_dense(const lmmc_mat_t* dense, lmmc_real_t eps, lmmc_sparse_mat_t* out_sparse);

/**
 * @brief 将稀疏矩阵展开为稠密矩阵，缺失项填零。
 *
 * @param[in]  sparse    输入稀疏矩阵，保持不变。
 * @param[out] out_dense 已创建且与 sparse 同维的稠密矩阵，内容被完全覆写。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或结构非法。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不匹配。
 * @note 使用输出缓冲区，无内存分配。
 */
lmmc_status_t lmmc_sparse_to_dense(const lmmc_sparse_mat_t* sparse, lmmc_mat_t* out_dense);

/**
 * @brief 计算稀疏矩阵转置，保持原存储格式。
 *
 * @param[in]  sparse         输入稀疏矩阵，保持不变。
 * @param[out] out_transposed 输出转置矩阵，调用方须用 ::lmmc_sparse_destroy 释放。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或结构非法。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 * @note 为输出数组分配堆内存。
 */
lmmc_status_t lmmc_sparse_transpose(const lmmc_sparse_mat_t* sparse, lmmc_sparse_mat_t* out_transposed);

/**
 * @brief 转换为 CSC，src 已为 CSC 时执行深拷贝。
 *
 * @param[in]  src 输入稀疏矩阵，保持不变。
 * @param[out] dst 输出 CSC，调用方须用 ::lmmc_sparse_destroy 释放。
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 * @note 为输出分配堆内存。
 */
lmmc_status_t lmmc_sparse_to_csc(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst);
/**
 * @brief 转换为 CSR，src 已为 CSR 时执行深拷贝。
 *
 * @param[in]  src 输入稀疏矩阵，保持不变。
 * @param[out] dst 输出 CSR，调用方须用 ::lmmc_sparse_destroy 释放。
 * @return 同 ::lmmc_sparse_to_csc。
 * @note 为输出分配堆内存。
 */
lmmc_status_t lmmc_sparse_to_csr(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst);

/** @brief 创建容量为 @p capacity 的空 COO 矩阵。 */
lmmc_status_t lmmc_sparse_coo_create(
    size_t rows, size_t cols, size_t capacity,
    lmmc_sparse_coo_t* out_coo
);

/** @brief 向 COO 追加一项非零元，达到容量时返回错误。 */
lmmc_status_t lmmc_sparse_coo_add_entry(
    lmmc_sparse_coo_t* coo,
    size_t row, size_t col, lmmc_real_t value
);

/**
 * @brief 将 COO 排序为 CSR，重复坐标按插入顺序累加，保留显式零。
 * 要求非零维度、nnz<=capacity、有效条目数组和范围内坐标。
 * 输入保持不变；非法描述符返回 INVALID_ARGUMENT，输出保持不变。
 */
lmmc_status_t lmmc_sparse_coo_to_csr(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csr
);

/** @brief 将 COO 转换为 CSC，校验、稳定合并与显式零语义同 COO→CSR。 */
lmmc_status_t lmmc_sparse_coo_to_csc(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csc
);

/** @brief 销毁 COO 矩阵并释放缓冲区。 */
void lmmc_sparse_coo_destroy(lmmc_sparse_coo_t* coo);

/**
 * @brief 创建 BSR 矩阵，数组内容未初始化。
 *
 * @param[in]  rows       块行数。
 * @param[in]  cols       块列数。
 * @param[in]  block_size 块维度 r，每块为 rxr。
 * @param[in]  nnz_blocks 非零块数量。
 * @param[out] out        输出 BSR 矩阵。
 */
lmmc_status_t lmmc_sparse_bsr_create(size_t rows, size_t cols, size_t block_size,
    size_t nnz_blocks, lmmc_sparse_bsr_t* out);

/**
 * @brief 将 BSR 矩阵展开为稠密矩阵。
 *
 * @param[in]  bsr 输入 BSR 矩阵。
 * @param[out] out 已创建的稠密矩阵，维度须为 (rows*block_size) x (cols*block_size)。
 */
lmmc_status_t lmmc_sparse_bsr_to_dense(const lmmc_sparse_bsr_t* bsr, lmmc_mat_t* out);

/**
 * @brief 将稠密矩阵转换为 BSR 格式。
 *
 * 丢弃块内全部元素绝对值均 <= eps 的块。
 *
 * @param[in]  dense      输入稠密矩阵，各维度须为 block_size 的整数倍。
 * @param[in]  block_size 块维度 r。
 * @param[in]  eps        丢弃阈值。
 * @param[out] out        输出 BSR 矩阵。
 */
lmmc_status_t lmmc_sparse_dense_to_bsr(const lmmc_mat_t* dense, size_t block_size,
    lmmc_real_t eps, lmmc_sparse_bsr_t* out);

/** @brief 销毁 BSR 矩阵，按所有权释放底层缓冲区。 */
void lmmc_sparse_bsr_destroy(lmmc_sparse_bsr_t* bsr);

/**
 * @brief 从完整 CSR 矩阵提取对称半存储。
 *
 * @param[in]  full 输入完整 CSR 方阵。
 * @param[in]  half 选择上三角或下三角。
 * @param[out] out  输出对称半存储 CSR 矩阵。
 */
lmmc_status_t lmmc_sparse_sym_csr_from_csr(const lmmc_sparse_mat_t* full,
    lmmc_sparse_sym_half_t half, lmmc_sparse_sym_csr_t* out);

/** @brief 销毁对称半存储 CSR 矩阵，按所有权释放底层缓冲区。 */
void lmmc_sparse_sym_csr_destroy(lmmc_sparse_sym_csr_t* s);

#ifdef __cplusplus
}
#endif

#endif
