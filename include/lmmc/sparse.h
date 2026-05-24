/**
 * @file sparse.h
 * @brief 稀疏矩阵数据结构、构造器与基本运算。
 *
 * 提供 CSR / CSC 两种压缩存储格式以及临时使用的 COO 三元组格式，
 * 涵盖建造器（builder）、SpMV、SpGEMM、转置、格式互转、稀疏 LU /
 * Cholesky 分解等接口。
 */
#ifndef LMMC_SPARSE_H
#define LMMC_SPARSE_H

#include <stddef.h>
#include "lmmc/dense.h"
#include "lmmc/status.h"
#include "lmmc/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 稀疏矩阵存储格式。
 */
typedef enum {
    LMMC_SPARSE_CSR = 0, /**< 压缩行存储 (Compressed Sparse Row)。 */
    LMMC_SPARSE_CSC = 1  /**< 压缩列存储 (Compressed Sparse Column)。 */
} lmmc_sparse_format_t;

/**
 * @brief 通用稀疏矩阵（CSR / CSC 共用同一结构）。
 *
 * 当 @c format = ::LMMC_SPARSE_CSR ：
 *  - @c row_ptr 长度为 @c rows+1 ，
 *  - @c col_idx 与 @c values 长度均为 @c nnz 。
 *
 * 当 @c format = ::LMMC_SPARSE_CSC ：
 *  - @c row_ptr 在此别名为 col_ptr，长度为 @c cols+1 ，
 *  - @c col_idx 在此别名为 row_idx，长度为 @c nnz 。
 */
typedef struct {
    size_t rows;                       /**< 行数。 */
    size_t cols;                       /**< 列数。 */
    size_t nnz;                        /**< 非零元个数。 */
    size_t* row_ptr;                   /**< CSR 行指针 / CSC 列指针数组。 */
    size_t* col_idx;                   /**< CSR 列索引 / CSC 行索引数组。 */
    lmmc_real_t* values;               /**< 非零元数值数组，长度 @c nnz 。 */
    lmmc_sparse_format_t format;       /**< 存储格式。 */
    int owns_data;                     /**< 是否拥有底层缓冲区。 */
} lmmc_sparse_mat_t;

/**
 * @brief 增量构造稀疏矩阵的辅助对象（不透明类型）。
 *
 * 适合 nnz 未知或乱序插入的场景。最终调用 ::lmmc_sparse_builder_build
 * 得到 ::lmmc_sparse_mat_t 。
 */
typedef struct lmmc_sparse_builder_t lmmc_sparse_builder_t;

/**
 * @brief 创建稀疏矩阵构造器。
 *
 * @param[in]  rows             行数。
 * @param[in]  cols             列数。
 * @param[in]  initial_capacity 初始预分配容量（条目数），可为 0 。
 * @param[out] out_builder      返回的构造器句柄。
 */
lmmc_status_t lmmc_sparse_builder_create(size_t rows, size_t cols, size_t initial_capacity, lmmc_sparse_builder_t** out_builder);

/**
 * @brief 向构造器追加一项非零元。重复 (row,col) 会被累加。
 */
lmmc_status_t lmmc_sparse_builder_add(lmmc_sparse_builder_t* builder, size_t row, size_t col, lmmc_real_t value);

/**
 * @brief 将构造器转换为指定格式的稀疏矩阵，构造器仍可继续使用或销毁。
 */
lmmc_status_t lmmc_sparse_builder_build(lmmc_sparse_builder_t* builder, lmmc_sparse_format_t format, lmmc_sparse_mat_t* out_sparse);

/** @brief 销毁稀疏矩阵构造器。 */
void lmmc_sparse_builder_destroy(lmmc_sparse_builder_t* builder);

/** @brief 创建 @p nnz 个非零的空 CSR 矩阵（数组内容未初始化）。 */
lmmc_status_t lmmc_sparse_create_csr(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse);
/** @brief 创建 @p nnz 个非零的空 CSC 矩阵（数组内容未初始化）。 */
lmmc_status_t lmmc_sparse_create_csc(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse);

/**
 * @brief 用外部缓冲区包装为 CSR 视图（不拥有内存）。
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
 * @brief 用外部缓冲区包装为 CSC 视图（不拥有内存）。
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

/** @brief 销毁稀疏矩阵，必要时释放底层缓冲区。 */
void lmmc_sparse_destroy(lmmc_sparse_mat_t* sparse);

/**
 * @brief 由稠密矩阵生成稀疏矩阵（CSR），绝对值不超过 @p eps 的元素被丢弃。
 */
lmmc_status_t lmmc_sparse_from_dense(const lmmc_mat_t* dense, lmmc_real_t eps, lmmc_sparse_mat_t* out_sparse);

/** @brief 将稀疏矩阵展开为稠密矩阵（缺失项视为 0）。 */
lmmc_status_t lmmc_sparse_to_dense(const lmmc_sparse_mat_t* sparse, lmmc_mat_t* out_dense);

/** @brief 稀疏矩阵-稠密向量乘法 @c y = A x 。 */
lmmc_status_t lmmc_sparse_mat_vec_mul(const lmmc_sparse_mat_t* sparse, const lmmc_vec_t* x, lmmc_vec_t* y);

/** @brief 稀疏矩阵转置（保持原格式）。 */
lmmc_status_t lmmc_sparse_transpose(const lmmc_sparse_mat_t* sparse, lmmc_sparse_mat_t* out_transposed);

/** @brief 稀疏矩阵 × 稠密矩阵 @c C = A B （结果稠密）。 */
lmmc_status_t lmmc_sparse_mat_mat_mul_dense(const lmmc_sparse_mat_t* sparse, const lmmc_mat_t* b, lmmc_mat_t* c);

/** @brief 稀疏矩阵-稀疏矩阵乘法（SpGEMM）：@c C = A B 。 */
lmmc_status_t lmmc_sparse_mat_mat_mul_sparse(const lmmc_sparse_mat_t* a, const lmmc_sparse_mat_t* b, lmmc_sparse_mat_t* c);

/** @brief 将矩阵格式转换为 CSC 。若 @p src 已是 CSC 则深拷贝。 */
lmmc_status_t lmmc_sparse_to_csc(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst);
/** @brief 将矩阵格式转换为 CSR 。若 @p src 已是 CSR 则深拷贝。 */
lmmc_status_t lmmc_sparse_to_csr(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst);

/* ===================== 稀疏 LU 分解 ===================== */

/**
 * @brief 稀疏 LU 分解上下文（不透明类型）。
 *
 * The @c col_perm field is populated by AMD (Approximate Minimum Degree)
 * reordering during the symbolic phase to reduce fill-in. The factorization
 * uses a three-stage pipeline: analyze (symbolic) -> factorize (numeric) -> solve.
 */
typedef struct lmmc_sparse_lu_t lmmc_sparse_lu_t;

/**
 * @brief 进行稀疏 LU 的符号分析，确定填充模式。
 */
lmmc_status_t lmmc_sparse_lu_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t** out_lu
);

/**
 * @brief 进行稀疏 LU 的数值分解（需先完成符号分析）。
 */
lmmc_status_t lmmc_sparse_lu_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t* lu
);

/**
 * @brief 利用已分解的 LU 因子求解 @c A x = b 。
 */
lmmc_status_t lmmc_sparse_lu_solve(
    const lmmc_sparse_lu_t* lu,
    const lmmc_vec_t* b,
    lmmc_vec_t* x
);

/** @brief 销毁稀疏 LU 上下文。 */
void lmmc_sparse_lu_destroy(lmmc_sparse_lu_t* lu);

/* ===================== 稀疏 Cholesky 分解 ===================== */

/**
 * @brief 稀疏 Cholesky 分解上下文（不透明类型，要求 @c A 对称正定）。
 *
 * The @c perm field is populated by AMD (Approximate Minimum Degree)
 * reordering during the symbolic phase to reduce fill-in. The factorization
 * uses a three-stage pipeline: analyze (symbolic) -> factorize (numeric) -> solve.
 */
typedef struct lmmc_sparse_chol_t lmmc_sparse_chol_t;

/** @brief 稀疏 Cholesky 符号分析。 */
lmmc_status_t lmmc_sparse_chol_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t** out_chol
);

/**
 * @brief 稀疏 Cholesky 数值分解。
 *
 * @return 若 @p a 非正定返回 ::LMMC_STATUS_NOT_POSITIVE_DEFINITE 。
 */
lmmc_status_t lmmc_sparse_chol_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t* chol
);

/** @brief 利用已分解的 Cholesky 因子求解 @c A x = b 。 */
lmmc_status_t lmmc_sparse_chol_solve(
    const lmmc_sparse_chol_t* chol,
    const lmmc_vec_t* b,
    lmmc_vec_t* x
);

/** @brief 销毁稀疏 Cholesky 上下文。 */
void lmmc_sparse_chol_destroy(lmmc_sparse_chol_t* chol);

/* ===================== COO 三元组接口 ===================== */

/**
 * @brief 三元组 (COO) 形式的稀疏矩阵临时表示。
 *
 * 拥有自身缓冲区，主要用于增量插入后转换为 CSR / CSC 。
 */
typedef struct {
    size_t rows;                /**< 行数。 */
    size_t cols;                /**< 列数。 */
    size_t nnz;                 /**< 当前已插入的条目数。 */
    size_t capacity;            /**< 当前容量。 */
    size_t* row_idx;            /**< 行下标数组，长度 @c capacity 。 */
    size_t* col_idx;            /**< 列下标数组，长度 @c capacity 。 */
    lmmc_real_t* values;        /**< 数值数组，长度 @c capacity 。 */
} lmmc_sparse_coo_t;

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

/** @brief 将 COO 转换为 CSR 矩阵（重复条目按相加合并）。 */
lmmc_status_t lmmc_sparse_coo_to_csr(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csr
);

/** @brief 将 COO 转换为 CSC 矩阵（重复条目按相加合并）。 */
lmmc_status_t lmmc_sparse_coo_to_csc(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csc
);

/** @brief 销毁 COO 矩阵并释放其缓冲区。 */
void lmmc_sparse_coo_destroy(lmmc_sparse_coo_t* coo);

/* ===================== 通用稀疏运算 ===================== */

/** @brief 计算 @c C = alpha * A + beta * B 。要求 @p a 与 @p b 同维同格式。 */
lmmc_status_t lmmc_sparse_add(
    lmmc_real_t alpha, const lmmc_sparse_mat_t* a,
    lmmc_real_t beta, const lmmc_sparse_mat_t* b,
    lmmc_sparse_mat_t* out_c
);

/** @brief 就地标量缩放：@f$A \leftarrow \alpha A@f$ 。 */
lmmc_status_t lmmc_sparse_scale(
    lmmc_sparse_mat_t* a,
    lmmc_real_t alpha
);

/** @brief 计算稀疏矩阵的 Frobenius 范数。 */
lmmc_status_t lmmc_sparse_norm_fro(
    const lmmc_sparse_mat_t* a,
    lmmc_real_t* out_norm
);

/** @brief 提取主对角线为稠密向量。要求 @c a->rows == a->cols 。 */
lmmc_status_t lmmc_sparse_diag(
    const lmmc_sparse_mat_t* a,
    lmmc_vec_t* out_diag
);

/* ===================== BSR (Block Sparse Row) 格式 ===================== */

/**
 * @brief 块稀疏行 (BSR) 格式矩阵。
 *
 * 每个非零块为 @c block_size × @c block_size 的稠密子矩阵，
 * 按行优先存储在 @c values 中。
 */
typedef struct {
    size_t rows;          /**< 块行数。 */
    size_t cols;          /**< 块列数。 */
    size_t block_size;    /**< r×r 块维度。 */
    size_t nnz_blocks;    /**< 非零块数量。 */
    size_t* row_ptr;      /**< 块行指针数组，长度 rows+1 。 */
    size_t* col_idx;      /**< 块列索引数组，长度 nnz_blocks 。 */
    lmmc_real_t* values;  /**< 块数值数组，长度 nnz_blocks * block_size^2 。 */
    int owns_data;        /**< 是否拥有底层缓冲区。 */
} lmmc_sparse_bsr_t;

/**
 * @brief 创建 BSR 矩阵（数组内容未初始化）。
 *
 * @param[in]  rows       块行数。
 * @param[in]  cols       块列数。
 * @param[in]  block_size 块维度 r（每块为 r×r）。
 * @param[in]  nnz_blocks 非零块数量。
 * @param[out] out        输出 BSR 矩阵。
 */
lmmc_status_t lmmc_sparse_bsr_create(size_t rows, size_t cols, size_t block_size,
    size_t nnz_blocks, lmmc_sparse_bsr_t* out);

/**
 * @brief 将 BSR 矩阵展开为稠密矩阵。
 *
 * @param[in]  bsr 输入 BSR 矩阵。
 * @param[out] out 输出稠密矩阵，必须已创建且维度为 (rows*block_size) × (cols*block_size)。
 */
lmmc_status_t lmmc_sparse_bsr_to_dense(const lmmc_sparse_bsr_t* bsr, lmmc_mat_t* out);

/**
 * @brief 将稠密矩阵转换为 BSR 格式。
 *
 * 绝对值不超过 @p eps 的块（块内所有元素绝对值均 <= eps）被丢弃。
 *
 * @param[in]  dense      输入稠密矩阵，维度必须为 block_size 的整数倍。
 * @param[in]  block_size 块维度 r 。
 * @param[in]  eps        丢弃阈值。
 * @param[out] out        输出 BSR 矩阵。
 */
lmmc_status_t lmmc_sparse_dense_to_bsr(const lmmc_mat_t* dense, size_t block_size,
    lmmc_real_t eps, lmmc_sparse_bsr_t* out);

/** @brief 销毁 BSR 矩阵，必要时释放底层缓冲区。 */
void lmmc_sparse_bsr_destroy(lmmc_sparse_bsr_t* bsr);

/* ===================== 对称半存储 CSR 格式 ===================== */

/**
 * @brief 对称半存储选择：上三角或下三角。
 */
typedef enum {
    LMMC_SPARSE_SYM_UPPER = 0, /**< 存储上三角（含对角线）。 */
    LMMC_SPARSE_SYM_LOWER = 1  /**< 存储下三角（含对角线）。 */
} lmmc_sparse_sym_half_t;

/**
 * @brief 对称半存储 CSR 格式矩阵。
 *
 * 仅存储上三角或下三角（含对角线），用于对称矩阵的紧凑表示。
 */
typedef struct {
    size_t n;                        /**< 矩阵阶数（方阵）。 */
    size_t nnz;                      /**< 存储的非零元个数（仅半三角）。 */
    size_t* row_ptr;                 /**< 行指针数组，长度 n+1 。 */
    size_t* col_idx;                 /**< 列索引数组，长度 nnz 。 */
    lmmc_real_t* values;             /**< 数值数组，长度 nnz 。 */
    lmmc_sparse_sym_half_t half;     /**< 存储的半三角类型。 */
    int owns_data;                   /**< 是否拥有底层缓冲区。 */
} lmmc_sparse_sym_csr_t;

/**
 * @brief 从完整 CSR 矩阵提取对称半存储。
 *
 * @param[in]  full 输入完整 CSR 矩阵（必须为方阵）。
 * @param[in]  half 选择存储上三角或下三角。
 * @param[out] out  输出对称半存储 CSR 矩阵。
 */
lmmc_status_t lmmc_sparse_sym_csr_from_csr(const lmmc_sparse_mat_t* full,
    lmmc_sparse_sym_half_t half, lmmc_sparse_sym_csr_t* out);

/**
 * @brief 对称半存储 SpMV：@c y = A * x 。
 *
 * 利用对称性，仅存储半三角但计算完整矩阵-向量乘积。
 *
 * @param[in]  A 对称半存储 CSR 矩阵。
 * @param[in]  x 输入向量，长度为 A->n 。
 * @param[out] y 输出向量，长度为 A->n 。
 */
lmmc_status_t lmmc_sparse_sym_spmv(const lmmc_sparse_sym_csr_t* A,
    const lmmc_vec_t* x, lmmc_vec_t* y);

/** @brief 销毁对称半存储 CSR 矩阵，必要时释放底层缓冲区。 */
void lmmc_sparse_sym_csr_destroy(lmmc_sparse_sym_csr_t* s);

#ifdef __cplusplus
}
#endif

#endif
