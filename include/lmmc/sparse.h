/**
 * @file sparse.h
 * @brief 稀疏矩阵数据结构,构造器与基本运算.
 *
 * 提供 CSR / CSC 两种压缩存储格式以及临时使用的 COO 三元组格式,
 * 涵盖建造器(builder),SpMV,SpGEMM,转置,格式互转,稀疏 LU /
 * Cholesky 分解等接口.
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
 * @brief 稀疏矩阵存储格式.
 */
typedef enum {
    LMMC_SPARSE_CSR = 0, /**< 压缩行存储 (Compressed Sparse Row). */
    LMMC_SPARSE_CSC = 1  /**< 压缩列存储 (Compressed Sparse Column). */
} lmmc_sparse_format_t;

/**
 * @brief 通用稀疏矩阵(CSR / CSC 共用同一结构).
 *
 * 当 @c format = ::LMMC_SPARSE_CSR :
 *  - @c row_ptr 长度为 @c rows+1 ,
 *  - @c col_idx 与 @c values 长度均为 @c nnz .
 *
 * 当 @c format = ::LMMC_SPARSE_CSC :
 *  - @c row_ptr 在此别名为 col_ptr,长度为 @c cols+1 ,
 *  - @c col_idx 在此别名为 row_idx,长度为 @c nnz .
 */
typedef struct {
    size_t rows;                       /**< 行数. */
    size_t cols;                       /**< 列数. */
    size_t nnz;                        /**< 非零元个数. */
    size_t* row_ptr;                   /**< CSR 行指针 / CSC 列指针数组. */
    size_t* col_idx;                   /**< CSR 列索引 / CSC 行索引数组. */
    lmmc_real_t* values;               /**< 非零元数值数组,长度 @c nnz . */
    lmmc_sparse_format_t format;       /**< 存储格式. */
    int owns_data;                     /**< 是否拥有底层缓冲区. */
} lmmc_sparse_mat_t;

/**
 * @brief 增量构造稀疏矩阵的辅助对象(不透明类型).
 *
 * 适合 nnz 未知或乱序插入的场景.最终调用 ::lmmc_sparse_builder_build
 * 得到 ::lmmc_sparse_mat_t .
 */
typedef struct lmmc_sparse_builder_t lmmc_sparse_builder_t;

/**
 * @brief 创建稀疏矩阵增量构造器.
 *
 * 适用于非零元个数未知或乱序插入的场景.构造器内部维护动态数组,
 * 容量不足时自动翻倍扩容.最终调用 ::lmmc_sparse_builder_build 生成
 * 压缩格式矩阵.
 *
 * @param[in]  rows             矩阵行数,必须 > 0.
 * @param[in]  cols             矩阵列数,必须 > 0.
 * @param[in]  initial_capacity 初始预分配条目数;0 时使用内部默认值 16.
 * @param[out] out_builder      返回的构造器句柄指针.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - rows/cols 为 0 或 out_builder 为 NULL.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存.调用方必须配对调用 ::lmmc_sparse_builder_destroy 释放.
 */
lmmc_status_t lmmc_sparse_builder_create(size_t rows, size_t cols, size_t initial_capacity, lmmc_sparse_builder_t** out_builder);

/**
 * @brief 向构造器追加一项非零元.
 *
 * 若 (row, col) 已存在,数值会在 build 时被累加合并.
 * 容量不足时自动翻倍扩容.
 *
 * @param[in,out] builder 构造器句柄.
 * @param[in]     row     行下标,必须 < rows.
 * @param[in]     col     列下标,必须 < cols.
 * @param[in]     value   非零元数值.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - builder 为 NULL 或下标越界.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 扩容时内存分配失败.
 *
 * @par 副作用
 * - 修改构造器内部状态(nnz 递增,可能触发 realloc).
 */
lmmc_status_t lmmc_sparse_builder_add(lmmc_sparse_builder_t* builder, size_t row, size_t col, lmmc_real_t value);

/**
 * @brief 将构造器中的条目转换为指定格式的稀疏矩阵.
 *
 * 重复 (row,col) 条目的数值会被累加合并.转换后构造器仍有效,
 * 可继续追加条目或再次 build.
 *
 * @param[in]     builder    构造器句柄.
 * @param[in]     format     目标格式(CSR 或 CSC).
 * @param[out]    out_sparse 输出稀疏矩阵,调用方需配对调用 ::lmmc_sparse_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 参数为 NULL.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存用于 out_sparse 的数组.构造器本身不被修改.
 */
lmmc_status_t lmmc_sparse_builder_build(lmmc_sparse_builder_t* builder, lmmc_sparse_format_t format, lmmc_sparse_mat_t* out_sparse);

/**
 * @brief 销毁稀疏矩阵构造器并释放其全部内部内存.
 *
 * 对 NULL 指针安全.销毁后句柄不可再使用.
 *
 * @param[in] builder 构造器句柄,可为 NULL.
 *
 * @par 副作用
 * - 释放构造器内部的动态数组和句柄本身.
 */
void lmmc_sparse_builder_destroy(lmmc_sparse_builder_t* builder);

/**
 * @brief 创建指定 nnz 的空 CSR 矩阵(数组已分配但内容未初始化).
 *
 * 调用方需手动填充 row_ptr,col_idx,values 数组后方可使用.
 * 适用于已知稀疏结构的批量构造场景.
 *
 * @param[in]  rows       行数,必须 > 0.
 * @param[in]  cols       列数,必须 > 0.
 * @param[in]  nnz        非零元个数,可为 0.
 * @param[out] out_sparse 输出矩阵,调用方需配对调用 ::lmmc_sparse_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - rows/cols 为 0 或 out_sparse 为 NULL.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存(owns_data=1).row_ptr 已清零,col_idx/values 内容未定义.
 */
lmmc_status_t lmmc_sparse_create_csr(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse);
/**
 * @brief 创建指定 nnz 的空 CSC 矩阵(数组已分配但内容未初始化).
 *
 * 语义同 ::lmmc_sparse_create_csr,但 format 为 CSC.
 * row_ptr 在 CSC 语义下为 col_ptr,col_idx 为 row_idx.
 *
 * @param[in]  rows       行数,必须 > 0.
 * @param[in]  cols       列数,必须 > 0.
 * @param[in]  nnz        非零元个数,可为 0.
 * @param[out] out_sparse 输出矩阵.
 *
 * @return 同 ::lmmc_sparse_create_csr.
 *
 * @par 副作用
 * - 分配堆内存(owns_data=1).
 */
lmmc_status_t lmmc_sparse_create_csc(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse);

/**
 * @brief 用外部缓冲区包装为 CSR 视图(不拥有内存,不分配).
 *
 * 内部会验证 row_ptr 的单调性和 col_idx 的范围合法性.
 * 传入缓冲区的所有权保留给调用方;::lmmc_sparse_destroy 仅销毁视图.
 *
 * @param[in]  rows       行数.
 * @param[in]  cols       列数.
 * @param[in]  nnz        非零元个数.
 * @param[in]  row_ptr    行指针数组,长度 rows+1,row_ptr[0]==0,row_ptr[rows]==nnz.
 * @param[in]  col_idx    列索引数组,长度 nnz,每个值 < cols.
 * @param[in]  values     数值数组,长度 nnz.
 * @param[out] out_sparse 输出矩阵视图.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 参数非法或结构验证失败.
 *
 * @par 副作用
 * - 无内存分配.对视图的写操作直接修改外部缓冲区.
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
 * @brief 用外部缓冲区包装为 CSC 视图(不拥有内存,不分配).
 *
 * 语义同 ::lmmc_sparse_wrap_csr,但 format 为 CSC.
 * col_ptr 对应 row_ptr 字段,row_idx 对应 col_idx 字段.
 *
 * @param[in]  rows       行数.
 * @param[in]  cols       列数.
 * @param[in]  nnz        非零元个数.
 * @param[in]  col_ptr    列指针数组,长度 cols+1.
 * @param[in]  row_idx    行索引数组,长度 nnz,每个值 < rows.
 * @param[in]  values     数值数组,长度 nnz.
 * @param[out] out_sparse 输出矩阵视图.
 *
 * @return 同 ::lmmc_sparse_wrap_csr.
 *
 * @par 副作用
 * - 无内存分配.
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
 * @brief 销毁稀疏矩阵,释放其拥有的底层缓冲区.
 *
 * 若 owns_data 非零则释放 row_ptr,col_idx,values;否则仅清零字段.
 * 对 NULL 指针安全.
 *
 * @param[in,out] sparse 待销毁的稀疏矩阵,可为 NULL.
 *
 * @par 副作用
 * - 若 owns_data:释放堆内存.
 * - 所有字段清零,销毁后不可再用于运算.
 */
void lmmc_sparse_destroy(lmmc_sparse_mat_t* sparse);

/**
 * @brief 由稠密矩阵生成 CSR 稀疏矩阵,绝对值 <= eps 的元素被丢弃.
 *
 * 两遍扫描:第一遍计数 nnz,第二遍填充数组.
 *
 * @param[in]  dense      输入稠密矩阵(不被修改).
 * @param[in]  eps        丢弃阈值(非负).
 * @param[out] out_sparse 输出 CSR 矩阵,调用方需配对调用 ::lmmc_sparse_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或 eps < 0.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存.dense 不被修改.
 */
lmmc_status_t lmmc_sparse_from_dense(const lmmc_mat_t* dense, lmmc_real_t eps, lmmc_sparse_mat_t* out_sparse);

/**
 * @brief 将稀疏矩阵展开为稠密矩阵(缺失项填 0).
 *
 * out_dense 必须已创建且维度与 sparse 一致.先清零再填入非零元.
 *
 * @param[in]  sparse    输入稀疏矩阵(不被修改).
 * @param[out] out_dense 输出稠密矩阵,内容被完全覆写.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或结构非法.
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不匹配.
 *
 * @par 副作用
 * - 覆写 out_dense->data.无内存分配.
 */
lmmc_status_t lmmc_sparse_to_dense(const lmmc_sparse_mat_t* sparse, lmmc_mat_t* out_dense);

/**
 * @brief 稀疏矩阵-稠密向量乘法:y = A * x.
 *
 * 支持 CSR 和 CSC 格式.CSR 时使用 4-展开内循环优化.
 * x->size 须等于 sparse->cols,y->size 须等于 sparse->rows.
 *
 * @param[in]  sparse 输入稀疏矩阵(不被修改).
 * @param[in]  x      输入向量(不被修改).
 * @param[out] y      输出向量,内容被完全覆写.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或结构非法.
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 向量长度与矩阵维度不匹配.
 *
 * @par 副作用
 * - 覆写 y->data.无内存分配.
 */
lmmc_status_t lmmc_sparse_mat_vec_mul(const lmmc_sparse_mat_t* sparse, const lmmc_vec_t* x, lmmc_vec_t* y);

/**
 * @brief 计算稀疏矩阵的转置(保持原格式).
 *
 * 输出矩阵的 format 与输入相同.内部分配新的数组.
 *
 * @param[in]  sparse         输入稀疏矩阵(不被修改).
 * @param[out] out_transposed 输出转置矩阵,调用方需配对调用 ::lmmc_sparse_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或结构非法.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存.sparse 不被修改.
 */
lmmc_status_t lmmc_sparse_transpose(const lmmc_sparse_mat_t* sparse, lmmc_sparse_mat_t* out_transposed);

/**
 * @brief 稀疏矩阵 x 稠密矩阵:C = A * B(结果为稠密矩阵).
 *
 * sparse 为 mxk,b 为 kxn,c 必须已创建为 mxn.
 * 内部先清零 c 再累加.支持 CSR 和 CSC 格式,使用 4-展开优化.
 *
 * @param[in]  sparse 输入稀疏矩阵(不被修改).
 * @param[in]  b      输入稠密矩阵(不被修改).
 * @param[out] c      输出稠密矩阵,内容被完全覆写.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL.
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 维度不匹配.
 *
 * @par 副作用
 * - 覆写 c->data.无内存分配.
 */
lmmc_status_t lmmc_sparse_mat_mat_mul_dense(const lmmc_sparse_mat_t* sparse, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 稀疏矩阵-稀疏矩阵乘法(SpGEMM):C = A * B.
 *
 * 输出为 CSR 格式.内部使用行累加器(marker + accumulator)两遍算法:
 * 第一遍计算 nnz 结构,第二遍填充数值.
 * 若输入为 CSC 格式会先内部转换为 CSR.
 *
 * @param[in]  a 左稀疏矩阵(不被修改).
 * @param[in]  b 右稀疏矩阵(不被修改),a->cols 须等于 b->rows.
 * @param[out] c 输出稀疏矩阵(CSR),调用方需配对调用 ::lmmc_sparse_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL.
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - a->cols != b->rows.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存.若输入为 CSC 则内部临时分配 CSR 副本并在返回前释放.
 */
lmmc_status_t lmmc_sparse_mat_mat_mul_sparse(const lmmc_sparse_mat_t* a, const lmmc_sparse_mat_t* b, lmmc_sparse_mat_t* c);

/**
 * @brief 将稀疏矩阵转换为 CSC 格式.若 src 已是 CSC 则执行深拷贝.
 *
 * @param[in]  src 输入稀疏矩阵(不被修改).
 * @param[out] dst 输出 CSC 矩阵,调用方需配对调用 ::lmmc_sparse_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存.src 不被修改.
 */
lmmc_status_t lmmc_sparse_to_csc(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst);
/**
 * @brief 将稀疏矩阵转换为 CSR 格式.若 src 已是 CSR 则执行深拷贝.
 *
 * @param[in]  src 输入稀疏矩阵(不被修改).
 * @param[out] dst 输出 CSR 矩阵,调用方需配对调用 ::lmmc_sparse_destroy 释放.
 *
 * @return 同 ::lmmc_sparse_to_csc.
 *
 * @par 副作用
 * - 分配堆内存.src 不被修改.
 */
lmmc_status_t lmmc_sparse_to_csr(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst);

/**
 * @brief 稀疏 LU 分解上下文(不透明类型).
 *
 * 符号阶段在 A+A^T 的去重对称简单图上执行确定性的贪心残余度排序。
 * 该排序只删除已消元顶点的关联边，不建模消元填充，因此不宣称为 AMD。
 * 分解按符号分析、数值分解与求解三个阶段执行。
 */
typedef struct lmmc_sparse_lu_t lmmc_sparse_lu_t;

/**
 * @brief 稀疏 LU 符号分析阶段：去重简单图 + 贪心残余度重排序。
 *
 * 分析矩阵的稀疏结构，计算确定性的度数/原索引顺序，并预分配 L/U
 * 因子的存储空间。分析结果可复用于多次数值分解
 * (当矩阵结构不变,仅数值变化时).
 *
 * @param[in]  a      输入方阵(不被修改),rows 须等于 cols.
 * @param[out] out_lu 返回的 LU 上下文句柄,调用方需配对调用 ::lmmc_sparse_lu_destroy 释放.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或矩阵非方阵.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败.
 *
 * @par 副作用
 * - 分配堆内存(LU 上下文 + 内部工作数组).a 不被修改.
 * - 若 a 为 CSR 格式,内部会临时转换为 CSC 并在返回前释放.
 */
lmmc_status_t lmmc_sparse_lu_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t** out_lu
);

/**
 * @brief 稀疏 LU 数值分解阶段（稠密列工作区左看算法 + 部分主元）。
 *
 * 在已完成符号分析的 lu 上下文中执行数值分解。
 * 每列散布到稠密工作区并由已有 L 列更新；矩阵数值变化但结构不变时可复用 lu。
 *
 * @param[in]     a  输入方阵(不被修改),维度须与符号分析时一致.
 * @param[in,out] lu 已完成符号分析的 LU 上下文,数值因子被覆写.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或维度不匹配.
 * - ::LMMC_STATUS_SINGULAR_MATRIX - 主元为零,矩阵奇异.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 因子数组扩容失败.
 *
 * @par 副作用
 * - 修改 lu 内部的 L/U 数值数组和行置换.
 * - 分配临时工作内存并在返回前释放.
 *
 */
lmmc_status_t lmmc_sparse_lu_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t* lu
);

/**
 * @brief 利用已分解的稀疏 LU 因子求解 A*x = b.
 *
 * 求解过程:x = Q * U^{-1} * L^{-1} * P * b,其中 P 为行置换,Q 为残余度列置换.
 * lu 必须已完成 symbolic + numeric 两阶段.
 *
 * @param[in]  lu LU 上下文(不被修改).
 * @param[in]  b  右端向量,长度须等于 lu->n.
 * @param[out] x  解向量,长度须等于 lu->n,内容被覆写.
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功.
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL.
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 向量长度与矩阵阶数不匹配.
 * - ::LMMC_STATUS_SINGULAR_MATRIX - U 对角线为零.
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 临时工作内存分配失败.
 *
 * @par 副作用
 * - 覆写 x->data.分配并释放长度为 n 的临时工作数组.
 */
lmmc_status_t lmmc_sparse_lu_solve(
    const lmmc_sparse_lu_t* lu,
    const lmmc_vec_t* b,
    lmmc_vec_t* x
);

/** @brief 销毁稀疏 LU 上下文. */
void lmmc_sparse_lu_destroy(lmmc_sparse_lu_t* lu);

/**
 * @brief 稀疏 Cholesky 分解上下文(不透明类型,要求 @c A 对称正定).
 *
 * 符号阶段在去重对称简单图上执行确定性的贪心残余度排序；
 * 数值阶段使用稠密列工作区的左看 Cholesky。该排序不建模填充，不是 AMD。
 * 分解按符号分析、数值分解与求解三个阶段执行。
 */
typedef struct lmmc_sparse_chol_t lmmc_sparse_chol_t;

/** @brief 稀疏 Cholesky 符号分析：去重简单图和贪心残余度重排序。 */
lmmc_status_t lmmc_sparse_chol_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t** out_chol
);

/**
 * @brief 使用稠密列工作区的左看稀疏 Cholesky 数值分解.
 *
 * @return 若 @p a 非正定返回 ::LMMC_STATUS_NOT_POSITIVE_DEFINITE .
 */
lmmc_status_t lmmc_sparse_chol_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t* chol
);

/** @brief 利用已分解的 Cholesky 因子求解 @c A x = b . */
lmmc_status_t lmmc_sparse_chol_solve(
    const lmmc_sparse_chol_t* chol,
    const lmmc_vec_t* b,
    lmmc_vec_t* x
);

/** @brief 销毁稀疏 Cholesky 上下文. */
void lmmc_sparse_chol_destroy(lmmc_sparse_chol_t* chol);

/**
 * @brief 三元组 (COO) 形式的稀疏矩阵临时表示.
 *
 * 拥有自身缓冲区,主要用于增量插入后转换为 CSR / CSC .
 */
typedef struct {
    size_t rows;                /**< 行数. */
    size_t cols;                /**< 列数. */
    size_t nnz;                 /**< 当前已插入的条目数. */
    size_t capacity;            /**< 当前容量. */
    size_t* row_idx;            /**< 行下标数组,长度 @c capacity . */
    size_t* col_idx;            /**< 列下标数组,长度 @c capacity . */
    lmmc_real_t* values;        /**< 数值数组,长度 @c capacity . */
} lmmc_sparse_coo_t;

/** @brief 创建容量为 @p capacity 的空 COO 矩阵. */
lmmc_status_t lmmc_sparse_coo_create(
    size_t rows, size_t cols, size_t capacity,
    lmmc_sparse_coo_t* out_coo
);

/** @brief 向 COO 追加一项非零元,达到容量时返回错误. */
lmmc_status_t lmmc_sparse_coo_add_entry(
    lmmc_sparse_coo_t* coo,
    size_t row, size_t col, lmmc_real_t value
);

/** @brief 将 COO 转换为 CSR 矩阵(重复条目按相加合并). */
lmmc_status_t lmmc_sparse_coo_to_csr(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csr
);

/** @brief 将 COO 转换为 CSC 矩阵(重复条目按相加合并). */
lmmc_status_t lmmc_sparse_coo_to_csc(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csc
);

/** @brief 销毁 COO 矩阵并释放其缓冲区. */
void lmmc_sparse_coo_destroy(lmmc_sparse_coo_t* coo);

/** @brief 计算 @c C = alpha * A + beta * B .要求 @p a 与 @p b 同维同格式. */
lmmc_status_t lmmc_sparse_add(
    lmmc_real_t alpha, const lmmc_sparse_mat_t* a,
    lmmc_real_t beta, const lmmc_sparse_mat_t* b,
    lmmc_sparse_mat_t* out_c
);

/** @brief 就地标量缩放:@f$A \leftarrow \alpha A@f$ . */
lmmc_status_t lmmc_sparse_scale(
    lmmc_sparse_mat_t* a,
    lmmc_real_t alpha
);

/** @brief 计算稀疏矩阵的 Frobenius 范数. */
lmmc_status_t lmmc_sparse_norm_fro(
    const lmmc_sparse_mat_t* a,
    lmmc_real_t* out_norm
);

/** @brief 提取主对角线为稠密向量.要求 @c a->rows == a->cols . */
lmmc_status_t lmmc_sparse_diag(
    const lmmc_sparse_mat_t* a,
    lmmc_vec_t* out_diag
);

/**
 * @brief 块稀疏行 (BSR) 格式矩阵.
 *
 * 每个非零块为 @c block_size x @c block_size 的稠密子矩阵,
 * 按行优先存储在 @c values 中.
 */
typedef struct {
    size_t rows;          /**< 块行数. */
    size_t cols;          /**< 块列数. */
    size_t block_size;    /**< rxr 块维度. */
    size_t nnz_blocks;    /**< 非零块数量. */
    size_t* row_ptr;      /**< 块行指针数组,长度 rows+1 . */
    size_t* col_idx;      /**< 块列索引数组,长度 nnz_blocks . */
    lmmc_real_t* values;  /**< 块数值数组,长度 nnz_blocks * block_size^2 . */
    int owns_data;        /**< 是否拥有底层缓冲区. */
} lmmc_sparse_bsr_t;

/**
 * @brief 创建 BSR 矩阵(数组内容未初始化).
 *
 * @param[in]  rows       块行数.
 * @param[in]  cols       块列数.
 * @param[in]  block_size 块维度 r(每块为 rxr).
 * @param[in]  nnz_blocks 非零块数量.
 * @param[out] out        输出 BSR 矩阵.
 */
lmmc_status_t lmmc_sparse_bsr_create(size_t rows, size_t cols, size_t block_size,
    size_t nnz_blocks, lmmc_sparse_bsr_t* out);

/**
 * @brief 将 BSR 矩阵展开为稠密矩阵.
 *
 * @param[in]  bsr 输入 BSR 矩阵.
 * @param[out] out 输出稠密矩阵,必须已创建且维度为 (rows*block_size) x (cols*block_size).
 */
lmmc_status_t lmmc_sparse_bsr_to_dense(const lmmc_sparse_bsr_t* bsr, lmmc_mat_t* out);

/**
 * @brief 将稠密矩阵转换为 BSR 格式.
 *
 * 绝对值不超过 @p eps 的块(块内所有元素绝对值均 <= eps)被丢弃.
 *
 * @param[in]  dense      输入稠密矩阵,维度必须为 block_size 的整数倍.
 * @param[in]  block_size 块维度 r .
 * @param[in]  eps        丢弃阈值.
 * @param[out] out        输出 BSR 矩阵.
 */
lmmc_status_t lmmc_sparse_dense_to_bsr(const lmmc_mat_t* dense, size_t block_size,
    lmmc_real_t eps, lmmc_sparse_bsr_t* out);

/** @brief 销毁 BSR 矩阵,必要时释放底层缓冲区. */
void lmmc_sparse_bsr_destroy(lmmc_sparse_bsr_t* bsr);

/**
 * @brief 对称半存储选择:上三角或下三角.
 */
typedef enum {
    LMMC_SPARSE_SYM_UPPER = 0, /**< 存储上三角(含对角线). */
    LMMC_SPARSE_SYM_LOWER = 1  /**< 存储下三角(含对角线). */
} lmmc_sparse_sym_half_t;

/**
 * @brief 对称半存储 CSR 格式矩阵.
 *
 * 仅存储上三角或下三角(含对角线),用于对称矩阵的紧凑表示.
 */
typedef struct {
    size_t n;                        /**< 矩阵阶数(方阵). */
    size_t nnz;                      /**< 存储的非零元个数(仅半三角). */
    size_t* row_ptr;                 /**< 行指针数组,长度 n+1 . */
    size_t* col_idx;                 /**< 列索引数组,长度 nnz . */
    lmmc_real_t* values;             /**< 数值数组,长度 nnz . */
    lmmc_sparse_sym_half_t half;     /**< 存储的半三角类型. */
    int owns_data;                   /**< 是否拥有底层缓冲区. */
} lmmc_sparse_sym_csr_t;

/**
 * @brief 从完整 CSR 矩阵提取对称半存储.
 *
 * @param[in]  full 输入完整 CSR 矩阵(必须为方阵).
 * @param[in]  half 选择存储上三角或下三角.
 * @param[out] out  输出对称半存储 CSR 矩阵.
 */
lmmc_status_t lmmc_sparse_sym_csr_from_csr(const lmmc_sparse_mat_t* full,
    lmmc_sparse_sym_half_t half, lmmc_sparse_sym_csr_t* out);

/**
 * @brief 对称半存储 SpMV:@c y = A * x .
 *
 * 利用对称性,仅存储半三角但计算完整矩阵-向量乘积.
 *
 * @param[in]  A 对称半存储 CSR 矩阵.
 * @param[in]  x 输入向量,长度为 A->n .
 * @param[out] y 输出向量,长度为 A->n .
 */
lmmc_status_t lmmc_sparse_sym_spmv(const lmmc_sparse_sym_csr_t* A,
    const lmmc_vec_t* x, lmmc_vec_t* y);

/** @brief 销毁对称半存储 CSR 矩阵,必要时释放底层缓冲区. */
void lmmc_sparse_sym_csr_destroy(lmmc_sparse_sym_csr_t* s);

#ifdef __cplusplus
}
#endif

#endif
