/**
 * @file precond.h
 * @brief 迭代求解器使用的预处理子接口。
 *
 * 当前支持：恒等（无预处理）、Jacobi（对角）、ILU(0) 和 ILUT 。
 */
#ifndef LMMC_PRECOND_H
#define LMMC_PRECOND_H

#include <stddef.h>
#include "lmmc/dense.h"
#include "lmmc/sparse.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 预处理子类型标识。 */
typedef enum {
    LMMC_PRECOND_NONE = 0,   /**< 恒等预处理，等价于无预处理。 */
    LMMC_PRECOND_JACOBI = 1, /**< Jacobi (对角线倒数)。 */
    LMMC_PRECOND_ILU0 = 2,   /**< 不完全 LU 分解，零填充。 */
    LMMC_PRECOND_ILUT = 3    /**< 阈值与最大填充控制的不完全 LU 分解。 */
} lmmc_precond_type_t;

/**
 * @brief 通用预处理子句柄。
 *
 * 内部 @c impl 指向具体类型相关的数据结构，调用方仅通过
 * ::lmmc_precond_apply / ::lmmc_precond_destroy 操作。
 */
typedef struct {
    lmmc_precond_type_t type;  /**< 预处理子类型。 */
    size_t size;               /**< 适用的向量长度。 */
    void* impl;                /**< 内部实现指针。 */
    int owns_data;             /**< 是否拥有内部数据所有权。 */
} lmmc_precond_t;

/** @brief 创建恒等预处理子（仅记录尺寸，apply 时直接复制）。 */
lmmc_status_t lmmc_precond_create_none(size_t size, lmmc_precond_t* out_precond);

/**
 * @brief 由稀疏矩阵的对角线创建 Jacobi 预处理子：M = diag(A)。
 *
 * 提取 A 的对角线元素并取倒数存储。若某对角元为零或过小（<1e-15），
 * 返回 SINGULAR_MATRIX。
 *
 * @param[in]  a          输入稀疏矩阵（CSR 格式，不被修改）。
 * @param[out] out_precond 输出预处理子句柄。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或结构非法。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 矩阵非方阵。
 * - ::LMMC_STATUS_SINGULAR_MATRIX — 对角线存在零元素。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存存储对角线倒数。调用方需配对调用 ::lmmc_precond_destroy 释放。
 */
lmmc_status_t lmmc_precond_create_jacobi(const lmmc_sparse_mat_t* a, lmmc_precond_t* out_precond);

/**
 * @brief 创建 ILU(0) 预处理子（零填充不完全 LU 分解）。
 *
 * 填充模式与输入矩阵 A 完全一致（不产生新的非零位置）。
 * 内部复制 A 的结构并就地执行 IKJ 版本的 ILU 分解。
 *
 * @param[in]  a          输入稀疏矩阵（CSR 格式，不被修改）。
 * @param[out] out_precond 输出预处理子句柄。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或结构非法。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 矩阵非方阵。
 * - ::LMMC_STATUS_SINGULAR_MATRIX — 对角线为零或缺失。
 * - ::LMMC_STATUS_NUMERICAL_FAILURE — 分解过程中出现 NaN/Inf。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存。调用方需配对调用 ::lmmc_precond_destroy 释放。
 */
lmmc_status_t lmmc_precond_create_ilu0(const lmmc_sparse_mat_t* a, lmmc_precond_t* out_precond);

/**
 * @brief 创建 ILUT 预处理子。
 *
 * @param[in]  a                输入稀疏矩阵。
 * @param[in]  drop_tol         小于此阈值的填充被丢弃（绝对值比较）。
 * @param[in]  max_fill_per_row 每行最多保留的非零元个数。
 * @param[out] out_precond      返回的预处理子句柄。
 */
lmmc_status_t lmmc_precond_create_ilut(
    const lmmc_sparse_mat_t* a,
    lmmc_real_t drop_tol,
    size_t max_fill_per_row,
    lmmc_precond_t* out_precond
);

/**
 * @brief 应用预处理子：求解 M*out = rhs。
 *
 * 根据 precond->type 分派到对应的求解逻辑：
 * - NONE：直接复制 rhs 到 out。
 * - JACOBI：逐元素乘以对角线倒数。
 * - ILU0/ILUT：前代 + 回代。
 *
 * @param[in]  precond 预处理子句柄（不被修改）。
 * @param[in]  rhs     右端向量（不被修改）。
 * @param[out] out     输出向量，长度须与 rhs 相同，内容被覆写。out 不可与 rhs 别名。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 向量长度与预处理子尺寸不匹配。
 *
 * @par 副作用
 * - 覆写 out->data。无内存分配。
 */
lmmc_status_t lmmc_precond_apply(const lmmc_precond_t* precond, const lmmc_vec_t* rhs, lmmc_vec_t* out);

/**
 * @brief 销毁预处理子，释放内部资源。
 *
 * 对 NULL 指针安全。销毁后句柄不可再使用。
 *
 * @param[in,out] precond 预处理子句柄，可为 NULL。
 *
 * @par 副作用
 * - 释放 impl 指向的内部数据结构。
 */
void lmmc_precond_destroy(lmmc_precond_t* precond);

#ifdef __cplusplus
}
#endif

#endif
