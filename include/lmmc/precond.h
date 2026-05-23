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

/** @brief 由稀疏矩阵 @p a 的对角线创建 Jacobi 预处理子。 */
lmmc_status_t lmmc_precond_create_jacobi(const lmmc_sparse_mat_t* a, lmmc_precond_t* out_precond);

/** @brief 创建 ILU(0) 预处理子，填充模式与 @p a 完全一致。 */
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
 * @brief 应用预处理子：求解 @c M @c out = @c rhs 。
 *
 * @p out 必须与 @p rhs 同长度且互不别名。
 */
lmmc_status_t lmmc_precond_apply(const lmmc_precond_t* precond, const lmmc_vec_t* rhs, lmmc_vec_t* out);

/** @brief 销毁预处理子，必要时释放内部资源。 */
void lmmc_precond_destroy(lmmc_precond_t* precond);

#ifdef __cplusplus
}
#endif

#endif
