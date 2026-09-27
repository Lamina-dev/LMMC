/** @file sparse_solvers.h */
#ifndef LMMC_SPARSE_SOLVERS_H
#define LMMC_SPARSE_SOLVERS_H

#include "lmmc/sparse_types.h"
#include "lmmc/dense_types.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 稀疏 LU 符号分析：去重简单图与贪心残余度重排序。
 *
 * 分析矩阵的稀疏结构，计算确定性的度数/原索引顺序，并预分配 L/U
 * 因子的存储空间。矩阵结构不变、仅数值变化时可复用分析结果。
 *
 * @param[in]  a      输入方阵，rows 须等于 cols。
 * @param[out] out_lu 返回的 LU 上下文句柄,调用方需配对调用 ::lmmc_sparse_lu_destroy 释放。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或矩阵非方阵。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存(LU 上下文 + 内部工作数组)。a 保持原值。
 * - 若 a 为 CSR 格式,内部会临时转换为 CSC 并在返回前释放。
 */
lmmc_status_t lmmc_sparse_lu_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t** out_lu
);

/**
 * @brief 稀疏 LU 数值分解：稠密列工作区左看算法与部分主元法。
 *
 * 每列散布到稠密工作区并由已有 L 列更新；矩阵数值变化但结构不变时可复用 lu。
 *
 * @param[in]     a  输入方阵，维度须与符号分析时一致。
 * @param[in,out] lu 已完成符号分析的 LU 上下文,数值因子被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL 或维度不匹配。
 * - ::LMMC_STATUS_SINGULAR_MATRIX - 主元为零,矩阵奇异。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 因子数组扩容失败。
 *
 * @par 副作用
 * - 修改 lu 内部的 L/U 数值数组和行置换。
 * - 分配临时工作内存并在返回前释放。
 */
lmmc_status_t lmmc_sparse_lu_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t* lu
);

/**
 * @brief 使用已分解的稀疏 LU 因子求解 A*x = b。
 *
 * 求解公式：x = Q * U^{-1} * L^{-1} * P * b,其中 P 为行置换,Q 为残余度列置换。
 * lu 必须已完成 符号分析与数值分解两阶段。
 *
 * @param[in]  lu LU 上下文。
 * @param[in]  b  右端向量,长度须等于 lu->n。
 * @param[out] x  解向量,长度须等于 lu->n,覆写原内容。
 *
 * @return
 * - ::LMMC_STATUS_OK - 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT - 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH - 向量长度与矩阵阶数不匹配。
 * - ::LMMC_STATUS_SINGULAR_MATRIX - U 对角线为零。
 * - ::LMMC_STATUS_ALLOCATION_FAILED - 临时工作内存分配失败。
 *
 * @par 副作用
 * - 覆写 x->data。分配并释放长度为 n 的临时工作数组。
 */
lmmc_status_t lmmc_sparse_lu_solve(
    const lmmc_sparse_lu_t* lu,
    const lmmc_vec_t* b,
    lmmc_vec_t* x
);

/** @brief 销毁稀疏 LU 上下文。 */
void lmmc_sparse_lu_destroy(lmmc_sparse_lu_t* lu);

/** @brief 稀疏 Cholesky 符号分析：去重简单图和贪心残余度重排序。 */
lmmc_status_t lmmc_sparse_chol_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t** out_chol
);

/**
 * @brief 使用稠密列工作区的左看稀疏 Cholesky 数值分解。
 *
 * @return 若 @p a 非正定返回 ::LMMC_STATUS_NOT_POSITIVE_DEFINITE 。
 */
lmmc_status_t lmmc_sparse_chol_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t* chol
);

/** @brief 使用已分解的 Cholesky 因子求解 A*x = b。 */
lmmc_status_t lmmc_sparse_chol_solve(
    const lmmc_sparse_chol_t* chol,
    const lmmc_vec_t* b,
    lmmc_vec_t* x
);

/** @brief 销毁稀疏 Cholesky 上下文。 */
void lmmc_sparse_chol_destroy(lmmc_sparse_chol_t* chol);

#ifdef __cplusplus
}
#endif

#endif
