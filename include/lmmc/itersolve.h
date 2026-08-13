/**
 * @file itersolve.h
 * @brief 稀疏线性系统的迭代求解器（CG / BiCGSTAB / GMRES / MINRES / LSQR）。
 *
 * 通用工作流程：
 *   1. 调用 ::lmmc_itersolve_default_config 取得默认参数。
 *   2. 根据需要修改容差、最大步数、日志回调等。
 *   3. 调用具体求解器并检查 ::lmmc_itersolve_result_t 。
 */
#ifndef LMMC_ITERSOLVE_H
#define LMMC_ITERSOLVE_H

#include <stddef.h>
#include "lmmc/numeric.h"
#include "lmmc/diagnostic.h"
#include "lmmc/precond.h"
#include "lmmc/sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 迭代求解的结果统计。
 */
typedef struct {
    int converged;                          /**< 是否收敛（非 0 表示已满足容差）。 */
    size_t num_iter;                        /**< 实际迭代步数。 */
    lmmc_real_t initial_residual_norm;      /**< 初始残差 2-范数。 */
    lmmc_real_t final_residual_norm;        /**< 终止时的残差 2-范数。 */
} lmmc_itersolve_result_t;

/**
 * @brief 矩阵-向量乘法算子回调类型（matrix-free 模式）。
 *
 * 用户提供的回调函数，计算 y = A * x 而无需显式存储矩阵 A 。
 *
 * @param[in]  x         输入向量。
 * @param[out] y         输出向量（由调用方预分配）。
 * @param[in]  user_data 用户透明上下文。
 * @return ::LMMC_STATUS_OK 成功；其它状态码表示失败。
 */
typedef lmmc_status_t (*lmmc_matvec_op_t)(
    const lmmc_vec_t* x, lmmc_vec_t* y, void* user_data);

/**
 * @brief 迭代求解器配置。
 */
typedef struct {
    lmmc_real_t abs_tol;                              /**< 残差绝对容差。 */
    lmmc_real_t rel_tol;                              /**< 残差相对容差（相对于初始残差）。 */
    size_t max_iter;                                  /**< 最大迭代步数。 */
    size_t restart;                                   /**< GMRES 重启长度（其它求解器忽略）。 */
    lmmc_diagnostic_sink_t diagnostics;               /**< 统一诊断出口。 */
    lmmc_matvec_op_t apply_op;                        /**< 矩阵-向量乘法算子回调，NULL 表示使用稀疏矩阵。 */
    void* op_user_data;                               /**< 传递给 @c apply_op 的上下文。 */
} lmmc_itersolve_config_t;

/**
 * @brief 根据问题规模生成迭代求解器默认配置。
 *
 * 默认值：abs_tol=1e-12, rel_tol=1e-8, max_iter=max(100, 20*n), restart=min(n,30)。
 *
 * @param[in]  problem_size 问题维度 n，必须 > 0。
 * @param[out] out_cfg      输出配置结构体。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — problem_size==0 或 out_cfg==NULL。
 *
 * @par 副作用
 * - 覆写 out_cfg 的所有字段。无内存分配。
 */
lmmc_status_t lmmc_itersolve_default_config(size_t problem_size, lmmc_itersolve_config_t* out_cfg);

/**
 * @brief 共轭梯度法（CG）求解对称正定稀疏系统 A*x = b。
 *
 * 要求 A 对称正定。x 作为初始猜测输入，收敛后存储解。
 * 收敛判据：||r||_2 <= abs_tol + rel_tol * ||b||_2。
 *
 * @param[in]     a          对称正定稀疏矩阵（不被修改）。
 * @param[in]     b          右端向量（不被修改）。
 * @param[in]     precond    预处理子，可为 NULL（无预处理）。
 * @param[in]     cfg        配置参数，可为 NULL（使用默认值）。
 * @param[in,out] x          初始猜测 / 输出解向量。
 * @param[out]    out_result 迭代统计结果，可为 NULL。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功（含收敛和达到最大步数两种情况，通过 out_result->converged 区分）。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或维度为 0。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 矩阵/向量/预处理子维度不匹配。
 * - ::LMMC_STATUS_NUMERICAL_FAILURE — 出现 NaN/Inf 或分母退化。
 *
 * @par 副作用
 * - 就地修改 x->data。分配并释放 4 个长度为 n 的临时向量。
 */
lmmc_status_t lmmc_cg_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
);

/**
 * @brief 稳定双共轭梯度法（BiCGSTAB）求解一般非对称稀疏系统 A*x = b。
 *
 * 适用于非对称或不定矩阵。x 作为初始猜测输入。
 *
 * @param[in]     a          稀疏矩阵（不被修改）。
 * @param[in]     b          右端向量（不被修改）。
 * @param[in]     precond    预处理子，可为 NULL。
 * @param[in]     cfg        配置参数，可为 NULL。
 * @param[in,out] x          初始猜测 / 输出解向量。
 * @param[out]    out_result 迭代统计结果，可为 NULL。
 *
 * @return 同 ::lmmc_cg_solve。
 *
 * @par 副作用
 * - 就地修改 x->data。分配并释放 9 个长度为 n 的临时向量。
 */
lmmc_status_t lmmc_bicgstab_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
);

/**
 * @brief 重启型 GMRES 求解一般非对称稀疏系统 A*x = b。
 *
 * 使用 Arnoldi 过程构建 Krylov 子空间，Givens 旋转求解最小残差。
 * 重启长度由 cfg->restart 控制（默认 30）。
 *
 * @param[in]     a          稀疏矩阵（不被修改）。
 * @param[in]     b          右端向量（不被修改）。
 * @param[in]     precond    预处理子，可为 NULL。
 * @param[in]     cfg        配置参数，可为 NULL。cfg->restart 控制重启长度。
 * @param[in,out] x          初始猜测 / 输出解向量。
 * @param[out]    out_result 迭代统计结果，可为 NULL。
 *
 * @return 同 ::lmmc_cg_solve。
 *
 * @par 副作用
 * - 就地修改 x->data。
 * - 分配 (restart+1) 个长度为 n 的 Arnoldi 基向量 + Hessenberg 矩阵等工作空间。
 */
lmmc_status_t lmmc_gmres_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
);

/**
 * @brief MINRES 求解对称（可能不定）稀疏系统 A x = b 。
 *
 * 当 cfg->apply_op 非 NULL 时使用 matrix-free 模式（此时 a 必须为 NULL）。
 * 若同时提供 a 和 cfg->apply_op ，返回 LMMC_STATUS_INVALID_ARGUMENT 。
 */
lmmc_status_t lmmc_minres_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
);

/**
 * @brief LSQR 求解最小二乘问题 min ||Ax - b||_2 。
 *
 * 基于 Golub-Kahan 双对角化。当 cfg->apply_op 非 NULL 时使用 matrix-free 模式
 * （此时 a 必须为 NULL）。若同时提供 a 和 cfg->apply_op ，
 * 返回 LMMC_STATUS_INVALID_ARGUMENT 。
 */
lmmc_status_t lmmc_lsqr_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
);

#ifdef __cplusplus
}
#endif

#endif
