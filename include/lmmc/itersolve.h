/**
 * @file itersolve.h
 * @brief 稀疏线性系统的迭代求解器（CG / BiCGSTAB / GMRES）。
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
 * @brief 迭代求解器逐步日志回调签名。
 *
 * @param iter         当前迭代步（从 0 起）。
 * @param residual_norm 当前残差 2-范数。
 * @param user_data    用户透明上下文。
 */
typedef void (*lmmc_itersolve_log_callback_t)(
    size_t iter,
    lmmc_real_t residual_norm,
    void* user_data
);

/**
 * @brief 迭代求解器配置。
 */
typedef struct {
    lmmc_real_t abs_tol;                              /**< 残差绝对容差。 */
    lmmc_real_t rel_tol;                              /**< 残差相对容差（相对于初始残差）。 */
    size_t max_iter;                                  /**< 最大迭代步数。 */
    size_t restart;                                   /**< GMRES 重启长度（其它求解器忽略）。 */
    int verbose;                                      /**< 非 0 时启用 stdout 日志。 */
    lmmc_itersolve_log_callback_t log_cb;             /**< 自定义日志回调，可为 NULL 。 */
    void* log_user_data;                              /**< 传递给 @c log_cb 的上下文。 */
} lmmc_itersolve_config_t;

/**
 * @brief 根据问题规模生成默认配置。
 *
 * @param[in]  problem_size 待求解问题维度。
 * @param[out] out_cfg      返回的配置结构。
 */
lmmc_status_t lmmc_itersolve_default_config(size_t problem_size, lmmc_itersolve_config_t* out_cfg);

/**
 * @brief 共轭梯度法求解对称正定稀疏系统 @c A x = b 。
 *
 * @param[in]     a          系数矩阵（CSR / CSC 均可，要求对称正定）。
 * @param[in]     b          右端向量。
 * @param[in]     precond    预处理子，可为 NULL 表示无预处理。
 * @param[in]     cfg        求解配置。
 * @param[in,out] x          初始猜测 / 输出解。
 * @param[out]    out_result 求解统计信息，可为 NULL 。
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
 * @brief 稳定双共轭梯度法（BiCGSTAB）求解一般非对称稀疏系统。
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
 * @brief 重启型 GMRES 求解一般非对称稀疏系统，由 @c cfg->restart 控制重启长度。
 */
lmmc_status_t lmmc_gmres_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
);

#ifdef __cplusplus
}
#endif

#endif
