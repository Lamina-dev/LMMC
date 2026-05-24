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
    int verbose;                                      /**< 非 0 时启用 stdout 日志。 */
    lmmc_itersolve_log_callback_t log_cb;             /**< 自定义日志回调，可为 NULL 。 */
    void* log_user_data;                              /**< 传递给 @c log_cb 的上下文。 */
    lmmc_matvec_op_t apply_op;                        /**< 矩阵-向量乘法算子回调，NULL 表示使用稀疏矩阵。 */
    void* op_user_data;                               /**< 传递给 @c apply_op 的上下文。 */
} lmmc_itersolve_config_t;

/**
 * @brief 根据问题规模生成默认配置。
 */
lmmc_status_t lmmc_itersolve_default_config(size_t problem_size, lmmc_itersolve_config_t* out_cfg);

/**
 * @brief 共轭梯度法求解对称正定稀疏系统 A x = b 。
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
 * @brief 重启型 GMRES 求解一般非对称稀疏系统。
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
