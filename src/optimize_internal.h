#ifndef LMMC_OPTIMIZE_INTERNAL_H
#define LMMC_OPTIMIZE_INTERNAL_H

#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/linear_algebra.h"
#include "lmmc/optimize.h"

/** @brief 工作区由本次调用拥有，视图仅借用存储，释放由调用方负责。 */
typedef struct {
    void* allocation;
    lmmc_real_t* next;
    size_t* pivots;
} lmmc_opt_workspace_t;

lmmc_status_t lmmc_opt_workspace_create(
    size_t n, size_t vectors, size_t matrices, size_t extra,
    int need_pivots, lmmc_opt_workspace_t* work);
lmmc_vec_t lmmc_opt_workspace_vector(lmmc_opt_workspace_t* work, size_t n);
lmmc_mat_t lmmc_opt_workspace_matrix(lmmc_opt_workspace_t* work, size_t n);

void lmmc_optimize_emit(const lmmc_optimize_config_t* cfg,
    const char* operation, size_t iteration,
    const lmmc_real_t* values, size_t value_count);
int lmmc_optimize_config_is_valid(const lmmc_optimize_config_t* cfg);
int lmmc_optimize_has_converged(lmmc_real_t residual,
    lmmc_real_t initial_residual, const lmmc_optimize_config_t* cfg);
int lmmc_optimize_arguments_valid(const lmmc_vec_t* x,
    const lmmc_optimize_config_t* cfg, const lmmc_optimize_result_t* out);
void lmmc_optimize_result_init(lmmc_optimize_result_t* out);
lmmc_real_t lmmc_optimize_norm(const lmmc_vec_t* v);
int lmmc_optimize_stop(lmmc_real_t norm, lmmc_real_t* initial,
    size_t iteration, const lmmc_optimize_config_t* cfg,
    lmmc_optimize_result_t* out);

/** @brief 临时向量由外层求解器一次性分配。 */
lmmc_status_t lmmc_optimize_jacobian(
    lmmc_opt_func_t F, lmmc_opt_jac_t J, void* user_data,
    const lmmc_vec_t* x, const lmmc_vec_t* Fx, lmmc_mat_t* matrix,
    lmmc_vec_t* x_pert, lmmc_vec_t* F_pert);
int lmmc_optimize_armijo(
    lmmc_opt_obj_t obj, void* user_data, const lmmc_vec_t* x,
    const lmmc_vec_t* direction, lmmc_vec_t* trial,
    lmmc_real_t* value, lmmc_real_t derivative, int attempts, int subtract);

#endif
