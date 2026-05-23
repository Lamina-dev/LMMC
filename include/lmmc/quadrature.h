#ifndef LMMC_QUADRATURE_H
#define LMMC_QUADRATURE_H

#include "lmmc/config.h"
#include "lmmc/status.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 被积函数类型 */
typedef lmmc_real_t (*lmmc_quad_func_t)(lmmc_real_t x, void* user_data);

/* 自适应积分结果 */
typedef struct {
    lmmc_real_t value;        /* 积分估计值 */
    lmmc_real_t error;        /* 误差估计 */
    size_t num_evals;         /* 函数求值次数 */
} lmmc_quad_result_t;

/* 复合梯形法 */
lmmc_status_t lmmc_quad_trapezoid(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t n,
    lmmc_real_t* out_result
);

/* 复合 Simpson 法 */
lmmc_status_t lmmc_quad_simpson(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t n,
    lmmc_real_t* out_result
);

/* Gauss-Legendre 求积 */
lmmc_status_t lmmc_quad_gauss_legendre(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t order,
    lmmc_real_t* out_result
);

/* 自适应 Gauss-Kronrod 求积 */
lmmc_status_t lmmc_quad_adaptive(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t abs_tol,
    lmmc_real_t rel_tol,
    size_t max_depth,
    lmmc_quad_result_t* out_result
);

#ifdef __cplusplus
}
#endif

#endif
