#ifndef LMMC_INTERP_H
#define LMMC_INTERP_H

#include "lmmc/config.h"
#include "lmmc/status.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 线性插值（无状态，直接计算） */
lmmc_status_t lmmc_interp_linear(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_real_t query_x,
    lmmc_real_t* out_y
);

/* 三次样条插值上下文 */
typedef struct lmmc_interp_cspline_t lmmc_interp_cspline_t;

lmmc_status_t lmmc_interp_cspline_create(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_interp_cspline_t** out_spline
);

lmmc_status_t lmmc_interp_cspline_eval(
    const lmmc_interp_cspline_t* spline,
    lmmc_real_t query_x,
    lmmc_real_t* out_y
);

void lmmc_interp_cspline_destroy(lmmc_interp_cspline_t* spline);

/* Lagrange 插值上下文（重心形式） */
typedef struct lmmc_interp_lagrange_t lmmc_interp_lagrange_t;

lmmc_status_t lmmc_interp_lagrange_create(
    const lmmc_real_t* xs,
    const lmmc_real_t* ys,
    size_t n,
    lmmc_interp_lagrange_t** out_lagrange
);

lmmc_status_t lmmc_interp_lagrange_eval(
    const lmmc_interp_lagrange_t* lagrange,
    lmmc_real_t query_x,
    lmmc_real_t* out_y
);

void lmmc_interp_lagrange_destroy(lmmc_interp_lagrange_t* lagrange);

#ifdef __cplusplus
}
#endif

#endif
