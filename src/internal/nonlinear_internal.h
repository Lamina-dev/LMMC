#ifndef LMMC_NONLINEAR_INTERNAL_H
#define LMMC_NONLINEAR_INTERNAL_H
#include <math.h>
#include <float.h>
#include "internal.h"
#include "lmmc/nonlinear.h"

void lmmc_nonlinear_do_log(const lmmc_nonlinear_config_t* cfg, size_t iter, lmmc_real_t x, lmmc_real_t f_x);

lmmc_real_t lmmc_nonlinear_midpoint(
    lmmc_real_t left, lmmc_real_t right);

lmmc_real_t lmmc_nonlinear_half_width(
    lmmc_real_t left, lmmc_real_t right);

lmmc_real_t lmmc_nonlinear_distance(
    lmmc_real_t left, lmmc_real_t right);

lmmc_status_t lmmc_nonlinear_load_and_validate_config(
    const lmmc_nonlinear_config_t* cfg,
    lmmc_nonlinear_config_t* out_cfg
);

lmmc_status_t lmmc_nonlinear_x_tolerance(
    lmmc_real_t x,
    const lmmc_nonlinear_config_t* cfg,
    lmmc_real_t* out_x_tol
);

void lmmc_nonlinear_reset_result(lmmc_nonlinear_result_t* out_result);

#endif
