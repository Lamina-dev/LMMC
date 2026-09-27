#ifndef LMMC_QUADRATURE_INTERNAL_H
#define LMMC_QUADRATURE_INTERNAL_H

#include "lmmc/quadrature.h"
#include "internal.h"

void lmmc_quad_center_half_length(
    lmmc_real_t a, lmmc_real_t b,
    lmmc_real_t* center, lmmc_real_t* half_length);
lmmc_real_t lmmc_quad_midpoint(lmmc_real_t a, lmmc_real_t b);

#endif
