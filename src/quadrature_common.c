#include "internal/quadrature_internal.h"

void lmmc_quad_center_half_length(
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t* center,
    lmmc_real_t* half_length)
{
    lmmc_interval_center_half_width(a, b, center, half_length);
}

lmmc_real_t lmmc_quad_midpoint(lmmc_real_t a, lmmc_real_t b)
{
    return lmmc_interval_midpoint(a, b);
}
