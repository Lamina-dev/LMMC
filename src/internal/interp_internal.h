#ifndef LMMC_INTERP_INTERNAL_H
#define LMMC_INTERP_INTERNAL_H

#include "lmmc/interp.h"

size_t interp_find_interval(const lmmc_real_t* xs, size_t n, lmmc_real_t query_x);
lmmc_real_t interp_interval_fraction(lmmc_real_t left, lmmc_real_t right,
                                     lmmc_real_t query);
int interp_check_strictly_increasing(const lmmc_real_t* xs, size_t n);
int interp_check_finite_values(const lmmc_real_t* values, size_t n);
int interp_check_samples(const lmmc_real_t* xs, const lmmc_real_t* ys,
                         size_t n, size_t minimum);
lmmc_status_t interp_barycentric_weights(const lmmc_real_t* nodes, size_t n,
                                        lmmc_real_t* weights);

typedef struct {
    const lmmc_real_t* xs;
    const lmmc_real_t* ys;
    const lmmc_real_t* zs;
    size_t nx;
    size_t ny;
    lmmc_real_t qx;
    lmmc_real_t qy;
} interp_grid_t;

lmmc_status_t interp_check_grid(const interp_grid_t* grid, size_t minimum,
                               const lmmc_real_t* out);

typedef struct {
    size_t n;
    const lmmc_real_t* ys;
    lmmc_real_t* h;
    lmmc_real_t* moments;
    lmmc_real_t* sub;
    lmmc_real_t* dia;
    lmmc_real_t* sup;
    lmmc_real_t* rhs;
    lmmc_real_t* auxiliary;
    lmmc_real_t* diagonal_copy;
} interp_spline_workspace_t;

void interp_spline_standard(interp_spline_workspace_t* work,
                            lmmc_spline_bc_t bc, lmmc_real_t deriv_left,
                            lmmc_real_t deriv_right);
void interp_spline_not_a_knot(interp_spline_workspace_t* work);
void interp_spline_periodic(interp_spline_workspace_t* work);

#endif
