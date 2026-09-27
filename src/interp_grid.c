#include "internal/interp_internal.h"

#include <math.h>

lmmc_status_t lmmc_interp_bilinear(
    const lmmc_real_t* xs, size_t nx,
    const lmmc_real_t* ys, size_t ny,
    const lmmc_real_t* zs,
    lmmc_real_t qx, lmmc_real_t qy,
    lmmc_real_t* out_z)
{
    size_t ix, iy;
    const interp_grid_t grid = {xs, ys, zs, nx, ny, qx, qy};
    lmmc_status_t status = interp_check_grid(&grid, 2, out_z);
    lmmc_real_t tx, ty;
    lmmc_real_t z00, z01, z10, z11, result;

    if (status != LMMC_STATUS_OK) {
        return status;
    }

    ix = interp_find_interval(xs, nx, qx);
    iy = interp_find_interval(ys, ny, qy);
    if (ix >= nx - 1) ix = nx - 2;
    if (iy >= ny - 1) iy = ny - 2;

    tx = (qx - xs[ix]) / (xs[ix + 1] - xs[ix]);
    ty = (qy - ys[iy]) / (ys[iy + 1] - ys[iy]);
    z00 = zs[ix * ny + iy];
    z01 = zs[ix * ny + (iy + 1)];
    z10 = zs[(ix + 1) * ny + iy];
    z11 = zs[(ix + 1) * ny + (iy + 1)];
    if (!isfinite(z00) || !isfinite(z01)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(z10) || !isfinite(z11)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    result = (1.0 - tx) * (1.0 - ty) * z00
           + (1.0 - tx) * ty * z01
           + tx * (1.0 - ty) * z10
           + tx * ty * z11;
    if (!isfinite(result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out_z = result;
    return LMMC_STATUS_OK;
}

/** @brief 选取四个连续节点，边界模板中的端点各出现一次。 */
static size_t interp_grid_stencil(const lmmc_real_t* xs, size_t n, lmmc_real_t q)
{
    size_t interval = interp_find_interval(xs, n, q);
    size_t start = interval > 0 ? interval - 1 : 0;
    return start > n - 4 ? n - 4 : start;
}

static lmmc_status_t interp_grid_basis(const lmmc_real_t* nodes, lmmc_real_t q,
                                      lmmc_real_t* basis)
{
    size_t i, j;
    lmmc_real_t denominator = 0.0;
    for (i = 0; i < 4; ++i) {
        if (q == nodes[i]) {
            for (j = 0; j < 4; ++j) basis[j] = i == j ? 1.0 : 0.0;
            return LMMC_STATUS_OK;
        }
    }
    lmmc_status_t status = interp_barycentric_weights(nodes, 4, basis);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    for (i = 0; i < 4; ++i) {
        const lmmc_real_t difference = q - nodes[i];
        if (!isfinite(difference)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        basis[i] /= difference;
        denominator += basis[i];
        if (!isfinite(basis[i]) || !isfinite(denominator)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    if (denominator == 0.0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    for (i = 0; i < 4; ++i) basis[i] /= denominator;
    return interp_check_finite_values(basis, 4) ?
        LMMC_STATUS_OK : LMMC_STATUS_NUMERICAL_FAILURE;
}

static int interp_grid_values_finite(const interp_grid_t* grid,
                                     size_t ix, size_t iy)
{
    size_t i;
    for (i = 0; i < 4; ++i) {
        if (!interp_check_finite_values(grid->zs + (ix + i) * grid->ny + iy, 4)) {
            return 0;
        }
    }
    return 1;
}

static lmmc_status_t interp_grid_tensor_product(const interp_grid_t* grid,
    size_t ix, size_t iy, const lmmc_real_t* bx, const lmmc_real_t* by,
    lmmc_real_t* out)
{
    size_t i, j;
    lmmc_real_t result = 0.0;
    for (i = 0; i < 4; ++i) {
        lmmc_real_t row = 0.0;
        for (j = 0; j < 4; ++j) {
            row += by[j] * grid->zs[(ix + i) * grid->ny + iy + j];
            if (!isfinite(row)) {
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }
        result += bx[i] * row;
        if (!isfinite(result)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_interp_bicubic(
    const lmmc_real_t* xs, size_t nx,
    const lmmc_real_t* ys, size_t ny,
    const lmmc_real_t* zs, lmmc_real_t qx, lmmc_real_t qy,
    lmmc_real_t* out_z)
{
    const interp_grid_t grid = {xs, ys, zs, nx, ny, qx, qy};
    size_t ix, iy;
    lmmc_real_t bx[4], by[4];
    lmmc_status_t status = interp_check_grid(&grid, 4, out_z);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    ix = interp_grid_stencil(xs, nx, qx);
    iy = interp_grid_stencil(ys, ny, qy);
    if (!interp_grid_values_finite(&grid, ix, iy)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = interp_grid_basis(xs + ix, qx, bx);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = interp_grid_basis(ys + iy, qy, by);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    return interp_grid_tensor_product(&grid, ix, iy, bx, by, out_z);
}
