#include "internal/interp_internal.h"

#include <string.h>

/** @brief Thomas 消元仅修改主对角线与右端项。 */
static void interp_tridiag_solve(size_t n, const lmmc_real_t* sub,
                                 lmmc_real_t* dia, const lmmc_real_t* sup,
                                 lmmc_real_t* rhs)
{
    size_t i;
    for (i = 1; i < n; ++i) {
        lmmc_real_t factor = sub[i] / dia[i - 1];
        dia[i] -= factor * sup[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    rhs[n - 1] /= dia[n - 1];
    for (i = n - 1; i > 0; --i) {
        rhs[i - 1] = (rhs[i - 1] - sup[i - 1] * rhs[i]) / dia[i - 1];
    }
}

static lmmc_real_t interp_spline_rhs(const interp_spline_workspace_t* w,
                                    size_t i)
{
    return 6.0 * ((w->ys[i + 1] - w->ys[i]) / w->h[i]
                 - (w->ys[i] - w->ys[i - 1]) / w->h[i - 1]);
}

static void interp_spline_interior(interp_spline_workspace_t* w,
                                    size_t first, size_t end, size_t offset)
{
    size_t i;
    for (i = first; i < end; ++i) {
        size_t row = i - offset;
        w->sub[row] = w->h[i - 1];
        w->dia[row] = 2.0 * (w->h[i - 1] + w->h[i]);
        w->sup[row] = w->h[i];
        w->rhs[row] = interp_spline_rhs(w, i);
    }
}

void interp_spline_standard(interp_spline_workspace_t* w,
                            lmmc_spline_bc_t bc, lmmc_real_t deriv_left,
                            lmmc_real_t deriv_right)
{
    size_t last = w->n - 1;
    const lmmc_real_t* h = w->h;
    const lmmc_real_t* ys = w->ys;
    interp_spline_interior(w, 1, last, 0);
    w->sub[0] = w->sup[0] = w->rhs[0] = 0.0;
    w->sub[last] = w->sup[last] = w->rhs[last] = 0.0;
    w->dia[0] = w->dia[last] = 1.0;
    if (bc == LMMC_SPLINE_CLAMPED) {
        w->dia[0] = 2.0 * h[0];
        w->sup[0] = h[0];
        w->rhs[0] = 6.0 * ((ys[1] - ys[0]) / h[0] - deriv_left);
        w->sub[last] = h[last - 1];
        w->dia[last] = 2.0 * h[last - 1];
        w->rhs[last] = 6.0 * (deriv_right - (ys[last] - ys[last - 1]) / h[last - 1]);
    }
    interp_tridiag_solve(w->n, w->sub, w->dia, w->sup, w->rhs);
    memcpy(w->moments, w->rhs, w->n * sizeof(lmmc_real_t));
}

void interp_spline_not_a_knot(interp_spline_workspace_t* w)
{
    size_t n = w->n, count = n - 2;
    const lmmc_real_t* h = w->h;
    lmmc_real_t* moments = w->moments;
    lmmc_real_t ha, hb;
    if (n == 3) {
        /** @brief 三个节点确定唯一二次多项式，其二阶导数为常数。 */
        lmmc_real_t value = interp_spline_rhs(w, 1) / (3.0 * (h[0] + h[1]));
        moments[0] = moments[1] = moments[2] = value;
        return;
    }
    /** @brief 利用三阶导数连续性消去端点的二阶导数。 */
    interp_spline_interior(w, 1, n - 1, 1);
    w->dia[0] = h[0] * (h[0] + h[1]) / h[1] + 2.0 * (h[0] + h[1]);
    w->sup[0] = h[1] - h[0] * h[0] / h[1];
    w->sub[0] = 0.0;
    ha = h[n - 3];
    hb = h[n - 2];
    w->sub[count - 1] = ha - hb * hb / ha;
    w->dia[count - 1] = 2.0 * (ha + hb) + hb * (ha + hb) / ha;
    w->sup[count - 1] = 0.0;
    interp_tridiag_solve(count, w->sub, w->dia, w->sup, w->rhs);
    memcpy(moments + 1, w->rhs, count * sizeof(lmmc_real_t));
    moments[0] = ((h[0] + h[1]) * moments[1] - h[0] * moments[2]) / h[1];
    moments[n - 1] = ((ha + hb) * moments[n - 2] - hb * moments[n - 3]) / ha;
}

void interp_spline_periodic(interp_spline_workspace_t* w)
{
    size_t count = w->n - 1, i;
    const lmmc_real_t* h = w->h;
    const lmmc_real_t* ys = w->ys;
    lmmc_real_t corner = h[count - 1], gamma, vy, vz;
    interp_spline_interior(w, 1, count, 0);
    w->dia[0] = 2.0 * (corner + h[0]);
    w->sup[0] = h[0];
    w->sub[0] = 0.0;
    w->rhs[0] = 6.0 * ((ys[1] - ys[0]) / h[0]
                       - (ys[count] - ys[count - 1]) / corner);
    gamma = -w->dia[0];
    /** @brief Sherman-Morrison 修正复用同一三对角系统，分别求解右端项 b 和 u。 */
    w->dia[0] -= gamma;
    w->dia[count - 1] -= corner * corner / gamma;
    w->sup[count - 1] = 0.0;
    for (i = 0; i < count; ++i) w->auxiliary[i] = 0.0;
    w->auxiliary[0] = gamma;
    w->auxiliary[count - 1] = corner;
    memcpy(w->diagonal_copy, w->dia, count * sizeof(lmmc_real_t));
    interp_tridiag_solve(count, w->sub, w->dia, w->sup, w->rhs);
    interp_tridiag_solve(count, w->sub, w->diagonal_copy, w->sup, w->auxiliary);
    vy = w->rhs[0] + (corner / gamma) * w->rhs[count - 1];
    vz = w->auxiliary[0] + (corner / gamma) * w->auxiliary[count - 1];
    for (i = 0; i < count; ++i) {
        w->moments[i] = w->rhs[i] - w->auxiliary[i] * vy / (1.0 + vz);
    }
    w->moments[count] = w->moments[0];
}
