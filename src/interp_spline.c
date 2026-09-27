#include "memory_bridge.h"
#include "internal/interp_internal.h"

#include <math.h>
#include <string.h>

struct lmmc_interp_cspline_t {
    size_t n;
    lmmc_real_t* xs;
    lmmc_real_t* ys;
    lmmc_real_t* coeffs; /**< 共 4*(n-1) 个系数，每段按 a、b、c、d 存储。 */
};

void lmmc_interp_cspline_destroy(lmmc_interp_cspline_t* spline)
{
    if (!spline) {
        return;
    }
    lmmc_memory_free(spline->xs);
    lmmc_memory_free(spline->ys);
    lmmc_memory_free(spline->coeffs);
    lmmc_memory_free(spline);
}

lmmc_status_t lmmc_interp_cspline_eval(
    const lmmc_interp_cspline_t* spline,
    lmmc_real_t query_x, lmmc_real_t* out_y)
{
    size_t seg;
    lmmc_real_t dx, a, b, c, d, result;
    if (!spline || !out_y || !isfinite(query_x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (query_x < spline->xs[0] || query_x > spline->xs[spline->n - 1]) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    if (query_x == spline->xs[spline->n - 1]) {
        *out_y = spline->ys[spline->n - 1];
        return LMMC_STATUS_OK;
    }
    seg = interp_find_interval(spline->xs, spline->n, query_x);
    dx = query_x - spline->xs[seg];
    a = spline->coeffs[4 * seg + 0];
    b = spline->coeffs[4 * seg + 1];
    c = spline->coeffs[4 * seg + 2];
    d = spline->coeffs[4 * seg + 3];
    result = a + dx * (b + dx * (c + dx * d));
    if (!isfinite(result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out_y = result;
    return LMMC_STATUS_OK;
}

/**
 * @internal
 * @brief 由二阶导数（矩）计算样条系数。
 */
static void cspline_compute_coeffs(lmmc_interp_cspline_t* spline,
                                    const lmmc_real_t* h,
                                    const lmmc_real_t* M)
{
    size_t i, nm1 = spline->n - 1;
    for (i = 0; i < nm1; i++) {
        lmmc_real_t ai = spline->ys[i];
        lmmc_real_t ci = M[i] / 2.0;
        lmmc_real_t di = (M[i + 1] - M[i]) / (6.0 * h[i]);
        lmmc_real_t bi = (spline->ys[i + 1] - spline->ys[i]) / h[i]
                       - h[i] * (2.0 * M[i] + M[i + 1]) / 6.0;
        spline->coeffs[4 * i + 0] = ai;
        spline->coeffs[4 * i + 1] = bi;
        spline->coeffs[4 * i + 2] = ci;
        spline->coeffs[4 * i + 3] = di;
    }
}

static int interp_spline_check_boundary(const lmmc_real_t* ys, size_t n,
                                        lmmc_spline_bc_t bc,
                                        lmmc_real_t left, lmmc_real_t right)
{
    if (bc < LMMC_SPLINE_NATURAL || bc > LMMC_SPLINE_PERIODIC) {
        return 0;
    }
    if (bc == LMMC_SPLINE_CLAMPED) {
        return isfinite(left) && isfinite(right);
    }
    if (bc == LMMC_SPLINE_PERIODIC && fabs(ys[0] - ys[n - 1]) > 1e-12) {
        return 0;
    }
    return 1;
}

static lmmc_interp_cspline_t* interp_spline_allocate(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n)
{
    lmmc_interp_cspline_t* spline = lmmc_memory_alloc(sizeof(*spline));
    if (!spline) {
        return NULL;
    }
    spline->n = n;
    spline->xs = lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    spline->ys = lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    spline->coeffs = lmmc_memory_alloc_array_2d(n - 1, 4, sizeof(lmmc_real_t));
    if (!spline->xs || !spline->ys || !spline->coeffs) {
        lmmc_interp_cspline_destroy(spline);
        return NULL;
    }
    memcpy(spline->xs, xs, n * sizeof(lmmc_real_t));
    memcpy(spline->ys, ys, n * sizeof(lmmc_real_t));
    return spline;
}

static int interp_spline_workspace_init(interp_spline_workspace_t* w,
                                        const lmmc_interp_cspline_t* spline,
                                        lmmc_spline_bc_t bc)
{
    size_t i, n = spline->n;
    size_t arrays = bc == LMMC_SPLINE_PERIODIC ? 8 : 6;
    lmmc_real_t* storage = lmmc_memory_alloc_array_2d(n, arrays, sizeof(lmmc_real_t));
    if (!storage) {
        return 0;
    }
    w->n = n;
    w->ys = spline->ys;
    w->h = storage;
    w->moments = storage + n;
    w->sub = storage + 2 * n;
    w->dia = storage + 3 * n;
    w->sup = storage + 4 * n;
    w->rhs = storage + 5 * n;
    w->auxiliary = NULL;
    w->diagonal_copy = NULL;
    if (bc == LMMC_SPLINE_PERIODIC) {
        w->auxiliary = storage + 6 * n;
        w->diagonal_copy = storage + 7 * n;
    }
    for (i = 0; i < n - 1; ++i) w->h[i] = spline->xs[i + 1] - spline->xs[i];
    return 1;
}

lmmc_status_t lmmc_interp_cspline_create_ex(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n,
    lmmc_spline_bc_t bc, lmmc_real_t deriv_left, lmmc_real_t deriv_right,
    lmmc_interp_cspline_t** out_spline)
{
    lmmc_interp_cspline_t* spline;
    interp_spline_workspace_t work;
    if (!out_spline) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out_spline = NULL;
    if (!interp_check_samples(xs, ys, n, 3)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!interp_spline_check_boundary(ys, n, bc, deriv_left, deriv_right)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    spline = interp_spline_allocate(xs, ys, n);
    if (!spline) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    if (!interp_spline_workspace_init(&work, spline, bc)) {
        lmmc_interp_cspline_destroy(spline);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    switch (bc) {
    case LMMC_SPLINE_NOT_A_KNOT:
        interp_spline_not_a_knot(&work);
        break;
    case LMMC_SPLINE_PERIODIC:
        interp_spline_periodic(&work);
        break;
    default:
        interp_spline_standard(&work, bc, deriv_left, deriv_right);
        break;
    }
    cspline_compute_coeffs(spline, work.h, work.moments);
    lmmc_memory_free(work.h);
    if (!interp_check_finite_values(spline->coeffs, (n - 1) * 4)) {
        lmmc_interp_cspline_destroy(spline);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out_spline = spline;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_interp_cspline_create(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n,
    lmmc_interp_cspline_t** out_spline)
{
    return lmmc_interp_cspline_create_ex(xs, ys, n, LMMC_SPLINE_NATURAL, 0.0, 0.0, out_spline);
}
