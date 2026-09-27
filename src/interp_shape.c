#include "memory_bridge.h"
#include "internal.h"
#include "internal/interp_internal.h"

#include <math.h>
#include <string.h>

struct lmmc_interp_pchip_t {
    size_t n;
    lmmc_real_t* xs;
    lmmc_real_t* ys;
    lmmc_real_t* d; /**< 各节点导数。 */
};

void lmmc_interp_pchip_destroy(lmmc_interp_pchip_t* p)
{
    if (!p) {
        return;
    }
    lmmc_memory_free(p->xs);
    lmmc_memory_free(p->ys);
    lmmc_memory_free(p->d);
    lmmc_memory_free(p);
}

static lmmc_real_t interp_pchip_endpoint(lmmc_real_t h1, lmmc_real_t h2,
                                          lmmc_real_t slope1, lmmc_real_t slope2)
{
    lmmc_real_t derivative = ((2.0 * h1 + h2) * slope1 - h1 * slope2) / (h1 + h2);
    if (derivative * slope1 <= 0.0) {
        return 0.0;
    }
    if (slope1 * slope2 <= 0.0 && fabs(derivative) > 3.0 * fabs(slope1)) {
        return 3.0 * slope1;
    }
    return derivative;
}

static void interp_pchip_derivatives(lmmc_interp_pchip_t* p,
                                     const lmmc_real_t* delta)
{
    const lmmc_real_t* xs = p->xs;
    size_t i, n = p->n;
    if (n == 2) {
        p->d[0] = p->d[1] = delta[0];
        return;
    }
    for (i = 1; i < n - 1; ++i) {
        if (delta[i - 1] * delta[i] <= 0.0) {
            p->d[i] = 0.0;
        } else {
            lmmc_real_t h1 = xs[i] - xs[i - 1];
            lmmc_real_t h2 = xs[i + 1] - xs[i];
            lmmc_real_t w1 = 2.0 * h2 + h1;
            lmmc_real_t w2 = h2 + 2.0 * h1;
            p->d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
        }
    }
    p->d[0] = interp_pchip_endpoint(xs[1] - xs[0], xs[2] - xs[1], delta[0], delta[1]);
    p->d[n - 1] = interp_pchip_endpoint(xs[n - 1] - xs[n - 2],
        xs[n - 2] - xs[n - 3], delta[n - 2], delta[n - 3]);
}

lmmc_status_t lmmc_interp_pchip_create(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n,
    lmmc_interp_pchip_t** out)
{
    lmmc_interp_pchip_t* p = NULL;
    lmmc_real_t* delta = NULL; /**< 各段斜率。 */
    size_t i;

    if (!out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (!interp_check_samples(xs, ys, n, 2)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    p = (lmmc_interp_pchip_t*)lmmc_memory_alloc(sizeof(*p));
    if (!p) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    p->n = n; p->xs = NULL; p->ys = NULL; p->d = NULL;

    p->xs = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    p->ys = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    p->d = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    delta = (lmmc_real_t*)lmmc_memory_alloc_array(n - 1, sizeof(lmmc_real_t));
    if (!p->xs || !p->ys || !p->d || !delta) {
        lmmc_memory_free(delta);
        lmmc_interp_pchip_destroy(p);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(p->xs, xs, n * sizeof(lmmc_real_t));
    memcpy(p->ys, ys, n * sizeof(lmmc_real_t));

    for (i = 0; i < n - 1; i++) {
        delta[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    }

    interp_pchip_derivatives(p, delta);
    if (!interp_check_finite_values(delta, n - 1) ||
        !interp_check_finite_values(p->d, n)) {
        lmmc_memory_free(delta);
        lmmc_interp_pchip_destroy(p);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    lmmc_memory_free(delta);
    *out = p;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_interp_pchip_eval(
    const lmmc_interp_pchip_t* p, lmmc_real_t x, lmmc_real_t* out_y)
{
    size_t seg;
    lmmc_real_t h, t, a, b, result;
    if (!p || !out_y || !isfinite(x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x < p->xs[0] || x > p->xs[p->n - 1]) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    if (x == p->xs[p->n - 1]) {
        *out_y = p->ys[p->n - 1];
        return LMMC_STATUS_OK;
    }

    seg = interp_find_interval(p->xs, p->n, x);
    h = p->xs[seg + 1] - p->xs[seg];
    t = (x - p->xs[seg]) / h;

    a = 1.0 - t;
    b = t;
    result = a * a * (1.0 + 2.0 * t) * p->ys[seg]
           + b * b * (3.0 - 2.0 * t) * p->ys[seg + 1]
           + t * a * a * h * p->d[seg]
           - b * b * a * h * p->d[seg + 1];
    if (!isfinite(result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out_y = result;
    return LMMC_STATUS_OK;
}

struct lmmc_interp_akima_t {
    size_t n;
    lmmc_real_t* xs;
    lmmc_real_t* ys;
    lmmc_real_t* d; /**< 各节点导数。 */
};

void lmmc_interp_akima_destroy(lmmc_interp_akima_t* a)
{
    if (!a) {
        return;
    }
    lmmc_memory_free(a->xs);
    lmmc_memory_free(a->ys);
    lmmc_memory_free(a->d);
    lmmc_memory_free(a);
}

static void interp_akima_derivatives(lmmc_interp_akima_t* a, lmmc_real_t* m)
{
    size_t i, n = a->n, nm1 = n - 1;
    const lmmc_real_t* xs = a->xs;
    const lmmc_real_t* ys = a->ys;
    /** @brief 将第 i 段斜率 (y[i+1]-y[i])/(x[i+1]-x[i]) 存于扩展数组 i+2 处，i=0..n-2。 */
    for (i = 0; i < nm1; i++) {
        m[i + 2] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    }

    /**
     * @brief 按 Akima 外推公式扩展边界斜率。
     * m[-1] = 2*m[0] - m[1]，m[-2] = 2*m[-1] - m[0]；
     * m[n-1] = 2*m[n-2] - m[n-3]，m[n] = 2*m[n-1] - m[n-2]。
     */
    m[1] = 2.0 * m[2] - m[3];       /**< 原斜率 m[-1]。 */
    m[0] = 2.0 * m[1] - m[2];       /**< 原斜率 m[-2]。 */
    m[nm1 + 2] = 2.0 * m[nm1 + 1] - m[nm1]; /**< 原斜率 m[n-1]。 */
    m[nm1 + 3] = 2.0 * m[nm1 + 2] - m[nm1 + 1]; /**< 原斜率 m[n]。 */

    /**
     * @brief 由扩展斜率计算 Akima 导数，m_ext[i+2] 对应第 i 段斜率。
     * w1 = |m_ext[i+3] - m_ext[i+2]|，w2 = |m_ext[i+1] - m_ext[i]|；
     * d[i] = (w1*m_ext[i+1] + w2*m_ext[i+2]) / (w1+w2)。
     */
    for (i = 0; i < n; i++) {
        lmmc_real_t w1 = fabs(m[i + 3] - m[i + 2]);
        lmmc_real_t w2 = fabs(m[i + 1] - m[i]);
        if (w1 + w2 < 1e-30) {
            /** @brief 斜率变化可忽略时取相邻斜率均值。 */
            a->d[i] = 0.5 * (m[i + 1] + m[i + 2]);
        } else {
            a->d[i] = (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2);
        }
    }
}

lmmc_status_t lmmc_interp_akima_create(
    const lmmc_real_t* xs, const lmmc_real_t* ys, size_t n,
    lmmc_interp_akima_t** out)
{
    lmmc_interp_akima_t* a = NULL;
    lmmc_real_t* m = NULL; /**< 扩展斜率 m[-2]..m[n] 存于索引 0..n+2。 */
    size_t slope_count;

    if (!out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (!interp_check_samples(xs, ys, n, 5)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_safe_add_size(n, 3, &slope_count)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    a = (lmmc_interp_akima_t*)lmmc_memory_alloc(sizeof(*a));
    if (!a) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    a->n = n; a->xs = NULL; a->ys = NULL; a->d = NULL;

    a->xs = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    a->ys = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    a->d = (lmmc_real_t*)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    m = (lmmc_real_t*)lmmc_memory_alloc_array(slope_count, sizeof(lmmc_real_t));
    if (!a->xs || !a->ys || !a->d || !m) {
        lmmc_memory_free(m);
        lmmc_interp_akima_destroy(a);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(a->xs, xs, n * sizeof(lmmc_real_t));
    memcpy(a->ys, ys, n * sizeof(lmmc_real_t));

    interp_akima_derivatives(a, m);
    if (!interp_check_finite_values(m, slope_count) ||
        !interp_check_finite_values(a->d, n)) {
        lmmc_memory_free(m);
        lmmc_interp_akima_destroy(a);
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    lmmc_memory_free(m);
    *out = a;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_interp_akima_eval(
    const lmmc_interp_akima_t* a, lmmc_real_t x, lmmc_real_t* out_y)
{
    size_t seg;
    lmmc_real_t h, t, omt, result;
    if (!a || !out_y || !isfinite(x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x < a->xs[0] || x > a->xs[a->n - 1]) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    if (x == a->xs[a->n - 1]) {
        *out_y = a->ys[a->n - 1];
        return LMMC_STATUS_OK;
    }

    seg = interp_find_interval(a->xs, a->n, x);
    h = a->xs[seg + 1] - a->xs[seg];
    t = (x - a->xs[seg]) / h;
    omt = 1.0 - t;

    result = omt * omt * (1.0 + 2.0 * t) * a->ys[seg]
           + t * t * (3.0 - 2.0 * t) * a->ys[seg + 1]
           + t * omt * omt * h * a->d[seg]
           - t * t * omt * h * a->d[seg + 1];
    if (!isfinite(result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out_y = result;
    return LMMC_STATUS_OK;
}
