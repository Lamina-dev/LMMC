#include "internal/quadrature_internal.h"

/**
 * @brief Gauss-Hermite 求积，权函数 exp(-x^2)，积分域 (-inf, +inf)。
 *
 * Golub-Welsch 算法以 Hermite 多项式的对称三对角 Jacobi 矩阵特征值为节点，
 * 以特征向量的首分量计算权重。
 * 物理学家形式的 Hermite 多项式满足 x H_n = H_{n+1}/2 + n H_{n-1}；
 * Jacobi 系数为 a_i = 0、b_i = sqrt(i/2)，i=1..n-1。
 */

#define LMMC_GH_MAX_ORDER 20

static size_t lmmc_quad_tridiag_block_end(
    const lmmc_real_t* diag, const lmmc_real_t* subdiag, size_t first, size_t n)
{
    size_t last = first;
    while (last < n - 1) {
        const lmmc_real_t dd = lmmc_abs(diag[last]) + lmmc_abs(diag[last + 1]);
        if (lmmc_abs(subdiag[last]) + dd == dd) break;
        ++last;
    }
    return last;
}

static void lmmc_quad_tridiag_sweep(
    lmmc_real_t* diag, lmmc_real_t* subdiag, lmmc_real_t* z,
    size_t first, size_t last)
{
    lmmc_real_t g = (diag[first + 1] - diag[first]) / (2.0 * subdiag[first]);
    lmmc_real_t r = sqrt(g * g + 1.0);
    const lmmc_real_t sign_g = g >= 0.0 ? 1.0 : -1.0;
    lmmc_real_t s = 1.0, c = 1.0, p = 0.0;
    g = diag[last] - diag[first] + subdiag[first] / (g + sign_g * r);
    for (size_t i = last; i > first; --i) {
        const lmmc_real_t fi = s * subdiag[i - 1];
        const lmmc_real_t bi = c * subdiag[i - 1];
        const lmmc_real_t zi = z[i];
        if (lmmc_abs(fi) >= lmmc_abs(g)) {
            c = g / fi;
            r = sqrt(c * c + 1.0);
            subdiag[i] = fi * r;
            s = 1.0 / r;
            c *= s;
        } else {
            s = fi / g;
            r = sqrt(s * s + 1.0);
            subdiag[i] = g * r;
            c = 1.0 / r;
            s *= c;
        }
        g = diag[i] - p;
        r = (diag[i - 1] - g) * s + 2.0 * c * bi;
        p = s * r;
        diag[i] = g + p;
        g = c * r - bi;
        z[i] = s * z[i - 1] + c * zi;
        z[i - 1] = c * z[i - 1] - s * zi;
    }
    diag[first] -= p;
    subdiag[first] = g;
}

static lmmc_status_t tridiag_eigvals_weights(
    lmmc_real_t* diag, lmmc_real_t* subdiag,
    lmmc_real_t* weights, size_t n, lmmc_real_t mu0,
    size_t iteration_limit)
{
    lmmc_real_t z[LMMC_GH_MAX_ORDER];
    for (size_t i = 0; i < n; i++) z[i] = (i == 0) ? 1.0 : 0.0;

    for (size_t l = 0; l < n; l++) {
        size_t iterations = 0;
        while (1) {
            const size_t m =
                lmmc_quad_tridiag_block_end(diag, subdiag, l, n);
            if (m == l) break;
            if (iterations++ >= iteration_limit) {
                return LMMC_STATUS_CONVERGENCE_FAILED;
            }
            lmmc_quad_tridiag_sweep(diag, subdiag, z, l, m);
            if (m < n - 1) subdiag[m] = 0.0;
        }
    }

    for (size_t i = 0; i < n; i++) {
        weights[i] = mu0 * z[i] * z[i];
    }
    return LMMC_STATUS_OK;
}

#ifdef LMMC_BUILD_TESTS
lmmc_status_t lmmc_quad_gauss_hermite_with_iteration_limit_for_test(
    lmmc_quad_func_t f, void* ud, size_t order,
    size_t iteration_limit, lmmc_real_t* out);
#endif

static lmmc_status_t lmmc_quad_gauss_hermite_impl(
    lmmc_quad_func_t f, void* ud, size_t order,
    size_t iteration_limit, lmmc_real_t* out)
{
    lmmc_real_t diag[LMMC_GH_MAX_ORDER];
    lmmc_real_t subdiag[LMMC_GH_MAX_ORDER];
    lmmc_real_t weights[LMMC_GH_MAX_ORDER];
    lmmc_real_t sum = 0.0;

    if (f == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (order < 1 || order > LMMC_GH_MAX_ORDER) {
        *out = 0.0;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < order; i++) {
        diag[i] = 0.0;
    }
    for (size_t i = 0; i < order - 1; i++) {
        subdiag[i] = sqrt((lmmc_real_t)(i + 1) / 2.0);
    }

    const lmmc_status_t eigensolver_status = tridiag_eigvals_weights(
        diag, subdiag, weights, order, sqrt(LMMC_CONST_PI),
        iteration_limit);
    if (eigensolver_status != LMMC_STATUS_OK) {
        return eigensolver_status;
    }

    *out = 0.0;
    for (size_t i = 0; i < order; i++) {
        const lmmc_real_t value = f(diag[i], ud);
        if (!isfinite(value)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        sum += weights[i] * value;
        if (!isfinite(sum)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    *out = sum;
    return LMMC_STATUS_OK;
}

#ifdef LMMC_BUILD_TESTS
lmmc_status_t lmmc_quad_gauss_hermite_with_iteration_limit_for_test(
    lmmc_quad_func_t f, void* ud, size_t order,
    size_t iteration_limit, lmmc_real_t* out)
{
    return lmmc_quad_gauss_hermite_impl(
        f, ud, order, iteration_limit, out);
}
#endif

lmmc_status_t lmmc_quad_gauss_hermite(
    lmmc_quad_func_t f, void* ud,
    size_t order, lmmc_real_t* out)
{
    return lmmc_quad_gauss_hermite_impl(f, ud, order, 100, out);
}


/**
 * @brief Gauss-Laguerre 求积，权函数 exp(-x)，积分域 [0, +inf)。
 *
 * Golub-Welsch 算法使用首一 Laguerre 多项式的三项递推：
 * x L_n(x) = L_{n+1}(x) + (2n+1) L_n(x) - n^2 L_{n-1}(x)。
 * Jacobi 系数为 alpha_i = 2*i + 1、beta_i = i（i=1..n-1）；
 * mu_0 为 exp(-x) 在 [0, inf) 上的积分，等于 1。
 */

#define LMMC_GL_LAG_MAX_ORDER 20

lmmc_status_t lmmc_quad_gauss_laguerre(
    lmmc_quad_func_t f, void* ud,
    size_t order, lmmc_real_t* out)
{
    lmmc_real_t diag[LMMC_GL_LAG_MAX_ORDER];
    lmmc_real_t subdiag[LMMC_GL_LAG_MAX_ORDER];
    lmmc_real_t weights[LMMC_GL_LAG_MAX_ORDER];
    lmmc_real_t sum = 0.0;

    if (f == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out = 0.0;
    if (order < 1 || order > LMMC_GL_LAG_MAX_ORDER) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < order; i++) {
        diag[i] = 2.0 * (lmmc_real_t)i + 1.0;
    }
    for (size_t i = 0; i < order - 1; i++) {
        subdiag[i] = (lmmc_real_t)(i + 1);
    }

    const lmmc_status_t eigensolver_status = tridiag_eigvals_weights(
        diag, subdiag, weights, order, 1.0, 100);
    if (eigensolver_status != LMMC_STATUS_OK) {
        return eigensolver_status;
    }

    for (size_t i = 0; i < order; i++) {
        const lmmc_real_t value = f(diag[i], ud);
        if (!isfinite(value)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        sum += weights[i] * value;
        if (!isfinite(sum)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    *out = sum;
    return LMMC_STATUS_OK;
}
