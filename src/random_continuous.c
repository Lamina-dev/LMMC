#include <math.h>
#include "random_internal.h"

lmmc_status_t lmmc_rng_gamma(
    lmmc_rng_t* rng,
    lmmc_real_t shape,
    lmmc_real_t scale,
    lmmc_real_t* out)
{
    double d, c, x, v, u;

    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(shape) || !isfinite(scale) ||
        shape <= 0.0 || scale <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /** @brief shape < 1 时采用采样恒等式 Gamma(shape) = Gamma(shape+1) * U^(1/shape)。 */
    if (shape < 1.0) {
        lmmc_real_t g;
        lmmc_status_t st = lmmc_rng_gamma(rng, shape + 1.0, 1.0, &g);
        if (st != LMMC_STATUS_OK) {
            return st;
        }

        u = u64_to_double01(xoshiro256ss_next(rng->state));
        while (u == 0.0) {
            u = u64_to_double01(xoshiro256ss_next(rng->state));
        }
        *out = scale * g * pow(u, 1.0 / shape);
        return LMMC_STATUS_OK;
    }

    /** @brief shape >= 1 时使用 Marsaglia-Tsang 法。 */
    d = shape - 1.0 / 3.0;
    c = 1.0 / sqrt(9.0 * d);

    for (;;) {
        do {
            x = lmmc_rng_standard_normal(rng->state);
            v = 1.0 + c * x;
        } while (v <= 0.0);

        v = v * v * v;
        u = u64_to_double01(xoshiro256ss_next(rng->state));

        /** @brief 以挤压界快速判定接受。 */
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
            *out = scale * d * v;
            return LMMC_STATUS_OK;
        }

        /** @brief 按完整接受条件判定候选值。 */
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) {
            *out = scale * d * v;
            return LMMC_STATUS_OK;
        }
    }
}

/** @brief 小形状参数的幂以对数分量保留至比值计算。 */
static lmmc_status_t beta_log_gamma_parts(
    lmmc_rng_t* rng, lmmc_real_t shape,
    lmmc_real_t* log_base, lmmc_real_t* log_u)
{
    lmmc_real_t g;
    lmmc_status_t st = lmmc_rng_gamma(rng, shape < 1.0 ? shape + 1.0 : shape, 1.0, &g);
    if (st != LMMC_STATUS_OK) { return st; }
    if (!isfinite(g) || g <= 0.0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *log_base = log(g);
    *log_u = 0.0;
    if (shape < 1.0) {
        unsigned attempt;
        for (attempt = 0; attempt < 64; ++attempt) {
            const double u = u64_to_double01(xoshiro256ss_next(rng->state));
            if (u != 0.0) {
                *log_u = log(u);
                return LMMC_STATUS_OK;
            }
        }
        return LMMC_STATUS_CONVERGENCE_FAILED;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t beta_ratio_from_log_parts(
    lmmc_real_t alpha, lmmc_real_t beta_param,
    lmmc_real_t a, lmmc_real_t b, lmmc_real_t ua, lmmc_real_t ub,
    lmmc_real_t* out)
{
    lmmc_real_t m, ta, tb, difference, q, value;
    m = fmin(1.0, fmin(alpha, beta_param));
    ta = alpha < 1.0 ? (m / alpha) * ua : 0.0;
    tb = beta_param < 1.0 ? (m / beta_param) * ub : 0.0;
    /**
     * @brief 先相减再除法，保持次正规形状参数下的计算稳定性。
     * 基值差保留在缩放项外，以保留极小 m 下缩放项相等时的差异。
     */
    difference = (tb - ta) / m + (b - a);
    if (isnan(difference)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    if (difference >= 0.0) {
        q = exp(-difference);
        value = q / (1.0 + q);
    } else {
        q = exp(difference);
        value = 1.0 / (1.0 + q);
    }
    if (!isfinite(value) || value < 0.0 || value > 1.0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = value;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_beta(
    lmmc_rng_t* rng,
    lmmc_real_t alpha,
    lmmc_real_t beta_param,
    lmmc_real_t* out)
{
    lmmc_real_t a, b, ua, ub;
    lmmc_status_t st;

    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(alpha) || !isfinite(beta_param) ||
        alpha <= 0.0 || beta_param <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = beta_log_gamma_parts(rng, alpha, &a, &ua);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    st = beta_log_gamma_parts(rng, beta_param, &b, &ub);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    return beta_ratio_from_log_parts(alpha, beta_param, a, b, ua, ub, out);
}

lmmc_status_t lmmc_rng_chi_squared(
    lmmc_rng_t* rng,
    lmmc_real_t df,
    lmmc_real_t* out)
{
    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(df) || df <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (df / 2.0 == 0.0) {
        lmmc_real_t g;
        lmmc_status_t st = lmmc_rng_gamma(rng, 1.0, 2.0, &g);
        double u;
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        do {
            u = u64_to_double01(xoshiro256ss_next(rng->state));
        } while (u == 0.0);
        /* Keep the unrepresentable shape df/2 in the Gamma power exponent. */
        *out = g * exp(2.0 * log(u) / df);
        return LMMC_STATUS_OK;
    }

    return lmmc_rng_gamma(rng, df / 2.0, 2.0, out);
}

lmmc_status_t lmmc_rng_student_t(
    lmmc_rng_t* rng,
    lmmc_real_t df,
    lmmc_real_t* out)
{
    lmmc_real_t z, chi2, denom, value;
    lmmc_status_t st;

    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(df) || df <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    z = lmmc_rng_standard_normal(rng->state);

    st = lmmc_rng_chi_squared(rng, df, &chi2);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    denom = sqrt(chi2 / df);
    if (denom == 0.0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    value = z / denom;
    if (!isfinite(value)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = value;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_f(
    lmmc_rng_t* rng,
    lmmc_real_t df1,
    lmmc_real_t df2,
    lmmc_real_t* out)
{
    lmmc_real_t chi1, chi2, denominator, value;
    lmmc_status_t st;

    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(df1) || !isfinite(df2) ||
        df1 <= 0.0 || df2 <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_rng_chi_squared(rng, df1, &chi1);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    st = lmmc_rng_chi_squared(rng, df2, &chi2);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    denominator = chi2 / df2;
    if (denominator == 0.0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    value = (chi1 / df1) / denominator;
    if (!isfinite(value)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = value;
    return LMMC_STATUS_OK;
}
