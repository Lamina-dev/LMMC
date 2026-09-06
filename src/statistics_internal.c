#include "statistics_internal.h"

#define SPECIAL_FUNCTION_MAX_ITERATIONS 1000
#define SPECIAL_FUNCTION_TOLERANCE 1e-15
#define LENTZ_TINY 1e-300

lmmc_real_t lmmc_stirling_error(lmmc_real_t x) {
    if (x < 16.0) {
        return lgamma(x + 1.0) -
               (x + 0.5) * log(x) + x -
               LMMC_LOG_SQRT_2PI;
    }

    {
        const lmmc_real_t inverse = 1.0 / x;
        const lmmc_real_t inverse_squared = inverse * inverse;
        return inverse *
               (1.0 / 12.0 -
                inverse_squared *
                    (1.0 / 360.0 -
                     inverse_squared *
                         (1.0 / 1260.0 -
                          inverse_squared / 1680.0)));
    }
}


lmmc_real_t lmmc_log_gamma_stirling_error(lmmc_real_t x) {
    if (x < 16.0) {
        return lgamma(x) -
               (x - 0.5) * log(x) + x -
               LMMC_LOG_SQRT_2PI;
    }

    {
        const lmmc_real_t inverse = 1.0 / x;
        const lmmc_real_t inverse_squared = inverse * inverse;
        return inverse *
               (1.0 / 12.0 -
                inverse_squared *
                    (1.0 / 360.0 -
                     inverse_squared *
                         (1.0 / 1260.0 -
                          inverse_squared / 1680.0)));
    }
}
lmmc_real_t lmmc_deviance_part(
    lmmc_real_t x, lmmc_real_t mean) {
    if (x == mean) return 0.0;
    if (x == 0.0) return mean;
    if (mean == 0.0) return INFINITY;

    if (fabs(x - mean) < 0.1 * (x + mean)) {
        const lmmc_real_t difference = x - mean;
        lmmc_real_t ratio = difference / (x + mean);
        const lmmc_real_t ratio_squared = ratio * ratio;
        lmmc_real_t sum = 0.5 * difference * ratio;
        lmmc_real_t term = x * ratio;
        size_t order;

        for (order = 1; order < 100; ++order) {
            const lmmc_real_t previous = sum;
            term *= ratio_squared;
            sum += term / (lmmc_real_t)(2 * order + 1);
            if (sum == previous) break;
        }
        return 2.0 * sum;
    }

    {
        const lmmc_real_t quotient = x / mean;
        const lmmc_real_t log_ratio =
            isfinite(quotient) ? log(quotient) :
                                 log(x) - log(mean);
        return x > mean ?
                   x * (log_ratio - 1.0) + mean :
                   x * log_ratio + mean - x;
    }
}

static lmmc_status_t store_regularized_probability(
    lmmc_real_t value, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(value) || value < 0.0 || value > 1.0)
        return LMMC_STATUS_NUMERICAL_FAILURE;
    *out = value;
    return LMMC_STATUS_OK;
}

/*
 * DLMF 8.12.3: leading uniform large-a expansion for Q(a, x).
 * The continued fraction and power series require O(sqrt(a)) work around
 * x == a; this expansion is both bounded-cost and stable in that region.
 */
static int regularized_gamma_large_central_upper(
    lmmc_real_t a, lmmc_real_t x, lmmc_real_t* upper) {
    lmmc_real_t delta, eta, eta_squared, c0, correction, value;

    if (a < 10000.0 || x <= 0.0) return 0;
    delta = x / a - 1.0;
    if (fabs(delta) > 0.1) return 0;

    eta_squared = 2.0 * (delta - log1p(delta));
    if (eta_squared < 0.0) {
        if (eta_squared > -DBL_EPSILON) {
            eta_squared = 0.0;
        } else {
            return 0;
        }
    }
    eta = copysign(sqrt(eta_squared), delta);
    if (fabs(eta) * sqrt(a) > 8.0) return 0;

    if (fabs(eta) < 1e-3) {
        c0 = -1.0 / 3.0 + eta / 12.0 -
             2.0 * eta * eta / 135.0;
    } else {
        c0 = 1.0 / delta - 1.0 / eta;
    }
    correction =
        exp(-0.5 * a * eta_squared) * c0 /
        (LMMC_SQRT_2PI * sqrt(a));
    value = 0.5 * erfc(eta * sqrt(0.5 * a)) + correction;
    if (!isfinite(value) || value < 0.0 || value > 1.0) return 0;

    *upper = value;
    return 1;
}

/**
 * @brief 正则化不完全伽马函数下尾 P(a, x) = gamma(a, x) / Gamma(a)。
 * 使用级数展开或连分数展开。
 */
lmmc_status_t regularized_gamma_lower(
    lmmc_real_t a, lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) return store_regularized_probability(0.0, out);
    {
        lmmc_real_t upper;
        if (regularized_gamma_large_central_upper(a, x, &upper))
            return store_regularized_probability(1.0 - upper, out);
    }
    if (x < a + 1.0) {
        lmmc_real_t sum = 1.0 / a;
        lmmc_real_t term = sum;
        lmmc_real_t ap = a;
        int i;
        for (i = 0; i < SPECIAL_FUNCTION_MAX_ITERATIONS; ++i) {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if (!isfinite(term) || !isfinite(sum))
                return LMMC_STATUS_NUMERICAL_FAILURE;
            if (fabs(term) < fabs(sum) * SPECIAL_FUNCTION_TOLERANCE)
                break;
        }
        if (i == SPECIAL_FUNCTION_MAX_ITERATIONS)
            return LMMC_STATUS_CONVERGENCE_FAILED;
        return store_regularized_probability(
            sum * exp(-x + a * log(x) - lgamma(a)), out);
    } else {
        lmmc_real_t upper;
        lmmc_status_t status = regularized_gamma_upper_cf(a, x, &upper);
        if (status != LMMC_STATUS_OK) return status;
        return store_regularized_probability(1.0 - upper, out);
    }
}

/**
 * @brief 正则化不完全伽马函数上尾 Q(a, x) = 1 - P(a, x)。
 * 使用 Lentz 连分数展开。
 */
lmmc_status_t regularized_gamma_upper_cf(
    lmmc_real_t a, lmmc_real_t x, lmmc_real_t* out) {
    lmmc_real_t f, c, d, delta;
    int i;

    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    f = x - a + 1.0;
    if (!isfinite(f)) return LMMC_STATUS_NUMERICAL_FAILURE;
    if (fabs(f) < LENTZ_TINY) f = copysign(LENTZ_TINY, f == 0.0 ? 1.0 : f);
    c = f;
    d = 0.0;

    for (i = 1; i <= SPECIAL_FUNCTION_MAX_ITERATIONS; ++i) {
        lmmc_real_t an = (lmmc_real_t)i * (a - (lmmc_real_t)i);
        lmmc_real_t bn = x - a + 2.0 * (lmmc_real_t)i + 1.0;
        if (!isfinite(an) || !isfinite(bn))
            return LMMC_STATUS_NUMERICAL_FAILURE;

        d = bn + an * d;
        if (fabs(d) < LENTZ_TINY)
            d = copysign(LENTZ_TINY, d == 0.0 ? 1.0 : d);
        d = 1.0 / d;
        c = bn + an / c;
        if (fabs(c) < LENTZ_TINY)
            c = copysign(LENTZ_TINY, c == 0.0 ? 1.0 : c);
        delta = c * d;
        f *= delta;
        if (!isfinite(c) || !isfinite(d) || !isfinite(delta) || !isfinite(f))
            return LMMC_STATUS_NUMERICAL_FAILURE;
        if (fabs(delta - 1.0) < SPECIAL_FUNCTION_TOLERANCE)
            break;
    }
    if (i > SPECIAL_FUNCTION_MAX_ITERATIONS)
        return LMMC_STATUS_CONVERGENCE_FAILED;
    return store_regularized_probability(
        exp(-x + a * log(x) - lgamma(a)) / f, out);
}

/**
 * @brief 正则化不完全伽马函数上尾 Q(a, x)。
 */
lmmc_status_t regularized_gamma_upper(
    lmmc_real_t a, lmmc_real_t x, lmmc_real_t* out) {
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) return store_regularized_probability(1.0, out);
    {
        lmmc_real_t upper;
        if (regularized_gamma_large_central_upper(a, x, &upper))
            return store_regularized_probability(upper, out);
    }
    if (x < a + 1.0) {
        lmmc_real_t lower;
        lmmc_status_t status = regularized_gamma_lower(a, x, &lower);
        if (status != LMMC_STATUS_OK) return status;
        return store_regularized_probability(1.0 - lower, out);
    }
    return regularized_gamma_upper_cf(a, x, out);
}

/**
 * @brief 正则化不完全贝塔函数的 Lentz 连分数。
 */
lmmc_status_t regularized_beta_cf(
    lmmc_real_t x, lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out) {
    lmmc_real_t qab = a + b;
    lmmc_real_t qap = a + 1.0;
    lmmc_real_t qam = a - 1.0;
    lmmc_real_t c = 1.0;
    lmmc_real_t d, f;
    int m;

    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!isfinite(qab) || !isfinite(qap) || !isfinite(qam))
        return LMMC_STATUS_NUMERICAL_FAILURE;
    d = 1.0 - qab * x / qap;
    if (fabs(d) < LENTZ_TINY)
        d = copysign(LENTZ_TINY, d == 0.0 ? 1.0 : d);
    d = 1.0 / d;
    f = d;

    for (m = 1; m <= SPECIAL_FUNCTION_MAX_ITERATIONS; ++m) {
        lmmc_real_t aa, del;
        int m2 = 2 * m;
        aa = (lmmc_real_t)m * (b - (lmmc_real_t)m) * x /
             ((qam + (lmmc_real_t)m2) * (a + (lmmc_real_t)m2));
        d = 1.0 + aa * d;
        if (fabs(d) < LENTZ_TINY)
            d = copysign(LENTZ_TINY, d == 0.0 ? 1.0 : d);
        c = 1.0 + aa / c;
        if (fabs(c) < LENTZ_TINY)
            c = copysign(LENTZ_TINY, c == 0.0 ? 1.0 : c);
        d = 1.0 / d;
        f *= c * d;

        aa = -(a + (lmmc_real_t)m) * (qab + (lmmc_real_t)m) * x /
             ((a + (lmmc_real_t)m2) * (qap + (lmmc_real_t)m2));
        d = 1.0 + aa * d;
        if (fabs(d) < LENTZ_TINY)
            d = copysign(LENTZ_TINY, d == 0.0 ? 1.0 : d);
        c = 1.0 + aa / c;
        if (fabs(c) < LENTZ_TINY)
            c = copysign(LENTZ_TINY, c == 0.0 ? 1.0 : c);
        d = 1.0 / d;
        del = c * d;
        f *= del;
        if (!isfinite(aa) || !isfinite(c) || !isfinite(d) ||
            !isfinite(del) || !isfinite(f))
            return LMMC_STATUS_NUMERICAL_FAILURE;
        if (fabs(del - 1.0) < SPECIAL_FUNCTION_TOLERANCE)
            break;
    }
    if (m > SPECIAL_FUNCTION_MAX_ITERATIONS)
        return LMMC_STATUS_CONVERGENCE_FAILED;
    if (!isfinite(f)) return LMMC_STATUS_NUMERICAL_FAILURE;
    *out = f;
    return LMMC_STATUS_OK;
}

/**
 * @brief 正则化不完全贝塔函数 I_x(a, b)。
 */
lmmc_status_t regularized_beta(
    lmmc_real_t x, lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out) {
    lmmc_real_t log_bt, bt, fraction, result;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= 0.0) return store_regularized_probability(0.0, out);
    if (x >= 1.0) return store_regularized_probability(1.0, out);
    if (x == 0.5 && a == b)
        return store_regularized_probability(0.5, out);

    log_bt = lgamma(a + b) - lgamma(a) - lgamma(b) +
             a * log(x) + b * log1p(-x);
    if (isnan(log_bt) || log_bt > log(DBL_MAX))
        return LMMC_STATUS_NUMERICAL_FAILURE;
    bt = exp(log_bt);
    if (!isfinite(bt)) return LMMC_STATUS_NUMERICAL_FAILURE;

    if (x < (a + 1.0) / (a + b + 2.0)) {
        status = regularized_beta_cf(x, a, b, &fraction);
        if (status != LMMC_STATUS_OK) return status;
        result = bt * fraction / a;
    } else {
        status = regularized_beta_cf(1.0 - x, b, a, &fraction);
        if (status != LMMC_STATUS_OK) return status;
        result = 1.0 - bt * fraction / b;
    }
    return store_regularized_probability(result, out);
}
