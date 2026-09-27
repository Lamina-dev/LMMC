#include "statistics_internal.h"

#define SPECIAL_FUNCTION_MAX_ITERATIONS 1000
#define SPECIAL_FUNCTION_TOLERANCE 1e-15
#define LENTZ_TINY 1e-300

static lmmc_real_t lentz_nonzero(lmmc_real_t value) {
    if (fabs(value) < LENTZ_TINY) {
        return copysign(LENTZ_TINY, value == 0.0 ? 1.0 : value);
    }
    return value;
}

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
    if (x == mean) { return 0.0; }
    if (x == 0.0) { return mean; }
    if (mean == 0.0) { return INFINITY; }

    if (fabs(x - mean) < 0.1 * x + 0.1 * mean) {
        const lmmc_real_t difference = x - mean;
        /** @brief 先除后加，在缩放域计算峰值附近的比值。 */
        const lmmc_real_t ratio = (difference / x) / (1.0 + mean / x);
        const lmmc_real_t ratio_squared = ratio * ratio;
        lmmc_real_t sum = 0.5 * difference * ratio;
        lmmc_real_t term = x * ratio;
        size_t order;

        for (order = 1; order < 100; ++order) {
            const lmmc_real_t previous = sum;
            term *= ratio_squared;
            sum += term / (lmmc_real_t)(2 * order + 1);
            if (sum == previous) { break; }
        }
        return 2.0 * sum;
    }

    {
        const lmmc_real_t quotient = x / mean;
        const lmmc_real_t log_ratio =
            quotient > 0.0 && isfinite(quotient) ? log(quotient) :
                                                 log(x) - log(mean);
        return x > mean ?
                   x * (log_ratio - 1.0) + mean :
                   x * log_ratio + mean - x;
    }
}

static lmmc_status_t store_regularized_probability(
    lmmc_real_t value, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(value) || value < 0.0 || value > 1.0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
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

    if (a < 10000.0 || x <= 0.0) { return 0; }
    delta = x / a - 1.0;
    if (fabs(delta) > 0.1) { return 0; }

    eta_squared = 2.0 * (delta - log1p(delta));
    if (eta_squared < 0.0) {
        if (eta_squared > -DBL_EPSILON) {
            eta_squared = 0.0;
        } else {
            return 0;
        }
    }
    eta = copysign(sqrt(eta_squared), delta);
    if (fabs(eta) * sqrt(a) > 8.0) { return 0; }

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
    if (!isfinite(value) || value < 0.0 || value > 1.0) { return 0; }

    *upper = value;
    return 1;
}

/**
 * @brief 正则化不完全伽马函数下尾 P(a, x) = gamma(a, x) / Gamma(a)。
 * 使用级数展开或连分数展开。
 */
lmmc_status_t regularized_gamma_lower(
    lmmc_real_t a, lmmc_real_t x, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= 0.0) { return store_regularized_probability(0.0, out); }
    {
        lmmc_real_t upper;
        if (regularized_gamma_large_central_upper(a, x, &upper)) { return store_regularized_probability(1.0 - upper, out); }
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
            if (!isfinite(term) || !isfinite(sum)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
            if (fabs(term) < fabs(sum) * SPECIAL_FUNCTION_TOLERANCE) { break; }
        }
        if (i == SPECIAL_FUNCTION_MAX_ITERATIONS) { return LMMC_STATUS_CONVERGENCE_FAILED; }
        return store_regularized_probability(
            sum * exp(-x + a * log(x) - lgamma(a)), out);
    } else {
        lmmc_real_t upper;
        lmmc_status_t status = regularized_gamma_upper_cf(a, x, &upper);
        if (status != LMMC_STATUS_OK) { return status; }
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

    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    f = x - a + 1.0;
    if (!isfinite(f)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    f = lentz_nonzero(f);
    c = f;
    d = 0.0;

    for (i = 1; i <= SPECIAL_FUNCTION_MAX_ITERATIONS; ++i) {
        lmmc_real_t an = (lmmc_real_t)i * (a - (lmmc_real_t)i);
        lmmc_real_t bn = x - a + 2.0 * (lmmc_real_t)i + 1.0;
        if (!isfinite(an) || !isfinite(bn)) { return LMMC_STATUS_NUMERICAL_FAILURE; }

        d = bn + an * d;
        d = lentz_nonzero(d);
        d = 1.0 / d;
        c = bn + an / c;
        c = lentz_nonzero(c);
        delta = c * d;
        f *= delta;
        if (!isfinite(c) || !isfinite(d) || !isfinite(delta) || !isfinite(f)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        if (fabs(delta - 1.0) < SPECIAL_FUNCTION_TOLERANCE) { break; }
    }
    if (i > SPECIAL_FUNCTION_MAX_ITERATIONS) { return LMMC_STATUS_CONVERGENCE_FAILED; }
    return store_regularized_probability(
        exp(-x + a * log(x) - lgamma(a)) / f, out);
}

/**
 * @brief 正则化不完全伽马函数上尾 Q(a, x)。
 */
lmmc_status_t regularized_gamma_upper(
    lmmc_real_t a, lmmc_real_t x, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= 0.0) { return store_regularized_probability(1.0, out); }
    {
        lmmc_real_t upper;
        if (regularized_gamma_large_central_upper(a, x, &upper)) { return store_regularized_probability(upper, out); }
    }
    if (x < a + 1.0) {
        lmmc_real_t lower;
        lmmc_status_t status = regularized_gamma_lower(a, x, &lower);
        if (status != LMMC_STATUS_OK) { return status; }
        return store_regularized_probability(1.0 - lower, out);
    }
    return regularized_gamma_upper_cf(a, x, out);
}

/**
 * @brief 正则化不完全贝塔函数的 Lentz 连分数。
 */
static lmmc_status_t regularized_beta_cf(
    lmmc_real_t x, lmmc_real_t complement_x,
    lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out) {
    const lmmc_real_t qap = a + 1.0;
    const lmmc_real_t qam = a - 1.0;
    lmmc_real_t c = 1.0;
    lmmc_real_t d, f;
    int m;

    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    d = complement_x + (1.0 - b) * x / qap;
    d = lentz_nonzero(d);
    d = 1.0 / d;
    f = d;

    for (m = 1; m <= SPECIAL_FUNCTION_MAX_ITERATIONS; ++m) {
        const lmmc_real_t order = (lmmc_real_t)m;
        const lmmc_real_t m2 = 2.0 * order;
        const lmmc_real_t even =
            ((b - order) * x / (a + m2)) * (order / (qam + m2));
        /**
         * @brief 合并两个 Lentz 步骤后再舍入单位项。
         * x 接近 1 时奇数项系数接近 -1，直接计算 1 + odd 以保留原始微小补量。
         */
        const lmmc_real_t one_plus_odd =
            complement_x +
            (a / (a + m2)) * ((1.0 + m2 - b) * x / (qap + m2)) +
            (order / (a + m2)) *
                ((2.0 + 3.0 * order - b) * x / (qap + m2));
        const lmmc_real_t c_offset = even / c;
        const lmmc_real_t d_offset = even * d;
        lmmc_real_t c_numerator = one_plus_odd + c_offset;
        lmmc_real_t d_denominator = one_plus_odd + d_offset;
        lmmc_real_t c_denominator = 1.0 + c_offset;
        lmmc_real_t d_numerator = 1.0 + d_offset;
        lmmc_real_t delta;

        c_numerator = lentz_nonzero(c_numerator);
        d_denominator = lentz_nonzero(d_denominator);
        c_denominator = lentz_nonzero(c_denominator);
        d_numerator = lentz_nonzero(d_numerator);
        c = c_numerator / c_denominator;
        d = d_numerator / d_denominator;
        delta = c_numerator / d_denominator;
        f *= delta;
        if (!isfinite(even) || !isfinite(one_plus_odd)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (!isfinite(c) || !isfinite(d)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (!isfinite(delta) || !isfinite(f)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (fabs(delta - 1.0) < SPECIAL_FUNCTION_TOLERANCE) { break; }
    }
    if (m > SPECIAL_FUNCTION_MAX_ITERATIONS) { return LMMC_STATUS_CONVERGENCE_FAILED; }
    if (!(f > 0.0)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *out = f;
    return LMMC_STATUS_OK;
}

/**
 * @brief 用倒数参数稳定计算大形状参数下 log Gamma 的 Stirling 修正项。
 */
static double beta_stirling_inverse(double inverse) {
    const double square = inverse * inverse;
    return inverse * (1.0 / 12.0 + square * (-1.0 / 360.0 +
        square * (1.0 / 1260.0 - square / 1680.0)));
}

/**
 * @brief 计算 large * (log1p(small/large) - small/large)。
 * 先乘 small，在比值平方下溢时保留修正量。
 */
static double beta_scaled_log1pmx(double small, double large) {
    const double ratio = small / large;
    if (ratio <= 0.125) {
        double polynomial = -1.0 / 32.0;
        int k;
        for (k = 31; k >= 2; --k) {
            polynomial = fma(polynomial, ratio,
                             (k % 2 ? 1.0 : -1.0) / (double)k);
        }
        return (small * ratio) * polynomial;
    }
    return large * (log1p(ratio) - ratio);
}

static double beta_ratio_correction(double small, double large) {
    const double ratio = small / large;
    return (small - 0.5) * log1p(ratio) +
        beta_scaled_log1pmx(small, large) +
        beta_stirling_inverse((1.0 / large) / (1.0 + ratio)) -
        lmmc_log_gamma_stirling_error(large);
}

lmmc_status_t lmmc_log_beta(double a, double b, double* out_log_beta) {
    double value;
    const double small = fmin(a, b), large = fmax(a, b);
    if (!out_log_beta || !isfinite(a) || !isfinite(b) || a <= 0.0 || b <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (large < 16.0) {
        value = lgamma(small) + lgamma(large) - lgamma(small + large);
    } else if (small < 16.0) {
        value = lgamma(small) -
            (small * log(large) + beta_ratio_correction(small, large));
    } else {
        const double ratio = small / large;
        const double logarithm = log1p(ratio);
        const double log_ratio = ratio > 0.0
            ? log(ratio) : log(small) - log(large);
        value = small * (log_ratio - logarithm) - large * logarithm +
            0.5 * (logarithm - log(small)) + LMMC_LOG_SQRT_2PI +
            lmmc_log_gamma_stirling_error(small) +
            lmmc_log_gamma_stirling_error(large) -
            beta_stirling_inverse((1.0 / large) / (1.0 + ratio));
    }
    if (isnan(value) || value == INFINITY) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *out_log_beta = value;
    return LMMC_STATUS_OK;
}

static int beta_log_kernel_arguments_valid(
    double x, double complement_x, double a, double b, const double* out_log_kernel) {
    if (!out_log_kernel || !isfinite(a) || !isfinite(b) || a <= 0.0 || b <= 0.0) {
        return 0;
    }
    if (!isfinite(x) || !isfinite(complement_x)) {
        return 0;
    }
    if (x <= 0.0 || x > 1.0 || complement_x <= 0.0 || complement_x > 1.0) {
        return 0;
    }
    return 1;
}

static lmmc_status_t beta_log_kernel_large_shapes(
    double x, double complement_x, double a, double b, double total, double* value) {
    if (isfinite(total)) {
        *value = lmmc_log_gamma_stirling_error(total) -
            lmmc_log_gamma_stirling_error(a) -
            lmmc_log_gamma_stirling_error(b) -
            lmmc_deviance_part(a, total * x) -
            lmmc_deviance_part(b, total * complement_x) +
            0.5 * (log(a) + log(b) - log(total)) - LMMC_LOG_SQRT_2PI;
    } else if (a == b && x == 0.5 && complement_x == 0.5) {
        *value = 0.5 * (log(a) - log(LMMC_PI)) - log(2.0) -
            (1.0 / a) / 8.0;
    } else {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    return LMMC_STATUS_OK;
}

static double beta_log_kernel_one_large_shape(
    double x, double complement_x, double a, double b,
    double log_x, double log_complement) {
    const double small = fmin(a, b), large = fmax(a, b);
    const double small_x = a < b ? x : complement_x;
    const double log_small_x = a < b ? log_x : log_complement;
    const double log_large_x = a < b ? log_complement : log_x;
    const double product = small_x * large;
    const double log_product = product > 0.0
        ? log(product) : log_small_x + log(large);
    return small * log_product + large * log_large_x - lgamma(small) +
        beta_ratio_correction(small, large);
}

lmmc_status_t lmmc_beta_log_kernel(
    double x, double complement_x, double a, double b, double* out_log_kernel) {
    const double total = a + b;
    double value;
    double log_x, log_complement;
    if (!beta_log_kernel_arguments_valid(x, complement_x, a, b, out_log_kernel)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    log_x = x <= 0.5 ? log(x) : log1p(-complement_x);
    log_complement = complement_x <= 0.5 ? log(complement_x) : log1p(-x);
    if (a >= 16.0 && b >= 16.0) {
        const lmmc_status_t status =
            beta_log_kernel_large_shapes(x, complement_x, a, b, total, &value);
        if (status != LMMC_STATUS_OK) { return status; }
    } else if (a >= 16.0 || b >= 16.0) {
        value = beta_log_kernel_one_large_shape(x, complement_x, a, b, log_x, log_complement);
    } else {
        double logarithm;
        const lmmc_status_t status = lmmc_log_beta(a, b, &logarithm);
        if (status != LMMC_STATUS_OK) { return status; }
        value = a * log_x + b * log_complement - logarithm;
    }
    if (isnan(value) || value == INFINITY) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *out_log_kernel = value;
    return LMMC_STATUS_OK;
}

static lmmc_status_t regularized_beta_pair(
    lmmc_real_t x, lmmc_real_t complement_x,
    lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out) {
    lmmc_real_t log_bt, fraction, result, shape;
    lmmc_status_t status;
    const int lower = x * (b + 1.0) < complement_x * (a + 1.0);
    if (x == 0.5 && a == b) { return store_regularized_probability(0.5, out); }
    if (!isfinite(a + b)) { return LMMC_STATUS_NUMERICAL_FAILURE; }

    status = lmmc_beta_log_kernel(x, complement_x, a, b, &log_bt);
    if (status != LMMC_STATUS_OK) { return status; }
    if (isnan(log_bt) || log_bt > log(DBL_MAX)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    if (lower) {
        status = regularized_beta_cf(x, complement_x, a, b, &fraction);
        shape = a;
    } else {
        status = regularized_beta_cf(complement_x, x, b, a, &fraction);
        shape = b;
    }
    if (status != LMMC_STATUS_OK) { return status; }
    /** @brief 在对数域合并前因子与连分式，保留极小前因子的贡献。 */
    result = exp(log_bt + log(fraction) - log(shape));
    if (!isfinite(result) || result < 0.0 || result > 1.0) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    return store_regularized_probability(lower ? result : 1.0 - result, out);
}

lmmc_status_t regularized_beta(
    lmmc_real_t x, lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= 0.0) { return store_regularized_probability(0.0, out); }
    if (x >= 1.0) { return store_regularized_probability(1.0, out); }
    return regularized_beta_pair(x, 1.0 - x, a, b, out);
}

lmmc_status_t regularized_beta_upper(
    lmmc_real_t x, lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= 0.0) { return store_regularized_probability(1.0, out); }
    if (x >= 1.0) { return store_regularized_probability(0.0, out); }
    return regularized_beta_pair(1.0 - x, x, b, a, out);
}
