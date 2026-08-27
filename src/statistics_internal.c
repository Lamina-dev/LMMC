#include "statistics_internal.h"

/**
 * @brief 正则化不完全伽马函数下尾 P(a, x) = gamma(a, x) / Gamma(a)。
 * 使用级数展开或连分数展开。
 */
lmmc_real_t regularized_gamma_lower(lmmc_real_t a, lmmc_real_t x) {
    if (x <= 0.0) return 0.0;
    if (x < a + 1.0) {
        /* Series expansion */
        lmmc_real_t sum = 1.0 / a;
        lmmc_real_t term = 1.0 / a;
        lmmc_real_t ap = a;
        int i;
        for (i = 0; i < 200; ++i) {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if (fabs(term) < fabs(sum) * 1e-15) break;
        }
        return sum * exp(-x + a * log(x) - lgamma(a));
    } else {
        /* Use complement: P = 1 - Q, where Q uses continued fraction */
        return 1.0 - regularized_gamma_upper_cf(a, x);
    }
}

/**
 * @brief 正则化不完全伽马函数上尾 Q(a, x) = 1 - P(a, x)。
 * 使用 Lentz 连分数展开。
 */
lmmc_real_t regularized_gamma_upper_cf(lmmc_real_t a, lmmc_real_t x) {
    /* Lentz continued fraction for Q(a, x) */
    lmmc_real_t f, c, d, delta;
    int i;
    lmmc_real_t tiny = 1e-30;

    f = tiny;
    c = tiny;
    d = 0.0;

    /* CF: b0 = 0, a1 = 1, b1 = x - a + 1, a_i = (i-1)*(a - (i-1)), b_i = x - a + 2*i - 1 */
    /* Modified Lentz: start with b0 = x - a + 1 */
    {
        lmmc_real_t b0 = x - a + 1.0;
        if (fabs(b0) < tiny) b0 = tiny;
        f = b0;
        c = b0;
        d = 0.0;
    }

    for (i = 1; i <= 200; ++i) {
        lmmc_real_t an = (lmmc_real_t)i * (a - (lmmc_real_t)i);
        lmmc_real_t bn = x - a + 2.0 * (lmmc_real_t)i + 1.0;

        d = bn + an * d;
        if (fabs(d) < tiny) d = tiny;
        d = 1.0 / d;

        c = bn + an / c;
        if (fabs(c) < tiny) c = tiny;

        delta = c * d;
        f *= delta;
        if (fabs(delta - 1.0) < 1e-15) break;
    }

    return exp(-x + a * log(x) - lgamma(a)) / f;
}

/**
 * @brief 正则化不完全贝塔函数 I_x(a, b)。
 * 使用连分数展开（Lentz 方法）。
 */
lmmc_real_t regularized_beta_cf(lmmc_real_t x, lmmc_real_t a, lmmc_real_t b) {
    lmmc_real_t qab, qap, qam, c, d, f;
    int m, m2;
    lmmc_real_t tiny = 1e-30;

    qab = a + b;
    qap = a + 1.0;
    qam = a - 1.0;

    c = 1.0;
    d = 1.0 - qab * x / qap;
    if (fabs(d) < tiny) d = tiny;
    d = 1.0 / d;
    f = d;

    for (m = 1; m <= 200; ++m) {
        lmmc_real_t aa, del;
        m2 = 2 * m;

        /* Even step */
        aa = (lmmc_real_t)m * (b - (lmmc_real_t)m) * x /
             ((qam + (lmmc_real_t)m2) * (a + (lmmc_real_t)m2));
        d = 1.0 + aa * d;
        if (fabs(d) < tiny) d = tiny;
        c = 1.0 + aa / c;
        if (fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        f *= c * d;

        /* Odd step */
        aa = -(a + (lmmc_real_t)m) * (qab + (lmmc_real_t)m) * x /
             ((a + (lmmc_real_t)m2) * (qap + (lmmc_real_t)m2));
        d = 1.0 + aa * d;
        if (fabs(d) < tiny) d = tiny;
        c = 1.0 + aa / c;
        if (fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        del = c * d;
        f *= del;

        if (fabs(del - 1.0) < 1e-15) break;
    }

    return f;
}

/**
 * @brief 正则化不完全贝塔函数 I_x(a, b)。
 */
lmmc_real_t regularized_beta(lmmc_real_t x, lmmc_real_t a, lmmc_real_t b) {
    lmmc_real_t bt;
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) +
             a * log(x) + b * log(1.0 - x));

    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * regularized_beta_cf(x, a, b) / a;
    } else {
        return 1.0 - bt * regularized_beta_cf(1.0 - x, b, a) / b;
    }
}
