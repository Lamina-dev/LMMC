#include <math.h>
#include "random_internal.h"

/**
 * @brief Poisson 分布采样。
 * lambda < 10 时使用逆变换法，lambda >= 10 时使用 Hörmann PTRD 法。
 */

/** @brief 以逆变换法采样小 lambda 的 Poisson 分布。 */
static size_t poisson_inversion(uint64_t* state, double lambda) {
    double L = exp(-lambda);
    double p = 1.0;
    size_t k = 0;

    do {
        k++;
        p *= u64_to_double01(xoshiro256ss_next(state));
    } while (p > L);

    return k - 1;
}

static lmmc_status_t poisson_store_sample(double sample, size_t* out) {
    const double exclusive_limit = (double)SIZE_MAX + 1.0;
    if (!isfinite(sample) || sample < 0.0 || sample >= exclusive_limit) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = (size_t)sample;
    return LMMC_STATUS_OK;
}

/**
 * @brief 以分解变换拒绝法 PTRD 采样 lambda >= 10 的 Poisson 分布。
 * @note 参考 Hörmann, "The transformed rejection method for generating Poisson
 * random variables", Insurance: Mathematics and Economics 12 (1993) 39-45。
 */
static lmmc_status_t poisson_ptrd(uint64_t* state, double lambda, size_t* out) {
    double smu = sqrt(lambda);
    double b = 0.931 + 2.53 * smu;
    double a = -0.059 + 0.02483 * b;
    double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    double vr = 0.9277 - 3.6224 / (b - 2.0);
    double us, v, u, k_real;

    for (;;) {
        v = u64_to_double01(xoshiro256ss_next(state));
        if (v <= 0.86 * vr) {
            u = v / vr - 0.43;
            k_real = floor((2.0 * a / (0.5 - fabs(u)) + b) * u + lambda + 0.445);
            if (k_real >= 0.0) {
                return poisson_store_sample(k_real, out);
            }
        }

        if (v >= vr) {
            u = u64_to_double01(xoshiro256ss_next(state)) - 0.5;
        } else {
            u = v / vr - 0.93;
            u = ((u >= 0.0) ? 0.5 : -0.5) - u;
            v = u64_to_double01(xoshiro256ss_next(state)) * vr;
        }

        us = 0.5 - fabs(u);
        if (us == 0.0 || (us < 0.013 && v > us)) {
            continue;
        }

        k_real = floor((2.0 * a / us + b) * u + lambda + 0.445);
        if (!isfinite(k_real)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (k_real < 0.0) {
            continue;
        }

        /**
         * @brief 接受条件为 log(v * inv_alpha / (a/(us*us) + b)) <=
         * -lambda + k*log(lambda) - lgamma(k+1)。
         * 整数候选值以 double 保存，接受后再检查输出范围。
         */
        v = v * inv_alpha / (a / (us * us) + b);
        {
            const double log_accept = k_real * log(lambda) - lambda - lgamma(k_real + 1.0);
            if (!isfinite(log_accept)) {
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            if (log(v) <= log_accept) {
                return poisson_store_sample(k_real, out);
            }
        }
    }
}

lmmc_status_t lmmc_rng_poisson(
    lmmc_rng_t* rng,
    lmmc_real_t lambda,
    size_t* out)
{
    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(lambda) || lambda <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (lambda < 10.0) {
        *out = poisson_inversion(rng->state, lambda);
    } else {
        return poisson_ptrd(rng->state, lambda, out);
    }

    return LMMC_STATUS_OK;
}


/**
 * @brief 二项分布采样。
 * np < 30 时使用自底向上的精确逆 CDF，其余使用
 * Kachitvichyanukul-Schmeiser BTPE 变换拒绝法。
 */
static uint64_t binomial_inversion(uint64_t* state, uint64_t n, double p) {
    const double q = 1.0 - p;
    const double qn = exp((double)n * log1p(-p));
    const double np = (double)n * p;
    const uint64_t bound = (uint64_t)fmin((double)n, np + 10.0 * sqrt(np * q + 1.0));
    uint64_t x = 0;
    double px = qn;
    double u = u64_to_double01(xoshiro256ss_next(state));

    while (u > px) {
        ++x;
        if (x > bound) {
            x = 0;
            px = qn;
            u = u64_to_double01(xoshiro256ss_next(state));
        } else {
            u -= px;
            px = (((double)(n - x + 1) * p) * px) / ((double)x * q);
        }
    }
    return x;
}

static int binomial_btpe_accept(
    uint64_t n, double r, double q, int64_t m, double nrq,
    double xm, int64_t y, double v) {
    const uint64_t k = y >= m ? (uint64_t)y - (uint64_t)m : (uint64_t)m - (uint64_t)y;
    if (k <= 20 || (double)k >= nrq / 2.0 - 1.0) {
        const double s = r / q;
        const double aa = s * ((double)n + 1.0);
        double f = 1.0;
        int64_t i;
        if (m < y) {
            for (i = m + 1; i <= y; ++i) {
                f *= aa / (double)i - s;
            }
        } else if (m > y) {
            for (i = y + 1; i <= m; ++i) {
                f /= aa / (double)i - s;
            }
        }
        if (v > f) {
            return 0;
        }
    } else {
        const double kd = (double)k;
        const double rho = (kd / nrq) *
            ((kd * (kd / 3.0 + 0.625) + 1.0 / 6.0) / nrq + 0.5);
        const double t = -kd * kd / (2.0 * nrq);
        const double logv = log(v);
        if (logv > t - rho) {
            const double x1 = (double)y + 1.0;
            const double f1 = (double)m + 1.0;
            const double z = (double)n + 1.0 - (double)m;
            const double w = (double)n - (double)y + 1.0;
            const double x2 = x1 * x1, f2 = f1 * f1, z2 = z * z, w2 = w * w;
            double accept;
            if (logv > t + rho) {
                return 0;
            }
            accept = xm * log(f1 / x1) + ((double)n - (double)m + 0.5) * log(z / w)
                   + ((double)y - (double)m) * log(w * r / (x1 * q))
                   + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0
                   + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0
                   - (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0
                   - (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0;
            if (logv > accept) {
                return 0;
            }
        }
    }
    return 1;
}

static uint64_t binomial_btpe(uint64_t* state, uint64_t n, double p) {
    const double r = fmin(p, 1.0 - p);
    const double q = 1.0 - r;
    const double fm = (double)n * r + r;
    const int64_t m = (int64_t)floor(fm);
    const double p1 = floor(2.195 * sqrt((double)n * r * q) - 4.6 * q) + 0.5;
    const double xm = (double)m + 0.5;
    const double xl = xm - p1;
    const double xr = xm + p1;
    const double c = 0.134 + 20.5 / (15.3 + (double)m);
    double a = (fm - xl) / (fm - xl * r);
    const double laml = a * (1.0 + a / 2.0);
    double lamr;
    const double p2 = p1 * (1.0 + 2.0 * c);
    const double p3 = p2 + c / laml;
    double p4;
    const double nrq = (double)n * r * q;

    a = (xr - fm) / (xr * q);
    lamr = a * (1.0 + a / 2.0);
    p4 = p3 + c / lamr;

    for (;;) {
        double u = u64_to_double01(xoshiro256ss_next(state)) * p4;
        double v = u64_to_double01(xoshiro256ss_next(state));
        double x;
        int64_t y;

        if (u <= p1) {
            uint64_t result;
            y = (int64_t)floor(xm - p1 * v + u);
            result = (uint64_t)y;
            return p > 0.5 ? n - result : result;
        } else if (u <= p2) {
            x = xl + (u - p1) / c;
            v = v * c + 1.0 - fabs((double)m - x + 0.5) / p1;
            if (v > 1.0) {
                continue;
            }
            y = (int64_t)floor(x);
        } else if (u <= p3) {
            if (v == 0.0) {
                continue;
            }
            y = (int64_t)floor(xl + log(v) / laml);
            if (y < 0) {
                continue;
            }
            v = v * (u - p2) * laml;
        } else {
            if (v == 0.0) {
                continue;
            }
            y = (int64_t)floor(xr - log(v) / lamr);
            if (y < 0 || (uint64_t)y > n) {
                continue;
            }
            v = v * (u - p3) * lamr;
        }

        if (!binomial_btpe_accept(n, r, q, m, nrq, xm, y, v)) {
            continue;
        }
        {
            uint64_t result = (uint64_t)y;
            return p > 0.5 ? n - result : result;
        }
    }
}

static uint64_t binomial_chunk(uint64_t* state, uint64_t n, double p) {
    const double working_p = p <= 0.5 ? p : 1.0 - p;
    uint64_t value;
    if ((double)n * working_p < 30.0) {
        value = binomial_inversion(state, n, working_p);
        return p <= 0.5 ? value : n - value;
    }
    return binomial_btpe(state, n, p);
}

lmmc_status_t lmmc_rng_binomial(
    lmmc_rng_t* rng,
    size_t n,
    lmmc_real_t p,
    size_t* out
) {
    const uint64_t max_chunk = (UINT64_C(1) << 53) - UINT64_C(1);
    uint64_t remaining;
    uint64_t total = 0;
    if (rng == NULL || out == NULL || !isfinite(p) || p < 0.0 || p > 1.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n == 0 || p == 0.0) {
        *out = 0;
        return LMMC_STATUS_OK;
    }
    if (p == 1.0) {
        *out = n;
        return LMMC_STATUS_OK;
    }

    remaining = (uint64_t)n;
    while (remaining != 0) {
        uint64_t chunk = remaining > max_chunk ? max_chunk : remaining;
        total += binomial_chunk(rng->state, chunk, p);
        remaining -= chunk;
    }
    *out = (size_t)total;
    return LMMC_STATUS_OK;
}
