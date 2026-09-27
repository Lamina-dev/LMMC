#include <math.h>
#include "lmmc/special_functions.h"
#include <float.h>
#include "statistics_internal.h"

/** @brief 通过 C99 erf 计算误差函数 erf(x)。 */
lmmc_status_t lmmc_erf(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    *out = erf(x);
    return LMMC_STATUS_OK;
}

/**
 * @brief 通过 C 标准库 erfc 计算互补误差函数 erfc(x) = 1 - erf(x)。
 *
 * 直接计算 erfc，保持大 |x| 时的尾部精度。
 */
lmmc_status_t lmmc_erfc(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    *out = erfc(x);
    return LMMC_STATUS_OK;
}

/**
 * @brief 使用 Lanczos 近似计算对数伽马函数（g=7，n=9）。
 *
 * ln(Gamma(x)) = (x - 0.5) * ln(x + g - 0.5) - (x + g - 0.5) + 0.5*ln(2*pi) + ln(Ag(x))，
 * 其中 Ag(x) 为 Lanczos 级数和。
 */
lmmc_status_t lmmc_lgamma(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= 0.0) { return LMMC_STATUS_INVALID_ARGUMENT; }

    /** @brief Numerical Recipes / Cephes 的 Lanczos 系数（g=7，n=9）。 */
    static const double coeff[9] = {
         0.99999999999980993,
       676.5203681218851,
      -1259.1392167224028,
        771.32342877765313,
       -176.61502916214059,
         12.507343278686905,
         -0.13857109526572012,
          9.9843695780195716e-6,
          1.5056327351493116e-7
    };

    double g = 7.0;
    double tmp, ser;
    int i;

    if (x < 0.5) {
        /** @brief 反射公式：Gamma(x) * Gamma(1-x) = pi / sin(pi*x)。 */
        /** @brief 对数形式：lgamma(x) = ln(pi) - ln(sin(pi*x)) - lgamma(1-x)。 */
        double sinpx = sin(LMMC_PI * x);
        if (fabs(sinpx) < 1e-300) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        lmmc_real_t lg1mx;
        lmmc_status_t st = lmmc_lgamma(1.0 - x, &lg1mx);
        if (st != LMMC_STATUS_OK) { return st; }
        *out = log(LMMC_PI) - log(fabs(sinpx)) - lg1mx;
        return LMMC_STATUS_OK;
    }

    x -= 1.0;
    tmp = x + g + 0.5;
    ser = coeff[0];
    for (i = 1; i < 9; i++) {
        ser += coeff[i] / (x + (double)i);
    }

    *out = 0.5 * log(2.0 * LMMC_PI) + (x + 0.5) * log(tmp) - tmp + log(ser);
    return LMMC_STATUS_OK;
}

/** @brief 计算伽马函数 tgamma(x) = exp(lgamma(x))，要求 x > 0。 */
lmmc_status_t lmmc_tgamma(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= 0.0) { return LMMC_STATUS_INVALID_ARGUMENT; }

    lmmc_real_t lg;
    lmmc_status_t st = lmmc_lgamma(x, &lg);
    if (st != LMMC_STATUS_OK) { return st; }

    if (lg > 709.0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    *out = exp(lg);
    return LMMC_STATUS_OK;
}

/** @brief 使用稳定的共享 log-beta 内核，仅拒绝最终结果溢出。 */
lmmc_status_t lmmc_beta(lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out) {
    double logarithm, value;
    lmmc_status_t status;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_log_beta(a, b, &logarithm);
    if (status != LMMC_STATUS_OK) { return status; }
    if (logarithm > log(DBL_MAX)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    value = exp(logarithm);
    if (!isfinite(value)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    *out = value;
    return LMMC_STATUS_OK;
}

/**
 * @brief 计算双伽马函数 psi(x) = d/dx ln(Gamma(x))。
 *
 * 按 psi(x+1) = psi(x) + 1/x 递推至 x >= 7，再使用渐近展开：
 * psi(x) ~ ln(x) - 1/(2x) - sum_{k=1}^{N} B_{2k}/(2k * x^{2k})，
 * 其中 B_{2k} 为 Bernoulli 数。
 */
lmmc_status_t lmmc_digamma(lmmc_real_t x, lmmc_real_t* out) {
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (x <= 0.0) { return LMMC_STATUS_INVALID_ARGUMENT; }

    double result = 0.0;

    while (x < 7.0) {
        result -= 1.0 / x;
        x += 1.0;
    }

    /** @brief 渐近展开：psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...。 */
    /** @brief Bernoulli 数：B2=1/6，B4=-1/30，B6=1/42，B8=-1/30，B10=5/66，B12=-691/2730。 */
    {
        double ix = 1.0 / x;
        double ix2 = ix * ix;

        /** @brief 展开系数为 B_{2k} / (2k)。 */
        static const double bernoulli_coeff[] = {
            1.0 / 12.0,       /**< 系数：B2/(2*1) = (1/6)/2。 */
           -1.0 / 120.0,      /**< 系数：B4/(2*2) = (-1/30)/4。 */
            1.0 / 252.0,      /**< 系数：B6/(2*3) = (1/42)/6。 */
           -1.0 / 240.0,      /**< 系数：B8/(2*4) = (-1/30)/8。 */
            5.0 / 660.0,      /**< 系数：B10/(2*5) = (5/66)/10。 */
           -691.0 / 32760.0,  /**< 系数：B12/(2*6) = (-691/2730)/12。 */
            1.0 / 12.0        /**< 系数：B14/(2*7) = (7/6)/14 = 1/12。 */
        };

        double sum = 0.0;
        double ix2k = ix2; /**< ix^(2k)，初值为 ix^2。 */
        int k;
        for (k = 0; k < 7; k++) {
            sum -= bernoulli_coeff[k] * ix2k;
            ix2k *= ix2;
        }

        result += log(x) - 0.5 * ix + sum;
    }

    *out = result;
    return LMMC_STATUS_OK;
}

/**
 * @brief 估计 binary64 运算下的工程误差包络。
 * @note 假设 log/exp/expm1 误差不超过 2 ULP；此包络不是精确算术证明，C 标准也不保证该误差界。
 */
static double lambertw_ulp(double value) {
    const double magnitude = fabs(value);
    return nextafter(magnitude, INFINITY) - magnitude;
}

static void lambertw_residual(
    double w, double z, int region, double target, double target_error,
    double* residual, double* radius, double* first, double* second) {
    const double unit = DBL_EPSILON / 2.0;
    if (region == 1) {
        /**
         * @brief 此处 w 表示 u = W+1，H(u) = u^2 sum (k-1)u^(k-2)/k!。
         * 固定 binary64 系数的舍入误差计入包络半径。
         */
        static const double coefficient[] = {
            0.5, 0.33333333333333331, 0.125, 0.033333333333333333,
            0.0069444444444444441, 0.0011904761904761906,
            0.00017361111111111112, 2.2045855379188714e-05,
            2.4801587301587302e-06, 2.5052108385441718e-07,
            2.296443268665491e-08, 1.9270852604185937e-09,
            1.4911969277048643e-10, 1.0706029224547743e-11,
            7.1692159985810778e-13, 4.498331606952833e-14,
            2.6552651846596585e-15, 1.4797143443923793e-16,
            7.8096034842931133e-18, 3.9145882126782523e-19,
            1.8683261924146203e-20, 8.5099743753875053e-22,
            3.7069964135210721e-23, 1.5472680682522736e-24,
            6.1989906580619936e-26, 2.3877593645868422e-27,
            8.855700940088562e-29, 3.1667896082053609e-30,
            1.0932964123566126e-31, 3.6483751246605535e-33,
            1.1781211340049704e-34
        };
        double p, absolute, square, value, exponential;
        int k;
        p = coefficient[30];
        absolute = fabs(p);
        for (k = 29; k >= 0; --k) {
            p = fma(p, w, coefficient[k]);
            absolute = fma(absolute, fabs(w), coefficient[k]);
        }
        square = w * w;
        value = square * p;
        *residual = value - target;
        *radius = (96.0 * unit / (1.0 - 96.0 * unit)) *
                      square * absolute + target_error +
                  unit * fabs(*residual) +
                  4.0e-36 * pow(fabs(w), 33.0);
        exponential = exp(w);
        *first = w * exponential;
        *second = (1.0 + w) * exponential;
    } else if (region == 2) {
        const double logarithm = log(fabs(w));
        double low;
        const double sum = lmmc_two_sum(w, logarithm, &low);
        *residual = (sum - target) + low;
        *radius = 2.0 * lambertw_ulp(logarithm) + target_error +
                  4.0 * unit * (fabs(sum - target) + fabs(low)) +
                  2.0 * unit * fabs(sum);
        *first = 1.0 + 1.0 / w;
        *second = -(1.0 / w) / w;
    } else {
        const double em1 = expm1(w);
        const double difference = w - z;
        *residual = fma(w, em1, difference);
        *radius = 2.0 * fabs(w) * lambertw_ulp(em1) +
                  unit * (fabs(difference) + fabs(*residual)) +
                  4.0 * nextafter(0.0, 1.0);
        *first = exp(w) * (1.0 + w);
        *second = exp(w) * (2.0 + w);
    }
}

typedef struct {
    double lo, hi, w, target, target_error, step;
    int region, increasing;
} lmmc_lambertw_state_t;

static void lambertw_initialize(double z, int branch, lmmc_lambertw_state_t* state) {
    state->target = 0.0;
    state->target_error = 0.0;
    state->step = INFINITY;
    if (z < -0.3) {
        const double e_hi = 2.718281828459045;
        const double e_lo = 1.4456468917292502e-16;
        const double high = fma(e_hi, z, 1.0);
        const double low = e_lo * z;
        double remainder;
        state->target = lmmc_two_sum(high, low, &remainder);
        state->target_error = fabs(remainder) +
            DBL_EPSILON * (fabs(high) + fabs(low)) + 2.0e-32 * fabs(z);
        state->region = 1;
        state->increasing = branch == 0;
        state->lo = state->increasing ? 0.0 : -1.0;
        state->hi = state->increasing ? 1.0 : 0.0;
        state->w = (state->increasing ? 1.0 : -1.0) * sqrt(2.0 * state->target) -
            (2.0 / 3.0) * state->target;
    } else if (branch == -1 || z > 1.0) {
        state->region = 2;
        state->increasing = 1;
        state->target = log(fabs(z));
        state->target_error = 2.0 * lambertw_ulp(state->target);
        if (branch == -1) {
            state->lo = 2.0 * state->target;
            state->hi = -1.0;
            state->w = state->target - log(-state->target);
        } else {
            state->lo = 0.5;
            state->hi = state->target + 1.0;
            state->w = z <= 2.718281828459045 ? 1.0 : state->target - log(state->target);
        }
    } else {
        state->region = 0;
        state->increasing = 1;
        state->lo = z < 0.0 ? -1.0 : 0.0;
        state->hi = z < 0.0 ? 0.0 : z;
        state->w = z < 0.0 ? z : z / (1.0 + z);
    }
    if (!(state->w > state->lo && state->w < state->hi)) {
        state->w = state->lo + (state->hi - state->lo) / 2.0;
    }
}

static int lambertw_accept_enclosure(
    double z, const lmmc_lambertw_state_t* state, double tolerance, double value) {
    if (fabs(state->step) <= tolerance) {
        double fl, el, fr, er, unused1, unused2;
        /** @brief 在目标容差内预留输出减法和端点舍入误差。 */
        const double offset = tolerance - 2.0 * lambertw_ulp(value);
        const double left = fmax(state->lo, state->w - offset);
        const double right = fmin(state->hi, state->w + offset);
        lambertw_residual(left, z, state->region, state->target, state->target_error,
                         &fl, &el, &unused1, &unused2);
        lambertw_residual(right, z, state->region, state->target, state->target_error,
                         &fr, &er, &unused1, &unused2);
        if ((state->increasing && fl + el <= 0.0 && fr - er >= 0.0) ||
            (!state->increasing && fl - el >= 0.0 && fr + er <= 0.0)) {
            return 1;
        }
    }
    return 0;
}

static void lambertw_safeguard(
    lmmc_lambertw_state_t* state, double f, double error, double first, double second) {
    double candidate, denominator;
    if (f > error) {
        if (state->increasing) { state->hi = state->w; } else { state->lo = state->w; }
    } else if (f < -error) {
        if (state->increasing) { state->lo = state->w; } else { state->hi = state->w; }
    }
    denominator = first - 0.5 * f * (second / first);
    candidate = state->w - f / denominator;
    if (!isfinite(candidate) || !(candidate > state->lo && candidate < state->hi)) {
        candidate = state->lo + (state->hi - state->lo) / 2.0;
    }
    state->step = candidate - state->w;
    if (candidate == state->w && fabs(f) > error) {
        candidate = state->lo + (state->hi - state->lo) / 2.0;
        state->step = candidate - state->w;
    }
    state->w = candidate;
}

static lmmc_status_t lambertw_iterate(
    double z, unsigned max_iterations, lmmc_lambertw_state_t* state, double* out) {
    unsigned iteration;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        double f, error, first, second;
        const double value = state->region == 1 ? state->w - 1.0 : state->w;
        const double tolerance = 64.0 * DBL_EPSILON * fabs(value) +
                                 2.0 * nextafter(0.0, 1.0);
        lambertw_residual(state->w, z, state->region, state->target, state->target_error,
                         &f, &error, &first, &second);
        if (!isfinite(f) || !isfinite(error)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
        if (lambertw_accept_enclosure(z, state, tolerance, value)) {
            *out = value;
            return LMMC_STATUS_OK;
        }
        lambertw_safeguard(state, f, error, first, second);
    }
    return LMMC_STATUS_CONVERGENCE_FAILED;
}

static lmmc_status_t lambertw_real(
    double z, int branch, unsigned max_iterations, double* out) {
    lmmc_lambertw_state_t state;
    if (!out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!isfinite(z)) { return LMMC_STATUS_NUMERICAL_FAILURE; }
    if (z < -LMMC_INV_E || (branch == -1 && z >= 0.0)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (z == -LMMC_INV_E) { *out = -1.0; return LMMC_STATUS_OK; }
    if (branch == 0 && z == 0.0) { *out = z; return LMMC_STATUS_OK; }
    if (branch == 0 && fabs(z) <= 0x1p-26) {
        /**
         * @brief 逆级数截断尾项界为 (e|z|)^4/(1-e|z|)，在此远小于 1 ULP。
         * 保留次正规数与带符号零的位表示。
         */
        *out = fabs(z) <= DBL_EPSILON / 4.0
            ? z : z * fma(z, fma(1.5, z, -1.0), 1.0);
        return LMMC_STATUS_OK;
    }
    lambertw_initialize(z, branch, &state);
    return lambertw_iterate(z, max_iterations, &state, out);
}

lmmc_status_t lmmc_lambertw(lmmc_real_t z, lmmc_real_t* out_res) {
    return lambertw_real(z, 0, 100, out_res);
}

lmmc_status_t lmmc_lambertw_wm1(lmmc_real_t z, lmmc_real_t* out_res) {
    return lambertw_real(z, -1, 100, out_res);
}
