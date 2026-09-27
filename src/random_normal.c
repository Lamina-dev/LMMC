#include <math.h>
#include "random_internal.h"

/**
 * @brief 使用 256 个矩形的 Ziggurat 正态采样。
 *
 * 在标准正态分布右半部（x >= 0）采样 |x|，再随机赋予符号。
 * 表布局：
 * - zig_xtab[0] = v/f(r)：含尾部的底层宽度。
 * - zig_xtab[1] = r：尾部截断点。
 * - zig_xtab[i]（i=2..255）：递减的 x 坐标。
 * - zig_xtab[256] = 0：分布峰值位置。
 *
 * r = 3.6541528853610088，v = 0.00492867323399 为半正态中每个矩形的面积。
 * @see Marsaglia & Tsang, "The Ziggurat Method for Generating Random Variables", JSS 2000.
 */

#define ZIG_N 256
#define ZIG_R 3.6541528853610088

static const double zig_xtab[ZIG_N + 1] = {
#include "ziggurat_table.inc"
};
static inline double zig_pdf(double x) {
    return exp(-0.5 * x * x);
}


/** @brief 用 Marsaglia 精确尾部算法采样标准正态的 |x| > r 部分。 */
static double zig_sample_tail(uint64_t* state) {
    double x, y, u1, u2;
    for (;;) {
        do {
            u1 = u64_to_double01(xoshiro256ss_next(state));
        } while (u1 == 0.0);
        do {
            u2 = u64_to_double01(xoshiro256ss_next(state));
        } while (u2 == 0.0);

        x = -log(u1) / ZIG_R;
        y = -log(u2);

        if (2.0 * y >= x * x) {
            return x + ZIG_R;
        }
    }
}

/** @brief 用 256 层 Ziggurat 采样标准正态：先采样 |x|，再随机赋予符号。 */
double lmmc_rng_standard_normal(uint64_t* state) {
    uint64_t u, u2;
    int i, sign;
    double x;


    for (;;) {
        u = xoshiro256ss_next(state);
        i = (int)(u & 0xFF);  /**< 层号，范围 0..255。 */
        sign = (u & 0x100) ? -1 : 1;  /**< 第 8 位决定符号。 */

        /** @brief 用独立随机数在 [0, xtab[i]) 内均匀采样。 */
        u2 = xoshiro256ss_next(state);
        x = u64_to_double01(u2) * zig_xtab[i];

        if (x < zig_xtab[i + 1]) {
            return sign * x;
        }

        /** @brief 第 0 层包含尾部。 */
        if (i == 0) {
            /** @brief 区间 [xtab[1], xtab[0]) 的候选值采用尾部采样。 */
            if (x < zig_xtab[1]) {
                return sign * x;
            }
            double tail = zig_sample_tail(state);
            return sign * tail;
        }

        /** @brief 楔形区以 (f(x) - f(xtab[i])) / (f(xtab[i+1]) - f(xtab[i])) 的概率接受。 */
        {
            double f_x = zig_pdf(x);
            double f_outer = zig_pdf(zig_xtab[i]);     /**< 外边缘密度，值较小。 */
            double f_inner = zig_pdf(zig_xtab[i + 1]); /**< 内边缘密度，值较大。 */
            double u_wedge = u64_to_double01(xoshiro256ss_next(state));

            if (u_wedge * (f_inner - f_outer) < (f_x - f_outer)) {
                return sign * x;
            }
        }
    }
}

lmmc_status_t lmmc_rng_normal(
    lmmc_rng_t* rng,
    lmmc_real_t mean,
    lmmc_real_t stddev,
    lmmc_real_t* out_value)
{
    if (rng == NULL || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(mean) || !isfinite(stddev) || stddev <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    *out_value = mean + stddev * lmmc_rng_standard_normal(rng->state);
    return LMMC_STATUS_OK;
}
