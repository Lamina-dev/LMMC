/**
 * @file fft.c
 * @brief 使用 radix-2 Cooley-Tukey 与 Bluestein chirp-z 实现任意长度 N 的 FFT。
 *
 * lmmc_fft、lmmc_fft_forward 与 lmmc_fft_inverse 处理任意 N，
 * lmmc_fft_radix4_pad_into 提供显式填充接口。
 *
 * 分派规则：
 *   - N 为 4 的幂：使用现有 radix-4 路径
 *   - N 为 2 的幂：使用原地迭代 radix-2 Cooley-Tukey
 *   - 其他 N：使用 Bluestein chirp-z，并填充到不小于 2N-1 的 2 的幂
 *
 * @see James W. Cooley and John W. Tukey,
 *      “An Algorithm for the Machine Calculation of Complex Fourier Series,” 1965.
 * @see Leo I. Bluestein,
 *      “A Linear Filtering Approach to the Computation of Discrete Fourier Transform,” 1970.
 */
#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/numeric.h"

/**
 * @brief Check if n is a power of 2.
 */
static int fft_is_power_of_two(size_t n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

/**
 * @brief Check if n is a power of 4.
 */
static int fft_is_power_of_four(size_t n) {
    if (n == 0) return 0;
    if ((n & (n - 1)) != 0) return 0;
    /** 2 的幂仅有一个置位；该位位于偶数位置时数值同时为 4 的幂。 */
    return (n & 0x5555555555555555ULL) != 0;
}

/**
 * @brief Compute the next power of 2 >= n.
 */
static size_t fft_next_power_of_two(size_t n) {
    size_t p = 1;
    if (n == 0) return 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

/**
 * @brief Bit-reversal permutation for power-of-2 length.
 */
static void fft_bit_reverse(lmmc_real_t* real, lmmc_real_t* imag, size_t n) {
    size_t i, j, bit;
    lmmc_real_t tmp;

    j = 0;
    for (i = 1; i < n; ++i) {
        bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if (i < j) {
            tmp = real[i]; real[i] = real[j]; real[j] = tmp;
            tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
        }
    }
}

/**
 * @brief In-place radix-2 Cooley-Tukey FFT for power-of-2 lengths.
 * @param inverse  If non-zero, compute inverse FFT (with 1/N normalization).
 */
static lmmc_status_t fft_radix2(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse) {
    size_t len, i, j;
    double angle_sign;

    if (n <= 1) return LMMC_STATUS_OK;

    fft_bit_reverse(real, imag, n);

    angle_sign = inverse ? 1.0 : -1.0;

    for (len = 2; len <= n; len <<= 1) {
        double angle = angle_sign * 2.0 * LMMC_PI / (double)len;
        double wlen_r = cos(angle);
        double wlen_i = sin(angle);

        for (i = 0; i < n; i += len) {
            double w_r = 1.0, w_i = 0.0;
            size_t half = len >> 1;

            for (j = 0; j < half; ++j) {
                size_t u_idx = i + j;
                size_t v_idx = i + j + half;

                /** 计算旋转因子与 x[v] 的乘积。 */
                double t_r = w_r * real[v_idx] - w_i * imag[v_idx];
                double t_i = w_r * imag[v_idx] + w_i * real[v_idx];

                /** 执行蝶形合并。 */
                real[v_idx] = real[u_idx] - t_r;
                imag[v_idx] = imag[u_idx] - t_i;
                real[u_idx] = real[u_idx] + t_r;
                imag[u_idx] = imag[u_idx] + t_i;

                /** 推进旋转因子。 */
                {
                    double new_w_r = w_r * wlen_r - w_i * wlen_i;
                    double new_w_i = w_r * wlen_i + w_i * wlen_r;
                    w_r = new_w_r;
                    w_i = new_w_i;
                }
            }
        }
    }

    if (inverse) {
        double scale = 1.0 / (double)n;
        for (i = 0; i < n; ++i) {
            real[i] *= scale;
            imag[i] *= scale;
        }
    }

    return LMMC_STATUS_OK;
}

/**
 * @brief 对任意长度 N 执行 Bluestein chirp-z 变换。
 *
 * 算法步骤：
 *   1. 计算 chirp 序列：w[k] = exp(±i*pi*k^2/N)
 *   2. 输入乘 chirp：a[k] = x[k] * conj(w[k])
 *   3. 构造零填充卷积核：b[k] = w[k]
 *   4. 通过 FFT 计算循环卷积
 *   5. 结果乘 chirp 并完成归一化
 */
static lmmc_status_t fft_bluestein(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse) {
    size_t m;           /* padded power-of-2 length */
    size_t k;
    double sign;
    lmmc_real_t *a_r = NULL, *a_i = NULL;
    lmmc_real_t *b_r = NULL, *b_i = NULL;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (n <= 1) {
        return LMMC_STATUS_OK;
    }

    /** 填充长度取不小于 2N-1 的最小 2 的幂。 */
    m = fft_next_power_of_two(2 * n - 1);

    /** 分配工作数组。 */
    a_r = (lmmc_real_t*)lmmc_alloc(m * sizeof(lmmc_real_t));
    a_i = (lmmc_real_t*)lmmc_alloc(m * sizeof(lmmc_real_t));
    b_r = (lmmc_real_t*)lmmc_alloc(m * sizeof(lmmc_real_t));
    b_i = (lmmc_real_t*)lmmc_alloc(m * sizeof(lmmc_real_t));

    if (a_r == NULL || a_i == NULL || b_r == NULL || b_i == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }

    /** 将工作数组初始化为零。 */
    memset(a_r, 0, m * sizeof(lmmc_real_t));
    memset(a_i, 0, m * sizeof(lmmc_real_t));
    memset(b_r, 0, m * sizeof(lmmc_real_t));
    memset(b_i, 0, m * sizeof(lmmc_real_t));

    /** 正向 DFT 使用 exp(-2*pi*i*k*n/N)，因此正向 chirp 相位符号为 -1，
     * 逆向相位符号为 +1。
     */
    sign = inverse ? 1.0 : -1.0;

    /** 构造 chirp 调制输入 a[k] = x[k] * exp(sign*i*pi*k^2/N)。 */
    for (k = 0; k < n; ++k) {
        double phase = sign * LMMC_PI * (double)(k * k) / (double)n;
        double c = cos(phase);
        double s = sin(phase);
        /** 将输入乘以 c + i*s。 */
        a_r[k] = real[k] * c - imag[k] * s;
        a_i[k] = imag[k] * c + real[k] * s;
    }

    /** 构造卷积核 b[k] = exp(-sign*i*pi*k^2/N)，并为负索引设置环绕项。 */
    for (k = 0; k < n; ++k) {
        double phase = -sign * LMMC_PI * (double)(k * k) / (double)n;
        double c = cos(phase);
        double s = sin(phase);
        b_r[k] = c;
        b_i[k] = s;
    }
    /** 设置环绕项 b[m-k] = b[k]，k = 1..n-1。 */
    for (k = 1; k < n; ++k) {
        b_r[m - k] = b_r[k];
        b_i[m - k] = b_i[k];
    }

    /** 对 a 与 b 执行长度 m 的正向 radix-2 FFT。 */
    st = fft_radix2(a_r, a_i, m, 0);
    if (st != LMMC_STATUS_OK) goto cleanup;

    st = fft_radix2(b_r, b_i, m, 0);
    if (st != LMMC_STATUS_OK) goto cleanup;

    /** 逐点计算 a = a * b。 */
    for (k = 0; k < m; ++k) {
        double tr = a_r[k] * b_r[k] - a_i[k] * b_i[k];
        double ti = a_r[k] * b_i[k] + a_i[k] * b_r[k];
        a_r[k] = tr;
        a_i[k] = ti;
    }

    /* Inverse FFT of the product */
    st = fft_radix2(a_r, a_i, m, 1);
    if (st != LMMC_STATUS_OK) goto cleanup;

    /* Extract result: X[k] = a[k] * exp(sign * i * pi * k^2 / N) */
    for (k = 0; k < n; ++k) {
        double phase = sign * LMMC_PI * (double)(k * k) / (double)n;
        double c = cos(phase);
        double s = sin(phase);
        /* result[k] = a[k] * exp(sign*i*pi*k^2/N) = a[k] * (c + i*s) */
        real[k] = a_r[k] * c - a_i[k] * s;
        imag[k] = a_i[k] * c + a_r[k] * s;
    }

    /* For inverse FFT, normalize by 1/N */
    if (inverse) {
        double scale = 1.0 / (double)n;
        for (k = 0; k < n; ++k) {
            real[k] *= scale;
            imag[k] *= scale;
        }
    }

cleanup:
    if (a_r) lmmc_free(a_r);
    if (a_i) lmmc_free(a_i);
    if (b_r) lmmc_free(b_r);
    if (b_i) lmmc_free(b_i);
    return st;
}

lmmc_status_t lmmc_fft(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse) {
    if (real == NULL || imag == NULL || n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n == 1) {
        /* DFT of length 1 is identity */
        return LMMC_STATUS_OK;
    }

    /* Dispatch: power-of-4 -> radix-4, power-of-2 -> radix-2, else -> Bluestein */
    if (fft_is_power_of_four(n)) {
        return lmmc_fft_radix4(real, imag, n, inverse ? 1 : 0);
    } else if (fft_is_power_of_two(n)) {
        return fft_radix2(real, imag, n, inverse ? 1 : 0);
    } else {
        return fft_bluestein(real, imag, n, inverse ? 1 : 0);
    }
}

lmmc_status_t lmmc_fft_forward(lmmc_real_t* real, lmmc_real_t* imag, size_t n) {
    return lmmc_fft(real, imag, n, 0);
}

lmmc_status_t lmmc_fft_inverse(lmmc_real_t* real, lmmc_real_t* imag, size_t n) {
    return lmmc_fft(real, imag, n, 1);
}

lmmc_status_t lmmc_fft_radix4_pad_into(
    const lmmc_real_t* real_in, const lmmc_real_t* imag_in, size_t n,
    lmmc_real_t* real_out, lmmc_real_t* imag_out, size_t* out_nfft)
{
    size_t nfft = 0;
    lmmc_status_t st;

    if (real_in == NULL || imag_in == NULL || real_out == NULL ||
        imag_out == NULL || out_nfft == NULL || n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_fft_radix4_next_size(n, &nfft);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    /* Copy input data */
    memcpy(real_out, real_in, n * sizeof(lmmc_real_t));
    memcpy(imag_out, imag_in, n * sizeof(lmmc_real_t));

    /* Zero-pad the remainder */
    if (nfft > n) {
        memset(real_out + n, 0, (nfft - n) * sizeof(lmmc_real_t));
        memset(imag_out + n, 0, (nfft - n) * sizeof(lmmc_real_t));
    }

    *out_nfft = nfft;
    return LMMC_STATUS_OK;
}
