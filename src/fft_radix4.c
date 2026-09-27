#include <math.h>
#include "internal.h"
#include "lmmc/fft.h"
#include "lmmc/numeric_scalar.h"


static size_t lmmc_reverse_base4(size_t value, unsigned digits) {
    size_t reversed = 0;
    unsigned i = 0;

    for (i = 0; i < digits; ++i) {
        reversed = (reversed << 2) | (value & (size_t)3);
        value >>= 2;
    }
    return reversed;
}

lmmc_status_t lmmc_fft_radix4_next_size(size_t n, size_t* out_nfft) {
    size_t nfft = 1;

    if (out_nfft == NULL || n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    while (nfft < n) {
        if (nfft > ((size_t)-1) / (size_t)4) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        nfft *= (size_t)4;
    }

    *out_nfft = nfft;
    return LMMC_STATUS_OK;
}

/** @brief 蝶形运算先完成全部读取与旋转，再写回各元素。 */
static void fft_radix4_butterfly(
    lmmc_real_t* real, lmmc_real_t* imag, size_t quarter,
    double angle, int inverse)
{
    lmmc_real_t ar[4], ai[4];
    size_t k;
    ar[0] = real[0];
    ai[0] = imag[0];
    for (k = 1; k < 4; ++k) {
        const double phase = angle * (double)k;
        const double c = cos(phase);
        const double s = sin(phase);
        ar[k] = real[k * quarter] * c - imag[k * quarter] * s;
        ai[k] = real[k * quarter] * s + imag[k * quarter] * c;
    }

    real[0] = ((ar[0] + ar[1]) + ar[2]) + ar[3];
    imag[0] = ((ai[0] + ai[1]) + ai[2]) + ai[3];
    real[2 * quarter] = ((ar[0] - ar[1]) + ar[2]) - ar[3];
    imag[2 * quarter] = ((ai[0] - ai[1]) + ai[2]) - ai[3];
    if (!inverse) {
        real[quarter] = ((ar[0] + ai[1]) - ar[2]) - ai[3];
        imag[quarter] = ((ai[0] - ar[1]) - ai[2]) + ar[3];
        real[3 * quarter] = ((ar[0] - ai[1]) - ar[2]) + ai[3];
        imag[3 * quarter] = ((ai[0] + ar[1]) - ai[2]) - ar[3];
    } else {
        real[quarter] = ((ar[0] - ai[1]) - ar[2]) + ai[3];
        imag[quarter] = ((ai[0] + ar[1]) - ai[2]) - ar[3];
        real[3 * quarter] = ((ar[0] + ai[1]) - ar[2]) - ai[3];
        imag[3 * quarter] = ((ai[0] - ar[1]) - ai[2]) + ar[3];
    }
}

static void fft_radix4_stage(
    lmmc_real_t* real, lmmc_real_t* imag, size_t n,
    size_t group, int inverse)
{
    const size_t quarter = group / 4;
    const double step = (inverse ? 2.0 : -2.0) * LMMC_PI / (double)group;
    size_t i, j;
    for (i = 0; i < n; i += group) {
        for (j = 0; j < quarter; ++j) {
            fft_radix4_butterfly(real + i + j, imag + i + j,
                                 quarter, step * (double)j, inverse);
        }
    }
}

static lmmc_status_t lmmc_fft_radix4_core(
    lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse)
{
    unsigned digits = 0;
    size_t i, len;
    if (!lmmc_is_power_of_four(n, &digits)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n <= 1) { return LMMC_STATUS_OK; }
    for (i = 0; i < n; ++i) {
        const size_t r = lmmc_reverse_base4(i, digits);
        if (r > i) {
            lmmc_swap(&real[i], &real[r]);
            lmmc_swap(&imag[i], &imag[r]);
        }
    }
    for (len = 4; len <= n; len *= 4) {
        fft_radix4_stage(real, imag, n, len, inverse);
        if (len == n) { break; }
    }
    if (inverse) {
        const double scale = 1.0 / (double)n;
        for (i = 0; i < n; ++i) {
            real[i] *= scale;
            imag[i] *= scale;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_fft_radix4(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse) {
    unsigned digits = 0;
    lmmc_storage_envelope_t real_envelope;
    lmmc_storage_envelope_t imag_envelope;

    if (real == NULL || imag == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_is_power_of_four(n, &digits) ||
        !lmmc_storage_envelope_checked(
            real, 1, n, n, sizeof(lmmc_real_t), &real_envelope) ||
        !lmmc_storage_envelope_checked(
            imag, 1, n, n, sizeof(lmmc_real_t), &imag_envelope) ||
        lmmc_storage_envelopes_overlap(&real_envelope, &imag_envelope)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    return lmmc_fft_radix4_core(real, imag, n, inverse ? 1 : 0);
}

lmmc_status_t lmmc_fft_radix4_forward(lmmc_real_t* real, lmmc_real_t* imag, size_t n) {
    return lmmc_fft_radix4(real, imag, n, 0);
}

lmmc_status_t lmmc_fft_radix4_inverse(lmmc_real_t* real, lmmc_real_t* imag, size_t n) {
    return lmmc_fft_radix4(real, imag, n, 1);
}
