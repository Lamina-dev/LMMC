/**
 * @file example_fft_radix4.c
 * @brief 演示 LMMC 中 fft radix4 相关接口的使用。
 *
 * 展示 lmmc_fft_radix4_pad_into 显式填充辅助函数，以及
 * lmmc_fft 通用 N 点 FFT 入口。
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"

int main(void) {
    enum { LOGICAL_N = 10, MAX_NFFT = 16 };
    lmmc_status_t st = LMMC_STATUS_OK;
    size_t nfft = 0;
    size_t i = 0;
    int equal = 0;
    double real_auto[LOGICAL_N] = {0.0};
    double imag_auto[LOGICAL_N] = {0.0};
    double real_padded[MAX_NFFT] = {0.0};
    double imag_padded[MAX_NFFT] = {0.0};

    for (i = 0; i < LOGICAL_N; ++i) {
        double x = (double)i;
        real_auto[i] = sin(0.35 * x) + 0.25 * cos(0.12 * x);
        imag_auto[i] = 0.0;
    }

    /* Use lmmc_fft_radix4_pad_into for explicit zero-padding to next power of 4 */
    st = lmmc_fft_radix4_pad_into(real_auto, imag_auto, LOGICAL_N,
                                   real_padded, imag_padded, &nfft);
    if (st != LMMC_STATUS_OK || nfft > MAX_NFFT) {
        printf("pad_into failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    /* Now use the strict radix-4 on the padded data */
    st = lmmc_fft_radix4_forward(real_padded, imag_padded, nfft);
    if (st != LMMC_STATUS_OK) {
        printf("radix4 forward FFT failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    /* Also demonstrate lmmc_fft for arbitrary-length N-point FFT */
    st = lmmc_fft_forward(real_auto, imag_auto, LOGICAL_N);
    if (st != LMMC_STATUS_OK) {
        printf("N-point forward FFT failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    printf("logical_n = %d\n", LOGICAL_N);
    printf("padded_n = %zu\n", nfft);
    printf("N-point FFT result (first %d bins):\n", LOGICAL_N);
    for (i = 0; i < LOGICAL_N; ++i) {
        printf("  k=%2zu: % .8f %+.8fi\n", i, real_auto[i], imag_auto[i]);
    }

    /* Verify round-trip with lmmc_fft */
    {
        double real_rt[LOGICAL_N], imag_rt[LOGICAL_N];
        for (i = 0; i < LOGICAL_N; ++i) {
            double x = (double)i;
            real_rt[i] = sin(0.35 * x) + 0.25 * cos(0.12 * x);
            imag_rt[i] = 0.0;
        }
        st = lmmc_fft_forward(real_rt, imag_rt, LOGICAL_N);
        if (st != LMMC_STATUS_OK) { printf("fwd failed\n"); return 1; }
        st = lmmc_fft_inverse(real_rt, imag_rt, LOGICAL_N);
        if (st != LMMC_STATUS_OK) { printf("inv failed\n"); return 1; }

        for (i = 0; i < LOGICAL_N; ++i) {
            double x = (double)i;
            double expected = sin(0.35 * x) + 0.25 * cos(0.12 * x);
            st = lmmc_double_nearly_equal_tol(real_rt[i], expected, 1e-9, 1e-9, &equal);
            if (st != LMMC_STATUS_OK || equal != 1) {
                printf("round-trip mismatch @ k=%zu\n", i);
                return 1;
            }
        }
    }

    printf("verified\n");
    return 0;
}
