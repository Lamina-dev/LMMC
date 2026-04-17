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
    double real_ref[MAX_NFFT] = {0.0};
    double imag_ref[MAX_NFFT] = {0.0};

    st = lmmc_fft_radix4_next_size(LOGICAL_N, &nfft);
    if (st != LMMC_STATUS_OK || nfft > MAX_NFFT) {
        printf("bad next_size: %s\n", lmmc_status_string(st));
        return 1;
    }

    for (i = 0; i < LOGICAL_N; ++i) {
        double x = (double)i;
        real_auto[i] = sin(0.35 * x) + 0.25 * cos(0.12 * x);
        imag_auto[i] = 0.0;
        real_ref[i] = real_auto[i];
        imag_ref[i] = imag_auto[i];
    }

    st = lmmc_fft_radix4_forward(real_auto, imag_auto, LOGICAL_N);
    if (st != LMMC_STATUS_OK) {
        printf("auto-pad forward FFT failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_fft_radix4_forward(real_ref, imag_ref, nfft);
    if (st != LMMC_STATUS_OK) {
        printf("explicit-pad forward FFT failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    printf("ll = %d\n", LOGICAL_N);
    printf("l = %zu\n", nfft);
    printf("1st %d bins:\n", LOGICAL_N);
    for (i = 0; i < LOGICAL_N; ++i) {
        printf("  k=%2zu: % .8f %+.8fi\n", i, real_auto[i], imag_auto[i]);
    }

    for (i = 0; i < LOGICAL_N; ++i) {
        st = lmmc_double_nearly_equal_tol(real_auto[i], real_ref[i], 1e-10, 1e-10, &equal);
        if (st != LMMC_STATUS_OK || equal != 1) {
            printf("real mismatch @ k=%zu\n", i);
            return 1;
        }
        st = lmmc_double_nearly_equal_tol(imag_auto[i], imag_ref[i], 1e-10, 1e-10, &equal);
        if (st != LMMC_STATUS_OK || equal != 1) {
            printf("imag mismatch @ k=%zu\n", i);
            return 1;
        }
    }

    printf("verified\n");
    return 0;
}
