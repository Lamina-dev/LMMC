#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"

static int lmmc_test_close(double a, double b, double abs_tol, double rel_tol) {
    int equal = 0;
    lmmc_status_t st = lmmc_double_nearly_equal_tol(a, b, abs_tol, rel_tol, &equal);
    return st == LMMC_STATUS_OK && equal == 1;
}

int main(void) {
    lmmc_status_t st = LMMC_STATUS_OK;
    int equal = 0;
    int rc = 0;

    st = lmmc_double_nearly_equal(0.1 + 0.2, 0.3, &equal);
    if (st != LMMC_STATUS_OK || equal != 1) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal(1.0, 1.0 + 1e-6, &equal);
    if (st != LMMC_STATUS_OK || equal != 0) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal_tol(1.0, 1.0 + 1e-6, 1e-9, 1e-5, &equal);
    if (st != LMMC_STATUS_OK || equal != 1) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal_tol(1.0, 1.0, -1.0, 1e-9, &equal);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal_tol(1.0, 1.0, 1e-9, 1e-9, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal_tol(NAN, 1.0, 1e-9, 1e-9, &equal);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal_tol(INFINITY, INFINITY, 1e-9, 1e-9, &equal);
    if (st != LMMC_STATUS_OK || equal != 1) {
        rc = 1;
        goto done;
    }

    {
        size_t i = 0;
        const size_t n = 16;
        double real[16] = {0.0};
        double imag[16] = {0.0};

        real[0] = 1.0;
        st = lmmc_fft_radix4_forward(real, imag, n);
        if (st != LMMC_STATUS_OK) {
            rc = 1;
            goto done;
        }

        for (i = 0; i < n; ++i) {
            if (!lmmc_test_close(real[i], 1.0, 1e-12, 1e-10) || !lmmc_test_close(imag[i], 0.0, 1e-12, 1e-10)) {
                rc = 1;
                goto done;
            }
        }
    }

    {
        size_t i = 0;
        const size_t n = 16;
        double real[16] = {0.0};
        double imag[16] = {0.0};
        double real_ref[16] = {0.0};
        double imag_ref[16] = {0.0};

        for (i = 0; i < n; ++i) {
            double x = (double)i;
            real[i] = sin(0.31 * x) + 0.5 * cos(0.17 * x);
            imag[i] = cos(0.21 * x) - 0.4 * sin(0.11 * x);
        }

        memcpy(real_ref, real, sizeof(real));
        memcpy(imag_ref, imag, sizeof(imag));

        st = lmmc_fft_radix4_forward(real, imag, n);
        if (st != LMMC_STATUS_OK) {
            rc = 1;
            goto done;
        }
        st = lmmc_fft_radix4_inverse(real, imag, n);
        if (st != LMMC_STATUS_OK) {
            rc = 1;
            goto done;
        }

        for (i = 0; i < n; ++i) {
            if (!lmmc_test_close(real[i], real_ref[i], 1e-10, 1e-10) || !lmmc_test_close(imag[i], imag_ref[i], 1e-10, 1e-10)) {
                rc = 1;
                goto done;
            }
        }
    }

    {
        size_t nfft = 0;

        st = lmmc_fft_radix4_next_size(10, &nfft);
        if (st != LMMC_STATUS_OK || nfft != 16) {
            rc = 1;
            goto done;
        }

        st = lmmc_fft_radix4_next_size(16, &nfft);
        if (st != LMMC_STATUS_OK || nfft != 16) {
            rc = 1;
            goto done;
        }

        st = lmmc_fft_radix4_next_size(0, &nfft);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1;
            goto done;
        }

        st = lmmc_fft_radix4_next_size(4, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1;
            goto done;
        }
    }

    {
        size_t i = 0;
        const size_t n = 10;
        const size_t nfft = 16;
        double real_auto[10] = {0.0};
        double imag_auto[10] = {0.0};
        double real_ref[16] = {0.0};
        double imag_ref[16] = {0.0};

        for (i = 0; i < n; ++i) {
            double x = (double)i;
            real_auto[i] = sin(0.27 * x) + 0.2 * cos(0.13 * x);
            imag_auto[i] = cos(0.19 * x) - 0.3 * sin(0.07 * x);
            real_ref[i] = real_auto[i];
            imag_ref[i] = imag_auto[i];
        }

        st = lmmc_fft_radix4_forward(real_auto, imag_auto, n);
        if (st != LMMC_STATUS_OK) {
            rc = 1;
            goto done;
        }

        st = lmmc_fft_radix4_forward(real_ref, imag_ref, nfft);
        if (st != LMMC_STATUS_OK) {
            rc = 1;
            goto done;
        }

        for (i = 0; i < n; ++i) {
            if (!lmmc_test_close(real_auto[i], real_ref[i], 1e-10, 1e-10) || !lmmc_test_close(imag_auto[i], imag_ref[i], 1e-10, 1e-10)) {
                rc = 1;
                goto done;
            }
        }
    }

    {
        size_t i = 0;
        const size_t n = 10;
        const size_t nfft = 16;
        double real_auto[10] = {0.0};
        double imag_auto[10] = {0.0};
        double real_ref[16] = {0.0};
        double imag_ref[16] = {0.0};

        for (i = 0; i < n; ++i) {
            double x = (double)i;
            real_auto[i] = 0.5 * cos(0.41 * x) - 0.2 * sin(0.23 * x);
            imag_auto[i] = 0.6 * sin(0.17 * x) + 0.1 * cos(0.29 * x);
            real_ref[i] = real_auto[i];
            imag_ref[i] = imag_auto[i];
        }

        st = lmmc_fft_radix4_inverse(real_auto, imag_auto, n);
        if (st != LMMC_STATUS_OK) {
            rc = 1;
            goto done;
        }

        st = lmmc_fft_radix4_inverse(real_ref, imag_ref, nfft);
        if (st != LMMC_STATUS_OK) {
            rc = 1;
            goto done;
        }

        for (i = 0; i < n; ++i) {
            if (!lmmc_test_close(real_auto[i], real_ref[i], 1e-10, 1e-10) || !lmmc_test_close(imag_auto[i], imag_ref[i], 1e-10, 1e-10)) {
                rc = 1;
                goto done;
            }
        }
    }

    {
        double real4[4] = {0.0};
        double imag4[4] = {0.0};

        st = lmmc_fft_radix4(NULL, imag4, 4, 0);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1;
            goto done;
        }
        st = lmmc_fft_radix4(real4, NULL, 4, 0);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1;
            goto done;
        }
        st = lmmc_fft_radix4(real4, imag4, 0, 0);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1;
            goto done;
        }
        st = lmmc_fft_radix4(real4, imag4, 4, 2);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1;
            goto done;
        }
    }

done:
    if (rc != 0) {
        printf("numeric test failed\n");
    }
    return rc;
}
