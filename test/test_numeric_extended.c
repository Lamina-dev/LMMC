/**
 * @file test_numeric_extended.c
 * 针对 LMMC 中 numeric extended 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    int rc = 0;
    lmmc_status_t st = LMMC_STATUS_OK;
    int test_section = 0;


    test_section = 1;
    {
        lmmc_real_t res = 0.0;


        st = lmmc_atan2(0.0, 1.0, &res);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(res, 0.0, 1e-12)) {
            rc = 1; goto done;
        }


        st = lmmc_atan2(1.0, 0.0, &res);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(res, LMMC_PI / 2.0, 1e-12)) {
            rc = 1; goto done;
        }


        st = lmmc_atan2(0.0, -1.0, &res);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(res, LMMC_PI, 1e-12)) {
            rc = 1; goto done;
        }


        st = lmmc_atan2(-1.0, 0.0, &res);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(res, -LMMC_PI / 2.0, 1e-12)) {
            rc = 1; goto done;
        }
    }


    test_section = 2;
    {
        lmmc_real_t s = 0.0, c = 0.0;

        st = lmmc_sincos(0.0, &s, &c);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(s, 0.0, 1e-15)) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(c, 1.0, 1e-15)) { rc = 1; goto done; }
    }


    test_section = 3;
    {
        lmmc_real_t res = 0.0;


        st = lmmc_hypot(3.0, 4.0, &res);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(res, 5.0, 1e-12)) {
            rc = 1; goto done;
        }


        st = lmmc_hypot(1e300, 1e300, &res);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        if (!isfinite(res)) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(res, LMMC_SQRT2 * 1e300, 1e290)) {
            rc = 1; goto done;
        }
    }


    test_section = 5;
    {
        lmmc_real_t res = 0.0;


        st = lmmc_expm1(0.0, &res);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(res, 0.0, 1e-15)) {
            rc = 1; goto done;
        }


        st = lmmc_log1p(0.0, &res);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(res, 0.0, 1e-15)) {
            rc = 1; goto done;
        }


        st = lmmc_expm1(1e-15, &res);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        if (!lmmc_test_nearly_equal(res, 1e-15, 1e-28)) {
            rc = 1; goto done;
        }


        st = lmmc_log1p(1e-15, &res);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        if (!lmmc_test_nearly_equal(res, 1e-15, 1e-28)) {
            rc = 1; goto done;
        }
    }


    test_section = 7;
    {
        lmmc_real_t res = 0.0;


        st = lmmc_lambertw(0.0, &res);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(res, 0.0, 1e-12)) {
            rc = 1; goto done;
        }


        st = lmmc_lambertw(1.0, &res);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(res, 0.5671432904097838, 1e-10)) {
            rc = 1; goto done;
        }


        st = lmmc_lambertw(exp(1.0), &res);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(res, 1.0, 1e-10)) {
            rc = 1; goto done;
        }
    }


    test_section = 8;
    {
        lmmc_real_t test_values[] = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0};
        size_t n_tests = sizeof(test_values) / sizeof(test_values[0]);

        for (size_t i = 0; i < n_tests; i++) {
            lmmc_real_t x = test_values[i];
            lmmc_real_t w = 0.0;
            st = lmmc_lambertw(x, &w);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


            lmmc_real_t reconstructed = w * exp(w);
            if (!lmmc_test_nearly_equal(reconstructed, x, 1e-10)) {
                rc = 1; goto done;
            }
        }
    }


    test_section = 9;
    {
        const size_t n = 16;
        double real[16], imag[16];
        double real_orig[16], imag_orig[16];


        for (size_t i = 0; i < n; i++) {
            real[i] = sin(0.5 * (double)i) + 0.3 * cos(1.2 * (double)i);
            imag[i] = cos(0.7 * (double)i) - 0.2 * sin(0.9 * (double)i);
        }
        memcpy(real_orig, real, sizeof(real));
        memcpy(imag_orig, imag, sizeof(imag));

        st = lmmc_fft_radix4_forward(real, imag, n);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_fft_radix4_inverse(real, imag, n);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        for (size_t i = 0; i < n; i++) {
            if (!lmmc_test_nearly_equal(real[i], real_orig[i], 1e-10) ||
                !lmmc_test_nearly_equal(imag[i], imag_orig[i], 1e-10)) {
                rc = 1; goto done;
            }
        }


        {
            const size_t n2 = 64;
            double real2[64], imag2[64];
            double real2_orig[64], imag2_orig[64];

            for (size_t i = 0; i < n2; i++) {
                real2[i] = sin(0.3 * (double)i) + cos(0.7 * (double)i);
                imag2[i] = 0.0;
            }
            memcpy(real2_orig, real2, sizeof(real2));
            memcpy(imag2_orig, imag2, sizeof(imag2));

            st = lmmc_fft_radix4_forward(real2, imag2, n2);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

            st = lmmc_fft_radix4_inverse(real2, imag2, n2);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

            for (size_t i = 0; i < n2; i++) {
                if (!lmmc_test_nearly_equal(real2[i], real2_orig[i], 1e-10) ||
                    !lmmc_test_nearly_equal(imag2[i], imag2_orig[i], 1e-10)) {
                    rc = 1; goto done;
                }
            }
        }
    }


    test_section = 10;
    {

        {
            const size_t n = 4;
            double real[4] = {1.0, 2.0, 3.0, 4.0};
            double imag[4] = {0.0, 0.0, 0.0, 0.0};


            double time_energy = 0.0;
            for (size_t i = 0; i < n; i++) {
                time_energy += real[i] * real[i] + imag[i] * imag[i];
            }


            st = lmmc_fft_radix4_forward(real, imag, n);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


            double freq_energy = 0.0;
            for (size_t i = 0; i < n; i++) {
                freq_energy += real[i] * real[i] + imag[i] * imag[i];
            }


            if (!lmmc_test_nearly_equal(time_energy, freq_energy / (double)n, 1e-10)) {
                rc = 1; goto done;
            }
        }


        {
            const size_t n = 16;
            double real[16], imag[16];

            for (size_t i = 0; i < n; i++) {
                real[i] = sin(0.4 * (double)i) + 0.5 * cos(1.1 * (double)i);
                imag[i] = 0.2 * sin(0.8 * (double)i);
            }

            double time_energy = 0.0;
            for (size_t i = 0; i < n; i++) {
                time_energy += real[i] * real[i] + imag[i] * imag[i];
            }

            st = lmmc_fft_radix4_forward(real, imag, n);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

            double freq_energy = 0.0;
            for (size_t i = 0; i < n; i++) {
                freq_energy += real[i] * real[i] + imag[i] * imag[i];
            }

            if (!lmmc_test_nearly_equal(time_energy, freq_energy / (double)n, 1e-10)) {
                rc = 1; goto done;
            }
        }


        {
            const size_t n = 64;
            double real[64], imag[64];

            for (size_t i = 0; i < n; i++) {
                real[i] = cos(0.2 * (double)i) - 0.3 * sin(0.6 * (double)i);
                imag[i] = 0.0;
            }

            double time_energy = 0.0;
            for (size_t i = 0; i < n; i++) {
                time_energy += real[i] * real[i] + imag[i] * imag[i];
            }

            st = lmmc_fft_radix4_forward(real, imag, n);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

            double freq_energy = 0.0;
            for (size_t i = 0; i < n; i++) {
                freq_energy += real[i] * real[i] + imag[i] * imag[i];
            }

            if (!lmmc_test_nearly_equal(time_energy, freq_energy / (double)n, 1e-10)) {
                rc = 1; goto done;
            }
        }


        {
            const size_t n = 256;
            double real[256], imag[256];

            for (size_t i = 0; i < n; i++) {
                real[i] = sin(0.1 * (double)i) + 0.7 * cos(0.3 * (double)i);
                imag[i] = 0.4 * cos(0.5 * (double)i);
            }

            double time_energy = 0.0;
            for (size_t i = 0; i < n; i++) {
                time_energy += real[i] * real[i] + imag[i] * imag[i];
            }

            st = lmmc_fft_radix4_forward(real, imag, n);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

            double freq_energy = 0.0;
            for (size_t i = 0; i < n; i++) {
                freq_energy += real[i] * real[i] + imag[i] * imag[i];
            }

            if (!lmmc_test_nearly_equal(time_energy, freq_energy / (double)n, 1e-10)) {
                rc = 1; goto done;
            }
        }
    }


    test_section = 11;
    {
        lmmc_real_t res = 0.0;


        st = lmmc_lambertw(-0.5, &res);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1; goto done;
        }


        st = lmmc_lambertw(-1.0, &res);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1; goto done;
        }
    }


    test_section = 12;
    {
        lmmc_real_t res = 0.0;

        st = lmmc_nextafter(1.0, 2.0, &res);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


        lmmc_real_t expected = 1.0 + DBL_EPSILON;
        if (!lmmc_test_nearly_equal(res, expected, 0.0)) {
            rc = 1; goto done;
        }


        st = lmmc_nextafter(1.0, 0.0, &res);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (res >= 1.0) { rc = 1; goto done; }
    }

done:
    if (rc != 0) {
        printf("numeric extended test failed at section %d\n", test_section);
    }
    return rc;
}
