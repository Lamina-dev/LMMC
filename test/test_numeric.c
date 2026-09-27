/**
 * @file test_numeric.c
 * 针对 LMMC 中 numeric 相关接口的单元测试。
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static int lmmc_test_close(double a, double b, double abs_tol, double rel_tol) {
    int equal = 0;
    lmmc_status_t st = lmmc_double_nearly_equal_tol(a, b, abs_tol, rel_tol, &equal);
    return st == LMMC_STATUS_OK && equal == 1;
}

static void test_comparison_precision(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    int equal = 0;
    st = lmmc_double_nearly_equal(0.1 + 0.2, 0.3, &equal);
    assert_false(st != LMMC_STATUS_OK || equal != 1);

    st = lmmc_double_nearly_equal(1.0, 1.0 + 1e-6, &equal);
    assert_false(st != LMMC_STATUS_OK || equal != 0);

    st = lmmc_double_nearly_equal_tol(1.0, 1.0 + 1e-6, 1e-9, 1e-5, &equal);
    assert_false(st != LMMC_STATUS_OK || equal != 1);
}

static void test_comparison_arguments(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    int equal = 0;
    st = lmmc_double_nearly_equal_tol(1.0, 1.0, -1.0, 1e-9, &equal);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_double_nearly_equal_tol(1.0, 1.0, 1e-9, 1e-9, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_double_nearly_equal_tol(NAN, 1.0, 1e-9, 1e-9, &equal);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_comparison_infinity_and_tolerance(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    int equal = 0;
    st = lmmc_double_nearly_equal_tol(INFINITY, INFINITY, 1e-9, 1e-9, &equal);
    assert_false(st != LMMC_STATUS_OK || equal != 1);

    equal = 7;
    st = lmmc_double_nearly_equal_tol(1.0, 1.15, 0.1, 0.1, &equal);
    assert_false(st != LMMC_STATUS_OK || equal != 0);
}

static void test_comparison_transactional_errors(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    int equal = 0;
    equal = 7;
    st = lmmc_approx_eq(1.0, 1.0, -1.0, &equal);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT || equal != 7);

    st = lmmc_approx_eq(1.0, 1.0, NAN, &equal);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT || equal != 7);
}

static void test_remainder_values(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t remainder = 7.0;
    st = lmmc_fmod(-3.5, INFINITY, &remainder);
    assert_false(st != LMMC_STATUS_OK || remainder != -3.5);
    st = lmmc_fmod(-4.0, 2.0, &remainder);
    assert_false(st != LMMC_STATUS_OK || remainder != 0.0 || !signbit(remainder));
    st = lmmc_fmod(DBL_MAX, DBL_MIN, &remainder);
    assert_false(st != LMMC_STATUS_OK || remainder != 0.0);
}

static void test_remainder_domains(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t remainder = 7.0;
    st = lmmc_fmod(1.0, 0.0, &remainder);
    assert_false(st != LMMC_STATUS_OUT_OF_RANGE || remainder != 7.0);
    st = lmmc_fmod(INFINITY, 2.0, &remainder);
    if (st != LMMC_STATUS_OUT_OF_RANGE || remainder != 7.0) {
        fail_msg("fmod infinite dividend must report a domain error\n");
    }
    st = lmmc_fmod(1.0, NAN, &remainder);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT || remainder != 7.0);
    st = lmmc_fmod(NAN, 1.0, &remainder);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT || remainder != 7.0);
}

static void test_hypot_extremes(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t length = 7.0;
    st = lmmc_hypot(DBL_MAX, DBL_MAX, &length);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || length != 7.0) {
        fail_msg("hypot overflow must fail without modifying output\n");
    }
    st = lmmc_hypot(NAN, 1.0, &length);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT || length != 7.0);
    st = lmmc_hypot(1.0, INFINITY, &length);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT || length != 7.0);
    st = lmmc_hypot(DBL_MAX / 2.0, DBL_MAX / 2.0, &length);
    assert_false(st != LMMC_STATUS_OK || !isfinite(length) ||
                 fabs(length / (DBL_MAX / 2.0) - sqrt(2.0)) > 1e-15);
    st = lmmc_hypot(DBL_MIN / 2.0, 0.0, &length);
    assert_false(st != LMMC_STATUS_OK || length != DBL_MIN / 2.0);
}

static void test_log2_domain(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t logarithm = 7.0;
    st = lmmc_log2(-1.0, &logarithm);
    assert_false(st != LMMC_STATUS_OUT_OF_RANGE || logarithm != 7.0);
    st = lmmc_log2(0.0, &logarithm);
    assert_false(st != LMMC_STATUS_OUT_OF_RANGE || logarithm != 7.0);
    st = lmmc_log2(8.0, &logarithm);
    assert_false(st != LMMC_STATUS_OK || logarithm != 3.0);
}

static void test_exp2_overflow(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t exponential = 7.0;
    st = lmmc_exp2(1024.0, &exponential);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE || exponential != 7.0);
    st = lmmc_exp2(10.0, &exponential);
    assert_false(st != LMMC_STATUS_OK || exponential != 1024.0);
}

static void test_expm1_overflow(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t exponential = 7.0;
    st = lmmc_expm1(1000.0, &exponential);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE || exponential != 7.0);
    st = lmmc_expm1(1.0, &exponential);
    assert_false(st != LMMC_STATUS_OK ||
                 fabs(exponential - 1.718281828459045) > 1e-15);
}

static void test_ldexp_range(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t scaled = 7.0;
    st = lmmc_ldexp(1.0, 1024, &scaled);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE || scaled != 7.0);
    st = lmmc_ldexp(1.0, -1075, &scaled);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE || scaled != 7.0);
    st = lmmc_ldexp(0.5, 4, &scaled);
    assert_false(st != LMMC_STATUS_OK || scaled != 8.0);
}

static void test_log1p_domain(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t logarithm = 7.0;
    st = lmmc_log1p(-2.0, &logarithm);
    assert_false(st != LMMC_STATUS_OUT_OF_RANGE || logarithm != 7.0);
    st = lmmc_log1p(-1.0, &logarithm);
    assert_false(st != LMMC_STATUS_OUT_OF_RANGE || logarithm != 7.0);
    st = lmmc_log1p(3.0, &logarithm);
    assert_false(st != LMMC_STATUS_OK || fabs(logarithm - log(4.0)) > 1e-15);
}

static void test_fft_impulse(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    size_t i = 0;
    const size_t n = 16;
    double real[16] = {0.0};
    double imag[16] = {0.0};

    LMMC_REAL_SET_D(&real[0], 1.0);
    st = lmmc_fft_radix4_forward(real, imag, n);
    assert_false(st != LMMC_STATUS_OK);

    for (i = 0; i < n; ++i) {
        assert_false(!lmmc_test_close(real[i], 1.0, 1e-12, 1e-10) || !lmmc_test_close(imag[i], 0.0, 1e-12, 1e-10));
    }
}

static void test_fft_complex_roundtrip(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
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
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_fft_radix4_inverse(real, imag, n);
    assert_false(st != LMMC_STATUS_OK);

    for (i = 0; i < n; ++i) {
        assert_false(!lmmc_test_close(real[i], real_ref[i], 1e-10, 1e-10) || !lmmc_test_close(imag[i], imag_ref[i], 1e-10, 1e-10));
    }
}

static void test_fft_next_size(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    size_t nfft = 0;

    st = lmmc_fft_radix4_next_size(10, &nfft);
    assert_false(st != LMMC_STATUS_OK || nfft != 16);

    st = lmmc_fft_radix4_next_size(16, &nfft);
    assert_false(st != LMMC_STATUS_OK || nfft != 16);

    st = lmmc_fft_radix4_next_size(0, &nfft);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_fft_radix4_next_size(4, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_fft_strict_length(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
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
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    for (i = 0; i < n; ++i) {
        double x = (double)i;
        real_auto[i] = sin(0.27 * x) + 0.2 * cos(0.13 * x);
        imag_auto[i] = cos(0.19 * x) - 0.3 * sin(0.07 * x);
    }
    st = lmmc_fft_forward(real_auto, imag_auto, n);
    assert_false(st != LMMC_STATUS_OK);

    st = lmmc_fft_radix4_forward(real_ref, imag_ref, nfft);
    assert_false(st != LMMC_STATUS_OK);
}

static void test_fft_arbitrary_roundtrip(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    size_t i = 0;
    const size_t n = 10;
    double real_auto[10] = {0.0};
    double imag_auto[10] = {0.0};
    double real_orig[10] = {0.0};
    double imag_orig[10] = {0.0};

    for (i = 0; i < n; ++i) {
        double x = (double)i;
        real_auto[i] = 0.5 * cos(0.41 * x) - 0.2 * sin(0.23 * x);
        imag_auto[i] = 0.6 * sin(0.17 * x) + 0.1 * cos(0.29 * x);
        real_orig[i] = real_auto[i];
        imag_orig[i] = imag_auto[i];
    }

    st = lmmc_fft_radix4_inverse(real_auto, imag_auto, n);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_fft_forward(real_auto, imag_auto, n);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_fft_inverse(real_auto, imag_auto, n);
    assert_false(st != LMMC_STATUS_OK);

    for (i = 0; i < n; ++i) {
        assert_false(!lmmc_test_close(real_auto[i], real_orig[i], 1e-9, 1e-9) || !lmmc_test_close(imag_auto[i], imag_orig[i], 1e-9, 1e-9));
    }
}

static void test_fft_arguments(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    double real4[4] = {0.0};
    double imag4[4] = {0.0};

    st = lmmc_fft_radix4(NULL, imag4, 4, 0);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    st = lmmc_fft_radix4(real4, NULL, 4, 0);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    st = lmmc_fft_radix4(real4, imag4, 0, 0);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    st = lmmc_fft_radix4(real4, imag4, 4, 2);
    assert_false(st != LMMC_STATUS_OK);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_comparison_precision),
        cmocka_unit_test(test_comparison_arguments),
        cmocka_unit_test(test_comparison_infinity_and_tolerance),
        cmocka_unit_test(test_comparison_transactional_errors),
        cmocka_unit_test(test_remainder_values),
        cmocka_unit_test(test_remainder_domains),
        cmocka_unit_test(test_hypot_extremes),
        cmocka_unit_test(test_log2_domain),
        cmocka_unit_test(test_exp2_overflow),
        cmocka_unit_test(test_expm1_overflow),
        cmocka_unit_test(test_ldexp_range),
        cmocka_unit_test(test_log1p_domain),
        cmocka_unit_test(test_fft_impulse),
        cmocka_unit_test(test_fft_complex_roundtrip),
        cmocka_unit_test(test_fft_next_size),
        cmocka_unit_test(test_fft_strict_length),
        cmocka_unit_test(test_fft_arbitrary_roundtrip),
        cmocka_unit_test(test_fft_arguments),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
