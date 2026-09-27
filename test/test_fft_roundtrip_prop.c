/**
 * @file test_fft_roundtrip_prop.c
 * FFT 往返精度属性测试。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define RNG_SEED UINT64_C(0xCAFEBABE1234)
#define NUM_ROUNDTRIP_TRIALS 200
#define NUM_SPECTRAL_TRIALS 100

static uint64_t g_rng_state = RNG_SEED;

static void rng_seed(uint64_t s) { g_rng_state = (s == 0) ? UINT64_C(1) : s; }

static uint64_t rng_u64(void) {
    uint64_t x = g_rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng_state = x;
    return x * UINT64_C(2685821657736338717);
}

/**
 * @brief Generate a uniform random double in [lo, hi).
 */
static double rng_uniform(double lo, double hi) {
    double u = (double)(rng_u64() >> 11) * (1.0 / 9007199254740992.0);
    return lo + (hi - lo) * u;
}

/**
 * @brief Generate a random integer in [lo, hi] inclusive.
 */
static size_t rng_int(size_t lo, size_t hi) {
    if (lo >= hi) {
        return lo;
    }
    uint64_t range = (uint64_t)(hi - lo + 1);
    return lo + (size_t)(rng_u64() % range);
}

/**
 * @brief Generate a random N value from a mix of:
 *   - Powers of 2
 *   - Powers of 4
 *   - Primes
 *   - Arbitrary values in [1, 2048]
 */
static size_t generate_n(int trial_idx) {
    /* Some fixed interesting values */
    static const size_t powers_of_2[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    static const size_t powers_of_4[] = {1, 4, 16, 64, 256, 1024};
    static const size_t primes[] = {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                                    47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 127,
                                    131, 251, 257, 509, 521, 1021, 1031, 1999, 2039};

    switch (trial_idx % 5) {
    case 0:
        /* Power of 2 */
        return powers_of_2[rng_int(0, 11)];
    case 1:
        /* Power of 4 */
        return powers_of_4[rng_int(0, 5)];
    case 2:
        /* Prime */
        return primes[rng_int(0, 36)];
    case 3:
        /* Small arbitrary */
        return rng_int(1, 64);
    default:
        /* Larger arbitrary */
        return rng_int(1, 2048);
    }
}

typedef struct {
    lmmc_real_t real_orig[2048];
    lmmc_real_t imag_orig[2048];
    lmmc_real_t real_work[2048];
    lmmc_real_t imag_work[2048];
} roundtrip_workspace_t;

static int setup(void **state) {
    roundtrip_workspace_t *work = calloc(1, sizeof(*work));
    assert_non_null(work);
    *state = work;
    rng_seed(RNG_SEED);
    return 0;
}

static int teardown(void **state) {
    free(*state);
    return 0;
}

static double generate_roundtrip_input(roundtrip_workspace_t *work, size_t N) {
    double max_abs = 0.0;
    for (size_t j = 0; j < N; j++) {
        work->real_orig[j] = rng_uniform(-10.0, 10.0);
        work->imag_orig[j] = rng_uniform(-10.0, 10.0);
        work->real_work[j] = work->real_orig[j];
        work->imag_work[j] = work->imag_orig[j];
        double mag = sqrt(work->real_orig[j] * work->real_orig[j] +
                          work->imag_orig[j] * work->imag_orig[j]);
        if (mag > max_abs) {
            max_abs = mag;
        }
    }
    return max_abs;
}

static void check_roundtrip_error(
    const roundtrip_workspace_t *work, size_t N, double max_abs) {
    double max_err = 0.0;
    for (size_t j = 0; j < N; j++) {
        double err_re = fabs(work->real_work[j] - work->real_orig[j]);
        double err_im = fabs(work->imag_work[j] - work->imag_orig[j]);
        double err = (err_re > err_im) ? err_re : err_im;
        if (err > max_err) {
            max_err = err;
        }
    }
    const double threshold = 1e-9 * (1.0 + max_abs);
    assert_false(max_err > threshold);
}

static void check_spectral_bins(
    const lmmc_real_t *real_buf, const lmmc_real_t *imag_buf, size_t N, size_t k) {
    const double tol = 1e-9 * sqrt((double)N);
    double peak_mag = 0.0;
    double max_off_peak = 0.0;
    for (size_t j = 0; j < N; j++) {
        double mag = sqrt(real_buf[j] * real_buf[j] + imag_buf[j] * imag_buf[j]);
        if (j == k) {
            peak_mag = mag;
        } else if (mag > max_off_peak) {
            max_off_peak = mag;
        }
    }
    assert_false(fabs(peak_mag - (double)N) > tol);
    assert_false(max_off_peak > tol);
}

static void test_fft_rejects_unrepresentable_workspace(void **state) {
    (void)state;
    lmmc_real_t real = 0.0;
    lmmc_real_t imag = 0.0;
    lmmc_status_t status = lmmc_fft_forward(&real, &imag, SIZE_MAX);
    assert_false(status != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_fft_roundtrip(void **state) {
    roundtrip_workspace_t *work = *state;
    for (int trial = 0; trial < NUM_ROUNDTRIP_TRIALS; ++trial) {
        const size_t N = generate_n(trial);
        const double max_abs = generate_roundtrip_input(work, N);
        assert_int_equal(lmmc_fft_forward(work->real_work, work->imag_work, N), LMMC_STATUS_OK);
        assert_int_equal(lmmc_fft_inverse(work->real_work, work->imag_work, N), LMMC_STATUS_OK);
        check_roundtrip_error(work, N, max_abs);
    }
}

static void test_spectral_purity(void **state) {
    roundtrip_workspace_t *work = *state;
    for (int trial = 0; trial < NUM_SPECTRAL_TRIALS; ++trial) {
        size_t N = generate_n(trial);
        if (N <= 1) {
            N = 2 + rng_int(0, 100);
        }
        const size_t k = rng_int(0, N - 1);
        for (size_t j = 0; j < N; ++j) {
            const double angle = 2.0 * LMMC_PI * (double)k * (double)j / (double)N;
            work->real_work[j] = cos(angle);
            work->imag_work[j] = sin(angle);
        }
        assert_int_equal(lmmc_fft_forward(work->real_work, work->imag_work, N), LMMC_STATUS_OK);
        check_spectral_bins(work->real_work, work->imag_work, N, k);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_fft_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_spectral_purity, setup, teardown),
        cmocka_unit_test(test_fft_rejects_unrepresentable_workspace),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
