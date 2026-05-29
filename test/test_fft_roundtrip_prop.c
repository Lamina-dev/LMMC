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

/* ===================== Simple xorshift64 RNG ===================== */

#define RNG_SEED 0xCAFEBABE1234ULL
#define NUM_ROUNDTRIP_TRIALS 200
#define NUM_SPECTRAL_TRIALS 100

static uint64_t g_rng_state = RNG_SEED;

static void rng_seed(uint64_t s) { g_rng_state = (s == 0) ? 1ULL : s; }

static uint64_t rng_u64(void)
{
    uint64_t x = g_rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng_state = x;
    return x * 2685821657736338717ULL;
}

/**
 * @brief Generate a uniform random double in [lo, hi).
 */
static double rng_uniform(double lo, double hi)
{
    double u = (double)(rng_u64() >> 11) * (1.0 / 9007199254740992.0);
    return lo + (hi - lo) * u;
}

/**
 * @brief Generate a random integer in [lo, hi] inclusive.
 */
static size_t rng_int(size_t lo, size_t hi)
{
    if (lo >= hi) return lo;
    uint64_t range = (uint64_t)(hi - lo + 1);
    return lo + (size_t)(rng_u64() % range);
}

/* ===================== Helper: generate interesting N values ===================== */

/**
 * @brief Generate a random N value from a mix of:
 *   - Powers of 2
 *   - Powers of 4
 *   - Primes
 *   - Arbitrary values in [1, 2048]
 */
static size_t generate_n(int trial_idx)
{
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

/* ===================== Property Test: Round-trip FFT ===================== */

/**
 * @brief Test that forward FFT followed by inverse FFT recovers the original
 *        input within 1e-9 * (1 + max|x|) for various N in [1, 2048].
 *
 * For each trial:
 *   1. Generate random complex input of length N
 *   2. Copy the input
 *   3. Apply forward FFT then inverse FFT
 *   4. Verify max_i |x_recovered - x_original| <= 1e-9 * (1 + max|x|)
 *
 * Returns 0 on success, 1 on failure.
 */
static int test_fft_roundtrip(void)
{
    int failures = 0;
    int i;

    printf("  Testing FFT round-trip with %d random inputs...\n", NUM_ROUNDTRIP_TRIALS);

    for (i = 0; i < NUM_ROUNDTRIP_TRIALS; i++) {
        size_t N = generate_n(i);
        lmmc_real_t *real_orig = NULL, *imag_orig = NULL;
        lmmc_real_t *real_work = NULL, *imag_work = NULL;
        lmmc_status_t st;
        double max_abs = 0.0;
        double max_err = 0.0;
        double threshold;
        size_t j;

        /* Allocate arrays */
        real_orig = (lmmc_real_t*)malloc(N * sizeof(lmmc_real_t));
        imag_orig = (lmmc_real_t*)malloc(N * sizeof(lmmc_real_t));
        real_work = (lmmc_real_t*)malloc(N * sizeof(lmmc_real_t));
        imag_work = (lmmc_real_t*)malloc(N * sizeof(lmmc_real_t));

        if (!real_orig || !imag_orig || !real_work || !imag_work) {
            printf("    FAIL: allocation failed for N=%zu\n", N);
            free(real_orig); free(imag_orig);
            free(real_work); free(imag_work);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        /* Generate random complex input and compute max|x| */
        for (j = 0; j < N; j++) {
            real_orig[j] = rng_uniform(-10.0, 10.0);
            imag_orig[j] = rng_uniform(-10.0, 10.0);
            real_work[j] = real_orig[j];
            imag_work[j] = imag_orig[j];

            double mag = sqrt(real_orig[j] * real_orig[j] + imag_orig[j] * imag_orig[j]);
            if (mag > max_abs) max_abs = mag;
        }

        /* Forward FFT */
        st = lmmc_fft_forward(real_work, imag_work, N);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: lmmc_fft_forward returned %d for N=%zu\n", (int)st, N);
            free(real_orig); free(imag_orig);
            free(real_work); free(imag_work);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        /* Inverse FFT */
        st = lmmc_fft_inverse(real_work, imag_work, N);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: lmmc_fft_inverse returned %d for N=%zu\n", (int)st, N);
            free(real_orig); free(imag_orig);
            free(real_work); free(imag_work);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        /* Compute max error */
        for (j = 0; j < N; j++) {
            double err_re = fabs(real_work[j] - real_orig[j]);
            double err_im = fabs(imag_work[j] - imag_orig[j]);
            double err = (err_re > err_im) ? err_re : err_im;
            if (err > max_err) max_err = err;
        }

        /* Check tolerance: max error <= 1e-9 * (1 + max|x|) */
        threshold = 1e-9 * (1.0 + max_abs);

        if (max_err > threshold) {
            printf("    FAIL: N=%zu, max_err=%.3e > threshold=%.3e (max|x|=%.3e)\n",
                   N, max_err, threshold, max_abs);
            failures++;
            if (failures >= 5) {
                printf("    ... stopping after 5 failures\n");
                free(real_orig); free(imag_orig);
                free(real_work); free(imag_work);
                break;
            }
        }

        free(real_orig); free(imag_orig);
        free(real_work); free(imag_work);
    }

    if (failures > 0) {
        printf("    Round-trip: %d/%d failures\n", failures, NUM_ROUNDTRIP_TRIALS);
        return 1;
    }

    printf("    Round-trip: all %d trials passed\n", NUM_ROUNDTRIP_TRIALS);
    return 0;
}

/* ===================== Property Test: Spectral Purity ===================== */

/**
 * @brief Test single-sinusoid spectral purity.
 *
 * For each trial:
 *   1. Create a complex sinusoid of integer frequency k and length N:
 *      x[n] = exp(2*pi*i*k*n/N)
 *   2. Apply forward FFT
 *   3. Verify the spectrum has a single non-zero magnitude bin (bin k)
 *      within 1e-9 * sqrt(N) tolerance for all other bins.
 *
 * Returns 0 on success, 1 on failure.
 */
static int test_spectral_purity(void)
{
    int failures = 0;
    int i;

    printf("  Testing spectral purity with %d random sinusoids...\n", NUM_SPECTRAL_TRIALS);

    for (i = 0; i < NUM_SPECTRAL_TRIALS; i++) {
        size_t N = generate_n(i);
        size_t k;
        lmmc_real_t *real_buf = NULL, *imag_buf = NULL;
        lmmc_status_t st;
        double tol;
        double peak_mag = 0.0;
        double max_off_peak = 0.0;
        size_t j;

        /* Skip N=1 since there's only one bin */
        if (N <= 1) {
            N = 2 + rng_int(0, 100);
        }

        /* Choose a random integer frequency k in [0, N-1] */
        k = rng_int(0, N - 1);

        /* Allocate arrays */
        real_buf = (lmmc_real_t*)malloc(N * sizeof(lmmc_real_t));
        imag_buf = (lmmc_real_t*)malloc(N * sizeof(lmmc_real_t));

        if (!real_buf || !imag_buf) {
            printf("    FAIL: allocation failed for N=%zu\n", N);
            free(real_buf); free(imag_buf);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        /* Generate complex sinusoid: x[n] = exp(2*pi*i*k*n/N) */
        for (j = 0; j < N; j++) {
            double angle = 2.0 * LMMC_PI * (double)k * (double)j / (double)N;
            real_buf[j] = cos(angle);
            imag_buf[j] = sin(angle);
        }

        /* Forward FFT */
        st = lmmc_fft_forward(real_buf, imag_buf, N);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: lmmc_fft_forward returned %d for N=%zu, k=%zu\n",
                   (int)st, N, k);
            free(real_buf); free(imag_buf);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        /* Tolerance for off-peak bins */
        tol = 1e-9 * sqrt((double)N);

        /* Check spectrum: bin k should have magnitude ~N, all others ~0 */
        for (j = 0; j < N; j++) {
            double mag = sqrt(real_buf[j] * real_buf[j] + imag_buf[j] * imag_buf[j]);
            if (j == k) {
                peak_mag = mag;
            } else {
                if (mag > max_off_peak) max_off_peak = mag;
            }
        }

        /* The peak bin should have magnitude approximately N */
        if (fabs(peak_mag - (double)N) > tol) {
            printf("    FAIL: N=%zu, k=%zu, peak_mag=%.6e, expected=%.1f, tol=%.3e\n",
                   N, k, peak_mag, (double)N, tol);
            failures++;
            if (failures >= 5) {
                printf("    ... stopping after 5 failures\n");
                free(real_buf); free(imag_buf);
                break;
            }
        }
        /* All off-peak bins should be within tolerance of zero */
        else if (max_off_peak > tol) {
            printf("    FAIL: N=%zu, k=%zu, max_off_peak=%.3e > tol=%.3e\n",
                   N, k, max_off_peak, tol);
            failures++;
            if (failures >= 5) {
                printf("    ... stopping after 5 failures\n");
                free(real_buf); free(imag_buf);
                break;
            }
        }

        free(real_buf); free(imag_buf);
    }

    if (failures > 0) {
        printf("    Spectral purity: %d/%d failures\n", failures, NUM_SPECTRAL_TRIALS);
        return 1;
    }

    printf("    Spectral purity: all %d trials passed\n", NUM_SPECTRAL_TRIALS);
    return 0;
}

/* ===================== Main ===================== */

typedef int (*test_func_t)(void);
typedef struct { const char *name; test_func_t func; } test_entry_t;

int main(void)
{
    test_entry_t tests[] = {
        {"FFT_roundtrip_property",      test_fft_roundtrip},
        {"FFT_spectral_purity_property", test_spectral_purity},
    };
    size_t n_tests = sizeof(tests) / sizeof(tests[0]);
    size_t i;
    size_t n_passed = 0;
    size_t n_failed = 0;

    printf("=== Property 1: Round-trip FFT ===\n");
    printf("Verify forward then inverse recovers input within 1e-9 * (1 + max|x|)\n");
    printf("Test single-sinusoid spectral purity\n\n");

    rng_seed(RNG_SEED);

    for (i = 0; i < n_tests; i++) {
        printf("[%zu/%zu] %s ...\n", i + 1, n_tests, tests[i].name);
        fflush(stdout);
        if (tests[i].func() == 0) {
            printf("  PASS\n\n");
            n_passed++;
        } else {
            printf("  FAIL\n\n");
            n_failed++;
        }
    }

    printf("=== Results: %zu passed, %zu failed ===\n", n_passed, n_failed);
    return (n_failed > 0) ? 1 : 0;
}
