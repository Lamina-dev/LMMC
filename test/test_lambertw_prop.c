/**
 * @file test_lambertw_prop.c
 * Lambert W 恒等式属性测试。
 */
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include "lmmc/lmmc.h"

/* ===================== Simple xorshift64 RNG ===================== */

#define RNG_SEED 0xDEADBEEF42ULL
#define NUM_TRIALS 2000  /* At least 1000 per branch */

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

/* ===================== Property Test: W_0 branch ===================== */

/**
 * @brief Test W_0 identity: W(z)*exp(W(z)) == z for random z in [-1/e, large].
 *
 * Generates random z values across the W_0 domain:
 * - Near the branch point: z in [-1/e, -1/e + 0.01]
 * - Small positive: z in [0, 1]
 * - Medium: z in [1, 100]
 * - Large: z in [100, 1e6]
 *
 * Returns 0 on success, 1 on failure.
 */
static int test_w0_identity(void)
{
    int failures = 0;
    int i;
    double neg_inv_e = -LMMC_INV_E;

    printf("  Testing W_0 branch with %d random inputs...\n", NUM_TRIALS);

    for (i = 0; i < NUM_TRIALS; i++) {
        double z;
        double w = 0.0;
        double residual, threshold;
        lmmc_status_t st;

        /* Generate z in different sub-ranges of the W_0 domain */
        switch (i % 4) {
            case 0:
                /* Near branch point: [-1/e, -1/e + 0.05] */
                z = rng_uniform(neg_inv_e, neg_inv_e + 0.05);
                break;
            case 1:
                /* Small positive: [0, 1] */
                z = rng_uniform(0.0, 1.0);
                break;
            case 2:
                /* Medium: [1, 100] */
                z = rng_uniform(1.0, 100.0);
                break;
            case 3:
                /* Large: [100, 1e6] */
                z = rng_uniform(100.0, 1e6);
                break;
        }

        st = lmmc_lambertw(z, &w);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: W0(%.17g) returned status %d\n", z, (int)st);
            failures++;
            if (failures >= 5) {
                printf("    ... stopping after 5 failures\n");
                return 1;
            }
            continue;
        }

        /* Verify identity: |W(z)*exp(W(z)) - z| <= 1e-10 * (1 + |z|) */
        residual = fabs(w * exp(w) - z);
        threshold = 1e-10 * (1.0 + fabs(z));

        if (residual > threshold) {
            printf("    FAIL: W0(%.17g) = %.17g\n", z, w);
            printf("           w*exp(w) = %.17g\n", w * exp(w));
            printf("           residual = %.3e > threshold = %.3e\n",
                   residual, threshold);
            failures++;
            if (failures >= 5) {
                printf("    ... stopping after 5 failures\n");
                return 1;
            }
        }
    }

    if (failures > 0) {
        printf("    W_0 branch: %d/%d failures\n", failures, NUM_TRIALS);
        return 1;
    }

    printf("    W_0 branch: all %d trials passed\n", NUM_TRIALS);
    return 0;
}

/* ===================== Property Test: W_{-1} branch ===================== */

/**
 * @brief Test W_{-1} identity: W(z)*exp(W(z)) == z for random z in [-1/e, 0).
 *
 * Generates random z values across the W_{-1} domain:
 * - Near the branch point: z in [-1/e, -1/e + 0.01]
 * - Middle of domain: z in [-0.3, -0.05]
 * - Near zero: z in [-0.05, -1e-10]
 * - Very near zero: z in [-1e-10, -1e-300]
 *
 * Returns 0 on success, 1 on failure.
 */
static int test_wm1_identity(void)
{
    int failures = 0;
    int i;
    double neg_inv_e = -LMMC_INV_E;

    printf("  Testing W_{-1} branch with %d random inputs...\n", NUM_TRIALS);

    for (i = 0; i < NUM_TRIALS; i++) {
        double z;
        double w = 0.0;
        double residual, threshold;
        lmmc_status_t st;

        /* Generate z in different sub-ranges of the W_{-1} domain [-1/e, 0) */
        switch (i % 4) {
            case 0:
                /* Near branch point: [-1/e, -1/e + 0.01] */
                z = rng_uniform(neg_inv_e, neg_inv_e + 0.01);
                break;
            case 1:
                /* Middle of domain: [-0.3, -0.05] */
                z = rng_uniform(-0.3, -0.05);
                break;
            case 2:
                /* Near zero: [-0.05, -1e-6] */
                z = rng_uniform(-0.05, -1e-6);
                break;
            case 3:
                /* Very near zero (log-uniform): [-1e-6, -1e-100] */
                {
                    /* Generate log-uniform in exponent range to cover small values */
                    double exp_val = rng_uniform(-100.0, -6.0);
                    z = -pow(10.0, exp_val);
                }
                break;
        }

        st = lmmc_lambertw_wm1(z, &w);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: Wm1(%.17g) returned status %d\n", z, (int)st);
            failures++;
            if (failures >= 5) {
                printf("    ... stopping after 5 failures\n");
                return 1;
            }
            continue;
        }

        /* Verify W_{-1} is on the correct branch: w <= -1 */
        if (w > -1.0 + 1e-10) {
            printf("    FAIL: Wm1(%.17g) = %.17g (should be <= -1)\n", z, w);
            failures++;
            if (failures >= 5) {
                printf("    ... stopping after 5 failures\n");
                return 1;
            }
            continue;
        }

        /* Verify identity: |W(z)*exp(W(z)) - z| <= 1e-10 * (1 + |z|) */
        residual = fabs(w * exp(w) - z);
        threshold = 1e-10 * (1.0 + fabs(z));

        if (residual > threshold) {
            printf("    FAIL: Wm1(%.17g) = %.17g\n", z, w);
            printf("           w*exp(w) = %.17g\n", w * exp(w));
            printf("           residual = %.3e > threshold = %.3e\n",
                   residual, threshold);
            failures++;
            if (failures >= 5) {
                printf("    ... stopping after 5 failures\n");
                return 1;
            }
        }
    }

    if (failures > 0) {
        printf("    W_{-1} branch: %d/%d failures\n", failures, NUM_TRIALS);
        return 1;
    }

    printf("    W_{-1} branch: all %d trials passed\n", NUM_TRIALS);
    return 0;
}

/* ===================== Main ===================== */

typedef int (*test_func_t)(void);
typedef struct { const char *name; test_func_t func; } test_entry_t;

int main(void)
{
    test_entry_t tests[] = {
        {"W0_identity_property",   test_w0_identity},
        {"Wm1_identity_property",  test_wm1_identity},
    };
    size_t n_tests = sizeof(tests) / sizeof(tests[0]);
    size_t i;
    size_t n_passed = 0;
    size_t n_failed = 0;

    printf("=== Lambert W identity ===\n");
    printf("Verify W(z) * exp(W(z)) == z within 1e-10 * (1 + |z|)\n\n");

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
