/**
 * @file test_dist_roundtrip_prop.c
 * @brief Property-based test for distribution CDF/quantile round-trip and erf accuracy.
 *
 * **Property 10: Distribution round-trip**
 * - Verify |quantile(cdf(x)) - x| <= 1e-8 for x in bulk of each continuous distribution
 * - Test erf accuracy against C standard library erf(x) for x in [-5, 5]
 *
 * **Validates: Requirements 16.4, 16.5**
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "lmmc/lmmc.h"

/* ===================== Simple xorshift64 RNG ===================== */

#define RNG_SEED 0xDEADBEEF4321ULL
#define NUM_DIST_TRIALS 200
#define NUM_ERF_TRIALS 500

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

/* ===================== Distribution Round-Trip Tests ===================== */

/**
 * @brief Test normal distribution round-trip: quantile(cdf(x)) == x
 *
 * Generate random x within 2.5 sigma of mu (bulk of normal).
 * Skip cases where p lands at 0 or 1 (extreme tails).
 * Returns 0 on success, 1 on failure.
 */
static int test_normal_roundtrip(void)
{
    int failures = 0;
    int i;

    printf("  Testing normal distribution round-trip (%d trials)...\n", NUM_DIST_TRIALS);

    for (i = 0; i < NUM_DIST_TRIALS; i++) {
        double mu = rng_uniform(-5.0, 5.0);
        double sigma = rng_uniform(0.5, 3.0);
        /* Stay within 2.5 sigma of mean to remain in bulk */
        double x = mu + rng_uniform(-2.5, 2.5) * sigma;
        lmmc_real_t p, x_recovered;
        lmmc_status_t st;

        st = lmmc_dist_normal_cdf(x, mu, sigma, &p);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: normal cdf returned %d for x=%.6e, mu=%.6e, sigma=%.6e\n",
                   (int)st, x, mu, sigma);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        /* Skip extreme probabilities where quantile is undefined or unstable */
        if (p <= 1e-10 || p >= 1.0 - 1e-10) continue;

        st = lmmc_dist_normal_quantile(p, mu, sigma, &x_recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: normal quantile returned %d for p=%.15e\n", (int)st, p);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        double err = fabs(x_recovered - x);
        if (err > 1e-8) {
            printf("    FAIL: normal round-trip |%.15e - %.15e| = %.3e > 1e-8 "
                   "(mu=%.3e, sigma=%.3e, p=%.15e)\n",
                   x_recovered, x, err, mu, sigma, p);
            failures++;
            if (failures >= 5) break;
        }
    }

    if (failures > 0) {
        printf("    Normal round-trip: %d/%d failures\n", failures, NUM_DIST_TRIALS);
        return 1;
    }
    printf("    Normal round-trip: all %d trials passed\n", NUM_DIST_TRIALS);
    return 0;
}

/**
 * @brief Test t-distribution round-trip: quantile(cdf(x)) == x
 *
 * Generate random x in [-3, 3] (bulk), df in [2, 50].
 */
static int test_t_roundtrip(void)
{
    int failures = 0;
    int i;

    printf("  Testing t-distribution round-trip (%d trials)...\n", NUM_DIST_TRIALS);

    for (i = 0; i < NUM_DIST_TRIALS; i++) {
        double x = rng_uniform(-3.0, 3.0);
        double df = rng_uniform(2.0, 50.0);
        lmmc_real_t p, x_recovered;
        lmmc_status_t st;

        st = lmmc_dist_t_cdf(x, df, &p);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: t cdf returned %d for x=%.6e, df=%.6e\n", (int)st, x, df);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        st = lmmc_dist_t_quantile(p, df, &x_recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: t quantile returned %d for p=%.15e, df=%.6e\n", (int)st, p, df);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        double err = fabs(x_recovered - x);
        if (err > 1e-8) {
            printf("    FAIL: t round-trip |%.15e - %.15e| = %.3e > 1e-8 (df=%.3e, p=%.15e)\n",
                   x_recovered, x, err, df, p);
            failures++;
            if (failures >= 5) break;
        }
    }

    if (failures > 0) {
        printf("    t round-trip: %d/%d failures\n", failures, NUM_DIST_TRIALS);
        return 1;
    }
    printf("    t round-trip: all %d trials passed\n", NUM_DIST_TRIALS);
    return 0;
}

/**
 * @brief Test chi-squared distribution round-trip: quantile(cdf(x)) == x
 *
 * Generate random x in [0.5, 20] (bulk), df in [2, 30].
 */
static int test_chi2_roundtrip(void)
{
    int failures = 0;
    int i;

    printf("  Testing chi2 distribution round-trip (%d trials)...\n", NUM_DIST_TRIALS);

    for (i = 0; i < NUM_DIST_TRIALS; i++) {
        double df = rng_uniform(2.0, 30.0);
        /* Generate x in bulk: around the mean (df) +/- 2*sqrt(2*df) */
        double spread = 2.0 * sqrt(2.0 * df);
        double x = rng_uniform(fmax(0.5, df - spread), df + spread);
        lmmc_real_t p, x_recovered;
        lmmc_status_t st;

        st = lmmc_dist_chi2_cdf(x, df, &p);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: chi2 cdf returned %d for x=%.6e, df=%.6e\n", (int)st, x, df);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        /* Skip extreme probabilities where quantile inversion is less stable */
        if (p < 0.001 || p > 0.999) continue;

        st = lmmc_dist_chi2_quantile(p, df, &x_recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: chi2 quantile returned %d for p=%.15e, df=%.6e\n", (int)st, p, df);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        double err = fabs(x_recovered - x);
        if (err > 1e-8) {
            printf("    FAIL: chi2 round-trip |%.15e - %.15e| = %.3e > 1e-8 (df=%.3e, p=%.15e)\n",
                   x_recovered, x, err, df, p);
            failures++;
            if (failures >= 5) break;
        }
    }

    if (failures > 0) {
        printf("    chi2 round-trip: %d/%d failures\n", failures, NUM_DIST_TRIALS);
        return 1;
    }
    printf("    chi2 round-trip: all %d trials passed\n", NUM_DIST_TRIALS);
    return 0;
}

/**
 * @brief Test F-distribution round-trip: quantile(cdf(x)) == x
 *
 * Generate random x in bulk of F distribution, df1 in [3, 20], df2 in [5, 30].
 * The F distribution bulk is around its mean df2/(df2-2).
 */
static int test_f_roundtrip(void)
{
    int failures = 0;
    int i;

    printf("  Testing F-distribution round-trip (%d trials)...\n", NUM_DIST_TRIALS);

    for (i = 0; i < NUM_DIST_TRIALS; i++) {
        double df1 = rng_uniform(3.0, 20.0);
        double df2 = rng_uniform(5.0, 30.0);
        /* F distribution mean is df2/(df2-2) for df2>2; stay in central region */
        double mean_f = df2 / (df2 - 2.0);
        double x = rng_uniform(mean_f * 0.5, mean_f * 2.0);
        lmmc_real_t p, x_recovered;
        lmmc_status_t st;

        st = lmmc_dist_f_cdf(x, df1, df2, &p);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: F cdf returned %d for x=%.6e, df1=%.6e, df2=%.6e\n",
                   (int)st, x, df1, df2);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        /* Skip probabilities outside the central bulk [0.1, 0.9] */
        if (p < 0.1 || p > 0.9) continue;

        st = lmmc_dist_f_quantile(p, df1, df2, &x_recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: F quantile returned %d for p=%.15e\n", (int)st, p);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        double err = fabs(x_recovered - x);
        if (err > 1e-8) {
            printf("    FAIL: F round-trip |%.15e - %.15e| = %.3e > 1e-8 "
                   "(df1=%.3e, df2=%.3e, p=%.15e)\n",
                   x_recovered, x, err, df1, df2, p);
            failures++;
            if (failures >= 5) break;
        }
    }

    if (failures > 0) {
        printf("    F round-trip: %d/%d failures\n", failures, NUM_DIST_TRIALS);
        return 1;
    }
    printf("    F round-trip: all %d trials passed\n", NUM_DIST_TRIALS);
    return 0;
}

/**
 * @brief Test gamma distribution round-trip: quantile(cdf(x)) == x
 *
 * Generate random x in bulk, shape in [2, 10], scale in [0.5, 3].
 * Stay in the central region of the distribution.
 */
static int test_gamma_roundtrip(void)
{
    int failures = 0;
    int i;

    printf("  Testing gamma distribution round-trip (%d trials)...\n", NUM_DIST_TRIALS);

    for (i = 0; i < NUM_DIST_TRIALS; i++) {
        double shape = rng_uniform(2.0, 10.0);
        double scale = rng_uniform(0.5, 3.0);
        /* Mean = shape*scale, stddev = sqrt(shape)*scale */
        double mean_g = shape * scale;
        double stddev_g = sqrt(shape) * scale;
        /* Stay within 1.5 stddev of mean to remain in bulk */
        double x = rng_uniform(fmax(0.5, mean_g - 1.5 * stddev_g), mean_g + 1.5 * stddev_g);
        lmmc_real_t p, x_recovered;
        lmmc_status_t st;

        st = lmmc_dist_gamma_cdf(x, shape, scale, &p);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: gamma cdf returned %d for x=%.6e, shape=%.6e, scale=%.6e\n",
                   (int)st, x, shape, scale);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        /* Skip extreme probabilities where quantile inversion is less stable */
        if (p < 0.01 || p > 0.99) continue;

        st = lmmc_dist_gamma_quantile(p, shape, scale, &x_recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: gamma quantile returned %d for p=%.15e\n", (int)st, p);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        double err = fabs(x_recovered - x);
        if (err > 1e-8) {
            printf("    FAIL: gamma round-trip |%.15e - %.15e| = %.3e > 1e-8 "
                   "(shape=%.3e, scale=%.3e, p=%.15e)\n",
                   x_recovered, x, err, shape, scale, p);
            failures++;
            if (failures >= 5) break;
        }
    }

    if (failures > 0) {
        printf("    Gamma round-trip: %d/%d failures\n", failures, NUM_DIST_TRIALS);
        return 1;
    }
    printf("    Gamma round-trip: all %d trials passed\n", NUM_DIST_TRIALS);
    return 0;
}

/**
 * @brief Test beta distribution round-trip: quantile(cdf(x)) == x
 *
 * Generate random x in [0.05, 0.95] (bulk of beta), alpha in [1, 10], beta in [1, 10].
 */
static int test_beta_roundtrip(void)
{
    int failures = 0;
    int i;

    printf("  Testing beta distribution round-trip (%d trials)...\n", NUM_DIST_TRIALS);

    for (i = 0; i < NUM_DIST_TRIALS; i++) {
        double alpha = rng_uniform(1.0, 10.0);
        double beta_param = rng_uniform(1.0, 10.0);
        /* Beta is on [0,1]; generate in bulk avoiding extremes */
        double x = rng_uniform(0.05, 0.95);
        lmmc_real_t p, x_recovered;
        lmmc_status_t st;

        st = lmmc_dist_beta_cdf(x, alpha, beta_param, &p);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: beta cdf returned %d for x=%.6e, alpha=%.6e, beta=%.6e\n",
                   (int)st, x, alpha, beta_param);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        /* Skip extreme probabilities */
        if (p < 0.001 || p > 0.999) continue;

        st = lmmc_dist_beta_quantile(p, alpha, beta_param, &x_recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: beta quantile returned %d for p=%.15e\n", (int)st, p);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        double err = fabs(x_recovered - x);
        if (err > 1e-8) {
            printf("    FAIL: beta round-trip |%.15e - %.15e| = %.3e > 1e-8 "
                   "(alpha=%.3e, beta=%.3e, p=%.15e)\n",
                   x_recovered, x, err, alpha, beta_param, p);
            failures++;
            if (failures >= 5) break;
        }
    }

    if (failures > 0) {
        printf("    Beta round-trip: %d/%d failures\n", failures, NUM_DIST_TRIALS);
        return 1;
    }
    printf("    Beta round-trip: all %d trials passed\n", NUM_DIST_TRIALS);
    return 0;
}

/* ===================== Erf Accuracy Test ===================== */

/**
 * @brief Test erf accuracy against C standard library erf(x) for x in [-5, 5].
 *
 * Verify |lmmc_erf(x) - erf(x)| <= 1e-12 for random x values.
 * Returns 0 on success, 1 on failure.
 */
static int test_erf_accuracy(void)
{
    int failures = 0;
    int i;

    printf("  Testing erf accuracy against C stdlib (%d trials)...\n", NUM_ERF_TRIALS);

    for (i = 0; i < NUM_ERF_TRIALS; i++) {
        double x = rng_uniform(-5.0, 5.0);
        lmmc_real_t lmmc_result;
        lmmc_status_t st;

        st = lmmc_erf(x, &lmmc_result);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: lmmc_erf returned %d for x=%.6e\n", (int)st, x);
            failures++;
            if (failures >= 5) break;
            continue;
        }

        double reference = erf(x);
        double err = fabs(lmmc_result - reference);

        if (err > 1e-12) {
            printf("    FAIL: erf(%.15e): lmmc=%.15e, ref=%.15e, |err|=%.3e > 1e-12\n",
                   x, lmmc_result, reference, err);
            failures++;
            if (failures >= 5) break;
        }
    }

    if (failures > 0) {
        printf("    Erf accuracy: %d/%d failures\n", failures, NUM_ERF_TRIALS);
        return 1;
    }
    printf("    Erf accuracy: all %d trials passed\n", NUM_ERF_TRIALS);
    return 0;
}

/* ===================== Main ===================== */

typedef int (*test_func_t)(void);
typedef struct { const char *name; test_func_t func; } test_entry_t;

int main(void)
{
    test_entry_t tests[] = {
        {"normal_roundtrip",  test_normal_roundtrip},
        {"t_roundtrip",       test_t_roundtrip},
        {"chi2_roundtrip",    test_chi2_roundtrip},
        {"f_roundtrip",       test_f_roundtrip},
        {"gamma_roundtrip",   test_gamma_roundtrip},
        {"beta_roundtrip",    test_beta_roundtrip},
        {"erf_accuracy",      test_erf_accuracy},
    };
    size_t n_tests = sizeof(tests) / sizeof(tests[0]);
    size_t i;
    size_t n_passed = 0;
    size_t n_failed = 0;

    printf("=== Property 10: Distribution round-trip ===\n");
    printf("Validates: Requirements 16.4, 16.5\n");
    printf("Verify |quantile(cdf(x)) - x| <= 1e-8 for continuous distributions\n");
    printf("Test erf accuracy against high-precision reference for x in [-5, 5]\n\n");

    lmmc_init();
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

    lmmc_deinit();
    return (n_failed > 0) ? 1 : 0;
}
