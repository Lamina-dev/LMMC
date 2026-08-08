/**
 * @file test_rng_uniformity_prop.c
 * RNG 均匀性属性测试。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "lmmc/lmmc.h"

/* ===================== Chi-squared CDF via incomplete gamma ===================== */

/**
 * @brief Compute the regularized lower incomplete gamma function P(a, x)
 *        using the series expansion.
 *
 * P(a, x) = gamma(a, x) / Gamma(a)
 *         = e^{-x} * x^a * sum_{n=0}^{inf} x^n / (a*(a+1)*...*(a+n))
 *
 * Used for computing the chi-squared CDF: chi2_cdf(x, k) = P(k/2, x/2).
 */
static double regularized_gamma_p(double a, double x)
{
    double sum, term;
    int n;
    const int max_iter = 2000;
    const double eps = 1e-15;

    if (x < 0.0) return 0.0;
    if (x == 0.0) return 0.0;

    /* For large x relative to a, use the complement: P = 1 - Q */
    if (x > a + 200.0) {
        /* Use continued fraction for Q(a,x) = 1 - P(a,x) */
        /* Lentz's method for the continued fraction */
        double f, c, d, delta;
        const double tiny = 1e-300;

        f = tiny;
        c = tiny;
        d = 0.0;

        /* CF: Q(a,x) = e^{-x} * x^a / Gamma(a) * 1/(x-a+1+ K)
         * where K is the continued fraction.
         * Using the form: b0 + a1/(b1 + a2/(b2 + ...))
         * with b_n = x - a + 2n + 1, a_n = n*(a-n)
         */
        {
            double b0 = x - a + 1.0;
            f = (fabs(b0) < tiny) ? tiny : b0;
            c = f;
            d = 0.0;

            for (n = 1; n <= max_iter; n++) {
                double an = (double)n * (a - (double)n);
                double bn = x - a + 2.0 * n + 1.0;

                d = bn + an * d;
                if (fabs(d) < tiny) d = tiny;
                d = 1.0 / d;

                c = bn + an / c;
                if (fabs(c) < tiny) c = tiny;

                delta = c * d;
                f *= delta;

                if (fabs(delta - 1.0) < eps) break;
            }

            /* Q(a,x) = e^{-x + a*ln(x) - lgamma(a)} / f */
            double log_q = -x + a * log(x) - lgamma(a) - log(f);
            if (log_q < -700.0) return 1.0; /* Q is negligible */
            return 1.0 - exp(log_q);
        }
    }

    /* Series expansion: P(a,x) = e^{-x} * x^a / Gamma(a) * sum_{n=0}^inf x^n / (a+1)...(a+n) */
    sum = 1.0 / a;
    term = 1.0 / a;

    for (n = 1; n <= max_iter; n++) {
        term *= x / (a + (double)n);
        sum += term;
        if (fabs(term) < eps * fabs(sum)) break;
    }

    /* P(a,x) = e^{-x + a*ln(x) - lgamma(a)} * sum */
    double log_p = -x + a * log(x) - lgamma(a) + log(sum);
    if (log_p < -700.0) return 0.0;
    if (log_p > 0.0) return 1.0; /* numerical overflow guard */
    return exp(log_p);
}

/**
 * @brief Compute the chi-squared CDF: P(chi2 <= x | k degrees of freedom).
 *
 * chi2_cdf(x, k) = P(k/2, x/2) where P is the regularized lower incomplete gamma.
 */
static double chi2_cdf(double x, int k)
{
    if (x <= 0.0) return 0.0;
    return regularized_gamma_p((double)k / 2.0, x / 2.0);
}

/* ===================== Property Test: Uniformity ===================== */

#define NUM_SAMPLES 1000000
#define NUM_BINS    256

/**
 * @brief Test RNG uniformity via chi-squared goodness-of-fit.
 *
 * Draws 1,000,000 64-bit samples from the RNG, bins them into 256 equal-width
 * bins over the unsigned 64-bit range [0, 2^64), computes the chi-squared
 * statistic against the uniform distribution, and verifies the p-value > 1e-6.
 *
 * Returns 0 on success, 1 on failure.
 */
static int test_rng_uniformity_chi_squared(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    size_t bins[NUM_BINS];
    double expected;
    double chi2_stat;
    double p_value;
    int i;

    printf("  Drawing %d samples, binning into %d bins...\n", NUM_SAMPLES, NUM_BINS);

    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) {
        printf("    FAIL: lmmc_rng_create returned %d\n", (int)st);
        return 1;
    }

    /* Seed with a fixed value for reproducibility of the property test */
    lmmc_rng_seed(rng, 0xABCDEF0123456789ULL);

    memset(bins, 0, sizeof(bins));

    /* Draw samples and bin them.
     * Each bin covers a range of 2^64 / 256 = 2^56 values.
     * Bin index = sample >> 56 (top 8 bits).
     */
    for (i = 0; i < NUM_SAMPLES; i++) {
        uint64_t sample = lmmc_rng_next_u64(rng);
        size_t bin_idx = (size_t)(sample >> 56);  /* top 8 bits -> 0..255 */
        bins[bin_idx]++;
    }

    /* Expected count per bin under uniform distribution */
    expected = (double)NUM_SAMPLES / (double)NUM_BINS;

    /* Compute chi-squared statistic */
    chi2_stat = 0.0;
    for (i = 0; i < NUM_BINS; i++) {
        double observed = (double)bins[i];
        double diff = observed - expected;
        chi2_stat += (diff * diff) / expected;
    }

    /* Compute p-value: P(chi2 > chi2_stat | df=255) = 1 - CDF(chi2_stat, 255) */
    p_value = 1.0 - chi2_cdf(chi2_stat, NUM_BINS - 1);

    printf("    Chi-squared statistic: %.4f (df=%d)\n", chi2_stat, NUM_BINS - 1);
    printf("    p-value: %.6e\n", p_value);
    printf("    Expected range for chi2 with df=255: ~200-310 (95%% interval)\n");

    if (p_value <= 1e-6) {
        printf("    FAIL: p-value %.6e <= 1e-6, RNG output is not uniform\n", p_value);
        printf("    Bin counts (first 16): ");
        for (i = 0; i < 16; i++) {
            printf("%zu ", bins[i]);
        }
        printf("...\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    printf("    PASS: p-value %.6e > 1e-6\n", p_value);
    lmmc_rng_destroy(rng);
    return 0;
}

/* ===================== Property Test: Unique Seeds ===================== */

static int test_rng_unique_sequences(void)
{
    lmmc_rng_t* rng1 = NULL;
    lmmc_rng_t* rng2 = NULL;
    lmmc_status_t st;
    uint64_t seq1[16], seq2[16];
    int found_diff = 0;
    int i;

    printf("  Creating two RNGs without explicit seed...\n");

    st = lmmc_rng_create(&rng1);
    if (st != LMMC_STATUS_OK) {
        printf("    FAIL: lmmc_rng_create (rng1) returned %d\n", (int)st);
        return 1;
    }

    st = lmmc_rng_create(&rng2);
    if (st != LMMC_STATUS_OK) {
        printf("    FAIL: lmmc_rng_create (rng2) returned %d\n", (int)st);
        lmmc_rng_destroy(rng1);
        return 1;
    }

    /* Draw 16 samples from each */
    for (i = 0; i < 16; i++) {
        seq1[i] = lmmc_rng_next_u64(rng1);
        seq2[i] = lmmc_rng_next_u64(rng2);
    }

    /* Check that at least one position differs */
    for (i = 0; i < 16; i++) {
        if (seq1[i] != seq2[i]) {
            found_diff = 1;
            printf("    Sequences differ at position %d\n", i);
            printf("      rng1[%d] = 0x%016llx\n", i, (unsigned long long)seq1[i]);
            printf("      rng2[%d] = 0x%016llx\n", i, (unsigned long long)seq2[i]);
            break;
        }
    }

    if (!found_diff) {
        printf("    FAIL: Two RNGs created without explicit seed produced\n");
        printf("          identical first 16 samples!\n");
        printf("    Sequence: ");
        for (i = 0; i < 16; i++) {
            printf("0x%016llx ", (unsigned long long)seq1[i]);
        }
        printf("\n");
        lmmc_rng_destroy(rng1);
        lmmc_rng_destroy(rng2);
        return 1;
    }

    printf("    PASS: Two RNGs produce different sequences\n");
    lmmc_rng_destroy(rng1);
    lmmc_rng_destroy(rng2);
    return 0;
}

/* ===================== Main ===================== */

typedef int (*test_func_t)(void);
typedef struct { const char *name; test_func_t func; } test_entry_t;

int main(void)
{
    test_entry_t tests[] = {
        {"RNG_uniformity_chi_squared", test_rng_uniformity_chi_squared},
        {"RNG_unique_sequences",       test_rng_unique_sequences},
    };
    size_t n_tests = sizeof(tests) / sizeof(tests[0]);
    size_t i;
    size_t n_passed = 0;
    size_t n_failed = 0;

    printf("=== Property 8: RNG uniformity ===\n");
    printf("- Chi-squared test on 1M samples binned into 256 bins\n");
    printf("- Two RNGs created in same second produce different sequences\n\n");

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
