/**
 * @file test_rng_uniformity_prop.c
 * RNG 均匀性属性测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "lmmc/lmmc.h"

/**
 * @brief Compute the regularized lower incomplete gamma function P(a, x)
 *        using the series expansion.
 *
 * P(a, x) = gamma(a, x) / Gamma(a)
 *         = e^{-x} * x^a * sum_{n=0}^{inf} x^n / (a*(a+1)*...*(a+n))
 *
 * Used for computing the chi-squared CDF: chi2_cdf(x, k) = P(k/2, x/2).
 */
struct test_fixture {
    lmmc_rng_t *rng;
    lmmc_rng_t *rng1;
    lmmc_rng_t *rng2;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_rng_destroy(fixture->rng);
    lmmc_rng_destroy(fixture->rng1);
    lmmc_rng_destroy(fixture->rng2);
    free(fixture);
    *state = NULL;
    return 0;
}

static double gamma_p_complement(double a, double x, int max_iter, double eps) {
    int n;
    double f, c, d, delta;
    const double tiny = 1e-300;

    f = tiny;
    c = tiny;
    d = 0.0;

    /**
     * @brief Q(a,x) = e^{-x} * x^a / Gamma(a) / (x-a+1+K)，K 为连分式。
     * 分母形式为 b0 + a1/(b1 + a2/(b2 + ...))，
     * 其中 b_n = x - a + 2n + 1，a_n = n*(a-n)。
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
            if (fabs(d) < tiny) {
                d = tiny;
            }
            d = 1.0 / d;

            c = bn + an / c;
            if (fabs(c) < tiny) {
                c = tiny;
            }

            delta = c * d;
            f *= delta;

            if (fabs(delta - 1.0) < eps) {
                break;
            }
        }

        double log_q = -x + a * log(x) - lgamma(a) - log(f);
        if (log_q < -700.0) {
            return 1.0;
        } /**< Q 可忽略，P 取 1。 */
        return 1.0 - exp(log_q);
    }
}

static double regularized_gamma_p(double a, double x) {
    double sum, term;
    int n;
    const int max_iter = 2000;
    const double eps = 1e-15;

    if (x < 0.0) {
        return 0.0;
    }
    if (x == 0.0) {
        return 0.0;
    }

    /* For large x relative to a, use the complement: P = 1 - Q */
    if (x > a + 200.0) {
        return gamma_p_complement(a, x, max_iter, eps);
    }

    /* Series expansion: P(a,x) = e^{-x} * x^a / Gamma(a) * sum_{n=0}^inf x^n / (a+1)...(a+n) */
    sum = 1.0 / a;
    term = 1.0 / a;

    for (n = 1; n <= max_iter; n++) {
        term *= x / (a + (double)n);
        sum += term;
        if (fabs(term) < eps * fabs(sum)) {
            break;
        }
    }

    /* P(a,x) = e^{-x + a*ln(x) - lgamma(a)} * sum */
    double log_p = -x + a * log(x) - lgamma(a) + log(sum);
    if (log_p < -700.0) {
        return 0.0;
    }
    if (log_p > 0.0) {
        return 1.0;
    } /**< 将数值溢出的概率限制为 1。 */
    return exp(log_p);
}

/**
 * @brief Compute the chi-squared CDF: P(chi2 <= x | k degrees of freedom).
 *
 * chi2_cdf(x, k) = P(k/2, x/2) where P is the regularized lower incomplete gamma.
 */
static double chi2_cdf(double x, int k) {
    if (x <= 0.0) {
        return 0.0;
    }
    return regularized_gamma_p((double)k / 2.0, x / 2.0);
}

#define NUM_SAMPLES 1000000
#define NUM_BINS 256

/**
 * @brief Test RNG uniformity via chi-squared goodness-of-fit.
 *
 * Draws 1,000,000 64-bit samples from the RNG, bins them into 256 equal-width
 * bins over the unsigned 64-bit range [0, 2^64), computes the chi-squared
 * statistic against the uniform distribution, and verifies the p-value > 1e-6.
 */
static void test_rng_uniformity_chi_squared(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    size_t bins[NUM_BINS];
    double expected;
    double chi2_stat;
    double p_value;
    int i;

    st = lmmc_rng_create(&fixture->rng);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    FAIL: lmmc_rng_create returned %d\n", (int)st);
    }

    /* Seed with a fixed value for reproducibility of the property test */
    assert_int_equal(lmmc_rng_seed(fixture->rng, UINT64_C(0xABCDEF0123456789)), LMMC_STATUS_OK);

    memset(bins, 0, sizeof(bins));

    /* Draw samples and bin them.
     * Each bin covers a range of 2^64 / 256 = 2^56 values.
     * Bin index = sample >> 56 (top 8 bits).
     */
    for (i = 0; i < NUM_SAMPLES; i++) {
        uint64_t sample = lmmc_rng_next_u64(fixture->rng);
        size_t bin_idx = (size_t)(sample >> 56); /* top 8 bits -> 0..255 */
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

    if (!(p_value > 1e-6)) {
        fail_msg("RNG chi-squared goodness-of-fit failed: "
                 "statistic=%.4f, p-value=%.6e",
                 chi2_stat, p_value);
    }
}

static void test_rng_unique_sequences(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    uint64_t seq1[16], seq2[16];
    int found_diff = 0;
    int i;

    st = lmmc_rng_create(&fixture->rng1);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    FAIL: lmmc_rng_create (rng1) returned %d\n", (int)st);
    }

    st = lmmc_rng_create(&fixture->rng2);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    FAIL: lmmc_rng_create (rng2) returned %d\n", (int)st);
    }

    /* Draw 16 samples from each */
    for (i = 0; i < 16; i++) {
        seq1[i] = lmmc_rng_next_u64(fixture->rng1);
        seq2[i] = lmmc_rng_next_u64(fixture->rng2);
    }

    /* Check that at least one position differs */
    for (i = 0; i < 16; i++) {
        if (seq1[i] != seq2[i]) {
            found_diff = 1;
            break;
        }
    }

    if (!found_diff) {
        fail_msg("independently created RNGs produced the same first "
                 "16 raw samples");
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_rng_uniformity_chi_squared, setup, teardown),
        cmocka_unit_test_setup_teardown(test_rng_unique_sequences, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
