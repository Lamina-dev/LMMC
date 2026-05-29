/**
 * @file test_random_dist_stats.c
 * 随机分布采样器统计测试（均值/方差 + 卡方拟合优度）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lmmc/lmmc.h"

#define NUM_SAMPLES 1000000
#define NUM_BINS    256
#define SIGMA_TOL   5.0

/* ===================== Chi-squared CDF via incomplete gamma ===================== */

static double regularized_gamma_p(double a, double x)
{
    double sum, term;
    int n;
    const int max_iter = 2000;
    const double eps = 1e-15;

    if (x < 0.0) return 0.0;
    if (x == 0.0) return 0.0;

    if (x > a + 200.0) {
        double f, c, d, delta;
        const double tiny = 1e-300;
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

        double log_q = -x + a * log(x) - lgamma(a) - log(f);
        if (log_q < -700.0) return 1.0;
        return 1.0 - exp(log_q);
    }

    sum = 1.0 / a;
    term = 1.0 / a;

    for (n = 1; n <= max_iter; n++) {
        term *= x / (a + (double)n);
        sum += term;
        if (fabs(term) < eps * fabs(sum)) break;
    }

    double log_p = -x + a * log(x) - lgamma(a) + log(sum);
    if (log_p < -700.0) return 0.0;
    if (log_p > 0.0) return 1.0;
    return exp(log_p);
}

static double chi2_cdf(double x, int k)
{
    if (x <= 0.0) return 0.0;
    return regularized_gamma_p((double)k / 2.0, x / 2.0);
}

/* ===================== Helper: check mean/variance ===================== */

/**
 * @brief Check that empirical mean and variance are within 5*sigma/sqrt(N)
 *        of the theoretical values.
 *
 * For the mean: the standard error is sigma/sqrt(N), so tolerance = 5*sigma/sqrt(N).
 * For the variance: the standard error of sample variance is sigma^2*sqrt(2/(N-1)),
 *   so tolerance = 5*sigma^2*sqrt(2/(N-1)).
 */
static int check_mean_variance(const char* name,
                               double emp_mean, double emp_var,
                               double theo_mean, double theo_var,
                               size_t n)
{
    double sigma = sqrt(theo_var);
    double sqrt_n = sqrt((double)n);
    double mean_tol = SIGMA_TOL * sigma / sqrt_n;
    double var_tol = SIGMA_TOL * theo_var * sqrt(2.0 / ((double)n - 1.0));
    int ok = 1;

    double mean_err = fabs(emp_mean - theo_mean);
    double var_err = fabs(emp_var - theo_var);

    if (mean_err > mean_tol) {
        printf("    FAIL [%s]: mean error %.6e > tolerance %.6e\n",
               name, mean_err, mean_tol);
        printf("      empirical mean = %.8f, theoretical = %.8f\n",
               emp_mean, theo_mean);
        ok = 0;
    }

    if (var_err > var_tol) {
        printf("    FAIL [%s]: variance error %.6e > tolerance %.6e\n",
               name, var_err, var_tol);
        printf("      empirical var = %.8f, theoretical = %.8f\n",
               emp_var, theo_var);
        ok = 0;
    }

    if (ok) {
        printf("    PASS [%s]: mean=%.6f (theo=%.6f, err=%.2e), var=%.6f (theo=%.6f, err=%.2e)\n",
               name, emp_mean, theo_mean, mean_err, emp_var, theo_var, var_err);
    }

    return ok ? 0 : 1;
}

/* ===================== Test: Normal (Ziggurat) ===================== */

static int test_normal_mean_variance(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;

    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) return 1;
    lmmc_rng_seed(rng, 42);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_normal(rng, 0.0, 1.0, &val);
        if (st != LMMC_STATUS_OK) { lmmc_rng_destroy(rng); return 1; }
        sum += val;
        sum_sq += val * val;
    }

    lmmc_rng_destroy(rng);

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);

    return check_mean_variance("Normal(0,1)", emp_mean, emp_var, 0.0, 1.0, NUM_SAMPLES);
}

/* ===================== Test: Gamma ===================== */

static int test_gamma_mean_variance(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double shape = 3.0, scale = 2.0;

    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) return 1;
    lmmc_rng_seed(rng, 123);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_gamma(rng, shape, scale, &val);
        if (st != LMMC_STATUS_OK) { lmmc_rng_destroy(rng); return 1; }
        sum += val;
        sum_sq += val * val;
    }

    lmmc_rng_destroy(rng);

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* Gamma(shape, scale): mean = shape*scale, var = shape*scale^2 */
    double theo_mean = shape * scale;
    double theo_var = shape * scale * scale;

    return check_mean_variance("Gamma(3,2)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

/* ===================== Test: Beta ===================== */

static int test_beta_mean_variance(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double alpha = 2.0, beta_p = 5.0;

    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) return 1;
    lmmc_rng_seed(rng, 456);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_beta(rng, alpha, beta_p, &val);
        if (st != LMMC_STATUS_OK) { lmmc_rng_destroy(rng); return 1; }
        sum += val;
        sum_sq += val * val;
    }

    lmmc_rng_destroy(rng);

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* Beta(a,b): mean = a/(a+b), var = ab/((a+b)^2*(a+b+1)) */
    double theo_mean = alpha / (alpha + beta_p);
    double theo_var = (alpha * beta_p) /
                      ((alpha + beta_p) * (alpha + beta_p) * (alpha + beta_p + 1.0));

    return check_mean_variance("Beta(2,5)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

/* ===================== Test: Chi-squared ===================== */

static int test_chi_squared_mean_variance(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double df = 5.0;

    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) return 1;
    lmmc_rng_seed(rng, 789);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_chi_squared(rng, df, &val);
        if (st != LMMC_STATUS_OK) { lmmc_rng_destroy(rng); return 1; }
        sum += val;
        sum_sq += val * val;
    }

    lmmc_rng_destroy(rng);

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* Chi2(df): mean = df, var = 2*df */
    double theo_mean = df;
    double theo_var = 2.0 * df;

    return check_mean_variance("ChiSq(5)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

/* ===================== Test: Student-t ===================== */

static int test_student_t_mean_variance(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double df = 10.0;  /* df > 2 so variance is finite */

    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) return 1;
    lmmc_rng_seed(rng, 1001);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_student_t(rng, df, &val);
        if (st != LMMC_STATUS_OK) { lmmc_rng_destroy(rng); return 1; }
        sum += val;
        sum_sq += val * val;
    }

    lmmc_rng_destroy(rng);

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* t(df): mean = 0 (for df > 1), var = df/(df-2) (for df > 2) */
    double theo_mean = 0.0;
    double theo_var = df / (df - 2.0);

    return check_mean_variance("StudentT(10)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

/* ===================== Test: F distribution ===================== */

static int test_f_mean_variance(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double df1 = 5.0, df2 = 20.0;  /* df2 > 4 so variance is finite */

    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) return 1;
    lmmc_rng_seed(rng, 2002);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_f(rng, df1, df2, &val);
        if (st != LMMC_STATUS_OK) { lmmc_rng_destroy(rng); return 1; }
        sum += val;
        sum_sq += val * val;
    }

    lmmc_rng_destroy(rng);

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* F(d1,d2): mean = d2/(d2-2) for d2>2, var = 2*d2^2*(d1+d2-2)/(d1*(d2-2)^2*(d2-4)) for d2>4 */
    double theo_mean = df2 / (df2 - 2.0);
    double theo_var = (2.0 * df2 * df2 * (df1 + df2 - 2.0)) /
                      (df1 * (df2 - 2.0) * (df2 - 2.0) * (df2 - 4.0));

    return check_mean_variance("F(5,20)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

/* ===================== Test: Poisson ===================== */

static int test_poisson_mean_variance(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double lambda = 15.0;  /* Use lambda >= 10 to exercise PTRD path */

    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) return 1;
    lmmc_rng_seed(rng, 3003);

    for (i = 0; i < NUM_SAMPLES; i++) {
        size_t val;
        st = lmmc_rng_poisson(rng, lambda, &val);
        if (st != LMMC_STATUS_OK) { lmmc_rng_destroy(rng); return 1; }
        sum += (double)val;
        sum_sq += (double)val * (double)val;
    }

    lmmc_rng_destroy(rng);

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* Poisson(lambda): mean = lambda, var = lambda */
    double theo_mean = lambda;
    double theo_var = lambda;

    return check_mean_variance("Poisson(15)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

/* ===================== Test: Binomial ===================== */

static int test_binomial_mean_variance(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    size_t n_trials = 50;
    double p = 0.3;

    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) return 1;
    lmmc_rng_seed(rng, 4004);

    for (i = 0; i < NUM_SAMPLES; i++) {
        size_t val;
        st = lmmc_rng_binomial(rng, n_trials, p, &val);
        if (st != LMMC_STATUS_OK) { lmmc_rng_destroy(rng); return 1; }
        sum += (double)val;
        sum_sq += (double)val * (double)val;
    }

    lmmc_rng_destroy(rng);

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* Binomial(n,p): mean = n*p, var = n*p*(1-p) */
    double theo_mean = (double)n_trials * p;
    double theo_var = (double)n_trials * p * (1.0 - p);

    return check_mean_variance("Binomial(50,0.3)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

/* ===================== Test: Ziggurat Normal Chi-squared GOF ===================== */

/**
 * @brief Chi-squared goodness-of-fit test for the Ziggurat normal sampler.
 *
 * Draws 1,000,000 samples from N(0,1), bins into 256 equal-probability bins
 * (using the normal quantile function to determine bin edges), computes the
 * chi-squared statistic, and verifies p-value > 1e-6.
 */
static int test_normal_chi_squared_gof(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    size_t bins[NUM_BINS];
    size_t i;
    double chi2_stat;
    double p_value;
    double expected;
    double bin_edges[NUM_BINS + 1];

    printf("  Chi-squared GOF for Ziggurat normal (256 bins, 1M samples)...\n");

    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) {
        printf("    FAIL: lmmc_rng_create returned %d\n", (int)st);
        return 1;
    }
    lmmc_rng_seed(rng, 0xDEADBEEF12345678ULL);

    /* Compute bin edges using equal-probability bins.
     * Each bin has probability 1/NUM_BINS under N(0,1).
     * Edge[i] = quantile(i / NUM_BINS) for i = 0..NUM_BINS.
     * Edge[0] = -inf, Edge[NUM_BINS] = +inf.
     */
    bin_edges[0] = -1e30;  /* effectively -infinity */
    bin_edges[NUM_BINS] = 1e30;  /* effectively +infinity */
    for (i = 1; i < NUM_BINS; i++) {
        lmmc_real_t q;
        double p_val = (double)i / (double)NUM_BINS;
        st = lmmc_dist_normal_quantile(p_val, 0.0, 1.0, &q);
        if (st != LMMC_STATUS_OK) {
            printf("    FAIL: lmmc_dist_normal_quantile returned %d for p=%.6f\n",
                   (int)st, p_val);
            lmmc_rng_destroy(rng);
            return 1;
        }
        bin_edges[i] = (double)q;
    }

    memset(bins, 0, sizeof(bins));

    /* Draw samples and bin them */
    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_normal(rng, 0.0, 1.0, &val);
        if (st != LMMC_STATUS_OK) { lmmc_rng_destroy(rng); return 1; }

        /* Binary search for the bin */
        size_t lo = 0, hi = NUM_BINS - 1;
        while (lo < hi) {
            size_t mid = (lo + hi) / 2;
            if ((double)val < bin_edges[mid + 1]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        bins[lo]++;
    }

    lmmc_rng_destroy(rng);

    /* Expected count per bin (equal-probability bins) */
    expected = (double)NUM_SAMPLES / (double)NUM_BINS;

    /* Compute chi-squared statistic */
    chi2_stat = 0.0;
    for (i = 0; i < NUM_BINS; i++) {
        double observed = (double)bins[i];
        double diff = observed - expected;
        chi2_stat += (diff * diff) / expected;
    }

    /* p-value: P(chi2 > chi2_stat | df = NUM_BINS - 1) = 1 - CDF */
    p_value = 1.0 - chi2_cdf(chi2_stat, NUM_BINS - 1);

    printf("    Chi-squared statistic: %.4f (df=%d)\n", chi2_stat, NUM_BINS - 1);
    printf("    p-value: %.6e\n", p_value);

    if (p_value <= 1e-6) {
        printf("    FAIL: p-value %.6e <= 1e-6\n", p_value);
        return 1;
    }

    printf("    PASS: p-value %.6e > 1e-6\n", p_value);
    return 0;
}

/* ===================== Main ===================== */

int main(void)
{
    int rc = 0;
    int failures = 0;

    printf("=== Random Distribution Statistical Tests ===\n");
    printf("- 1M samples per distribution, mean/variance within 5*sigma/sqrt(N)\n");
    printf("- Chi-squared GOF for Ziggurat normal (256 bins, alpha=1e-6)\n\n");

    printf("[1/9] Normal (Ziggurat) mean/variance...\n");
    if (test_normal_mean_variance() != 0) { failures++; rc = 1; }

    printf("[2/9] Gamma mean/variance...\n");
    if (test_gamma_mean_variance() != 0) { failures++; rc = 1; }

    printf("[3/9] Beta mean/variance...\n");
    if (test_beta_mean_variance() != 0) { failures++; rc = 1; }

    printf("[4/9] Chi-squared mean/variance...\n");
    if (test_chi_squared_mean_variance() != 0) { failures++; rc = 1; }

    printf("[5/9] Student-t mean/variance...\n");
    if (test_student_t_mean_variance() != 0) { failures++; rc = 1; }

    printf("[6/9] F distribution mean/variance...\n");
    if (test_f_mean_variance() != 0) { failures++; rc = 1; }

    printf("[7/9] Poisson mean/variance...\n");
    if (test_poisson_mean_variance() != 0) { failures++; rc = 1; }

    printf("[8/9] Binomial mean/variance...\n");
    if (test_binomial_mean_variance() != 0) { failures++; rc = 1; }

    printf("[9/9] Ziggurat Normal chi-squared GOF...\n");
    if (test_normal_chi_squared_gof() != 0) { failures++; rc = 1; }

    printf("\n=== Results: %d/9 passed, %d/9 failed ===\n", 9 - failures, failures);
    return rc;
}
