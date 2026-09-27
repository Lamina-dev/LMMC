/**
 * @file test_random_dist_stats.c
 * 随机分布采样器统计测试（均值/方差 + 卡方拟合优度）。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "lmmc/lmmc.h"

#define NUM_SAMPLES 1000000
#define NUM_BINS 256
#define SIGMA_TOL 5.0

struct test_fixture {
    lmmc_rng_t *rng;
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
    free(fixture);
    *state = NULL;
    return 0;
}

static double gamma_p_complement(double a, double x, int max_iter, double eps) {
    int n;
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
    }
    return 1.0 - exp(log_q);
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

    if (x > a + 200.0) {
        return gamma_p_complement(a, x, max_iter, eps);
    }

    sum = 1.0 / a;
    term = 1.0 / a;

    for (n = 1; n <= max_iter; n++) {
        term *= x / (a + (double)n);
        sum += term;
        if (fabs(term) < eps * fabs(sum)) {
            break;
        }
    }

    double log_p = -x + a * log(x) - lgamma(a) + log(sum);
    if (log_p < -700.0) {
        return 0.0;
    }
    if (log_p > 0.0) {
        return 1.0;
    }
    return exp(log_p);
}

static double chi2_cdf(double x, int k) {
    if (x <= 0.0) {
        return 0.0;
    }
    return regularized_gamma_p((double)k / 2.0, x / 2.0);
}

/**
 * @brief Check that empirical mean and variance are within 5*sigma/sqrt(N)
 *        of the theoretical values.
 *
 * For the mean: the standard error is sigma/sqrt(N), so tolerance = 5*sigma/sqrt(N).
 * For the variance: the standard error of sample variance is sigma^2*sqrt(2/(N-1)),
 *   so tolerance = 5*sigma^2*sqrt(2/(N-1)).
 */
static void check_mean_variance(const char *name,
                                double emp_mean, double emp_var,
                                double theo_mean, double theo_var,
                                size_t n) {
    const double mean_tol = SIGMA_TOL * sqrt(theo_var / (double)n);
    const double var_tol = SIGMA_TOL * theo_var * sqrt(2.0 / ((double)n - 1.0));
    if (!isfinite(emp_mean) || fabs(emp_mean - theo_mean) > mean_tol) {
        fail_msg("%s mean: empirical %.17g, theoretical %.17g, tolerance %.17g",
                 name, emp_mean, theo_mean, mean_tol);
    }
    if (!isfinite(emp_var) || fabs(emp_var - theo_var) > var_tol) {
        fail_msg("%s variance: empirical %.17g, theoretical %.17g, tolerance %.17g",
                 name, emp_var, theo_var, var_tol);
    }
}

static void test_normal_mean_variance(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 42), LMMC_STATUS_OK);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_normal(fixture->rng, 0.0, 1.0, &val);
        assert_false(st != LMMC_STATUS_OK);
        sum += val;
        sum_sq += val * val;
    }

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);

    check_mean_variance("Normal(0,1)", emp_mean, emp_var, 0.0, 1.0, NUM_SAMPLES);
}

static void test_gamma_mean_variance(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double shape = 3.0, scale = 2.0;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 123), LMMC_STATUS_OK);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_gamma(fixture->rng, shape, scale, &val);
        assert_false(st != LMMC_STATUS_OK);
        sum += val;
        sum_sq += val * val;
    }

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* Gamma(shape, scale): mean = shape*scale, var = shape*scale^2 */
    double theo_mean = shape * scale;
    double theo_var = shape * scale * scale;

    check_mean_variance("Gamma(3,2)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

static void test_beta_mean_variance(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double alpha = 2.0, beta_p = 5.0;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 456), LMMC_STATUS_OK);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_beta(fixture->rng, alpha, beta_p, &val);
        assert_int_equal(st, LMMC_STATUS_OK);
        assert_true(isfinite(val));
        assert_true(val >= 0.0 && val <= 1.0);
        sum += val;
        sum_sq += val * val;
    }

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* Beta(a,b): mean = a/(a+b), var = ab/((a+b)^2*(a+b+1)) */
    double theo_mean = alpha / (alpha + beta_p);
    double theo_var = (alpha * beta_p) /
                      ((alpha + beta_p) * (alpha + beta_p) * (alpha + beta_p + 1.0));

    check_mean_variance("Beta(2,5)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

static void test_beta_small_shape_moments(void **state) {
    struct test_fixture *fixture = *state;
    const double shapes[][2] = {{0.001, 0.001}, {0.001, 0.003}, {0.003, 0.001}, {0.5, 0.5}};
    const size_t count = 100000;

    size_t shape, i;
    assert_false(lmmc_rng_create(&fixture->rng) != LMMC_STATUS_OK);
    for (shape = 0; shape < sizeof(shapes) / sizeof(shapes[0]); ++shape) {
        const double a = shapes[shape][0], b = shapes[shape][1];
        const double mean = a / (a + b);
        const double second = mean * (a + 1.0) / (a + b + 1.0);
        double sum = 0.0, sum_sq = 0.0;
        assert_int_equal(lmmc_rng_seed(fixture->rng, 1 + shape), LMMC_STATUS_OK);
        for (i = 0; i < count; ++i) {
            double value;
            assert_int_equal(lmmc_rng_beta(fixture->rng, a, b, &value),
                             LMMC_STATUS_OK);
            assert_true(isfinite(value));
            assert_true(value >= 0.0 && value <= 1.0);
            sum += value;
            sum_sq += value * value;
        }
        if (!(fabs(sum / count - mean) <= 0.01) || !(fabs(sum_sq / count - second) <= 0.01)) {
            fail_msg("Beta(%g,%g) moments: mean=%g second=%g\n", a, b, sum / count, sum_sq / count);
        }
    }
}

static void test_chi_squared_mean_variance(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double df = 5.0;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 789), LMMC_STATUS_OK);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_chi_squared(fixture->rng, df, &val);
        assert_false(st != LMMC_STATUS_OK);
        sum += val;
        sum_sq += val * val;
    }

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* Chi2(df): mean = df, var = 2*df */
    double theo_mean = df;
    double theo_var = 2.0 * df;

    check_mean_variance("ChiSq(5)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

static void test_student_t_mean_variance(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double df = 10.0; /* df > 2 so variance is finite */

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 1001), LMMC_STATUS_OK);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_student_t(fixture->rng, df, &val);
        assert_false(st != LMMC_STATUS_OK);
        sum += val;
        sum_sq += val * val;
    }

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* t(df): mean = 0 (for df > 1), var = df/(df-2) (for df > 2) */
    double theo_mean = 0.0;
    double theo_var = df / (df - 2.0);

    check_mean_variance("StudentT(10)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

static void test_f_mean_variance(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double df1 = 5.0, df2 = 20.0; /* df2 > 4 so variance is finite */

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 2002), LMMC_STATUS_OK);

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_f(fixture->rng, df1, df2, &val);
        assert_false(st != LMMC_STATUS_OK);
        sum += val;
        sum_sq += val * val;
    }

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* F(d1,d2): mean = d2/(d2-2) for d2>2, var = 2*d2^2*(d1+d2-2)/(d1*(d2-2)^2*(d2-4)) for d2>4 */
    double theo_mean = df2 / (df2 - 2.0);
    double theo_var = (2.0 * df2 * df2 * (df1 + df2 - 2.0)) /
                      (df1 * (df2 - 2.0) * (df2 - 2.0) * (df2 - 4.0));

    check_mean_variance("F(5,20)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

static void test_poisson_mean_variance(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    double lambda = 15.0; /* Use lambda >= 10 to exercise PTRD path */

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 3003), LMMC_STATUS_OK);

    for (i = 0; i < NUM_SAMPLES; i++) {
        size_t val;
        st = lmmc_rng_poisson(fixture->rng, lambda, &val);
        assert_false(st != LMMC_STATUS_OK);
        sum += (double)val;
        sum_sq += (double)val * (double)val;
    }

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* Poisson(lambda): mean = lambda, var = lambda */
    double theo_mean = lambda;
    double theo_var = lambda;

    check_mean_variance("Poisson(15)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

static void test_binomial_mean_variance(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double sum = 0.0, sum_sq = 0.0;
    size_t i;
    size_t n_trials = 50;
    double p = 0.3;

    st = lmmc_rng_create(&fixture->rng);
    assert_false(st != LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 4004), LMMC_STATUS_OK);

    for (i = 0; i < NUM_SAMPLES; i++) {
        size_t val;
        st = lmmc_rng_binomial(fixture->rng, n_trials, p, &val);
        assert_false(st != LMMC_STATUS_OK);
        sum += (double)val;
        sum_sq += (double)val * (double)val;
    }

    double emp_mean = sum / NUM_SAMPLES;
    double emp_var = (sum_sq / NUM_SAMPLES) - (emp_mean * emp_mean);
    /* Binomial(n,p): mean = n*p, var = n*p*(1-p) */
    double theo_mean = (double)n_trials * p;
    double theo_var = (double)n_trials * p * (1.0 - p);

    check_mean_variance("Binomial(50,0.3)", emp_mean, emp_var, theo_mean, theo_var, NUM_SAMPLES);
}

/**
 * @brief Chi-squared goodness-of-fit test for the normal sampler.
 *
 * Draws 1,000,000 samples from N(0,1), bins into 256 equal-probability bins
 * (using the normal quantile function to determine bin edges), computes the
 * chi-squared statistic, and verifies p-value > 1e-6.
 */
static void normal_probability_bin_edges(double *bin_edges) {
    /**
     * @brief N(0,1) 下每个分箱的概率为 1/NUM_BINS。
     * 有限极值哨兵保持原有尾部分箱归属。
     */
    bin_edges[0] = -1e30;
    bin_edges[NUM_BINS] = 1e30;
    for (size_t i = 1; i < NUM_BINS; i++) {
        lmmc_real_t q;
        double p_val = (double)i / (double)NUM_BINS;
        lmmc_status_t st = lmmc_dist_normal_quantile(p_val, 0.0, 1.0, &q);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    FAIL: lmmc_dist_normal_quantile returned %d for p=%.6f\n", (int)st, p_val);
        }
        bin_edges[i] = (double)q;
    }
}

static void test_normal_chi_squared_gof(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    size_t bins[NUM_BINS];
    size_t i;
    double chi2_stat;
    double p_value;
    double expected;
    double bin_edges[NUM_BINS + 1];

    st = lmmc_rng_create(&fixture->rng);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    FAIL: lmmc_rng_create returned %d\n", (int)st);
    }
    assert_int_equal(lmmc_rng_seed(fixture->rng, UINT64_C(0xDEADBEEF12345678)), LMMC_STATUS_OK);

    normal_probability_bin_edges(bin_edges);

    memset(bins, 0, sizeof(bins));

    /* Draw samples and bin them */
    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t val;
        st = lmmc_rng_normal(fixture->rng, 0.0, 1.0, &val);
        assert_false(st != LMMC_STATUS_OK);

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


    if (!(p_value > 1e-6)) {
        fail_msg("    FAIL: p-value %.6e <= 1e-6\n", p_value);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_normal_mean_variance, setup, teardown),
        cmocka_unit_test_setup_teardown(test_gamma_mean_variance, setup, teardown),
        cmocka_unit_test_setup_teardown(test_beta_mean_variance, setup, teardown),
        cmocka_unit_test_setup_teardown(test_beta_small_shape_moments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_chi_squared_mean_variance, setup, teardown),
        cmocka_unit_test_setup_teardown(test_student_t_mean_variance, setup, teardown),
        cmocka_unit_test_setup_teardown(test_f_mean_variance, setup, teardown),
        cmocka_unit_test_setup_teardown(test_poisson_mean_variance, setup, teardown),
        cmocka_unit_test_setup_teardown(test_binomial_mean_variance, setup, teardown),
        cmocka_unit_test_setup_teardown(test_normal_chi_squared_gof, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
