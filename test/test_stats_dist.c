/**
 * @file test_stats_dist.c
 * 概率分布与描述性统计测试。
 */
#include <stdio.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("FAIL: %s\n", msg); } \
} while(0)

static void test_median(void) {
    lmmc_real_t data_odd[] = {5.0, 1.0, 3.0, 2.0, 4.0};
    lmmc_vec_t v_odd = {5, data_odd, 0};
    lmmc_real_t med;

    CHECK(lmmc_vec_median(&v_odd, &med) == LMMC_STATUS_OK, "median odd status");
    CHECK(lmmc_test_nearly_equal(med, 3.0, 1e-12), "median odd value");

    lmmc_real_t data_even[] = {1.0, 2.0, 3.0, 4.0};
    lmmc_vec_t v_even = {4, data_even, 0};
    CHECK(lmmc_vec_median(&v_even, &med) == LMMC_STATUS_OK, "median even status");
    CHECK(lmmc_test_nearly_equal(med, 2.5, 1e-12), "median even value");
}

static void test_quantile(void) {
    lmmc_real_t data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    lmmc_vec_t v = {5, data, 0};
    lmmc_real_t q;

    CHECK(lmmc_vec_quantile(&v, 0.0, &q) == LMMC_STATUS_OK, "quantile p=0");
    CHECK(lmmc_test_nearly_equal(q, 1.0, 1e-12), "quantile p=0 val");

    CHECK(lmmc_vec_quantile(&v, 1.0, &q) == LMMC_STATUS_OK, "quantile p=1");
    CHECK(lmmc_test_nearly_equal(q, 5.0, 1e-12), "quantile p=1 val");

    CHECK(lmmc_vec_quantile(&v, 0.5, &q) == LMMC_STATUS_OK, "quantile p=0.5");
    CHECK(lmmc_test_nearly_equal(q, 3.0, 1e-12), "quantile p=0.5 val");

    CHECK(lmmc_vec_quantile(&v, 0.25, &q) == LMMC_STATUS_OK, "quantile p=0.25");
    CHECK(lmmc_test_nearly_equal(q, 2.0, 1e-12), "quantile p=0.25 val");
}

static void test_histogram(void) {
    lmmc_real_t data[] = {0.5, 1.5, 2.5, 3.5, 4.5};
    lmmc_vec_t v = {5, data, 0};
    lmmc_real_t edges[6];
    size_t counts[5];

    CHECK(lmmc_vec_histogram(&v, 5, edges, counts) == LMMC_STATUS_OK, "histogram status");
    /* Each bin should have 1 element */
    size_t total = 0;
    for (int i = 0; i < 5; i++) total += counts[i];
    CHECK(total == 5, "histogram total count");
}

static void test_normal_dist(void) {
    lmmc_real_t val;

    /* PDF at mean */
    CHECK(lmmc_dist_normal_pdf(0.0, 0.0, 1.0, &val) == LMMC_STATUS_OK, "normal pdf status");
    CHECK(lmmc_test_nearly_equal(val, 0.3989422804014327, 1e-10), "normal pdf(0,0,1)");

    /* CDF at 0 */
    CHECK(lmmc_dist_normal_cdf(0.0, 0.0, 1.0, &val) == LMMC_STATUS_OK, "normal cdf status");
    CHECK(lmmc_test_nearly_equal(val, 0.5, 1e-10), "normal cdf(0,0,1)");

    /* CDF at 1.96 */
    CHECK(lmmc_dist_normal_cdf(1.96, 0.0, 1.0, &val) == LMMC_STATUS_OK, "normal cdf 1.96");
    CHECK(lmmc_test_nearly_equal(val, 0.9750021048517796, 1e-6), "normal cdf(1.96)");

    /* Quantile at 0.5 */
    CHECK(lmmc_dist_normal_quantile(0.5, 0.0, 1.0, &val) == LMMC_STATUS_OK, "normal quantile");
    CHECK(lmmc_test_nearly_equal(val, 0.0, 1e-8), "normal quantile(0.5)=0");

    /* Round-trip: quantile(cdf(x)) == x */
    lmmc_real_t cdf_val;
    lmmc_dist_normal_cdf(1.5, 0.0, 1.0, &cdf_val);
    lmmc_dist_normal_quantile(cdf_val, 0.0, 1.0, &val);
    CHECK(lmmc_test_nearly_equal(val, 1.5, 1e-8), "normal round-trip x=1.5");
}

static void test_t_dist(void) {
    lmmc_real_t val;

    /* t CDF at 0 should be 0.5 */
    CHECK(lmmc_dist_t_cdf(0.0, 5.0, &val) == LMMC_STATUS_OK, "t cdf status");
    CHECK(lmmc_test_nearly_equal(val, 0.5, 1e-10), "t cdf(0, df=5)=0.5");

    /* Round-trip */
    lmmc_real_t cdf_val;
    lmmc_dist_t_cdf(2.0, 10.0, &cdf_val);
    lmmc_dist_t_quantile(cdf_val, 10.0, &val);
    CHECK(lmmc_test_nearly_equal(val, 2.0, 1e-6), "t round-trip x=2 df=10");
}

static void test_chi2_dist(void) {
    lmmc_real_t val;

    /* chi2 CDF at 0 should be 0 */
    CHECK(lmmc_dist_chi2_cdf(0.0, 5.0, &val) == LMMC_STATUS_OK, "chi2 cdf(0)");
    CHECK(lmmc_test_nearly_equal(val, 0.0, 1e-10), "chi2 cdf(0)=0");

    /* Round-trip */
    lmmc_real_t cdf_val;
    lmmc_dist_chi2_cdf(5.0, 3.0, &cdf_val);
    lmmc_dist_chi2_quantile(cdf_val, 3.0, &val);
    CHECK(lmmc_test_nearly_equal(val, 5.0, 1e-6), "chi2 round-trip x=5 df=3");
}

static void test_f_dist(void) {
    lmmc_real_t val;

    /* F CDF at 0 should be 0 */
    CHECK(lmmc_dist_f_cdf(0.0, 5.0, 10.0, &val) == LMMC_STATUS_OK, "f cdf(0)");
    CHECK(lmmc_test_nearly_equal(val, 0.0, 1e-10), "f cdf(0)=0");

    /* Round-trip */
    lmmc_real_t cdf_val;
    lmmc_dist_f_cdf(2.0, 5.0, 10.0, &cdf_val);
    lmmc_dist_f_quantile(cdf_val, 5.0, 10.0, &val);
    CHECK(lmmc_test_nearly_equal(val, 2.0, 1e-5), "f round-trip x=2 df1=5 df2=10");
}

static void test_gamma_dist(void) {
    lmmc_real_t val;

    /* Gamma CDF at 0 should be 0 */
    CHECK(lmmc_dist_gamma_cdf(0.0, 2.0, 1.0, &val) == LMMC_STATUS_OK, "gamma cdf(0)");
    CHECK(lmmc_test_nearly_equal(val, 0.0, 1e-10), "gamma cdf(0)=0");

    /* Gamma(1, 1) = Exponential(1): CDF(1) = 1 - e^{-1} */
    CHECK(lmmc_dist_gamma_cdf(1.0, 1.0, 1.0, &val) == LMMC_STATUS_OK, "gamma=exp cdf");
    CHECK(lmmc_test_nearly_equal(val, 1.0 - exp(-1.0), 1e-10), "gamma=exp cdf(1)");

    /* Round-trip */
    lmmc_real_t cdf_val;
    lmmc_dist_gamma_cdf(3.0, 2.0, 1.5, &cdf_val);
    lmmc_dist_gamma_quantile(cdf_val, 2.0, 1.5, &val);
    CHECK(lmmc_test_nearly_equal(val, 3.0, 1e-6), "gamma round-trip");
}

static void test_beta_dist(void) {
    lmmc_real_t val;

    /* Beta CDF at 0 should be 0, at 1 should be 1 */
    CHECK(lmmc_dist_beta_cdf(0.0, 2.0, 3.0, &val) == LMMC_STATUS_OK, "beta cdf(0)");
    CHECK(lmmc_test_nearly_equal(val, 0.0, 1e-10), "beta cdf(0)=0");
    CHECK(lmmc_dist_beta_cdf(1.0, 2.0, 3.0, &val) == LMMC_STATUS_OK, "beta cdf(1)");
    CHECK(lmmc_test_nearly_equal(val, 1.0, 1e-10), "beta cdf(1)=1");

    /* Beta(1,1) = Uniform(0,1): CDF(0.5) = 0.5 */
    CHECK(lmmc_dist_beta_cdf(0.5, 1.0, 1.0, &val) == LMMC_STATUS_OK, "beta uniform");
    CHECK(lmmc_test_nearly_equal(val, 0.5, 1e-10), "beta(1,1) cdf(0.5)=0.5");

    /* Round-trip */
    lmmc_real_t cdf_val;
    lmmc_dist_beta_cdf(0.3, 2.0, 5.0, &cdf_val);
    lmmc_dist_beta_quantile(cdf_val, 2.0, 5.0, &val);
    CHECK(lmmc_test_nearly_equal(val, 0.3, 1e-6), "beta round-trip");
}

static void test_binomial_dist(void) {
    lmmc_real_t val;

    /* Binomial(10, 0.5) PMF at k=5 */
    CHECK(lmmc_dist_binomial_pmf(5, 10, 0.5, &val) == LMMC_STATUS_OK, "binom pmf");
    /* C(10,5) * 0.5^10 = 252/1024 = 0.24609375 */
    CHECK(lmmc_test_nearly_equal(val, 0.24609375, 1e-10), "binom pmf(5,10,0.5)");

    /* CDF should sum to 1 at k=n */
    CHECK(lmmc_dist_binomial_cdf(10, 10, 0.5, &val) == LMMC_STATUS_OK, "binom cdf(n)");
    CHECK(lmmc_test_nearly_equal(val, 1.0, 1e-10), "binom cdf(n)=1");
}

static void test_poisson_dist(void) {
    lmmc_real_t val;

    /* Poisson(3) PMF at k=0: e^{-3} */
    CHECK(lmmc_dist_poisson_pmf(0, 3.0, &val) == LMMC_STATUS_OK, "poisson pmf(0)");
    CHECK(lmmc_test_nearly_equal(val, exp(-3.0), 1e-10), "poisson pmf(0,3)");

    /* Poisson(3) PMF at k=3: 3^3 * e^{-3} / 3! = 27*e^{-3}/6 */
    CHECK(lmmc_dist_poisson_pmf(3, 3.0, &val) == LMMC_STATUS_OK, "poisson pmf(3)");
    CHECK(lmmc_test_nearly_equal(val, 27.0 * exp(-3.0) / 6.0, 1e-10), "poisson pmf(3,3)");
}

int main(void) {
    if (lmmc_init() != LMMC_STATUS_OK) return 1;

    test_median();
    test_quantile();
    test_histogram();
    test_normal_dist();
    test_t_dist();
    test_chi2_dist();
    test_f_dist();
    test_gamma_dist();
    test_beta_dist();
    test_binomial_dist();
    test_poisson_dist();

    printf("\n=== Stats Distribution Tests ===\n");
    printf("Passed: %d, Failed: %d\n", g_pass, g_fail);

    if (lmmc_deinit() != LMMC_STATUS_OK) return 1;
    return g_fail > 0 ? 1 : 0;
}
