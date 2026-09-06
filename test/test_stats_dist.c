/**
 * @file test_stats_dist.c
 * 概率分布与描述性统计测试。
 */
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
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

    {
        lmmc_real_t data_extreme[] = {DBL_MAX, DBL_MAX};
        lmmc_vec_t v_extreme = {2, data_extreme, 0};
        CHECK(lmmc_vec_median(&v_extreme, &med) == LMMC_STATUS_OK,
              "median accepts equal extreme finite values");
        CHECK(med == DBL_MAX,
              "even median avoids overflowing a representable midpoint");
    }

    {
        lmmc_real_t data_nonfinite[] = {1.0, NAN, 3.0};
        lmmc_vec_t v_nonfinite = {3, data_nonfinite, 0};
        med = 17.0;
        CHECK(lmmc_vec_median(&v_nonfinite, &med) ==
                  LMMC_STATUS_NUMERICAL_FAILURE,
              "median rejects nonfinite observations");
        CHECK(med == 17.0,
              "failed median evaluation preserves the output");
    }
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

    q = 17.0;
    CHECK(lmmc_vec_quantile(&v, NAN, &q) ==
              LMMC_STATUS_INVALID_ARGUMENT,
          "quantile rejects a NaN probability before integer conversion");
    CHECK(q == 17.0,
          "invalid quantile probability preserves the output");

    {
        lmmc_real_t data_nonfinite[] = {1.0, NAN, 3.0};
        lmmc_vec_t v_nonfinite = {3, data_nonfinite, 0};
        CHECK(lmmc_vec_quantile(&v_nonfinite, 0.5, &q) ==
                  LMMC_STATUS_NUMERICAL_FAILURE,
              "quantile rejects nonfinite observations");
        CHECK(q == 17.0,
              "failed quantile evaluation preserves the output");
    }
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

    {
        lmmc_real_t extreme_data[] = {-DBL_MAX, DBL_MAX};
        lmmc_vec_t extreme = {2, extreme_data, 0};
        lmmc_real_t extreme_edges[3];
        size_t extreme_counts[2];

        CHECK(lmmc_vec_histogram(
                  &extreme, 2, extreme_edges, extreme_counts) ==
                  LMMC_STATUS_OK,
              "histogram accepts a finite full-range dataset");
        CHECK(extreme_edges[0] == -DBL_MAX &&
                  extreme_edges[1] == 0.0 &&
                  extreme_edges[2] == DBL_MAX,
              "histogram constructs finite equal-width full-range edges");
        CHECK(extreme_counts[0] == 1 && extreme_counts[1] == 1,
              "histogram bins both full-range endpoints");
    }

    {
        lmmc_real_t nonfinite_data[] = {1.0, NAN};
        lmmc_vec_t nonfinite = {2, nonfinite_data, 0};
        lmmc_real_t preserved_edges[3] = {7.0, 8.0, 9.0};
        size_t preserved_counts[2] = {10, 11};

        CHECK(lmmc_vec_histogram(
                  &nonfinite, 2, preserved_edges, preserved_counts) ==
                  LMMC_STATUS_NUMERICAL_FAILURE,
              "histogram rejects nonfinite observations");
        CHECK(preserved_edges[0] == 7.0 &&
                  preserved_edges[1] == 8.0 &&
                  preserved_edges[2] == 9.0 &&
                  preserved_counts[0] == 10 &&
                  preserved_counts[1] == 11,
              "failed histogram evaluation preserves output arrays");
    }
}

static void test_normal_dist(void) {
    lmmc_real_t val;

    /* PDF at mean */
    CHECK(lmmc_dist_normal_pdf(0.0, 0.0, 1.0, &val) == LMMC_STATUS_OK, "normal pdf status");
    CHECK(lmmc_test_nearly_equal(val, 0.3989422804014327, 1e-10), "normal pdf(0,0,1)");

    val = 123.0;
    CHECK(lmmc_dist_normal_pdf(
              0.0, 0.0, DBL_MIN * 0.0625, &val) ==
              LMMC_STATUS_NUMERICAL_FAILURE,
          "normal pdf reports an unrepresentable density");
    CHECK(val == 123.0,
          "failed normal pdf evaluation preserves the output");

    /* CDF at 0 */
    CHECK(lmmc_dist_normal_cdf(0.0, 0.0, 1.0, &val) == LMMC_STATUS_OK, "normal cdf status");
    CHECK(lmmc_test_nearly_equal(val, 0.5, 1e-10), "normal cdf(0,0,1)");

    /* CDF at 1.96 */
    CHECK(lmmc_dist_normal_cdf(1.96, 0.0, 1.0, &val) == LMMC_STATUS_OK, "normal cdf 1.96");
    CHECK(lmmc_test_nearly_equal(val, 0.9750021048517796, 1e-6), "normal cdf(1.96)");

    {
        const lmmc_real_t expected = 0.5 * erfc(10.0 / sqrt(2.0));
        CHECK(lmmc_dist_normal_cdf(-10.0, 0.0, 1.0, &val) ==
                  LMMC_STATUS_OK,
              "normal lower-tail CDF status");
        CHECK(fabs(val - expected) <= expected * 1e-12,
              "normal CDF preserves a representable lower tail");
    }

    {
        const lmmc_real_t expected_cdf =
            0.5 * erfc(-2.0 / sqrt(2.0));
        const lmmc_real_t expected_pdf =
            exp(-2.0) / sqrt(2.0 * LMMC_PI) / DBL_MAX;
        CHECK(lmmc_dist_normal_cdf(
                  DBL_MAX, -DBL_MAX, DBL_MAX, &val) ==
                  LMMC_STATUS_OK,
              "normal CDF accepts an overflowing raw centered difference");
        CHECK(fabs(val - expected_cdf) < 1e-15,
              "normal CDF standardizes extreme finite parameters without overflow");
        CHECK(lmmc_dist_normal_pdf(
                  DBL_MAX, -DBL_MAX, DBL_MAX, &val) ==
                  LMMC_STATUS_OK,
              "normal PDF accepts an overflowing raw centered difference");
        CHECK(val > 0.0 &&
                  fabs(val - expected_pdf) <= expected_pdf * 1e-12,
              "normal PDF preserves a representable extreme-scale density");
    }

    /* Quantile at 0.5 */
    CHECK(lmmc_dist_normal_quantile(0.5, 0.0, 1.0, &val) == LMMC_STATUS_OK, "normal quantile");
    CHECK(lmmc_test_nearly_equal(val, 0.0, 1e-8), "normal quantile(0.5)=0");

    /* Round-trip: quantile(cdf(x)) == x */
    lmmc_real_t cdf_val;
    lmmc_dist_normal_cdf(1.5, 0.0, 1.0, &cdf_val);
    lmmc_dist_normal_quantile(cdf_val, 0.0, 1.0, &val);
    CHECK(lmmc_test_nearly_equal(val, 1.5, 1e-8), "normal round-trip x=1.5");

    {
        const lmmc_real_t sigma = DBL_MAX * 0.5;
        const lmmc_real_t p = 0.0013498980316300933;
        CHECK(lmmc_dist_normal_quantile(
                  p, DBL_MAX, sigma, &val) == LMMC_STATUS_OK,
              "normal quantile fuses a cancelling location-scale transform");
        CHECK(isfinite(val) &&
                  fabs(val / DBL_MAX + 0.5) < 1e-8,
              "normal quantile preserves a representable cancelling result");
    }
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

    {
        const lmmc_real_t extreme = DBL_MAX;
        const lmmc_real_t expected = atan(1.0 / extreme) / LMMC_PI;
        CHECK(lmmc_dist_t_cdf(-extreme, 1.0, &val) == LMMC_STATUS_OK,
              "Cauchy lower-tail CDF accepts an extreme finite variate");
        CHECK(val > 0.0 &&
                  fabs(val - expected) <= expected * 1e-12,
              "Student t CDF preserves a representable extreme lower tail");
    }

    {
        const lmmc_real_t extreme = 1.0e200;
        const lmmc_real_t df = 0.1;
        const lmmc_real_t log_power =
            2.0 * log(extreme) - log(df);
        const lmmc_real_t expected = exp(
            lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0) -
            0.5 * (log(df) + log(LMMC_PI)) -
            ((df + 1.0) / 2.0) * log_power);
        CHECK(lmmc_dist_t_pdf(extreme, df, &val) == LMMC_STATUS_OK,
              "Student t PDF accepts an overflowing raw square");
        CHECK(val > 0.0 &&
                  fabs(val - expected) <= expected * 1e-12,
              "Student t PDF preserves a representable heavy tail");
    }

    {
        const lmmc_real_t expected_cdf =
            0.84134474606854294859;
        const lmmc_real_t expected_quantile =
            1.95996398454005423552;
        CHECK(lmmc_dist_t_pdf(0.0, DBL_MAX, &val) ==
                  LMMC_STATUS_OK,
              "extreme-degree Student t PDF remains computable");
        CHECK(fabs(val - 1.0 / sqrt(2.0 * LMMC_PI)) < 1e-15,
              "extreme-degree Student t PDF reaches the normal limit");
        CHECK(lmmc_dist_t_cdf(1.0, DBL_MAX, &val) ==
                  LMMC_STATUS_OK,
              "extreme-degree Student t CDF remains computable");
        CHECK(fabs(val - expected_cdf) < 1e-15,
              "extreme-degree Student t CDF reaches the normal limit");
        CHECK(lmmc_dist_t_quantile(0.975, DBL_MAX, &val) ==
                  LMMC_STATUS_OK,
              "extreme-degree Student t quantile remains computable");
        CHECK(fabs(val - expected_quantile) < 5e-8,
              "extreme-degree Student t quantile reaches the normal limit");
    }
}

static void test_chi2_dist(void) {
    lmmc_real_t val;

    {
        const lmmc_real_t expected =
            0.5 / sqrt(LMMC_PI) / sqrt(DBL_MAX);
        CHECK(lmmc_dist_chi2_pdf(DBL_MAX, DBL_MAX, &val) ==
                  LMMC_STATUS_OK,
              "extreme-degree central chi-square density remains computable");
        CHECK(val > 0.0 &&
                  fabs(val / expected - 1.0) < 1e-12,
              "chi-square density avoids large log-gamma cancellation");
    }

    /* chi2 CDF at 0 should be 0 */
    CHECK(lmmc_dist_chi2_cdf(0.0, 5.0, &val) == LMMC_STATUS_OK, "chi2 cdf(0)");
    CHECK(lmmc_test_nearly_equal(val, 0.0, 1e-10), "chi2 cdf(0)=0");

    /* Round-trip */
    lmmc_real_t cdf_val;
    lmmc_dist_chi2_cdf(5.0, 3.0, &cdf_val);
    lmmc_dist_chi2_quantile(cdf_val, 3.0, &val);
    CHECK(lmmc_test_nearly_equal(val, 5.0, 1e-6), "chi2 round-trip x=5 df=3");

    {
        const lmmc_real_t expected =
            erf(sqrt(DBL_TRUE_MIN) / sqrt(2.0));
        CHECK(lmmc_dist_chi2_cdf(DBL_TRUE_MIN, 1.0, &val) ==
                  LMMC_STATUS_OK,
              "chi-square CDF accepts a halving-underflow variate");
        CHECK(val > 0.0 &&
                  fabs(val / expected - 1.0) < 1e-12,
              "chi-square CDF preserves a representable extreme lower tail");
    }
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

    {
        const lmmc_real_t expected =
            sqrt(DBL_MAX / (8.0 * LMMC_PI));
        CHECK(lmmc_dist_f_pdf(1.0, DBL_MAX, DBL_MAX, &val) ==
                  LMMC_STATUS_OK,
              "equal extreme-degree F density remains computable");
        CHECK(isfinite(val) &&
                  fabs(val / expected - 1.0) < 1e-12,
              "F density avoids large log-gamma cancellation");
    }

    CHECK(lmmc_dist_f_pdf(0.8, 32.0, 40.0, &val) ==
              LMMC_STATUS_OK,
          "stable F-density kernel accepts its large-shape boundary");
    CHECK(fabs(val - 1.1859038610805410002) <
              1.1859038610805410002e-12,
          "stable F-density kernel matches the reference density");
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

    {
        const lmmc_real_t expected =
            erf(sqrt(DBL_MIN) / sqrt(DBL_MAX));
        CHECK(lmmc_dist_gamma_cdf(
                  DBL_MIN, 0.5, DBL_MAX, &val) ==
                  LMMC_STATUS_OK,
              "gamma CDF accepts an underflowing standardized variate");
        CHECK(val > 0.0 &&
                  fabs(val / expected - 1.0) < 1e-12,
              "gamma CDF preserves a representable subnormal-scale tail");
    }

    {
        const lmmc_real_t shape = 1.0e18;
        const lmmc_real_t expected =
            1.0 / sqrt(2.0 * LMMC_PI * shape);
        CHECK(lmmc_dist_gamma_pdf(shape, shape, 1.0, &val) ==
                  LMMC_STATUS_OK,
              "large-shape central gamma density remains computable");
        CHECK(isfinite(val) &&
                  fabs(val / expected - 1.0) < 1e-12,
              "large-shape gamma PDF avoids log-gamma cancellation");
    }

    CHECK(lmmc_dist_gamma_pdf(30.0, 16.0, 2.0, &val) ==
              LMMC_STATUS_OK,
          "stable gamma PDF kernel accepts its large-shape boundary");
    CHECK(fabs(val - 0.051217933332267094) <
              0.051217933332267094e-12,
          "stable gamma PDF kernel matches the reference density");
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

    {
        const lmmc_real_t shape = 1.0e18;
        const lmmc_real_t expected =
            sqrt(4.0 * shape / LMMC_PI);
        CHECK(lmmc_dist_beta_pdf(0.5, shape, shape, &val) ==
                  LMMC_STATUS_OK,
              "large symmetric beta center density remains computable");
        CHECK(isfinite(val) &&
                  fabs(val / expected - 1.0) < 1e-12,
              "large symmetric beta PDF avoids log-gamma cancellation");
    }

    {
        const lmmc_real_t shape = DBL_MAX;
        const lmmc_real_t expected =
            2.0 * sqrt(shape / LMMC_PI);
        CHECK(lmmc_dist_beta_pdf(0.5, shape, shape, &val) ==
                  LMMC_STATUS_OK,
              "overflowing-shape-sum beta center remains computable");
        CHECK(isfinite(val) &&
                  fabs(val / expected - 1.0) < 1e-12,
              "symmetric beta PDF avoids overflowing alpha+beta");
    }

    CHECK(lmmc_dist_beta_pdf(0.4, 16.0, 20.0, &val) ==
              LMMC_STATUS_OK,
          "stable beta PDF kernel accepts its large-shape boundary");
    CHECK(fabs(val - 4.2502261912068876) <
              4.2502261912068876e-12,
          "stable beta PDF kernel matches the reference density");

    /* Round-trip */
    lmmc_real_t cdf_val;
    lmmc_dist_beta_cdf(0.3, 2.0, 5.0, &cdf_val);
    lmmc_dist_beta_quantile(cdf_val, 2.0, 5.0, &val);
    CHECK(lmmc_test_nearly_equal(val, 0.3, 1e-6), "beta round-trip");
}

static void test_quantile_tail_regressions(void) {
    lmmc_real_t q, cdf;
    lmmc_status_t status;

    status = lmmc_dist_chi2_quantile(0.01, 0.1, &q);
    CHECK(status == LMMC_STATUS_OK, "chi-square lower-tail quantile status");
    if (status == LMMC_STATUS_OK) {
        status = lmmc_dist_chi2_cdf(q, 0.1, &cdf);
        CHECK(status == LMMC_STATUS_OK && fabs(cdf - 0.01) < 1e-10,
              "chi-square lower-tail quantile round-trip");
    }

    status = lmmc_dist_gamma_quantile(0.01, 2.0, 2.0, &q);
    CHECK(status == LMMC_STATUS_OK, "gamma lower-tail quantile status");
    if (status == LMMC_STATUS_OK) {
        status = lmmc_dist_gamma_cdf(q, 2.0, 2.0, &cdf);
        CHECK(status == LMMC_STATUS_OK && fabs(cdf - 0.01) < 1e-10,
              "gamma lower-tail quantile round-trip");
    }

    status = lmmc_dist_f_quantile(0.5, 100.0, 1.0, &q);
    CHECK(status == LMMC_STATUS_OK, "F median quantile status");
    if (status == LMMC_STATUS_OK) {
        status = lmmc_dist_f_cdf(q, 100.0, 1.0, &cdf);
        CHECK(status == LMMC_STATUS_OK && fabs(cdf - 0.5) < 1e-10,
              "F median quantile round-trip");
    }

    status = lmmc_dist_beta_quantile(0.99, 100.0, 2.0, &q);
    CHECK(status == LMMC_STATUS_OK, "beta upper-tail quantile status");
    if (status == LMMC_STATUS_OK) {
        status = lmmc_dist_beta_cdf(q, 100.0, 2.0, &cdf);
        CHECK(status == LMMC_STATUS_OK && fabs(cdf - 0.99) < 1e-10,
              "beta upper-tail quantile round-trip");
    }

}

static void test_density_failure_contracts(void) {
    lmmc_real_t val = 123.0;

    CHECK(lmmc_dist_chi2_pdf(0.0, 1.0, &val) ==
              LMMC_STATUS_NUMERICAL_FAILURE,
          "chi-square pdf reports an infinite boundary density");
    CHECK(val == 123.0,
          "failed chi-square pdf preserves the output");
    CHECK(lmmc_dist_f_pdf(0.0, 1.0, 2.0, &val) ==
              LMMC_STATUS_NUMERICAL_FAILURE,
          "F pdf reports an infinite boundary density");
    CHECK(val == 123.0, "failed F pdf preserves the output");
    CHECK(lmmc_dist_gamma_pdf(0.0, 0.5, 1.0, &val) ==
              LMMC_STATUS_NUMERICAL_FAILURE,
          "gamma pdf reports an infinite boundary density");
    CHECK(val == 123.0, "failed gamma pdf preserves the output");
    CHECK(lmmc_dist_beta_pdf(0.0, 0.5, 1.0, &val) ==
              LMMC_STATUS_NUMERICAL_FAILURE,
          "beta pdf reports an infinite boundary density");
    CHECK(val == 123.0, "failed beta pdf preserves the output");
}

static void test_cdf_large_parameter_contracts(void) {
    const lmmc_real_t gamma_center =
        0.5 + 1.0 / (3.0 * sqrt(2.0 * acos(-1.0) * 100000.0));
    lmmc_real_t val = 123.0;
    CHECK(lmmc_dist_beta_cdf(0.5, DBL_MAX, DBL_MAX, &val) ==
              LMMC_STATUS_OK,
          "extreme symmetric beta CDF remains computable");
    CHECK(val == 0.5,
          "extreme symmetric beta CDF is exactly one half at its center");
    CHECK(lmmc_dist_f_cdf(1.0, DBL_MAX, DBL_MAX, &val) ==
              LMMC_STATUS_OK,
          "equal-degree extreme F CDF remains computable");
    CHECK(val == 0.5,
          "equal-degree F CDF is exactly one half at one");

    CHECK(lmmc_dist_gamma_cdf(100000.0, 100000.0, 1.0, &val) ==
              LMMC_STATUS_OK,
          "large-shape central gamma CDF remains computable");
    CHECK(fabs(val - gamma_center) < 1e-8,
          "large-shape central gamma CDF has the expected asymptotic value");

    CHECK(lmmc_dist_beta_cdf(0.5, 10000000.0, 10000000.0, &val) ==
              LMMC_STATUS_OK,
          "symmetric large-shape beta CDF remains computable");
    CHECK(val == 0.5,
          "symmetric beta CDF is exactly one half at its center");

    CHECK(lmmc_dist_binomial_cdf(9999999, 19999999, 0.5, &val) ==
              LMMC_STATUS_OK,
          "large central binomial CDF remains computable");
    CHECK(val == 0.5,
          "odd-trial symmetric binomial CDF is exactly one half");

    CHECK(lmmc_dist_poisson_cdf(99999, 100000.0, &val) ==
              LMMC_STATUS_OK,
          "large-rate central Poisson CDF remains computable");
    CHECK(fabs(val - (1.0 - gamma_center)) < 1e-8,
          "large-rate central Poisson CDF has the expected asymptotic value");
}

static void test_continuous_dist_rejects_nonfinite(void) {
    lmmc_real_t val;

    CHECK(lmmc_dist_normal_pdf(NAN, 0.0, 1.0, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "normal pdf rejects NaN variate");
    CHECK(lmmc_dist_normal_cdf(0.0, NAN, 1.0, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "normal cdf rejects NaN mean");
    CHECK(lmmc_dist_normal_quantile(NAN, 0.0, 1.0, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "normal quantile rejects NaN probability");
    CHECK(lmmc_dist_normal_quantile(
              0.75, DBL_MAX, DBL_MAX, &val) ==
              LMMC_STATUS_NUMERICAL_FAILURE,
          "normal quantile reports an unrepresentable finite-input result");
    CHECK(lmmc_dist_t_cdf(0.0, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "t cdf rejects NaN degrees of freedom");
    CHECK(lmmc_dist_chi2_pdf(NAN, 2.0, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "chi-square pdf rejects NaN variate");
    CHECK(lmmc_dist_f_cdf(1.0, INFINITY, 2.0, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "F cdf rejects infinite degrees of freedom");
    CHECK(lmmc_dist_gamma_quantile(0.5, 1.0, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "gamma quantile rejects NaN scale");
    CHECK(lmmc_dist_beta_cdf(0.5, NAN, 1.0, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "beta cdf rejects NaN shape");
    CHECK(lmmc_dist_binomial_pmf(1, 2, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "binomial pmf rejects NaN probability");
    CHECK(lmmc_dist_binomial_cdf(1, 2, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "binomial cdf rejects NaN probability");
    CHECK(lmmc_dist_poisson_pmf(1, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "Poisson pmf rejects NaN rate");
    CHECK(lmmc_dist_poisson_cdf(1, NAN, &val) == LMMC_STATUS_INVALID_ARGUMENT,
          "Poisson cdf rejects NaN rate");
}

static void test_binomial_dist(void) {
    lmmc_real_t val;

    /* Binomial(10, 0.5) PMF at k=5 */
    CHECK(lmmc_dist_binomial_pmf(5, 10, 0.5, &val) == LMMC_STATUS_OK, "binom pmf");
    /* C(10,5) * 0.5^10 = 252/1024 = 0.24609375 */
    CHECK(lmmc_test_nearly_equal(val, 0.24609375, 1e-10), "binom pmf(5,10,0.5)");

    CHECK(lmmc_dist_binomial_pmf(0, SIZE_MAX, 0.5, &val) ==
              LMMC_STATUS_OK,
          "binomial pmf accepts the full size_t trial range");
    CHECK(isfinite(val) && val == 0.0,
          "binomial pmf underflows a valid extreme tail to zero");

    {
        const size_t large_n = (size_t)1000000000000000000ULL;
        const lmmc_real_t expected =
            sqrt(2.0 / (acos(-1.0) * (lmmc_real_t)large_n));
        CHECK(lmmc_dist_binomial_pmf(
                  large_n / 2, large_n, 0.5, &val) ==
                  LMMC_STATUS_OK,
              "central large-n binomial PMF remains computable");
        CHECK(isfinite(val) &&
                  fabs(val / expected - 1.0) < 1e-12,
              "central large-n binomial PMF avoids log-gamma cancellation");
    }

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

    {
        const size_t large_k = (size_t)1000000000000000000ULL;
        const lmmc_real_t lambda = 1.0e18;
        const lmmc_real_t expected =
            1.0 / sqrt(2.0 * acos(-1.0) * lambda);
        CHECK(lmmc_dist_poisson_pmf(
                  large_k, lambda, &val) == LMMC_STATUS_OK,
              "large-rate central Poisson PMF remains computable");
        CHECK(isfinite(val) &&
                  fabs(val / expected - 1.0) < 1e-12,
              "large-rate Poisson PMF avoids log-gamma cancellation");
    }
    CHECK(lmmc_dist_poisson_cdf(0, 40.0, &val) == LMMC_STATUS_OK,
          "Poisson upper-gamma tail status");
    CHECK(fabs(val - exp(-40.0)) <= exp(-40.0) * 1e-12,
          "Poisson CDF preserves a representable small tail");
    CHECK(lmmc_dist_poisson_cdf(SIZE_MAX, 40.0, &val) == LMMC_STATUS_OK,
          "Poisson CDF accepts the full size_t support");
    CHECK(val == 1.0, "Poisson CDF at SIZE_MAX is one");
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
    test_quantile_tail_regressions();
    test_density_failure_contracts();
    test_cdf_large_parameter_contracts();
    test_continuous_dist_rejects_nonfinite();
    test_binomial_dist();
    test_poisson_dist();

    printf("\n=== Stats Distribution Tests ===\n");
    printf("Passed: %d, Failed: %d\n", g_pass, g_fail);

    if (lmmc_deinit() != LMMC_STATUS_OK) return 1;
    return g_fail > 0 ? 1 : 0;
}
