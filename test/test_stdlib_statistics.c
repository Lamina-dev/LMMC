#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"
#include <float.h>

static void test_empty_inputs_quantile_bounds(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_real_t values[] = {1, 2, 3, 4};
    lmmc_real_t scaled_values[] = {2, 4, 6, 8};

    assert_true(lmmc_std_stats_mean(values, 0, &out) == LMMC_STATUS_EMPTY_INPUT);
    assert_true(lmmc_std_stats_median(values, 0, &out) == LMMC_STATUS_EMPTY_INPUT);
    assert_true(lmmc_std_stats_var(values, 0, &out) == LMMC_STATUS_EMPTY_INPUT);
    assert_true(lmmc_std_stats_std(values, 0, &out) == LMMC_STATUS_EMPTY_INPUT);
    assert_true(lmmc_std_stats_quantile(values, 0, 0.5, &out) == LMMC_STATUS_EMPTY_INPUT);
    assert_true(lmmc_std_stats_quantile(values, 4, -0.1, &out) == LMMC_STATUS_OUT_OF_RANGE);
    assert_true(lmmc_std_stats_quantile(values, 4, 1.1, &out) == LMMC_STATUS_OUT_OF_RANGE);
    assert_true(lmmc_std_stats_cov(values, scaled_values, 0, &out) == LMMC_STATUS_EMPTY_INPUT);
    assert_true(lmmc_std_stats_corr(values, scaled_values, 0, &out) == LMMC_STATUS_EMPTY_INPUT);
}

static void test_sample_statistics(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_real_t values[] = {1, 2, 3, 4};
    lmmc_real_t scaled_values[] = {2, 4, 6, 8};

    assert_true(lmmc_std_stats_mean(values, 4, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 2.5));

    assert_true(lmmc_std_stats_median(values, 4, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 2.5));

    assert_true(lmmc_std_stats_var(values, 4, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 5.0 / 3.0));

    assert_true(lmmc_std_stats_std(values, 4, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, sqrt(5.0 / 3.0)));

    assert_true(lmmc_std_stats_quantile(values, 4, 0.5, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 2.5));

    assert_true(lmmc_std_stats_cov(values, scaled_values, 4, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 10.0 / 3.0));

    assert_true(lmmc_std_stats_corr(values, scaled_values, 4, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 1.0));
}

static void test_neighboring_double_correlation(void **state) {
    (void)state;
    const lmmc_real_t xs[2][2] = {{1, nextafter(1, 0)}, {1, nextafter(1, 2)}};
    const lmmc_real_t ys[2][2] = {{0, 1}, {2, 1}};
    lmmc_real_t out = 17;
    for (size_t i = 0; i < 2; ++i) {
        assert_true(lmmc_std_stats_corr(xs[i], ys[i], 2, &out) == LMMC_STATUS_OK);
        assert_true(isfinite(out) && fabs(out + 1) <= 8 * DBL_EPSILON);
    }
    const lmmc_real_t constant[] = {1, 1};
    const lmmc_real_t nonfinite[] = {1, INFINITY};
    out = 17;
    assert_true(lmmc_std_stats_corr(constant, ys[0], 2, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(out == 17);
    assert_true(lmmc_std_stats_corr(nonfinite, ys[0], 2, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(out == 17);
}

static void test_invalid_arguments(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_real_t values[] = {1, 2, 3, 4};
    lmmc_real_t scaled_values[] = {2, 4, 6, 8};

    assert_true(lmmc_std_stats_mean(NULL, 4, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_mean(values, 4, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_median(NULL, 4, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_median(values, 4, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_var(NULL, 4, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_var(values, 4, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_std(NULL, 4, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_std(values, 4, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_quantile(NULL, 4, 0.5, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_quantile(values, 4, 0.5, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_cov(NULL, scaled_values, 4, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_cov(values, NULL, 4, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_cov(values, scaled_values, 4, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_corr(NULL, scaled_values, 4, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_corr(values, NULL, 4, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_corr(values, scaled_values, 4, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_nonfinite_inputs(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_real_t values[] = {1, 2, 3, 4};
    lmmc_real_t nonfinite_values[] = {1, NAN, 3};

    assert_true(lmmc_std_stats_mean(nonfinite_values, 3, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_median(nonfinite_values, 3, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_var(nonfinite_values, 3, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_std(nonfinite_values, 3, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_quantile(values, 4, NAN, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_cov(values, nonfinite_values, 3, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_corr(values, nonfinite_values, 3, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_zero_variance(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_real_t scaled_values[] = {2, 4, 6, 8};
    lmmc_real_t constant_values[] = {1, 1, 1, 1};

    assert_true(lmmc_std_stats_corr(constant_values, scaled_values, 4, &out) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_normal_distribution(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_stats_normal_pdf(0, 0, 1, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.3989422804014327));
    assert_true(lmmc_std_stats_normal_cdf(0, 0, 1, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.5));
    assert_true(lmmc_std_stats_normal_quantile(0.5, 0, 1, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.0));
}

static void test_student_t_distribution(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_stats_t_pdf(0, 1, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 1.0 / LMMC_PI));
    assert_true(lmmc_std_stats_t_cdf(0, 1, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.5));
    assert_true(lmmc_std_stats_t_quantile(0.5, 1, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.0));
}

static void test_chi_squared_distribution(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_stats_chi2_pdf(2, 2, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.5 * exp(-1.0)));
    assert_true(lmmc_std_stats_chi2_cdf(0, 2, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.0));
    assert_true(lmmc_std_stats_chi2_quantile(1.0 - exp(-1.0), 2, &out) == LMMC_STATUS_OK);
    assert_true(fabs((double)(out - 2.0)) <= 1e-8);
}

static void test_fisher_f_distribution(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_stats_f_pdf(1, 2, 2, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.25));
    assert_true(lmmc_std_stats_f_cdf(1, 2, 2, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.5));
    assert_true(lmmc_std_stats_f_quantile(0.5, 2, 2, &out) == LMMC_STATUS_OK);
    assert_true(fabs((double)(out - 1.0)) <= 1e-8);
}

static void test_gamma_distribution(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_stats_gamma_pdf(2, 1, 2, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.5 * exp(-1.0)));
    assert_true(lmmc_std_stats_gamma_cdf(0, 1, 2, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.0));
    assert_true(lmmc_std_stats_gamma_quantile(1.0 - exp(-1.0), 1, 2, &out) == LMMC_STATUS_OK);
    assert_true(fabs((double)(out - 2.0)) <= 1e-8);
}

static void test_beta_distribution(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_stats_beta_pdf(0.5, 1, 1, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 1.0));
    assert_true(lmmc_std_stats_beta_cdf(0.5, 1, 1, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.5));
    assert_true(lmmc_std_stats_beta_quantile(0.5, 1, 1, &out) == LMMC_STATUS_OK);
    assert_true(fabs((double)(out - 0.5)) <= 1e-8);
}

static void test_beta_large_shape_normalization(void **state) {
    (void)state;
    const double a = 1e16;
    const double x = nextafter(1.0, 0.0);
    const double power = exp((a - 1) * log(x));
    const double expected_one = a * power;
    const double expected_two = a * ((a + 1) * (1 - x)) * power;
    lmmc_real_t out = 17;
    assert_true(lmmc_std_stats_beta_pdf(x, a, 1, &out) == LMMC_STATUS_OK);
    assert_true(isfinite(out) && fabs(out / expected_one - 1) <= 2e-12);
    assert_true(lmmc_std_stats_beta_pdf(1 - x, 1, a, &out) == LMMC_STATUS_OK);
    assert_true(isfinite(out) && fabs(out / expected_one - 1) <= 2e-12);
    assert_true(lmmc_std_stats_beta_pdf(x, a, 2, &out) == LMMC_STATUS_OK);
    assert_true(isfinite(out) && fabs(out / expected_two - 1) <= 2e-12);
    out = 17;
    assert_true(lmmc_std_stats_beta_pdf(x, 0, 1, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(out == 17);
}

static void test_discrete_distribution(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_stats_binomial_pmf(2, 4, 0.5, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.375));
    assert_true(lmmc_std_stats_binomial_cdf(4, 4, 0.5, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 1.0));
    assert_true(lmmc_std_stats_poisson_pmf(2, 2, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 2.0 * exp(-2.0)));
    assert_true(lmmc_std_stats_poisson_cdf(0, 1, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, exp(-1.0)));
}

static void test_distribution_failures(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_stats_normal_pdf(0, 0, 0, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_normal_quantile(0, 0, 1, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_t_pdf(0, 0, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_chi2_pdf(0, 1, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_f_pdf(0, 1, 2, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_gamma_pdf(0, 0.5, 1, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_beta_pdf(0, 0.5, 1, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_binomial_pmf(0, 1, NAN, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_poisson_pmf(0, NAN, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_stats_poisson_pmf(0, 1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_sample_size_boundary(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_real_t single_value[] = {1};
    lmmc_real_t single_scaled_value[] = {2};

    assert_true(lmmc_std_stats_var(single_value, 1, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_std(single_value, 1, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_cov(single_value, single_scaled_value, 1, &out) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_stats_corr(single_value, single_scaled_value, 1, &out) ==
                LMMC_STATUS_INVALID_ARGUMENT);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_empty_inputs_quantile_bounds),
        cmocka_unit_test(test_sample_statistics),
        cmocka_unit_test(test_neighboring_double_correlation),
        cmocka_unit_test(test_invalid_arguments),
        cmocka_unit_test(test_nonfinite_inputs),
        cmocka_unit_test(test_zero_variance),
        cmocka_unit_test(test_normal_distribution),
        cmocka_unit_test(test_student_t_distribution),
        cmocka_unit_test(test_chi_squared_distribution),
        cmocka_unit_test(test_fisher_f_distribution),
        cmocka_unit_test(test_gamma_distribution),
        cmocka_unit_test(test_beta_distribution),
        cmocka_unit_test(test_beta_large_shape_normalization),
        cmocka_unit_test(test_discrete_distribution),
        cmocka_unit_test(test_distribution_failures),
        cmocka_unit_test(test_sample_size_boundary),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
