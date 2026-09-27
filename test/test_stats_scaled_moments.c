/**
 * @file test_stats_scaled_moments.c
 * @brief 缩放统计矩与 Pearson 相关系数测试。
 */
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

struct test_fixture {
    lmmc_vec_t empty;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->empty);
    free(fixture);
    *state = NULL;
    return 0;
}

static void test_adjacent_stddev(void **state) {
    (void)state;
    lmmc_status_t st;

    double std;
    const double base = 0x1p600;
    const double adjacent = nextafter(base, INFINITY);
    const double spacing = adjacent - base;
    lmmc_real_t nearby_data[] = {base, adjacent};
    lmmc_vec_t nearby = {2, nearby_data, 0};

    st = lmmc_vec_stddev_population(&nearby, &std);
    assert_false(st != LMMC_STATUS_OK || !isfinite(std) || !(fabs(std / (spacing * 0.5) - 1.0) <= 1e-12));
    st = lmmc_vec_stddev_sample(&nearby, &std);
    assert_false(st != LMMC_STATUS_OK || !isfinite(std) || !(fabs(std / (spacing / sqrt(2.0)) - 1.0) <= 1e-12));
}
static void test_extreme_vector(void **state) {
    (void)state;
    lmmc_status_t st;

    double mean, std, corr;
    lmmc_real_t extreme_x_data[] = {-DBL_MAX, DBL_MAX};
    lmmc_real_t extreme_y_data[] = {-DBL_MAX, DBL_MAX};
    lmmc_vec_t extreme_x = {2, extreme_x_data, 0};
    lmmc_vec_t extreme_y = {2, extreme_y_data, 0};
    mean = 17.0;
    st = lmmc_vec_mean(&extreme_x, &mean);
    assert_false(st != LMMC_STATUS_OK || mean != 0.0);
    st = lmmc_vec_stddev_population(&extreme_x, &std);
    assert_false(st != LMMC_STATUS_OK || !isfinite(std) || !(fabs(std / DBL_MAX - 1.0) <= 1e-12));
    st = lmmc_vec_stddev_sample(&extreme_x, &std);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);

    corr = 17.0;
    st = lmmc_vec_correlation_population(
        &extreme_x, &extreme_y, &corr);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(corr, 1.0, 1e-12));
}
static void test_scaled_stddev(void **state) {
    (void)state;
    lmmc_status_t st;

    double std;
    const int exponents[] = {-600, 600};
    for (size_t i = 0; i < sizeof(exponents) / sizeof(exponents[0]); ++i) {
        const double scale = scalbn(1.0, exponents[i]);
        lmmc_real_t scaled_data[] = {scale, -scale};
        lmmc_vec_t scaled = {2, scaled_data, 0};

        st = lmmc_vec_stddev_population(&scaled, &std);
        assert_false(st != LMMC_STATUS_OK || !isfinite(std) || !(fabs(std / scale - 1.0) <= 1e-12));
        st = lmmc_vec_stddev_sample(&scaled, &std);
        assert_false(st != LMMC_STATUS_OK || !isfinite(std) || !(fabs(std / (sqrt(2.0) * scale) - 1.0) <= 1e-12));
    }
}
typedef enum {
    SCALED_VARIANCE_POPULATION,
    SCALED_VARIANCE_SAMPLE,
    SCALED_COVARIANCE_POPULATION
} scaled_moment_operation_t;

static lmmc_status_t run_scaled_moment(
    scaled_moment_operation_t operation, const lmmc_vec_t* values,
    lmmc_real_t* out)
{
    switch (operation) {
    case SCALED_VARIANCE_POPULATION:
        return lmmc_vec_variance_population(values, out);
    case SCALED_VARIANCE_SAMPLE:
        return lmmc_vec_variance_sample(values, out);
    case SCALED_COVARIANCE_POPULATION:
        return lmmc_vec_covariance_population(values, values, out);
    }
    return LMMC_STATUS_INVALID_ARGUMENT;
}

static void test_scaled_moment_statuses(void **state) {
    (void)state;
    static const scaled_moment_operation_t operations[] = {
        SCALED_VARIANCE_POPULATION,
        SCALED_VARIANCE_SAMPLE,
        SCALED_COVARIANCE_POPULATION
    };
    static const int exponents[] = {-600, 600};

    for (size_t operation = 0;
         operation < sizeof(operations) / sizeof(operations[0]);
         ++operation) {
        for (size_t exponent = 0;
             exponent < sizeof(exponents) / sizeof(exponents[0]);
             ++exponent) {
            const double scale = scalbn(1.0, exponents[exponent]);
            lmmc_real_t data[] = {scale, -scale};
            const lmmc_vec_t values = {2, data, 0};
            lmmc_real_t result = 17.0;
            const lmmc_status_t status =
                run_scaled_moment(operations[operation], &values, &result);
            if (exponents[exponent] < 0) {
                assert_int_equal(status, LMMC_STATUS_OK);
                assert_true(result == 0.0);
            } else {
                assert_int_equal(status, LMMC_STATUS_NUMERICAL_FAILURE);
                assert_true(result == 17.0);
            }
        }
    }
}

static void check_correlation(lmmc_real_t *x_data, lmmc_real_t *y_data,
                              size_t count, double expected) {
    lmmc_vec_t x = {count, x_data, 0};
    lmmc_vec_t y = {count, y_data, 0};
    double value = 17.0;
    assert_true(lmmc_vec_correlation_population(&x, &y, &value) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(value, expected, 8.0 * DBL_EPSILON));
    assert_true(lmmc_vec_correlation_sample(&x, &y, &value) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(value, expected, 8.0 * DBL_EPSILON));
    assert_true(lmmc_vec_correlation_population(&y, &x, &value) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(value, expected, 8.0 * DBL_EPSILON));
    assert_true(lmmc_vec_correlation_sample(&y, &x, &value) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(value, expected, 8.0 * DBL_EPSILON));
}

static void test_correlation_neighboring_values(void **state) {
    (void)state;
    lmmc_real_t x[] = {1.0, nextafter(1.0, 0.0)};
    lmmc_real_t y[] = {0.0, 1.0};
    check_correlation(x, y, 2, -1.0);
    x[1] = nextafter(1.0, 2.0);
    y[0] = 2.0;
    check_correlation(x, y, 2, -1.0);
    x[0] = x[1];
    x[1] = 1.0;
    y[0] = 1.0;
    y[1] = 2.0;
    check_correlation(x, y, 2, -1.0);
    {
        lmmc_real_t near_x[] = {1.0, 1.0 + DBL_EPSILON, 1.0 + 2.0 * DBL_EPSILON};
        lmmc_real_t near_y[] = {0.0, 1.0, 1.0};
        check_correlation(near_x, near_y, 3, sqrt(3.0) / 2.0);
    }
}

static void test_correlation_scale_invariance(void **state) {
    (void)state;
    const int exponents[] = {-600, 0, 600};
    for (size_t i = 0; i < sizeof(exponents) / sizeof(exponents[0]); ++i) {
        for (size_t j = 0; j < sizeof(exponents) / sizeof(exponents[0]); ++j) {
            lmmc_real_t x[3], y[3];
            for (size_t k = 0; k < 3; ++k) {
                x[k] = scalbn((double)k - 1.0 + 0x1p40, exponents[i]);
                y[k] = scalbn((k == 0 ? 0.0 : 1.0) - 0x1p30, exponents[j]);
            }
            check_correlation(x, y, 3, sqrt(3.0) / 2.0);
            for (size_t k = 0; k < 3; ++k) {
                x[k] = -x[k];
            }
            check_correlation(x, y, 3, -sqrt(3.0) / 2.0);
        }
    }
    {
        lmmc_real_t x[] = {-DBL_MAX, DBL_MAX};
        lmmc_real_t y[] = {DBL_MAX, -DBL_MAX};
        check_correlation(x, y, 2, -1.0);
        x[0] = y[1] = 0.0;
        x[1] = y[0] = nextafter(0.0, 1.0);
        check_correlation(x, y, 2, -1.0);
    }
    {
        lmmc_real_t x[] = {-1.0, 0.0, 1.0};
        lmmc_real_t y[] = {1.0, -2.0, 1.0};
        check_correlation(x, y, 3, 0.0);
    }
}

static void test_correlation_failures_preserve_output(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t x_data[] = {1.0, 2.0};
    lmmc_real_t y_data[] = {3.0, 3.0};
    lmmc_vec_t x = {2, x_data, 0};
    lmmc_vec_t y = {2, y_data, 0};

    typedef lmmc_status_t (*correlation_fn)(const lmmc_vec_t *, const lmmc_vec_t *, lmmc_real_t *);
    const correlation_fn functions[] = {
        lmmc_vec_correlation_population, lmmc_vec_correlation_sample};
    for (size_t i = 0; i < sizeof(functions) / sizeof(functions[0]); ++i) {
        const correlation_fn correlation = functions[i];
        double value = 17.0;
        y.size = 2;
        y_data[1] = 3.0;
        assert_true(correlation(&x, &y, &value) == LMMC_STATUS_NUMERICAL_FAILURE);
        assert_true(value == 17.0);
        y_data[1] = NAN;
        assert_true(correlation(&x, &y, &value) == LMMC_STATUS_NUMERICAL_FAILURE);
        assert_true(value == 17.0);
        y_data[1] = INFINITY;
        assert_true(correlation(&y, &x, &value) == LMMC_STATUS_NUMERICAL_FAILURE);
        assert_true(value == 17.0);
        assert_true(correlation(&fixture->empty, &x, &value) == LMMC_STATUS_INVALID_ARGUMENT);
        assert_true(correlation(NULL, &x, &value) == LMMC_STATUS_INVALID_ARGUMENT);
        assert_true(value == 17.0);
        assert_true(correlation(&x, &x, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
        y.size = 1;
        assert_true(correlation(&x, &y, &value) == LMMC_STATUS_DIMENSION_MISMATCH);
        assert_true(correlation(&y, &y, &value) ==
                    (i == 0 ? LMMC_STATUS_NUMERICAL_FAILURE : LMMC_STATUS_INVALID_ARGUMENT));
        assert_true(value == 17.0);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_adjacent_stddev, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_vector, setup, teardown),
        cmocka_unit_test_setup_teardown(test_scaled_stddev, setup, teardown),
        cmocka_unit_test_setup_teardown(test_scaled_moment_statuses, setup, teardown),
        cmocka_unit_test_setup_teardown(test_correlation_neighboring_values, setup, teardown),
        cmocka_unit_test_setup_teardown(test_correlation_scale_invariance, setup, teardown),
        cmocka_unit_test_setup_teardown(test_correlation_failures_preserve_output, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
