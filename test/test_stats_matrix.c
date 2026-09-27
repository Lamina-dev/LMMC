/**
 * @file test_stats_matrix.c
 * @brief 矩阵统计量测试。
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
    lmmc_vec_t means;
    lmmc_mat_t cov_p;
    lmmc_mat_t cov_s;
    lmmc_mat_t corr_p;
    lmmc_mat_t corr_s;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->means);
    lmmc_mat_destroy(&fixture->cov_p);
    lmmc_mat_destroy(&fixture->cov_s);
    lmmc_mat_destroy(&fixture->corr_p);
    lmmc_mat_destroy(&fixture->corr_s);
    free(fixture);
    *state = NULL;
    return 0;
}

static void test_column_mean(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    lmmc_real_t input[] = {1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0};
    lmmc_mat_t data = {4, 2, 2, input, 0};
    st = lmmc_vec_create(2, &fixture->means);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_mat_column_mean(&data, &fixture->means);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(fixture->means.data[0], 2.5, 1e-12) || !lmmc_test_nearly_equal(fixture->means.data[1], 5.0, 1e-12));
}
static void test_covariance_population(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    lmmc_real_t input[] = {1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0};
    lmmc_mat_t data = {4, 2, 2, input, 0};
    st = lmmc_mat_create(2, 2, &fixture->cov_p);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_mat_covariance_population(&data, &fixture->cov_p);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(fixture->cov_p.data[0], 1.25, 1e-12) || !lmmc_test_nearly_equal(fixture->cov_p.data[1], 2.5, 1e-12) || !lmmc_test_nearly_equal(fixture->cov_p.data[2], 2.5, 1e-12) || !lmmc_test_nearly_equal(fixture->cov_p.data[3], 5.0, 1e-12));
}
static void test_covariance_sample(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    lmmc_real_t input[] = {1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0};
    lmmc_mat_t data = {4, 2, 2, input, 0};
    st = lmmc_mat_create(2, 2, &fixture->cov_s);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_mat_covariance_sample(&data, &fixture->cov_s);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(fixture->cov_s.data[0], 1.6666666666666667, 1e-12) || !lmmc_test_nearly_equal(fixture->cov_s.data[1], 3.3333333333333335, 1e-12) || !lmmc_test_nearly_equal(fixture->cov_s.data[2], 3.3333333333333335, 1e-12) || !lmmc_test_nearly_equal(fixture->cov_s.data[3], 6.666666666666667, 1e-12));
}
static void test_correlation_population(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    lmmc_real_t input[] = {1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0};
    lmmc_mat_t data = {4, 2, 2, input, 0};
    st = lmmc_mat_create(2, 2, &fixture->corr_p);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_mat_correlation_population(&data, &fixture->corr_p);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(fixture->corr_p.data[0], 1.0, 1e-12) || !lmmc_test_nearly_equal(fixture->corr_p.data[1], 1.0, 1e-12) || !lmmc_test_nearly_equal(fixture->corr_p.data[2], 1.0, 1e-12) || !lmmc_test_nearly_equal(fixture->corr_p.data[3], 1.0, 1e-12));
}
static void test_correlation_sample(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    lmmc_real_t input[] = {1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0};
    lmmc_mat_t data = {4, 2, 2, input, 0};
    st = lmmc_mat_create(2, 2, &fixture->corr_s);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_mat_correlation_sample(&data, &fixture->corr_s);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(fixture->corr_s.data[0], 1.0, 1e-12) || !lmmc_test_nearly_equal(fixture->corr_s.data[1], 1.0, 1e-12) || !lmmc_test_nearly_equal(fixture->corr_s.data[2], 1.0, 1e-12) || !lmmc_test_nearly_equal(fixture->corr_s.data[3], 1.0, 1e-12));
}
static void test_matrix_extremes(void **state) {
    (void)state;
    lmmc_status_t st;

    lmmc_real_t mean_data[2];
    lmmc_vec_t means = {2, mean_data, 0};
    lmmc_real_t extreme_data[] = {
        -DBL_MAX, -DBL_MAX,
        DBL_MAX, DBL_MAX};
    lmmc_real_t extreme_output[4] = {17.0, 17.0, 17.0, 17.0};
    lmmc_mat_t extreme_matrix = {2, 2, 2, extreme_data, 0};
    lmmc_mat_t extreme_correlation = {
        2, 2, 2, extreme_output, 0};

    st = lmmc_mat_column_mean(&extreme_matrix, &means);
    assert_false(st != LMMC_STATUS_OK || means.data[0] != 0.0 || means.data[1] != 0.0);

    st = lmmc_mat_correlation_population(
        &extreme_matrix, &extreme_correlation);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(extreme_output[0], 1.0, 1e-12) || !lmmc_test_nearly_equal(extreme_output[1], 1.0, 1e-12) || !lmmc_test_nearly_equal(extreme_output[2], 1.0, 1e-12) || !lmmc_test_nearly_equal(extreme_output[3], 1.0, 1e-12));
    st = lmmc_mat_covariance_population(&extreme_matrix, &extreme_correlation);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);
    st = lmmc_mat_covariance_sample(&extreme_matrix, &extreme_correlation);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);
}

static void check_column_vector_correlation(const lmmc_mat_t *data, double expected) {
    lmmc_real_t x_data[3], y_data[3];
    lmmc_vec_t x = {data->rows, x_data, 0};
    lmmc_vec_t y = {data->rows, y_data, 0};
    double vector_result;
    for (size_t row = 0; row < data->rows; ++row) {
        x_data[row] = data->data[data->stride * row];
        y_data[row] = data->data[data->stride * row + 1];
    }
    assert_int_equal(lmmc_vec_correlation_population(&x, &y, &vector_result), LMMC_STATUS_OK);
    assert_true(expected == vector_result);
}

static void test_correlation_strided_columns(void **state) {
    (void)state;
    typedef lmmc_status_t (*correlation_fn)(const lmmc_mat_t *, lmmc_mat_t *);
    const correlation_fn functions[] = {
        lmmc_mat_correlation_population, lmmc_mat_correlation_sample};
    for (size_t sample = 0; sample < 2; ++sample) {
        for (size_t fixture = 0; fixture < 3; ++fixture) {
            lmmc_real_t input[] = {
                1.0, 0.0, NAN, NAN,
                nextafter(1.0, 0.0), 1.0, NAN, NAN,
                0.0, 0.0, NAN, NAN};
            lmmc_real_t output[] = {17.0, 17.0, 17.0, 17.0, 17.0, 17.0};
            lmmc_mat_t data = {2, 2, 4, input, 0};
            lmmc_mat_t result = {2, 2, 3, output, 0};
            double expected = -1.0;
            if (fixture == 1) {
                input[1] = 2.0;
                input[4] = nextafter(1.0, 2.0);
            } else if (fixture == 2) {
                data.rows = 3;
                input[0] = -0x1p600;
                input[1] = 0.0;
                input[4] = 0.0;
                input[5] = 0x1p-600;
                input[8] = 0x1p600;
                input[9] = 0x1p-600;
                expected = sqrt(3.0) / 2.0;
            }
            assert_true(functions[sample](&data, &result) == LMMC_STATUS_OK);
            assert_true(output[0] == 1.0 && output[4] == 1.0);
            assert_true(lmmc_test_nearly_equal(output[1], expected, 8.0 * DBL_EPSILON));
            assert_true(output[1] == output[3]);
            assert_true(output[2] == 17.0 && output[5] == 17.0);
            check_column_vector_correlation(&data, output[1]);
            for (size_t row = 0; row < data.rows; ++row) {
                input[4 * row + 1] = 3.0;
            }
            assert_true(functions[sample](&data, &result) == LMMC_STATUS_NUMERICAL_FAILURE);
            input[1] = NAN;
            assert_true(functions[sample](&data, &result) == LMMC_STATUS_NUMERICAL_FAILURE);
            input[1] = INFINITY;
            assert_true(functions[sample](&data, &result) == LMMC_STATUS_NUMERICAL_FAILURE);
            data.rows = 1;
            assert_true(functions[sample](&data, &result) ==
                        (sample == 0 ? LMMC_STATUS_NUMERICAL_FAILURE : LMMC_STATUS_INVALID_ARGUMENT));
            data.rows = 0;
            assert_true(functions[sample](&data, &result) == LMMC_STATUS_INVALID_ARGUMENT);
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_column_mean, setup, teardown),
        cmocka_unit_test_setup_teardown(test_covariance_population, setup, teardown),
        cmocka_unit_test_setup_teardown(test_covariance_sample, setup, teardown),
        cmocka_unit_test_setup_teardown(test_correlation_population, setup, teardown),
        cmocka_unit_test_setup_teardown(test_correlation_sample, setup, teardown),
        cmocka_unit_test_setup_teardown(test_matrix_extremes, setup, teardown),
        cmocka_unit_test_setup_teardown(test_correlation_strided_columns, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
