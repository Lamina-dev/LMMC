/**
 * @file test_stats_boundaries.c
 * @brief 统计接口的边界测试。
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
    lmmc_vec_t one;
    lmmc_vec_t const_x;
    lmmc_vec_t lin_y;
    lmmc_vec_t x;
    lmmc_vec_t short_y;
    lmmc_vec_t nan_vec;
    lmmc_mat_t one_row;
    lmmc_mat_t one_row_cov;
    lmmc_mat_t bad_cov_shape;
    lmmc_mat_t nan_mat;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->one);
    lmmc_vec_destroy(&fixture->const_x);
    lmmc_vec_destroy(&fixture->lin_y);
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->short_y);
    lmmc_vec_destroy(&fixture->nan_vec);
    lmmc_mat_destroy(&fixture->one_row);
    lmmc_mat_destroy(&fixture->one_row_cov);
    lmmc_mat_destroy(&fixture->bad_cov_shape);
    lmmc_mat_destroy(&fixture->nan_mat);
    free(fixture);
    *state = NULL;
    return 0;
}

static void test_single_observation(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    double var, std;
    st = lmmc_vec_create(1, &fixture->one);
    assert_false(st != LMMC_STATUS_OK);
    LMMC_REAL_SET_D(&fixture->one.data[0], 42.0);

    st = lmmc_vec_variance_sample(&fixture->one, &var);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_vec_variance_population(&fixture->one, &var);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(var, 0.0, 1e-12));

    st = lmmc_vec_stddev_sample(&fixture->one, &std);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    st = lmmc_vec_stddev_population(&fixture->one, &std);
    assert_false(st != LMMC_STATUS_OK || std != 0.0);
}
static void test_constant_correlation(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    double corr;
    st = lmmc_vec_create(3, &fixture->const_x);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_vec_create(3, &fixture->lin_y);
    assert_false(st != LMMC_STATUS_OK);
    LMMC_REAL_SET_D(&fixture->const_x.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->const_x.data[1], 1.0);
    LMMC_REAL_SET_D(&fixture->const_x.data[2], 1.0);
    LMMC_REAL_SET_D(&fixture->lin_y.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->lin_y.data[1], 2.0);
    LMMC_REAL_SET_D(&fixture->lin_y.data[2], 3.0);

    st = lmmc_vec_correlation_sample(&fixture->const_x, &fixture->lin_y, &corr);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);
}
static void test_vector_dimensions(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    double cov;
    st = lmmc_vec_create(4, &fixture->x);
    assert_false(st != LMMC_STATUS_OK);
    LMMC_REAL_SET_D(&fixture->x.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->x.data[1], 2.0);
    LMMC_REAL_SET_D(&fixture->x.data[2], 3.0);
    LMMC_REAL_SET_D(&fixture->x.data[3], 4.0);
    st = lmmc_vec_create(3, &fixture->short_y);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_vec_covariance_population(&fixture->x, &fixture->short_y, &cov);
    assert_false(st != LMMC_STATUS_DIMENSION_MISMATCH);
}
static void test_nonfinite_vector(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    double mean, std;
    st = lmmc_vec_create(2, &fixture->nan_vec);
    assert_false(st != LMMC_STATUS_OK);
    LMMC_REAL_SET_D(&fixture->nan_vec.data[0], 1.0);
    fixture->nan_vec.data[1] = NAN;

    st = lmmc_vec_mean(&fixture->nan_vec, &mean);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);
    st = lmmc_vec_stddev_population(&fixture->nan_vec, &std);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);
    st = lmmc_vec_stddev_sample(&fixture->nan_vec, &std);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);
}
static void test_matrix_sample_size(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    st = lmmc_mat_create(1, 2, &fixture->one_row);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_mat_create(2, 2, &fixture->one_row_cov);
    assert_false(st != LMMC_STATUS_OK);
    LMMC_REAL_SET_D(&fixture->one_row.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->one_row.data[1], 2.0);

    st = lmmc_mat_covariance_sample(&fixture->one_row, &fixture->one_row_cov);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}
static void test_matrix_dimensions(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    lmmc_real_t input[] = {1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0};
    lmmc_mat_t data = {4, 2, 2, input, 0};
    st = lmmc_mat_create(3, 3, &fixture->bad_cov_shape);
    assert_false(st != LMMC_STATUS_OK);

    st = lmmc_mat_covariance_population(&data, &fixture->bad_cov_shape);
    assert_false(st != LMMC_STATUS_DIMENSION_MISMATCH);
}
static void test_nonfinite_matrix(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    lmmc_real_t mean_data[2];
    lmmc_vec_t means = {2, mean_data, 0};
    st = lmmc_mat_create(2, 2, &fixture->nan_mat);
    assert_false(st != LMMC_STATUS_OK);
    LMMC_REAL_SET_D(&fixture->nan_mat.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->nan_mat.data[1], 2.0);
    LMMC_REAL_SET_D(&fixture->nan_mat.data[2], 3.0);
    fixture->nan_mat.data[3] = NAN;

    st = lmmc_mat_column_mean(&fixture->nan_mat, &means);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_single_observation, setup, teardown),
        cmocka_unit_test_setup_teardown(test_constant_correlation, setup, teardown),
        cmocka_unit_test_setup_teardown(test_vector_dimensions, setup, teardown),
        cmocka_unit_test_setup_teardown(test_nonfinite_vector, setup, teardown),
        cmocka_unit_test_setup_teardown(test_matrix_sample_size, setup, teardown),
        cmocka_unit_test_setup_teardown(test_matrix_dimensions, setup, teardown),
        cmocka_unit_test_setup_teardown(test_nonfinite_matrix, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
