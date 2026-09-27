#include <stdlib.h>
/**
 * @file test_tensor.c
 * 针对 LMMC 中 tensor 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static lmmc_status_t lmmc_tensor_set_values(lmmc_tensor3_t *tensor, const double *values) {
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    size_t idx = 0;
    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                lmmc_status_t st = lmmc_tensor3_set(tensor, i, j, k, values[idx++]);
                if (st != LMMC_STATUS_OK) {
                    return st;
                }
            }
        }
    }
    return LMMC_STATUS_OK;
}

static void lmmc_tensor_expect_values(const lmmc_tensor3_t *tensor, const double *expected, double eps) {
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    size_t idx = 0;
    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                double v = 0.0;
                lmmc_status_t st = lmmc_tensor3_get(tensor, i, j, k, &v);
                assert_false(st != LMMC_STATUS_OK);
                assert_true(lmmc_test_nearly_equal(v, expected[idx++], eps));
            }
        }
    }
}

static void lmmc_mat_expect_values(const lmmc_mat_t *mat, const double *expected, double eps) {
    size_t i = 0;
    size_t j = 0;
    size_t idx = 0;
    for (i = 0; i < mat->rows; ++i) {
        for (j = 0; j < mat->cols; ++j) {
            double v = mat->data[i * mat->stride + j];
            assert_true(lmmc_test_nearly_equal(v, expected[idx++], eps));
        }
    }
}

typedef struct {
    lmmc_tensor3_t t;
    lmmc_tensor3_t wrapped;
    lmmc_tensor3_t a;
    lmmc_tensor3_t b;
    lmmc_tensor3_t out;
    lmmc_tensor3_t mismatch;
    lmmc_tensor3_t non_finite;
    lmmc_tensor3_t b_zero;
    lmmc_tensor3_t reshaped;
    lmmc_tensor3_t sliced;
    lmmc_tensor3_t non_contig;
    lmmc_mat_t axis0;
    lmmc_mat_t axis1;
    lmmc_mat_t axis2;
    lmmc_mat_t axis_bad_shape;
    lmmc_mat_t axis_non_finite;
    double v;
    double n;
    double sum_v;
    double max_v;
    double min_v;
    double raw[8];
    double raw_non_contig[16];
} test_fixture_t;

static const double a_vals[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
static const double b_vals[8] = {2.0, 4.0, 1.0, -2.0, 0.5, 2.0, 4.0, 8.0};
static const double add_expected[8] = {3.0, 6.0, 4.0, 2.0, 5.5, 8.0, 11.0, 16.0};
static const double sub_expected[8] = {-1.0, -2.0, 2.0, 6.0, 4.5, 4.0, 3.0, 0.0};
static const double mul_expected[8] = {2.0, 8.0, 3.0, -8.0, 2.5, 12.0, 28.0, 64.0};
static const double div_expected[8] = {0.5, 0.5, 3.0, -2.0, 10.0, 3.0, 1.75, 1.0};
static const double scale_expected[8] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
static const double axis0_expected[4] = {6.0, 8.0, 10.0, 12.0};
static const double axis1_expected[4] = {4.0, 6.0, 12.0, 14.0};
static const double axis2_expected[4] = {3.0, 7.0, 11.0, 15.0};

static void test_tensor_storage_access(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_status_t st;

    st = lmmc_tensor3_create(0, 2, 2, &fixture->t);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_create(2, 2, 2, &fixture->t);
    assert_false(st != LMMC_STATUS_OK);

    st = lmmc_tensor3_fill(&fixture->t, 1.0);
    assert_false(st != LMMC_STATUS_OK);

    st = lmmc_tensor3_set(&fixture->t, 1, 0, 1, 3.0);
    assert_false(st != LMMC_STATUS_OK);

    st = lmmc_tensor3_set(&fixture->t, 2, 0, 0, 1.0);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_get(&fixture->t, 1, 0, 1, &fixture->v);
    assert_false(st != LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(fixture->v, 3.0, 1e-12));

    st = lmmc_tensor3_get(&fixture->t, 0, 0, 0, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_wrap(2, 2, 2, 0, 2, 1, fixture->raw, &fixture->wrapped);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_norm_fro(&fixture->t, &fixture->n);
    assert_false(st != LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(fixture->n, 4.0, 1e-12));
}

static lmmc_status_t create_tensor_arithmetic_operands(test_fixture_t *fixture) {
    lmmc_status_t st;

    st = lmmc_tensor3_create(2, 2, 2, &fixture->a);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_tensor3_create(2, 2, 2, &fixture->b);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_tensor3_create(2, 2, 2, &fixture->out);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_tensor3_create(2, 2, 1, &fixture->mismatch);
}

static lmmc_status_t create_tensor_axis_outputs(test_fixture_t *fixture) {
    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, &fixture->axis0);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_mat_create(2, 2, &fixture->axis1);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_mat_create(2, 2, &fixture->axis2);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_mat_create(2, 1, &fixture->axis_bad_shape);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_mat_create(1, 1, &fixture->axis_non_finite);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    st = lmmc_tensor_set_values(&fixture->a, a_vals);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_tensor_set_values(&fixture->b, b_vals);
}

static void test_tensor_arithmetic(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_status_t st;

    st = lmmc_tensor3_add(&fixture->a, &fixture->b, &fixture->out);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_tensor_expect_values(&fixture->out, add_expected, 1e-12);

    st = lmmc_tensor3_sub(&fixture->a, &fixture->b, &fixture->out);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_tensor_expect_values(&fixture->out, sub_expected, 1e-12);

    st = lmmc_tensor3_mul(&fixture->a, &fixture->b, &fixture->out);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_tensor_expect_values(&fixture->out, mul_expected, 1e-12);

    st = lmmc_tensor3_div(&fixture->a, &fixture->b, &fixture->out);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_tensor_expect_values(&fixture->out, div_expected, 1e-12);

    st = lmmc_tensor3_scale(&fixture->a, 0.5, &fixture->out);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_tensor_expect_values(&fixture->out, scale_expected, 1e-12);
}

static void test_tensor_axis_sums(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_status_t st;

    st = lmmc_tensor3_sum_axis(&fixture->a, 0, &fixture->axis0);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_mat_expect_values(&fixture->axis0, axis0_expected, 1e-12);

    st = lmmc_tensor3_sum_axis(&fixture->a, 1, &fixture->axis1);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_mat_expect_values(&fixture->axis1, axis1_expected, 1e-12);

    st = lmmc_tensor3_sum_axis(&fixture->a, 2, &fixture->axis2);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_mat_expect_values(&fixture->axis2, axis2_expected, 1e-12);
}

static void test_tensor_reshape_view_alias(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_status_t st;

    st = lmmc_tensor3_reshape_view(&fixture->a, 1, 4, 2, &fixture->reshaped);
    assert_false(st != LMMC_STATUS_OK || fixture->reshaped.owns_data != 0);
    st = lmmc_tensor3_set(&fixture->reshaped, 0, 3, 1, 123.0);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_get(&fixture->a, 1, 1, 1, &fixture->v);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(fixture->v, 123.0, 1e-12));
}

static void test_tensor_slice_view_alias(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_status_t st;

    st = lmmc_tensor3_slice_view(&fixture->a, 0, 2, 0, 1, 0, 2, &fixture->sliced);
    assert_false(st != LMMC_STATUS_OK || fixture->sliced.owns_data != 0);
    st = lmmc_tensor3_get(&fixture->sliced, 1, 0, 1, &fixture->v);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(fixture->v, 6.0, 1e-12));
    st = lmmc_tensor3_set(&fixture->sliced, 0, 0, 0, 77.0);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_get(&fixture->a, 0, 0, 0, &fixture->v);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(fixture->v, 77.0, 1e-12));
}

static void test_tensor_reductions(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_status_t st;

    st = lmmc_tensor3_set(&fixture->a, 1, 1, 1, 8.0);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_set(&fixture->a, 0, 0, 0, 1.0);
    assert_false(st != LMMC_STATUS_OK);

    st = lmmc_tensor3_sum(&fixture->a, &fixture->sum_v);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(fixture->sum_v, 36.0, 1e-12));

    st = lmmc_tensor3_max(&fixture->a, &fixture->max_v);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(fixture->max_v, 8.0, 1e-12));

    st = lmmc_tensor3_min(&fixture->a, &fixture->min_v);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(fixture->min_v, 1.0, 1e-12));
}

static void test_tensor_shape_and_null_errors(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_status_t st;

    st = lmmc_tensor3_add(&fixture->a, &fixture->mismatch, &fixture->out);
    assert_false(st != LMMC_STATUS_DIMENSION_MISMATCH);

    st = lmmc_tensor3_scale(&fixture->a, 1.0, &fixture->mismatch);
    assert_false(st != LMMC_STATUS_DIMENSION_MISMATCH);

    st = lmmc_tensor3_add(&fixture->a, &fixture->b, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_sum(&fixture->a, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_sum_axis(&fixture->a, 3, &fixture->axis0);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_sum_axis(&fixture->a, 0, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_tensor_view_errors(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_status_t st;

    st = lmmc_tensor3_reshape_view(&fixture->a, 3, 3, 1, &fixture->reshaped);
    assert_false(st != LMMC_STATUS_DIMENSION_MISMATCH);

    st = lmmc_tensor3_reshape_view(&fixture->a, 2, 2, 2, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_wrap(2, 2, 2, 8, 4, 2, fixture->raw_non_contig, &fixture->non_contig);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_reshape_view(&fixture->non_contig, 1, 4, 2, &fixture->reshaped);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_slice_view(&fixture->a, 1, 1, 0, 1, 0, 1, &fixture->sliced);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_slice_view(&fixture->a, 0, 3, 0, 1, 0, 1, &fixture->sliced);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_slice_view(&fixture->a, 0, 1, 0, 1, 0, 1, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_tensor_arithmetic_errors(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_status_t st;

    st = lmmc_tensor3_sum_axis(&fixture->a, 0, &fixture->axis_bad_shape);
    assert_false(st != LMMC_STATUS_DIMENSION_MISMATCH);

    st = lmmc_tensor3_scale(&fixture->a, INFINITY, &fixture->out);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_create(2, 2, 2, &fixture->b_zero);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor_set_values(&fixture->b_zero, b_vals);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_set(&fixture->b_zero, 0, 0, 0, 0.0);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_div(&fixture->a, &fixture->b_zero, &fixture->out);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_tensor_nonfinite_reductions(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_status_t st;

    st = lmmc_tensor3_create(1, 1, 1, &fixture->non_finite);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_set(&fixture->non_finite, 0, 0, 0, NAN);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_sum(&fixture->non_finite, &fixture->sum_v);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);

    st = lmmc_tensor3_sum_axis(&fixture->non_finite, 0, &fixture->axis_non_finite);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);
}

static int teardown(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_mat_destroy(&fixture->axis_non_finite);
    lmmc_mat_destroy(&fixture->axis_bad_shape);
    lmmc_mat_destroy(&fixture->axis2);
    lmmc_mat_destroy(&fixture->axis1);
    lmmc_mat_destroy(&fixture->axis0);
    lmmc_tensor3_destroy(&fixture->non_contig);
    lmmc_tensor3_destroy(&fixture->sliced);
    lmmc_tensor3_destroy(&fixture->reshaped);
    lmmc_tensor3_destroy(&fixture->t);
    lmmc_tensor3_destroy(&fixture->wrapped);
    lmmc_tensor3_destroy(&fixture->a);
    lmmc_tensor3_destroy(&fixture->b);
    lmmc_tensor3_destroy(&fixture->out);
    lmmc_tensor3_destroy(&fixture->mismatch);
    lmmc_tensor3_destroy(&fixture->non_finite);
    lmmc_tensor3_destroy(&fixture->b_zero);
    free(fixture);
    *state = NULL;
    return 0;
}

static int setup(void **state) {
    test_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    lmmc_status_t st = create_tensor_arithmetic_operands(fixture);
    if (st == LMMC_STATUS_OK) {
        st = create_tensor_axis_outputs(fixture);
    }
    if (st != LMMC_STATUS_OK) {
        teardown(state);
        return st;
    }
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_tensor_storage_access, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_arithmetic, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_axis_sums, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_reshape_view_alias, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_slice_view_alias, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_reductions, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_shape_and_null_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_view_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_arithmetic_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_nonfinite_reductions, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
