#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static int lmmc_tensor_set_values(lmmc_tensor_t* tensor, const double* values) {
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    size_t idx = 0;
    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                lmmc_status_t st = lmmc_tensor_set(tensor, i, j, k, values[idx++]);
                if (st != LMMC_STATUS_OK) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

static int lmmc_tensor_expect_values(const lmmc_tensor_t* tensor, const double* expected, double eps) {
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    size_t idx = 0;
    for (i = 0; i < tensor->dim0; ++i) {
        for (j = 0; j < tensor->dim1; ++j) {
            for (k = 0; k < tensor->dim2; ++k) {
                double v = 0.0;
                lmmc_status_t st = lmmc_tensor_get(tensor, i, j, k, &v);
                if (st != LMMC_STATUS_OK) {
                    return 1;
                }
                if (!lmmc_test_nearly_equal(v, expected[idx++], eps)) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

static int lmmc_mat_expect_values(const lmmc_mat_t* mat, const double* expected, double eps) {
    size_t i = 0;
    size_t j = 0;
    size_t idx = 0;
    for (i = 0; i < mat->rows; ++i) {
        for (j = 0; j < mat->cols; ++j) {
            double v = mat->data[i * mat->stride + j];
            if (!lmmc_test_nearly_equal(v, expected[idx++], eps)) {
                return 1;
            }
        }
    }
    return 0;
}

int main(void) {
    lmmc_tensor_t t = {0};
    lmmc_tensor_t wrapped = {0};
    lmmc_tensor_t a = {0};
    lmmc_tensor_t b = {0};
    lmmc_tensor_t out = {0};
    lmmc_tensor_t mismatch = {0};
    lmmc_tensor_t non_finite = {0};
    lmmc_tensor_t b_zero = {0};
    lmmc_tensor_t reshaped = {0};
    lmmc_tensor_t sliced = {0};
    lmmc_tensor_t non_contig = {0};
    lmmc_mat_t axis0 = {0};
    lmmc_mat_t axis1 = {0};
    lmmc_mat_t axis2 = {0};
    lmmc_mat_t axis_bad_shape = {0};
    lmmc_mat_t axis_non_finite = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    double v = 0.0;
    double n = 0.0;
    double sum_v = 0.0;
    double max_v = 0.0;
    double min_v = 0.0;
    double raw[8] = {0.0};
    double raw_non_contig[16] = {0.0};
    const double a_vals[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const double b_vals[8] = {2.0, 4.0, 1.0, -2.0, 0.5, 2.0, 4.0, 8.0};
    const double add_expected[8] = {3.0, 6.0, 4.0, 2.0, 5.5, 8.0, 11.0, 16.0};
    const double sub_expected[8] = {-1.0, -2.0, 2.0, 6.0, 4.5, 4.0, 3.0, 0.0};
    const double mul_expected[8] = {2.0, 8.0, 3.0, -8.0, 2.5, 12.0, 28.0, 64.0};
    const double div_expected[8] = {0.5, 0.5, 3.0, -2.0, 10.0, 3.0, 1.75, 1.0};
    const double scale_expected[8] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    const double axis0_expected[4] = {6.0, 8.0, 10.0, 12.0};
    const double axis1_expected[4] = {4.0, 6.0, 12.0, 14.0};
    const double axis2_expected[4] = {3.0, 7.0, 11.0, 15.0};
    int rc = 0;

    st = lmmc_tensor3_create(0, 2, 2, &t);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor3_create(2, 2, 2, &t);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_fill(&t, 1.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_set(&t, 1, 0, 1, 3.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_set(&t, 2, 0, 0, 1.0);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_get(&t, 1, 0, 1, &v);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    if (!lmmc_test_nearly_equal(v, 3.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_get(&t, 0, 0, 0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor3_wrap(2, 2, 2, 0, 2, 1, raw, &wrapped);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_norm_fro(&t, &n);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    if (!lmmc_test_nearly_equal(n, 4.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor3_create(2, 2, 2, &a);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor3_create(2, 2, 2, &b);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor3_create(2, 2, 2, &out);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor3_create(2, 2, 1, &mismatch);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(2, 2, &axis0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &axis1);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &axis2);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 1, &axis_bad_shape);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(1, 1, &axis_non_finite);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    if (lmmc_tensor_set_values(&a, a_vals) != 0 || lmmc_tensor_set_values(&b, b_vals) != 0) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_add(&a, &b, &out);
    if (st != LMMC_STATUS_OK || lmmc_tensor_expect_values(&out, add_expected, 1e-12) != 0) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_sub(&a, &b, &out);
    if (st != LMMC_STATUS_OK || lmmc_tensor_expect_values(&out, sub_expected, 1e-12) != 0) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_mul(&a, &b, &out);
    if (st != LMMC_STATUS_OK || lmmc_tensor_expect_values(&out, mul_expected, 1e-12) != 0) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_div(&a, &b, &out);
    if (st != LMMC_STATUS_OK || lmmc_tensor_expect_values(&out, div_expected, 1e-12) != 0) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_scale(&a, 0.5, &out);
    if (st != LMMC_STATUS_OK || lmmc_tensor_expect_values(&out, scale_expected, 1e-12) != 0) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_sum_axis(&a, 0, &axis0);
    if (st != LMMC_STATUS_OK || lmmc_mat_expect_values(&axis0, axis0_expected, 1e-12) != 0) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_sum_axis(&a, 1, &axis1);
    if (st != LMMC_STATUS_OK || lmmc_mat_expect_values(&axis1, axis1_expected, 1e-12) != 0) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_sum_axis(&a, 2, &axis2);
    if (st != LMMC_STATUS_OK || lmmc_mat_expect_values(&axis2, axis2_expected, 1e-12) != 0) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_reshape_view(&a, 1, 4, 2, &reshaped);
    if (st != LMMC_STATUS_OK || reshaped.owns_data != 0) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor_set(&reshaped, 0, 3, 1, 123.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor_get(&a, 1, 1, 1, &v);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(v, 123.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_slice_view(&a, 0, 2, 0, 1, 0, 2, &sliced);
    if (st != LMMC_STATUS_OK || sliced.owns_data != 0) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor_get(&sliced, 1, 0, 1, &v);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(v, 6.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor_set(&sliced, 0, 0, 0, 77.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor_get(&a, 0, 0, 0, &v);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(v, 77.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_set(&a, 1, 1, 1, 8.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor_set(&a, 0, 0, 0, 1.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_sum(&a, &sum_v);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(sum_v, 36.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_max(&a, &max_v);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(max_v, 8.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_min(&a, &min_v);
    if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(min_v, 1.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_add(&a, &mismatch, &out);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_scale(&a, 1.0, &mismatch);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_add(&a, &b, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_sum(&a, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_sum_axis(&a, 3, &axis0);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_sum_axis(&a, 0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_reshape_view(&a, 3, 3, 1, &reshaped);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_reshape_view(&a, 2, 2, 2, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor3_wrap(2, 2, 2, 8, 4, 2, raw_non_contig, &non_contig);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor_reshape_view(&non_contig, 1, 4, 2, &reshaped);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_slice_view(&a, 1, 1, 0, 1, 0, 1, &sliced);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_slice_view(&a, 0, 3, 0, 1, 0, 1, &sliced);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_slice_view(&a, 0, 1, 0, 1, 0, 1, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_sum_axis(&a, 0, &axis_bad_shape);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_scale(&a, INFINITY, &out);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor3_create(2, 2, 2, &b_zero);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    if (lmmc_tensor_set_values(&b_zero, b_vals) != 0) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor_set(&b_zero, 0, 0, 0, 0.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor_div(&a, &b_zero, &out);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor3_create(1, 1, 1, &non_finite);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor_set(&non_finite, 0, 0, 0, NAN);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor_sum(&non_finite, &sum_v);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_sum_axis(&non_finite, 0, &axis_non_finite);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        rc = 1;
        goto cleanup;
    }

cleanup:
    lmmc_mat_destroy(&axis_non_finite);
    lmmc_mat_destroy(&axis_bad_shape);
    lmmc_mat_destroy(&axis2);
    lmmc_mat_destroy(&axis1);
    lmmc_mat_destroy(&axis0);
    lmmc_tensor_destroy(&non_contig);
    lmmc_tensor_destroy(&sliced);
    lmmc_tensor_destroy(&reshaped);
    lmmc_tensor_destroy(&t);
    lmmc_tensor_destroy(&wrapped);
    lmmc_tensor_destroy(&a);
    lmmc_tensor_destroy(&b);
    lmmc_tensor_destroy(&out);
    lmmc_tensor_destroy(&mismatch);
    lmmc_tensor_destroy(&non_finite);
    lmmc_tensor_destroy(&b_zero);

    if (rc != 0) {
        printf("tensor test failed\n");
    }
    return rc;
}
