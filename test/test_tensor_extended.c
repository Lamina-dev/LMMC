/**
 * @file test_tensor_extended.c
 * 针对 LMMC 中 tensor extended 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static int tensor_set_values(lmmc_tensor_t* tensor, const double* values) {
    size_t idx = 0;
    for (size_t i = 0; i < tensor->dim0; ++i) {
        for (size_t j = 0; j < tensor->dim1; ++j) {
            for (size_t k = 0; k < tensor->dim2; ++k) {
                lmmc_status_t st = lmmc_tensor_set(tensor, i, j, k, values[idx++]);
                if (st != LMMC_STATUS_OK) return 1;
            }
        }
    }
    return 0;
}

int main(void) {
    int rc = 0;
    lmmc_tensor_t a = {0};
    lmmc_tensor_t b = {0};
    lmmc_tensor_t out = {0};
    lmmc_tensor_t t1x1x1 = {0};
    lmmc_tensor_t t1_b = {0};
    lmmc_tensor_t t1_out = {0};
    lmmc_tensor_t reshaped = {0};
    lmmc_tensor_t sliced = {0};
    lmmc_tensor_t scaled = {0};
    lmmc_tensor_t zeros = {0};
    lmmc_mat_t sum_ax0 = {0};
    lmmc_mat_t sum_ax1 = {0};
    lmmc_mat_t sum_ax2 = {0};
    lmmc_status_t st;


    st = lmmc_tensor3_create(2, 3, 4, &a);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    st = lmmc_tensor3_create(2, 3, 4, &b);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    st = lmmc_tensor3_create(2, 3, 4, &out);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                double val = (double)(i * 12 + j * 4 + k + 1);
                st = lmmc_tensor_set(&a, i, j, k, val);
                if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
                st = lmmc_tensor_set(&b, i, j, k, val * 2.0);
                if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            }
        }
    }


    st = lmmc_tensor_add(&a, &b, &out);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                double va, vb, vo;
                lmmc_tensor_get(&a, i, j, k, &va);
                lmmc_tensor_get(&b, i, j, k, &vb);
                lmmc_tensor_get(&out, i, j, k, &vo);
                if (!lmmc_test_nearly_equal(vo, va + vb, 1e-12)) { rc = 1; goto done; }
            }
        }
    }


    st = lmmc_tensor_mul(&a, &b, &out);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                double va, vb, vo;
                lmmc_tensor_get(&a, i, j, k, &va);
                lmmc_tensor_get(&b, i, j, k, &vb);
                lmmc_tensor_get(&out, i, j, k, &vo);
                if (!lmmc_test_nearly_equal(vo, va * vb, 1e-12)) { rc = 1; goto done; }
            }
        }
    }


    {
        lmmc_tensor_t b_with_zero = {0};
        st = lmmc_tensor3_create(2, 3, 4, &b_with_zero);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_tensor_fill(&b_with_zero, 2.0);
        if (st != LMMC_STATUS_OK) { lmmc_tensor_destroy(&b_with_zero); rc = 1; goto done; }

        st = lmmc_tensor_set(&b_with_zero, 0, 0, 0, 0.0);
        if (st != LMMC_STATUS_OK) { lmmc_tensor_destroy(&b_with_zero); rc = 1; goto done; }
        st = lmmc_tensor_div(&a, &b_with_zero, &out);
        if (st != LMMC_STATUS_NUMERICAL_FAILURE) { lmmc_tensor_destroy(&b_with_zero); rc = 1; goto done; }
        lmmc_tensor_destroy(&b_with_zero);
    }


    st = lmmc_tensor3_create(2, 3, 4, &scaled);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


    st = lmmc_tensor_scale(&a, 0.0, &scaled);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                double vo;
                lmmc_tensor_get(&scaled, i, j, k, &vo);
                if (!lmmc_test_nearly_equal(vo, 0.0, 1e-12)) { rc = 1; goto done; }
            }
        }
    }


    st = lmmc_tensor_scale(&a, 1.0, &scaled);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                double va, vo;
                lmmc_tensor_get(&a, i, j, k, &va);
                lmmc_tensor_get(&scaled, i, j, k, &vo);
                if (!lmmc_test_nearly_equal(vo, va, 1e-12)) { rc = 1; goto done; }
            }
        }
    }


    st = lmmc_mat_create(3, 4, &sum_ax0);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    st = lmmc_mat_create(2, 4, &sum_ax1);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    st = lmmc_mat_create(2, 3, &sum_ax2);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

    st = lmmc_tensor_sum_axis(&a, 0, &sum_ax0);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

    for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 4; ++k) {
            double v0, v1;
            lmmc_tensor_get(&a, 0, j, k, &v0);
            lmmc_tensor_get(&a, 1, j, k, &v1);
            double expected = v0 + v1;
            double actual = sum_ax0.data[j * sum_ax0.stride + k];
            if (!lmmc_test_nearly_equal(actual, expected, 1e-12)) { rc = 1; goto done; }
        }
    }

    st = lmmc_tensor_sum_axis(&a, 1, &sum_ax1);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            double expected = 0.0;
            for (size_t j = 0; j < 3; ++j) {
                double v;
                lmmc_tensor_get(&a, i, j, k, &v);
                expected += v;
            }
            double actual = sum_ax1.data[i * sum_ax1.stride + k];
            if (!lmmc_test_nearly_equal(actual, expected, 1e-12)) { rc = 1; goto done; }
        }
    }

    st = lmmc_tensor_sum_axis(&a, 2, &sum_ax2);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < 4; ++k) {
                double v;
                lmmc_tensor_get(&a, i, j, k, &v);
                expected += v;
            }
            double actual = sum_ax2.data[i * sum_ax2.stride + j];
            if (!lmmc_test_nearly_equal(actual, expected, 1e-12)) { rc = 1; goto done; }
        }
    }


    st = lmmc_tensor_reshape_view(&a, 4, 6, 1, &reshaped);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

    if (reshaped.dim0 * reshaped.dim1 * reshaped.dim2 != a.dim0 * a.dim1 * a.dim2) {
        rc = 1; goto done;
    }

    if (reshaped.owns_data != 0) { rc = 1; goto done; }
    if (reshaped.data != a.data) { rc = 1; goto done; }


    {
        lmmc_tensor_t reshaped2 = {0};
        st = lmmc_tensor_reshape_view(&a, 1, 24, 1, &reshaped2);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (reshaped2.dim0 * reshaped2.dim1 * reshaped2.dim2 != 24) { rc = 1; goto done; }
        lmmc_tensor_destroy(&reshaped2);
    }


    st = lmmc_tensor_slice_view(&a, 0, 2, 1, 3, 0, 2, &sliced);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    if (sliced.dim0 != 2 || sliced.dim1 != 2 || sliced.dim2 != 2) { rc = 1; goto done; }

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                double vs, va;
                lmmc_tensor_get(&sliced, i, j, k, &vs);
                lmmc_tensor_get(&a, i, j + 1, k, &va);
                if (!lmmc_test_nearly_equal(vs, va, 1e-12)) { rc = 1; goto done; }
            }
        }
    }


    {
        lmmc_tensor_t bad_slice = {0};
        st = lmmc_tensor_slice_view(&a, 0, 5, 0, 1, 0, 1, &bad_slice);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }


    st = lmmc_tensor3_create(1, 1, 1, &t1x1x1);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    st = lmmc_tensor_set(&t1x1x1, 0, 0, 0, 42.0);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


    st = lmmc_tensor3_create(1, 1, 1, &t1_b);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    st = lmmc_tensor_set(&t1_b, 0, 0, 0, 8.0);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    st = lmmc_tensor3_create(1, 1, 1, &t1_out);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

    st = lmmc_tensor_add(&t1x1x1, &t1_b, &t1_out);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    {
        double v;
        lmmc_tensor_get(&t1_out, 0, 0, 0, &v);
        if (!lmmc_test_nearly_equal(v, 50.0, 1e-12)) { rc = 1; goto done; }
    }


    st = lmmc_tensor_mul(&t1x1x1, &t1_b, &t1_out);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    {
        double v;
        lmmc_tensor_get(&t1_out, 0, 0, 0, &v);
        if (!lmmc_test_nearly_equal(v, 336.0, 1e-12)) { rc = 1; goto done; }
    }


    st = lmmc_tensor_scale(&t1x1x1, 3.0, &t1_out);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    {
        double v;
        lmmc_tensor_get(&t1_out, 0, 0, 0, &v);
        if (!lmmc_test_nearly_equal(v, 126.0, 1e-12)) { rc = 1; goto done; }
    }


    {
        double s, mx, mn;
        st = lmmc_tensor_sum(&t1x1x1, &s);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(s, 42.0, 1e-12)) { rc = 1; goto done; }
        st = lmmc_tensor_max(&t1x1x1, &mx);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(mx, 42.0, 1e-12)) { rc = 1; goto done; }
        st = lmmc_tensor_min(&t1x1x1, &mn);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(mn, 42.0, 1e-12)) { rc = 1; goto done; }
    }


    {
        double norm;
        st = lmmc_tensor_norm_fro(&t1x1x1, &norm);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(norm, 42.0, 1e-12)) { rc = 1; goto done; }
    }


    st = lmmc_tensor3_create(2, 3, 4, &zeros);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    st = lmmc_tensor_fill(&zeros, 0.0);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
    {
        double norm;
        st = lmmc_tensor_norm_fro(&zeros, &norm);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(norm, 0.0, 1e-12)) { rc = 1; goto done; }
    }


    {
        lmmc_tensor_t ones = {0};
        st = lmmc_tensor3_create(3, 4, 5, &ones);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_tensor_fill(&ones, 1.0);
        if (st != LMMC_STATUS_OK) { lmmc_tensor_destroy(&ones); rc = 1; goto done; }
        double norm;
        st = lmmc_tensor_norm_fro(&ones, &norm);
        if (st != LMMC_STATUS_OK) { lmmc_tensor_destroy(&ones); rc = 1; goto done; }
        double expected_norm = sqrt(3.0 * 4.0 * 5.0);
        if (!lmmc_test_nearly_equal(norm, expected_norm, 1e-12)) {
            lmmc_tensor_destroy(&ones);
            rc = 1; goto done;
        }
        lmmc_tensor_destroy(&ones);
    }


    {
        double sum_val, max_val, min_val;

        double expected_sum = 0.0;
        double expected_max = -1e300;
        double expected_min = 1e300;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    double v;
                    lmmc_tensor_get(&a, i, j, k, &v);
                    expected_sum += v;
                    if (v > expected_max) expected_max = v;
                    if (v < expected_min) expected_min = v;
                }
            }
        }

        st = lmmc_tensor_sum(&a, &sum_val);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(sum_val, expected_sum, 1e-10)) { rc = 1; goto done; }

        st = lmmc_tensor_max(&a, &max_val);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(max_val, expected_max, 1e-12)) { rc = 1; goto done; }

        st = lmmc_tensor_min(&a, &min_val);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(min_val, expected_min, 1e-12)) { rc = 1; goto done; }
    }


    {
        lmmc_tensor_t bad_reshape = {0};

        st = lmmc_tensor_reshape_view(&a, 2, 3, 5, &bad_reshape);
        if (st != LMMC_STATUS_DIMENSION_MISMATCH) { rc = 1; goto done; }


        st = lmmc_tensor_reshape_view(&a, 5, 5, 1, &bad_reshape);
        if (st != LMMC_STATUS_DIMENSION_MISMATCH) { rc = 1; goto done; }


        st = lmmc_tensor_reshape_view(&a, 1, 1, 1, &bad_reshape);
        if (st != LMMC_STATUS_DIMENSION_MISMATCH) { rc = 1; goto done; }
    }

done:
    lmmc_mat_destroy(&sum_ax0);
    lmmc_mat_destroy(&sum_ax1);
    lmmc_mat_destroy(&sum_ax2);
    lmmc_tensor_destroy(&zeros);
    lmmc_tensor_destroy(&scaled);
    lmmc_tensor_destroy(&sliced);
    lmmc_tensor_destroy(&reshaped);
    lmmc_tensor_destroy(&t1_out);
    lmmc_tensor_destroy(&t1_b);
    lmmc_tensor_destroy(&t1x1x1);
    lmmc_tensor_destroy(&out);
    lmmc_tensor_destroy(&b);
    lmmc_tensor_destroy(&a);

    if (rc != 0) {
        printf("tensor_extended test failed\n");
    }
    return rc;
}
