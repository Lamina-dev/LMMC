/**
 * @file test_svd_pinv_finite.c
 * @brief 验证伪逆结果有限性、错误优先级及存储重叠时的输出原子性。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/dense.h"
#include "lmmc/eigen.h"

static void scalar_failure(lmmc_real_t input, lmmc_real_t tolerance) {
    lmmc_real_t output[] = {37.0, -91.0};
    const lmmc_real_t sentinel[] = {37.0, -91.0};
    lmmc_mat_t a, pinv;
    assert_int_equal(lmmc_mat_wrap(1, 1, 1, &input, &a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_wrap(1, 1, 2, output, &pinv), LMMC_STATUS_OK);
    assert_int_equal(lmmc_pinv(&a, tolerance, &pinv), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(memcmp(output, sentinel, sizeof(output)) == 0);
}

static void test_subnormal_reciprocal(void **state) {
    (void)state;
    scalar_failure(1e-309, 0.0);
}

static void test_nonfinite_inputs(void **state) {
    (void)state;
    const lmmc_real_t values[] = {NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        scalar_failure(values[i], 0.0);
    }
}

static void test_nonfinite_tolerances(void **state) {
    (void)state;
    const lmmc_real_t values[] = {NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        scalar_failure(2.0, values[i]);
    }
}

static void test_later_reciprocal_failure(void **state) {
    (void)state;
    lmmc_real_t input[] = {1e-308, 0.0, 0.0, 1e-309};
    lmmc_real_t output[] = {11, 12, 13, 14, 15, 16};
    const lmmc_real_t sentinel[] = {11, 12, 13, 14, 15, 16};
    lmmc_mat_t a, pinv;
    assert_int_equal(lmmc_mat_wrap(2, 2, 2, input, &a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_wrap(2, 2, 3, output, &pinv), LMMC_STATUS_OK);
    assert_int_equal(lmmc_pinv(&a, 0.0, &pinv), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(memcmp(output, sentinel, sizeof(output)) == 0);
}

static void test_failed_alias_unchanged(void **state) {
    (void)state;
    lmmc_real_t input[] = {1e-308, 0.0, 0.0, 1e-309};
    lmmc_real_t sentinel[4];
    memcpy(sentinel, input, sizeof(input));
    lmmc_mat_t a;
    assert_int_equal(lmmc_mat_wrap(2, 2, 2, input, &a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_pinv(&a, 0.0, &a), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(memcmp(input, sentinel, sizeof(input)) == 0);
}

static void check_alias_result(size_t offset) {
    lmmc_real_t storage[] = {1, 2, 91, 3, 5, 92, 93};
    const lmmc_real_t expected[] = {-5, 2, 3, -1};
    lmmc_mat_t a, pinv;
    assert_int_equal(lmmc_mat_wrap(2, 2, 3, storage, &a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_wrap(2, 2, 3, storage + offset, &pinv), LMMC_STATUS_OK);
    assert_int_equal(lmmc_pinv(&a, 0.0, &pinv), LMMC_STATUS_OK);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            lmmc_real_t value = storage[offset + 3 * i + j];
            assert_true(isfinite(value) && fabs(value - expected[2 * i + j]) <= 1e-10);
        }
    }
    assert_true(storage[6] == 93);
    if (offset == 0) {
        assert_true(storage[2] == 91 && storage[5] == 92);
    } else {
        assert_true(storage[0] == 1 && storage[3] == 3);
    }
}

static void test_in_place_stride(void **state) {
    (void)state;
    check_alias_result(0);
}

static void test_partial_overlap(void **state) {
    (void)state;
    check_alias_result(1);
}

static void test_error_precedence(void **state) {
    (void)state;
    lmmc_real_t input = NAN;
    lmmc_real_t output[] = {71, 72};
    lmmc_mat_t a, wrong_shape;
    assert_int_equal(lmmc_mat_wrap(1, 1, 1, &input, &a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_wrap(1, 2, 2, output, &wrong_shape), LMMC_STATUS_OK);
    assert_int_equal(lmmc_pinv(NULL, NAN, &wrong_shape), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_pinv(&a, NAN, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_pinv(&a, NAN, &wrong_shape), LMMC_STATUS_DIMENSION_MISMATCH);
    assert_true(output[0] == 71 && output[1] == 72);
}

static void test_threshold_semantics(void **state) {
    (void)state;
    lmmc_real_t input[] = {2, 0, 0, 1};
    lmmc_real_t output[4];
    lmmc_mat_t a, pinv;
    assert_int_equal(lmmc_mat_wrap(2, 2, 2, input, &a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_wrap(2, 2, 2, output, &pinv), LMMC_STATUS_OK);
    assert_int_equal(lmmc_pinv(&a, -1.0, &pinv), LMMC_STATUS_OK);
    assert_true(output[0] == 0.5 && output[1] == 0 && output[2] == 0 && output[3] == 1);
    assert_int_equal(lmmc_pinv(&a, 1.0, &pinv), LMMC_STATUS_OK);
    assert_true(output[0] == 0.5 && output[1] == 0 && output[2] == 0 && output[3] == 0);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_subnormal_reciprocal),
        cmocka_unit_test(test_nonfinite_inputs),
        cmocka_unit_test(test_nonfinite_tolerances),
        cmocka_unit_test(test_later_reciprocal_failure),
        cmocka_unit_test(test_failed_alias_unchanged),
        cmocka_unit_test(test_in_place_stride),
        cmocka_unit_test(test_partial_overlap),
        cmocka_unit_test(test_error_precedence),
        cmocka_unit_test(test_threshold_semantics),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
