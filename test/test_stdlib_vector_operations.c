#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"
#include "internal_test_hooks.h"
#include <float.h>

struct test_fixture {
    lmmc_vec_t cross_result;
    lmmc_std_bool_vec_t bool_vec_result;
};

static int setup(void **state) {
    *state = test_calloc(1, sizeof(struct test_fixture));
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_memory_fail_reset_for_test();
    lmmc_std_bool_vec_destroy(&fixture->bool_vec_result);
    lmmc_vec_destroy(&fixture->cross_result);
    test_free(fixture);
    return 0;
}

static void test_products_norm(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_dot(&vec_a, &vec_b, &out), LMMC_STATUS_OK);
    assert_true(close_real(out, 32));
    assert_int_equal(lmmc_std_linalg_norm(&vec_a, &out), LMMC_STATUS_OK);
    assert_true(close_real(out, sqrt(14.0)));
    assert_int_equal(lmmc_std_linalg_cross(&vec_a, &vec_b, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], -3));
    assert_true(close_real(fixture->cross_result.data[1], 6));
    assert_true(close_real(fixture->cross_result.data[2], -3));
}

static void test_addition_scaling(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_add(&vec_a, &vec_b, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 5));
    assert_true(close_real(fixture->cross_result.data[1], 7));
    assert_true(close_real(fixture->cross_result.data[2], 9));
    lmmc_vec_destroy(&fixture->cross_result);

    assert_int_equal(lmmc_std_linalg_vec_scale(&vec_a, 2, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 2));
    assert_true(close_real(fixture->cross_result.data[1], 4));
    assert_true(close_real(fixture->cross_result.data[2], 6));
    assert_true(fixture->cross_result.owns_data && fixture->cross_result.data != vec_a_values);
    fixture->cross_result.data[0] = 9;
    lmmc_vec_destroy(&fixture->cross_result);
    assert_true(vec_a_values[0] == 1);

    assert_int_equal(lmmc_std_linalg_vec_add_scalar(&vec_a, 10, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 11));
    assert_true(close_real(fixture->cross_result.data[1], 12));
    assert_true(close_real(fixture->cross_result.data[2], 13));
}

static void test_scaling_failure_outputs(void **state) {
    (void)state;
    lmmc_real_t values[] = {NAN}, sentinel[] = {123, 456};
    lmmc_vec_t vec = {1, values, 0};
    lmmc_vec_t rejected = {2, sentinel, 0}, result = rejected;
    lmmc_status_t invalid_status, allocation_status;

    lmmc_memory_fail_after_for_test(0);
    invalid_status = lmmc_std_linalg_vec_scale(&vec, 1, &rejected);
    values[0] = DBL_MAX;
    allocation_status = lmmc_std_linalg_vec_scale(&vec, 1, &result);
    lmmc_memory_fail_reset_for_test();

    assert_int_equal(invalid_status, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(allocation_status, LMMC_STATUS_ALLOCATION_FAILED);
    assert_true(rejected.size == 2);
    assert_true(rejected.data == sentinel && !rejected.owns_data);
    assert_true(result.size == 2);
    assert_true(result.data == sentinel && !result.owns_data);

    assert_int_equal(lmmc_std_linalg_vec_scale(&vec, 2, &result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(result.size == 0);
    assert_true(result.data == NULL && !result.owns_data);
    assert_true(values[0] == DBL_MAX);
    assert_true(sentinel[0] == 123 && sentinel[1] == 456);
    lmmc_vec_destroy(&result);
}

static void test_subtraction(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_sub(&vec_b, &vec_a, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 3));
    assert_true(close_real(fixture->cross_result.data[1], 3));
    assert_true(close_real(fixture->cross_result.data[2], 3));
    lmmc_vec_destroy(&fixture->cross_result);

    assert_int_equal(lmmc_std_linalg_vec_sub_scalar(&vec_a, 1, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 0));
    assert_true(close_real(fixture->cross_result.data[1], 1));
    assert_true(close_real(fixture->cross_result.data[2], 2));
    lmmc_vec_destroy(&fixture->cross_result);

    assert_int_equal(lmmc_std_linalg_scalar_sub_vec(10, &vec_a, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 9));
    assert_true(close_real(fixture->cross_result.data[1], 8));
    assert_true(close_real(fixture->cross_result.data[2], 7));
}

static void test_multiplication(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_mul(&vec_a, &vec_b, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 4));
    assert_true(close_real(fixture->cross_result.data[1], 10));
    assert_true(close_real(fixture->cross_result.data[2], 18));
    lmmc_vec_destroy(&fixture->cross_result);

    assert_int_equal(lmmc_std_linalg_vec_mul_scalar(&vec_a, 2, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 2));
    assert_true(close_real(fixture->cross_result.data[1], 4));
    assert_true(close_real(fixture->cross_result.data[2], 6));
}

static void test_division(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_div(&vec_b, &vec_a, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 4));
    assert_true(close_real(fixture->cross_result.data[1], 2.5));
    assert_true(close_real(fixture->cross_result.data[2], 2));
    lmmc_vec_destroy(&fixture->cross_result);

    assert_int_equal(lmmc_std_linalg_vec_div_scalar(&vec_b, 2, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 2));
    assert_true(close_real(fixture->cross_result.data[1], 2.5));
    assert_true(close_real(fixture->cross_result.data[2], 3));
    lmmc_vec_destroy(&fixture->cross_result);

    assert_int_equal(lmmc_std_linalg_scalar_div_vec(12, &vec_a, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 12));
    assert_true(close_real(fixture->cross_result.data[1], 6));
    assert_true(close_real(fixture->cross_result.data[2], 4));
}

static void test_powers(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_pow_values[] = {2, 3, 4};
    lmmc_real_t vec_pow_exp_values[] = {3, 2, 0.5};
    lmmc_real_t vec_negative_values[] = {-1, 2, 3};
    lmmc_real_t vec_negative_integer_exp_values[] = {3, 2, 1};
    lmmc_vec_t vec_pow = {3, vec_pow_values, 0};
    lmmc_vec_t vec_pow_exp = {3, vec_pow_exp_values, 0};
    lmmc_vec_t vec_negative = {3, vec_negative_values, 0};
    lmmc_vec_t vec_negative_integer_exp = {3, vec_negative_integer_exp_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_pow(&vec_pow, &vec_pow_exp, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 8));
    assert_true(close_real(fixture->cross_result.data[1], 9));
    assert_true(close_real(fixture->cross_result.data[2], 2));
    lmmc_vec_destroy(&fixture->cross_result);

    assert_int_equal(lmmc_std_linalg_vec_pow_scalar(&vec_pow, 2, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], 4));
    assert_true(close_real(fixture->cross_result.data[1], 9));
    assert_true(close_real(fixture->cross_result.data[2], 16));
    lmmc_vec_destroy(&fixture->cross_result);

    assert_int_equal(lmmc_std_linalg_vec_pow(&vec_negative, &vec_negative_integer_exp, &fixture->cross_result), LMMC_STATUS_OK);
    assert_true(fixture->cross_result.size == 3);
    assert_true(close_real(fixture->cross_result.data[0], -1));
    assert_true(close_real(fixture->cross_result.data[1], 4));
    assert_true(close_real(fixture->cross_result.data[2], 3));
}

static void test_comparison(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_compare_scalar(&vec_a, LMMC_STD_COMPARE_GT, 1, &fixture->bool_vec_result), LMMC_STATUS_OK);
    assert_true(fixture->bool_vec_result.size == 3);
    assert_true(fixture->bool_vec_result.data[0] == 0);
    assert_true(fixture->bool_vec_result.data[1] == 1);
    assert_true(fixture->bool_vec_result.data[2] == 1);
    lmmc_std_bool_vec_destroy(&fixture->bool_vec_result);

    assert_int_equal(lmmc_std_linalg_vec_compare(&vec_a, &vec_b, LMMC_STD_COMPARE_LT, &fixture->bool_vec_result), LMMC_STATUS_OK);
    assert_true(fixture->bool_vec_result.size == 3);
    assert_true(fixture->bool_vec_result.data[0] == 1);
    assert_true(fixture->bool_vec_result.data[1] == 1);
    assert_true(fixture->bool_vec_result.data[2] == 1);
    lmmc_std_bool_vec_destroy(NULL);
}

static void test_products_scaling_norm_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_dot(NULL, &vec_b, &out), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_dot(&vec_a, NULL, &out), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_dot(&vec_a, &vec_b, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_cross(NULL, &vec_b, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_cross(&vec_a, NULL, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_cross(&vec_a, &vec_b, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_scale(NULL, 2, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_scale(&vec_a, 2, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_norm(NULL, &out), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_norm(&vec_a, NULL), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_addition_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_add(NULL, &vec_b, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_add(&vec_a, NULL, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_add(&vec_a, &vec_b, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_add_scalar(NULL, 10, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_add_scalar(&vec_a, 10, NULL), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_subtraction_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_sub(NULL, &vec_b, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_sub(&vec_a, NULL, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_sub(&vec_a, &vec_b, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_sub_scalar(NULL, 1, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_sub_scalar(&vec_a, 1, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_scalar_sub_vec(1, NULL, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_scalar_sub_vec(1, &vec_a, NULL), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_multiplication_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_mul(NULL, &vec_b, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_mul(&vec_a, NULL, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_mul(&vec_a, &vec_b, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_mul_scalar(NULL, 2, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_mul_scalar(&vec_a, 2, NULL), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_division_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_div(NULL, &vec_b, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_div(&vec_a, NULL, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_div(&vec_a, &vec_b, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_div_scalar(NULL, 2, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_div_scalar(&vec_a, 2, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_scalar_div_vec(2, NULL, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_scalar_div_vec(2, &vec_a, NULL), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_powers_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_pow(NULL, &vec_b, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_pow(&vec_a, NULL, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_pow(&vec_a, &vec_b, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_pow_scalar(NULL, 2, &fixture->cross_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_pow_scalar(&vec_a, 2, NULL), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_comparison_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_compare(NULL, &vec_b, LMMC_STD_COMPARE_EQ, &fixture->bool_vec_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_compare(&vec_a, NULL, LMMC_STD_COMPARE_EQ, &fixture->bool_vec_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_compare(&vec_a, &vec_b, LMMC_STD_COMPARE_EQ, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_compare_scalar(NULL, LMMC_STD_COMPARE_GT, 1, &fixture->bool_vec_result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_compare_scalar(&vec_a, LMMC_STD_COMPARE_GT, 1, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_std_linalg_vec_compare_scalar(
                         &vec_a,
                         (lmmc_std_compare_op_t)99,
                         1,
                         &fixture->bool_vec_result),
                     LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_products_scaling_norm_numerical_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_real_t vec_nonfinite_values[] = {1, NAN, 3};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};
    lmmc_vec_t vec_nonfinite = {3, vec_nonfinite_values, 0};

    assert_int_equal(lmmc_std_linalg_dot(&vec_nonfinite, &vec_b, &out), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_cross(&vec_nonfinite, &vec_b, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_scale(&vec_a, NAN, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_scale(&vec_nonfinite, 2, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_norm(&vec_nonfinite, &out), LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_addition_numerical_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_real_t vec_nonfinite_values[] = {1, NAN, 3};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};
    lmmc_vec_t vec_nonfinite = {3, vec_nonfinite_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_add(&vec_nonfinite, &vec_b, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_add_scalar(&vec_a, NAN, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_add_scalar(&vec_nonfinite, 10, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_subtraction_numerical_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_real_t vec_nonfinite_values[] = {1, NAN, 3};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};
    lmmc_vec_t vec_nonfinite = {3, vec_nonfinite_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_sub(&vec_nonfinite, &vec_b, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_sub_scalar(&vec_a, NAN, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_sub_scalar(&vec_nonfinite, 1, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_scalar_sub_vec(NAN, &vec_a, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_scalar_sub_vec(1, &vec_nonfinite, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_multiplication_numerical_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_real_t vec_nonfinite_values[] = {1, NAN, 3};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};
    lmmc_vec_t vec_nonfinite = {3, vec_nonfinite_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_mul(&vec_nonfinite, &vec_b, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_mul_scalar(&vec_a, NAN, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_mul_scalar(&vec_nonfinite, 2, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_division_numerical_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_real_t vec_nonfinite_values[] = {1, NAN, 3};
    lmmc_real_t vec_zero_values[] = {1, 0, 3};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};
    lmmc_vec_t vec_nonfinite = {3, vec_nonfinite_values, 0};
    lmmc_vec_t vec_zero = {3, vec_zero_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_div(&vec_nonfinite, &vec_b, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_div(&vec_b, &vec_zero, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_div_scalar(&vec_a, NAN, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_div_scalar(&vec_a, 0, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_div_scalar(&vec_nonfinite, 2, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_scalar_div_vec(NAN, &vec_a, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_scalar_div_vec(2, &vec_nonfinite, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_scalar_div_vec(2, &vec_zero, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_powers_numerical_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_real_t vec_nonfinite_values[] = {1, NAN, 3};
    lmmc_real_t vec_zero_values[] = {1, 0, 3};
    lmmc_real_t vec_negative_values[] = {-1, 2, 3};
    lmmc_real_t vec_negative_exp_values[] = {0.5, 2, 3};
    lmmc_real_t vec_zero_negative_exp_values[] = {2, -1, 1};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};
    lmmc_vec_t vec_nonfinite = {3, vec_nonfinite_values, 0};
    lmmc_vec_t vec_zero = {3, vec_zero_values, 0};
    lmmc_vec_t vec_negative = {3, vec_negative_values, 0};
    lmmc_vec_t vec_negative_exp = {3, vec_negative_exp_values, 0};
    lmmc_vec_t vec_zero_negative_exp = {3, vec_zero_negative_exp_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_pow(&vec_nonfinite, &vec_b, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_pow(&vec_negative, &vec_negative_exp, &fixture->cross_result), LMMC_STATUS_OUT_OF_RANGE);
    assert_int_equal(lmmc_std_linalg_vec_pow(&vec_zero, &vec_zero_negative_exp, &fixture->cross_result), LMMC_STATUS_OUT_OF_RANGE);
    assert_int_equal(lmmc_std_linalg_vec_pow_scalar(&vec_a, NAN, &fixture->cross_result), LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_pow_scalar(&vec_negative, 0.5, &fixture->cross_result), LMMC_STATUS_OUT_OF_RANGE);
}

static void test_comparison_numerical_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_real_t vec_nonfinite_values[] = {1, NAN, 3};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};
    lmmc_vec_t vec_nonfinite = {3, vec_nonfinite_values, 0};

    assert_int_equal(lmmc_std_linalg_vec_compare(
                         &vec_nonfinite,
                         &vec_b,
                         LMMC_STD_COMPARE_EQ,
                         &fixture->bool_vec_result),
                     LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(lmmc_std_linalg_vec_compare_scalar(&vec_a, LMMC_STD_COMPARE_GT, NAN, &fixture->bool_vec_result), LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_short_values[] = {1, 2};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_short = {2, vec_short_values, 0};

    assert_int_equal(lmmc_std_linalg_dot(&vec_a, &vec_short, &out), LMMC_STATUS_DIMENSION_MISMATCH);
    assert_int_equal(lmmc_std_linalg_cross(&vec_a, &vec_short, &fixture->cross_result), LMMC_STATUS_DIMENSION_MISMATCH);
    assert_int_equal(lmmc_std_linalg_vec_add(&vec_a, &vec_short, &fixture->cross_result), LMMC_STATUS_DIMENSION_MISMATCH);
    assert_int_equal(lmmc_std_linalg_vec_sub(&vec_a, &vec_short, &fixture->cross_result), LMMC_STATUS_DIMENSION_MISMATCH);
    assert_int_equal(lmmc_std_linalg_vec_mul(&vec_a, &vec_short, &fixture->cross_result), LMMC_STATUS_DIMENSION_MISMATCH);
    assert_int_equal(lmmc_std_linalg_vec_div(&vec_a, &vec_short, &fixture->cross_result), LMMC_STATUS_DIMENSION_MISMATCH);
    assert_int_equal(lmmc_std_linalg_vec_pow(&vec_a, &vec_short, &fixture->cross_result), LMMC_STATUS_DIMENSION_MISMATCH);
    assert_int_equal(lmmc_std_linalg_vec_compare(&vec_a, &vec_short, LMMC_STD_COMPARE_EQ, &fixture->bool_vec_result), LMMC_STATUS_DIMENSION_MISMATCH);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_products_norm, setup, teardown),
        cmocka_unit_test_setup_teardown(test_addition_scaling, setup, teardown),
        cmocka_unit_test(test_scaling_failure_outputs),
        cmocka_unit_test_setup_teardown(test_subtraction, setup, teardown),
        cmocka_unit_test_setup_teardown(test_multiplication, setup, teardown),
        cmocka_unit_test_setup_teardown(test_division, setup, teardown),
        cmocka_unit_test_setup_teardown(test_powers, setup, teardown),
        cmocka_unit_test_setup_teardown(test_comparison, setup, teardown),
        cmocka_unit_test_setup_teardown(test_products_scaling_norm_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_addition_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_subtraction_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_multiplication_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_division_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_powers_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_comparison_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_products_scaling_norm_numerical_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_addition_numerical_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_subtraction_numerical_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_multiplication_numerical_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_division_numerical_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_powers_numerical_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_comparison_numerical_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_dimension_mismatch, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
