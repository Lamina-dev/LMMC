#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"

struct test_fixture {
    lmmc_vec_t shape_result;
    lmmc_mat_t result;
    lmmc_vec_t matvec_result;
    lmmc_mat_t mat;
    lmmc_mat_t rhs;
    lmmc_std_bool_mat_t bool_mat_result;
    lmmc_mat_t rectangular;
    lmmc_mat_t singular;
};

static int setup(void **state) {
    *state = test_calloc(1, sizeof(struct test_fixture));
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->singular);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_vec_destroy(&fixture->matvec_result);
    lmmc_mat_destroy(&fixture->result);
    lmmc_vec_destroy(&fixture->shape_result);
    test_free(fixture);
    return 0;
}

static void test_shape_transforms_constructors_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_mat_t invalid_stride = {0};
    size_t rows = 0;
    size_t cols = 0;
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_shape(&invalid_stride, &rows, &cols) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_shape_vec(&invalid_stride, &fixture->shape_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_transpose(&invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_adjoint(&invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_trace(&invalid_stride, &out) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_vec_destroy(&fixture->shape_result);
}

static void test_products_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_real_t vec_short_values[] = {1, 2};
    lmmc_vec_t vec_short = {2, vec_short_values, 0};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_matmul(&invalid_stride, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_matmul(&fixture->mat, &invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_matvec(&invalid_stride, &vec_short, &fixture->matvec_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_vec_destroy(&fixture->matvec_result);
}

static void test_addition_scaling_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_add(&invalid_stride, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_add(&fixture->mat, &invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_add_scalar(&invalid_stride, 10, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_scale(&invalid_stride, 2, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_subtraction_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_sub(&invalid_stride, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_sub(&fixture->mat, &invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_sub_scalar(&invalid_stride, 1, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_multiplication_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_mul_elem(&invalid_stride, &fixture->rhs, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_mul_elem(&fixture->mat, &invalid_stride, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_division_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_div(&invalid_stride, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_div(&fixture->mat, &invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_div_scalar(&invalid_stride, 2, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_powers_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_pow_elem(&invalid_stride, &fixture->rhs, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_pow_elem(&fixture->mat, &invalid_stride, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_pow_scalar(&invalid_stride, 2, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_comparison_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_compare(
                    &invalid_stride,
                    &fixture->rhs,
                    LMMC_STD_COMPARE_EQ,
                    &fixture->bool_mat_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_compare(
                    &fixture->mat,
                    &invalid_stride,
                    LMMC_STD_COMPARE_EQ,
                    &fixture->bool_mat_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_compare_scalar(
                    &invalid_stride,
                    LMMC_STD_COMPARE_GT,
                    1,
                    &fixture->bool_mat_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
}

static void test_addition_scaling_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_add(&fixture->mat, &fixture->rectangular, &fixture->result) == LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_subtraction_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_sub(&fixture->mat, &fixture->rectangular, &fixture->result) == LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_multiplication_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_mul_elem(&fixture->mat, &fixture->rectangular, &fixture->result) ==
                LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_division_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_div(&fixture->mat, &fixture->rectangular, &fixture->result) == LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_powers_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_pow_elem(&fixture->mat, &fixture->rectangular, &fixture->result) ==
                LMMC_STATUS_DIMENSION_MISMATCH);
    assert_true(lmmc_std_linalg_mat_pow_int(&fixture->rectangular, 2, &fixture->result) == LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_comparison_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_compare(&fixture->mat, &fixture->rectangular, LMMC_STD_COMPARE_EQ, &fixture->bool_mat_result) ==
                LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
}

static void test_products_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_matmul(&fixture->mat, &fixture->rectangular, &fixture->result) == LMMC_STATUS_DIMENSION_MISMATCH);
    assert_true(lmmc_std_linalg_matvec(&fixture->mat, &vec_a, &fixture->matvec_result) == LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_vec_destroy(&fixture->matvec_result);
}

static void test_powers_singularity(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t singular_values[] = {1, 2, 2, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->singular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->singular, singular_values);

    assert_true(lmmc_std_linalg_mat_pow_int(&fixture->singular, -1, &fixture->result) == LMMC_STATUS_SINGULAR_MATRIX);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->singular);
}

static void test_shape_transforms_constructors_rectangular_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_trace(&fixture->rectangular, &out) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->rectangular);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_shape_transforms_constructors_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_products_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_addition_scaling_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_subtraction_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_multiplication_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_division_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_powers_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_comparison_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_addition_scaling_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_subtraction_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_multiplication_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_division_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_powers_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_comparison_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_products_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_powers_singularity, setup, teardown),
        cmocka_unit_test_setup_teardown(test_shape_transforms_constructors_rectangular_errors, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
