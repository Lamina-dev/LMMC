#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"

struct test_fixture {
    lmmc_vec_t shape_result;
    lmmc_mat_t mat;
    lmmc_mat_t result;
    lmmc_vec_t matvec_result;
    lmmc_mat_t rhs;
    lmmc_std_bool_mat_t bool_mat_result;
};

static int setup(void **state) {
    *state = test_calloc(1, sizeof(struct test_fixture));
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_vec_destroy(&fixture->matvec_result);
    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_vec_destroy(&fixture->shape_result);
    test_free(fixture);
    return 0;
}

static void test_shape_transforms_constructors_null_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    size_t rows = 0;
    size_t cols = 0;
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_shape(NULL, &rows, &cols) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_shape_vec(NULL, &fixture->shape_result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_shape_vec(&fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_transpose(NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_adjoint(NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_trace(NULL, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_shape(&fixture->mat, NULL, &cols) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_shape(&fixture->mat, &rows, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_transpose(&fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_adjoint(&fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_trace(&fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_vec_destroy(&fixture->shape_result);
}

static void test_products_null_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_real_t vec_short_values[] = {1, 2};
    lmmc_vec_t vec_short = {2, vec_short_values, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_matmul(NULL, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_matmul(&fixture->mat, NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_matmul(&fixture->mat, &fixture->rhs, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_matvec(NULL, &vec_short, &fixture->matvec_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_matvec(&fixture->mat, NULL, &fixture->matvec_result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_matvec(&fixture->mat, &vec_short, NULL) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_vec_destroy(&fixture->matvec_result);
}

static void test_addition_scaling_null_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_add(NULL, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_add(&fixture->mat, NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_add(&fixture->mat, &fixture->rhs, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_add_scalar(NULL, 10, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_add_scalar(&fixture->mat, 10, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_scale(NULL, 2, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_scale(&fixture->mat, 2, NULL) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_subtraction_null_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_sub(NULL, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_sub(&fixture->mat, NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_sub(&fixture->mat, &fixture->rhs, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_sub_scalar(NULL, 1, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_sub_scalar(&fixture->mat, 1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_scalar_sub_mat(1, NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_scalar_sub_mat(1, &fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_multiplication_null_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_mul_elem(NULL, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_mul_elem(&fixture->mat, NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_mul_elem(&fixture->mat, &fixture->rhs, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_mul_scalar(NULL, 2, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_mul_scalar(&fixture->mat, 2, NULL) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_division_null_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_div(NULL, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_div(&fixture->mat, NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_div(&fixture->mat, &fixture->rhs, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_div_scalar(NULL, 2, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_div_scalar(&fixture->mat, 2, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_scalar_div_mat(2, NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_scalar_div_mat(2, &fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_powers_null_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_pow_elem(NULL, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_pow_elem(&fixture->mat, NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_pow_elem(&fixture->mat, &fixture->rhs, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_pow_scalar(NULL, 2, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_pow_scalar(&fixture->mat, 2, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_pow_int(NULL, 2, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_pow_int(&fixture->mat, 2, NULL) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_comparison_null_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_compare(NULL, &fixture->rhs, LMMC_STD_COMPARE_EQ, &fixture->bool_mat_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_compare(&fixture->mat, NULL, LMMC_STD_COMPARE_EQ, &fixture->bool_mat_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_compare(&fixture->mat, &fixture->rhs, LMMC_STD_COMPARE_EQ, NULL) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_compare_scalar(NULL, LMMC_STD_COMPARE_GE, 1, &fixture->bool_mat_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_compare_scalar(&fixture->mat, LMMC_STD_COMPARE_GE, 1, NULL) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_compare_scalar(
                    &fixture->mat,
                    (lmmc_std_compare_op_t)99,
                    1,
                    &fixture->bool_mat_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
}

static void test_shape_transforms_constructors_invalid_constructors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};

    assert_true(lmmc_std_linalg_eye(0, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_eye(2, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_diag(NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_diag(&vec_a, NULL) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_shape_transforms_constructors_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_products_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_addition_scaling_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_subtraction_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_multiplication_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_division_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_powers_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_comparison_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_shape_transforms_constructors_invalid_constructors, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
