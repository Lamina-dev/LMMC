#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"
#include "internal_test_hooks.h"

#include <float.h>

struct test_fixture {
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
    lmmc_memory_fail_reset_for_test();
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_vec_destroy(&fixture->matvec_result);
    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mat);
    test_free(fixture);
    return 0;
}

static void test_shape_transforms_constructors_nonfinite_inputs(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    fixture->mat.data[0] = NAN;
    assert_true(lmmc_std_linalg_transpose(&fixture->mat, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_adjoint(&fixture->mat, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_trace(&fixture->mat, &out) == LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_products_nonfinite_inputs(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_real_t vec_short_values[] = {1, 2};
    lmmc_vec_t vec_short = {2, vec_short_values, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    fixture->mat.data[0] = NAN;
    assert_true(lmmc_std_linalg_matmul(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_matvec(&fixture->mat, &vec_short, &fixture->matvec_result) ==
                LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_vec_destroy(&fixture->matvec_result);
}

static void test_addition_scaling_nonfinite_inputs(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    fixture->mat.data[0] = NAN;
    assert_true(lmmc_std_linalg_mat_add(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_add_scalar(&fixture->mat, 10, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_add_scalar(&fixture->rhs, NAN, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_scale(&fixture->mat, 2, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_scale(&fixture->rhs, NAN, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_scale_nonfinite_before_allocation(void **state) {
    (void)state;
    lmmc_real_t value[] = {NAN};
    lmmc_real_t sentinel[] = {123, 456};
    lmmc_mat_t mat = {1, 1, 1, value, 0};
    lmmc_mat_t rejected = {2, 1, 1, sentinel, 0};
    lmmc_mat_t result = rejected;
    lmmc_status_t invalid_status;
    lmmc_status_t allocation_status;

    lmmc_memory_fail_after_for_test(0);
    invalid_status = lmmc_std_linalg_mat_scale(&mat, 1, &rejected);
    value[0] = 1;
    allocation_status = lmmc_std_linalg_mat_scale(&mat, 1, &result);
    lmmc_memory_fail_reset_for_test();

    assert_true(invalid_status == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(allocation_status == LMMC_STATUS_ALLOCATION_FAILED);
    assert_true(rejected.rows == 2 && rejected.cols == 1 && rejected.stride == 1);
    assert_true(rejected.data == sentinel && !rejected.owns_data);
    assert_true(result.rows == 2 && result.cols == 1 && result.stride == 1);
    assert_true(result.data == sentinel && !result.owns_data);
    assert_true(sentinel[0] == 123 && sentinel[1] == 456);
    assert_true(value[0] == 1);
    lmmc_mat_destroy(&result);
    lmmc_mat_destroy(&rejected);
}

static void test_scale_overflow_clears_output(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t value[] = {DBL_MAX};
    lmmc_mat_t mat = {1, 1, 1, value, 0};

    assert_true(lmmc_std_linalg_mat_scale(&mat, 2, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(fixture->result.rows == 0 && fixture->result.cols == 0 && fixture->result.stride == 0);
    assert_true(fixture->result.data == NULL && !fixture->result.owns_data);
    assert_true(value[0] == DBL_MAX);
    lmmc_mat_destroy(&fixture->result);
}

static void test_subtraction_nonfinite_inputs(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    fixture->mat.data[0] = NAN;
    assert_true(lmmc_std_linalg_mat_sub(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_sub_scalar(&fixture->mat, 1, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_sub_scalar(&fixture->rhs, NAN, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_scalar_sub_mat(1, &fixture->mat, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_scalar_sub_mat(NAN, &fixture->rhs, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_multiplication_nonfinite_inputs(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    fixture->mat.data[0] = NAN;
    assert_true(lmmc_std_linalg_mat_mul_elem(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_mul_scalar(&fixture->mat, 2, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_mul_scalar(&fixture->rhs, NAN, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_division_nonfinite_inputs(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    fixture->mat.data[0] = NAN;
    assert_true(lmmc_std_linalg_mat_div(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_div_scalar(&fixture->mat, 2, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_div_scalar(&fixture->rhs, NAN, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_scalar_div_mat(2, &fixture->mat, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_scalar_div_mat(NAN, &fixture->rhs, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_powers_nonfinite_inputs(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    fixture->mat.data[0] = NAN;
    assert_true(lmmc_std_linalg_mat_pow_elem(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_pow_scalar(&fixture->mat, 2, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_pow_scalar(&fixture->rhs, NAN, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_comparison_nonfinite_inputs(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    fixture->mat.data[0] = NAN;
    assert_true(lmmc_std_linalg_mat_compare(&fixture->mat, &fixture->rhs, LMMC_STD_COMPARE_EQ, &fixture->bool_mat_result) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_compare_scalar(&fixture->mat, LMMC_STD_COMPARE_GT, 1, &fixture->bool_mat_result) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_mat_compare_scalar(&fixture->rhs, LMMC_STD_COMPARE_GT, NAN, &fixture->bool_mat_result) ==
                LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
}

static void test_shape_transforms_constructors_nonfinite_constructor(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_nonfinite_values[] = {1, NAN, 3};
    lmmc_vec_t vec_nonfinite = {3, vec_nonfinite_values, 0};

    assert_true(lmmc_std_linalg_diag(&vec_nonfinite, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_mat_destroy(&fixture->result);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_shape_transforms_constructors_nonfinite_inputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_products_nonfinite_inputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_addition_scaling_nonfinite_inputs, setup, teardown),
        cmocka_unit_test(test_scale_nonfinite_before_allocation),
        cmocka_unit_test_setup_teardown(test_scale_overflow_clears_output, setup, teardown),
        cmocka_unit_test_setup_teardown(test_subtraction_nonfinite_inputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_multiplication_nonfinite_inputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_division_nonfinite_inputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_powers_nonfinite_inputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_comparison_nonfinite_inputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_shape_transforms_constructors_nonfinite_constructor, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
