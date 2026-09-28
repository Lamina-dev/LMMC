#include <float.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"

struct test_fixture {
    lmmc_vec_t shape_result;
    lmmc_mat_t mat;
    lmmc_mat_t rectangular;
    lmmc_vec_t matvec_result;
    lmmc_mat_t rhs;
    lmmc_mat_t result;
    lmmc_std_bool_mat_t bool_mat_result;
};

static int setup(void **state) {
    *state = test_calloc(1, sizeof(struct test_fixture));
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_vec_destroy(&fixture->matvec_result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_vec_destroy(&fixture->shape_result);
    test_free(fixture);
    return 0;
}

static void test_shape(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    size_t rows = 0;
    size_t cols = 0;
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_shape(&fixture->mat, &rows, &cols) == LMMC_STATUS_OK);
    assert_true(rows == 2);
    assert_true(cols == 2);
    assert_true(lmmc_std_linalg_shape_vec(&fixture->rectangular, &fixture->shape_result) == LMMC_STATUS_OK);
    assert_true(fixture->shape_result.size == 2);
    assert_true(close_real(fixture->shape_result.data[0], 3));
    assert_true(close_real(fixture->shape_result.data[1], 2));
    lmmc_vec_destroy(&fixture->shape_result);

    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_products(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_real_t vec_short_values[] = {1, 2};
    lmmc_vec_t vec_short = {2, vec_short_values, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_matvec(&fixture->mat, &vec_short, &fixture->matvec_result) == LMMC_STATUS_OK);
    assert_true(fixture->matvec_result.size == 2);
    assert_true(close_real(fixture->matvec_result.data[0], 5));
    assert_true(close_real(fixture->matvec_result.data[1], 11));
    lmmc_vec_destroy(&fixture->matvec_result);

    assert_true(lmmc_std_linalg_matmul(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 19));
    assert_true(mat_close_at(&fixture->result, 0, 1, 22));
    assert_true(mat_close_at(&fixture->result, 1, 0, 43));
    assert_true(mat_close_at(&fixture->result, 1, 1, 50));
    lmmc_mat_destroy(&fixture->result);

    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_addition_scaling(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_add(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 6));
    assert_true(mat_close_at(&fixture->result, 0, 1, 8));
    assert_true(mat_close_at(&fixture->result, 1, 0, 10));
    assert_true(mat_close_at(&fixture->result, 1, 1, 12));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_mat_scale(&fixture->mat, 2, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 2));
    assert_true(mat_close_at(&fixture->result, 0, 1, 4));
    assert_true(mat_close_at(&fixture->result, 1, 0, 6));
    assert_true(mat_close_at(&fixture->result, 1, 1, 8));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_mat_add_scalar(&fixture->mat, 10, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 11));
    assert_true(mat_close_at(&fixture->result, 0, 1, 12));
    assert_true(mat_close_at(&fixture->result, 1, 0, 13));
    assert_true(mat_close_at(&fixture->result, 1, 1, 14));
    lmmc_mat_destroy(&fixture->result);

    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_strided_scale_copy(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t values[] = {1, -0.0, NAN, 3, 4, NAN};
    lmmc_real_t snapshot[6];
    lmmc_mat_t mat = {2, 2, 3, values, 0};
    memcpy(snapshot, values, sizeof(values));

    assert_true(lmmc_std_linalg_mat_scale(&mat, 1, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2 && fixture->result.cols == 2);
    assert_true(fixture->result.owns_data && fixture->result.data != values);
    assert_true(fixture->result.data[0] == 1);
    assert_true(fixture->result.data[1] == 0 && signbit(fixture->result.data[1]));
    assert_true(fixture->result.data[fixture->result.stride] == 3);
    assert_true(fixture->result.data[fixture->result.stride + 1] == 4);

    fixture->result.data[0] = 9;
    lmmc_mat_destroy(&fixture->result);
    assert_true(mat.data == values && !mat.owns_data);
    assert_true(memcmp(values, snapshot, sizeof(values)) == 0);
}

static void test_subtraction(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_sub(&fixture->rhs, &fixture->mat, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 4));
    assert_true(mat_close_at(&fixture->result, 0, 1, 4));
    assert_true(mat_close_at(&fixture->result, 1, 0, 4));
    assert_true(mat_close_at(&fixture->result, 1, 1, 4));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_mat_sub_scalar(&fixture->mat, 1, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 0));
    assert_true(mat_close_at(&fixture->result, 0, 1, 1));
    assert_true(mat_close_at(&fixture->result, 1, 0, 2));
    assert_true(mat_close_at(&fixture->result, 1, 1, 3));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_scalar_sub_mat(10, &fixture->mat, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 9));
    assert_true(mat_close_at(&fixture->result, 0, 1, 8));
    assert_true(mat_close_at(&fixture->result, 1, 0, 7));
    assert_true(mat_close_at(&fixture->result, 1, 1, 6));
    lmmc_mat_destroy(&fixture->result);

    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_multiplication(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_mul_elem(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 5));
    assert_true(mat_close_at(&fixture->result, 0, 1, 12));
    assert_true(mat_close_at(&fixture->result, 1, 0, 21));
    assert_true(mat_close_at(&fixture->result, 1, 1, 32));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_mat_mul_scalar(&fixture->mat, 2, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 2));
    assert_true(mat_close_at(&fixture->result, 0, 1, 4));
    assert_true(mat_close_at(&fixture->result, 1, 0, 6));
    assert_true(mat_close_at(&fixture->result, 1, 1, 8));
    lmmc_mat_destroy(&fixture->result);

    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_division(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_div(&fixture->rhs, &fixture->mat, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 5));
    assert_true(mat_close_at(&fixture->result, 0, 1, 3));
    assert_true(mat_close_at(&fixture->result, 1, 0, 7.0 / 3.0));
    assert_true(mat_close_at(&fixture->result, 1, 1, 2));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_mat_div_scalar(&fixture->rhs, 2, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 2.5));
    assert_true(mat_close_at(&fixture->result, 0, 1, 3));
    assert_true(mat_close_at(&fixture->result, 1, 0, 3.5));
    assert_true(mat_close_at(&fixture->result, 1, 1, 4));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_scalar_div_mat(12, &fixture->mat, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 12));
    assert_true(mat_close_at(&fixture->result, 0, 1, 6));
    assert_true(mat_close_at(&fixture->result, 1, 0, 4));
    assert_true(mat_close_at(&fixture->result, 1, 1, 3));
    lmmc_mat_destroy(&fixture->result);

    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_division_by_subnormal_scalar(void **state) {
    (void)state;
    const lmmc_real_t tiny = DBL_MIN / 16.0;
    lmmc_real_t values[] = {tiny, 0.0, -tiny, 0.0};
    lmmc_mat_t input = {2, 1, 2, values, 0};
    lmmc_mat_t result = {0};

    assert_true(lmmc_std_linalg_mat_div_scalar(&input, tiny, &result) == LMMC_STATUS_OK);
    assert_true(result.rows == 2 && result.cols == 1);
    assert_true(result.data[0] == 1.0 && result.data[result.stride] == -1.0);
    lmmc_mat_destroy(&result);
}

static void test_elementwise_powers(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_pow_elem(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 1));
    assert_true(mat_close_at(&fixture->result, 0, 1, 64));
    assert_true(mat_close_at(&fixture->result, 1, 0, 2187));
    assert_true(mat_close_at(&fixture->result, 1, 1, 65536));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_mat_pow_scalar(&fixture->mat, 2, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 1));
    assert_true(mat_close_at(&fixture->result, 0, 1, 4));
    assert_true(mat_close_at(&fixture->result, 1, 0, 9));
    assert_true(mat_close_at(&fixture->result, 1, 1, 16));
    lmmc_mat_destroy(&fixture->result);

    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_integer_powers(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_mat_pow_int(&fixture->mat, 2, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 7));
    assert_true(mat_close_at(&fixture->result, 0, 1, 10));
    assert_true(mat_close_at(&fixture->result, 1, 0, 15));
    assert_true(mat_close_at(&fixture->result, 1, 1, 22));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_mat_pow_int(&fixture->mat, 0, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, 1));
    assert_true(mat_close_at(&fixture->result, 0, 1, 0));
    assert_true(mat_close_at(&fixture->result, 1, 0, 0));
    assert_true(mat_close_at(&fixture->result, 1, 1, 1));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_mat_pow_int(&fixture->mat, -1, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 0, -2));
    assert_true(mat_close_at(&fixture->result, 0, 1, 1));
    assert_true(mat_close_at(&fixture->result, 1, 0, 1.5));
    assert_true(mat_close_at(&fixture->result, 1, 1, -0.5));
    lmmc_mat_destroy(&fixture->result);

    lmmc_mat_destroy(&fixture->mat);
}

static void test_comparison(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_mat_compare_scalar(&fixture->mat, LMMC_STD_COMPARE_GE, 3, &fixture->bool_mat_result) ==
                LMMC_STATUS_OK);
    assert_true(fixture->bool_mat_result.rows == 2);
    assert_true(fixture->bool_mat_result.cols == 2);
    assert_true(fixture->bool_mat_result.data[0] == 0);
    assert_true(fixture->bool_mat_result.data[1] == 0);
    assert_true(fixture->bool_mat_result.data[fixture->bool_mat_result.stride] == 1);
    assert_true(fixture->bool_mat_result.data[fixture->bool_mat_result.stride + 1] == 1);
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);

    assert_true(lmmc_std_linalg_mat_compare(&fixture->mat, &fixture->rhs, LMMC_STD_COMPARE_LE, &fixture->bool_mat_result) ==
                LMMC_STATUS_OK);
    assert_true(fixture->bool_mat_result.rows == 2);
    assert_true(fixture->bool_mat_result.cols == 2);
    assert_true(fixture->bool_mat_result.data[0] == 1);
    assert_true(fixture->bool_mat_result.data[1] == 1);
    assert_true(fixture->bool_mat_result.data[fixture->bool_mat_result.stride] == 1);
    assert_true(fixture->bool_mat_result.data[fixture->bool_mat_result.stride + 1] == 1);
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
    lmmc_std_bool_mat_destroy(NULL);

    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_power_domain_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    fixture->mat.data[0] = -1;
    assert_true(lmmc_std_linalg_mat_pow_scalar(&fixture->mat, 0.5, &fixture->result) == LMMC_STATUS_OUT_OF_RANGE);
    fixture->mat.data[0] = 0;
    assert_true(lmmc_std_linalg_mat_pow_scalar(&fixture->mat, -1, &fixture->result) == LMMC_STATUS_OUT_OF_RANGE);
    fixture->mat.data[0] = 1;

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_division_by_zero(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    fixture->rhs.data[0] = 0;
    assert_true(lmmc_std_linalg_mat_div(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_scalar_div_mat(12, &fixture->rhs, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    fixture->rhs.data[0] = 5;

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_transpose_adjoint(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_transpose(&fixture->mat, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 2);
    assert_true(fixture->result.cols == 2);
    assert_true(mat_close_at(&fixture->result, 0, 1, 3));
    assert_true(mat_close_at(&fixture->result, 1, 0, 2));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_adjoint(&fixture->mat, &fixture->result) == LMMC_STATUS_OK);
    assert_true(mat_close_at(&fixture->result, 0, 1, 3));
    assert_true(mat_close_at(&fixture->result, 1, 0, 2));
    lmmc_mat_destroy(&fixture->result);

    lmmc_mat_destroy(&fixture->mat);
}

static void test_constructors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};

    assert_true(lmmc_std_linalg_eye(3, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 3);
    assert_true(fixture->result.cols == 3);
    assert_true(mat_close_at(&fixture->result, 0, 0, 1));
    assert_true(mat_close_at(&fixture->result, 1, 1, 1));
    assert_true(mat_close_at(&fixture->result, 2, 2, 1));
    assert_true(mat_close_at(&fixture->result, 0, 1, 0));
    assert_true(mat_close_at(&fixture->result, 1, 2, 0));
    lmmc_mat_destroy(&fixture->result);

    assert_true(lmmc_std_linalg_diag(&vec_a, &fixture->result) == LMMC_STATUS_OK);
    assert_true(fixture->result.rows == 3);
    assert_true(fixture->result.cols == 3);
    assert_true(mat_close_at(&fixture->result, 0, 0, 1));
    assert_true(mat_close_at(&fixture->result, 1, 1, 2));
    assert_true(mat_close_at(&fixture->result, 2, 2, 3));
    assert_true(mat_close_at(&fixture->result, 0, 2, 0));
    assert_true(mat_close_at(&fixture->result, 2, 0, 0));
    lmmc_mat_destroy(&fixture->result);
}

static void test_trace(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_trace(&fixture->mat, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 5));

    lmmc_mat_destroy(&fixture->mat);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_shape, setup, teardown),
        cmocka_unit_test_setup_teardown(test_products, setup, teardown),
        cmocka_unit_test_setup_teardown(test_addition_scaling, setup, teardown),
        cmocka_unit_test_setup_teardown(test_strided_scale_copy, setup, teardown),
        cmocka_unit_test_setup_teardown(test_subtraction, setup, teardown),
        cmocka_unit_test_setup_teardown(test_multiplication, setup, teardown),
        cmocka_unit_test_setup_teardown(test_division, setup, teardown),
        cmocka_unit_test(test_division_by_subnormal_scalar),
        cmocka_unit_test_setup_teardown(test_elementwise_powers, setup, teardown),
        cmocka_unit_test_setup_teardown(test_integer_powers, setup, teardown),
        cmocka_unit_test_setup_teardown(test_comparison, setup, teardown),
        cmocka_unit_test_setup_teardown(test_power_domain_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_division_by_zero, setup, teardown),
        cmocka_unit_test_setup_teardown(test_transpose_adjoint, setup, teardown),
        cmocka_unit_test_setup_teardown(test_constructors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_trace, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
