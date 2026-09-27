#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"

struct test_fixture {
    lmmc_mat_t mat;
    lmmc_mat_t result;
    lmmc_mat_t rectangular;
    lmmc_mat_t rhs;
    lmmc_eigen_gen_full_result_t eig_result;
    lmmc_std_eig_table_t eig_table;
    lmmc_svd_result_t svd_result;
    lmmc_std_svd_table_t svd_table;
    lmmc_mat_t mismatched_rhs;
    lmmc_mat_t singular;
};

static int setup(void **state) {
    *state = test_calloc(1, sizeof(struct test_fixture));
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->singular);
    lmmc_mat_destroy(&fixture->mismatched_rhs);
    lmmc_std_svd_table_destroy(&fixture->svd_table);
    lmmc_svd_result_destroy(&fixture->svd_result);
    lmmc_std_eig_table_destroy(&fixture->eig_table);
    lmmc_eigen_gen_full_result_destroy(&fixture->eig_result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mat);
    test_free(fixture);
    return 0;
}

static void test_determinant(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_det(&fixture->mat, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, -2));

    lmmc_mat_destroy(&fixture->mat);
}

static void test_inverse(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_inv(&fixture->mat, &fixture->result) == LMMC_STATUS_OK);
    assert_true(mat_close_at(&fixture->result, 0, 0, -2));
    assert_true(mat_close_at(&fixture->result, 0, 1, 1));
    assert_true(mat_close_at(&fixture->result, 1, 0, 1.5));
    assert_true(mat_close_at(&fixture->result, 1, 1, -0.5));
    lmmc_mat_destroy(&fixture->result);

    lmmc_mat_destroy(&fixture->mat);
}

static void test_rank(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    size_t rank = 0;
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_rank(&fixture->rectangular, &rank) == LMMC_STATUS_OK);
    assert_true(rank == 1);

    lmmc_mat_destroy(&fixture->rectangular);
}

static void test_left_solve(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_solve_left(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_OK);
    assert_true(mat_close_at(&fixture->result, 0, 0, -3));
    assert_true(mat_close_at(&fixture->result, 0, 1, -4));
    assert_true(mat_close_at(&fixture->result, 1, 0, 4));
    assert_true(mat_close_at(&fixture->result, 1, 1, 5));
    lmmc_mat_destroy(&fixture->result);

    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_right_solve(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_solve_right(&fixture->rhs, &fixture->mat, &fixture->result) == LMMC_STATUS_OK);
    assert_true(mat_close_at(&fixture->result, 0, 0, -1));
    assert_true(mat_close_at(&fixture->result, 0, 1, 2));
    assert_true(mat_close_at(&fixture->result, 1, 0, -2));
    assert_true(mat_close_at(&fixture->result, 1, 1, 3));
    lmmc_mat_destroy(&fixture->result);

    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_eigenvalues(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_eig(&fixture->mat, &fixture->eig_result) == LMMC_STATUS_OK);
    assert_true(fixture->eig_result.real_parts.size == 2);
    assert_true(fixture->eig_result.imag_parts.size == 2);
    assert_true(close_real(fixture->eig_result.imag_parts.data[0], 0));
    assert_true(close_real(fixture->eig_result.imag_parts.data[1], 0));
    lmmc_eigen_gen_full_result_destroy(&fixture->eig_result);

    lmmc_mat_destroy(&fixture->mat);
}

static void test_eigen_table(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    const lmmc_mat_t *named = NULL;
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_eig_table(&fixture->mat, &fixture->eig_table) == LMMC_STATUS_OK);
    lmmc_std_eig_table_destroy(NULL);
    named = lmmc_std_eig_table_get(&fixture->eig_table, "values_real");
    assert_true(named);
    assert_true(named->rows == 2);
    assert_true(named->cols == 1);
    assert_true(isfinite((double)named->data[0]));
    assert_true(isfinite((double)named->data[named->stride]));
    assert_true(lmmc_std_eig_table_count(&fixture->eig_table) == 4);
    assert_true(strcmp(lmmc_std_eig_table_key(&fixture->eig_table, 0), "values_real") == 0);
    assert_true(strcmp(lmmc_std_eig_table_key(&fixture->eig_table, 3), "vectors_imag") == 0);
    assert_true(lmmc_std_eig_table_count(NULL) == 0);
    assert_true(lmmc_std_eig_table_key(&fixture->eig_table, 4) == NULL);
    assert_true(lmmc_std_eig_table_key(NULL, 0) == NULL);
    assert_true(lmmc_std_eig_table_get(&fixture->eig_table, "missing") == NULL);
    assert_true(lmmc_std_eig_table_get(&fixture->eig_table, NULL) == NULL);
    assert_true(lmmc_std_eig_table_get(NULL, "values_real") == NULL);
    named = lmmc_std_eig_table_get(&fixture->eig_table, "vectors_real");
    assert_true(named);
    assert_true(named->rows == 2);
    assert_true(named->cols == 2);
    lmmc_std_eig_table_destroy(&fixture->eig_table);

    lmmc_mat_destroy(&fixture->mat);
}

static void test_singular_values(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_svd(&fixture->mat, &fixture->svd_result) == LMMC_STATUS_OK);
    assert_true(fixture->svd_result.sigma.size == 2);
    assert_true(fixture->svd_result.U.rows == 2);
    assert_true(fixture->svd_result.Vt.cols == 2);
    assert_true(fixture->svd_result.sigma.data[0] >= fixture->svd_result.sigma.data[1]);
    lmmc_svd_result_destroy(&fixture->svd_result);

    lmmc_mat_destroy(&fixture->mat);
}

static void test_svd_table(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    const lmmc_mat_t *named = NULL;
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_svd_table(&fixture->mat, &fixture->svd_table) == LMMC_STATUS_OK);
    lmmc_std_svd_table_destroy(NULL);
    named = lmmc_std_svd_table_get(&fixture->svd_table, "S");
    assert_true(named);
    assert_true(named->rows == 2);
    assert_true(named->cols == 2);
    assert_true(named->data[0] >= named->data[named->stride + 1]);
    assert_true(lmmc_std_svd_table_count(&fixture->svd_table) == 3);
    assert_true(strcmp(lmmc_std_svd_table_key(&fixture->svd_table, 0), "U") == 0);
    assert_true(strcmp(lmmc_std_svd_table_key(&fixture->svd_table, 2), "Vt") == 0);
    assert_true(lmmc_std_svd_table_count(NULL) == 0);
    assert_true(lmmc_std_svd_table_key(&fixture->svd_table, 3) == NULL);
    assert_true(lmmc_std_svd_table_key(NULL, 0) == NULL);
    assert_true(lmmc_std_svd_table_get(&fixture->svd_table, "sigma") == NULL);
    assert_true(lmmc_std_svd_table_get(&fixture->svd_table, NULL) == NULL);
    assert_true(lmmc_std_svd_table_get(NULL, "S") == NULL);
    named = lmmc_std_svd_table_get(&fixture->svd_table, "U");
    assert_true(named);
    assert_true(named->rows == 2);
    assert_true(named->cols == 2);
    lmmc_std_svd_table_destroy(&fixture->svd_table);

    lmmc_mat_destroy(&fixture->mat);
}

static void test_rectangular_eigen_error(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_eig(&fixture->rectangular, &fixture->eig_result) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_eigen_gen_full_result_destroy(&fixture->eig_result);
    lmmc_mat_destroy(&fixture->rectangular);
}

static void test_determinant_inverse_rank_null_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    size_t rank = 0;
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_det(NULL, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_rank(NULL, &rank) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_inv(NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_det(&fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_rank(&fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_inv(&fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_linear_solves_null_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_solve_left(NULL, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_solve_left(&fixture->mat, NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_solve_right(NULL, &fixture->mat, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_solve_right(&fixture->rhs, NULL, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_solve_left(&fixture->mat, &fixture->rhs, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_solve_right(&fixture->rhs, &fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_decompositions_null_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    assert_true(lmmc_std_linalg_eig(NULL, &fixture->eig_result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_svd(NULL, &fixture->svd_result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_eig_table(NULL, &fixture->eig_table) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_svd_table(NULL, &fixture->svd_table) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_eig(&fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_svd(&fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_eig_table(&fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_svd_table(&fixture->mat, NULL) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_std_svd_table_destroy(&fixture->svd_table);
    lmmc_std_eig_table_destroy(&fixture->eig_table);
    lmmc_svd_result_destroy(&fixture->svd_result);
    lmmc_eigen_gen_full_result_destroy(&fixture->eig_result);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_determinant_inverse_rank_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_mat_t invalid_stride = {0};
    size_t rank = 0;
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_det(&invalid_stride, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_rank(&invalid_stride, &rank) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_inv(&invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
}

static void test_linear_solves_invalid_stride(void **state) {
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

    assert_true(lmmc_std_linalg_solve_left(&invalid_stride, &fixture->rhs, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_solve_left(&fixture->mat, &invalid_stride, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_solve_right(&invalid_stride, &fixture->mat, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_solve_right(&fixture->rhs, &invalid_stride, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_decompositions_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_mat_t invalid_stride = {0};
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_eig(&invalid_stride, &fixture->eig_result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_svd(&invalid_stride, &fixture->svd_result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_eig_table(&invalid_stride, &fixture->eig_table) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_svd_table(&invalid_stride, &fixture->svd_table) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_std_svd_table_destroy(&fixture->svd_table);
    lmmc_std_eig_table_destroy(&fixture->eig_table);
    lmmc_svd_result_destroy(&fixture->svd_result);
    lmmc_eigen_gen_full_result_destroy(&fixture->eig_result);
    lmmc_mat_destroy(&invalid_stride);
}

static void test_linear_solves_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(1, 1, &fixture->mismatched_rhs) == LMMC_STATUS_OK);
    fixture->mismatched_rhs.data[0] = 1;

    assert_true(lmmc_std_linalg_solve_left(&fixture->mat, &fixture->mismatched_rhs, &fixture->result) ==
                LMMC_STATUS_DIMENSION_MISMATCH);
    assert_true(lmmc_std_linalg_solve_right(&fixture->mismatched_rhs, &fixture->mat, &fixture->result) ==
                LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mismatched_rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_determinant_inverse_rank_singularity(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t singular_values[] = {1, 2, 2, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->singular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->singular, singular_values);

    assert_true(lmmc_std_linalg_inv(&fixture->singular, &fixture->result) == LMMC_STATUS_SINGULAR_MATRIX);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->singular);
}

static void test_linear_solves_singularity(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t singular_values[] = {1, 2, 2, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->singular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->singular, singular_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    assert_true(lmmc_std_linalg_solve_left(&fixture->singular, &fixture->rhs, &fixture->result) == LMMC_STATUS_SINGULAR_MATRIX);
    assert_true(lmmc_std_linalg_solve_right(&fixture->rhs, &fixture->singular, &fixture->result) == LMMC_STATUS_SINGULAR_MATRIX);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->singular);
}

static void test_determinant_inverse_rank_nonfinite_inputs(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    size_t rank = 0;
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    fixture->mat.data[0] = NAN;
    assert_true(lmmc_std_linalg_det(&fixture->mat, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_rank(&fixture->mat, &rank) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_inv(&fixture->mat, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_linear_solves_nonfinite_inputs(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);

    fixture->mat.data[0] = NAN;
    assert_true(lmmc_std_linalg_solve_left(&fixture->mat, &fixture->rhs, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_solve_right(&fixture->rhs, &fixture->mat, &fixture->result) == LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_decompositions_nonfinite_inputs(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);

    fixture->mat.data[0] = NAN;
    fixture->eig_table.values_real.rows = 123;
    fixture->svd_table.U.rows = 123;
    assert_true(lmmc_std_linalg_eig(&fixture->mat, &fixture->eig_result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_svd(&fixture->mat, &fixture->svd_result) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_eig_table(&fixture->mat, &fixture->eig_table) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_linalg_svd_table(&fixture->mat, &fixture->svd_table) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(fixture->eig_table.values_real.rows == 0);
    assert_true(fixture->svd_table.U.rows == 0);

    lmmc_std_svd_table_destroy(&fixture->svd_table);
    lmmc_std_eig_table_destroy(&fixture->eig_table);
    lmmc_svd_result_destroy(&fixture->svd_result);
    lmmc_eigen_gen_full_result_destroy(&fixture->eig_result);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_determinant_inverse_rank_rectangular_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_det(&fixture->rectangular, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_inv(&fixture->rectangular, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_determinant, setup, teardown),
        cmocka_unit_test_setup_teardown(test_inverse, setup, teardown),
        cmocka_unit_test_setup_teardown(test_rank, setup, teardown),
        cmocka_unit_test_setup_teardown(test_left_solve, setup, teardown),
        cmocka_unit_test_setup_teardown(test_right_solve, setup, teardown),
        cmocka_unit_test_setup_teardown(test_eigenvalues, setup, teardown),
        cmocka_unit_test_setup_teardown(test_eigen_table, setup, teardown),
        cmocka_unit_test_setup_teardown(test_singular_values, setup, teardown),
        cmocka_unit_test_setup_teardown(test_svd_table, setup, teardown),
        cmocka_unit_test_setup_teardown(test_rectangular_eigen_error, setup, teardown),
        cmocka_unit_test_setup_teardown(test_determinant_inverse_rank_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_linear_solves_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_decompositions_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_determinant_inverse_rank_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_linear_solves_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_decompositions_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_linear_solves_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_determinant_inverse_rank_singularity, setup, teardown),
        cmocka_unit_test_setup_teardown(test_linear_solves_singularity, setup, teardown),
        cmocka_unit_test_setup_teardown(test_determinant_inverse_rank_nonfinite_inputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_linear_solves_nonfinite_inputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_decompositions_nonfinite_inputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_determinant_inverse_rank_rectangular_errors, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
