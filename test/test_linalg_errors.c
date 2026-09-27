#include <stdlib.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static int setup_matrix(void **state) {
    lmmc_mat_t *a = calloc(1, sizeof(*a));
    assert_non_null(a);
    *state = a;
    return 0;
}

static int teardown_matrix(void **state) {
    lmmc_mat_t *a = *state;
    lmmc_mat_destroy(a);
    free(a);
    return 0;
}

static void test_solver_null_arguments(void **state) {
    (void)state;
    lmmc_vec_t b = {0}, x = {0};
    size_t piv[2] = {0};
    double tau[2] = {0};

    assert_int_equal(lmmc_lu_solve(NULL, piv, &b, &x), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_cholesky_solve(NULL, &b, &x), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_qr_solve(NULL, tau, &b, &x), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_factorization_null_matrices(void **state) {
    (void)state;
    size_t piv[2] = {0};
    double tau[2] = {0};

    assert_int_equal(lmmc_lu_decompose_inplace(NULL, piv, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_cholesky_decompose_inplace(NULL), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_qr_decompose_inplace(NULL, tau, 2), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_lu_null_pivots(void **state) {
    lmmc_mat_t *a = *state;
    assert_int_equal(lmmc_mat_create(2, 2, a), LMMC_STATUS_OK);
    a->data[0] = 1.0;
    a->data[1] = 0.0;
    a->data[2] = 0.0;
    a->data[3] = 1.0;

    assert_int_equal(lmmc_lu_decompose_inplace(a, NULL, NULL), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_qr_null_tau(void **state) {
    lmmc_mat_t *a = *state;
    assert_int_equal(lmmc_mat_create(3, 2, a), LMMC_STATUS_OK);
    for (size_t i = 0; i < 6; i++)
        a->data[i] = 1.0;

    assert_int_equal(lmmc_qr_decompose_inplace(a, NULL, 2), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_lu_rectangular_arguments(void **state) {
    lmmc_mat_t *a = *state;
    size_t piv[2] = {0};
    assert_int_equal(lmmc_mat_create(2, 3, a), LMMC_STATUS_OK);
    for (size_t i = 0; i < 6; i++)
        a->data[i] = 1.0;

    assert_int_equal(lmmc_lu_decompose_inplace(a, piv, NULL), LMMC_STATUS_DIMENSION_MISMATCH);
}

static void test_cholesky_rectangular_arguments(void **state) {
    lmmc_mat_t *a = *state;
    assert_int_equal(lmmc_mat_create(2, 3, a), LMMC_STATUS_OK);
    for (size_t i = 0; i < 6; i++)
        a->data[i] = 1.0;

    assert_int_equal(lmmc_cholesky_decompose_inplace(a), LMMC_STATUS_DIMENSION_MISMATCH);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_solver_null_arguments),
        cmocka_unit_test(test_factorization_null_matrices),
        cmocka_unit_test_setup_teardown(test_lu_null_pivots, setup_matrix, teardown_matrix),
        cmocka_unit_test_setup_teardown(test_qr_null_tau, setup_matrix, teardown_matrix),
        cmocka_unit_test_setup_teardown(test_lu_rectangular_arguments, setup_matrix, teardown_matrix),
        cmocka_unit_test_setup_teardown(test_cholesky_rectangular_arguments, setup_matrix, teardown_matrix),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
