/**
 * @file test_sparse_extended_addition.c
 * @brief 稀疏矩阵加法测试。
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define TEST_EPS_TIGHT 1e-12
#define TEST_EPS_NORMAL 1e-10

#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_sparse_mat_t addition_sa;
    lmmc_sparse_mat_t addition_sb;
    lmmc_sparse_mat_t addition_sc;
    lmmc_mat_t addition_dc_sparse;
    lmmc_sparse_mat_t check_empty_transpose_zero_t;
    lmmc_mat_t empty_operations_zero_dense;
    lmmc_sparse_mat_t empty_operations_zero_sparse;
    lmmc_vec_t empty_operations_x;
    lmmc_vec_t empty_operations_y;
    lmmc_sparse_coo_t coo_duplicates_coo;
    lmmc_sparse_mat_t coo_duplicates_sparse;
    lmmc_mat_t coo_duplicates_dense;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->coo_duplicates_dense);
    lmmc_sparse_destroy(&fixture->coo_duplicates_sparse);
    lmmc_sparse_coo_destroy(&fixture->coo_duplicates_coo);
    lmmc_vec_destroy(&fixture->empty_operations_y);
    lmmc_vec_destroy(&fixture->empty_operations_x);
    lmmc_sparse_destroy(&fixture->empty_operations_zero_sparse);
    lmmc_mat_destroy(&fixture->empty_operations_zero_dense);
    lmmc_sparse_destroy(&fixture->check_empty_transpose_zero_t);
    lmmc_mat_destroy(&fixture->addition_dc_sparse);
    lmmc_sparse_destroy(&fixture->addition_sc);
    lmmc_sparse_destroy(&fixture->addition_sb);
    lmmc_sparse_destroy(&fixture->addition_sa);
    free(fixture);
    return 0;
}

static void test_addition(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double a_data[] = {
        1.0, 0.0, 2.0,
        0.0, 3.0, 0.0,
        4.0, 0.0, 5.0};
    double b_data[] = {
        0.0, 6.0, 0.0,
        7.0, 0.0, 8.0,
        0.0, 9.0, 0.0};

    st = lmmc_test_build_sparse(a_data, 3, 3, &fixture->addition_sa);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_test_build_sparse(b_data, 3, 3, &fixture->addition_sb);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_add(1.0, &fixture->addition_sa, 1.0, &fixture->addition_sb, &fixture->addition_sc);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_mat_create(3, 3, &fixture->addition_dc_sparse);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_to_dense(&fixture->addition_sc, &fixture->addition_dc_sparse);
    assert_true(st == LMMC_STATUS_OK);

    double expected[] = {1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0};
    for (size_t i = 0; i < 9; i++) {
        assert_true(lmmc_test_nearly_equal(fixture->addition_dc_sparse.data[i], expected[i], TEST_EPS_TIGHT));
    }

    lmmc_mat_destroy(&fixture->addition_dc_sparse);
    lmmc_sparse_destroy(&fixture->addition_sc);
    lmmc_sparse_destroy(&fixture->addition_sb);
    lmmc_sparse_destroy(&fixture->addition_sa);
}

static void check_empty_transpose(struct test_fixture *fixture, const lmmc_sparse_mat_t *zero_sparse) {

    lmmc_status_t st = lmmc_sparse_transpose(zero_sparse, &fixture->check_empty_transpose_zero_t);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->check_empty_transpose_zero_t.nnz == 0);
    lmmc_sparse_destroy(&fixture->check_empty_transpose_zero_t);
}

static void test_empty_operations(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_mat_create(3, 3, &fixture->empty_operations_zero_dense);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_from_dense(&fixture->empty_operations_zero_dense, 0.0, &fixture->empty_operations_zero_sparse);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(fixture->empty_operations_zero_sparse.nnz == 0);

    st = lmmc_vec_create(3, &fixture->empty_operations_x);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_fill(&fixture->empty_operations_x, 5.0);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(3, &fixture->empty_operations_y);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_mat_vec_mul(&fixture->empty_operations_zero_sparse, &fixture->empty_operations_x, &fixture->empty_operations_y);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < 3; i++) {
        assert_true(lmmc_test_nearly_equal(fixture->empty_operations_y.data[i], 0.0, TEST_EPS_TIGHT));
    }

    check_empty_transpose(fixture, &fixture->empty_operations_zero_sparse);

    lmmc_vec_destroy(&fixture->empty_operations_y);
    lmmc_vec_destroy(&fixture->empty_operations_x);
    lmmc_sparse_destroy(&fixture->empty_operations_zero_sparse);
    lmmc_mat_destroy(&fixture->empty_operations_zero_dense);
}

static void test_coo_duplicates(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_coo_create(3, 3, 8, &fixture->coo_duplicates_coo);
    assert_true(st == LMMC_STATUS_OK);

    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo_duplicates_coo, 0, 0, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo_duplicates_coo, 0, 0, 2.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo_duplicates_coo, 0, 0, 3.0), LMMC_STATUS_OK);

    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo_duplicates_coo, 1, 1, 5.0), LMMC_STATUS_OK);

    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo_duplicates_coo, 2, 2, 4.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo_duplicates_coo, 2, 2, 6.0), LMMC_STATUS_OK);

    st = lmmc_sparse_coo_to_csr(&fixture->coo_duplicates_coo, &fixture->coo_duplicates_sparse);
    lmmc_sparse_coo_destroy(&fixture->coo_duplicates_coo);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(fixture->coo_duplicates_sparse.nnz == 3);

    st = lmmc_mat_create(3, 3, &fixture->coo_duplicates_dense);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_to_dense(&fixture->coo_duplicates_sparse, &fixture->coo_duplicates_dense);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(((lmmc_test_nearly_equal(fixture->coo_duplicates_dense.data[0], 6.0, TEST_EPS_TIGHT)) && (lmmc_test_nearly_equal(fixture->coo_duplicates_dense.data[4], 5.0, TEST_EPS_TIGHT))) && (lmmc_test_nearly_equal(fixture->coo_duplicates_dense.data[8], 10.0, TEST_EPS_TIGHT)));
    lmmc_mat_destroy(&fixture->coo_duplicates_dense);
    lmmc_sparse_destroy(&fixture->coo_duplicates_sparse);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_addition, setup, teardown),
        cmocka_unit_test_setup_teardown(test_empty_operations, setup, teardown),
        cmocka_unit_test_setup_teardown(test_coo_duplicates, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
