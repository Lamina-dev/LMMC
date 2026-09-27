/**
 * @file test_sparse_coo.c
 * 针对 LMMC 中 sparse coo 相关接口的单元测试。
 */
#include <stdio.h>
#include "test_common.h"
#include <math.h>
#include "lmmc/lmmc.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_sparse_coo_t coo;
    lmmc_sparse_mat_t sparse;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    assert_non_null(fixture);
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_destroy(&fixture->sparse);
    lmmc_sparse_coo_destroy(&fixture->coo);
    free(fixture);
    *state = NULL;
    return 0;
}

static void test_coo_create(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_coo_create(3, 4, 8, &fixture->coo);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->coo.rows == 3);
    assert_true(fixture->coo.cols == 4);
    assert_true(fixture->coo.nnz == 0);
    lmmc_sparse_coo_destroy(&fixture->coo);

    st = lmmc_sparse_coo_create(5, 5, 0, &fixture->coo);
    assert_true(st == LMMC_STATUS_OK);
    lmmc_sparse_coo_destroy(&fixture->coo);

    st = lmmc_sparse_coo_create(3, 3, 4, NULL);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_sparse_coo_create(0, 3, 4, &fixture->coo);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_sparse_coo_create(3, 0, 4, &fixture->coo);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_coo_add_entry(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_coo_create(3, 3, 2, &fixture->coo);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_coo_add_entry(&fixture->coo, 0, 0, 1.0);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->coo.nnz == 1);

    st = lmmc_sparse_coo_add_entry(&fixture->coo, 1, 2, 3.5);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->coo.nnz == 2);

    st = lmmc_sparse_coo_add_entry(&fixture->coo, 2, 1, 7.0);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->coo.nnz == 3);

    assert_true(fixture->coo.row_idx[0] == 0 && fixture->coo.col_idx[0] == 0);
    assert_true(fixture->coo.values[0] == 1.0);
    assert_true(fixture->coo.row_idx[1] == 1 && fixture->coo.col_idx[1] == 2);
    assert_true(fixture->coo.values[1] == 3.5);
    assert_true(fixture->coo.row_idx[2] == 2 && fixture->coo.col_idx[2] == 1);
    assert_true(fixture->coo.values[2] == 7.0);

    st = lmmc_sparse_coo_add_entry(&fixture->coo, 3, 0, 1.0);
    assert_true(st == LMMC_STATUS_INDEX_OUT_OF_BOUNDS);

    st = lmmc_sparse_coo_add_entry(&fixture->coo, 0, 3, 1.0);
    assert_true(st == LMMC_STATUS_INDEX_OUT_OF_BOUNDS);

    st = lmmc_sparse_coo_add_entry(NULL, 0, 0, 1.0);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_coo_to_csr_basic(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_coo_create(3, 3, 8, &fixture->coo);
    assert_true(st == LMMC_STATUS_OK);

    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 2, 2, 5.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 0, 0, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 1, 1, 3.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 0, 2, 2.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 2, 0, 4.0), LMMC_STATUS_OK);

    st = lmmc_sparse_coo_to_csr(&fixture->coo, &fixture->sparse);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->sparse.rows == 3);
    assert_true(fixture->sparse.cols == 3);
    assert_true(fixture->sparse.nnz == 5);
    assert_true(fixture->sparse.format == LMMC_SPARSE_CSR);

    assert_true(fixture->sparse.row_ptr[0] == 0);
    assert_true(fixture->sparse.row_ptr[1] == 2);
    assert_true(fixture->sparse.row_ptr[2] == 3);
    assert_true(fixture->sparse.row_ptr[3] == 5);

    assert_true(fixture->sparse.col_idx[0] == 0 && fixture->sparse.values[0] == 1.0);
    assert_true(fixture->sparse.col_idx[1] == 2 && fixture->sparse.values[1] == 2.0);

    assert_true(fixture->sparse.col_idx[2] == 1 && fixture->sparse.values[2] == 3.0);

    assert_true(fixture->sparse.col_idx[3] == 0 && fixture->sparse.values[3] == 4.0);
    assert_true(fixture->sparse.col_idx[4] == 2 && fixture->sparse.values[4] == 5.0);
}

static void test_coo_to_csr_duplicates(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_coo_create(3, 3, 8, &fixture->coo);
    assert_true(st == LMMC_STATUS_OK);

    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 0, 0, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 0, 0, 2.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 0, 0, 3.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 1, 1, 5.0), LMMC_STATUS_OK);

    st = lmmc_sparse_coo_to_csr(&fixture->coo, &fixture->sparse);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->sparse.nnz == 2);

    assert_true(fixture->sparse.row_ptr[0] == 0);
    assert_true(fixture->sparse.row_ptr[1] == 1);
    assert_true(fixture->sparse.col_idx[0] == 0);
    assert_true(fabs(fixture->sparse.values[0] - 6.0) < 1e-15);

    assert_true(fixture->sparse.row_ptr[2] == 2);
    assert_true(fixture->sparse.col_idx[1] == 1);
    assert_true(fixture->sparse.values[1] == 5.0);

    assert_true(fixture->sparse.row_ptr[3] == 2);
}

static void test_coo_to_csr_empty(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_coo_create(3, 3, 4, &fixture->coo);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_coo_to_csr(&fixture->coo, &fixture->sparse);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->sparse.nnz == 0);
    assert_true(fixture->sparse.rows == 3);
    assert_true(fixture->sparse.cols == 3);
}

static void test_coo_to_csc_basic(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_coo_create(3, 3, 8, &fixture->coo);
    assert_true(st == LMMC_STATUS_OK);

    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 2, 2, 5.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 0, 0, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 1, 1, 3.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 0, 2, 2.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 2, 0, 4.0), LMMC_STATUS_OK);

    st = lmmc_sparse_coo_to_csc(&fixture->coo, &fixture->sparse);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->sparse.rows == 3);
    assert_true(fixture->sparse.cols == 3);
    assert_true(fixture->sparse.nnz == 5);
    assert_true(fixture->sparse.format == LMMC_SPARSE_CSC);

    assert_true(fixture->sparse.row_ptr[0] == 0);
    assert_true(fixture->sparse.row_ptr[1] == 2);
    assert_true(fixture->sparse.row_ptr[2] == 3);
    assert_true(fixture->sparse.row_ptr[3] == 5);

    assert_true(fixture->sparse.col_idx[0] == 0 && fixture->sparse.values[0] == 1.0);
    assert_true(fixture->sparse.col_idx[1] == 2 && fixture->sparse.values[1] == 4.0);

    assert_true(fixture->sparse.col_idx[2] == 1 && fixture->sparse.values[2] == 3.0);

    assert_true(fixture->sparse.col_idx[3] == 0 && fixture->sparse.values[3] == 2.0);
    assert_true(fixture->sparse.col_idx[4] == 2 && fixture->sparse.values[4] == 5.0);
}

static void test_coo_to_csc_duplicates(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_coo_create(3, 3, 8, &fixture->coo);
    assert_true(st == LMMC_STATUS_OK);

    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 1, 2, 2.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 1, 2, 3.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo, 0, 0, 1.0), LMMC_STATUS_OK);

    st = lmmc_sparse_coo_to_csc(&fixture->coo, &fixture->sparse);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->sparse.nnz == 2);

    assert_true(fixture->sparse.row_ptr[0] == 0);
    assert_true(fixture->sparse.row_ptr[1] == 1);
    assert_true(fixture->sparse.col_idx[0] == 0);
    assert_true(fixture->sparse.values[0] == 1.0);

    assert_true(fixture->sparse.row_ptr[2] == 1);

    assert_true(fixture->sparse.row_ptr[3] == 2);
    assert_true(fixture->sparse.col_idx[1] == 1);
    assert_true(fabs(fixture->sparse.values[1] - 5.0) < 1e-15);
}

static void reject_invalid_coo_descriptors(const lmmc_sparse_coo_t *coo,
                                           lmmc_status_t (*convert)(const lmmc_sparse_coo_t *, lmmc_sparse_mat_t *),
                                           lmmc_sparse_mat_t *a) {

    for (int damage = 0; damage < 8; ++damage) {
        lmmc_sparse_coo_t bad = *coo;
        size_t bad_rows[] = {2, 0, 1, 1};
        size_t bad_cols[] = {3, 1, 2, 2};
        switch (damage) {
        case 0: {
            bad.rows = 0;
            break;
        }
        case 1: {
            bad.cols = 0;
            break;
        }
        case 2: {
            bad.capacity = 3;
            break;
        }
        case 3: {
            bad.row_idx = NULL;
            break;
        }
        case 4: {
            bad.col_idx = NULL;
            break;
        }
        case 5: {
            bad.values = NULL;
            break;
        }
        case 6: {
            bad.row_idx = bad_rows;
            break;
        }
        case 7: {
            bad.col_idx = bad_cols;
            break;
        }
        }
        a->rows = 17;
        assert_true(convert(&bad, a) == LMMC_STATUS_INVALID_ARGUMENT);
        assert_true(a->rows == 17 && a->row_ptr == NULL);
    }
}

static void test_coo_stable_sum_and_invalid_descriptors(void **state) {
    struct test_fixture *fixture = *state;

    size_t rows[] = {1, 0, 1, 1};
    size_t cols[] = {2, 1, 2, 2};
    double values[] = {0x1p54, 4, -0x1p54, 1};
    lmmc_sparse_coo_t coo = {0};
    coo.rows = 2;
    coo.cols = 3;
    coo.nnz = 4;
    coo.capacity = 4;
    coo.row_idx = rows;
    coo.col_idx = cols;
    coo.values = values;
    for (int format = 0; format < 2; ++format) {

        lmmc_status_t (*convert)(const lmmc_sparse_coo_t *, lmmc_sparse_mat_t *) =
            format ? lmmc_sparse_coo_to_csc : lmmc_sparse_coo_to_csr;
        assert_true(convert(&coo, &fixture->sparse) == LMMC_STATUS_OK);
        assert_true(fixture->sparse.nnz == 2 && fixture->sparse.values[0] == 4 && fixture->sparse.values[1] == 1);
        lmmc_sparse_destroy(&fixture->sparse);
        reject_invalid_coo_descriptors(&coo, convert, &fixture->sparse);
        lmmc_sparse_coo_t empty = {0};
        empty.rows = 2;
        empty.cols = 3;
        assert_true(convert(&empty, &fixture->sparse) == LMMC_STATUS_OK);
        assert_true(fixture->sparse.nnz == 0 && fixture->sparse.row_ptr[format ? 3 : 2] == 0);
        lmmc_sparse_destroy(&fixture->sparse);
    }
    assert_true(values[0] == 0x1p54 && values[2] == -0x1p54 && rows[0] == 1);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_coo_stable_sum_and_invalid_descriptors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_coo_create, setup, teardown),
        cmocka_unit_test_setup_teardown(test_coo_add_entry, setup, teardown),
        cmocka_unit_test_setup_teardown(test_coo_to_csr_basic, setup, teardown),
        cmocka_unit_test_setup_teardown(test_coo_to_csr_duplicates, setup, teardown),
        cmocka_unit_test_setup_teardown(test_coo_to_csr_empty, setup, teardown),
        cmocka_unit_test_setup_teardown(test_coo_to_csc_basic, setup, teardown),
        cmocka_unit_test_setup_teardown(test_coo_to_csc_duplicates, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
