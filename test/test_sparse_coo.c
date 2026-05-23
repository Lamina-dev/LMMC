/**
 * @file test_sparse_coo.c
 * @brief 针对 LMMC 中 sparse coo 相关接口的单元测试。
 *
 * @internal
 */
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "lmmc/lmmc.h"

static int test_coo_create(void) {
    lmmc_sparse_coo_t coo = {0};
    lmmc_status_t st;

    printf("  test_coo_create...\n");


    st = lmmc_sparse_coo_create(3, 4, 8, &coo);
    assert(st == LMMC_STATUS_OK);
    assert(coo.rows == 3);
    assert(coo.cols == 4);
    assert(coo.nnz == 0);
    assert(coo.capacity == 8);
    lmmc_sparse_coo_destroy(&coo);


    st = lmmc_sparse_coo_create(5, 5, 0, &coo);
    assert(st == LMMC_STATUS_OK);
    assert(coo.capacity == 16);
    lmmc_sparse_coo_destroy(&coo);


    st = lmmc_sparse_coo_create(3, 3, 4, NULL);
    assert(st == LMMC_STATUS_INVALID_ARGUMENT);


    st = lmmc_sparse_coo_create(0, 3, 4, &coo);
    assert(st == LMMC_STATUS_INVALID_ARGUMENT);


    st = lmmc_sparse_coo_create(3, 0, 4, &coo);
    assert(st == LMMC_STATUS_INVALID_ARGUMENT);

    printf("  test_coo_create PASSED\n");
    return 0;
}

static int test_coo_add_entry(void) {
    lmmc_sparse_coo_t coo = {0};
    lmmc_status_t st;

    printf("  test_coo_add_entry...\n");

    st = lmmc_sparse_coo_create(3, 3, 2, &coo);
    assert(st == LMMC_STATUS_OK);


    st = lmmc_sparse_coo_add_entry(&coo, 0, 0, 1.0);
    assert(st == LMMC_STATUS_OK);
    assert(coo.nnz == 1);

    st = lmmc_sparse_coo_add_entry(&coo, 1, 2, 3.5);
    assert(st == LMMC_STATUS_OK);
    assert(coo.nnz == 2);
    assert(coo.capacity == 2);


    st = lmmc_sparse_coo_add_entry(&coo, 2, 1, 7.0);
    assert(st == LMMC_STATUS_OK);
    assert(coo.nnz == 3);
    assert(coo.capacity == 4);


    assert(coo.row_idx[0] == 0 && coo.col_idx[0] == 0);
    assert(coo.values[0] == 1.0);
    assert(coo.row_idx[1] == 1 && coo.col_idx[1] == 2);
    assert(coo.values[1] == 3.5);
    assert(coo.row_idx[2] == 2 && coo.col_idx[2] == 1);
    assert(coo.values[2] == 7.0);


    st = lmmc_sparse_coo_add_entry(&coo, 3, 0, 1.0);
    assert(st == LMMC_STATUS_INDEX_OUT_OF_BOUNDS);


    st = lmmc_sparse_coo_add_entry(&coo, 0, 3, 1.0);
    assert(st == LMMC_STATUS_INDEX_OUT_OF_BOUNDS);


    st = lmmc_sparse_coo_add_entry(NULL, 0, 0, 1.0);
    assert(st == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_sparse_coo_destroy(&coo);
    printf("  test_coo_add_entry PASSED\n");
    return 0;
}

static int test_coo_to_csr_basic(void) {
    lmmc_sparse_coo_t coo = {0};
    lmmc_sparse_mat_t csr = {0};
    lmmc_status_t st;

    printf("  test_coo_to_csr_basic...\n");


    st = lmmc_sparse_coo_create(3, 3, 8, &coo);
    assert(st == LMMC_STATUS_OK);


    lmmc_sparse_coo_add_entry(&coo, 2, 2, 5.0);
    lmmc_sparse_coo_add_entry(&coo, 0, 0, 1.0);
    lmmc_sparse_coo_add_entry(&coo, 1, 1, 3.0);
    lmmc_sparse_coo_add_entry(&coo, 0, 2, 2.0);
    lmmc_sparse_coo_add_entry(&coo, 2, 0, 4.0);

    st = lmmc_sparse_coo_to_csr(&coo, &csr);
    assert(st == LMMC_STATUS_OK);
    assert(csr.rows == 3);
    assert(csr.cols == 3);
    assert(csr.nnz == 5);
    assert(csr.format == LMMC_SPARSE_CSR);


    assert(csr.row_ptr[0] == 0);
    assert(csr.row_ptr[1] == 2);
    assert(csr.row_ptr[2] == 3);
    assert(csr.row_ptr[3] == 5);


    assert(csr.col_idx[0] == 0 && csr.values[0] == 1.0);
    assert(csr.col_idx[1] == 2 && csr.values[1] == 2.0);

    assert(csr.col_idx[2] == 1 && csr.values[2] == 3.0);

    assert(csr.col_idx[3] == 0 && csr.values[3] == 4.0);
    assert(csr.col_idx[4] == 2 && csr.values[4] == 5.0);

    lmmc_sparse_coo_destroy(&coo);
    lmmc_sparse_destroy(&csr);
    printf("  test_coo_to_csr_basic PASSED\n");
    return 0;
}

static int test_coo_to_csr_duplicates(void) {
    lmmc_sparse_coo_t coo = {0};
    lmmc_sparse_mat_t csr = {0};
    lmmc_status_t st;

    printf("  test_coo_to_csr_duplicates...\n");


    st = lmmc_sparse_coo_create(3, 3, 8, &coo);
    assert(st == LMMC_STATUS_OK);

    lmmc_sparse_coo_add_entry(&coo, 0, 0, 1.0);
    lmmc_sparse_coo_add_entry(&coo, 0, 0, 2.0);
    lmmc_sparse_coo_add_entry(&coo, 0, 0, 3.0);
    lmmc_sparse_coo_add_entry(&coo, 1, 1, 5.0);

    st = lmmc_sparse_coo_to_csr(&coo, &csr);
    assert(st == LMMC_STATUS_OK);
    assert(csr.nnz == 2);


    assert(csr.row_ptr[0] == 0);
    assert(csr.row_ptr[1] == 1);
    assert(csr.col_idx[0] == 0);
    assert(fabs(csr.values[0] - 6.0) < 1e-15);


    assert(csr.row_ptr[2] == 2);
    assert(csr.col_idx[1] == 1);
    assert(csr.values[1] == 5.0);


    assert(csr.row_ptr[3] == 2);

    lmmc_sparse_coo_destroy(&coo);
    lmmc_sparse_destroy(&csr);
    printf("  test_coo_to_csr_duplicates PASSED\n");
    return 0;
}

static int test_coo_to_csr_empty(void) {
    lmmc_sparse_coo_t coo = {0};
    lmmc_sparse_mat_t csr = {0};
    lmmc_status_t st;

    printf("  test_coo_to_csr_empty...\n");

    st = lmmc_sparse_coo_create(3, 3, 4, &coo);
    assert(st == LMMC_STATUS_OK);


    st = lmmc_sparse_coo_to_csr(&coo, &csr);
    assert(st == LMMC_STATUS_OK);
    assert(csr.nnz == 0);
    assert(csr.rows == 3);
    assert(csr.cols == 3);

    lmmc_sparse_coo_destroy(&coo);
    lmmc_sparse_destroy(&csr);
    printf("  test_coo_to_csr_empty PASSED\n");
    return 0;
}

static int test_coo_to_csc_basic(void) {
    lmmc_sparse_coo_t coo = {0};
    lmmc_sparse_mat_t csc = {0};
    lmmc_status_t st;

    printf("  test_coo_to_csc_basic...\n");


    st = lmmc_sparse_coo_create(3, 3, 8, &coo);
    assert(st == LMMC_STATUS_OK);

    lmmc_sparse_coo_add_entry(&coo, 2, 2, 5.0);
    lmmc_sparse_coo_add_entry(&coo, 0, 0, 1.0);
    lmmc_sparse_coo_add_entry(&coo, 1, 1, 3.0);
    lmmc_sparse_coo_add_entry(&coo, 0, 2, 2.0);
    lmmc_sparse_coo_add_entry(&coo, 2, 0, 4.0);

    st = lmmc_sparse_coo_to_csc(&coo, &csc);
    assert(st == LMMC_STATUS_OK);
    assert(csc.rows == 3);
    assert(csc.cols == 3);
    assert(csc.nnz == 5);
    assert(csc.format == LMMC_SPARSE_CSC);


    assert(csc.row_ptr[0] == 0);
    assert(csc.row_ptr[1] == 2);
    assert(csc.row_ptr[2] == 3);
    assert(csc.row_ptr[3] == 5);


    assert(csc.col_idx[0] == 0 && csc.values[0] == 1.0);
    assert(csc.col_idx[1] == 2 && csc.values[1] == 4.0);

    assert(csc.col_idx[2] == 1 && csc.values[2] == 3.0);

    assert(csc.col_idx[3] == 0 && csc.values[3] == 2.0);
    assert(csc.col_idx[4] == 2 && csc.values[4] == 5.0);

    lmmc_sparse_coo_destroy(&coo);
    lmmc_sparse_destroy(&csc);
    printf("  test_coo_to_csc_basic PASSED\n");
    return 0;
}

static int test_coo_to_csc_duplicates(void) {
    lmmc_sparse_coo_t coo = {0};
    lmmc_sparse_mat_t csc = {0};
    lmmc_status_t st;

    printf("  test_coo_to_csc_duplicates...\n");

    st = lmmc_sparse_coo_create(3, 3, 8, &coo);
    assert(st == LMMC_STATUS_OK);


    lmmc_sparse_coo_add_entry(&coo, 1, 2, 2.0);
    lmmc_sparse_coo_add_entry(&coo, 1, 2, 3.0);
    lmmc_sparse_coo_add_entry(&coo, 0, 0, 1.0);

    st = lmmc_sparse_coo_to_csc(&coo, &csc);
    assert(st == LMMC_STATUS_OK);
    assert(csc.nnz == 2);


    assert(csc.row_ptr[0] == 0);
    assert(csc.row_ptr[1] == 1);
    assert(csc.col_idx[0] == 0);
    assert(csc.values[0] == 1.0);


    assert(csc.row_ptr[2] == 1);


    assert(csc.row_ptr[3] == 2);
    assert(csc.col_idx[1] == 1);
    assert(fabs(csc.values[1] - 5.0) < 1e-15);

    lmmc_sparse_coo_destroy(&coo);
    lmmc_sparse_destroy(&csc);
    printf("  test_coo_to_csc_duplicates PASSED\n");
    return 0;
}

static int test_coo_destroy_null(void) {
    printf("  test_coo_destroy_null...\n");

    lmmc_sparse_coo_destroy(NULL);
    printf("  test_coo_destroy_null PASSED\n");
    return 0;
}

int main(void) {
    int failures = 0;

    printf("=== Sparse COO Tests ===\n");

    failures += test_coo_create();
    failures += test_coo_add_entry();
    failures += test_coo_to_csr_basic();
    failures += test_coo_to_csr_duplicates();
    failures += test_coo_to_csr_empty();
    failures += test_coo_to_csc_basic();
    failures += test_coo_to_csc_duplicates();
    failures += test_coo_destroy_null();

    if (failures == 0) {
        printf("\nAll Sparse COO Tests PASSED!\n");
    } else {
        printf("\n%d test(s) FAILED!\n", failures);
    }
    return failures;
}
