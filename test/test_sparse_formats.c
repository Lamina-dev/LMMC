/**
 * @file test_sparse_formats.c
 * BSR 格式和对称半存储 CSR 格式的单元测试。
 */
#include <stdio.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_mat_t dense;
    lmmc_mat_t recovered;
    lmmc_sparse_bsr_t bsr;
    lmmc_sparse_mat_t full_csr;
    lmmc_sparse_sym_csr_t sym;
    lmmc_vec_t x;
    lmmc_vec_t y_full;
    lmmc_vec_t y_sym;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    assert_non_null(fixture);
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->y_sym);
    lmmc_vec_destroy(&fixture->y_full);
    lmmc_vec_destroy(&fixture->x);
    lmmc_sparse_sym_csr_destroy(&fixture->sym);
    lmmc_sparse_destroy(&fixture->full_csr);
    lmmc_sparse_bsr_destroy(&fixture->bsr);
    lmmc_mat_destroy(&fixture->recovered);
    lmmc_mat_destroy(&fixture->dense);
    free(fixture);
    *state = NULL;
    return 0;
}

static void check_bsr_dense_roundtrip(lmmc_mat_t *dense, lmmc_mat_t *recovered, lmmc_sparse_bsr_t *bsr) {

    lmmc_status_t st;
    st = lmmc_mat_create(4, 4, recovered);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_bsr_to_dense(bsr, recovered);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            lmmc_real_t orig = dense->data[i * dense->stride + j];
            lmmc_real_t rec = recovered->data[i * recovered->stride + j];
            assert_true(lmmc_test_nearly_equal(orig, rec, 1e-15));
        }
    }
}

static void test_bsr_roundtrip(void **state) {
    struct test_fixture *fixture = *state;

    /* 创建一个 4x4 块带状稠密矩阵，block_size=2 */

    lmmc_status_t st;

    st = lmmc_mat_create(4, 4, &fixture->dense);
    assert_true(st == LMMC_STATUS_OK);

    /* 填充：块 (0,0) 和 (1,1) 非零，块 (0,1) 和 (1,0) 为零 */
    lmmc_real_t zero = 0.0;
    assert_int_equal(lmmc_mat_fill(&fixture->dense, zero), LMMC_STATUS_OK);

    /* 块 (0,0): rows 0-1, cols 0-1 */
    fixture->dense.data[0 * fixture->dense.stride + 0] = 1.0;
    fixture->dense.data[0 * fixture->dense.stride + 1] = 2.0;
    fixture->dense.data[1 * fixture->dense.stride + 0] = 3.0;
    fixture->dense.data[1 * fixture->dense.stride + 1] = 4.0;

    /* 块 (1,1): rows 2-3, cols 2-3 */
    fixture->dense.data[2 * fixture->dense.stride + 2] = 5.0;
    fixture->dense.data[2 * fixture->dense.stride + 3] = 6.0;
    fixture->dense.data[3 * fixture->dense.stride + 2] = 7.0;
    fixture->dense.data[3 * fixture->dense.stride + 3] = 8.0;

    /* 转换为 BSR */
    lmmc_real_t eps = 0.0;
    st = lmmc_sparse_dense_to_bsr(&fixture->dense, 2, eps, &fixture->bsr);
    assert_true(st == LMMC_STATUS_OK);

    /* 验证 BSR 结构 */
    assert_true((((fixture->bsr.rows == 2) && (fixture->bsr.cols == 2)) && (fixture->bsr.block_size == 2)) && (fixture->bsr.nnz_blocks == 2));

    check_bsr_dense_roundtrip(&fixture->dense, &fixture->recovered, &fixture->bsr);
}

static void create_upper_reference_matrix(lmmc_sparse_mat_t *full_csr) {

    lmmc_status_t st;
    st = lmmc_sparse_create_csr(3, 3, 7, full_csr);
    assert_true(st == LMMC_STATUS_OK);

    full_csr->row_ptr[0] = 0;
    full_csr->col_idx[0] = 0;
    full_csr->values[0] = 4.0;
    full_csr->col_idx[1] = 1;
    full_csr->values[1] = 1.0;
    full_csr->col_idx[2] = 2;
    full_csr->values[2] = 2.0;
    full_csr->row_ptr[1] = 3;
    full_csr->col_idx[3] = 0;
    full_csr->values[3] = 1.0;
    full_csr->col_idx[4] = 1;
    full_csr->values[4] = 3.0;
    full_csr->row_ptr[2] = 5;
    full_csr->col_idx[5] = 0;
    full_csr->values[5] = 2.0;
    full_csr->col_idx[6] = 2;
    full_csr->values[6] = 5.0;
    full_csr->row_ptr[3] = 7;
}

static void test_sym_csr_spmv(void **state) {
    struct test_fixture *fixture = *state;

    /* 创建一个 3x3 对称矩阵:
     * A = [4 1 2]
     *     [1 3 0]
     *     [2 0 5]
     */

    lmmc_status_t st;
    size_t i;

    create_upper_reference_matrix(&fixture->full_csr);

    /* 创建向量 x = [1, 2, 3] */
    st = lmmc_vec_create(3, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);
    fixture->x.data[0] = 1.0;
    fixture->x.data[1] = 2.0;
    fixture->x.data[2] = 3.0;

    /* 完整 CSR SpMV */
    st = lmmc_vec_create(3, &fixture->y_full);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_mat_vec_mul(&fixture->full_csr, &fixture->x, &fixture->y_full);
    assert_true(st == LMMC_STATUS_OK);

    /* 转换为对称半存储（上三角） */
    st = lmmc_sparse_sym_csr_from_csr(&fixture->full_csr, LMMC_SPARSE_SYM_UPPER, &fixture->sym);
    assert_true(st == LMMC_STATUS_OK);

    /* 对称 SpMV */
    st = lmmc_vec_create(3, &fixture->y_sym);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_sym_spmv(&fixture->sym, &fixture->x, &fixture->y_sym);
    assert_true(st == LMMC_STATUS_OK);

    /* 比较结果 */
    for (i = 0; i < 3; ++i) {
        assert_true(lmmc_test_nearly_equal(fixture->y_full.data[i], fixture->y_sym.data[i], 1e-12));
    }
}

static void create_lower_reference_matrix(lmmc_sparse_mat_t *full_csr) {

    lmmc_status_t st;
    st = lmmc_sparse_create_csr(3, 3, 7, full_csr);
    assert_true(st == LMMC_STATUS_OK);

    full_csr->row_ptr[0] = 0;
    full_csr->col_idx[0] = 0;
    full_csr->values[0] = 4.0;
    full_csr->col_idx[1] = 1;
    full_csr->values[1] = 1.0;
    full_csr->col_idx[2] = 2;
    full_csr->values[2] = 2.0;
    full_csr->row_ptr[1] = 3;
    full_csr->col_idx[3] = 0;
    full_csr->values[3] = 1.0;
    full_csr->col_idx[4] = 1;
    full_csr->values[4] = 3.0;
    full_csr->row_ptr[2] = 5;
    full_csr->col_idx[5] = 0;
    full_csr->values[5] = 2.0;
    full_csr->col_idx[6] = 2;
    full_csr->values[6] = 5.0;
    full_csr->row_ptr[3] = 7;
}

static void test_sym_csr_lower(void **state) {
    struct test_fixture *fixture = *state;

    /* 同样的 3x3 对称矩阵，但使用下三角存储 */

    lmmc_status_t st;
    size_t i;

    create_lower_reference_matrix(&fixture->full_csr);

    st = lmmc_vec_create(3, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);
    fixture->x.data[0] = 1.0;
    fixture->x.data[1] = 2.0;
    fixture->x.data[2] = 3.0;

    st = lmmc_vec_create(3, &fixture->y_full);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_mat_vec_mul(&fixture->full_csr, &fixture->x, &fixture->y_full);
    assert_true(st == LMMC_STATUS_OK);

    /* 下三角存储 */
    st = lmmc_sparse_sym_csr_from_csr(&fixture->full_csr, LMMC_SPARSE_SYM_LOWER, &fixture->sym);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_vec_create(3, &fixture->y_sym);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_sym_spmv(&fixture->sym, &fixture->x, &fixture->y_sym);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < 3; ++i) {
        assert_true(lmmc_test_nearly_equal(fixture->y_full.data[i], fixture->y_sym.data[i], 1e-12));
    }
}

static void test_bsr_error_cases(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    lmmc_real_t eps = 0.0;

    /* NULL output */
    st = lmmc_sparse_bsr_create(2, 2, 2, 1, NULL);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    /* Zero block_size */
    st = lmmc_sparse_bsr_create(2, 2, 0, 1, &fixture->bsr);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    /* Dense dimensions not multiple of block_size */
    st = lmmc_mat_create(5, 5, &fixture->dense);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_dense_to_bsr(&fixture->dense, 2, eps, &fixture->bsr);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_bsr_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_sym_csr_spmv, setup, teardown),
        cmocka_unit_test_setup_teardown(test_sym_csr_lower, setup, teardown),
        cmocka_unit_test_setup_teardown(test_bsr_error_cases, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
