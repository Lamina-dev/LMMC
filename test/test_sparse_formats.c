/**
 * @file test_sparse_formats.c
 * @brief BSR 格式和对称半存储 CSR 格式的单元测试。
 */
#include <stdio.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static int test_bsr_roundtrip(void) {
    /* 创建一个 4x4 块带状稠密矩阵，block_size=2 */
    lmmc_mat_t dense = {0}, recovered = {0};
    lmmc_sparse_bsr_t bsr = {0};
    lmmc_status_t st;
    size_t i, j;

    st = lmmc_mat_create(4, 4, &dense);
    if (st != LMMC_STATUS_OK) return 1;

    /* 填充：块 (0,0) 和 (1,1) 非零，块 (0,1) 和 (1,0) 为零 */
    lmmc_real_t zero = 0.0;
    lmmc_mat_fill(&dense, zero);

    /* 块 (0,0): rows 0-1, cols 0-1 */
    dense.data[0 * dense.stride + 0] = 1.0;
    dense.data[0 * dense.stride + 1] = 2.0;
    dense.data[1 * dense.stride + 0] = 3.0;
    dense.data[1 * dense.stride + 1] = 4.0;

    /* 块 (1,1): rows 2-3, cols 2-3 */
    dense.data[2 * dense.stride + 2] = 5.0;
    dense.data[2 * dense.stride + 3] = 6.0;
    dense.data[3 * dense.stride + 2] = 7.0;
    dense.data[3 * dense.stride + 3] = 8.0;

    /* 转换为 BSR */
    lmmc_real_t eps = 0.0;
    st = lmmc_sparse_dense_to_bsr(&dense, 2, eps, &bsr);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL: dense_to_bsr returned %d\n", (int)st);
        lmmc_mat_destroy(&dense);
        return 1;
    }

    /* 验证 BSR 结构 */
    if (bsr.rows != 2 || bsr.cols != 2 || bsr.block_size != 2 || bsr.nnz_blocks != 2) {
        printf("FAIL: BSR structure incorrect: rows=%zu cols=%zu bs=%zu nnz=%zu\n",
               bsr.rows, bsr.cols, bsr.block_size, bsr.nnz_blocks);
        lmmc_sparse_bsr_destroy(&bsr);
        lmmc_mat_destroy(&dense);
        return 1;
    }

    /* 转换回稠密 */
    st = lmmc_mat_create(4, 4, &recovered);
    if (st != LMMC_STATUS_OK) {
        lmmc_sparse_bsr_destroy(&bsr);
        lmmc_mat_destroy(&dense);
        return 1;
    }

    st = lmmc_sparse_bsr_to_dense(&bsr, &recovered);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL: bsr_to_dense returned %d\n", (int)st);
        lmmc_mat_destroy(&recovered);
        lmmc_sparse_bsr_destroy(&bsr);
        lmmc_mat_destroy(&dense);
        return 1;
    }

    /* 验证 round-trip */
    for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) {
            lmmc_real_t orig = dense.data[i * dense.stride + j];
            lmmc_real_t rec = recovered.data[i * recovered.stride + j];
            if (!lmmc_test_nearly_equal(orig, rec, 1e-15)) {
                printf("FAIL: BSR roundtrip mismatch at (%zu,%zu): %g vs %g\n",
                       i, j, orig, rec);
                lmmc_mat_destroy(&recovered);
                lmmc_sparse_bsr_destroy(&bsr);
                lmmc_mat_destroy(&dense);
                return 1;
            }
        }
    }

    lmmc_mat_destroy(&recovered);
    lmmc_sparse_bsr_destroy(&bsr);
    lmmc_mat_destroy(&dense);
    printf("PASS: test_bsr_roundtrip\n");
    return 0;
}

static int test_sym_csr_spmv(void) {
    /* 创建一个 3x3 对称矩阵:
     * A = [4 1 2]
     *     [1 3 0]
     *     [2 0 5]
     */
    lmmc_sparse_mat_t full_csr = {0};
    lmmc_sparse_sym_csr_t sym = {0};
    lmmc_vec_t x = {0}, y_full = {0}, y_sym = {0};
    lmmc_status_t st;
    size_t i;

    /* 构建完整 CSR：9 个非零元（含零元也存储以简化） */
    /* 实际非零元：(0,0)=4, (0,1)=1, (0,2)=2, (1,0)=1, (1,1)=3, (2,0)=2, (2,2)=5 */
    st = lmmc_sparse_create_csr(3, 3, 7, &full_csr);
    if (st != LMMC_STATUS_OK) return 1;

    /* Row 0: cols 0,1,2 */
    full_csr.row_ptr[0] = 0;
    full_csr.col_idx[0] = 0; full_csr.values[0] = 4.0;
    full_csr.col_idx[1] = 1; full_csr.values[1] = 1.0;
    full_csr.col_idx[2] = 2; full_csr.values[2] = 2.0;
    /* Row 1: cols 0,1 */
    full_csr.row_ptr[1] = 3;
    full_csr.col_idx[3] = 0; full_csr.values[3] = 1.0;
    full_csr.col_idx[4] = 1; full_csr.values[4] = 3.0;
    /* Row 2: cols 0,2 */
    full_csr.row_ptr[2] = 5;
    full_csr.col_idx[5] = 0; full_csr.values[5] = 2.0;
    full_csr.col_idx[6] = 2; full_csr.values[6] = 5.0;
    full_csr.row_ptr[3] = 7;

    /* 创建向量 x = [1, 2, 3] */
    st = lmmc_vec_create(3, &x);
    if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&full_csr); return 1; }
    x.data[0] = 1.0; x.data[1] = 2.0; x.data[2] = 3.0;

    /* 完整 CSR SpMV */
    st = lmmc_vec_create(3, &y_full);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr); return 1; }
    st = lmmc_sparse_mat_vec_mul(&full_csr, &x, &y_full);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL: full CSR SpMV returned %d\n", (int)st);
        lmmc_vec_destroy(&y_full); lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr);
        return 1;
    }

    /* 转换为对称半存储（上三角） */
    st = lmmc_sparse_sym_csr_from_csr(&full_csr, LMMC_SPARSE_SYM_UPPER, &sym);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL: sym_csr_from_csr returned %d\n", (int)st);
        lmmc_vec_destroy(&y_full); lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr);
        return 1;
    }

    /* 对称 SpMV */
    st = lmmc_vec_create(3, &y_sym);
    if (st != LMMC_STATUS_OK) {
        lmmc_sparse_sym_csr_destroy(&sym);
        lmmc_vec_destroy(&y_full); lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr);
        return 1;
    }
    st = lmmc_sparse_sym_spmv(&sym, &x, &y_sym);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL: sym_spmv returned %d\n", (int)st);
        lmmc_vec_destroy(&y_sym); lmmc_sparse_sym_csr_destroy(&sym);
        lmmc_vec_destroy(&y_full); lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr);
        return 1;
    }

    /* 比较结果 */
    for (i = 0; i < 3; ++i) {
        if (!lmmc_test_nearly_equal(y_full.data[i], y_sym.data[i], 1e-12)) {
            printf("FAIL: sym SpMV mismatch at [%zu]: full=%g sym=%g\n",
                   i, y_full.data[i], y_sym.data[i]);
            lmmc_vec_destroy(&y_sym); lmmc_sparse_sym_csr_destroy(&sym);
            lmmc_vec_destroy(&y_full); lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr);
            return 1;
        }
    }

    lmmc_vec_destroy(&y_sym);
    lmmc_sparse_sym_csr_destroy(&sym);
    lmmc_vec_destroy(&y_full);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&full_csr);
    printf("PASS: test_sym_csr_spmv\n");
    return 0;
}

static int test_sym_csr_lower(void) {
    /* 同样的 3x3 对称矩阵，但使用下三角存储 */
    lmmc_sparse_mat_t full_csr = {0};
    lmmc_sparse_sym_csr_t sym = {0};
    lmmc_vec_t x = {0}, y_full = {0}, y_sym = {0};
    lmmc_status_t st;
    size_t i;

    st = lmmc_sparse_create_csr(3, 3, 7, &full_csr);
    if (st != LMMC_STATUS_OK) return 1;

    full_csr.row_ptr[0] = 0;
    full_csr.col_idx[0] = 0; full_csr.values[0] = 4.0;
    full_csr.col_idx[1] = 1; full_csr.values[1] = 1.0;
    full_csr.col_idx[2] = 2; full_csr.values[2] = 2.0;
    full_csr.row_ptr[1] = 3;
    full_csr.col_idx[3] = 0; full_csr.values[3] = 1.0;
    full_csr.col_idx[4] = 1; full_csr.values[4] = 3.0;
    full_csr.row_ptr[2] = 5;
    full_csr.col_idx[5] = 0; full_csr.values[5] = 2.0;
    full_csr.col_idx[6] = 2; full_csr.values[6] = 5.0;
    full_csr.row_ptr[3] = 7;

    st = lmmc_vec_create(3, &x);
    if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&full_csr); return 1; }
    x.data[0] = 1.0; x.data[1] = 2.0; x.data[2] = 3.0;

    st = lmmc_vec_create(3, &y_full);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr); return 1; }
    st = lmmc_sparse_mat_vec_mul(&full_csr, &x, &y_full);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&y_full); lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr);
        return 1;
    }

    /* 下三角存储 */
    st = lmmc_sparse_sym_csr_from_csr(&full_csr, LMMC_SPARSE_SYM_LOWER, &sym);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL: sym_csr_from_csr (lower) returned %d\n", (int)st);
        lmmc_vec_destroy(&y_full); lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr);
        return 1;
    }

    st = lmmc_vec_create(3, &y_sym);
    if (st != LMMC_STATUS_OK) {
        lmmc_sparse_sym_csr_destroy(&sym);
        lmmc_vec_destroy(&y_full); lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr);
        return 1;
    }
    st = lmmc_sparse_sym_spmv(&sym, &x, &y_sym);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL: sym_spmv (lower) returned %d\n", (int)st);
        lmmc_vec_destroy(&y_sym); lmmc_sparse_sym_csr_destroy(&sym);
        lmmc_vec_destroy(&y_full); lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr);
        return 1;
    }

    for (i = 0; i < 3; ++i) {
        if (!lmmc_test_nearly_equal(y_full.data[i], y_sym.data[i], 1e-12)) {
            printf("FAIL: sym SpMV (lower) mismatch at [%zu]: full=%g sym=%g\n",
                   i, y_full.data[i], y_sym.data[i]);
            lmmc_vec_destroy(&y_sym); lmmc_sparse_sym_csr_destroy(&sym);
            lmmc_vec_destroy(&y_full); lmmc_vec_destroy(&x); lmmc_sparse_destroy(&full_csr);
            return 1;
        }
    }

    lmmc_vec_destroy(&y_sym);
    lmmc_sparse_sym_csr_destroy(&sym);
    lmmc_vec_destroy(&y_full);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&full_csr);
    printf("PASS: test_sym_csr_lower\n");
    return 0;
}

static int test_bsr_error_cases(void) {
    lmmc_sparse_bsr_t bsr = {0};
    lmmc_mat_t dense = {0};
    lmmc_status_t st;
    lmmc_real_t eps = 0.0;

    /* NULL output */
    st = lmmc_sparse_bsr_create(2, 2, 2, 1, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: bsr_create NULL should fail\n");
        return 1;
    }

    /* Zero block_size */
    st = lmmc_sparse_bsr_create(2, 2, 0, 1, &bsr);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: bsr_create zero block_size should fail\n");
        return 1;
    }

    /* Dense dimensions not multiple of block_size */
    st = lmmc_mat_create(5, 5, &dense);
    if (st != LMMC_STATUS_OK) return 1;
    st = lmmc_sparse_dense_to_bsr(&dense, 2, eps, &bsr);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: dense_to_bsr non-multiple should fail\n");
        lmmc_mat_destroy(&dense);
        return 1;
    }
    lmmc_mat_destroy(&dense);

    printf("PASS: test_bsr_error_cases\n");
    return 0;
}

int main(void) {
    int failures = 0;

    printf("Starting sparse format tests (BSR + Symmetric CSR)...\n");

    failures += test_bsr_roundtrip();
    failures += test_sym_csr_spmv();
    failures += test_sym_csr_lower();
    failures += test_bsr_error_cases();

    if (failures > 0) {
        printf("\n%d test(s) FAILED\n", failures);
        return 1;
    }
    printf("\nAll sparse format tests PASSED\n");
    return 0;
}
