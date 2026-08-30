/**
 * @file test_sparse_cholesky.c
 * 针对 LMMC 中 sparse cholesky 相关接口的单元测试。
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static int test_count = 0;
static int pass_count = 0;

#define TEST_ASSERT(cond, msg) do { \
    test_count++; \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
    } else { \
        pass_count++; \
    } \
} while(0)


static lmmc_status_t build_sparse_spd(const lmmc_real_t* data, size_t n,
                                       lmmc_sparse_mat_t* out)
{
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(n, n, n * n, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t val = data[i * n + j];
            if (val != 0.0) {
                st = lmmc_sparse_builder_add(builder, i, j, val);
                if (st != LMMC_STATUS_OK) {
                    lmmc_sparse_builder_destroy(builder);
                    return st;
                }
            }
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}


static void test_basic_3x3(void)
{
    printf("Test: Basic 3x3 SPD matrix solve\n");


    lmmc_real_t A_data[] = {
         4.0,  12.0, -16.0,
        12.0,  37.0, -43.0,
       -16.0, -43.0,  98.0
    };

    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_sparse_spd(A_data, 3, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build sparse SPD matrix");


    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "symbolic analysis succeeds");


    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "numeric factorization succeeds");


    st = lmmc_vec_create(3, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b vector");
    st = lmmc_vec_create(3, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x vector");

    b.data[0] = -20.0;
    b.data[1] = -43.0;
    b.data[2] = 192.0;

    st = lmmc_sparse_chol_solve(chol, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "solve succeeds");

    TEST_ASSERT(lmmc_test_nearly_equal(x.data[0], 1.0, 1e-10), "x[0] == 1.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[1], 2.0, 1e-10), "x[1] == 2.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[2], 3.0, 1e-10), "x[2] == 3.0");


    lmmc_sparse_chol_destroy(chol);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}


static void test_identity_2x2(void)
{
    printf("Test: 2x2 identity matrix\n");

    lmmc_real_t A_data[] = {
        1.0, 0.0,
        0.0, 1.0
    };

    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_sparse_spd(A_data, 2, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build identity matrix");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "symbolic analysis");

    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "numeric factorization");

    st = lmmc_vec_create(2, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b");
    st = lmmc_vec_create(2, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x");

    b.data[0] = 3.0;
    b.data[1] = 7.0;

    st = lmmc_sparse_chol_solve(chol, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "solve succeeds");

    TEST_ASSERT(lmmc_test_nearly_equal(x.data[0], 3.0, 1e-10), "x[0] == 3.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[1], 7.0, 1e-10), "x[1] == 7.0");

    lmmc_sparse_chol_destroy(chol);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}


static void test_not_positive_definite(void)
{
    printf("Test: Non-positive definite matrix detection\n");


    lmmc_real_t A_data[] = {
        1.0, 2.0,
        2.0, 1.0
    };

    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_status_t st;

    st = build_sparse_spd(A_data, 2, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build non-SPD matrix");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "symbolic analysis succeeds for non-SPD");

    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_NOT_POSITIVE_DEFINITE,
                "numeric returns NOT_POSITIVE_DEFINITE for non-SPD matrix");

    lmmc_sparse_chol_destroy(chol);
    lmmc_sparse_destroy(&A);
}


static void test_non_square_matrix(void)
{
    printf("Test: Non-square matrix detection\n");


    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_create_csc(2, 3, 0, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create non-square matrix");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_INVALID_ARGUMENT,
                "symbolic returns INVALID_ARGUMENT for non-square matrix");

    lmmc_sparse_destroy(&A);
}


static void test_null_pointers(void)
{
    printf("Test: NULL pointer validation\n");

    lmmc_sparse_chol_t* chol = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_chol_symbolic(NULL, &chol);
    TEST_ASSERT(st == LMMC_STATUS_INVALID_ARGUMENT, "symbolic with NULL matrix");

    lmmc_sparse_mat_t A = {0};
    A.rows = 2; A.cols = 2;
    st = lmmc_sparse_chol_symbolic(&A, NULL);
    TEST_ASSERT(st == LMMC_STATUS_INVALID_ARGUMENT, "symbolic with NULL output");


    lmmc_vec_t b = {0}, x = {0};
    st = lmmc_sparse_chol_solve(NULL, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_INVALID_ARGUMENT, "solve with NULL chol");


    lmmc_sparse_chol_destroy(NULL);
    TEST_ASSERT(1, "destroy NULL does not crash");
}


static void test_dimension_mismatch(void)
{
    printf("Test: Dimension mismatch in solve\n");

    lmmc_real_t A_data[] = {
        2.0, 1.0,
        1.0, 2.0
    };

    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_sparse_spd(A_data, 2, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build matrix");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "symbolic");

    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "numeric");


    st = lmmc_vec_create(3, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b (wrong size)");
    st = lmmc_vec_create(2, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x");

    st = lmmc_sparse_chol_solve(chol, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_DIMENSION_MISMATCH,
                "solve returns DIMENSION_MISMATCH for wrong-size b");

    lmmc_sparse_chol_destroy(chol);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}


static void test_4x4_spd(void)
{
    printf("Test: 4x4 SPD matrix solve\n");


    lmmc_real_t A_data[] = {
        3.0, 1.0, 1.0, 1.0,
        1.0, 4.0, 1.0, 1.0,
        1.0, 1.0, 5.0, 1.0,
        1.0, 1.0, 1.0, 6.0
    };

    lmmc_sparse_mat_t A = {0};
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_vec_t b = {0}, x = {0};
    lmmc_status_t st;

    st = build_sparse_spd(A_data, 4, &A);
    TEST_ASSERT(st == LMMC_STATUS_OK, "build 4x4 SPD matrix");

    st = lmmc_sparse_chol_symbolic(&A, &chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "symbolic");

    st = lmmc_sparse_chol_numeric(&A, chol);
    TEST_ASSERT(st == LMMC_STATUS_OK, "numeric");


    st = lmmc_vec_create(4, &b);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create b");
    st = lmmc_vec_create(4, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create x");

    b.data[0] = 12.0;
    b.data[1] = 16.0;
    b.data[2] = 22.0;
    b.data[3] = 30.0;

    st = lmmc_sparse_chol_solve(chol, &b, &x);
    TEST_ASSERT(st == LMMC_STATUS_OK, "solve succeeds");

    TEST_ASSERT(lmmc_test_nearly_equal(x.data[0], 1.0, 1e-10), "x[0] == 1.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[1], 2.0, 1e-10), "x[1] == 2.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[2], 3.0, 1e-10), "x[2] == 3.0");
    TEST_ASSERT(lmmc_test_nearly_equal(x.data[3], 4.0, 1e-10), "x[3] == 4.0");

    lmmc_sparse_chol_destroy(chol);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&A);
}

static void test_disconnected_duplicate_pattern(void)
{
    lmmc_sparse_mat_t matrix = {0};
    lmmc_sparse_chol_t* first = NULL;
    lmmc_sparse_chol_t* second = NULL;
    lmmc_vec_t b = {0}, x1 = {0}, x2 = {0};
    lmmc_status_t st;
    const size_t col_ptr[5] = {0, 1, 4, 7, 8};
    const size_t row_idx[8] = {0, 1, 2, 2, 1, 1, 2, 3};
    const lmmc_real_t values[8] = {4.0, 3.0, 0.5, 0.5, 0.5, 0.5, 3.0, 5.0};
    const lmmc_real_t rhs[4] = {4.0, 9.0, 11.0, 20.0};

    printf("Test: disconnected graph with duplicate symmetric entries\n");
    st = lmmc_sparse_create_csc(4, 4, 8, &matrix);
    TEST_ASSERT(st == LMMC_STATUS_OK, "create duplicate-pattern CSC");
    if (st != LMMC_STATUS_OK) return;
    memcpy(matrix.row_ptr, col_ptr, sizeof(col_ptr));
    memcpy(matrix.col_idx, row_idx, sizeof(row_idx));
    memcpy(matrix.values, values, sizeof(values));

    st = lmmc_sparse_chol_symbolic(&matrix, &first);
    TEST_ASSERT(st == LMMC_STATUS_OK, "first residual-degree symbolic analysis");
    if (st == LMMC_STATUS_OK) st = lmmc_sparse_chol_numeric(&matrix, first);
    TEST_ASSERT(st == LMMC_STATUS_OK, "first duplicate-pattern factorization");

    st = lmmc_sparse_chol_symbolic(&matrix, &second);
    TEST_ASSERT(st == LMMC_STATUS_OK, "second residual-degree symbolic analysis");
    if (st == LMMC_STATUS_OK) st = lmmc_sparse_chol_numeric(&matrix, second);
    TEST_ASSERT(st == LMMC_STATUS_OK, "second duplicate-pattern factorization");

    if (lmmc_vec_create(4, &b) != LMMC_STATUS_OK ||
        lmmc_vec_create(4, &x1) != LMMC_STATUS_OK ||
        lmmc_vec_create(4, &x2) != LMMC_STATUS_OK) {
        TEST_ASSERT(0, "create duplicate-pattern solve vectors");
        goto cleanup;
    }
    memcpy(b.data, rhs, sizeof(rhs));
    st = lmmc_sparse_chol_solve(first, &b, &x1);
    TEST_ASSERT(st == LMMC_STATUS_OK, "first duplicate-pattern solve");
    st = lmmc_sparse_chol_solve(second, &b, &x2);
    TEST_ASSERT(st == LMMC_STATUS_OK, "second duplicate-pattern solve");
    for (size_t i = 0; i < 4; ++i) {
        TEST_ASSERT(lmmc_test_nearly_equal(x1.data[i], (lmmc_real_t)(i + 1), 1e-10),
                    "duplicate-pattern solution is exact");
        TEST_ASSERT(lmmc_test_nearly_equal(x1.data[i], x2.data[i], 1e-12),
                    "duplicate-pattern ordering is deterministic");
    }

cleanup:
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&x1);
    lmmc_vec_destroy(&x2);
    lmmc_sparse_chol_destroy(first);
    lmmc_sparse_chol_destroy(second);
    lmmc_sparse_destroy(&matrix);
}

int main(void)
{
    printf("=== Sparse Cholesky Factorization Tests ===\n\n");

    test_basic_3x3();
    test_identity_2x2();
    test_not_positive_definite();
    test_non_square_matrix();
    test_null_pointers();
    test_dimension_mismatch();
    test_4x4_spd();
    test_disconnected_duplicate_pattern();

    printf("\n=== Results: %d/%d tests passed ===\n", pass_count, test_count);

    return (pass_count == test_count) ? 0 : 1;
}
