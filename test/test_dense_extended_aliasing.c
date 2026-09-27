/**
 * @file test_dense_extended_aliasing.c
 * @brief 稠密矩阵扩展接口单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define TEST_EPS_TIGHT 1e-12
#define TEST_EPS_NORMAL 1e-10
#define TEST_EPS_LOOSE 1e-6

static void test_gemv_aliasing(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_real_t matrix_data[12] = {
        1.0, 2.0, 90.0, 90.0,
        3.0, 4.0, 90.0, 90.0,
        5.0, 6.0, 90.0, 90.0};
    lmmc_real_t vector_data[4] = {7.0, 8.0, 9.0, 10.0};
    lmmc_real_t matrix_before[12];
    lmmc_real_t vector_before[4];
    lmmc_mat_t a = {2, 2, 4, matrix_data, 0};
    lmmc_vec_t x = {2, vector_data, 0};
    lmmc_vec_t y_over_x = {2, vector_data + 1, 0};
    lmmc_vec_t y_over_a = {2, matrix_data + 4, 0};
    lmmc_vec_t bad_y = {1, matrix_data + 4, 0};

    memcpy(matrix_before, matrix_data, sizeof(matrix_data));
    memcpy(vector_before, vector_data, sizeof(vector_data));
    st = lmmc_mat_gemv(1.0, &a, 0, &x, 0.0, &y_over_x);
    if (st != LMMC_STATUS_INVALID_ARGUMENT ||
        memcmp(matrix_before, matrix_data, sizeof(matrix_data)) != 0 ||
        memcmp(vector_before, vector_data, sizeof(vector_data)) != 0) {
        fail_msg("5.13 FAIL: GEMV x/y overlap was not rejected atomically\n");
    }

    st = lmmc_mat_vec_mul(&a, &x, &y_over_a);
    if (st != LMMC_STATUS_INVALID_ARGUMENT ||
        memcmp(matrix_before, matrix_data, sizeof(matrix_data)) != 0 ||
        memcmp(vector_before, vector_data, sizeof(vector_data)) != 0) {
        fail_msg("5.13 FAIL: mat_vec_mul A/y overlap was not rejected atomically\n");
    }

    st = lmmc_mat_gemv(1.0, &a, 0, &x, 0.0, &bad_y);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH ||
        memcmp(matrix_before, matrix_data, sizeof(matrix_data)) != 0) {
        fail_msg("5.13 FAIL: GEMV dimension precedence or atomicity mismatch\n");
    }
    return;
}

static void test_gemm_aliasing(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_real_t matrix_data[12] = {
        1.0, 2.0, 90.0, 90.0,
        3.0, 4.0, 90.0, 90.0,
        5.0, 6.0, 90.0, 90.0};
    lmmc_real_t vector_data[4] = {7.0, 8.0, 9.0, 10.0};
    lmmc_real_t matrix_before[12];
    lmmc_real_t vector_before[4];
    lmmc_mat_t a = {2, 2, 4, matrix_data, 0};

    memcpy(matrix_before, matrix_data, sizeof(matrix_data));
    memcpy(vector_before, vector_data, sizeof(vector_data));
    lmmc_mat_t b = {2, 2, 2, vector_data, 0};
    lmmc_mat_t c_over_a = a;
    st = lmmc_mat_gemm(
        1.0, &a, 0, &b, 0, 0.0, &c_over_a);
    if (st != LMMC_STATUS_INVALID_ARGUMENT ||
        memcmp(matrix_before, matrix_data, sizeof(matrix_data)) != 0 ||
        memcmp(vector_before, vector_data, sizeof(vector_data)) != 0) {
        fail_msg("5.13 FAIL: GEMM A/C overlap was not rejected atomically\n");
    }
    return;
}

static void test_transpose_aliasing(void **state) {
    (void)state;

    lmmc_status_t st;
    lmmc_real_t storage[20];
    lmmc_real_t before[20];
    lmmc_mat_t src = {2, 3, 4, storage, 0};
    lmmc_mat_t overlapping_dst = {3, 2, 3, storage + 6, 0};
    lmmc_mat_t overflow_src = {2, 2, SIZE_MAX, storage, 0};
    lmmc_mat_t separate_dst = {2, 2, 2, storage + 12, 0};

    for (size_t i = 0; i < 20; ++i)
        storage[i] = (lmmc_real_t)(i + 1);
    memcpy(before, storage, sizeof(storage));
    st = lmmc_mat_transpose_to(&src, &overlapping_dst);
    if (st != LMMC_STATUS_INVALID_ARGUMENT ||
        memcmp(before, storage, sizeof(storage)) != 0) {
        fail_msg("5.14 FAIL: transpose overlap was not rejected atomically\n");
    }

    st = lmmc_mat_transpose_to(&overflow_src, &separate_dst);
    if (st != LMMC_STATUS_INVALID_ARGUMENT ||
        memcmp(before, storage, sizeof(storage)) != 0) {
        fail_msg("5.14 FAIL: transpose envelope overflow was not rejected atomically\n");
    }
    return;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_gemv_aliasing),
        cmocka_unit_test(test_gemm_aliasing),
        cmocka_unit_test(test_transpose_aliasing),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
