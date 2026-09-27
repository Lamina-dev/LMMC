/**
 * @file test_eigen_extended_svd.c
 * @brief 奇异值分解扩展接口测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"
#include "test_common.h"

#define TOL 1e-10
#define TOL_LOOSE 1e-8
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])
struct test_fixture {
    struct test_svd_reconstruction_resources {
        lmmc_mat_t mat;
        lmmc_svd_result_t result;
    } test_svd_reconstruction;
    struct test_svd_reconstruction_4x3_resources {
        lmmc_mat_t mat;
        lmmc_svd_result_t result;
    } test_svd_reconstruction_4x3;
};

static void test_svd_reconstruction(void **state) {
    struct test_fixture *fixture = *state;
    struct test_svd_reconstruction_resources *resources = &fixture->test_svd_reconstruction;

    size_t m = 3, n = 3;
    assert_int_equal(lmmc_mat_create(m, n, &resources->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0};
    memcpy(resources->mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_svd(&resources->mat, &resources->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("SVD should succeed, got %d", (int)s);
    }

    size_t p = (m < n) ? m : n;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < p; k++) {
                sum += MAT_ELEM(&resources->result.U, i, k) *
                       resources->result.sigma.data[k] *
                       MAT_ELEM(&resources->result.Vt, k, j);
            }
            if (!(fabs(sum - data[i * n + j]) < TOL_LOOSE)) {
                fail_msg("SVD recon A[%" PRIuMAX "][%" PRIuMAX "]: expected %f, got %f", (uintmax_t)(i), (uintmax_t)(j), data[i * n + j], sum);
            }
        }
    }

    lmmc_svd_result_destroy(&resources->result);
    lmmc_mat_destroy(&resources->mat);
}

static void test_svd_reconstruction_4x3(void **state) {
    struct test_fixture *fixture = *state;
    struct test_svd_reconstruction_4x3_resources *resources = &fixture->test_svd_reconstruction_4x3;

    size_t m = 4, n = 3;
    assert_int_equal(lmmc_mat_create(m, n, &resources->mat), LMMC_STATUS_OK);
    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 13.0};
    memcpy(resources->mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_svd(&resources->mat, &resources->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("4x3 SVD should succeed, got %d", (int)s);
    }

    size_t p = (m < n) ? m : n;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < p; k++) {
                sum += MAT_ELEM(&resources->result.U, i, k) *
                       resources->result.sigma.data[k] *
                       MAT_ELEM(&resources->result.Vt, k, j);
            }
            if (!(fabs(sum - data[i * n + j]) < TOL_LOOSE)) {
                fail_msg("4x3 SVD recon A[%" PRIuMAX "][%" PRIuMAX "]: expected %f, got %f", (uintmax_t)(i), (uintmax_t)(j), data[i * n + j], sum);
            }
        }
    }

    lmmc_svd_result_destroy(&resources->result);
    lmmc_mat_destroy(&resources->mat);
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->test_svd_reconstruction.mat);
    lmmc_svd_result_destroy(&fixture->test_svd_reconstruction.result);
    lmmc_mat_destroy(&fixture->test_svd_reconstruction_4x3.mat);
    lmmc_svd_result_destroy(&fixture->test_svd_reconstruction_4x3.result);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_svd_reconstruction, setup, teardown),
        cmocka_unit_test_setup_teardown(test_svd_reconstruction_4x3, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
