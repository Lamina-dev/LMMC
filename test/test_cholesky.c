/**
 * @file test_cholesky.c
 * 针对 LMMC 中 cholesky 相关接口的单元测试。
 */
#include <stdio.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    lmmc_mat_t a = {0};
    lmmc_vec_t b = {0};
    lmmc_vec_t x = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;


    st = lmmc_mat_create(3, 3, &a);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }

    LMMC_REAL_SET_D(&a.data[0], 4.0);  LMMC_REAL_SET_D(&a.data[1], 12.0); LMMC_REAL_SET_D(&a.data[2], -16.0);
    LMMC_REAL_SET_D(&a.data[3], 12.0); LMMC_REAL_SET_D(&a.data[4], 37.0); LMMC_REAL_SET_D(&a.data[5], -43.0);
    LMMC_REAL_SET_D(&a.data[6], -16.0); LMMC_REAL_SET_D(&a.data[7], -43.0); LMMC_REAL_SET_D(&a.data[8], 98.0);

    st = lmmc_cholesky_decompose_inplace(&a);
    if (st != LMMC_STATUS_OK) {
        printf("Cholesky decomposition failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }


    if (!lmmc_test_nearly_equal(a.data[0], 2.0, 1e-10) ||
        !lmmc_test_nearly_equal(a.data[3], 6.0, 1e-10) ||
        !lmmc_test_nearly_equal(a.data[4], 1.0, 1e-10) ||
        !lmmc_test_nearly_equal(a.data[6], -8.0, 1e-10) ||
        !lmmc_test_nearly_equal(a.data[7], 5.0, 1e-10) ||
        !lmmc_test_nearly_equal(a.data[8], 3.0, 1e-10)) {
        printf("Cholesky L factors incorrect\n");
        rc = 1;
        goto cleanup;
    }


    st = lmmc_vec_create(3, &b);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    st = lmmc_vec_create(3, &x);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }

    LMMC_REAL_SET_D(&b.data[0], -20.0);
    LMMC_REAL_SET_D(&b.data[1], -43.0);
    LMMC_REAL_SET_D(&b.data[2], 192.0);

    st = lmmc_cholesky_solve(&a, &b, &x);
    if (st != LMMC_STATUS_OK) {
        printf("Cholesky solve failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    if (!lmmc_test_nearly_equal(x.data[0], 1.0, 1e-10) ||
        !lmmc_test_nearly_equal(x.data[1], 2.0, 1e-10) ||
        !lmmc_test_nearly_equal(x.data[2], 3.0, 1e-10)) {
        printf("Cholesky solution incorrect: [%f, %f, %f]\n", x.data[0], x.data[1], x.data[2]);
        rc = 1;
        goto cleanup;
    }


    lmmc_mat_t a_npd = {0};
    st = lmmc_mat_create(2, 2, &a_npd);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    LMMC_REAL_SET_D(&a_npd.data[0], 1.0); LMMC_REAL_SET_D(&a_npd.data[1], 2.0);
    LMMC_REAL_SET_D(&a_npd.data[2], 2.0); LMMC_REAL_SET_D(&a_npd.data[3], 1.0);
    st = lmmc_cholesky_decompose_inplace(&a_npd);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        printf("Expected NUMERICAL_FAILURE for non-positive definite matrix, got %s\n", lmmc_status_string(st));
        rc = 1;
    }
    lmmc_mat_destroy(&a_npd);

cleanup:
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_mat_destroy(&a);

    if (rc != 0) {
        printf("cholesky test failed\n");
    } else {
        printf("cholesky test passed\n");
    }
    return rc;
}
