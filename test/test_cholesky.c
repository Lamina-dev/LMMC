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

    // Test 1: Standard 3x3 SPD matrix
    // A = [ 4  12 -16 ]
    //     [ 12 37 -43 ]
    //     [-16 -43 98 ]
    // L = [ 2   0   0 ]
    //     [ 6   1   0 ]
    //     [-8   5   3 ]
    st = lmmc_mat_create(3, 3, &a);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    
    a.data[0] = 4.0;  a.data[1] = 12.0; a.data[2] = -16.0;
    a.data[3] = 12.0; a.data[4] = 37.0; a.data[5] = -43.0;
    a.data[6] = -16.0; a.data[7] = -43.0; a.data[8] = 98.0;

    st = lmmc_cholesky_decompose_inplace(&a);
    if (st != LMMC_STATUS_OK) {
        printf("Cholesky decomposition failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    // Verify L
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

    // Test 2: Solve Ax = b
    // Using the same A, let x_true = [1, 2, 3]^T
    // b = A * x_true = [4*1 + 12*2 - 16*3] = [4 + 24 - 48] = [-20]
    //                  [12*1 + 37*2 - 43*3] = [12 + 74 - 129] = [-43]
    //                  [-16*1 - 43*2 + 98*3] = [-16 - 86 + 294] = [192]
    st = lmmc_vec_create(3, &b);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    st = lmmc_vec_create(3, &x);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }

    b.data[0] = -20.0;
    b.data[1] = -43.0;
    b.data[2] = 192.0;

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

    // Test 3: Non-positive definite matrix
    lmmc_mat_t a_npd = {0};
    st = lmmc_mat_create(2, 2, &a_npd);
    a_npd.data[0] = 1.0; a_npd.data[1] = 2.0;
    a_npd.data[2] = 2.0; a_npd.data[3] = 1.0; // Det = 1 - 4 = -3
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
