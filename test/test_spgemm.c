#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    lmmc_mat_t a_dense = {0}, b_dense = {0}, c_ref = {0}, c_res_dense = {0};
    lmmc_sparse_mat_t a = {0}, b = {0}, c = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    // A = [ 1 0 2 ]
    //     [ 0 3 0 ]
    // B = [ 0 4 ]
    //     [ 5 0 ]
    //     [ 0 6 ]
    // C = A * B = [ 1*0+0*5+2*0  1*4+0*0+2*6 ] = [ 0 16 ]
    //             [ 0*0+3*5+0*0  0*4+3*0+0*6 ]   [ 15 0 ]

    st = lmmc_mat_create(2, 3, &a_dense);
    a_dense.data[0] = 1.0; a_dense.data[1] = 0.0; a_dense.data[2] = 2.0;
    a_dense.data[3] = 0.0; a_dense.data[4] = 3.0; a_dense.data[5] = 0.0;

    st = lmmc_mat_create(3, 2, &b_dense);
    b_dense.data[0] = 0.0; b_dense.data[1] = 4.0;
    b_dense.data[2] = 5.0; b_dense.data[3] = 0.0;
    b_dense.data[4] = 0.0; b_dense.data[5] = 6.0;

    st = lmmc_sparse_from_dense(&a_dense, 1e-14, &a);
    st = lmmc_sparse_from_dense(&b_dense, 1e-14, &b);

    st = lmmc_sparse_mat_mat_mul_sparse(&a, &b, &c);
    if (st != LMMC_STATUS_OK) {
        printf("SpGEMM failed: %s\n", lmmc_status_string(st));
        rc = 1; goto cleanup;
    }

    st = lmmc_mat_create(2, 2, &c_res_dense);
    st = lmmc_sparse_to_dense(&c, &c_res_dense);

    if (!lmmc_test_nearly_equal(c_res_dense.data[0], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(c_res_dense.data[1], 16.0, 1e-12) ||
        !lmmc_test_nearly_equal(c_res_dense.data[2], 15.0, 1e-12) ||
        !lmmc_test_nearly_equal(c_res_dense.data[3], 0.0, 1e-12)) {
        printf("SpGEMM result incorrect\n");
        rc = 1; goto cleanup;
    }

    // Test with identity
    lmmc_sparse_mat_t eye = {0}, res_eye = {0};
    lmmc_mat_t eye_dense = {0};
    lmmc_mat_create(3, 3, &eye_dense);
    eye_dense.data[0] = 1.0; eye_dense.data[4] = 1.0; eye_dense.data[8] = 1.0;
    lmmc_sparse_from_dense(&eye_dense, 1e-14, &eye);

    st = lmmc_sparse_mat_mat_mul_sparse(&a, &eye, &res_eye);
    if (st != LMMC_STATUS_OK || res_eye.nnz != a.nnz) {
        rc = 1; goto cleanup_eye;
    }
    
cleanup_eye:
    lmmc_sparse_destroy(&res_eye);
    lmmc_sparse_destroy(&eye);
    lmmc_mat_destroy(&eye_dense);

cleanup:
    lmmc_mat_destroy(&c_res_dense);
    lmmc_sparse_destroy(&c);
    lmmc_sparse_destroy(&b);
    lmmc_sparse_destroy(&a);
    lmmc_mat_destroy(&b_dense);
    lmmc_mat_destroy(&a_dense);

    if (rc == 0) printf("SpGEMM test passed\n");
    return rc;
}
