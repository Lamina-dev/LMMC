#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    lmmc_mat_t dense = {0};
    lmmc_mat_t dense_t = {0};
    lmmc_mat_t zero_dense = {0};
    lmmc_mat_t bmat = {0};
    lmmc_mat_t cmat = {0};
    lmmc_mat_t bbad = {0};
    lmmc_mat_t bzero = {0};
    lmmc_mat_t czero = {0};
    lmmc_sparse_mat_t sparse = {0};
    lmmc_sparse_mat_t sparse_t = {0};
    lmmc_sparse_mat_t zero_sparse = {0};
    lmmc_sparse_mat_t bad_sparse = {0};
    lmmc_vec_t x = {0};
    lmmc_vec_t y = {0};
    lmmc_vec_t zx = {0};
    lmmc_vec_t zy = {0};
    lmmc_vec_t x_bad = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_mat_create(3, 3, &dense);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(3, &x);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(3, &y);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    dense.data[0] = 4.0; dense.data[1] = 0.0; dense.data[2] = 0.0;
    dense.data[3] = 0.0; dense.data[4] = 5.0; dense.data[5] = 1.0;
    dense.data[6] = 2.0; dense.data[7] = 0.0; dense.data[8] = 3.0;

    x.data[0] = 1.0;
    x.data[1] = 2.0;
    x.data[2] = -1.0;

    st = lmmc_sparse_from_dense(&dense, 1e-14, &sparse);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_sparse_mat_vec_mul(&sparse, &x, &y);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    if (!lmmc_test_nearly_equal(y.data[0], 4.0, 1e-12) ||
        !lmmc_test_nearly_equal(y.data[1], 9.0, 1e-12) ||
        !lmmc_test_nearly_equal(y.data[2], -1.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_sparse_transpose(&sparse, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_sparse_transpose(&sparse, &sparse_t);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(3, 3, &dense_t);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_sparse_to_dense(&sparse_t, &dense_t);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    if (!lmmc_test_nearly_equal(dense_t.data[0], 4.0, 1e-12) ||
        !lmmc_test_nearly_equal(dense_t.data[1], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(dense_t.data[2], 2.0, 1e-12) ||
        !lmmc_test_nearly_equal(dense_t.data[3], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(dense_t.data[4], 5.0, 1e-12) ||
        !lmmc_test_nearly_equal(dense_t.data[5], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(dense_t.data[6], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(dense_t.data[7], 1.0, 1e-12) ||
        !lmmc_test_nearly_equal(dense_t.data[8], 3.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(3, 2, &bmat);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(3, 2, &cmat);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    bmat.data[0] = 1.0; bmat.data[1] = 2.0;
    bmat.data[2] = 0.0; bmat.data[3] = 1.0;
    bmat.data[4] = 3.0; bmat.data[5] = -1.0;

    st = lmmc_sparse_mat_mat_mul_dense(&sparse, &bmat, &cmat);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    if (!lmmc_test_nearly_equal(cmat.data[0], 4.0, 1e-12) ||
        !lmmc_test_nearly_equal(cmat.data[1], 8.0, 1e-12) ||
        !lmmc_test_nearly_equal(cmat.data[2], 3.0, 1e-12) ||
        !lmmc_test_nearly_equal(cmat.data[3], 4.0, 1e-12) ||
        !lmmc_test_nearly_equal(cmat.data[4], 11.0, 1e-12) ||
        !lmmc_test_nearly_equal(cmat.data[5], 1.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(2, 2, &bbad);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_sparse_mat_mat_mul_dense(&sparse, &bbad, &cmat);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_sparse_create_csr(0, 3, 0, &bad_sparse);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    {
        size_t bad_row_ptr[3] = {0, 1, 0};
        size_t bad_col_idx[1] = {0};
        double bad_values[1] = {1.0};
        st = lmmc_sparse_wrap_csr(2, 2, 1, bad_row_ptr, bad_col_idx, bad_values, &bad_sparse);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1;
            goto cleanup;
        }
    }

    st = lmmc_mat_create(4, 4, &zero_dense);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_sparse_from_dense(&zero_dense, 0.0, &zero_sparse);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    if (zero_sparse.nnz != 0) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(4, &zx);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(4, &zy);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_fill(&zx, 3.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_sparse_mat_vec_mul(&zero_sparse, &zx, &zy);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    if (!lmmc_test_nearly_equal(zy.data[0], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(zy.data[1], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(zy.data[2], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(zy.data[3], 0.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(4, 2, &bzero);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(4, 2, &czero);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    bzero.data[0] = 1.0; bzero.data[1] = 0.0;
    bzero.data[2] = 2.0; bzero.data[3] = 1.0;
    bzero.data[4] = 3.0; bzero.data[5] = 2.0;
    bzero.data[6] = 4.0; bzero.data[7] = 3.0;

    st = lmmc_sparse_mat_mat_mul_dense(&zero_sparse, &bzero, &czero);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    if (!lmmc_test_nearly_equal(czero.data[0], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(czero.data[1], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(czero.data[2], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(czero.data[3], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(czero.data[4], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(czero.data[5], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(czero.data[6], 0.0, 1e-12) ||
        !lmmc_test_nearly_equal(czero.data[7], 0.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(3, &x_bad);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_sparse_mat_vec_mul(&zero_sparse, &x_bad, &zy);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
    }

cleanup:
    lmmc_mat_destroy(&czero);
    lmmc_mat_destroy(&bzero);
    lmmc_mat_destroy(&bbad);
    lmmc_mat_destroy(&cmat);
    lmmc_mat_destroy(&bmat);
    lmmc_mat_destroy(&dense_t);
    lmmc_vec_destroy(&x_bad);
    lmmc_vec_destroy(&zy);
    lmmc_vec_destroy(&zx);
    lmmc_sparse_destroy(&sparse_t);
    lmmc_sparse_destroy(&sparse);
    lmmc_sparse_destroy(&zero_sparse);
    lmmc_sparse_destroy(&bad_sparse);
    lmmc_vec_destroy(&y);
    lmmc_vec_destroy(&x);
    lmmc_mat_destroy(&zero_dense);
    lmmc_mat_destroy(&dense);

    if (rc != 0) {
        printf("sparse test failed\n");
    }
    return rc;
}
