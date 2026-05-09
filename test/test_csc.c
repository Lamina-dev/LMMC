#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    lmmc_mat_t dense = {0};
    lmmc_sparse_mat_t csr = {0};
    lmmc_sparse_mat_t csc = {0};
    lmmc_sparse_mat_t csr_back = {0};
    lmmc_vec_t x = {0};
    lmmc_vec_t y_csr = {0};
    lmmc_vec_t y_csc = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    // 1. Create a dense matrix and convert to CSR
    st = lmmc_mat_create(3, 3, &dense);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    dense.data[0] = 1.0; dense.data[1] = 0.0; dense.data[2] = 2.0;
    dense.data[3] = 3.0; dense.data[4] = 4.0; dense.data[5] = 0.0;
    dense.data[6] = 0.0; dense.data[7] = 5.0; dense.data[8] = 6.0;

    st = lmmc_sparse_from_dense(&dense, 1e-14, &csr);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }

    // 2. Convert CSR to CSC
    st = lmmc_sparse_to_csc(&csr, &csc);
    if (st != LMMC_STATUS_OK) {
        printf("CSR to CSC failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    if (csc.format != LMMC_SPARSE_CSC) {
        printf("CSC format indicator incorrect\n");
        rc = 1;
        goto cleanup;
    }

    // 3. Verify CSC SpMV
    st = lmmc_vec_create(3, &x);
    st = lmmc_vec_create(3, &y_csr);
    st = lmmc_vec_create(3, &y_csc);
    x.data[0] = 1.0; x.data[1] = 1.0; x.data[2] = 1.0;

    st = lmmc_sparse_mat_vec_mul(&csr, &x, &y_csr);
    st = lmmc_sparse_mat_vec_mul(&csc, &x, &y_csc);

    for (size_t i = 0; i < 3; ++i) {
        if (!lmmc_test_nearly_equal(y_csr.data[i], y_csc.data[i], 1e-12)) {
            printf("CSC SpMV mismatch at row %zu: CSR=%f, CSC=%f\n", i, y_csr.data[i], y_csc.data[i]);
            rc = 1;
            goto cleanup;
        }
    }

    // 4. Convert back to CSR and verify
    st = lmmc_sparse_to_csr(&csc, &csr_back);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }

    if (csr_back.nnz != csr.nnz || csr_back.rows != csr.rows || csr_back.cols != csr.cols) {
        rc = 1;
        goto cleanup;
    }

    for (size_t i = 0; i <= csr.rows; ++i) {
        if (csr_back.row_ptr[i] != csr.row_ptr[i]) {
            rc = 1;
            goto cleanup;
        }
    }

    for (size_t i = 0; i < csr.nnz; ++i) {
        if (csr_back.col_idx[i] != csr.col_idx[i] || !lmmc_test_nearly_equal(csr_back.values[i], csr.values[i], 1e-12)) {
            rc = 1;
            goto cleanup;
        }
    }

    // 5. Test CSC Transpose (Should result in CSC of transposed matrix)
    lmmc_sparse_mat_t csc_t = {0};
    st = lmmc_sparse_transpose(&csc, &csc_t);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    if (csc_t.format != LMMC_SPARSE_CSC || csc_t.rows != 3 || csc_t.cols != 3) {
        rc = 1;
        goto cleanup;
    }
    
    // Verify values via dense matrix
    lmmc_mat_t t_dense;
    lmmc_mat_create(3, 3, &t_dense);
    st = lmmc_sparse_to_dense(&csc_t, &t_dense);
    if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&t_dense); goto cleanup; }
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (!lmmc_test_nearly_equal(t_dense.data[i * t_dense.stride + j], dense.data[j * dense.stride + i], 1e-12)) {
                printf("Transpose mismatch at (%zu, %zu): expected %f, got %f\n", i, j, dense.data[j * dense.stride + i], t_dense.data[i * t_dense.stride + j]);
                rc = 1;
            }
        }
    }
    lmmc_mat_destroy(&t_dense);

    lmmc_sparse_destroy(&csc_t);

cleanup:
    lmmc_vec_destroy(&y_csc);
    lmmc_vec_destroy(&y_csr);
    lmmc_vec_destroy(&x);
    lmmc_sparse_destroy(&csr_back);
    lmmc_sparse_destroy(&csc);
    lmmc_sparse_destroy(&csr);
    lmmc_mat_destroy(&dense);

    if (rc != 0) {
        printf("CSC test failed\n");
    } else {
        printf("CSC test passed\n");
    }
    return rc;
}
