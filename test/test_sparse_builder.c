#include <stdio.h>
#include <assert.h>
#include "lmmc/lmmc.h"

int main(void) {
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_sparse_mat_t sparse = {0};
    lmmc_mat_t dense = {0};
    lmmc_status_t st;

    printf("Testing Sparse Builder (COO -> CSR)...\n");

    /* 1. Create builder for a 3x3 matrix */
    st = lmmc_sparse_builder_create(3, 3, 4, &builder);
    if (st != LMMC_STATUS_OK) goto cleanup;

    /* 2. Add entries in random order */
    /*
       Matrix:
       [ 1.0  0.0  2.0 ]
       [ 0.0  3.0  0.0 ]
       [ 4.0  0.0  5.0 ]
    */
    st = lmmc_sparse_builder_add(builder, 2, 2, 5.0);
    if (st != LMMC_STATUS_OK) {  goto cleanup; }
    st = lmmc_sparse_builder_add(builder, 0, 0, 1.0);
    if (st != LMMC_STATUS_OK) {  goto cleanup; }
    st = lmmc_sparse_builder_add(builder, 1, 1, 3.0);
    if (st != LMMC_STATUS_OK) {  goto cleanup; }
    st = lmmc_sparse_builder_add(builder, 0, 2, 2.0);
    if (st != LMMC_STATUS_OK) {  goto cleanup; }
    st = lmmc_sparse_builder_add(builder, 2, 0, 4.0);
    if (st != LMMC_STATUS_OK) {  goto cleanup; }

    /* 3. Build CSR matrix */
    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, &sparse);
    if (st != LMMC_STATUS_OK) goto cleanup;
    if (sparse.nnz != 5) {
        st = LMMC_STATUS_DIMENSION_MISMATCH; // or another appropriate error
        goto cleanup;
    }

    /* 4. Verify entries */
    /* Note: Builder doesn't guarantee sorted columns within a row unless we add a sort step.
       But it should contain the correct data. */
    
    printf("Matrix built successfully with %zu non-zeros.\n", sparse.nnz);

    /* 5. Convert to dense for easy visual verification */
    st = lmmc_mat_create(3, 3, &dense);
    if (st != LMMC_STATUS_OK) { goto cleanup; }
    st = lmmc_sparse_to_dense(&sparse, &dense);
    if (st != LMMC_STATUS_OK) { goto cleanup; }

    printf("Dense representation:\n");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            printf("%.1f ", dense.data[i * dense.stride + j]);
        }
        printf("\n");
    }

    assert(dense.data[0 * dense.stride + 0] == 1.0);
    assert(dense.data[0 * dense.stride + 1] == 0.0);
    assert(dense.data[0 * dense.stride + 2] == 2.0);
    assert(dense.data[1 * dense.stride + 0] == 0.0);
    assert(dense.data[1 * dense.stride + 1] == 3.0);
    assert(dense.data[1 * dense.stride + 2] == 0.0);
    assert(dense.data[2 * dense.stride + 0] == 4.0);
    assert(dense.data[2 * dense.stride + 1] == 0.0);
    assert(dense.data[2 * dense.stride + 2] == 5.0);

    cleanup:
    lmmc_mat_destroy(&dense);
    lmmc_sparse_destroy(&sparse);
    lmmc_sparse_builder_destroy(builder);

    if (st == LMMC_STATUS_OK) {
        printf("Sparse Builder Test Passed!\n");
        return 0;
    } else {
        printf("Sparse Builder Test Failed!\n");
        return 1;
    }
}
