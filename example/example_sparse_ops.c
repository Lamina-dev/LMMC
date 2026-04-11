#include <stdio.h>
#include "lmmc/lmmc.h"

static void print_mat(const char* name, const lmmc_mat_t* m) {
    size_t i = 0;
    size_t j = 0;
    printf("%s =\n", name);
    for (i = 0; i < m->rows; ++i) {
        printf("  [");
        for (j = 0; j < m->cols; ++j) {
            printf("%8.3f", m->data[i * m->stride + j]);
            if (j + 1 < m->cols) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

int main(void) {
    lmmc_mat_t a_dense = {0};
    lmmc_mat_t at_dense = {0};
    lmmc_mat_t b = {0};
    lmmc_mat_t c = {0};
    lmmc_sparse_mat_t a_sparse = {0};
    lmmc_sparse_mat_t at_sparse = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_mat_create(3, 3, &a_dense);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create a_dense failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    a_dense.data[0] = 4.0; a_dense.data[1] = 0.0; a_dense.data[2] = 0.0;
    a_dense.data[3] = 0.0; a_dense.data[4] = 5.0; a_dense.data[5] = 1.0;
    a_dense.data[6] = 2.0; a_dense.data[7] = 0.0; a_dense.data[8] = 3.0;

    st = lmmc_sparse_from_dense(&a_dense, 1e-14, &a_sparse);
    if (st != LMMC_STATUS_OK) {
        printf("sparse_from_dense failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_sparse_transpose(&a_sparse, &at_sparse);
    if (st != LMMC_STATUS_OK) {
        printf("sparse_transpose failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(3, 3, &at_dense);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create at_dense failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_sparse_to_dense(&at_sparse, &at_dense);
    if (st != LMMC_STATUS_OK) {
        printf("sparse_to_dense for transpose failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(3, 2, &b);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create b failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(3, 2, &c);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create c failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    b.data[0] = 1.0; b.data[1] = 2.0;
    b.data[2] = 0.0; b.data[3] = 1.0;
    b.data[4] = 3.0; b.data[5] = -1.0;

    st = lmmc_sparse_mat_mat_mul_dense(&a_sparse, &b, &c);
    if (st != LMMC_STATUS_OK) {
        printf("sparse_mat_mat_mul_dense failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    print_mat("A (dense)", &a_dense);
    print_mat("A^T (dense)", &at_dense);
    print_mat("B", &b);
    print_mat("C = A * B", &c);

cleanup:
    lmmc_mat_destroy(&c);
    lmmc_mat_destroy(&b);
    lmmc_mat_destroy(&at_dense);
    lmmc_sparse_destroy(&at_sparse);
    lmmc_sparse_destroy(&a_sparse);
    lmmc_mat_destroy(&a_dense);
    return rc;
}
