#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    lmmc_mat_t a = {0};
    lmmc_mat_t singular = {0};
    lmmc_mat_t rect = {0};
    lmmc_vec_t b = {0};
    lmmc_vec_t x = {0};
    lmmc_vec_t b_bad = {0};
    size_t piv[3] = {0};
    size_t piv2[2] = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_mat_create(3, 3, &a);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(3, &b);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(3, &x);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    a.data[0] = 3.0; a.data[1] = 2.0; a.data[2] = -1.0;
    a.data[3] = 2.0; a.data[4] = -2.0; a.data[5] = 4.0;
    a.data[6] = -1.0; a.data[7] = 0.5; a.data[8] = -1.0;
    b.data[0] = 1.0;
    b.data[1] = -2.0;
    b.data[2] = 0.0;

    st = lmmc_lu_decompose_inplace(&a, piv, NULL);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_lu_solve(&a, piv, &b, &x);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    if (!lmmc_test_nearly_equal(x.data[0], 1.0, 1e-10) ||
        !lmmc_test_nearly_equal(x.data[1], -2.0, 1e-10) ||
        !lmmc_test_nearly_equal(x.data[2], -2.0, 1e-10)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(2, 2, &singular);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    singular.data[0] = 1.0; singular.data[1] = 2.0;
    singular.data[2] = 2.0; singular.data[3] = 4.0;
    st = lmmc_lu_decompose_inplace(&singular, piv2, NULL);
    if (st != LMMC_STATUS_SINGULAR_MATRIX) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(2, 3, &rect);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_lu_decompose_inplace(&rect, piv2, NULL);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(2, &b_bad);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_lu_solve(&a, piv, &b_bad, &x);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
    }

cleanup:
    lmmc_vec_destroy(&b_bad);
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_mat_destroy(&rect);
    lmmc_mat_destroy(&singular);
    lmmc_mat_destroy(&a);

    if (rc != 0) {
        printf("lu test failed\n");
    }
    return rc;
}
