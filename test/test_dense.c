#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    lmmc_mat_t a = {0};
    lmmc_mat_t b = {0};
    lmmc_mat_t c = {0};
    lmmc_mat_t left_bad = {0};
    lmmc_mat_t right_bad = {0};
    lmmc_mat_t out_bad = {0};
    lmmc_vec_t zero_vec = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_mat_create(2, 2, &a);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &b);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &c);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    a.data[0] = 1.0; a.data[1] = 2.0;
    a.data[2] = 3.0; a.data[3] = 4.0;

    b.data[0] = 2.0; b.data[1] = 0.0;
    b.data[2] = 1.0; b.data[3] = 2.0;

    st = lmmc_mat_mul(&a, &b, &c);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    if (!lmmc_test_nearly_equal(c.data[0], 4.0, 1e-12) ||
        !lmmc_test_nearly_equal(c.data[1], 4.0, 1e-12) ||
        !lmmc_test_nearly_equal(c.data[2], 10.0, 1e-12) ||
        !lmmc_test_nearly_equal(c.data[3], 8.0, 1e-12)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(0, 2, &left_bad);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(0, &zero_vec);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(2, 3, &left_bad);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &right_bad);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_mat_create(2, 2, &out_bad);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_mul(&left_bad, &right_bad, &out_bad);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
    }

cleanup:
    lmmc_vec_destroy(&zero_vec);
    lmmc_mat_destroy(&out_bad);
    lmmc_mat_destroy(&right_bad);
    lmmc_mat_destroy(&left_bad);
    lmmc_mat_destroy(&c);
    lmmc_mat_destroy(&b);
    lmmc_mat_destroy(&a);

    if (rc != 0) {
        printf("dense test failed\n");
    }
    return rc;
}
