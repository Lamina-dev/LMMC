#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    lmmc_mat_t a = {0};
    lmmc_mat_t tau_short_mat = {0};
    lmmc_mat_t singular_qr = {0};
    lmmc_vec_t b = {0};
    lmmc_vec_t x = {0};
    lmmc_vec_t b_bad = {0};
    lmmc_vec_t singular_b = {0};
    lmmc_vec_t singular_x = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    double tau[2] = {0.0, 0.0};
    double tau_short[1] = {0.0};
    double tau_singular[2] = {0.0, 0.0};
    int rc = 0;

    st = lmmc_mat_create(3, 2, &tau_short_mat);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_qr_decompose_inplace(&tau_short_mat, tau_short, 1);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(4, 2, &a);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(4, &b);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(2, &x);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    LMMC_REAL_SET_D(&a.data[0], 1.0); LMMC_REAL_SET_D(&a.data[1], 0.0);
    LMMC_REAL_SET_D(&a.data[2], 1.0); LMMC_REAL_SET_D(&a.data[3], 1.0);
    LMMC_REAL_SET_D(&a.data[4], 1.0); LMMC_REAL_SET_D(&a.data[5], 2.0);
    LMMC_REAL_SET_D(&a.data[6], 1.0); LMMC_REAL_SET_D(&a.data[7], 3.0);

    LMMC_REAL_SET_D(&b.data[0], 1.0);
    LMMC_REAL_SET_D(&b.data[1], 3.0);
    LMMC_REAL_SET_D(&b.data[2], 5.0);
    LMMC_REAL_SET_D(&b.data[3], 7.0);

    st = lmmc_qr_decompose_inplace(&a, tau, 2);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_qr_solve(&a, tau, &b, &x);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    if (!lmmc_test_nearly_equal(x.data[0], 1.0, 1e-9) ||
        !lmmc_test_nearly_equal(x.data[1], 2.0, 1e-9)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(3, &b_bad);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_qr_solve(&a, tau, &b_bad, &x);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(3, 2, &singular_qr);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(3, &singular_b);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(2, &singular_x);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    LMMC_REAL_SET_D(&singular_qr.data[0], 1.0); LMMC_REAL_SET_D(&singular_qr.data[1], 2.0);
    LMMC_REAL_SET_D(&singular_qr.data[2], 0.0); LMMC_REAL_SET_D(&singular_qr.data[3], 0.0);
    LMMC_REAL_SET_D(&singular_qr.data[4], 0.0); LMMC_REAL_SET_D(&singular_qr.data[5], 0.0);

    LMMC_REAL_SET_D(&singular_b.data[0], 1.0);
    LMMC_REAL_SET_D(&singular_b.data[1], 0.0);
    LMMC_REAL_SET_D(&singular_b.data[2], 0.0);

    st = lmmc_qr_decompose_inplace(&singular_qr, tau_singular, 2);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_qr_solve(&singular_qr, tau_singular, &singular_b, &singular_x);
    if (st != LMMC_STATUS_SINGULAR_MATRIX) {
        rc = 1;
    }

cleanup:
    lmmc_vec_destroy(&singular_x);
    lmmc_vec_destroy(&singular_b);
    lmmc_vec_destroy(&b_bad);
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_mat_destroy(&singular_qr);
    lmmc_mat_destroy(&tau_short_mat);
    lmmc_mat_destroy(&a);

    if (rc != 0) {
        printf("qr test failed\n");
    }
    return rc;
}
