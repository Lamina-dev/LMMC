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

    lmmc_mat_t a_copy = {0};
    lmmc_vec_t v2 = {0};
    lmmc_vec_t v_wrap = {0};
    lmmc_real_t wrap_data[2];
    LMMC_REAL_INIT(&wrap_data[0]); LMMC_REAL_SET_D(&wrap_data[0], 5.0);
    LMMC_REAL_INIT(&wrap_data[1]); LMMC_REAL_SET_D(&wrap_data[1], 6.0);

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

    LMMC_REAL_SET_D(&a.data[0], 1.0); LMMC_REAL_SET_D(&a.data[1], 2.0);
    LMMC_REAL_SET_D(&a.data[2], 3.0); LMMC_REAL_SET_D(&a.data[3], 4.0);

    LMMC_REAL_SET_D(&b.data[0], 2.0); LMMC_REAL_SET_D(&b.data[1], 0.0);
    LMMC_REAL_SET_D(&b.data[2], 1.0); LMMC_REAL_SET_D(&b.data[3], 2.0);

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

    st = lmmc_mat_create(2, 2, &a_copy);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    st = lmmc_mat_copy(&a, &a_copy);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    if (!lmmc_test_nearly_equal(a_copy.data[0], 1.0, 1e-12) ||
        !lmmc_test_nearly_equal(a_copy.data[1], 2.0, 1e-12) ||
        !lmmc_test_nearly_equal(a_copy.data[2], 3.0, 1e-12) ||
        !lmmc_test_nearly_equal(a_copy.data[3], 4.0, 1e-12)) {
        rc = 1; goto cleanup;
    }

    lmmc_real_t fnorm = 0.0;
    st = lmmc_mat_norm_fro(&a_copy, &fnorm);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    if (!lmmc_test_nearly_equal(fnorm, 5.47722557505, 1e-6)) { // sqrt(1+4+9+16) = sqrt(30) = 5.477225...
        rc = 1; goto cleanup;
    }

    lmmc_mat_t a_trans = {0};
    st = lmmc_mat_create(2, 2, &a_trans);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    st = lmmc_mat_transpose_to(&a_copy, &a_trans);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    if (!lmmc_test_nearly_equal(a_trans.data[0], 1.0, 1e-12) ||
        !lmmc_test_nearly_equal(a_trans.data[1], 3.0, 1e-12) ||
        !lmmc_test_nearly_equal(a_trans.data[2], 2.0, 1e-12) ||
        !lmmc_test_nearly_equal(a_trans.data[3], 4.0, 1e-12)) {
        rc = 1; goto cleanup;
    }

    lmmc_mat_t m_wrap = {0};
    st = lmmc_mat_wrap(2, 1, 1, wrap_data, &m_wrap);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    if (!lmmc_test_nearly_equal(m_wrap.data[0], 5.0, 1e-12) ||
        !lmmc_test_nearly_equal(m_wrap.data[1], 6.0, 1e-12)) {
        rc = 1; goto cleanup;
    }

    st = lmmc_vec_wrap(2, wrap_data, &v_wrap);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    if (!lmmc_test_nearly_equal(v_wrap.data[0], 5.0, 1e-12) ||
        !lmmc_test_nearly_equal(v_wrap.data[1], 6.0, 1e-12)) {
        rc = 1; goto cleanup;
    }

    st = lmmc_vec_create(2, &v2);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    st = lmmc_mat_vec_mul(&a, &v_wrap, &v2);
    if (st != LMMC_STATUS_OK) { rc = 1; goto cleanup; }
    if (!lmmc_test_nearly_equal(v2.data[0], 17.0, 1e-12) ||
        !lmmc_test_nearly_equal(v2.data[1], 39.0, 1e-12)) {
        rc = 1; goto cleanup;
    }

cleanup:
    lmmc_vec_destroy(&v2);
    lmmc_vec_destroy(&v_wrap);
    lmmc_mat_destroy(&m_wrap);
    lmmc_mat_destroy(&a_trans);
    lmmc_mat_destroy(&a_copy);
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
