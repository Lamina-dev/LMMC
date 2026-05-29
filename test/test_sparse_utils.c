/**
 * @file test_sparse_utils.c
 * 针对 LMMC 中 sparse utils 相关接口的单元测试。
 */
#include <stdio.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"


static int test_sparse_scale(void) {

    lmmc_mat_t dense = {0};
    lmmc_sparse_mat_t sparse = {0};
    lmmc_status_t st;

    st = lmmc_mat_create(3, 3, &dense);
    if (st != LMMC_STATUS_OK) return 1;

    LMMC_REAL_SET_D(&dense.data[0], 4.0); LMMC_REAL_SET_D(&dense.data[1], 0.0); LMMC_REAL_SET_D(&dense.data[2], 0.0);
    LMMC_REAL_SET_D(&dense.data[3], 0.0); LMMC_REAL_SET_D(&dense.data[4], 5.0); LMMC_REAL_SET_D(&dense.data[5], 1.0);
    LMMC_REAL_SET_D(&dense.data[6], 2.0); LMMC_REAL_SET_D(&dense.data[7], 0.0); LMMC_REAL_SET_D(&dense.data[8], 3.0);

    st = lmmc_sparse_from_dense(&dense, 1e-14, &sparse);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&dense); return 1; }


    lmmc_real_t alpha;
    LMMC_REAL_SET_D(&alpha, 2.0);
    st = lmmc_sparse_scale(&sparse, alpha);
    if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense); return 1; }


    lmmc_mat_t result = {0};
    st = lmmc_mat_create(3, 3, &result);
    if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense); return 1; }

    st = lmmc_sparse_to_dense(&sparse, &result);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&result); lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense); return 1; }


    if (!lmmc_test_nearly_equal(result.data[0], 8.0, 1e-12) ||
        !lmmc_test_nearly_equal(result.data[4], 10.0, 1e-12) ||
        !lmmc_test_nearly_equal(result.data[5], 2.0, 1e-12) ||
        !lmmc_test_nearly_equal(result.data[6], 4.0, 1e-12) ||
        !lmmc_test_nearly_equal(result.data[8], 6.0, 1e-12)) {
        printf("FAIL: test_sparse_scale values incorrect\n");
        lmmc_mat_destroy(&result); lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense);
        return 1;
    }

    lmmc_mat_destroy(&result);
    lmmc_sparse_destroy(&sparse);
    lmmc_mat_destroy(&dense);
    printf("PASS: test_sparse_scale\n");
    return 0;
}

static int test_sparse_scale_null(void) {
    lmmc_status_t st = lmmc_sparse_scale(NULL, 2.0);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: test_sparse_scale_null expected INVALID_ARGUMENT\n");
        return 1;
    }
    printf("PASS: test_sparse_scale_null\n");
    return 0;
}

static int test_sparse_norm_fro(void) {

    lmmc_mat_t dense = {0};
    lmmc_sparse_mat_t sparse = {0};
    lmmc_status_t st;
    lmmc_real_t norm;

    st = lmmc_mat_create(3, 3, &dense);
    if (st != LMMC_STATUS_OK) return 1;

    LMMC_REAL_SET_D(&dense.data[0], 3.0); LMMC_REAL_SET_D(&dense.data[1], 0.0); LMMC_REAL_SET_D(&dense.data[2], 0.0);
    LMMC_REAL_SET_D(&dense.data[3], 0.0); LMMC_REAL_SET_D(&dense.data[4], 4.0); LMMC_REAL_SET_D(&dense.data[5], 0.0);
    LMMC_REAL_SET_D(&dense.data[6], 0.0); LMMC_REAL_SET_D(&dense.data[7], 0.0); LMMC_REAL_SET_D(&dense.data[8], 0.0);

    st = lmmc_sparse_from_dense(&dense, 1e-14, &sparse);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&dense); return 1; }

    st = lmmc_sparse_norm_fro(&sparse, &norm);
    if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense); return 1; }

    if (!lmmc_test_nearly_equal(norm, 5.0, 1e-12)) {
        printf("FAIL: test_sparse_norm_fro expected 5.0, got %f\n", norm);
        lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense);
        return 1;
    }

    lmmc_sparse_destroy(&sparse);
    lmmc_mat_destroy(&dense);
    printf("PASS: test_sparse_norm_fro\n");
    return 0;
}

static int test_sparse_norm_fro_null(void) {
    lmmc_real_t norm;
    lmmc_status_t st = lmmc_sparse_norm_fro(NULL, &norm);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: test_sparse_norm_fro_null expected INVALID_ARGUMENT\n");
        return 1;
    }
    printf("PASS: test_sparse_norm_fro_null\n");
    return 0;
}

static int test_sparse_diag(void) {

    lmmc_mat_t dense = {0};
    lmmc_sparse_mat_t sparse = {0};
    lmmc_vec_t diag = {0};
    lmmc_status_t st;

    st = lmmc_mat_create(3, 3, &dense);
    if (st != LMMC_STATUS_OK) return 1;

    LMMC_REAL_SET_D(&dense.data[0], 4.0); LMMC_REAL_SET_D(&dense.data[1], 0.0); LMMC_REAL_SET_D(&dense.data[2], 0.0);
    LMMC_REAL_SET_D(&dense.data[3], 0.0); LMMC_REAL_SET_D(&dense.data[4], 5.0); LMMC_REAL_SET_D(&dense.data[5], 1.0);
    LMMC_REAL_SET_D(&dense.data[6], 2.0); LMMC_REAL_SET_D(&dense.data[7], 0.0); LMMC_REAL_SET_D(&dense.data[8], 3.0);

    st = lmmc_sparse_from_dense(&dense, 1e-14, &sparse);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&dense); return 1; }

    st = lmmc_sparse_diag(&sparse, &diag);
    if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense); return 1; }

    if (diag.size != 3) {
        printf("FAIL: test_sparse_diag expected size 3, got %zu\n", diag.size);
        lmmc_vec_destroy(&diag); lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense);
        return 1;
    }

    if (!lmmc_test_nearly_equal(diag.data[0], 4.0, 1e-12) ||
        !lmmc_test_nearly_equal(diag.data[1], 5.0, 1e-12) ||
        !lmmc_test_nearly_equal(diag.data[2], 3.0, 1e-12)) {
        printf("FAIL: test_sparse_diag values incorrect\n");
        lmmc_vec_destroy(&diag); lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense);
        return 1;
    }

    lmmc_vec_destroy(&diag);
    lmmc_sparse_destroy(&sparse);
    lmmc_mat_destroy(&dense);
    printf("PASS: test_sparse_diag\n");
    return 0;
}

static int test_sparse_diag_non_square(void) {

    lmmc_mat_t dense = {0};
    lmmc_sparse_mat_t sparse = {0};
    lmmc_vec_t diag = {0};
    lmmc_status_t st;

    st = lmmc_mat_create(2, 3, &dense);
    if (st != LMMC_STATUS_OK) return 1;

    LMMC_REAL_SET_D(&dense.data[0], 1.0); LMMC_REAL_SET_D(&dense.data[1], 2.0); LMMC_REAL_SET_D(&dense.data[2], 3.0);
    LMMC_REAL_SET_D(&dense.data[3], 4.0); LMMC_REAL_SET_D(&dense.data[4], 5.0); LMMC_REAL_SET_D(&dense.data[5], 6.0);

    st = lmmc_sparse_from_dense(&dense, 1e-14, &sparse);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&dense); return 1; }

    st = lmmc_sparse_diag(&sparse, &diag);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: test_sparse_diag_non_square expected INVALID_ARGUMENT, got %d\n", (int)st);
        lmmc_vec_destroy(&diag); lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense);
        return 1;
    }

    lmmc_sparse_destroy(&sparse);
    lmmc_mat_destroy(&dense);
    printf("PASS: test_sparse_diag_non_square\n");
    return 0;
}

static int test_sparse_add(void) {

    lmmc_mat_t denseA = {0}, denseB = {0};
    lmmc_sparse_mat_t sparseA = {0}, sparseB = {0}, sparseC = {0};
    lmmc_mat_t result = {0};
    lmmc_status_t st;

    st = lmmc_mat_create(3, 3, &denseA);
    if (st != LMMC_STATUS_OK) return 1;
    st = lmmc_mat_create(3, 3, &denseB);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&denseA); return 1; }


    LMMC_REAL_SET_D(&denseA.data[0], 1.0); LMMC_REAL_SET_D(&denseA.data[1], 0.0); LMMC_REAL_SET_D(&denseA.data[2], 2.0);
    LMMC_REAL_SET_D(&denseA.data[3], 0.0); LMMC_REAL_SET_D(&denseA.data[4], 3.0); LMMC_REAL_SET_D(&denseA.data[5], 0.0);
    LMMC_REAL_SET_D(&denseA.data[6], 4.0); LMMC_REAL_SET_D(&denseA.data[7], 0.0); LMMC_REAL_SET_D(&denseA.data[8], 5.0);


    LMMC_REAL_SET_D(&denseB.data[0], 0.0); LMMC_REAL_SET_D(&denseB.data[1], 6.0); LMMC_REAL_SET_D(&denseB.data[2], 0.0);
    LMMC_REAL_SET_D(&denseB.data[3], 7.0); LMMC_REAL_SET_D(&denseB.data[4], 0.0); LMMC_REAL_SET_D(&denseB.data[5], 8.0);
    LMMC_REAL_SET_D(&denseB.data[6], 0.0); LMMC_REAL_SET_D(&denseB.data[7], 9.0); LMMC_REAL_SET_D(&denseB.data[8], 0.0);

    st = lmmc_sparse_from_dense(&denseA, 1e-14, &sparseA);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&denseA); lmmc_mat_destroy(&denseB); return 1; }
    st = lmmc_sparse_from_dense(&denseB, 1e-14, &sparseB);
    if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&sparseA); lmmc_mat_destroy(&denseA); lmmc_mat_destroy(&denseB); return 1; }


    lmmc_real_t alpha, beta;
    LMMC_REAL_SET_D(&alpha, 2.0);
    LMMC_REAL_SET_D(&beta, 3.0);
    st = lmmc_sparse_add(alpha, &sparseA, beta, &sparseB, &sparseC);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL: test_sparse_add lmmc_sparse_add returned %d\n", (int)st);
        lmmc_sparse_destroy(&sparseA); lmmc_sparse_destroy(&sparseB);
        lmmc_mat_destroy(&denseA); lmmc_mat_destroy(&denseB);
        return 1;
    }


    st = lmmc_mat_create(3, 3, &result);
    if (st != LMMC_STATUS_OK) {
        lmmc_sparse_destroy(&sparseC); lmmc_sparse_destroy(&sparseA); lmmc_sparse_destroy(&sparseB);
        lmmc_mat_destroy(&denseA); lmmc_mat_destroy(&denseB);
        return 1;
    }
    st = lmmc_sparse_to_dense(&sparseC, &result);
    if (st != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&result); lmmc_sparse_destroy(&sparseC);
        lmmc_sparse_destroy(&sparseA); lmmc_sparse_destroy(&sparseB);
        lmmc_mat_destroy(&denseA); lmmc_mat_destroy(&denseB);
        return 1;
    }


    double expected[9] = {2.0, 18.0, 4.0, 21.0, 6.0, 24.0, 8.0, 27.0, 10.0};
    int i;
    for (i = 0; i < 9; ++i) {
        if (!lmmc_test_nearly_equal(result.data[i], expected[i], 1e-12)) {
            printf("FAIL: test_sparse_add result[%d] = %f, expected %f\n", i, result.data[i], expected[i]);
            lmmc_mat_destroy(&result); lmmc_sparse_destroy(&sparseC);
            lmmc_sparse_destroy(&sparseA); lmmc_sparse_destroy(&sparseB);
            lmmc_mat_destroy(&denseA); lmmc_mat_destroy(&denseB);
            return 1;
        }
    }

    lmmc_mat_destroy(&result);
    lmmc_sparse_destroy(&sparseC);
    lmmc_sparse_destroy(&sparseA);
    lmmc_sparse_destroy(&sparseB);
    lmmc_mat_destroy(&denseA);
    lmmc_mat_destroy(&denseB);
    printf("PASS: test_sparse_add\n");
    return 0;
}

static int test_sparse_add_dimension_mismatch(void) {

    lmmc_mat_t denseA = {0}, denseB = {0};
    lmmc_sparse_mat_t sparseA = {0}, sparseB = {0}, sparseC = {0};
    lmmc_status_t st;

    st = lmmc_mat_create(2, 3, &denseA);
    if (st != LMMC_STATUS_OK) return 1;
    st = lmmc_mat_create(3, 3, &denseB);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&denseA); return 1; }

    LMMC_REAL_SET_D(&denseA.data[0], 1.0); LMMC_REAL_SET_D(&denseA.data[1], 2.0); LMMC_REAL_SET_D(&denseA.data[2], 3.0);
    LMMC_REAL_SET_D(&denseA.data[3], 4.0); LMMC_REAL_SET_D(&denseA.data[4], 5.0); LMMC_REAL_SET_D(&denseA.data[5], 6.0);

    LMMC_REAL_SET_D(&denseB.data[0], 1.0); LMMC_REAL_SET_D(&denseB.data[1], 0.0); LMMC_REAL_SET_D(&denseB.data[2], 0.0);
    LMMC_REAL_SET_D(&denseB.data[3], 0.0); LMMC_REAL_SET_D(&denseB.data[4], 1.0); LMMC_REAL_SET_D(&denseB.data[5], 0.0);
    LMMC_REAL_SET_D(&denseB.data[6], 0.0); LMMC_REAL_SET_D(&denseB.data[7], 0.0); LMMC_REAL_SET_D(&denseB.data[8], 1.0);

    st = lmmc_sparse_from_dense(&denseA, 1e-14, &sparseA);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&denseA); lmmc_mat_destroy(&denseB); return 1; }
    st = lmmc_sparse_from_dense(&denseB, 1e-14, &sparseB);
    if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&sparseA); lmmc_mat_destroy(&denseA); lmmc_mat_destroy(&denseB); return 1; }

    lmmc_real_t alpha, beta;
    LMMC_REAL_SET_D(&alpha, 1.0);
    LMMC_REAL_SET_D(&beta, 1.0);
    st = lmmc_sparse_add(alpha, &sparseA, beta, &sparseB, &sparseC);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        printf("FAIL: test_sparse_add_dimension_mismatch expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_sparse_destroy(&sparseC); lmmc_sparse_destroy(&sparseA); lmmc_sparse_destroy(&sparseB);
        lmmc_mat_destroy(&denseA); lmmc_mat_destroy(&denseB);
        return 1;
    }

    lmmc_sparse_destroy(&sparseA);
    lmmc_sparse_destroy(&sparseB);
    lmmc_mat_destroy(&denseA);
    lmmc_mat_destroy(&denseB);
    printf("PASS: test_sparse_add_dimension_mismatch\n");
    return 0;
}

int main(void) {
    int rc = 0;

    rc |= test_sparse_scale();
    rc |= test_sparse_scale_null();
    rc |= test_sparse_norm_fro();
    rc |= test_sparse_norm_fro_null();
    rc |= test_sparse_diag();
    rc |= test_sparse_diag_non_square();
    rc |= test_sparse_add();
    rc |= test_sparse_add_dimension_mismatch();

    if (rc == 0) {
        printf("\nAll sparse utility tests PASSED\n");
    } else {
        printf("\nSome sparse utility tests FAILED\n");
    }
    return rc;
}
