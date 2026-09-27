/**
 * @file test_sparse_utils.c
 * 针对 LMMC 中 sparse utils 相关接口的单元测试。
 */
#include <stdio.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_mat_t denseA;
    lmmc_mat_t denseB;
    lmmc_mat_t result;
    lmmc_sparse_mat_t sparseA;
    lmmc_sparse_mat_t sparseB;
    lmmc_sparse_mat_t sparseC;
    lmmc_vec_t diag;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    return fixture ? 0 : -1;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->diag);
    lmmc_sparse_destroy(&fixture->sparseC);
    lmmc_sparse_destroy(&fixture->sparseB);
    lmmc_sparse_destroy(&fixture->sparseA);
    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->denseB);
    lmmc_mat_destroy(&fixture->denseA);
    free(fixture);
    *state = NULL;
    return 0;
}

static void test_sparse_scale(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_mat_create(3, 3, &fixture->denseA);
    assert_true(st == LMMC_STATUS_OK);

    LMMC_REAL_SET_D(&fixture->denseA.data[0], 4.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[1], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[2], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[3], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[4], 5.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[5], 1.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[6], 2.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[7], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[8], 3.0);

    st = lmmc_sparse_from_dense(&fixture->denseA, 1e-14, &fixture->sparseA);
    assert_true(st == LMMC_STATUS_OK);

    lmmc_real_t alpha;
    LMMC_REAL_SET_D(&alpha, 2.0);
    st = lmmc_sparse_scale(&fixture->sparseA, alpha);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_mat_create(3, 3, &fixture->result);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_to_dense(&fixture->sparseA, &fixture->result);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(((((lmmc_test_nearly_equal(fixture->result.data[0], 8.0, 1e-12)) && (lmmc_test_nearly_equal(fixture->result.data[4], 10.0, 1e-12))) && (lmmc_test_nearly_equal(fixture->result.data[5], 2.0, 1e-12))) && (lmmc_test_nearly_equal(fixture->result.data[6], 4.0, 1e-12))) && (lmmc_test_nearly_equal(fixture->result.data[8], 6.0, 1e-12)));
}

static void test_sparse_scale_null(void **state) {
    (void)state;

    lmmc_status_t st = lmmc_sparse_scale(NULL, 2.0);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_sparse_norm_fro(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    lmmc_real_t norm;

    st = lmmc_mat_create(3, 3, &fixture->denseA);
    assert_true(st == LMMC_STATUS_OK);

    LMMC_REAL_SET_D(&fixture->denseA.data[0], 3.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[1], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[2], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[3], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[4], 4.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[5], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[6], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[7], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[8], 0.0);

    st = lmmc_sparse_from_dense(&fixture->denseA, 1e-14, &fixture->sparseA);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_norm_fro(&fixture->sparseA, &norm);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(lmmc_test_nearly_equal(norm, 5.0, 1e-12));
}

static void test_sparse_norm_fro_null(void **state) {
    (void)state;

    lmmc_real_t norm;
    lmmc_status_t st = lmmc_sparse_norm_fro(NULL, &norm);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_sparse_diag(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_mat_create(3, 3, &fixture->denseA);
    assert_true(st == LMMC_STATUS_OK);

    LMMC_REAL_SET_D(&fixture->denseA.data[0], 4.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[1], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[2], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[3], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[4], 5.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[5], 1.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[6], 2.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[7], 0.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[8], 3.0);

    st = lmmc_sparse_from_dense(&fixture->denseA, 1e-14, &fixture->sparseA);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_diag(&fixture->sparseA, &fixture->diag);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(fixture->diag.size == 3);

    assert_true(((lmmc_test_nearly_equal(fixture->diag.data[0], 4.0, 1e-12)) && (lmmc_test_nearly_equal(fixture->diag.data[1], 5.0, 1e-12))) && (lmmc_test_nearly_equal(fixture->diag.data[2], 3.0, 1e-12)));
}

static void test_sparse_diag_non_square(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_mat_create(2, 3, &fixture->denseA);
    assert_true(st == LMMC_STATUS_OK);

    LMMC_REAL_SET_D(&fixture->denseA.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[1], 2.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[2], 3.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[3], 4.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[4], 5.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[5], 6.0);

    st = lmmc_sparse_from_dense(&fixture->denseA, 1e-14, &fixture->sparseA);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_diag(&fixture->sparseA, &fixture->diag);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void create_sparse_sum(lmmc_mat_t *denseA, lmmc_mat_t *denseB, lmmc_sparse_mat_t *sparseA, lmmc_sparse_mat_t *sparseB, lmmc_sparse_mat_t *sparseC) {

    lmmc_status_t st;
    st = lmmc_mat_create(3, 3, denseA);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_mat_create(3, 3, denseB);
    assert_true(st == LMMC_STATUS_OK);

    LMMC_REAL_SET_D(&denseA->data[0], 1.0);
    LMMC_REAL_SET_D(&denseA->data[1], 0.0);
    LMMC_REAL_SET_D(&denseA->data[2], 2.0);
    LMMC_REAL_SET_D(&denseA->data[3], 0.0);
    LMMC_REAL_SET_D(&denseA->data[4], 3.0);
    LMMC_REAL_SET_D(&denseA->data[5], 0.0);
    LMMC_REAL_SET_D(&denseA->data[6], 4.0);
    LMMC_REAL_SET_D(&denseA->data[7], 0.0);
    LMMC_REAL_SET_D(&denseA->data[8], 5.0);

    LMMC_REAL_SET_D(&denseB->data[0], 0.0);
    LMMC_REAL_SET_D(&denseB->data[1], 6.0);
    LMMC_REAL_SET_D(&denseB->data[2], 0.0);
    LMMC_REAL_SET_D(&denseB->data[3], 7.0);
    LMMC_REAL_SET_D(&denseB->data[4], 0.0);
    LMMC_REAL_SET_D(&denseB->data[5], 8.0);
    LMMC_REAL_SET_D(&denseB->data[6], 0.0);
    LMMC_REAL_SET_D(&denseB->data[7], 9.0);
    LMMC_REAL_SET_D(&denseB->data[8], 0.0);

    st = lmmc_sparse_from_dense(denseA, 1e-14, sparseA);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_from_dense(denseB, 1e-14, sparseB);
    assert_true(st == LMMC_STATUS_OK);

    lmmc_real_t alpha, beta;
    LMMC_REAL_SET_D(&alpha, 2.0);
    LMMC_REAL_SET_D(&beta, 3.0);
    st = lmmc_sparse_add(alpha, sparseA, beta, sparseB, sparseC);
    assert_true(st == LMMC_STATUS_OK);
}

static void test_sparse_add(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    create_sparse_sum(&fixture->denseA, &fixture->denseB, &fixture->sparseA, &fixture->sparseB, &fixture->sparseC);

    st = lmmc_mat_create(3, 3, &fixture->result);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_to_dense(&fixture->sparseC, &fixture->result);
    assert_true(st == LMMC_STATUS_OK);

    double expected[9] = {2.0, 18.0, 4.0, 21.0, 6.0, 24.0, 8.0, 27.0, 10.0};
    int i;
    for (i = 0; i < 9; ++i) {
        assert_true(lmmc_test_nearly_equal(fixture->result.data[i], expected[i], 1e-12));
    }
}

static void test_sparse_add_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_mat_create(2, 3, &fixture->denseA);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_mat_create(3, 3, &fixture->denseB);
    assert_true(st == LMMC_STATUS_OK);

    LMMC_REAL_SET_D(&fixture->denseA.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[1], 2.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[2], 3.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[3], 4.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[4], 5.0);
    LMMC_REAL_SET_D(&fixture->denseA.data[5], 6.0);

    LMMC_REAL_SET_D(&fixture->denseB.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->denseB.data[1], 0.0);
    LMMC_REAL_SET_D(&fixture->denseB.data[2], 0.0);
    LMMC_REAL_SET_D(&fixture->denseB.data[3], 0.0);
    LMMC_REAL_SET_D(&fixture->denseB.data[4], 1.0);
    LMMC_REAL_SET_D(&fixture->denseB.data[5], 0.0);
    LMMC_REAL_SET_D(&fixture->denseB.data[6], 0.0);
    LMMC_REAL_SET_D(&fixture->denseB.data[7], 0.0);
    LMMC_REAL_SET_D(&fixture->denseB.data[8], 1.0);

    st = lmmc_sparse_from_dense(&fixture->denseA, 1e-14, &fixture->sparseA);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_from_dense(&fixture->denseB, 1e-14, &fixture->sparseB);
    assert_true(st == LMMC_STATUS_OK);

    lmmc_real_t alpha, beta;
    LMMC_REAL_SET_D(&alpha, 1.0);
    LMMC_REAL_SET_D(&beta, 1.0);
    st = lmmc_sparse_add(alpha, &fixture->sparseA, beta, &fixture->sparseB, &fixture->sparseC);
    assert_true(st == LMMC_STATUS_DIMENSION_MISMATCH);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_sparse_scale, setup, teardown),
        cmocka_unit_test(test_sparse_scale_null),
        cmocka_unit_test_setup_teardown(test_sparse_norm_fro, setup, teardown),
        cmocka_unit_test(test_sparse_norm_fro_null),
        cmocka_unit_test_setup_teardown(test_sparse_diag, setup, teardown),
        cmocka_unit_test_setup_teardown(test_sparse_diag_non_square, setup, teardown),
        cmocka_unit_test_setup_teardown(test_sparse_add, setup, teardown),
        cmocka_unit_test_setup_teardown(test_sparse_add_dimension_mismatch, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
