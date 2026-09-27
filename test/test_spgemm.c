/**
 * @file test_spgemm.c
 * 针对 LMMC 中 spgemm 相关接口的单元测试。
 */
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_mat_t a_dense;
    lmmc_mat_t b_dense;
    lmmc_mat_t c_res_dense;
    lmmc_sparse_mat_t a;
    lmmc_sparse_mat_t b;
    lmmc_sparse_mat_t c;
    lmmc_sparse_mat_t discovery_order_and_cancellation_a;
    lmmc_sparse_mat_t discovery_order_and_cancellation_b;
    lmmc_sparse_mat_t discovery_order_and_cancellation_product;
    lmmc_sparse_mat_t discovery_order_and_cancellation_csc;
    lmmc_sparse_mat_t discovery_order_and_cancellation_csr;
    lmmc_sparse_mat_t discovery_order_and_cancellation_sum;
    lmmc_mat_t discovery_order_and_cancellation_dense;
    lmmc_vec_t discovery_order_and_cancellation_x;
    lmmc_vec_t discovery_order_and_cancellation_y;
    lmmc_sparse_mat_t identity_product_eye;
    lmmc_sparse_mat_t identity_product_res_eye;
    lmmc_mat_t identity_product_eye_dense;
};

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->identity_product_eye_dense);
    lmmc_sparse_destroy(&fixture->identity_product_res_eye);
    lmmc_sparse_destroy(&fixture->identity_product_eye);
    lmmc_vec_destroy(&fixture->discovery_order_and_cancellation_y);
    lmmc_vec_destroy(&fixture->discovery_order_and_cancellation_x);
    lmmc_mat_destroy(&fixture->discovery_order_and_cancellation_dense);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_sum);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_csr);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_csc);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_product);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_b);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_a);
    lmmc_mat_destroy(&fixture->a_dense);
    lmmc_mat_destroy(&fixture->b_dense);
    lmmc_sparse_destroy(&fixture->a);
    lmmc_sparse_destroy(&fixture->b);
    lmmc_sparse_destroy(&fixture->c);
    lmmc_mat_destroy(&fixture->c_res_dense);
    free(fixture);
    *state = NULL;
    return 0;
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    lmmc_status_t st;

    {
        lmmc_mat_t *a_dense = &fixture->a_dense;
        lmmc_mat_t *b_dense = &fixture->b_dense;
        lmmc_sparse_mat_t *a = &fixture->a;
        lmmc_sparse_mat_t *b = &fixture->b;

        st = lmmc_mat_create(2, 3, a_dense);
        if (st != LMMC_STATUS_OK)
            goto cleanup;
        LMMC_REAL_SET_D(&a_dense->data[0], 1.0);
        LMMC_REAL_SET_D(&a_dense->data[1], 0.0);
        LMMC_REAL_SET_D(&a_dense->data[2], 2.0);
        LMMC_REAL_SET_D(&a_dense->data[3], 0.0);
        LMMC_REAL_SET_D(&a_dense->data[4], 3.0);
        LMMC_REAL_SET_D(&a_dense->data[5], 0.0);

        st = lmmc_mat_create(3, 2, b_dense);
        if (st != LMMC_STATUS_OK)
            goto cleanup;
        LMMC_REAL_SET_D(&b_dense->data[0], 0.0);
        LMMC_REAL_SET_D(&b_dense->data[1], 4.0);
        LMMC_REAL_SET_D(&b_dense->data[2], 5.0);
        LMMC_REAL_SET_D(&b_dense->data[3], 0.0);
        LMMC_REAL_SET_D(&b_dense->data[4], 0.0);
        LMMC_REAL_SET_D(&b_dense->data[5], 6.0);

        st = lmmc_sparse_from_dense(a_dense, 1e-14, a);
        if (st != LMMC_STATUS_OK)
            goto cleanup;
        st = lmmc_sparse_from_dense(b_dense, 1e-14, b);
        if (st != LMMC_STATUS_OK)
            goto cleanup;
    }
    return 0;

cleanup:
    teardown(state);
    return st;
}

static void test_sparse_matrix_product(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *c_res_dense = &fixture->c_res_dense;
    lmmc_sparse_mat_t *a = &fixture->a;
    lmmc_sparse_mat_t *b = &fixture->b;
    lmmc_sparse_mat_t *c = &fixture->c;

    lmmc_status_t st;

    st = lmmc_sparse_mat_mat_mul_sparse(a, b, c);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_mat_create(2, 2, c_res_dense);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_to_dense(c, c_res_dense);
    assert_true(st == LMMC_STATUS_OK);

    assert_true((((lmmc_test_nearly_equal(c_res_dense->data[0], 0.0, 1e-12)) && (lmmc_test_nearly_equal(c_res_dense->data[1], 16.0, 1e-12))) && (lmmc_test_nearly_equal(c_res_dense->data[2], 15.0, 1e-12))) && (lmmc_test_nearly_equal(c_res_dense->data[3], 0.0, 1e-12)));
}

static void test_discovery_order_and_cancellation(void **state) {
    struct test_fixture *fixture = *state;

    size_t ap[] = {0, 2}, ai[] = {0, 1};
    size_t bp[] = {0, 1, 2}, bi[] = {1, 0};
    double av[] = {1, 1}, bv[] = {2, 3};

    assert_true(lmmc_sparse_wrap_csr(1, 2, 2, ap, ai, av, &fixture->discovery_order_and_cancellation_a) == LMMC_STATUS_OK);
    assert_true(lmmc_sparse_wrap_csr(2, 2, 2, bp, bi, bv, &fixture->discovery_order_and_cancellation_b) == LMMC_STATUS_OK);
    assert_true(lmmc_sparse_mat_mat_mul_sparse(&fixture->discovery_order_and_cancellation_a, &fixture->discovery_order_and_cancellation_b, &fixture->discovery_order_and_cancellation_product) == LMMC_STATUS_OK);
    assert_true(fixture->discovery_order_and_cancellation_product.nnz == 2 && fixture->discovery_order_and_cancellation_product.col_idx[0] == 0 && fixture->discovery_order_and_cancellation_product.col_idx[1] == 1);
    assert_true(lmmc_sparse_to_csc(&fixture->discovery_order_and_cancellation_product, &fixture->discovery_order_and_cancellation_csc) == LMMC_STATUS_OK);
    assert_true(lmmc_sparse_to_csr(&fixture->discovery_order_and_cancellation_csc, &fixture->discovery_order_and_cancellation_csr) == LMMC_STATUS_OK);
    assert_true(lmmc_sparse_add(1, &fixture->discovery_order_and_cancellation_product, 1, &fixture->discovery_order_and_cancellation_csr, &fixture->discovery_order_and_cancellation_sum) == LMMC_STATUS_OK);
    assert_true(lmmc_mat_create(1, 2, &fixture->discovery_order_and_cancellation_dense) == LMMC_STATUS_OK);
    assert_true(lmmc_sparse_to_dense(&fixture->discovery_order_and_cancellation_sum, &fixture->discovery_order_and_cancellation_dense) == LMMC_STATUS_OK);
    assert_true(fixture->discovery_order_and_cancellation_dense.data[0] == 6 && fixture->discovery_order_and_cancellation_dense.data[1] == 4);
    assert_true(lmmc_vec_create(2, &fixture->discovery_order_and_cancellation_x) == LMMC_STATUS_OK);
    assert_true(lmmc_vec_create(1, &fixture->discovery_order_and_cancellation_y) == LMMC_STATUS_OK);
    fixture->discovery_order_and_cancellation_x.data[0] = 2;
    fixture->discovery_order_and_cancellation_x.data[1] = 5;
    assert_true(lmmc_sparse_mat_vec_mul(&fixture->discovery_order_and_cancellation_csr, &fixture->discovery_order_and_cancellation_x, &fixture->discovery_order_and_cancellation_y) == LMMC_STATUS_OK && fixture->discovery_order_and_cancellation_y.data[0] == 16);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_sum);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_csr);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_csc);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_product);
    bi[1] = 1;
    bv[1] = -2;
    assert_true(lmmc_sparse_mat_mat_mul_sparse(&fixture->discovery_order_and_cancellation_a, &fixture->discovery_order_and_cancellation_b, &fixture->discovery_order_and_cancellation_product) == LMMC_STATUS_OK);
    assert_true(fixture->discovery_order_and_cancellation_product.nnz == 1 && fixture->discovery_order_and_cancellation_product.col_idx[0] == 1 && fixture->discovery_order_and_cancellation_product.values[0] == 0);
    assert_true(lmmc_sparse_to_dense(&fixture->discovery_order_and_cancellation_product, &fixture->discovery_order_and_cancellation_dense) == LMMC_STATUS_OK);
    assert_true(fixture->discovery_order_and_cancellation_dense.data[0] == 0 && fixture->discovery_order_and_cancellation_dense.data[1] == 0);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_product);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_a);
    lmmc_sparse_destroy(&fixture->discovery_order_and_cancellation_b);
    lmmc_vec_destroy(&fixture->discovery_order_and_cancellation_x);
    lmmc_vec_destroy(&fixture->discovery_order_and_cancellation_y);
    lmmc_mat_destroy(&fixture->discovery_order_and_cancellation_dense);
}

static void test_identity_product(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *a = &fixture->a;

    assert_int_equal(
        lmmc_mat_create(3, 3, &fixture->identity_product_eye_dense),
        LMMC_STATUS_OK);
    fixture->identity_product_eye_dense.data[0] = 1.0;
    fixture->identity_product_eye_dense.data[4] = 1.0;
    fixture->identity_product_eye_dense.data[8] = 1.0;
    assert_int_equal(
        lmmc_sparse_from_dense(
            &fixture->identity_product_eye_dense, 1e-14,
            &fixture->identity_product_eye),
        LMMC_STATUS_OK);

    assert_int_equal(
        lmmc_sparse_mat_mat_mul_sparse(
            a, &fixture->identity_product_eye,
            &fixture->identity_product_res_eye),
        LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_mat_create(a->rows, a->cols, &fixture->c_res_dense),
        LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_sparse_to_dense(
            &fixture->identity_product_res_eye, &fixture->c_res_dense),
        LMMC_STATUS_OK);
    for (size_t row = 0; row < a->rows; ++row) {
        for (size_t col = 0; col < a->cols; ++col) {
            const size_t expected_index =
                row * fixture->a_dense.stride + col;
            const size_t actual_index =
                row * fixture->c_res_dense.stride + col;
            assert_true(lmmc_test_nearly_equal(
                fixture->c_res_dense.data[actual_index],
                fixture->a_dense.data[expected_index], 1e-12));
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_discovery_order_and_cancellation, setup, teardown),
        cmocka_unit_test_setup_teardown(test_sparse_matrix_product, setup, teardown),
        cmocka_unit_test_setup_teardown(test_identity_product, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
