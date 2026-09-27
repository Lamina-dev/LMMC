/**
 * @file test_dense.c
 * 针对 LMMC 中 dense 相关接口的单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

struct test_fixture {
    lmmc_mat_t a;
    lmmc_mat_t b;
    lmmc_mat_t c;
    lmmc_vec_t x;
    lmmc_vec_t y;
    lmmc_real_t wrap_data[2];
};

static void test_matrix_product(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_mat_t *b = &fixture->b;
    lmmc_mat_t *c = &fixture->c;
    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, b);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(2, 2, c);
    assert_int_equal(st, LMMC_STATUS_OK);

    LMMC_REAL_SET_D(&b->data[0], 2.0);
    LMMC_REAL_SET_D(&b->data[1], 0.0);
    LMMC_REAL_SET_D(&b->data[2], 1.0);
    LMMC_REAL_SET_D(&b->data[3], 2.0);

    st = lmmc_mat_mul(a, b, c);
    assert_int_equal(st, LMMC_STATUS_OK);

    assert_false(!lmmc_test_nearly_equal(c->data[0], 4.0, 1e-12) ||
                 !lmmc_test_nearly_equal(c->data[1], 4.0, 1e-12) ||
                 !lmmc_test_nearly_equal(c->data[2], 10.0, 1e-12) ||
                 !lmmc_test_nearly_equal(c->data[3], 8.0, 1e-12));
}

static void test_dense_dimension_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_mat_t *b = &fixture->b;
    lmmc_mat_t *c = &fixture->c;
    lmmc_vec_t *x = &fixture->x;
    lmmc_status_t st;

    st = lmmc_mat_create(0, 2, a);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_vec_create(0, x);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_mat_create(2, 3, a);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(2, 2, b);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(2, 2, c);
    assert_int_equal(st, LMMC_STATUS_OK);

    st = lmmc_mat_mul(a, b, c);
    assert_int_equal(st, LMMC_STATUS_DIMENSION_MISMATCH);
}

static void test_copy_and_frobenius_norm(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_mat_t *b = &fixture->b;
    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, b);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_copy(a, b);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_false(!lmmc_test_nearly_equal(b->data[0], 1.0, 1e-12) ||
                 !lmmc_test_nearly_equal(b->data[1], 2.0, 1e-12) ||
                 !lmmc_test_nearly_equal(b->data[2], 3.0, 1e-12) ||
                 !lmmc_test_nearly_equal(b->data[3], 4.0, 1e-12));

    lmmc_real_t fnorm = 0.0;
    st = lmmc_mat_norm_fro(b, &fnorm);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(fnorm, 5.47722557505, 1e-6));
}

static void test_transpose(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_mat_t *b = &fixture->b;
    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, b);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_transpose_to(a, b);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_false(!lmmc_test_nearly_equal(b->data[0], 1.0, 1e-12) ||
                 !lmmc_test_nearly_equal(b->data[1], 3.0, 1e-12) ||
                 !lmmc_test_nearly_equal(b->data[2], 2.0, 1e-12) ||
                 !lmmc_test_nearly_equal(b->data[3], 4.0, 1e-12));
}

static void test_wrapped_values(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_t *x = &fixture->x;
    lmmc_real_t *wrap_data = fixture->wrap_data;
    lmmc_mat_t *a = &fixture->a;
    lmmc_status_t st;

    st = lmmc_mat_wrap(2, 1, 1, wrap_data, a);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_false(!lmmc_test_nearly_equal(a->data[0], 5.0, 1e-12) ||
                 !lmmc_test_nearly_equal(a->data[1], 6.0, 1e-12));

    st = lmmc_vec_wrap(2, wrap_data, x);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_false(!lmmc_test_nearly_equal(x->data[0], 5.0, 1e-12) ||
                 !lmmc_test_nearly_equal(x->data[1], 6.0, 1e-12));
}

static void test_wrapped_matrix_vector_product(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *x = &fixture->x;
    lmmc_vec_t *y = &fixture->y;
    lmmc_status_t st;

    st = lmmc_vec_create(2, y);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_vec_mul(a, x, y);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_false(!lmmc_test_nearly_equal(y->data[0], 17.0, 1e-12) ||
                 !lmmc_test_nearly_equal(y->data[1], 39.0, 1e-12));
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->a);
    lmmc_mat_destroy(&fixture->b);
    lmmc_mat_destroy(&fixture->c);
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->y);
    free(fixture);
    *state = NULL;
    return 0;
}

static int setup_empty(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    if (fixture == NULL) {
        return -1;
    }
    LMMC_REAL_INIT(&fixture->wrap_data[0]);
    LMMC_REAL_SET_D(&fixture->wrap_data[0], 5.0);
    LMMC_REAL_INIT(&fixture->wrap_data[1]);
    LMMC_REAL_SET_D(&fixture->wrap_data[1], 6.0);
    return 0;
}

static int setup_matrix(void **state) {
    if (setup_empty(state) != 0) {
        return -1;
    }
    struct test_fixture *fixture = *state;
    lmmc_status_t st = lmmc_mat_create(2, 2, &fixture->a);
    if (st != LMMC_STATUS_OK) {
        teardown(state);
        return st;
    }
    LMMC_REAL_SET_D(&fixture->a.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->a.data[1], 2.0);
    LMMC_REAL_SET_D(&fixture->a.data[2], 3.0);
    LMMC_REAL_SET_D(&fixture->a.data[3], 4.0);
    return 0;
}

static int setup_wrapped_product(void **state) {
    if (setup_matrix(state) != 0) {
        return -1;
    }
    struct test_fixture *fixture = *state;
    lmmc_status_t st = lmmc_vec_wrap(2, fixture->wrap_data, &fixture->x);
    if (st != LMMC_STATUS_OK) {
        teardown(state);
        return st;
    }
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_matrix_product, setup_matrix, teardown),
        cmocka_unit_test_setup_teardown(test_dense_dimension_errors, setup_empty, teardown),
        cmocka_unit_test_setup_teardown(test_copy_and_frobenius_norm, setup_matrix, teardown),
        cmocka_unit_test_setup_teardown(test_transpose, setup_matrix, teardown),
        cmocka_unit_test_setup_teardown(test_wrapped_values, setup_empty, teardown),
        cmocka_unit_test_setup_teardown(test_wrapped_matrix_vector_product, setup_wrapped_product, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
