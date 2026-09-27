/**
 * @file test_svd.c
 * 针对 LMMC 中 svd 相关接口的单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"

#define TOL 1e-10
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])

struct test_fixture {
    lmmc_mat_t mat, pinv;
    lmmc_svd_result_t result;
};

static void test_null_input(void **state) {
    struct test_fixture *fixture = *state;


    lmmc_status_t s;

    s = lmmc_svd(NULL, &fixture->result);
    assert_int_equal(s, LMMC_STATUS_INVALID_ARGUMENT);

    assert_int_equal(lmmc_mat_create(3, 2, &fixture->mat), LMMC_STATUS_OK);
    s = lmmc_svd(&fixture->mat, NULL);
    assert_int_equal(s, LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_1x1(void **state) {
    struct test_fixture *fixture = *state;


    assert_int_equal(lmmc_mat_create(1, 1, &fixture->mat), LMMC_STATUS_OK);
    fixture->mat.data[0] = 5.0;

    lmmc_status_t s = lmmc_svd(&fixture->mat, &fixture->result);
    assert_int_equal(s, LMMC_STATUS_OK);
    assert_true(fabs(fixture->result.sigma.data[0] - 5.0) < TOL);
}

static void test_2x2_diagonal(void **state) {
    struct test_fixture *fixture = *state;


    assert_int_equal(lmmc_mat_create(2, 2, &fixture->mat), LMMC_STATUS_OK);
    fixture->mat.data[0] = 3.0;
    fixture->mat.data[1] = 0.0;
    fixture->mat.data[2] = 0.0;
    fixture->mat.data[3] = 7.0;

    lmmc_status_t s = lmmc_svd(&fixture->mat, &fixture->result);
    assert_int_equal(s, LMMC_STATUS_OK);

    assert_true(fabs(fixture->result.sigma.data[0] - 7.0) < TOL);
    assert_true(fabs(fixture->result.sigma.data[1] - 3.0) < TOL);
}

static void test_extreme_finite_scale(void **state) {
    struct test_fixture *fixture = *state;

    const lmmc_real_t scale = 1e200;
    const lmmc_real_t golden_ratio = (1.0 + sqrt(5.0)) * 0.5;

    assert_int_equal(lmmc_mat_create(2, 2, &fixture->mat), LMMC_STATUS_OK);
    fixture->mat.data[0] = scale;
    fixture->mat.data[1] = scale;
    fixture->mat.data[2] = 0.0;
    fixture->mat.data[3] = scale;

    lmmc_status_t s = lmmc_svd(&fixture->mat, &fixture->result);
    assert_int_equal(s, LMMC_STATUS_OK);
    assert_true(isfinite(fixture->result.sigma.data[0]) && isfinite(fixture->result.sigma.data[1]));
    assert_true(fabs(fixture->result.sigma.data[0] / scale - golden_ratio) < TOL &&
                fabs(fixture->result.sigma.data[1] / scale - 1.0 / golden_ratio) < TOL);
}

static void test_3x3_reconstruction(void **state) {
    struct test_fixture *fixture = *state;


    size_t m = 3, n = 3;
    assert_int_equal(lmmc_mat_create(m, n, &fixture->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0};
    for (size_t i = 0; i < m * n; i++)
        fixture->mat.data[i] = data[i];

    lmmc_status_t s = lmmc_svd(&fixture->mat, &fixture->result);
    assert_int_equal(s, LMMC_STATUS_OK);

    size_t p = (m < n) ? m : n;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t sum = 0.0;
            for (size_t k = 0; k < p; k++) {
                sum += MAT_ELEM(&fixture->result.U, i, k) *
                       fixture->result.sigma.data[k] *
                       MAT_ELEM(&fixture->result.Vt, k, j);
            }
            assert_true(fabs(sum - data[i * n + j]) < 1e-8);
        }
    }
}

static void test_orthogonality(void **state) {
    struct test_fixture *fixture = *state;


    size_t m = 4, n = 3;
    assert_int_equal(lmmc_mat_create(m, n, &fixture->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 13.0};
    for (size_t i = 0; i < m * n; i++)
        fixture->mat.data[i] = data[i];

    lmmc_status_t s = lmmc_svd(&fixture->mat, &fixture->result);
    assert_int_equal(s, LMMC_STATUS_OK);

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            lmmc_real_t dot = 0.0;
            for (size_t k = 0; k < m; k++) {
                dot += MAT_ELEM(&fixture->result.U, k, i) * MAT_ELEM(&fixture->result.U, k, j);
            }
            lmmc_real_t expected = (i == j) ? 1.0 : 0.0;
            assert_true(fabs(dot - expected) < TOL);
        }
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t dot = 0.0;
            for (size_t k = 0; k < n; k++) {
                dot += MAT_ELEM(&fixture->result.Vt, i, k) * MAT_ELEM(&fixture->result.Vt, j, k);
            }
            lmmc_real_t expected = (i == j) ? 1.0 : 0.0;
            assert_true(fabs(dot - expected) < TOL);
        }
    }
}

static void test_descending_order(void **state) {
    struct test_fixture *fixture = *state;


    size_t m = 4, n = 3;
    assert_int_equal(lmmc_mat_create(m, n, &fixture->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
        2.0, 1.0, 4.0};
    for (size_t i = 0; i < m * n; i++)
        fixture->mat.data[i] = data[i];

    lmmc_status_t s = lmmc_svd(&fixture->mat, &fixture->result);
    assert_int_equal(s, LMMC_STATUS_OK);

    size_t p = (m < n) ? m : n;
    for (size_t i = 0; i + 1 < p; i++) {
        assert_true(fixture->result.sigma.data[i] >= fixture->result.sigma.data[i + 1]);
    }

    for (size_t i = 0; i < p; i++) {
        assert_true(isfinite(fixture->result.sigma.data[i]) && fixture->result.sigma.data[i] >= 0.0);
    }
}

static void test_wide_matrix(void **state) {
    struct test_fixture *fixture = *state;


    size_t m = 2, n = 4;
    assert_int_equal(lmmc_mat_create(m, n, &fixture->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0};
    for (size_t i = 0; i < m * n; i++)
        fixture->mat.data[i] = data[i];

    lmmc_status_t s = lmmc_svd(&fixture->mat, &fixture->result);
    assert_int_equal(s, LMMC_STATUS_OK);

    assert_true(fixture->result.U.rows == m && fixture->result.U.cols == m);
    assert_true(fixture->result.sigma.size == m);
    assert_true(fixture->result.Vt.rows == n && fixture->result.Vt.cols == n);

    size_t p = m;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t sum = 0.0;
            for (size_t k = 0; k < p; k++) {
                sum += MAT_ELEM(&fixture->result.U, i, k) *
                       fixture->result.sigma.data[k] *
                       MAT_ELEM(&fixture->result.Vt, k, j);
            }
            assert_true(fabs(sum - data[i * n + j]) < 1e-8);
        }
    }
}

static void test_cond(void **state) {
    struct test_fixture *fixture = *state;


    assert_int_equal(lmmc_mat_create(2, 2, &fixture->mat), LMMC_STATUS_OK);

    fixture->mat.data[0] = 3.0;
    fixture->mat.data[1] = 0.0;
    fixture->mat.data[2] = 0.0;
    fixture->mat.data[3] = 1.0;

    lmmc_real_t cond_val;
    lmmc_status_t s = lmmc_cond(&fixture->mat, &cond_val);
    assert_int_equal(s, LMMC_STATUS_OK);
    assert_true(fabs(cond_val - 3.0) < TOL);
}

static void test_cond_singular(void **state) {
    struct test_fixture *fixture = *state;


    assert_int_equal(lmmc_mat_create(2, 2, &fixture->mat), LMMC_STATUS_OK);

    fixture->mat.data[0] = 1.0;
    fixture->mat.data[1] = 0.0;
    fixture->mat.data[2] = 0.0;
    fixture->mat.data[3] = 0.0;

    lmmc_real_t cond_val;
    lmmc_status_t s = lmmc_cond(&fixture->mat, &cond_val);
    assert_int_equal(s, LMMC_STATUS_OK);
    assert_true(isinf(cond_val));
}

static void test_pinv_null(void **state) {
    struct test_fixture *fixture = *state;


    assert_int_equal(lmmc_mat_create(2, 2, &fixture->mat), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(2, 2, &fixture->pinv), LMMC_STATUS_OK);

    lmmc_status_t s = lmmc_pinv(NULL, 0.0, &fixture->pinv);
    assert_int_equal(s, LMMC_STATUS_INVALID_ARGUMENT);

    s = lmmc_pinv(&fixture->mat, 0.0, NULL);
    assert_int_equal(s, LMMC_STATUS_INVALID_ARGUMENT);
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    assert_non_null(fixture);
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_svd_result_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->pinv);
    lmmc_mat_destroy(&fixture->mat);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_null_input, setup, teardown),
        cmocka_unit_test_setup_teardown(test_1x1, setup, teardown),
        cmocka_unit_test_setup_teardown(test_2x2_diagonal, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_finite_scale, setup, teardown),
        cmocka_unit_test_setup_teardown(test_3x3_reconstruction, setup, teardown),
        cmocka_unit_test_setup_teardown(test_orthogonality, setup, teardown),
        cmocka_unit_test_setup_teardown(test_descending_order, setup, teardown),
        cmocka_unit_test_setup_teardown(test_wide_matrix, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cond, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cond_singular, setup, teardown),
        cmocka_unit_test_setup_teardown(test_pinv_null, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
