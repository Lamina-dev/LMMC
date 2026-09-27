/**
 * @file test_eigen_sym.c
 * 针对 LMMC 中 eigen sym 相关接口的单元测试。
 */
#include <inttypes.h>
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

struct test_fixture {
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
};

static void test_null_input(void **state) {
    struct test_fixture *fixture = *state;


    lmmc_status_t s;

    s = lmmc_eigen_symmetric(NULL, &fixture->result);
    if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("eigen_symmetric(NULL, ...) should return INVALID_ARGUMENT, got %d", (int)s);
    }

    assert_int_equal(lmmc_mat_create(3, 3, &fixture->mat), LMMC_STATUS_OK);
    s = lmmc_eigen_symmetric(&fixture->mat, NULL);
    if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("eigen_symmetric(..., NULL) should return INVALID_ARGUMENT, got %d", (int)s);
    }
}

static void test_non_square(void **state) {
    struct test_fixture *fixture = *state;


    assert_int_equal(lmmc_mat_create(2, 3, &fixture->mat), LMMC_STATUS_OK);

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("eigen_symmetric on 2x3 should return INVALID_ARGUMENT, got %d", (int)s);
    }
}

static void test_1x1(void **state) {
    struct test_fixture *fixture = *state;


    assert_int_equal(lmmc_mat_create(1, 1, &fixture->mat), LMMC_STATUS_OK);
    fixture->mat.data[0] = 5.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("1x1 should succeed, got %d", (int)s);
    }
    if (!(fabs(fixture->result.eigenvalues.data[0] - 5.0) < TOL)) {
        fail_msg("1x1 eigenvalue should be 5.0, got %f", fixture->result.eigenvalues.data[0]);
    }
    if (!(fabs(fixture->result.eigenvectors.data[0] - 1.0) < TOL)) {
        fail_msg("1x1 eigenvector should be 1.0, got %f", fixture->result.eigenvectors.data[0]);
    }
}

static void test_2x2_diagonal(void **state) {
    struct test_fixture *fixture = *state;


    assert_int_equal(lmmc_mat_create(2, 2, &fixture->mat), LMMC_STATUS_OK);
    fixture->mat.data[0] = 3.0;
    fixture->mat.data[1] = 0.0;
    fixture->mat.data[2] = 0.0;
    fixture->mat.data[3] = 7.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("2x2 diagonal should succeed, got %d", (int)s);
    }

    if (!(fabs(fixture->result.eigenvalues.data[0] - 3.0) < TOL)) {
        fail_msg("eigenvalue[0] should be 3.0, got %f", fixture->result.eigenvalues.data[0]);
    }
    if (!(fabs(fixture->result.eigenvalues.data[1] - 7.0) < TOL)) {
        fail_msg("eigenvalue[1] should be 7.0, got %f", fixture->result.eigenvalues.data[1]);
    }
}

static void test_2x2_symmetric(void **state) {
    struct test_fixture *fixture = *state;


    assert_int_equal(lmmc_mat_create(2, 2, &fixture->mat), LMMC_STATUS_OK);
    fixture->mat.data[0] = 2.0;
    fixture->mat.data[1] = 1.0;
    fixture->mat.data[2] = 1.0;
    fixture->mat.data[3] = 2.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("2x2 symmetric should succeed, got %d", (int)s);
    }

    if (!(fabs(fixture->result.eigenvalues.data[0] - 1.0) < TOL)) {
        fail_msg("eigenvalue[0] should be 1.0, got %f", fixture->result.eigenvalues.data[0]);
    }
    if (!(fabs(fixture->result.eigenvalues.data[1] - 3.0) < TOL)) {
        fail_msg("eigenvalue[1] should be 3.0, got %f", fixture->result.eigenvalues.data[1]);
    }
}

static void test_3x3_identity(void **state) {
    struct test_fixture *fixture = *state;


    assert_int_equal(lmmc_mat_create(3, 3, &fixture->mat), LMMC_STATUS_OK);
    fixture->mat.data[0] = 1.0;
    fixture->mat.data[1] = 0.0;
    fixture->mat.data[2] = 0.0;
    fixture->mat.data[3] = 0.0;
    fixture->mat.data[4] = 1.0;
    fixture->mat.data[5] = 0.0;
    fixture->mat.data[6] = 0.0;
    fixture->mat.data[7] = 0.0;
    fixture->mat.data[8] = 1.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("3x3 identity should succeed, got %d", (int)s);
    }

    for (int i = 0; i < 3; i++) {
        if (!(fabs(fixture->result.eigenvalues.data[i] - 1.0) < TOL)) {
            fail_msg("eigenvalue[%d] should be 1.0, got %f", i, fixture->result.eigenvalues.data[i]);
        }
    }
}

static void test_3x3_symmetric(void **state) {
    struct test_fixture *fixture = *state;


    assert_int_equal(lmmc_mat_create(3, 3, &fixture->mat), LMMC_STATUS_OK);

    fixture->mat.data[0] = 4.0;
    fixture->mat.data[1] = 1.0;
    fixture->mat.data[2] = 1.0;
    fixture->mat.data[3] = 1.0;
    fixture->mat.data[4] = 4.0;
    fixture->mat.data[5] = 1.0;
    fixture->mat.data[6] = 1.0;
    fixture->mat.data[7] = 1.0;
    fixture->mat.data[8] = 4.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("3x3 symmetric should succeed, got %d", (int)s);
    }

    if (!(fabs(fixture->result.eigenvalues.data[0] - 3.0) < TOL)) {
        fail_msg("eigenvalue[0] should be 3.0, got %f", fixture->result.eigenvalues.data[0]);
    }
    if (!(fabs(fixture->result.eigenvalues.data[1] - 3.0) < TOL)) {
        fail_msg("eigenvalue[1] should be 3.0, got %f", fixture->result.eigenvalues.data[1]);
    }
    if (!(fabs(fixture->result.eigenvalues.data[2] - 6.0) < TOL)) {
        fail_msg("eigenvalue[2] should be 6.0, got %f", fixture->result.eigenvalues.data[2]);
    }
}

static void test_orthogonality(void **state) {
    struct test_fixture *fixture = *state;


    size_t n = 4;
    assert_int_equal(lmmc_mat_create(n, n, &fixture->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {
        5.0, 1.0, 2.0, 0.0,
        1.0, 4.0, 1.0, 1.0,
        2.0, 1.0, 6.0, 2.0,
        0.0, 1.0, 2.0, 3.0};
    for (size_t i = 0; i < n * n; i++) {
        fixture->mat.data[i] = data[i];
    }

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("4x4 symmetric should succeed, got %d", (int)s);
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t dot = 0.0;
            for (size_t k = 0; k < n; k++) {
                dot += fixture->result.eigenvectors.data[k * n + i] *
                       fixture->result.eigenvectors.data[k * n + j];
            }
            lmmc_real_t expected = (i == j) ? 1.0 : 0.0;
            if (!(fabs(dot - expected) < TOL)) {
                fail_msg("V^T*V[%" PRIuMAX "][%" PRIuMAX "] should be %f, got %f", (uintmax_t)(i), (uintmax_t)(j), expected, dot);
            }
        }
    }
}

static void test_reconstruction(void **state) {
    struct test_fixture *fixture = *state;


    size_t n = 3;
    assert_int_equal(lmmc_mat_create(n, n, &fixture->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {
        2.0, -1.0, 0.0,
        -1.0, 2.0, -1.0,
        0.0, -1.0, 2.0};
    for (size_t i = 0; i < n * n; i++) {
        fixture->mat.data[i] = data[i];
    }

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("3x3 tridiag should succeed, got %d", (int)s);
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t sum = 0.0;
            for (size_t k = 0; k < n; k++) {

                sum += fixture->result.eigenvectors.data[i * n + k] *
                       fixture->result.eigenvalues.data[k] *
                       fixture->result.eigenvectors.data[j * n + k];
            }
            if (!(fabs(sum - data[i * n + j]) < 1e-8)) {
                fail_msg("Reconstruction A[%" PRIuMAX "][%" PRIuMAX "]: expected %f, got %f", (uintmax_t)(i), (uintmax_t)(j), data[i * n + j], sum);
            }
        }
    }
}

static void test_ascending_order(void **state) {
    struct test_fixture *fixture = *state;


    size_t n = 4;
    assert_int_equal(lmmc_mat_create(n, n, &fixture->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {
        10.0, 1.0, 2.0, 3.0,
        1.0, 5.0, 1.0, 2.0,
        2.0, 1.0, 3.0, 1.0,
        3.0, 2.0, 1.0, 1.0};
    for (size_t i = 0; i < n * n; i++) {
        fixture->mat.data[i] = data[i];
    }

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("4x4 should succeed, got %d", (int)s);
    }

    for (size_t i = 0; i < n - 1; i++) {
        if (!(isfinite(fixture->result.eigenvalues.data[i]) && isfinite(fixture->result.eigenvalues.data[i + 1]) &&
              fixture->result.eigenvalues.data[i] <= fixture->result.eigenvalues.data[i + 1])) {
            fail_msg("eigenvalues not ascending: [%" PRIuMAX "]=%f > [%" PRIuMAX "]=%f", (uintmax_t)(i), fixture->result.eigenvalues.data[i], (uintmax_t)(i + 1), fixture->result.eigenvalues.data[i + 1]);
        }
    }
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    assert_non_null(fixture);
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_eigen_sym_result_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mat);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_null_input, setup, teardown),
        cmocka_unit_test_setup_teardown(test_non_square, setup, teardown),
        cmocka_unit_test_setup_teardown(test_1x1, setup, teardown),
        cmocka_unit_test_setup_teardown(test_2x2_diagonal, setup, teardown),
        cmocka_unit_test_setup_teardown(test_2x2_symmetric, setup, teardown),
        cmocka_unit_test_setup_teardown(test_3x3_identity, setup, teardown),
        cmocka_unit_test_setup_teardown(test_3x3_symmetric, setup, teardown),
        cmocka_unit_test_setup_teardown(test_orthogonality, setup, teardown),
        cmocka_unit_test_setup_teardown(test_reconstruction, setup, teardown),
        cmocka_unit_test_setup_teardown(test_ascending_order, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
