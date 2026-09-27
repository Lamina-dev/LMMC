/**
 * @file test_eigen_extended_symmetric.c
 * @brief 对称矩阵特征分解测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"
#include "test_common.h"

#define TOL 1e-10
#define TOL_LOOSE 1e-8
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])
struct test_fixture {
    lmmc_mat_t mat;
    lmmc_eigen_sym_result_t result;
};

static void assert_av_equals_vd(
    const lmmc_mat_t* matrix, const lmmc_eigen_sym_result_t* result,
    lmmc_real_t tolerance)
{
    const size_t n = matrix->rows;
    for (size_t eigenvector = 0; eigenvector < n; ++eigenvector) {
        for (size_t row = 0; row < n; ++row) {
            lmmc_real_t av = 0.0;
            for (size_t column = 0; column < n; ++column) {
                av += MAT_ELEM(matrix, row, column) *
                    MAT_ELEM(&result->eigenvectors, column, eigenvector);
            }
            const lmmc_real_t vd =
                result->eigenvalues.data[eigenvector] *
                MAT_ELEM(&result->eigenvectors, row, eigenvector);
            if (!(fabs(av - vd) < tolerance)) {
                fail_msg(
                    "A*V != V*D at [%" PRIuMAX "][%" PRIuMAX
                    "]: %f vs %f",
                    (uintmax_t)row, (uintmax_t)eigenvector, av, vd);
            }
        }
    }
}

static void test_diagonal_eigenvalues(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_mat_create(4, 4, &fixture->mat), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_fill(&fixture->mat, 0.0),
                     LMMC_STATUS_OK);
    fixture->mat.data[0] = 5.0;
    fixture->mat.data[5] = 2.0;
    fixture->mat.data[10] = 8.0;
    fixture->mat.data[15] = 1.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("diagonal eigen should succeed, got %d", (int)s);
    }

    double expected[] = {1.0, 2.0, 5.0, 8.0};
    for (int i = 0; i < 4; i++) {
        if (!(fabs(fixture->result.eigenvalues.data[i] - expected[i]) < TOL)) {
            fail_msg("eigenvalue[%d]: expected %f, got %f", i, expected[i], fixture->result.eigenvalues.data[i]);
        }
    }
}

static void test_av_equals_vd_2x2(void **state) {
    struct test_fixture *fixture = *state;

    size_t n = 2;
    assert_int_equal(lmmc_mat_create(n, n, &fixture->mat), LMMC_STATUS_OK);

    fixture->mat.data[0] = 4.0;
    fixture->mat.data[1] = 1.0;
    fixture->mat.data[2] = 1.0;
    fixture->mat.data[3] = 3.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("2x2 eigen should succeed, got %d", (int)s);
    }

    assert_av_equals_vd(&fixture->mat, &fixture->result, TOL);
}

static void test_av_equals_vd_3x3(void **state) {
    struct test_fixture *fixture = *state;

    size_t n = 3;
    assert_int_equal(lmmc_mat_create(n, n, &fixture->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0};
    memcpy(fixture->mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("3x3 eigen should succeed, got %d", (int)s);
    }

    assert_av_equals_vd(&fixture->mat, &fixture->result, TOL);
}

static void test_av_equals_vd_5x5(void **state) {
    struct test_fixture *fixture = *state;

    size_t n = 5;
    assert_int_equal(lmmc_mat_create(n, n, &fixture->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {
        6.0, 2.0, 1.0, 0.0, 0.0,
        2.0, 5.0, 2.0, 1.0, 0.0,
        1.0, 2.0, 6.0, 2.0, 1.0,
        0.0, 1.0, 2.0, 5.0, 2.0,
        0.0, 0.0, 1.0, 2.0, 6.0};
    memcpy(fixture->mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("5x5 eigen should succeed, got %d", (int)s);
    }

    assert_av_equals_vd(&fixture->mat, &fixture->result, TOL);
}

static void test_eigenvector_orthogonality(void **state) {
    struct test_fixture *fixture = *state;

    size_t n = 4;
    assert_int_equal(lmmc_mat_create(n, n, &fixture->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {
        5.0, 1.0, 2.0, 0.0,
        1.0, 4.0, 1.0, 1.0,
        2.0, 1.0, 6.0, 2.0,
        0.0, 1.0, 2.0, 3.0};
    memcpy(fixture->mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("4x4 eigen should succeed, got %d", (int)s);
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double dot = 0.0;
            for (size_t k = 0; k < n; k++) {
                dot += fixture->result.eigenvectors.data[k * n + i] *
                       fixture->result.eigenvectors.data[k * n + j];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            if (!(fabs(dot - expected) < TOL)) {
                fail_msg("V^T*V[%" PRIuMAX "][%" PRIuMAX "] should be %f, got %f", (uintmax_t)(i), (uintmax_t)(j), expected, dot);
            }
        }
    }
}

static void test_repeated_eigenvalues(void **state) {
    struct test_fixture *fixture = *state;

    size_t n = 3;
    assert_int_equal(lmmc_mat_create(n, n, &fixture->mat), LMMC_STATUS_OK);

    assert_int_equal(lmmc_mat_fill(&fixture->mat, 0.0),
                     LMMC_STATUS_OK);
    fixture->mat.data[0] = 2.0;
    fixture->mat.data[4] = 2.0;
    fixture->mat.data[8] = 5.0;

    lmmc_status_t s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("repeated eigen should succeed, got %d", (int)s);
    }

    if (!(fabs(fixture->result.eigenvalues.data[0] - 2.0) < TOL)) {
        fail_msg("eigenvalue[0] should be 2.0, got %f", fixture->result.eigenvalues.data[0]);
    }
    if (!(fabs(fixture->result.eigenvalues.data[1] - 2.0) < TOL)) {
        fail_msg("eigenvalue[1] should be 2.0, got %f", fixture->result.eigenvalues.data[1]);
    }
    if (!(fabs(fixture->result.eigenvalues.data[2] - 5.0) < TOL)) {
        fail_msg("eigenvalue[2] should be 5.0, got %f", fixture->result.eigenvalues.data[2]);
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double dot = 0.0;
            for (size_t k = 0; k < n; k++) {
                dot += fixture->result.eigenvectors.data[k * n + i] *
                       fixture->result.eigenvectors.data[k * n + j];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            if (!(fabs(dot - expected) < TOL)) {
                fail_msg("Repeated: V^T*V[%" PRIuMAX "][%" PRIuMAX "] should be %f, got %f", (uintmax_t)(i), (uintmax_t)(j), expected, dot);
            }
        }
    }
}

static void test_symmetric_eigen_extreme_householder_scale(void **state) {
    struct test_fixture *fixture = *state;

    const double scale = 1.0e200;
    const double expected = sqrt(2.0);

    lmmc_status_t s = lmmc_mat_create(3, 3, &fixture->mat);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("extreme-scale symmetric matrix creation failed");
    }
    assert_int_equal(lmmc_mat_fill(&fixture->mat, 0.0),
                     LMMC_STATUS_OK);
    MAT_ELEM(&fixture->mat, 0, 1) = scale;
    MAT_ELEM(&fixture->mat, 1, 0) = scale;
    MAT_ELEM(&fixture->mat, 0, 2) = scale;
    MAT_ELEM(&fixture->mat, 2, 0) = scale;

    s = lmmc_eigen_symmetric(&fixture->mat, &fixture->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("symmetric eigen should survive representable Householder scale, got %d", (int)s);
    }
    if (!(isfinite(fixture->result.eigenvalues.data[0]) &&
          isfinite(fixture->result.eigenvalues.data[1]) &&
          isfinite(fixture->result.eigenvalues.data[2]))) {
        fail_msg("extreme-scale symmetric eigenvalues must be finite");
    }
    if (!(fabs(fixture->result.eigenvalues.data[0] / scale + expected) < 1.0e-10 &&
          fabs(fixture->result.eigenvalues.data[1] / scale) < 1.0e-10 &&
          fabs(fixture->result.eigenvalues.data[2] / scale - expected) < 1.0e-10)) {
        fail_msg("extreme-scale symmetric matrix preserves analytic eigenvalues");
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
        cmocka_unit_test_setup_teardown(test_diagonal_eigenvalues, setup, teardown),
        cmocka_unit_test_setup_teardown(test_av_equals_vd_2x2, setup, teardown),
        cmocka_unit_test_setup_teardown(test_av_equals_vd_3x3, setup, teardown),
        cmocka_unit_test_setup_teardown(test_av_equals_vd_5x5, setup, teardown),
        cmocka_unit_test_setup_teardown(test_eigenvector_orthogonality, setup, teardown),
        cmocka_unit_test_setup_teardown(test_repeated_eigenvalues, setup, teardown),
        cmocka_unit_test_setup_teardown(test_symmetric_eigen_extreme_householder_scale, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
