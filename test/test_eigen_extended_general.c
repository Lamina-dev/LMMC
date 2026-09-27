/**
 * @file test_eigen_extended_general.c
 * @brief 通用特征分解测试。
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
    struct test_nonsymmetric_eigenvalues_resources {
        lmmc_mat_t mat;
        lmmc_eigen_gen_result_t result;
    } test_nonsymmetric_eigenvalues;
    struct test_general_eigen_extreme_householder_scale_resources {
        lmmc_mat_t mat;
        lmmc_eigen_gen_result_t result;
    } test_general_eigen_extreme_householder_scale;
};

static void test_nonsymmetric_eigenvalues(void **state) {
    struct test_fixture *fixture = *state;
    struct test_nonsymmetric_eigenvalues_resources *resources = &fixture->test_nonsymmetric_eigenvalues;

    size_t n = 3;
    assert_int_equal(lmmc_mat_create(n, n, &resources->mat), LMMC_STATUS_OK);

    lmmc_real_t data[] = {
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0};
    memcpy(resources->mat.data, data, sizeof(data));

    lmmc_status_t s = lmmc_eigen_general(&resources->mat, &resources->result);
    assert_int_equal(s, LMMC_STATUS_OK);

    assert_true(resources->result.real_parts.size == 3);

    int found_real = 0;
    for (size_t i = 0; i < n; i++) {
        assert_true(isfinite(resources->result.real_parts.data[i]) && isfinite(resources->result.imag_parts.data[i]));
        if (fabs(resources->result.imag_parts.data[i]) < TOL) {
            assert_true(fabs(resources->result.real_parts.data[i] - 1.0) < TOL);
            found_real = 1;
        }
    }
    assert_true(found_real);

    int found_pos_imag = 0, found_neg_imag = 0;
    for (size_t i = 0; i < n; i++) {
        if (resources->result.imag_parts.data[i] > TOL)
            found_pos_imag = 1;
        if (resources->result.imag_parts.data[i] < -TOL)
            found_neg_imag = 1;
    }
    assert_true(found_pos_imag && found_neg_imag);

    lmmc_eigen_gen_result_destroy(&resources->result);
    lmmc_mat_destroy(&resources->mat);
}

static void test_general_eigen_extreme_householder_scale(void **state) {
    struct test_fixture *fixture = *state;
    struct test_general_eigen_extreme_householder_scale_resources *resources = &fixture->test_general_eigen_extreme_householder_scale;

    const double scale = 1.0e200;
    int found_one = 0, found_two = 0, found_three = 0;

    lmmc_status_t s = lmmc_mat_create(3, 3, &resources->mat);
    assert_int_equal(s, LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_fill(&resources->mat, 0.0),
                     LMMC_STATUS_OK);
    MAT_ELEM(&resources->mat, 0, 0) = scale;
    MAT_ELEM(&resources->mat, 1, 0) = scale;
    MAT_ELEM(&resources->mat, 1, 1) = 2.0 * scale;
    MAT_ELEM(&resources->mat, 2, 0) = scale;
    MAT_ELEM(&resources->mat, 2, 2) = 3.0 * scale;

    s = lmmc_eigen_general(&resources->mat, &resources->result);
    assert_int_equal(s, LMMC_STATUS_OK);
    for (size_t i = 0; i < 3; ++i) {
        assert_true(isfinite(resources->result.real_parts.data[i]) &&
                    isfinite(resources->result.imag_parts.data[i]));
        assert_true(fabs(resources->result.imag_parts.data[i]) < 1.0e-12 * scale);
        found_one |= fabs(resources->result.real_parts.data[i] / scale - 1.0) < 1.0e-10;
        found_two |= fabs(resources->result.real_parts.data[i] / scale - 2.0) < 1.0e-10;
        found_three |= fabs(resources->result.real_parts.data[i] / scale - 3.0) < 1.0e-10;
    }
    assert_true(found_one && found_two && found_three);

    lmmc_eigen_gen_result_destroy(&resources->result);
    lmmc_mat_destroy(&resources->mat);
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->test_nonsymmetric_eigenvalues.mat);
    lmmc_eigen_gen_result_destroy(&fixture->test_nonsymmetric_eigenvalues.result);
    lmmc_mat_destroy(&fixture->test_general_eigen_extreme_householder_scale.mat);
    lmmc_eigen_gen_result_destroy(&fixture->test_general_eigen_extreme_householder_scale.result);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_nonsymmetric_eigenvalues, setup, teardown),
        cmocka_unit_test_setup_teardown(test_general_eigen_extreme_householder_scale, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
