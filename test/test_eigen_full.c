/**
 * @file test_eigen_full.c
 * Unit tests for lmmc_eigen_general_full (eigenvalues + eigenvectors).
 */
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"
#include "test_common.h"

#define TOL 1e-8
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])

typedef struct {
    lmmc_mat_t mat;
    lmmc_eigen_gen_full_result_t result;
    lmmc_real_t vr[20];
    lmmc_real_t vi[20];
    uint64_t random_state;
} eigen_fixture;

static int setup(void **state) {
    eigen_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    eigen_fixture *fixture = *state;
    lmmc_eigen_gen_full_result_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mat);
    free(fixture);
    return 0;
}

/** Verify ||A*v - lambda*v||_2 <= tol * (1 + ||A||_F) * ||v||_2. */
static void verify_eigenpair_residual(const lmmc_mat_t *A, size_t n,
                                      lmmc_real_t re, lmmc_real_t im,
                                      const lmmc_real_t *vr, const lmmc_real_t *vi,
                                      double tol) {
    double norm_A = 0.0;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            norm_A += MAT_ELEM(A, i, j) * MAT_ELEM(A, i, j);
    norm_A = sqrt(norm_A);

    double norm_v = 0.0;
    for (size_t i = 0; i < n; i++)
        norm_v += vr[i] * vr[i] + vi[i] * vi[i];
    norm_v = sqrt(norm_v);
    assert_true(norm_v >= 1e-15);

    double residual_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double Avr_i = 0.0;
        double Avi_i = 0.0;
        for (size_t j = 0; j < n; j++) {
            Avr_i += MAT_ELEM(A, i, j) * vr[j];
            Avi_i += MAT_ELEM(A, i, j) * vi[j];
        }
        /* (re + i*im)*(vr + i*vi) = (re*vr - im*vi) + i*(re*vi + im*vr). */
        double lv_real = re * vr[i] - im * vi[i];
        double lv_imag = re * vi[i] + im * vr[i];
        double dr = Avr_i - lv_real;
        double di = Avi_i - lv_imag;
        residual_sq += dr * dr + di * di;
    }
    double residual = sqrt(residual_sq);
    double bound = tol * (1.0 + norm_A) * norm_v;
    assert_true(isfinite(residual));
    assert_true(residual <= bound);
}

static void verify_eigenpairs(eigen_fixture *fixture, size_t n) {
    lmmc_eigen_gen_full_result_t *result = &fixture->result;
    assert_true(n <= 20);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            fixture->vr[j] = MAT_ELEM(&result->vectors_real, j, i);
            fixture->vi[j] = MAT_ELEM(&result->vectors_imag, j, i);
        }
        verify_eigenpair_residual(&fixture->mat, n,
                                  result->real_parts.data[i], result->imag_parts.data[i],
                                  fixture->vr, fixture->vi, TOL);
    }
}

/* Simple LCG for reproducible random matrices. */
static double next_uniform(eigen_fixture *fixture, double lo, double hi) {
    fixture->random_state = fixture->random_state * UINT64_C(1664525) + UINT64_C(1013904223);
    double u = (double)(fixture->random_state & UINT64_C(0xFFFFFFFF)) / 4294967296.0;
    return lo + (hi - lo) * u;
}

static void test_invalid_inputs(void **state) {
    eigen_fixture *fixture = *state;
    lmmc_mat_t *mat = &fixture->mat;
    lmmc_eigen_gen_full_result_t *result = &fixture->result;
    assert_int_equal(lmmc_eigen_general_full(NULL, result), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_mat_create(3, 3, mat), LMMC_STATUS_OK);
    assert_int_equal(lmmc_eigen_general_full(mat, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    lmmc_mat_destroy(mat);
    assert_int_equal(lmmc_mat_create(2, 3, mat), LMMC_STATUS_OK);
    assert_int_equal(lmmc_eigen_general_full(mat, result), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_1x1(void **state) {
    eigen_fixture *fixture = *state;
    lmmc_mat_t *mat = &fixture->mat;
    lmmc_eigen_gen_full_result_t *result = &fixture->result;
    assert_int_equal(lmmc_mat_create(1, 1, mat), LMMC_STATUS_OK);
    MAT_ELEM(mat, 0, 0) = 5.0;
    assert_int_equal(lmmc_eigen_general_full(mat, result), LMMC_STATUS_OK);
    assert_true(fabs(result->real_parts.data[0] - 5.0) < TOL);
    assert_true(fabs(result->imag_parts.data[0]) < TOL);
    assert_true(fabs(MAT_ELEM(&result->vectors_real, 0, 0) - 1.0) < TOL);
}

static void test_diagonal_2x2(void **state) {
    eigen_fixture *fixture = *state;
    lmmc_mat_t *mat = &fixture->mat;
    lmmc_eigen_gen_full_result_t *result = &fixture->result;
    assert_int_equal(lmmc_mat_create(2, 2, mat), LMMC_STATUS_OK);
    MAT_ELEM(mat, 0, 0) = 3.0;
    MAT_ELEM(mat, 0, 1) = 0.0;
    MAT_ELEM(mat, 1, 0) = 0.0;
    MAT_ELEM(mat, 1, 1) = 7.0;
    assert_int_equal(lmmc_eigen_general_full(mat, result), LMMC_STATUS_OK);
    verify_eigenpairs(fixture, 2);
    for (size_t i = 0; i < 2; i++) {
        if (fabs(result->imag_parts.data[i]) < TOL) {
            for (size_t j = 0; j < 2; j++)
                assert_true(fabs(MAT_ELEM(&result->vectors_imag, j, i)) < TOL);
        }
    }
}

static void test_extreme_scale_2x2(void **state) {
    eigen_fixture *fixture = *state;
    lmmc_mat_t *mat = &fixture->mat;
    lmmc_eigen_gen_full_result_t *result = &fixture->result;
    const double scale = 1.0e200;
    assert_int_equal(lmmc_mat_create(2, 2, mat), LMMC_STATUS_OK);
    MAT_ELEM(mat, 0, 0) = scale;
    MAT_ELEM(mat, 0, 1) = scale;
    MAT_ELEM(mat, 1, 0) = 0.0;
    MAT_ELEM(mat, 1, 1) = 2.0 * scale;
    assert_int_equal(lmmc_eigen_general_full(mat, result), LMMC_STATUS_OK);
    int found_one = 0;
    int found_two = 0;
    for (size_t col = 0; col < 2; ++col) {
        double lambda = result->real_parts.data[col] / scale;
        double v0 = MAT_ELEM(&result->vectors_real, 0, col);
        double v1 = MAT_ELEM(&result->vectors_real, 1, col);
        double r0 = (1.0 - lambda) * v0 + v1;
        double r1 = (2.0 - lambda) * v1;
        assert_true(isfinite(lambda) && isfinite(v0) && isfinite(v1));
        assert_true(hypot(r0, r1) < 1.0e-10);
        found_one |= fabs(lambda - 1.0) < 1.0e-10;
        found_two |= fabs(lambda - 2.0) < 1.0e-10;
    }
    assert_true(found_one && found_two);
}

static void test_widely_separated_2x2(void **state) {
    eigen_fixture *fixture = *state;
    lmmc_mat_t *mat = &fixture->mat;
    lmmc_eigen_gen_full_result_t *result = &fixture->result;
    assert_int_equal(lmmc_mat_create(2, 2, mat), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_fill(mat, 0.0), LMMC_STATUS_OK);
    MAT_ELEM(mat, 0, 0) = 1.0;
    MAT_ELEM(mat, 1, 1) = 1.0e-200;
    assert_int_equal(lmmc_eigen_general_full(mat, result), LMMC_STATUS_OK);
    int found_large = 0;
    int found_small = 0;
    for (size_t i = 0; i < 2; ++i) {
        found_large |= fabs(result->real_parts.data[i] - 1.0) < 1.0e-12;
        found_small |= fabs(result->real_parts.data[i] / 1.0e-200 - 1.0) < 1.0e-12;
    }
    assert_true(found_large && found_small);
}

static void test_extreme_scale_3x3(void **state) {
    eigen_fixture *fixture = *state;
    lmmc_mat_t *mat = &fixture->mat;
    lmmc_eigen_gen_full_result_t *result = &fixture->result;
    const double scale = 1.0e200;
    assert_int_equal(lmmc_mat_create(3, 3, mat), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_fill(mat, 0.0), LMMC_STATUS_OK);
    MAT_ELEM(mat, 0, 0) = scale;
    MAT_ELEM(mat, 1, 0) = scale;
    MAT_ELEM(mat, 1, 1) = 2.0 * scale;
    MAT_ELEM(mat, 2, 0) = scale;
    MAT_ELEM(mat, 2, 2) = 3.0 * scale;
    assert_int_equal(lmmc_eigen_general_full(mat, result), LMMC_STATUS_OK);
    int found[3] = {0, 0, 0};
    for (size_t col = 0; col < 3; ++col) {
        const double lambda = result->real_parts.data[col] / scale;
        double v[3];
        for (size_t row = 0; row < 3; ++row) {
            v[row] = MAT_ELEM(&result->vectors_real, row, col);
            assert_true(isfinite(v[row]));
        }
        const double r0 = (1.0 - lambda) * v[0];
        const double r1 = v[0] + (2.0 - lambda) * v[1];
        const double r2 = v[0] + (3.0 - lambda) * v[2];
        assert_true(hypot(hypot(r0, r1), r2) < 1.0e-8);
        found[0] |= fabs(lambda - 1.0) < 1.0e-8;
        found[1] |= fabs(lambda - 2.0) < 1.0e-8;
        found[2] |= fabs(lambda - 3.0) < 1.0e-8;
    }
    assert_true(found[0] && found[1] && found[2]);
}

static void test_rotation_2x2(void **state) {
    eigen_fixture *fixture = *state;
    lmmc_mat_t *mat = &fixture->mat;
    lmmc_eigen_gen_full_result_t *result = &fixture->result;
    double theta = 0.7;
    double c = cos(theta), s = sin(theta);
    assert_int_equal(lmmc_mat_create(2, 2, mat), LMMC_STATUS_OK);
    MAT_ELEM(mat, 0, 0) = c;
    MAT_ELEM(mat, 0, 1) = -s;
    MAT_ELEM(mat, 1, 0) = s;
    MAT_ELEM(mat, 1, 1) = c;
    assert_int_equal(lmmc_eigen_general_full(mat, result), LMMC_STATUS_OK);
    assert_true(fabs(result->imag_parts.data[0]) > 0.1);
    assert_true(fabs(result->imag_parts.data[0] + result->imag_parts.data[1]) < TOL);
    verify_eigenpairs(fixture, 2);
    /* Conjugate eigenvectors share real parts and negate imaginary parts. */
    for (size_t j = 0; j < 2; j++) {
        assert_true(fabs(MAT_ELEM(&result->vectors_real, j, 0) -
                         MAT_ELEM(&result->vectors_real, j, 1)) < TOL);
        assert_true(fabs(MAT_ELEM(&result->vectors_imag, j, 0) +
                         MAT_ELEM(&result->vectors_imag, j, 1)) < TOL);
    }
}

static void test_diagonal_3x3(void **state) {
    eigen_fixture *fixture = *state;
    lmmc_mat_t *mat = &fixture->mat;
    assert_int_equal(lmmc_mat_create(3, 3, mat), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_fill(mat, 0.0), LMMC_STATUS_OK);
    MAT_ELEM(mat, 0, 0) = 1.0;
    MAT_ELEM(mat, 1, 1) = 2.0;
    MAT_ELEM(mat, 2, 2) = 3.0;
    assert_int_equal(lmmc_eigen_general_full(mat, &fixture->result), LMMC_STATUS_OK);
    verify_eigenpairs(fixture, 3);
}

static void check_random_matrix(eigen_fixture *fixture, size_t n) {
    lmmc_mat_t *mat = &fixture->mat;
    lmmc_eigen_gen_full_result_t *result = &fixture->result;
    assert_int_equal(lmmc_mat_create(n, n, mat), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            MAT_ELEM(mat, i, j) = (lmmc_real_t)next_uniform(fixture, -2.0, 2.0);
    assert_int_equal(lmmc_eigen_general_full(mat, result), LMMC_STATUS_OK);
    verify_eigenpairs(fixture, n);
    for (size_t i = 0; i < n; i++) {
        if (fabs(result->imag_parts.data[i]) < TOL) {
            for (size_t j = 0; j < n; j++)
                assert_true(fabs(MAT_ELEM(&result->vectors_imag, j, i)) <= TOL);
        }
    }
    lmmc_eigen_gen_full_result_destroy(result);
    lmmc_mat_destroy(mat);
}

static void test_random_5x5(void **state) {
    eigen_fixture *fixture = *state;
    fixture->random_state = UINT64_C(11111);
    for (int trial = 0; trial < 5; trial++)
        check_random_matrix(fixture, 5);
}

static void test_random_10x10(void **state) {
    eigen_fixture *fixture = *state;
    fixture->random_state = UINT64_C(22222);
    for (int trial = 0; trial < 3; trial++)
        check_random_matrix(fixture, 10);
}

static void test_random_20x20(void **state) {
    eigen_fixture *fixture = *state;
    fixture->random_state = UINT64_C(33333);
    for (int trial = 0; trial < 2; trial++)
        check_random_matrix(fixture, 20);
}

static void test_destroy_null(void **state) {
    (void)state;
    lmmc_eigen_gen_full_result_destroy(NULL);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_invalid_inputs, setup, teardown),
        cmocka_unit_test(test_destroy_null),
        cmocka_unit_test_setup_teardown(test_1x1, setup, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_2x2, setup, teardown),
        cmocka_unit_test_setup_teardown(test_rotation_2x2, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_scale_2x2, setup, teardown),
        cmocka_unit_test_setup_teardown(test_widely_separated_2x2, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_scale_3x3, setup, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_3x3, setup, teardown),
        cmocka_unit_test_setup_teardown(test_random_5x5, setup, teardown),
        cmocka_unit_test_setup_teardown(test_random_10x10, setup, teardown),
        cmocka_unit_test_setup_teardown(test_random_20x20, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
