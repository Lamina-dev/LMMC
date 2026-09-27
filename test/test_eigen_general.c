/**
 * @file test_eigen_general.c
 * 针对 LMMC 中 eigen general 相关接口的单元测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"
#include "test_common.h"

#define TOL 1e-8

typedef struct {
    lmmc_mat_t mat;
    lmmc_eigen_gen_result_t result;
    int matched[3];
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
    lmmc_eigen_gen_result_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->mat);
    free(fixture);
    return 0;
}

typedef struct {
    double re;
    double im;
} cplx_t;

static cplx_t cplx_make(double r, double i) {
    cplx_t z;
    z.re = r;
    z.im = i;
    return z;
}

static cplx_t cplx_mul(cplx_t a, cplx_t b) {
    cplx_t r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

static double cplx_abs(cplx_t a) {
    return sqrt(a.re * a.re + a.im * a.im);
}

static cplx_t poly_eval_complex(const double *coeffs, int degree, cplx_t lambda) {
    cplx_t acc = cplx_make(coeffs[degree], 0.0);
    int k;
    for (k = degree - 1; k >= 0; k--) {
        acc = cplx_mul(acc, lambda);
        acc.re += coeffs[k];
    }
    return acc;
}

static void build_char_poly(const lmmc_real_t *A, size_t n, double *coeffs) {
    if (n == 2) {
        double a = A[0], b = A[1];
        double c = A[2], d = A[3];
        double tr = a + d;
        double det = a * d - b * c;

        coeffs[0] = det;
        coeffs[1] = -tr;
        coeffs[2] = 1.0;
        return;
    }
    if (n == 3) {
        double a11 = A[0], a12 = A[1], a13 = A[2];
        double a21 = A[3], a22 = A[4], a23 = A[5];
        double a31 = A[6], a32 = A[7], a33 = A[8];
        double tr = a11 + a22 + a33;

        double m11 = a22 * a33 - a23 * a32;
        double m22 = a11 * a33 - a13 * a31;
        double m33 = a11 * a22 - a12 * a21;
        double c1 = m11 + m22 + m33;

        double det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31);

        coeffs[0] = det;
        coeffs[1] = -c1;
        coeffs[2] = tr;
        coeffs[3] = -1.0;
        return;
    }
    fail_msg("Unsupported characteristic polynomial dimension: %" PRIuMAX, (uintmax_t)(n));
}

static void verify_char_poly_roots(const lmmc_real_t *A, size_t n,
                                   const lmmc_eigen_gen_result_t *result,
                                   double tol) {
    double coeffs[4];
    build_char_poly(A, n, coeffs);

    double anorm = 0.0;
    for (size_t i = 0; i < n * n; i++) {
        if (fabs(A[i]) > anorm)
            anorm = fabs(A[i]);
    }
    double scale = 1.0;
    for (size_t k = 0; k < n; k++)
        scale *= (1.0 + anorm);
    double scaled_tol = tol * (scale + 1.0);

    for (size_t i = 0; i < n; i++) {
        cplx_t lambda = cplx_make(result->real_parts.data[i],
                                  result->imag_parts.data[i]);
        cplx_t p = poly_eval_complex(coeffs, (int)n, lambda);
        double mag = cplx_abs(p);
        assert_true(isfinite(mag));
        assert_true(mag <= scaled_tol);
    }
}

static void verify_conjugate_pairs(eigen_fixture *fixture, double tol) {
    const lmmc_eigen_gen_result_t *result = &fixture->result;
    size_t n = result->real_parts.size;
    assert_true(n <= 3);
    int *matched = fixture->matched;
    memset(matched, 0, sizeof(fixture->matched));

    for (size_t i = 0; i < n; i++) {
        if (matched[i])
            continue;
        double im = result->imag_parts.data[i];
        if (fabs(im) <= tol) {
            matched[i] = 1;
            continue;
        }
        double re = result->real_parts.data[i];
        int found = 0;
        for (size_t j = 0; j < n; j++) {
            if (matched[j] || j == i)
                continue;
            double rj = result->real_parts.data[j];
            double ij = result->imag_parts.data[j];
            if (fabs(re - rj) <= tol && fabs(im + ij) <= tol) {
                matched[i] = 1;
                matched[j] = 1;
                found = 1;
                break;
            }
        }
        assert_true(found);
    }
}

static void check_eigen_properties(eigen_fixture *fixture,
                                   const lmmc_real_t *A, size_t n) {
    lmmc_mat_t *mat = &fixture->mat;
    assert_int_equal(lmmc_mat_create(n, n, mat), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            mat->data[i * mat->stride + j] = A[i * n + j];
        }
    }

    assert_int_equal(lmmc_eigen_general(mat, &fixture->result), LMMC_STATUS_OK);
    verify_char_poly_roots(A, n, &fixture->result, TOL);
    verify_conjugate_pairs(fixture, TOL);
    lmmc_eigen_gen_result_destroy(&fixture->result);
    lmmc_mat_destroy(mat);
}

static void test_diagonal_2x2(void **state) {
    lmmc_real_t A[] = {
        3.0, 0.0,
        0.0, 7.0};
    check_eigen_properties(*state, A, 2);
}

static void test_diagonal_3x3(void **state) {
    lmmc_real_t A[] = {
        -1.5, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 4.5};
    check_eigen_properties(*state, A, 3);
}

static void test_block_diagonal_3x3(void **state) {
    lmmc_real_t A[] = {
        2.0, 1.0, 0.0,
        1.0, 2.0, 0.0,
        0.0, 0.0, 5.0};
    check_eigen_properties(*state, A, 3);
}

static void test_rotation_2x2(void **state) {
    double theta = 0.7;
    double c = cos(theta);
    double s = sin(theta);
    lmmc_real_t A[] = {
        c, -s,
        s, c};
    check_eigen_properties(*state, A, 2);
}

static void test_skew_symmetric_2x2(void **state) {
    lmmc_real_t A[] = {
        0.0, -2.0,
        2.0, 0.0};
    check_eigen_properties(*state, A, 2);
}

static double next_uniform(eigen_fixture *fixture, double lo, double hi) {

    fixture->random_state = fixture->random_state * UINT64_C(1664525) + UINT64_C(1013904223);
    double u = (double)(fixture->random_state & UINT64_C(0xFFFFFFFF)) / 4294967296.0;
    return lo + (hi - lo) * u;
}

static void test_random_2x2_matrices(void **state) {
    eigen_fixture *fixture = *state;
    int trials = 8;
    int i;
    fixture->random_state = UINT64_C(12345);
    for (i = 0; i < trials; i++) {
        lmmc_real_t A[4];
        A[0] = (lmmc_real_t)next_uniform(fixture, -2.0, 2.0);
        A[1] = (lmmc_real_t)next_uniform(fixture, -2.0, 2.0);
        A[2] = (lmmc_real_t)next_uniform(fixture, -2.0, 2.0);
        A[3] = (lmmc_real_t)next_uniform(fixture, -2.0, 2.0);
        check_eigen_properties(fixture, A, 2);
    }
}

static void test_random_3x3_matrices(void **state) {
    eigen_fixture *fixture = *state;
    int trials = 8;
    int i, k;
    fixture->random_state = UINT64_C(67890);
    for (i = 0; i < trials; i++) {
        lmmc_real_t A[9];
        for (k = 0; k < 9; k++) {
            A[k] = (lmmc_real_t)next_uniform(fixture, -2.0, 2.0);
        }
        check_eigen_properties(fixture, A, 3);
    }
}

static void test_invalid_inputs(void **state) {
    eigen_fixture *fixture = *state;
    lmmc_mat_t *mat = &fixture->mat;
    lmmc_eigen_gen_result_t *result = &fixture->result;

    assert_int_equal(lmmc_eigen_general(NULL, result),
                     LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_mat_create(3, 3, mat), LMMC_STATUS_OK);
    assert_int_equal(lmmc_eigen_general(mat, NULL),
                     LMMC_STATUS_INVALID_ARGUMENT);
    lmmc_mat_destroy(mat);

    assert_int_equal(lmmc_mat_create(2, 3, mat), LMMC_STATUS_OK);
    assert_int_equal(lmmc_eigen_general(mat, result),
                     LMMC_STATUS_INVALID_ARGUMENT);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_invalid_inputs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_2x2, setup, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_3x3, setup, teardown),
        cmocka_unit_test_setup_teardown(test_block_diagonal_3x3, setup, teardown),
        cmocka_unit_test_setup_teardown(test_rotation_2x2, setup, teardown),
        cmocka_unit_test_setup_teardown(test_skew_symmetric_2x2, setup, teardown),
        cmocka_unit_test_setup_teardown(test_random_2x2_matrices, setup, teardown),
        cmocka_unit_test_setup_teardown(test_random_3x3_matrices, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
