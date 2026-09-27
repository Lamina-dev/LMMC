/**
 * @file test_eigen_gen_prop.c
 * 针对 LMMC 中 eigen gen prop 相关接口的单元测试。
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

#define RNG_SEED UINT64_C(0xC0FFEE)

#define TRACE_TOL 1e-7

#define STRUCT_TOL 1e-9

#define IMAG_ZERO_TOL 1e-9

static uint64_t g_rng_state = RNG_SEED;

static void rng_seed(uint64_t s) { g_rng_state = (s == 0) ? UINT64_C(1) : s; }

static uint64_t rng_u64(void) {
    uint64_t x = g_rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng_state = x;
    return x * UINT64_C(2685821657736338717);
}

static double rng_uniform(double lo, double hi) {

    double u = (double)(rng_u64() >> 11) * (1.0 / 9007199254740992.0);
    return lo + (hi - lo) * u;
}

static void fill_mat_from_array(lmmc_mat_t *mat, const lmmc_real_t *src, size_t n) {
    size_t i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            mat->data[i * mat->stride + j] = src[i * n + j];
        }
    }
}

static double mat_trace(const lmmc_real_t *A, size_t n) {
    size_t i;
    double t = 0.0;
    for (i = 0; i < n; i++)
        t += A[i * n + i];
    return t;
}

static double mat_det_via_lu(const lmmc_real_t *A, size_t n) {
    double *M;
    size_t i, j, k, p;
    double sign;
    double det;

    M = (double *)malloc(n * n * sizeof(double));
    assert_non_null(M);
    for (i = 0; i < n * n; i++)
        M[i] = (double)A[i];

    sign = 1.0;
    for (k = 0; k < n; k++) {

        p = k;
        for (i = k + 1; i < n; i++) {
            if (fabs(M[i * n + k]) > fabs(M[p * n + k]))
                p = i;
        }
        if (fabs(M[p * n + k]) < 1e-300) {
            free(M);
            return 0.0;
        }
        if (p != k) {
            for (j = 0; j < n; j++) {
                double tmp = M[k * n + j];
                M[k * n + j] = M[p * n + j];
                M[p * n + j] = tmp;
            }
            sign = -sign;
        }
        for (i = k + 1; i < n; i++) {
            double f = M[i * n + k] / M[k * n + k];
            for (j = k + 1; j < n; j++) {
                M[i * n + j] -= f * M[k * n + j];
            }
            M[i * n + k] = 0.0;
        }
    }
    det = sign;
    for (i = 0; i < n; i++)
        det *= M[i * n + i];
    free(M);
    return det;
}

static void check_conjugate_pairs(const lmmc_eigen_gen_result_t *result, double tol) {
    size_t n = result->real_parts.size;
    int *matched = (int *)calloc(n, sizeof(int));
    size_t i, j;
    assert_non_null(matched);
    for (i = 0; i < n; i++) {
        double im;
        double re;
        int found;
        if (matched[i])
            continue;
        im = result->imag_parts.data[i];
        if (fabs(im) <= IMAG_ZERO_TOL) {
            matched[i] = 1;
            continue;
        }
        re = result->real_parts.data[i];
        found = 0;
        for (j = 0; j < n; j++) {
            double rj, ij;
            if (j == i || matched[j])
                continue;
            rj = result->real_parts.data[j];
            ij = result->imag_parts.data[j];
            if (fabs(re - rj) <= tol && fabs(im + ij) <= tol) {
                matched[i] = 1;
                matched[j] = 1;
                found = 1;
                break;
            }
        }
        if (!found) {
            free(matched);
            fail_msg("no conjugate partner for lambda_%" PRIuMAX " = %.6f + %.6fi", (uintmax_t)(i), re, im);
        }
    }
    free(matched);
}

static void check_trace_property(const lmmc_real_t *A, size_t n,
                                 const lmmc_eigen_gen_result_t *result,
                                 double tol, const char *label) {
    size_t i;
    double sum = 0.0;
    double tr = mat_trace(A, n);
    double scale;
    double diff;
    for (i = 0; i < n; i++)
        sum += result->real_parts.data[i];
    scale = fabs(tr) + 1.0;
    diff = fabs(sum - tr);
    if (!lmmc_test_nearly_equal(sum, tr, tol * scale)) {
        fail_msg("[%s] sum(real_parts)=%.10g but trace(A)=%.10g, diff=%.3e tol=%.3e",
                 label, sum, tr, diff, tol * scale);
    }
}

static void check_det_property(const lmmc_real_t *A, size_t n,
                               const lmmc_eigen_gen_result_t *result,
                               double tol, const char *label) {
    size_t i, j;
    int *used = NULL;
    double prod = 1.0;
    double det = mat_det_via_lu(A, n);
    double scale;
    double diff;

    used = (int *)calloc(n, sizeof(int));
    assert_non_null(used);

    for (i = 0; i < n; i++) {
        double re_i, im_i;
        if (used[i])
            continue;
        re_i = result->real_parts.data[i];
        im_i = result->imag_parts.data[i];
        if (fabs(im_i) <= IMAG_ZERO_TOL) {
            prod *= re_i;
            used[i] = 1;
            continue;
        }

        for (j = i + 1; j < n; j++) {
            double re_j, im_j;
            if (used[j])
                continue;
            re_j = result->real_parts.data[j];
            im_j = result->imag_parts.data[j];
            if (fabs(re_i - re_j) <= 1e-6 && fabs(im_i + im_j) <= 1e-6) {
                prod *= (re_i * re_i + im_i * im_i);
                used[i] = 1;
                used[j] = 1;
                break;
            }
        }
        if (!used[i]) {
            free(used);
            fail_msg("[%s] unmatched complex eigenvalue at index %" PRIuMAX, label, (uintmax_t)(i));
        }
    }
    free(used);

    scale = fabs(det) + 1.0;
    diff = fabs(prod - det);
    if (!lmmc_test_nearly_equal(prod, det, tol * scale)) {
        fail_msg("[%s] product of eigenvalues = %.10g, det(A) = %.10g, "
                 "diff=%.3e tol=%.3e",
                 label, prod, det, diff, tol * scale);
    }
}

typedef struct {
    lmmc_mat_t mat;
    lmmc_eigen_gen_result_t result;
} eigen_fixture_t;

static int setup_eigen(void **state) {
    eigen_fixture_t *f = calloc(1, sizeof(*f));
    assert_non_null(f);
    *state = f;
    return 0;
}

static int teardown_eigen(void **state) {
    eigen_fixture_t *f = *state;
    lmmc_eigen_gen_result_destroy(&f->result);
    lmmc_mat_destroy(&f->mat);
    free(f);
    return 0;
}

static void run_eigen_general(eigen_fixture_t *f, const lmmc_real_t *A,
                              size_t n, const char *label) {
    assert_int_equal(lmmc_mat_create(n, n, &f->mat), LMMC_STATUS_OK);
    fill_mat_from_array(&f->mat, A, n);
    lmmc_status_t st = lmmc_eigen_general(&f->mat, &f->result);
    if (st != LMMC_STATUS_OK) {
        fail_msg("[%s] lmmc_eigen_general returned %d", label, (int)st);
    }
    lmmc_mat_destroy(&f->mat);
}

static void test_rotation_matrix(void **state) {
    eigen_fixture_t *f = *state;
    const double theta = 0.7;
    const double c = cos(theta);
    const double s = sin(theta);
    lmmc_real_t A[4] = {
        (lmmc_real_t)c,
        -(lmmc_real_t)s,
        (lmmc_real_t)s,
        (lmmc_real_t)c,
    };
    lmmc_eigen_gen_result_t *result = &f->result;
    run_eigen_general(f, A, 2, "rotation_2x2");

    size_t found_pos = 0, found_neg = 0;
    for (size_t i = 0; i < 2; i++) {
        double re = result->real_parts.data[i];
        double im = result->imag_parts.data[i];
        if (fabs(re - c) <= STRUCT_TOL && fabs(im - s) <= STRUCT_TOL)
            found_pos = 1;
        if (fabs(re - c) <= STRUCT_TOL && fabs(im + s) <= STRUCT_TOL)
            found_neg = 1;
    }
    if (!found_pos || !found_neg) {
        fail_msg("rotation matrix eigenvalues not cos±i*sin "
                 "(found_pos=%" PRIuMAX ", found_neg=%" PRIuMAX ", got [%.6f%+.6fi, %.6f%+.6fi])",
                 (uintmax_t)(found_pos), (uintmax_t)(found_neg), result->real_parts.data[0], result->imag_parts.data[0], result->real_parts.data[1], result->imag_parts.data[1]);
    }
    check_conjugate_pairs(result, STRUCT_TOL);
    check_trace_property(A, 2, result, STRUCT_TOL, "rotation_2x2");
    check_det_property(A, 2, result, STRUCT_TOL, "rotation_2x2");
}

static void test_diagonal_matrix(void **state) {
    eigen_fixture_t *f = *state;
    lmmc_real_t A[9] = {
        2.5,
        0.0,
        0.0,
        0.0,
        -1.5,
        0.0,
        0.0,
        0.0,
        4.0,
    };
    lmmc_eigen_gen_result_t *result = &f->result;
    run_eigen_general(f, A, 3, "diagonal_3x3");

    for (size_t i = 0; i < 3; i++) {
        assert_true(lmmc_test_nearly_equal(result->imag_parts.data[i], 0.0, STRUCT_TOL));
    }
    double expected[3] = {2.5, -1.5, 4.0};
    int matched[3] = {0, 0, 0};
    for (size_t i = 0; i < 3; i++) {
        int found = 0;
        for (size_t j = 0; j < 3; j++) {
            if (matched[j])
                continue;
            if (fabs(result->real_parts.data[i] - expected[j]) <= STRUCT_TOL) {
                matched[j] = 1;
                found = 1;
                break;
            }
        }
        if (!found) {
            fail_msg("diagonal eigenvalue %.6f not in expected set", result->real_parts.data[i]);
        }
    }
    check_trace_property(A, 3, result, STRUCT_TOL, "diagonal_3x3");
    check_det_property(A, 3, result, STRUCT_TOL, "diagonal_3x3");
}

static void test_random_matrices(void **state) {
    eigen_fixture_t *f = *state;
    rng_seed(RNG_SEED);
    for (int t = 0; t < 32; t++) {
        lmmc_real_t A[25];
        char label[32];
        size_t n = (size_t)(2 + (int)(rng_u64() % 4));
        for (size_t i = 0; i < n * n; i++) {
            A[i] = (lmmc_real_t)rng_uniform(-2.0, 2.0);
        }
        snprintf(label, sizeof(label), "random_%dx%d_#%d", (int)n, (int)n, t);
        run_eigen_general(f, A, n, label);
        check_conjugate_pairs(&f->result, TRACE_TOL);
        check_trace_property(A, n, &f->result, TRACE_TOL, label);
        lmmc_eigen_gen_result_destroy(&f->result);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_rotation_matrix, setup_eigen, teardown_eigen),
        cmocka_unit_test_setup_teardown(test_diagonal_matrix, setup_eigen, teardown_eigen),
        cmocka_unit_test_setup_teardown(test_random_matrices, setup_eigen, teardown_eigen),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
