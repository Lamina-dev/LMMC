/**
 * @file test_eigen_sym_property.c
 * 针对 LMMC 中 eigen sym property 相关接口的单元测试。
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
#include "test_common.h"

#define NUM_ITERATIONS 120

#define BASE_TOL 1e-9

struct test_fixture {
    struct test_eigen_sym_property_resources {
        lmmc_mat_t A;
        lmmc_eigen_sym_result_t res;
    } test_eigen_sym_property;
    struct test_eigen_sym_2x2_focused_resources {
        lmmc_mat_t A;
        lmmc_eigen_sym_result_t res;
    } test_eigen_sym_2x2_focused;
};

static double rand_double(double range) {
    return ((double)rand() / (double)RAND_MAX) * 2.0 * range - range;
}

static size_t rand_size(size_t min_n, size_t max_n) {
    return min_n + (size_t)(rand() % (int)(max_n - min_n + 1));
}

static void make_random_symmetric(lmmc_mat_t *A, double range) {
    size_t n = A->rows;
    size_t i, j;
    for (i = 0; i < n; ++i) {
        A->data[i * n + i] = rand_double(range);
        for (j = i + 1; j < n; ++j) {
            double v = rand_double(range);
            A->data[i * n + j] = v;
            A->data[j * n + i] = v;
        }
    }
}

static double mat_fro_norm(const lmmc_mat_t *A) {
    size_t n = A->rows;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double v = A->data[i * n + j];
            sum += v * v;
        }
    }
    return sqrt(sum);
}

static void check_eigen_orthogonality(int *iter, size_t *n, const lmmc_real_t *V) {
    double ortho_tol = BASE_TOL * (double)(*n);
    for (size_t i = 0; i < (*n); ++i) {
        for (size_t j = 0; j < (*n); ++j) {
            double dot = 0.0;
            for (size_t k = 0; k < (*n); ++k) {
                dot += V[k * (*n) + i] * V[k * (*n) + j];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            double diff = fabs(dot - expected);
            if (!(diff <= ortho_tol)) {
                fail_msg("orthogonality mismatch (V^T*V)[%" PRIuMAX "][%" PRIuMAX "]: expected %.1f got %.15g diff=%.3e tol=%.3e (iter=%d, n=%" PRIuMAX ")", (uintmax_t)(i), (uintmax_t)(j), expected, dot, diff, ortho_tol, (*iter), (uintmax_t)((*n)));
            }
        }
    }
}

static void test_eigen_sym_property(void **state) {
    struct test_fixture *fixture = *state;
    struct test_eigen_sym_property_resources *resources = &fixture->test_eigen_sym_property;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_size(2, 6);

        lmmc_status_t st = lmmc_mat_create(n, n, &resources->A);
        if (!(st == LMMC_STATUS_OK)) {
            fail_msg("mat_create failed (iter=%d, n=%" PRIuMAX ")", iter, (uintmax_t)(n));
        }

        double range = (iter % 4 == 0)   ? 1.0
                       : (iter % 4 == 1) ? 10.0
                       : (iter % 4 == 2) ? 0.1
                                         : 100.0;
        make_random_symmetric(&resources->A, range);

        double a_norm = mat_fro_norm(&resources->A);
        assert_true(isfinite(a_norm));
        double tol = BASE_TOL * (a_norm + 1.0) * (double)n;

        st = lmmc_eigen_symmetric(&resources->A, &resources->res);
        if (!(st == LMMC_STATUS_OK)) {
            fail_msg("eigen_symmetric failed status=%d (iter=%d, n=%" PRIuMAX ")", (int)st, iter, (uintmax_t)(n));
        }

        for (size_t i = 0; i + 1 < n; ++i) {
            if (!(resources->res.eigenvalues.data[i] <= resources->res.eigenvalues.data[i + 1] + tol)) {
                fail_msg("eigenvalues not ascending: lambda[%" PRIuMAX "]=%.15g > lambda[%" PRIuMAX "]=%.15g (iter=%d, n=%" PRIuMAX ")", (uintmax_t)(i), resources->res.eigenvalues.data[i], (uintmax_t)(i + 1), resources->res.eigenvalues.data[i + 1], iter, (uintmax_t)(n));
            }
        }

        const lmmc_real_t *V = resources->res.eigenvectors.data;
        const lmmc_real_t *lam = resources->res.eigenvalues.data;

        assert_int_equal(lmmc_test_check_eigen_reconstruction(iter, n, resources->A.data, tol, V, lam), 0);

        check_eigen_orthogonality(&iter, &n, V);

        lmmc_eigen_sym_result_destroy(&resources->res);
        lmmc_mat_destroy(&resources->A);
    }
}

static void test_eigen_sym_2x2_focused(void **state) {
    struct test_fixture *fixture = *state;
    struct test_eigen_sym_2x2_focused_resources *resources = &fixture->test_eigen_sym_2x2_focused;
    int iter;

    for (iter = 0; iter < 30; iter++) {

        assert_int_equal(lmmc_mat_create(2, 2, &resources->A), LMMC_STATUS_OK);

        double a = rand_double(10.0);
        double b = rand_double(10.0);
        double d = rand_double(10.0);
        resources->A.data[0] = a;
        resources->A.data[1] = b;
        resources->A.data[2] = b;
        resources->A.data[3] = d;

        lmmc_status_t st = lmmc_eigen_symmetric(&resources->A, &resources->res);
        if (!(st == LMMC_STATUS_OK)) {
            fail_msg("2x2 eigen failed (iter=%d)", iter);
        }

        double trace = a + d;
        double disc = sqrt((a - d) * (a - d) + 4.0 * b * b);
        double lam1 = (trace - disc) / 2.0;
        double lam2 = (trace + disc) / 2.0;

        double tol = 1e-10 * (fabs(a) + fabs(b) + fabs(d) + 1.0);

        if (!(fabs(resources->res.eigenvalues.data[0] - lam1) < tol)) {
            fail_msg("2x2 lambda[0]: expected %.15g got %.15g (iter=%d)", lam1, resources->res.eigenvalues.data[0], iter);
        }
        if (!(fabs(resources->res.eigenvalues.data[1] - lam2) < tol)) {
            fail_msg("2x2 lambda[1]: expected %.15g got %.15g (iter=%d)", lam2, resources->res.eigenvalues.data[1], iter);
        }

        lmmc_eigen_sym_result_destroy(&resources->res);
        lmmc_mat_destroy(&resources->A);
    }
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->test_eigen_sym_property.A);
    lmmc_eigen_sym_result_destroy(&fixture->test_eigen_sym_property.res);
    lmmc_mat_destroy(&fixture->test_eigen_sym_2x2_focused.A);
    lmmc_eigen_sym_result_destroy(&fixture->test_eigen_sym_2x2_focused.res);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const unsigned int seed = 0x4553594Du;
    srand(seed);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_eigen_sym_property, setup, teardown),
        cmocka_unit_test_setup_teardown(test_eigen_sym_2x2_focused, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
