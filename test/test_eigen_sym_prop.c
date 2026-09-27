/**
 * @file test_eigen_sym_prop.c
 * 针对 LMMC 中 eigen sym prop 相关接口的单元测试。
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

#define NUM_ITERATIONS 80

#define BASE_TOL 1e-9

struct test_fixture {
    struct make_random_symmetric_BplusBT_resources {
        double *B;
    } make_random_symmetric_BplusBT;
    struct make_random_symmetric_BBT_resources {
        double *B;
    } make_random_symmetric_BBT;
    struct test_eigen_sym_property_resources {
        lmmc_mat_t A;
        lmmc_eigen_sym_result_t res;
    } test_eigen_sym_property;
};

static double rand_double(double range) {
    return ((double)rand() / (double)RAND_MAX) * 2.0 * range - range;
}

static size_t rand_size(size_t min_n, size_t max_n) {
    return min_n + (size_t)(rand() % (int)(max_n - min_n + 1));
}

static void make_random_symmetric_BplusBT(struct test_fixture *fixture, lmmc_mat_t *A, double range) {
    struct make_random_symmetric_BplusBT_resources *resources = &fixture->make_random_symmetric_BplusBT;
    size_t n = A->rows;
    resources->B = (double *)malloc(n * n * sizeof(double));
    assert_non_null(resources->B);
    for (size_t i = 0; i < n * n; ++i) {
        resources->B[i] = rand_double(range);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A->data[i * n + j] = resources->B[i * n + j] + resources->B[j * n + i];
        }
    }
    free(resources->B);
    resources->B = NULL;
}

static void make_random_symmetric_BBT(struct test_fixture *fixture, lmmc_mat_t *A, double range) {
    struct make_random_symmetric_BBT_resources *resources = &fixture->make_random_symmetric_BBT;
    size_t n = A->rows;
    resources->B = (double *)malloc(n * n * sizeof(double));
    assert_non_null(resources->B);
    for (size_t i = 0; i < n * n; ++i) {
        resources->B[i] = rand_double(range);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) {
                s += resources->B[i * n + k] * resources->B[j * n + k];
            }
            A->data[i * n + j] = s;
        }
    }
    free(resources->B);
    resources->B = NULL;
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

static void check_eigen_orthogonality(int *iter, size_t *n, double *tol_ortho, const lmmc_real_t *V) {
    for (size_t i = 0; i < (*n); ++i) {
        for (size_t j = 0; j < (*n); ++j) {
            double dot = 0.0;
            for (size_t k = 0; k < (*n); ++k) {
                dot += V[k * (*n) + i] * V[k * (*n) + j];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            double diff = fabs(dot - expected);
            if (!(diff <= (*tol_ortho))) {
                fail_msg("orthogonality mismatch (V^T*V)[%" PRIuMAX "][%" PRIuMAX "]: expected %.1f got %.15g diff=%.3e tol=%.3e (iter=%d, n=%" PRIuMAX ")", (uintmax_t)(i), (uintmax_t)(j), expected, dot, diff, (*tol_ortho), (*iter), (uintmax_t)((*n)));
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

        double range = (iter % 3 == 0)   ? 1.0
                       : (iter % 3 == 1) ? 10.0
                                         : 0.1;

        if (iter % 2 == 0) {
            make_random_symmetric_BplusBT(fixture, &resources->A, range);
        } else {
            make_random_symmetric_BBT(fixture, &resources->A, range);
        }

        double a_norm = mat_fro_norm(&resources->A);
        assert_true(isfinite(a_norm));

        double tol = BASE_TOL * (a_norm + 1.0) * (double)n * (double)n;
        double tol_ortho = BASE_TOL * (double)n * (double)n;

        st = lmmc_eigen_symmetric(&resources->A, &resources->res);
        if (!(st == LMMC_STATUS_OK)) {
            fail_msg("eigen_symmetric failed status=%d (iter=%d, n=%" PRIuMAX ")", (int)st, iter, (uintmax_t)(n));
        }

        const lmmc_real_t *V = resources->res.eigenvectors.data;
        const lmmc_real_t *lam = resources->res.eigenvalues.data;

        for (size_t i = 0; i + 1 < n; ++i) {
            if (!(lam[i] <= lam[i + 1] + tol)) {
                fail_msg("eigenvalues not ascending: lambda[%" PRIuMAX "]=%.15g > lambda[%" PRIuMAX "]=%.15g (iter=%d, n=%" PRIuMAX ")", (uintmax_t)(i), lam[i], (uintmax_t)(i + 1), lam[i + 1], iter, (uintmax_t)(n));
            }
        }

        assert_int_equal(lmmc_test_check_eigen_reconstruction(iter, n, resources->A.data, tol, V, lam), 0);

        check_eigen_orthogonality(&iter, &n, &tol_ortho, V);

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
    free(fixture->make_random_symmetric_BplusBT.B);
    free(fixture->make_random_symmetric_BBT.B);
    lmmc_mat_destroy(&fixture->test_eigen_sym_property.A);
    lmmc_eigen_sym_result_destroy(&fixture->test_eigen_sym_property.res);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    srand(0xC0FFEEu);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_eigen_sym_property, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
