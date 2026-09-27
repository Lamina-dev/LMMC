/**
 * @file test_bicgstab_autogen.c
 * 针对 LMMC 中 bicgstab autogen 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "lmmc/lmmc.h"

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_vec_t run_method_checked_x;
    lmmc_mat_t random_bicgstab_system_a_dense;
    lmmc_sparse_mat_t random_bicgstab_system_a_sparse;
    lmmc_vec_t random_bicgstab_system_x_true;
    lmmc_vec_t random_bicgstab_system_b;
    lmmc_precond_t random_bicgstab_system_jacobi;
    lmmc_precond_t random_bicgstab_system_ilu0;
    lmmc_precond_t random_bicgstab_system_ilut;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_precond_destroy(&fixture->random_bicgstab_system_ilut);
    lmmc_precond_destroy(&fixture->random_bicgstab_system_ilu0);
    lmmc_precond_destroy(&fixture->random_bicgstab_system_jacobi);
    lmmc_vec_destroy(&fixture->random_bicgstab_system_b);
    lmmc_vec_destroy(&fixture->random_bicgstab_system_x_true);
    lmmc_sparse_destroy(&fixture->random_bicgstab_system_a_sparse);
    lmmc_mat_destroy(&fixture->random_bicgstab_system_a_dense);
    lmmc_vec_destroy(&fixture->run_method_checked_x);
    free(fixture);
    return 0;
}

static double rand_unit(void) {
    return (double)rand() / (double)RAND_MAX;
}

static double rand_sym(void) {
    return 2.0 * rand_unit() - 1.0;
}

static double vec_error_norm2(const lmmc_vec_t *x, const lmmc_vec_t *y) {
    size_t i = 0;
    double s = 0.0;
    for (i = 0; i < x->size; ++i) {
        double d = x->data[i] - y->data[i];
        s += d * d;
    }
    return sqrt(s);
}

static void fill_diagonal_dominant_matrix(int n, double density, lmmc_mat_t *a_dense) {
    int i = 0;
    int j = 0;
    for (i = 0; i < n; ++i) {
        double row_sum_abs = 0.0;
        for (j = 0; j < n; ++j) {
            double v = 0.0;
            if (i == j) {
                continue;
            }
            if (abs(i - j) <= 2 || rand_unit() < density) {
                v = 0.2 * rand_sym();
                if (fabs(v) < 0.02) {
                    v = (v >= 0.0) ? 0.02 : -0.02;
                }
                a_dense->data[(size_t)i * a_dense->stride + (size_t)j] = v;
                row_sum_abs += fabs(v);
            }
        }
        a_dense->data[(size_t)i * a_dense->stride + (size_t)i] = row_sum_abs + 1.2 + 0.1 * rand_unit();
    }
}

static lmmc_status_t generate_diagonal_dominant_system(
    int n,
    double density,
    lmmc_mat_t *a_dense,
    lmmc_sparse_mat_t *a_sparse,
    lmmc_vec_t *x_true,
    lmmc_vec_t *b) {
    int i = 0;
    lmmc_status_t st = LMMC_STATUS_OK;

    st = lmmc_mat_create((size_t)n, (size_t)n, a_dense);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create((size_t)n, x_true);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create((size_t)n, b);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    fill_diagonal_dominant_matrix(n, density, a_dense);

    st = lmmc_sparse_from_dense(a_dense, 1e-14, a_sparse);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    for (i = 0; i < n; ++i) {
        x_true->data[i] = rand_sym();
    }

    return lmmc_sparse_mat_vec_mul(a_sparse, x_true, b);
}

static void run_method_checked(struct test_fixture *fixture,
                               const lmmc_sparse_mat_t *a,
                               const lmmc_vec_t *b,
                               const lmmc_vec_t *x_true,
                               const lmmc_precond_t *precond,
                               const lmmc_itersolve_config_t *cfg,
                               double tol) {

    lmmc_itersolve_result_t result = {0};
    lmmc_status_t st = lmmc_vec_create(b->size, &fixture->run_method_checked_x);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_bicgstab_solve(a, b, precond, cfg, &fixture->run_method_checked_x, &result);
    assert_true((st == LMMC_STATUS_OK) && (result.converged == 1));

    double error = vec_error_norm2(&fixture->run_method_checked_x, x_true);
    assert_true(isfinite(error) && error <= tol);

    lmmc_vec_destroy(&fixture->run_method_checked_x);
}

static void test_random_bicgstab_system(void **state) {
    struct test_fixture *fixture = *state;

    const int n = 20;
    const double density = 0.10;

    lmmc_itersolve_config_t cfg = {0};
    lmmc_status_t st = LMMC_STATUS_OK;

    st = generate_diagonal_dominant_system(n, density, &fixture->random_bicgstab_system_a_dense, &fixture->random_bicgstab_system_a_sparse, &fixture->random_bicgstab_system_x_true, &fixture->random_bicgstab_system_b);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_itersolve_default_config((size_t)n, &cfg);
    assert_true(st == LMMC_STATUS_OK);

    cfg.max_iter = (size_t)(n * 25);
    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 1e-10;

    st = lmmc_precond_create_jacobi(&fixture->random_bicgstab_system_a_sparse, &fixture->random_bicgstab_system_jacobi);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_precond_create_ilu0(&fixture->random_bicgstab_system_a_sparse, &fixture->random_bicgstab_system_ilu0);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_precond_create_ilut(&fixture->random_bicgstab_system_a_sparse, 1e-10, 8, &fixture->random_bicgstab_system_ilut);
    assert_true(st == LMMC_STATUS_OK);

    run_method_checked(fixture, &fixture->random_bicgstab_system_a_sparse, &fixture->random_bicgstab_system_b, &fixture->random_bicgstab_system_x_true, NULL, &cfg, 1e-5);
    run_method_checked(fixture, &fixture->random_bicgstab_system_a_sparse, &fixture->random_bicgstab_system_b, &fixture->random_bicgstab_system_x_true, &fixture->random_bicgstab_system_jacobi, &cfg, 1e-5);
    run_method_checked(fixture, &fixture->random_bicgstab_system_a_sparse, &fixture->random_bicgstab_system_b, &fixture->random_bicgstab_system_x_true, &fixture->random_bicgstab_system_ilu0, &cfg, 1e-5);
    run_method_checked(fixture, &fixture->random_bicgstab_system_a_sparse, &fixture->random_bicgstab_system_b, &fixture->random_bicgstab_system_x_true, &fixture->random_bicgstab_system_ilut, &cfg, 1e-5);

    lmmc_precond_destroy(&fixture->random_bicgstab_system_ilut);
    lmmc_precond_destroy(&fixture->random_bicgstab_system_ilu0);
    lmmc_precond_destroy(&fixture->random_bicgstab_system_jacobi);
    lmmc_vec_destroy(&fixture->random_bicgstab_system_b);
    lmmc_vec_destroy(&fixture->random_bicgstab_system_x_true);
    lmmc_sparse_destroy(&fixture->random_bicgstab_system_a_sparse);
    lmmc_mat_destroy(&fixture->random_bicgstab_system_a_dense);
}

int main(void) {
    char names[200][48];
    struct CMUnitTest tests[200];
    srand(20260411u);
    for (int i = 0; i < 200; ++i) {
        snprintf(names[i], sizeof(names[i]), "bicgstab_random_system_%d", i);
        tests[i] = (struct CMUnitTest)cmocka_unit_test_setup_teardown(
            test_random_bicgstab_system, setup, teardown);
        tests[i].name = names[i];
    }
    return cmocka_run_group_tests(tests, NULL, NULL);
}
