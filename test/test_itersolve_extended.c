/**
 * @file test_itersolve_extended.c
 * 针对 LMMC 中 itersolve extended 相关接口的单元测试。
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

typedef struct {
    lmmc_sparse_builder_t *builder;
    lmmc_sparse_mat_t A;
    lmmc_vec_t b, x, x_precond, ones, Ax;
    lmmc_precond_t jacobi;
} solver_fixture_t;

static int setup(void **state) {
    solver_fixture_t *f = calloc(1, sizeof(*f));
    assert_non_null(f);
    *state = f;
    return 0;
}

static int teardown(void **state) {
    solver_fixture_t *f = *state;
    lmmc_precond_destroy(&f->jacobi);
    lmmc_vec_destroy(&f->Ax);
    lmmc_vec_destroy(&f->ones);
    lmmc_vec_destroy(&f->x_precond);
    lmmc_vec_destroy(&f->x);
    lmmc_vec_destroy(&f->b);
    lmmc_sparse_destroy(&f->A);
    lmmc_sparse_builder_destroy(f->builder);
    free(f);
    return 0;
}

static void build_spd_tridiag(solver_fixture_t *f, size_t n) {
    assert_int_equal(lmmc_sparse_builder_create(n, n, 3 * n, &f->builder), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; ++i) {
        assert_int_equal(lmmc_sparse_builder_add(f->builder, i, i, 4.0), LMMC_STATUS_OK);
        if (i > 0) {
            assert_int_equal(lmmc_sparse_builder_add(f->builder, i, i - 1, -1.0), LMMC_STATUS_OK);
        }
        if (i + 1 < n) {
            assert_int_equal(lmmc_sparse_builder_add(f->builder, i, i + 1, -1.0), LMMC_STATUS_OK);
        }
    }
    assert_int_equal(lmmc_sparse_builder_build(f->builder, LMMC_SPARSE_CSR, &f->A), LMMC_STATUS_OK);
    lmmc_sparse_builder_destroy(f->builder);
    f->builder = NULL;
}

static void build_nonsym(solver_fixture_t *f, size_t n) {
    assert_int_equal(lmmc_sparse_builder_create(n, n, 3 * n, &f->builder), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; ++i) {
        assert_int_equal(lmmc_sparse_builder_add(f->builder, i, i, 5.0), LMMC_STATUS_OK);
        if (i > 0) {
            assert_int_equal(lmmc_sparse_builder_add(f->builder, i, i - 1, -1.0), LMMC_STATUS_OK);
        }
        if (i + 1 < n) {
            assert_int_equal(lmmc_sparse_builder_add(f->builder, i, i + 1, -2.0), LMMC_STATUS_OK);
        }
    }
    assert_int_equal(lmmc_sparse_builder_build(f->builder, LMMC_SPARSE_CSR, &f->A), LMMC_STATUS_OK);
    lmmc_sparse_builder_destroy(f->builder);
    f->builder = NULL;
}

static void build_illcond(solver_fixture_t *f, size_t n) {
    assert_int_equal(lmmc_sparse_builder_create(n, n, 3 * n, &f->builder), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; ++i) {
        double diag_val = (i == 0) ? 1e-12 : (double)(i + 1) * 100.0;
        assert_int_equal(lmmc_sparse_builder_add(f->builder, i, i, diag_val), LMMC_STATUS_OK);
        if (i > 0) {
            assert_int_equal(lmmc_sparse_builder_add(f->builder, i, i - 1, -1.0), LMMC_STATUS_OK);
        }
        if (i + 1 < n) {
            assert_int_equal(lmmc_sparse_builder_add(f->builder, i, i + 1, -1.0), LMMC_STATUS_OK);
        }
    }
    assert_int_equal(lmmc_sparse_builder_build(f->builder, LMMC_SPARSE_CSR, &f->A), LMMC_STATUS_OK);
    lmmc_sparse_builder_destroy(f->builder);
    f->builder = NULL;
}

static void compute_rhs_ones(solver_fixture_t *f) {
    assert_int_equal(lmmc_vec_create(f->A.rows, &f->ones), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(&f->ones, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_mat_vec_mul(&f->A, &f->ones, &f->b), LMMC_STATUS_OK);
    lmmc_vec_destroy(&f->ones);
}

static double compute_residual(solver_fixture_t *f) {
    double norm = 0.0;
    assert_int_equal(lmmc_vec_create(f->b.size, &f->Ax), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_mat_vec_mul(&f->A, &f->x, &f->Ax), LMMC_STATUS_OK);
    for (size_t i = 0; i < f->b.size; ++i) {
        double diff = f->Ax.data[i] - f->b.data[i];
        norm += diff * diff;
    }
    lmmc_vec_destroy(&f->Ax);
    return sqrt(norm);
}

static void test_cg_spd_residual(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 10;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    build_spd_tridiag(f, n);
    assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    compute_rhs_ones(f);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 200;
    assert_int_equal(lmmc_cg_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(compute_residual(f) < 1e-10);
}

static void test_bicgstab_nonsymmetric_residual(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 10;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    build_nonsym(f, n);
    assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    compute_rhs_ones(f);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 200;
    assert_int_equal(lmmc_bicgstab_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(compute_residual(f) < 1e-10);
}

static void test_gmres_nonsymmetric_residual(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 10;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    build_nonsym(f, n);
    assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    compute_rhs_ones(f);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 200;
    cfg.restart = 30;
    assert_int_equal(lmmc_gmres_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(compute_residual(f) < 1e-10);
}

static void test_cg_size_sweep(void **state) {
    solver_fixture_t *f = *state;
    const size_t sizes[] = {5, 20, 100};
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        size_t n = sizes[si];
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t result = {0};
        build_spd_tridiag(f, n);
        assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
        compute_rhs_ones(f);
        assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
        assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
        cfg.max_iter = 500;
        assert_int_equal(lmmc_cg_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_OK);
        assert_true(result.converged);
        lmmc_vec_destroy(&f->x);
        lmmc_vec_destroy(&f->b);
        lmmc_sparse_destroy(&f->A);
    }
}

static void test_nonsymmetric_solver_size_sweeps(void **state) {
    solver_fixture_t *f = *state;
    const size_t sizes[] = {5, 20, 100};
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        size_t n = sizes[si];
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t result = {0};
        build_nonsym(f, n);
        assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
        compute_rhs_ones(f);
        assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
        assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
        cfg.max_iter = 500;
        assert_int_equal(lmmc_bicgstab_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_OK);
        assert_true(result.converged);
        assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
        cfg.restart = 30;
        memset(&result, 0, sizeof(result));
        assert_int_equal(lmmc_gmres_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_OK);
        assert_true(result.converged);
        lmmc_vec_destroy(&f->x);
        lmmc_vec_destroy(&f->b);
        lmmc_sparse_destroy(&f->A);
    }
}

static void test_solver_iteration_limits(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 10;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    build_spd_tridiag(f, n);
    assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    compute_rhs_ones(f);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 1;
    (void)lmmc_cg_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result);
    assert_false(result.converged);
    assert_true(result.num_iter <= 1);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    memset(&result, 0, sizeof(result));
    (void)lmmc_bicgstab_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result);
    assert_false(result.converged);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    memset(&result, 0, sizeof(result));
    cfg.restart = 30;
    (void)lmmc_gmres_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result);
    assert_false(result.converged);
}

static void test_solver_dimension_errors(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 5;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    build_spd_tridiag(f, n);
    assert_int_equal(lmmc_vec_create(n + 2, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(&f->b, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    assert_int_equal(lmmc_cg_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_DIMENSION_MISMATCH);
    memset(&result, 0, sizeof(result));
    assert_int_equal(lmmc_bicgstab_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_DIMENSION_MISMATCH);
    memset(&result, 0, sizeof(result));
    cfg.restart = 30;
    assert_int_equal(lmmc_gmres_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_DIMENSION_MISMATCH);
}

static void test_ill_conditioned_iteration_limit(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 10;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    build_illcond(f, n);
    assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(&f->b, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 5;
    cfg.abs_tol = 1e-15;
    (void)lmmc_cg_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result);
    assert_false(result.converged);
}

static void test_exact_initial_guess(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 10;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    build_spd_tridiag(f, n);
    assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(&f->x, 1.0), LMMC_STATUS_OK);
    compute_rhs_ones(f);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 200;
    assert_int_equal(lmmc_cg_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(result.num_iter <= 1);
    assert_int_equal(lmmc_vec_fill(&f->x, 1.0), LMMC_STATUS_OK);
    memset(&result, 0, sizeof(result));
    assert_int_equal(lmmc_bicgstab_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(result.num_iter <= 1);
    assert_int_equal(lmmc_vec_fill(&f->x, 1.0), LMMC_STATUS_OK);
    memset(&result, 0, sizeof(result));
    cfg.restart = 30;
    assert_int_equal(lmmc_gmres_solve(&f->A, &f->b, NULL, &cfg, &f->x, &result), LMMC_STATUS_OK);
    assert_true(result.converged);
    assert_true(result.num_iter <= 1);
}

static void test_jacobi_acceleration(void **state) {
    solver_fixture_t *f = *state;
    const size_t n = 50;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t res_none = {0}, res_precond = {0};
    build_spd_tridiag(f, n);
    assert_int_equal(lmmc_vec_create(n, &f->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->x_precond), LMMC_STATUS_OK);
    compute_rhs_ones(f);
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 500;
    assert_int_equal(lmmc_vec_fill(&f->x, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_cg_solve(&f->A, &f->b, NULL, &cfg, &f->x, &res_none), LMMC_STATUS_OK);
    assert_true(res_none.converged);
    assert_int_equal(lmmc_precond_create_jacobi(&f->A, &f->jacobi), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_fill(&f->x_precond, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_cg_solve(&f->A, &f->b, &f->jacobi, &cfg, &f->x_precond, &res_precond), LMMC_STATUS_OK);
    assert_true(res_precond.converged);
    assert_true(res_precond.num_iter <= res_none.num_iter);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_cg_spd_residual, setup, teardown),
        cmocka_unit_test_setup_teardown(test_bicgstab_nonsymmetric_residual, setup, teardown),
        cmocka_unit_test_setup_teardown(test_gmres_nonsymmetric_residual, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cg_size_sweep, setup, teardown),
        cmocka_unit_test_setup_teardown(test_nonsymmetric_solver_size_sweeps, setup, teardown),
        cmocka_unit_test_setup_teardown(test_solver_iteration_limits, setup, teardown),
        cmocka_unit_test_setup_teardown(test_solver_dimension_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_ill_conditioned_iteration_limit, setup, teardown),
        cmocka_unit_test_setup_teardown(test_exact_initial_guess, setup, teardown),
        cmocka_unit_test_setup_teardown(test_jacobi_acceleration, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
