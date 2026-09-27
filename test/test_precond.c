/**
 * @file test_precond.c
 * 针对 LMMC 中 precond 相关接口的单元测试。
 */
#include <math.h>
#include <stdlib.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

typedef struct {
    lmmc_sparse_builder_t *builder;
    lmmc_sparse_mat_t A;
    lmmc_precond_t jacobi, ilu0, ilut, none;
    lmmc_vec_t v, out, Ax;
} precond_fixture_t;

static int setup(void **state) {
    precond_fixture_t *f = calloc(1, sizeof(*f));
    assert_non_null(f);
    *state = f;
    return 0;
}

static int teardown(void **state) {
    precond_fixture_t *f = *state;
    lmmc_vec_destroy(&f->Ax);
    lmmc_vec_destroy(&f->out);
    lmmc_vec_destroy(&f->v);
    lmmc_precond_destroy(&f->none);
    lmmc_precond_destroy(&f->ilut);
    lmmc_precond_destroy(&f->ilu0);
    lmmc_precond_destroy(&f->jacobi);
    lmmc_sparse_destroy(&f->A);
    lmmc_sparse_builder_destroy(f->builder);
    free(f);
    return 0;
}

static void build_spd_tridiag(precond_fixture_t *f, size_t n) {
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

static void build_general_dd(precond_fixture_t *f, size_t n) {
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

static void test_jacobi_application(void **state) {
    precond_fixture_t *f = *state;
    build_spd_tridiag(f, 5);
    assert_int_equal(lmmc_precond_create_jacobi(&f->A, &f->jacobi), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(5, &f->v), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(5, &f->out), LMMC_STATUS_OK);
    for (size_t i = 0; i < 5; ++i) {
        f->v.data[i] = (lmmc_real_t)(i + 1);
    }
    assert_int_equal(lmmc_precond_apply(&f->jacobi, &f->v, &f->out), LMMC_STATUS_OK);
    for (size_t i = 0; i < 5; ++i) {
        lmmc_real_t expected = (lmmc_real_t)(i + 1) / 4.0;
        assert_true(lmmc_test_nearly_equal(f->out.data[i], expected, 1e-12));
    }
}

static void test_ilu0_application_residual(void **state) {
    precond_fixture_t *f = *state;
    lmmc_real_t residual_norm = 0.0;
    build_spd_tridiag(f, 5);
    assert_int_equal(lmmc_precond_create_ilu0(&f->A, &f->ilu0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(5, &f->v), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(5, &f->out), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(5, &f->Ax), LMMC_STATUS_OK);
    for (size_t i = 0; i < 5; ++i) {
        f->v.data[i] = (lmmc_real_t)(i + 1);
    }
    assert_int_equal(lmmc_precond_apply(&f->ilu0, &f->v, &f->out), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_mat_vec_mul(&f->A, &f->out, &f->Ax), LMMC_STATUS_OK);
    for (size_t i = 0; i < 5; ++i) {
        lmmc_real_t diff = f->Ax.data[i] - f->v.data[i];
        residual_norm += diff * diff;
    }
    assert_true(sqrt(residual_norm) <= 1e-10);
}

static void test_identity_application(void **state) {
    precond_fixture_t *f = *state;
    assert_int_equal(lmmc_precond_create_none(5, &f->none), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(5, &f->v), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(5, &f->out), LMMC_STATUS_OK);
    for (size_t i = 0; i < 5; ++i) {
        f->v.data[i] = (lmmc_real_t)(i * 3.14 + 1.0);
    }
    assert_int_equal(lmmc_precond_apply(&f->none, &f->v, &f->out), LMMC_STATUS_OK);
    for (size_t i = 0; i < 5; ++i) {
        assert_true(lmmc_test_nearly_equal(f->out.data[i], f->v.data[i], 1e-15));
    }
}

static void test_ilut_application_residual(void **state) {
    precond_fixture_t *f = *state;
    lmmc_real_t residual_norm = 0.0;
    build_spd_tridiag(f, 10);
    assert_int_equal(lmmc_precond_create_ilut(&f->A, 1e-3, 10, &f->ilut), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(10, &f->v), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(10, &f->out), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(10, &f->Ax), LMMC_STATUS_OK);
    for (size_t i = 0; i < 10; ++i) {
        f->v.data[i] = (lmmc_real_t)(i + 1);
    }
    assert_int_equal(lmmc_precond_apply(&f->ilut, &f->v, &f->out), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_mat_vec_mul(&f->A, &f->out, &f->Ax), LMMC_STATUS_OK);
    for (size_t i = 0; i < 10; ++i) {
        lmmc_real_t diff = f->Ax.data[i] - f->v.data[i];
        residual_norm += diff * diff;
    }
    assert_true(sqrt(residual_norm) <= 1e-6);
}

static void test_jacobi_convergence(void **state) {
    precond_fixture_t *f = *state;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t res_precond = {0}, res_none = {0};
    const size_t n = 50;
    build_spd_tridiag(f, n);
    assert_int_equal(lmmc_precond_create_jacobi(&f->A, &f->jacobi), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->v), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->out), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->Ax), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; ++i) {
        f->v.data[i] = 1.0;
    }
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 200;
    assert_int_equal(lmmc_cg_solve(&f->A, &f->v, &f->jacobi, &cfg, &f->out, &res_precond), LMMC_STATUS_OK);
    assert_int_equal(res_precond.converged, 1);
    assert_int_equal(lmmc_cg_solve(&f->A, &f->v, NULL, &cfg, &f->Ax, &res_none), LMMC_STATUS_OK);
    assert_int_equal(res_none.converged, 1);
    assert_true(res_precond.num_iter <= res_none.num_iter);
}

static void test_ilu0_solver_residual(void **state) {
    precond_fixture_t *f = *state;
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    lmmc_real_t residual_norm = 0.0;
    const size_t n = 20;
    build_general_dd(f, n);
    assert_int_equal(lmmc_precond_create_ilu0(&f->A, &f->ilu0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->v), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->out), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &f->Ax), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; ++i) {
        f->v.data[i] = (lmmc_real_t)(i + 1);
    }
    assert_int_equal(lmmc_itersolve_default_config(n, &cfg), LMMC_STATUS_OK);
    cfg.max_iter = 200;
    assert_int_equal(lmmc_bicgstab_solve(&f->A, &f->v, &f->ilu0, &cfg, &f->out, &result), LMMC_STATUS_OK);
    assert_int_equal(result.converged, 1);
    assert_int_equal(lmmc_sparse_mat_vec_mul(&f->A, &f->out, &f->Ax), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; ++i) {
        lmmc_real_t diff = f->Ax.data[i] - f->v.data[i];
        residual_norm += diff * diff;
    }
    assert_true(sqrt(residual_norm) <= 1e-8);
}

static void test_preconditioner_null_arguments(void **state) {
    precond_fixture_t *f = *state;
    assert_int_equal(lmmc_precond_create_jacobi(NULL, &f->jacobi), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_precond_create_ilu0(NULL, &f->ilu0), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_precond_create_ilut(NULL, 1e-3, 10, &f->ilut), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_zero_diagonal_rejection(void **state) {
    precond_fixture_t *f = *state;
    assert_int_equal(lmmc_sparse_builder_create(3, 3, 9, &f->builder), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_builder_add(f->builder, 0, 0, 2.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_builder_add(f->builder, 0, 1, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_builder_add(f->builder, 1, 0, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_builder_add(f->builder, 1, 1, 0.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_builder_add(f->builder, 1, 2, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_builder_add(f->builder, 2, 1, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_builder_add(f->builder, 2, 2, 3.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_builder_build(f->builder, LMMC_SPARSE_CSR, &f->A), LMMC_STATUS_OK);
    lmmc_sparse_builder_destroy(f->builder);
    f->builder = NULL;
    assert_int_equal(lmmc_precond_create_jacobi(&f->A, &f->jacobi), LMMC_STATUS_SINGULAR_MATRIX);
}

static void test_preconditioner_size_sweep(void **state) {
    precond_fixture_t *f = *state;
    const size_t sizes[] = {1, 5, 50, 200};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        size_t n = sizes[s];
        build_spd_tridiag(f, n);
        assert_int_equal(lmmc_precond_create_jacobi(&f->A, &f->jacobi), LMMC_STATUS_OK);
        lmmc_precond_destroy(&f->jacobi);
        assert_int_equal(lmmc_precond_create_ilu0(&f->A, &f->ilu0), LMMC_STATUS_OK);
        lmmc_precond_destroy(&f->ilu0);
        assert_int_equal(lmmc_precond_create_none(n, &f->none), LMMC_STATUS_OK);
        lmmc_precond_destroy(&f->none);
        lmmc_sparse_destroy(&f->A);
    }
}

static void test_preconditioner_lifetimes(void **state) {
    precond_fixture_t *f = *state;
    build_spd_tridiag(f, 10);
    assert_int_equal(lmmc_precond_create_jacobi(&f->A, &f->jacobi), LMMC_STATUS_OK);
    assert_int_equal(lmmc_precond_create_ilu0(&f->A, &f->ilu0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_precond_create_ilut(&f->A, 1e-3, 10, &f->ilut), LMMC_STATUS_OK);
    assert_int_equal(lmmc_precond_create_none(10, &f->none), LMMC_STATUS_OK);
    lmmc_precond_destroy(&f->none);
    lmmc_precond_destroy(&f->ilut);
    lmmc_precond_destroy(&f->ilu0);
    lmmc_precond_destroy(&f->jacobi);
    lmmc_precond_destroy(NULL);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_jacobi_application, setup, teardown),
        cmocka_unit_test_setup_teardown(test_ilu0_application_residual, setup, teardown),
        cmocka_unit_test_setup_teardown(test_identity_application, setup, teardown),
        cmocka_unit_test_setup_teardown(test_ilut_application_residual, setup, teardown),
        cmocka_unit_test_setup_teardown(test_jacobi_convergence, setup, teardown),
        cmocka_unit_test_setup_teardown(test_ilu0_solver_residual, setup, teardown),
        cmocka_unit_test_setup_teardown(test_preconditioner_null_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_zero_diagonal_rejection, setup, teardown),
        cmocka_unit_test_setup_teardown(test_preconditioner_size_sweep, setup, teardown),
        cmocka_unit_test_setup_teardown(test_preconditioner_lifetimes, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
