#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    lmmc_mat_t a_dense = {0};
    lmmc_sparse_mat_t a_sparse = {0};
    lmmc_vec_t b = {0};
    lmmc_vec_t x = {0};
    lmmc_vec_t b_bad = {0};
    lmmc_vec_t x_bad = {0};
    lmmc_vec_t x_nan = {0};
    lmmc_sparse_mat_t a_sing = {0};
    lmmc_mat_t sing_dense = {0};
    lmmc_precond_t jacobi = {0};
    lmmc_precond_t none_bad = {0};
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_mat_create(3, 3, &a_dense);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    LMMC_REAL_SET_D(&a_dense.data[0], 4.0); LMMC_REAL_SET_D(&a_dense.data[1], 1.0); LMMC_REAL_SET_D(&a_dense.data[2], 0.0);
    LMMC_REAL_SET_D(&a_dense.data[3], 1.0); LMMC_REAL_SET_D(&a_dense.data[4], 3.0); LMMC_REAL_SET_D(&a_dense.data[5], 1.0);
    LMMC_REAL_SET_D(&a_dense.data[6], 0.0); LMMC_REAL_SET_D(&a_dense.data[7], 1.0); LMMC_REAL_SET_D(&a_dense.data[8], 2.0);

    st = lmmc_sparse_from_dense(&a_dense, 1e-14, &a_sparse);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(3, &b);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(3, &x);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    LMMC_REAL_SET_D(&b.data[0], 6.0);
    LMMC_REAL_SET_D(&b.data[1], 10.0);
    LMMC_REAL_SET_D(&b.data[2], 8.0);

    st = lmmc_precond_create_jacobi(&a_sparse, &jacobi);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_itersolve_default_config(3, &cfg);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    cfg.max_iter = 100;

    st = lmmc_cg_solve(&a_sparse, &b, &jacobi, &cfg, &x, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1) {
        rc = 1;
        goto cleanup;
    }
    if (!lmmc_test_nearly_equal(x.data[0], 1.0, 1e-10) ||
        !lmmc_test_nearly_equal(x.data[1], 2.0, 1e-10) ||
        !lmmc_test_nearly_equal(x.data[2], 3.0, 1e-10)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_fill(&x, 0.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_cg_solve(&a_sparse, &b, NULL, &cfg, &x, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_fill(&x, 0.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    cfg.abs_tol = 1e-20;
    cfg.rel_tol = 1e-20;
    cfg.max_iter = 1;
    st = lmmc_cg_solve(&a_sparse, &b, &jacobi, &cfg, &x, &result);
    if (st != LMMC_STATUS_OK || result.converged != 0 || result.num_iter != 1) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(2, &b_bad);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(2, &x_bad);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_precond_create_none(2, &none_bad);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 1e-8;
    cfg.max_iter = 50;

    st = lmmc_cg_solve(&a_sparse, &b_bad, NULL, &cfg, &x, &result);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_cg_solve(&a_sparse, &b, &none_bad, &cfg, &x, &result);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(3, &x_nan);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    x_nan.data[0] = NAN;
    LMMC_REAL_SET_D(&x_nan.data[1], 0.0);
    LMMC_REAL_SET_D(&x_nan.data[2], 0.0);

    st = lmmc_cg_solve(&a_sparse, &b, &jacobi, &cfg, &x_nan, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(2, 2, &sing_dense);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    LMMC_REAL_SET_D(&sing_dense.data[0], 0.0); LMMC_REAL_SET_D(&sing_dense.data[1], 1.0);
    LMMC_REAL_SET_D(&sing_dense.data[2], 1.0); LMMC_REAL_SET_D(&sing_dense.data[3], 2.0);

    st = lmmc_sparse_from_dense(&sing_dense, 1e-14, &a_sing);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    {
        lmmc_precond_t jacobi_fail = {0};
        st = lmmc_precond_create_jacobi(&a_sing, &jacobi_fail);
        if (st != LMMC_STATUS_SINGULAR_MATRIX) {
            rc = 1;
            goto cleanup;
        }
        lmmc_precond_destroy(&jacobi_fail);
    }

cleanup:
    lmmc_sparse_destroy(&a_sing);
    lmmc_mat_destroy(&sing_dense);
    lmmc_precond_destroy(&none_bad);
    lmmc_precond_destroy(&jacobi);
    lmmc_vec_destroy(&x_nan);
    lmmc_vec_destroy(&x_bad);
    lmmc_vec_destroy(&b_bad);
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_sparse_destroy(&a_sparse);
    lmmc_mat_destroy(&a_dense);

    if (rc != 0) {
        printf("itersolve test failed\n");
    }
    return rc;
}
