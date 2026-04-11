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
    lmmc_vec_t x_nan = {0};
    lmmc_precond_t jacobi = {0};
    lmmc_precond_t ilu0 = {0};
    lmmc_precond_t ilut = {0};
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};

    lmmc_mat_t zero_dense = {0};
    lmmc_sparse_mat_t zero_sparse = {0};
    lmmc_sparse_mat_t large_sparse = {0};
    lmmc_vec_t zero_b = {0};
    lmmc_vec_t zero_x = {0};

    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_mat_create(3, 3, &a_dense);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    a_dense.data[0] = 4.0; a_dense.data[1] = 1.0; a_dense.data[2] = 0.0;
    a_dense.data[3] = 2.0; a_dense.data[4] = 3.0; a_dense.data[5] = 1.0;
    a_dense.data[6] = 0.0; a_dense.data[7] = 1.0; a_dense.data[8] = 2.0;

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

    b.data[0] = 6.0;
    b.data[1] = 11.0;
    b.data[2] = 8.0;

    st = lmmc_precond_create_jacobi(&a_sparse, &jacobi);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_precond_create_ilu0(&a_sparse, &ilu0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_precond_create_ilut(&a_sparse, 1e-12, 4, &ilut);
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

    st = lmmc_bicgstab_solve(&a_sparse, &b, NULL, &cfg, &x, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1) {
        rc = 1;
        goto cleanup;
    }
    if (!lmmc_test_nearly_equal(x.data[0], 1.0, 1e-9) ||
        !lmmc_test_nearly_equal(x.data[1], 2.0, 1e-9) ||
        !lmmc_test_nearly_equal(x.data[2], 3.0, 1e-9)) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_fill(&x, 0.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_bicgstab_solve(&a_sparse, &b, &jacobi, &cfg, &x, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_fill(&x, 0.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_bicgstab_solve(&a_sparse, &b, &ilu0, &cfg, &x, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_fill(&x, 0.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_bicgstab_solve(&a_sparse, &b, &ilut, &cfg, &x, &result);
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
    st = lmmc_bicgstab_solve(&a_sparse, &b, &jacobi, &cfg, &x, &result);
    if (st != LMMC_STATUS_OK || result.converged != 0 || result.num_iter != 1) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(2, &b_bad);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 1e-8;
    cfg.max_iter = 100;

    st = lmmc_precond_create_ilut(&a_sparse, -1.0, 4, &ilut);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_precond_create_ilut(&a_sparse, 1e-12, 0, &ilut);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto cleanup;
    }

    {
        const size_t n_large = 1500;
        size_t i = 0;
        lmmc_precond_t ilut_large = {0};

        st = lmmc_sparse_create_csr(n_large, n_large, n_large, &large_sparse);
        if (st != LMMC_STATUS_OK) {
            rc = 1;
            goto cleanup;
        }

        for (i = 0; i < n_large; ++i) {
            large_sparse.row_ptr[i] = i;
            large_sparse.col_idx[i] = i;
            large_sparse.values[i] = 1.0;
        }
        large_sparse.row_ptr[n_large] = n_large;

        st = lmmc_precond_create_ilut(&large_sparse, 1e-12, 4, &ilut_large);
        if (st != LMMC_STATUS_OK || ilut_large.impl == NULL) {
            rc = 1;
            goto cleanup;
        }

        lmmc_precond_destroy(&ilut_large);
    }

    st = lmmc_bicgstab_solve(&a_sparse, &b_bad, NULL, &cfg, &x, &result);
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
    x_nan.data[1] = 0.0;
    x_nan.data[2] = 0.0;

    st = lmmc_bicgstab_solve(&a_sparse, &b, &jacobi, &cfg, &x_nan, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(2, 2, &zero_dense);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_sparse_from_dense(&zero_dense, 0.0, &zero_sparse);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(2, &zero_b);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(2, &zero_x);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    zero_b.data[0] = 1.0;
    zero_b.data[1] = 1.0;

    {
        lmmc_precond_t ilu0_fail = {0};
        st = lmmc_precond_create_ilu0(&zero_sparse, &ilu0_fail);
        if (st != LMMC_STATUS_SINGULAR_MATRIX) {
            rc = 1;
            goto cleanup;
        }
        lmmc_precond_destroy(&ilu0_fail);
    }

    {
        lmmc_precond_t ilut_fail = {0};
        st = lmmc_precond_create_ilut(&zero_sparse, 1e-12, 2, &ilut_fail);
        if (st != LMMC_STATUS_SINGULAR_MATRIX) {
            rc = 1;
            goto cleanup;
        }
        lmmc_precond_destroy(&ilut_fail);
    }

    st = lmmc_bicgstab_solve(&zero_sparse, &zero_b, NULL, &cfg, &zero_x, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        rc = 1;
        goto cleanup;
    }

cleanup:
    lmmc_vec_destroy(&zero_x);
    lmmc_vec_destroy(&zero_b);
    lmmc_sparse_destroy(&large_sparse);
    lmmc_sparse_destroy(&zero_sparse);
    lmmc_mat_destroy(&zero_dense);
    lmmc_vec_destroy(&x_nan);
    lmmc_vec_destroy(&b_bad);
    lmmc_precond_destroy(&ilut);
    lmmc_precond_destroy(&ilu0);
    lmmc_precond_destroy(&jacobi);
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_sparse_destroy(&a_sparse);
    lmmc_mat_destroy(&a_dense);

    if (rc != 0) {
        printf("bicgstab test failed\n");
    }
    return rc;
}
