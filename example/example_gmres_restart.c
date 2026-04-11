#include <stdio.h>
#include "lmmc/lmmc.h"

int main(void) {
    lmmc_mat_t a_dense = {0};
    lmmc_sparse_mat_t a_sparse = {0};
    lmmc_vec_t b = {0};
    lmmc_vec_t x = {0};
    lmmc_precond_t jacobi = {0};
    lmmc_precond_t ilu0 = {0};
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_mat_create(3, 3, &a_dense);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    a_dense.data[0] = 4.0; a_dense.data[1] = 1.0; a_dense.data[2] = 0.0;
    a_dense.data[3] = 2.0; a_dense.data[4] = 3.0; a_dense.data[5] = 1.0;
    a_dense.data[6] = 0.0; a_dense.data[7] = 1.0; a_dense.data[8] = 2.0;

    st = lmmc_sparse_from_dense(&a_dense, 1e-14, &a_sparse);
    if (st != LMMC_STATUS_OK) {
        printf("sparse_from_dense failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_vec_create(3, &b);
    if (st != LMMC_STATUS_OK) {
        printf("vec_create b failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_vec_create(3, &x);
    if (st != LMMC_STATUS_OK) {
        printf("vec_create x failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    b.data[0] = 6.0;
    b.data[1] = 11.0;
    b.data[2] = 8.0;

    st = lmmc_itersolve_default_config(3, &cfg);
    if (st != LMMC_STATUS_OK) {
        printf("itersolve_default_config failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    cfg.max_iter = 60;
    cfg.restart = 2;

    st = lmmc_gmres_solve(&a_sparse, &b, NULL, &cfg, &x, &result);
    if (st != LMMC_STATUS_OK) {
        printf("gmres (none) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    printf("GMRES(m) (no preconditioner):\n");
    printf("  converged = %d\n", result.converged);
    printf("  iterations = %zu\n", result.num_iter);
    printf("  restart = %zu\n", cfg.restart);
    printf("  final_residual = %.12e\n", result.final_residual_norm);
    printf("  x = [%.8f, %.8f, %.8f]\n", x.data[0], x.data[1], x.data[2]);

    st = lmmc_vec_fill(&x, 0.0);
    if (st != LMMC_STATUS_OK) {
        printf("vec_fill failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_precond_create_jacobi(&a_sparse, &jacobi);
    if (st != LMMC_STATUS_OK) {
        printf("precond_create_jacobi failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_gmres_solve(&a_sparse, &b, &jacobi, &cfg, &x, &result);
    if (st != LMMC_STATUS_OK) {
        printf("gmres (jacobi) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    printf("\nGMRES(m) (Jacobi):\n");
    printf("  converged = %d\n", result.converged);
    printf("  iterations = %zu\n", result.num_iter);
    printf("  restart = %zu\n", cfg.restart);
    printf("  final_residual = %.12e\n", result.final_residual_norm);
    printf("  x = [%.8f, %.8f, %.8f]\n", x.data[0], x.data[1], x.data[2]);

    st = lmmc_vec_fill(&x, 0.0);
    if (st != LMMC_STATUS_OK) {
        printf("vec_fill failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_precond_create_ilu0(&a_sparse, &ilu0);
    if (st != LMMC_STATUS_OK) {
        printf("precond_create_ilu0 failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_gmres_solve(&a_sparse, &b, &ilu0, &cfg, &x, &result);
    if (st != LMMC_STATUS_OK) {
        printf("gmres (ilu0) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    printf("\nGMRES(m) (ILU0):\n");
    printf("  converged = %d\n", result.converged);
    printf("  iterations = %zu\n", result.num_iter);
    printf("  restart = %zu\n", cfg.restart);
    printf("  final_residual = %.12e\n", result.final_residual_norm);
    printf("  x = [%.8f, %.8f, %.8f]\n", x.data[0], x.data[1], x.data[2]);

cleanup:
    lmmc_precond_destroy(&ilu0);
    lmmc_precond_destroy(&jacobi);
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_sparse_destroy(&a_sparse);
    lmmc_mat_destroy(&a_dense);
    return rc;
}
