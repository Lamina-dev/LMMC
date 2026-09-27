/**
 * @file example_bicgstab_general.c
 * @brief 演示 LMMC 中 bicgstab general 相关接口的使用。
 */
#include <stdio.h>
#include "lmmc/lmmc.h"

static int create_system(lmmc_mat_t* a_dense, lmmc_sparse_mat_t* a_sparse, lmmc_vec_t* b, lmmc_vec_t* x) {
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_mat_create(3, 3, a_dense);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    a_dense->data[0] = 4.0; a_dense->data[1] = 1.0; a_dense->data[2] = 0.0;
    a_dense->data[3] = 2.0; a_dense->data[4] = 3.0; a_dense->data[5] = 1.0;
    a_dense->data[6] = 0.0; a_dense->data[7] = 1.0; a_dense->data[8] = 2.0;

    st = lmmc_sparse_from_dense(a_dense, 1e-14, a_sparse);
    if (st != LMMC_STATUS_OK) {
        printf("sparse_from_dense failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_vec_create(3, b);
    if (st != LMMC_STATUS_OK) {
        printf("vec_create b failed: %s\n", lmmc_status_string(st));
        return 1;
    }
    st = lmmc_vec_create(3, x);
    if (st != LMMC_STATUS_OK) {
        printf("vec_create x failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    b->data[0] = 6.0;
    b->data[1] = 11.0;
    b->data[2] = 8.0;
    return 0;
}

static int solve_unpreconditioned(const lmmc_sparse_mat_t* a_sparse, const lmmc_vec_t* b, lmmc_vec_t* x, lmmc_itersolve_config_t* cfg, lmmc_itersolve_result_t* result) {
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_itersolve_default_config(3, cfg);
    if (st != LMMC_STATUS_OK) {
        printf("itersolve_default_config failed: %s\n", lmmc_status_string(st));
        return 1;
    }
    cfg->max_iter = 100;

    st = lmmc_bicgstab_solve(a_sparse, b, NULL, cfg, x, result);
    if (st != LMMC_STATUS_OK) {
        printf("bicgstab (none) failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    printf("BiCGSTAB (no preconditioner):\n");
    printf("  converged = %d\n", result->converged);
    printf("  iterations = %zu\n", result->num_iter);
    printf("  final_residual = %.12e\n", result->final_residual_norm);
    printf("  x = [%.8f, %.8f, %.8f]\n", x->data[0], x->data[1], x->data[2]);
    return 0;
}

static int solve_jacobi(const lmmc_sparse_mat_t* a_sparse, const lmmc_vec_t* b, lmmc_vec_t* x, const lmmc_itersolve_config_t* cfg, lmmc_itersolve_result_t* result, lmmc_precond_t* jacobi) {
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_vec_fill(x, 0.0);
    if (st != LMMC_STATUS_OK) {
        printf("vec_fill failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_precond_create_jacobi(a_sparse, jacobi);
    if (st != LMMC_STATUS_OK) {
        printf("precond_create_jacobi failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_bicgstab_solve(a_sparse, b, jacobi, cfg, x, result);
    if (st != LMMC_STATUS_OK) {
        printf("bicgstab (jacobi) failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    printf("\nBiCGSTAB (Jacobi):\n");
    printf("  converged = %d\n", result->converged);
    printf("  iterations = %zu\n", result->num_iter);
    printf("  final_residual = %.12e\n", result->final_residual_norm);
    printf("  x = [%.8f, %.8f, %.8f]\n", x->data[0], x->data[1], x->data[2]);
    return 0;
}

static int solve_ilu0(const lmmc_sparse_mat_t* a_sparse, const lmmc_vec_t* b, lmmc_vec_t* x, const lmmc_itersolve_config_t* cfg, lmmc_itersolve_result_t* result, lmmc_precond_t* ilu0) {
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_vec_fill(x, 0.0);
    if (st != LMMC_STATUS_OK) {
        printf("vec_fill failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_precond_create_ilu0(a_sparse, ilu0);
    if (st != LMMC_STATUS_OK) {
        printf("precond_create_ilu0 failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_bicgstab_solve(a_sparse, b, ilu0, cfg, x, result);
    if (st != LMMC_STATUS_OK) {
        printf("bicgstab (ilu0) failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    printf("\nBiCGSTAB (ILU0):\n");
    printf("  converged = %d\n", result->converged);
    printf("  iterations = %zu\n", result->num_iter);
    printf("  final_residual = %.12e\n", result->final_residual_norm);
    printf("  x = [%.8f, %.8f, %.8f]\n", x->data[0], x->data[1], x->data[2]);
    return 0;
}

static int solve_ilut(const lmmc_sparse_mat_t* a_sparse, const lmmc_vec_t* b, lmmc_vec_t* x, const lmmc_itersolve_config_t* cfg, lmmc_itersolve_result_t* result, lmmc_precond_t* ilut) {
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_vec_fill(x, 0.0);
    if (st != LMMC_STATUS_OK) {
        printf("vec_fill failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_precond_create_ilut(a_sparse, 1e-12, 4, ilut);
    if (st != LMMC_STATUS_OK) {
        printf("precond_create_ilut failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_bicgstab_solve(a_sparse, b, ilut, cfg, x, result);
    if (st != LMMC_STATUS_OK) {
        printf("bicgstab (ilut) failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    printf("\nBiCGSTAB (ILUT, drop_tol=1e-12, fill=4):\n");
    printf("  converged = %d\n", result->converged);
    printf("  iterations = %zu\n", result->num_iter);
    printf("  final_residual = %.12e\n", result->final_residual_norm);
    printf("  x = [%.8f, %.8f, %.8f]\n", x->data[0], x->data[1], x->data[2]);
    return 0;
}

int main(void) {
    lmmc_mat_t a_dense = {0};
    lmmc_sparse_mat_t a_sparse = {0};
    lmmc_vec_t b = {0};
    lmmc_vec_t x = {0};
    lmmc_precond_t jacobi = {0};
    lmmc_precond_t ilu0 = {0};
    lmmc_precond_t ilut = {0};
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t result = {0};
    int rc = 0;

    rc = create_system(&a_dense, &a_sparse, &b, &x);
    if (rc != 0) {
            goto cleanup;
        }
    rc = solve_unpreconditioned(&a_sparse, &b, &x, &cfg, &result);
    if (rc != 0) {
            goto cleanup;
        }
    rc = solve_jacobi(&a_sparse, &b, &x, &cfg, &result, &jacobi);
    if (rc != 0) {
            goto cleanup;
        }
    rc = solve_ilu0(&a_sparse, &b, &x, &cfg, &result, &ilu0);
    if (rc != 0) {
            goto cleanup;
        }
    rc = solve_ilut(&a_sparse, &b, &x, &cfg, &result, &ilut);
    if (rc != 0) {
            goto cleanup;
        }

cleanup:
    lmmc_precond_destroy(&ilut);
    lmmc_precond_destroy(&ilu0);
    lmmc_precond_destroy(&jacobi);
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_sparse_destroy(&a_sparse);
    lmmc_mat_destroy(&a_dense);
    return rc;
}
