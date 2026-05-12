#include <stdio.h>
#include "lmmc/lmmc.h"

// 1. Define a custom logging callback
void my_solver_logger(size_t iter, lmmc_real_t residual_norm, void* user_data) {
    const char* prefix = (const char*)user_data;
    printf("[%s] Step %zu: Residual = %.4e\n", prefix, iter, residual_norm);
}

int main(void) {
    lmmc_mat_t a_dense = {0};
    lmmc_sparse_mat_t a = {0};
    lmmc_vec_t b = {0}, x = {0};
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t res = {0};
    lmmc_status_t st = LMMC_STATUS_OK;

    printf("=== LMMC Solver Logging Example ===\n\n");

    // 2. Setup a simple 10x10 Poisson-like problem (tridiagonal)
    const size_t n = 10;
    st = lmmc_mat_create(n, n, &a_dense);
    if (st != LMMC_STATUS_OK) return 1;
    st = lmmc_vec_create(n, &b);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a_dense); return 1; }
    st = lmmc_vec_create(n, &x);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&b); lmmc_mat_destroy(&a_dense); return 1; }

    for (size_t i = 0; i < n; ++i) {
        a_dense.data[i * n + i] = 2.0;
        if (i > 0) a_dense.data[i * n + (i - 1)] = -1.0;
        if (i < n - 1) a_dense.data[i * n + (i + 1)] = -1.0;
        b.data[i] = 1.0; // RHS
    }

    st = lmmc_sparse_from_dense(&a_dense, 1e-14, &a);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_mat_destroy(&a_dense);
        return 1;
    }

    st = lmmc_itersolve_default_config(n, &cfg);
    if (st != LMMC_STATUS_OK) {
        lmmc_sparse_destroy(&a); lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_mat_destroy(&a_dense);
        return 1;
    }
    cfg.log_cb = my_solver_logger;
    cfg.log_user_data = (void*)"CG-Poisson";
    cfg.max_iter = 100;
    cfg.rel_tol = 1e-6;

    printf("Starting CG solve with custom logger:\n");
    st = lmmc_cg_solve(&a, &b, NULL, &cfg, &x, &res);

    if (st == LMMC_STATUS_OK && res.converged) {
        printf("\nConvergence reached in %zu iterations.\n", res.num_iter);
        printf("Final residual: %.4e\n\n", res.final_residual_norm);
    }

    // 4. Demonstrate "verbose" built-in logging
    cfg.log_cb = NULL; // Disable custom logger
    cfg.verbose = 1;   // Enable built-in stdout logging
    lmmc_vec_fill(&x, 0.0); // Reset x
    
    printf("Starting BiCGSTAB solve with built-in verbose logging:\n");
    st = lmmc_bicgstab_solve(&a, &b, NULL, &cfg, &x, &res);

    if (st == LMMC_STATUS_OK && res.converged) {
        printf("\nConvergence reached in %zu iterations.\n", res.num_iter);
        printf("Final residual: %.4e\n\n", res.final_residual_norm);
    }

    // Cleanup
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_sparse_destroy(&a);
    lmmc_mat_destroy(&a_dense);

    return 0;
}
