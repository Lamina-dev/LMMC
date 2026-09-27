/**
 * @file example_solver_logging.c
 * @brief 演示 LMMC 中 solver logging 相关接口的使用。
 */
#include <stdio.h>
#include "lmmc/lmmc.h"


void my_solver_logger(const lmmc_diagnostic_t* diagnostic, void* user_data) {
    const char* prefix = (const char*)user_data;
    printf("[%s] Step %zu: Residual = %.4e\n", prefix, diagnostic->iteration,
           diagnostic->value_count > 0 ? diagnostic->values[0] : 0.0);
}

static lmmc_status_t create_poisson_system(
    size_t n, lmmc_mat_t* a_dense, lmmc_sparse_mat_t* a,
    lmmc_vec_t* b, lmmc_vec_t* x
) {
    lmmc_status_t st = lmmc_mat_create(n, n, a_dense);
    if (st != LMMC_STATUS_OK) {
            return st;
        }
    st = lmmc_vec_create(n, b);
    if (st != LMMC_STATUS_OK) {
            return st;
        }
    st = lmmc_vec_create(n, x);
    if (st != LMMC_STATUS_OK) {
            return st;
        }

    for (size_t i = 0; i < n; ++i) {
        a_dense->data[i * n + i] = 2.0;
        if (i > 0) a_dense->data[i * n + (i - 1)] = -1.0;
        if (i < n - 1) a_dense->data[i * n + (i + 1)] = -1.0;
        b->data[i] = 1.0;
    }
    return lmmc_sparse_from_dense(a_dense, 1e-14, a);
}

int main(void) {
    lmmc_mat_t a_dense = {0};
    lmmc_sparse_mat_t a = {0};
    lmmc_vec_t b = {0}, x = {0};
    lmmc_itersolve_config_t cfg = {0};
    lmmc_itersolve_result_t res = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    printf("=== LMMC Solver Logging Example ===\n\n");


    const size_t n = 10;
    st = create_poisson_system(n, &a_dense, &a, &b, &x);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    st = lmmc_itersolve_default_config(n, &cfg);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }
    cfg.diagnostics = (lmmc_diagnostic_sink_t){my_solver_logger, (void*)"CG-Poisson", LMMC_DIAGNOSTIC_TRACE};
    cfg.max_iter = 100;
    cfg.rel_tol = 1e-6;

    printf("Starting CG solve with custom logger:\n");
    st = lmmc_cg_solve(&a, &b, NULL, &cfg, &x, &res);

    if (st == LMMC_STATUS_OK && res.converged) {
        printf("\nConvergence reached in %zu iterations.\n", res.num_iter);
        printf("Final residual: %.4e\n\n", res.final_residual_norm);
    } else {
        rc = 1;
    }


    cfg.diagnostics.callback = NULL;
    st = lmmc_vec_fill(&x, 0.0);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto cleanup;
    }

    printf("Starting BiCGSTAB solve without diagnostics:\n");
    st = lmmc_bicgstab_solve(&a, &b, NULL, &cfg, &x, &res);

    if (st == LMMC_STATUS_OK && res.converged) {
        printf("\nConvergence reached in %zu iterations.\n", res.num_iter);
        printf("Final residual: %.4e\n\n", res.final_residual_norm);
    } else {
        rc = 1;
    }


cleanup:
    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_sparse_destroy(&a);
    lmmc_mat_destroy(&a_dense);

    return rc;
}
