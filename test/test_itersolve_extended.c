/**
 * @file test_itersolve_extended.c
 * @brief 针对 LMMC 中 itersolve extended 相关接口的单元测试。
 *
 * @internal
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"


static lmmc_status_t build_spd_tridiag(size_t n, lmmc_sparse_mat_t* out) {
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {
        lmmc_sparse_builder_add(builder, i, i, 4.0);
        if (i > 0) lmmc_sparse_builder_add(builder, i, i - 1, -1.0);
        if (i < n - 1) lmmc_sparse_builder_add(builder, i, i + 1, -1.0);
    }
    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}


static lmmc_status_t build_nonsym(size_t n, lmmc_sparse_mat_t* out) {
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {
        lmmc_sparse_builder_add(builder, i, i, 5.0);
        if (i > 0) lmmc_sparse_builder_add(builder, i, i - 1, -1.0);
        if (i < n - 1) lmmc_sparse_builder_add(builder, i, i + 1, -2.0);
    }
    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}


static double compute_residual(const lmmc_sparse_mat_t* A,
                               const lmmc_vec_t* x,
                               const lmmc_vec_t* b) {
    lmmc_vec_t Ax = {0};
    lmmc_vec_create(b->size, &Ax);
    lmmc_sparse_mat_vec_mul(A, x, &Ax);
    double norm = 0.0;
    for (size_t i = 0; i < b->size; i++) {
        double diff = Ax.data[i] - b->data[i];
        norm += diff * diff;
    }
    lmmc_vec_destroy(&Ax);
    return sqrt(norm);
}


static lmmc_status_t build_illcond(size_t n, lmmc_sparse_mat_t* out) {
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t i = 0; i < n; i++) {

        double diag_val = (i == 0) ? 1e-12 : (double)(i + 1) * 100.0;
        lmmc_sparse_builder_add(builder, i, i, diag_val);
        if (i > 0) lmmc_sparse_builder_add(builder, i, i - 1, -1.0);
        if (i < n - 1) lmmc_sparse_builder_add(builder, i, i + 1, -1.0);
    }
    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}


static void compute_rhs_ones(const lmmc_sparse_mat_t* A, lmmc_vec_t* b) {
    lmmc_vec_t ones = {0};
    lmmc_vec_create(A->rows, &ones);
    lmmc_vec_fill(&ones, 1.0);
    lmmc_sparse_mat_vec_mul(A, &ones, b);
    lmmc_vec_destroy(&ones);
}

int main(void) {
    int rc = 0;
    lmmc_status_t st;


    {
        const size_t n = 10;
        lmmc_sparse_mat_t A = {0};
        lmmc_vec_t b = {0}, x = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t result = {0};

        st = build_spd_tridiag(n, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_create(n, &b);
        lmmc_vec_create(n, &x);


        compute_rhs_ones(&A, &b);

        lmmc_vec_fill(&x, 0.0);
        lmmc_itersolve_default_config(n, &cfg);
        cfg.max_iter = 200;

        st = lmmc_cg_solve(&A, &b, NULL, &cfg, &x, &result);
        if (st != LMMC_STATUS_OK || !result.converged) {
            printf("8.1 CG SPD tridiag failed: st=%d converged=%d\n", st, result.converged);
            rc = 1;
        } else {
            double res = compute_residual(&A, &x, &b);
            if (res >= 1e-10) {
                printf("8.1 CG residual too large: %e\n", res);
                rc = 1;
            }
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_sparse_destroy(&A);
        if (rc) goto done;
    }


    {
        const size_t n = 10;
        lmmc_sparse_mat_t A = {0};
        lmmc_vec_t b = {0}, x = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t result = {0};

        st = build_nonsym(n, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_create(n, &b);
        lmmc_vec_create(n, &x);

        compute_rhs_ones(&A, &b);

        lmmc_vec_fill(&x, 0.0);
        lmmc_itersolve_default_config(n, &cfg);
        cfg.max_iter = 200;

        st = lmmc_bicgstab_solve(&A, &b, NULL, &cfg, &x, &result);
        if (st != LMMC_STATUS_OK || !result.converged) {
            printf("8.2 BiCGSTAB nonsym failed: st=%d converged=%d\n", st, result.converged);
            rc = 1;
        } else {
            double res = compute_residual(&A, &x, &b);
            if (res >= 1e-10) {
                printf("8.2 BiCGSTAB residual too large: %e\n", res);
                rc = 1;
            }
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_sparse_destroy(&A);
        if (rc) goto done;
    }


    {
        const size_t n = 10;
        lmmc_sparse_mat_t A = {0};
        lmmc_vec_t b = {0}, x = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t result = {0};

        st = build_nonsym(n, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_create(n, &b);
        lmmc_vec_create(n, &x);

        compute_rhs_ones(&A, &b);

        lmmc_vec_fill(&x, 0.0);
        lmmc_itersolve_default_config(n, &cfg);
        cfg.max_iter = 200;
        cfg.restart = 30;

        st = lmmc_gmres_solve(&A, &b, NULL, &cfg, &x, &result);
        if (st != LMMC_STATUS_OK || !result.converged) {
            printf("8.3 GMRES nonsym failed: st=%d converged=%d\n", st, result.converged);
            rc = 1;
        } else {
            double res = compute_residual(&A, &x, &b);
            if (res >= 1e-10) {
                printf("8.3 GMRES residual too large: %e\n", res);
                rc = 1;
            }
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_sparse_destroy(&A);
        if (rc) goto done;
    }


    {
        size_t sizes[] = {5, 20, 100};
        for (int si = 0; si < 3; si++) {
            size_t n = sizes[si];
            lmmc_sparse_mat_t A = {0};
            lmmc_vec_t b = {0}, x = {0};
            lmmc_itersolve_config_t cfg = {0};
            lmmc_itersolve_result_t result = {0};

            st = build_spd_tridiag(n, &A);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_vec_create(n, &b);
            lmmc_vec_create(n, &x);
            compute_rhs_ones(&A, &b);


            lmmc_vec_fill(&x, 0.0);
            lmmc_itersolve_default_config(n, &cfg);
            cfg.max_iter = 500;
            st = lmmc_cg_solve(&A, &b, NULL, &cfg, &x, &result);
            if (st != LMMC_STATUS_OK || !result.converged) {
                printf("8.4 CG size=%zu failed\n", n);
                lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
                rc = 1; goto done;
            }

            lmmc_vec_destroy(&x);
            lmmc_vec_destroy(&b);
            lmmc_sparse_destroy(&A);
        }


        for (int si = 0; si < 3; si++) {
            size_t n = sizes[si];
            lmmc_sparse_mat_t A = {0};
            lmmc_vec_t b = {0}, x = {0};
            lmmc_itersolve_config_t cfg = {0};
            lmmc_itersolve_result_t result = {0};

            st = build_nonsym(n, &A);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_vec_create(n, &b);
            lmmc_vec_create(n, &x);
            compute_rhs_ones(&A, &b);


            lmmc_vec_fill(&x, 0.0);
            lmmc_itersolve_default_config(n, &cfg);
            cfg.max_iter = 500;
            st = lmmc_bicgstab_solve(&A, &b, NULL, &cfg, &x, &result);
            if (st != LMMC_STATUS_OK || !result.converged) {
                printf("8.4 BiCGSTAB size=%zu failed\n", n);
                lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
                rc = 1; goto done;
            }


            lmmc_vec_fill(&x, 0.0);
            cfg.restart = 30;
            memset(&result, 0, sizeof(result));
            st = lmmc_gmres_solve(&A, &b, NULL, &cfg, &x, &result);
            if (st != LMMC_STATUS_OK || !result.converged) {
                printf("8.4 GMRES size=%zu failed\n", n);
                lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
                rc = 1; goto done;
            }

            lmmc_vec_destroy(&x);
            lmmc_vec_destroy(&b);
            lmmc_sparse_destroy(&A);
        }
    }


    {
        const size_t n = 10;
        lmmc_sparse_mat_t A = {0};
        lmmc_vec_t b = {0}, x = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t result = {0};

        st = build_spd_tridiag(n, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_create(n, &b);
        lmmc_vec_create(n, &x);
        compute_rhs_ones(&A, &b);

        lmmc_vec_fill(&x, 0.0);
        lmmc_itersolve_default_config(n, &cfg);
        cfg.max_iter = 1;


        st = lmmc_cg_solve(&A, &b, NULL, &cfg, &x, &result);
        if (result.converged) {
            printf("8.5 CG max_iter=1 unexpectedly converged\n");
            rc = 1;
        }
        if (result.num_iter > 1) {
            printf("8.5 CG num_iter=%zu expected <=1\n", result.num_iter);
            rc = 1;
        }


        lmmc_vec_fill(&x, 0.0);
        memset(&result, 0, sizeof(result));
        st = lmmc_bicgstab_solve(&A, &b, NULL, &cfg, &x, &result);
        if (result.converged) {
            printf("8.5 BiCGSTAB max_iter=1 unexpectedly converged\n");
            rc = 1;
        }


        lmmc_vec_fill(&x, 0.0);
        memset(&result, 0, sizeof(result));
        cfg.restart = 30;
        st = lmmc_gmres_solve(&A, &b, NULL, &cfg, &x, &result);
        if (result.converged) {
            printf("8.5 GMRES max_iter=1 unexpectedly converged\n");
            rc = 1;
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_sparse_destroy(&A);
        if (rc) goto done;
    }


    {
        const size_t n = 5;
        lmmc_sparse_mat_t A = {0};
        lmmc_vec_t b_wrong = {0}, x = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t result = {0};

        st = build_spd_tridiag(n, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        lmmc_vec_create(n + 2, &b_wrong);
        lmmc_vec_fill(&b_wrong, 1.0);
        lmmc_vec_create(n, &x);
        lmmc_vec_fill(&x, 0.0);
        lmmc_itersolve_default_config(n, &cfg);

        st = lmmc_cg_solve(&A, &b_wrong, NULL, &cfg, &x, &result);
        if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
            printf("8.6 CG dimension mismatch: expected DIMENSION_MISMATCH, got %d\n", st);
            rc = 1;
        }

        memset(&result, 0, sizeof(result));
        st = lmmc_bicgstab_solve(&A, &b_wrong, NULL, &cfg, &x, &result);
        if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
            printf("8.6 BiCGSTAB dimension mismatch: expected DIMENSION_MISMATCH, got %d\n", st);
            rc = 1;
        }

        memset(&result, 0, sizeof(result));
        cfg.restart = 30;
        st = lmmc_gmres_solve(&A, &b_wrong, NULL, &cfg, &x, &result);
        if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
            printf("8.6 GMRES dimension mismatch: expected DIMENSION_MISMATCH, got %d\n", st);
            rc = 1;
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b_wrong);
        lmmc_sparse_destroy(&A);
        if (rc) goto done;
    }


    {
        const size_t n = 10;
        lmmc_sparse_mat_t A = {0};
        lmmc_vec_t b = {0}, x = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t result = {0};

        st = build_illcond(n, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_create(n, &b);
        lmmc_vec_fill(&b, 1.0);
        lmmc_vec_create(n, &x);
        lmmc_vec_fill(&x, 0.0);

        lmmc_itersolve_default_config(n, &cfg);
        cfg.max_iter = 5;
        cfg.abs_tol = 1e-15;

        st = lmmc_cg_solve(&A, &b, NULL, &cfg, &x, &result);

        if (result.converged) {
            printf("8.8 CG ill-conditioned unexpectedly converged in %zu iters\n", result.num_iter);
            rc = 1;
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_sparse_destroy(&A);
        if (rc) goto done;
    }


    {
        const size_t n = 10;
        lmmc_sparse_mat_t A = {0};
        lmmc_vec_t b = {0}, x = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t result = {0};

        st = build_spd_tridiag(n, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_create(n, &b);
        lmmc_vec_create(n, &x);


        lmmc_vec_fill(&x, 1.0);
        compute_rhs_ones(&A, &b);

        lmmc_itersolve_default_config(n, &cfg);
        cfg.max_iter = 200;


        st = lmmc_cg_solve(&A, &b, NULL, &cfg, &x, &result);
        if (st != LMMC_STATUS_OK || !result.converged) {
            printf("8.9 CG exact init failed: st=%d converged=%d\n", st, result.converged);
            rc = 1;
        } else if (result.num_iter > 1) {
            printf("8.9 CG exact init took %zu iters (expected 0 or 1)\n", result.num_iter);
            rc = 1;
        }


        lmmc_vec_fill(&x, 1.0);
        memset(&result, 0, sizeof(result));
        st = lmmc_bicgstab_solve(&A, &b, NULL, &cfg, &x, &result);
        if (st != LMMC_STATUS_OK || !result.converged) {
            printf("8.9 BiCGSTAB exact init failed: st=%d converged=%d\n", st, result.converged);
            rc = 1;
        } else if (result.num_iter > 1) {
            printf("8.9 BiCGSTAB exact init took %zu iters (expected 0 or 1)\n", result.num_iter);
            rc = 1;
        }


        lmmc_vec_fill(&x, 1.0);
        memset(&result, 0, sizeof(result));
        cfg.restart = 30;
        st = lmmc_gmres_solve(&A, &b, NULL, &cfg, &x, &result);
        if (st != LMMC_STATUS_OK || !result.converged) {
            printf("8.9 GMRES exact init failed: st=%d converged=%d\n", st, result.converged);
            rc = 1;
        } else if (result.num_iter > 1) {
            printf("8.9 GMRES exact init took %zu iters (expected 0 or 1)\n", result.num_iter);
            rc = 1;
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_sparse_destroy(&A);
        if (rc) goto done;
    }


    {
        const size_t n = 50;
        lmmc_sparse_mat_t A = {0};
        lmmc_vec_t b = {0}, x_none = {0}, x_precond = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_itersolve_result_t res_none = {0}, res_precond = {0};
        lmmc_precond_t jacobi = {0};

        st = build_spd_tridiag(n, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_create(n, &b);
        lmmc_vec_create(n, &x_none);
        lmmc_vec_create(n, &x_precond);
        compute_rhs_ones(&A, &b);

        lmmc_itersolve_default_config(n, &cfg);
        cfg.max_iter = 500;


        lmmc_vec_fill(&x_none, 0.0);
        st = lmmc_cg_solve(&A, &b, NULL, &cfg, &x_none, &res_none);
        if (st != LMMC_STATUS_OK || !res_none.converged) {
            printf("8.10 CG no-precond failed\n");
            lmmc_vec_destroy(&x_none); lmmc_vec_destroy(&x_precond);
            lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
            rc = 1; goto done;
        }


        st = lmmc_precond_create_jacobi(&A, &jacobi);
        if (st != LMMC_STATUS_OK) {
            printf("8.10 Jacobi create failed\n");
            lmmc_vec_destroy(&x_none); lmmc_vec_destroy(&x_precond);
            lmmc_vec_destroy(&b); lmmc_sparse_destroy(&A);
            rc = 1; goto done;
        }

        lmmc_vec_fill(&x_precond, 0.0);
        st = lmmc_cg_solve(&A, &b, &jacobi, &cfg, &x_precond, &res_precond);
        if (st != LMMC_STATUS_OK || !res_precond.converged) {
            printf("8.10 CG with Jacobi failed\n");
            rc = 1;
        } else {

            if (res_precond.num_iter > res_none.num_iter) {
                printf("8.10 Jacobi did not accelerate: precond=%zu > none=%zu\n",
                       res_precond.num_iter, res_none.num_iter);
                rc = 1;
            }
        }

        lmmc_precond_destroy(&jacobi);
        lmmc_vec_destroy(&x_precond);
        lmmc_vec_destroy(&x_none);
        lmmc_vec_destroy(&b);
        lmmc_sparse_destroy(&A);
        if (rc) goto done;
    }

done:
    if (rc != 0) {
        printf("itersolve_extended test failed\n");
    }
    return rc;
}
