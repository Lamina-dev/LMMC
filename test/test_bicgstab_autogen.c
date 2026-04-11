#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "lmmc/lmmc.h"

static double rand_unit(void) {
    return (double)rand() / (double)RAND_MAX;
}

static double rand_sym(void) {
    return 2.0 * rand_unit() - 1.0;
}

static double vec_error_norm2(const lmmc_vec_t* x, const lmmc_vec_t* y) {
    size_t i = 0;
    double s = 0.0;
    for (i = 0; i < x->size; ++i) {
        double d = x->data[i] - y->data[i];
        s += d * d;
    }
    return sqrt(s);
}

static lmmc_status_t generate_diagonal_dominant_system(
    int n,
    double density,
    lmmc_mat_t* a_dense,
    lmmc_sparse_mat_t* a_sparse,
    lmmc_vec_t* x_true,
    lmmc_vec_t* b
) {
    int i = 0;
    int j = 0;
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

    st = lmmc_sparse_from_dense(a_dense, 1e-14, a_sparse);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    for (i = 0; i < n; ++i) {
        x_true->data[i] = rand_sym();
    }

    return lmmc_sparse_mat_vec_mul(a_sparse, x_true, b);
}

static int run_method_checked(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_vec_t* x_true,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    double tol
) {
    lmmc_vec_t x = {0};
    lmmc_itersolve_result_t result = {0};
    lmmc_status_t st = lmmc_vec_create(b->size, &x);
    if (st != LMMC_STATUS_OK) {
        return 0;
    }

    st = lmmc_bicgstab_solve(a, b, precond, cfg, &x, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1) {
        lmmc_vec_destroy(&x);
        return 0;
    }

    if (vec_error_norm2(&x, x_true) > tol) {
        lmmc_vec_destroy(&x);
        return 0;
    }

    lmmc_vec_destroy(&x);
    return 1;
}

int main(void) {
    int case_id = 0;
    const int cases = 200;
    const int n = 20;
    const double density = 0.10;

    srand(20260411u);

    for (case_id = 0; case_id < cases; ++case_id) {
        lmmc_mat_t a_dense = {0};
        lmmc_sparse_mat_t a_sparse = {0};
        lmmc_vec_t x_true = {0};
        lmmc_vec_t b = {0};
        lmmc_precond_t jacobi = {0};
        lmmc_precond_t ilu0 = {0};
        lmmc_precond_t ilut = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_status_t st = LMMC_STATUS_OK;

        st = generate_diagonal_dominant_system(n, density, &a_dense, &a_sparse, &x_true, &b);
        if (st != LMMC_STATUS_OK) {
            printf("autogen test failed: generate system case=%d status=%s\n", case_id, lmmc_status_string(st));
            return 1;
        }

        st = lmmc_itersolve_default_config((size_t)n, &cfg);
        if (st != LMMC_STATUS_OK) {
            printf("autogen test failed: default config case=%d status=%s\n", case_id, lmmc_status_string(st));
            return 1;
        }

        cfg.max_iter = (size_t)(n * 25);
        cfg.abs_tol = 1e-12;
        cfg.rel_tol = 1e-10;

        st = lmmc_precond_create_jacobi(&a_sparse, &jacobi);
        if (st != LMMC_STATUS_OK) {
            printf("autogen test failed: jacobi create case=%d status=%s\n", case_id, lmmc_status_string(st));
            return 1;
        }

        st = lmmc_precond_create_ilu0(&a_sparse, &ilu0);
        if (st != LMMC_STATUS_OK) {
            printf("autogen test failed: ilu0 create case=%d status=%s\n", case_id, lmmc_status_string(st));
            return 1;
        }

        st = lmmc_precond_create_ilut(&a_sparse, 1e-10, 8, &ilut);
        if (st != LMMC_STATUS_OK) {
            printf("autogen test failed: ilut create case=%d status=%s\n", case_id, lmmc_status_string(st));
            return 1;
        }

        if (!run_method_checked(&a_sparse, &b, &x_true, NULL, &cfg, 1e-5) ||
            !run_method_checked(&a_sparse, &b, &x_true, &jacobi, &cfg, 1e-5) ||
            !run_method_checked(&a_sparse, &b, &x_true, &ilu0, &cfg, 1e-5) ||
            !run_method_checked(&a_sparse, &b, &x_true, &ilut, &cfg, 1e-5)) {
            printf("autogen test failed: solve case=%d\n", case_id);
            return 1;
        }

        lmmc_precond_destroy(&ilut);
        lmmc_precond_destroy(&ilu0);
        lmmc_precond_destroy(&jacobi);
        lmmc_vec_destroy(&b);
        lmmc_vec_destroy(&x_true);
        lmmc_sparse_destroy(&a_sparse);
        lmmc_mat_destroy(&a_dense);
    }

    return 0;
}
