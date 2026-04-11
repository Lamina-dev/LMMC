#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmmc/lmmc.h"

typedef struct {
    int cases;
    int size;
    unsigned int seed;
    double density;
    double ilut_drop_tol;
    size_t ilut_fill;
} benchmark_options_t;

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
                v = 0.25 * rand_sym();
                if (fabs(v) < 0.03) {
                    v = (v >= 0.0) ? 0.03 : -0.03;
                }
                a_dense->data[(size_t)i * a_dense->stride + (size_t)j] = v;
                row_sum_abs += fabs(v);
            }
        }

        a_dense->data[(size_t)i * a_dense->stride + (size_t)i] = row_sum_abs + 1.5 + 0.2 * rand_unit();
    }

    st = lmmc_sparse_from_dense(a_dense, 1e-14, a_sparse);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    for (i = 0; i < n; ++i) {
        x_true->data[i] = rand_sym();
    }

    st = lmmc_sparse_mat_vec_mul(a_sparse, x_true, b);
    return st;
}

static void run_one_method(
    int case_id,
    const char* method,
    const lmmc_sparse_mat_t* a_sparse,
    const lmmc_vec_t* b,
    const lmmc_vec_t* x_true,
    const lmmc_precond_t* precond,
    lmmc_status_t precond_status,
    const lmmc_itersolve_config_t* cfg
) {
    lmmc_vec_t x = {0};
    lmmc_itersolve_result_t result = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    double err = NAN;

    if (precond_status != LMMC_STATUS_OK) {
        printf("%d,%s,%d,0,0,nan,nan\n", case_id, method, (int)precond_status);
        return;
    }

    st = lmmc_vec_create(b->size, &x);
    if (st != LMMC_STATUS_OK) {
        printf("%d,%s,%d,0,0,nan,nan\n", case_id, method, (int)st);
        return;
    }

    st = lmmc_bicgstab_solve(a_sparse, b, precond, cfg, &x, &result);
    if (st == LMMC_STATUS_OK) {
        err = vec_error_norm2(&x, x_true);
    }

    printf(
        "%d,%s,%d,%d,%zu,%.16e,%.16e\n",
        case_id,
        method,
        (int)st,
        result.converged,
        result.num_iter,
        result.final_residual_norm,
        err
    );

    lmmc_vec_destroy(&x);
}

static void parse_args(int argc, char** argv, benchmark_options_t* opts) {
    int i = 0;
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cases") == 0 && i + 1 < argc) {
            opts->cases = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            opts->size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            opts->seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--density") == 0 && i + 1 < argc) {
            opts->density = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--ilut-drop") == 0 && i + 1 < argc) {
            opts->ilut_drop_tol = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--ilut-fill") == 0 && i + 1 < argc) {
            opts->ilut_fill = (size_t)strtoul(argv[++i], NULL, 10);
        }
    }
}

int main(int argc, char** argv) {
    benchmark_options_t opts = {1000, 24, 20260411u, 0.08, 1e-10, 8};
    int case_id = 0;

    parse_args(argc, argv, &opts);

    if (opts.cases <= 0 || opts.size <= 2 || opts.density <= 0.0 || opts.density >= 1.0 ||
        opts.ilut_drop_tol < 0.0 || opts.ilut_fill == 0) {
        fprintf(stderr, "invalid arguments\n");
        return 1;
    }

    srand(opts.seed);

    printf("case_id,method,status,converged,iterations,final_residual,error_norm\n");

    for (case_id = 0; case_id < opts.cases; ++case_id) {
        lmmc_mat_t a_dense = {0};
        lmmc_sparse_mat_t a_sparse = {0};
        lmmc_vec_t x_true = {0};
        lmmc_vec_t b = {0};
        lmmc_precond_t jacobi = {0};
        lmmc_precond_t ilu0 = {0};
        lmmc_precond_t ilut = {0};
        lmmc_itersolve_config_t cfg = {0};
        lmmc_status_t st = LMMC_STATUS_OK;
        lmmc_status_t st_jacobi = LMMC_STATUS_OK;
        lmmc_status_t st_ilu0 = LMMC_STATUS_OK;
        lmmc_status_t st_ilut = LMMC_STATUS_OK;

        st = generate_diagonal_dominant_system(opts.size, opts.density, &a_dense, &a_sparse, &x_true, &b);
        if (st != LMMC_STATUS_OK) {
            fprintf(stderr, "system generation failed at case %d: %s\n", case_id, lmmc_status_string(st));
            lmmc_vec_destroy(&b);
            lmmc_vec_destroy(&x_true);
            lmmc_sparse_destroy(&a_sparse);
            lmmc_mat_destroy(&a_dense);
            return 2;
        }

        st = lmmc_itersolve_default_config((size_t)opts.size, &cfg);
        if (st != LMMC_STATUS_OK) {
            fprintf(stderr, "default config failed at case %d: %s\n", case_id, lmmc_status_string(st));
            lmmc_vec_destroy(&b);
            lmmc_vec_destroy(&x_true);
            lmmc_sparse_destroy(&a_sparse);
            lmmc_mat_destroy(&a_dense);
            return 3;
        }

        cfg.max_iter = (size_t)(opts.size * 20);
        if (cfg.max_iter < 200) {
            cfg.max_iter = 200;
        }
        cfg.abs_tol = 1e-12;
        cfg.rel_tol = 1e-10;

        st_jacobi = lmmc_precond_create_jacobi(&a_sparse, &jacobi);
        st_ilu0 = lmmc_precond_create_ilu0(&a_sparse, &ilu0);
        st_ilut = lmmc_precond_create_ilut(&a_sparse, opts.ilut_drop_tol, opts.ilut_fill, &ilut);

        run_one_method(case_id, "None", &a_sparse, &b, &x_true, NULL, LMMC_STATUS_OK, &cfg);
        run_one_method(case_id, "Jacobi", &a_sparse, &b, &x_true, &jacobi, st_jacobi, &cfg);
        run_one_method(case_id, "ILU0", &a_sparse, &b, &x_true, &ilu0, st_ilu0, &cfg);
        run_one_method(case_id, "ILUT", &a_sparse, &b, &x_true, &ilut, st_ilut, &cfg);

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
