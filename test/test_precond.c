/**
 * @file test_precond.c
 * 针对 LMMC 中 precond 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"


static lmmc_status_t build_spd_tridiag(size_t n, lmmc_sparse_mat_t* out) {
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st;
    size_t i;

    st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (i = 0; i < n; ++i) {
        st = lmmc_sparse_builder_add(builder, i, i, 4.0);
        if (st != LMMC_STATUS_OK) goto fail;
        if (i > 0) {
            st = lmmc_sparse_builder_add(builder, i, i - 1, -1.0);
            if (st != LMMC_STATUS_OK) goto fail;
        }
        if (i + 1 < n) {
            st = lmmc_sparse_builder_add(builder, i, i + 1, -1.0);
            if (st != LMMC_STATUS_OK) goto fail;
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, out);
fail:
    lmmc_sparse_builder_destroy(builder);
    return st;
}


static lmmc_status_t build_general_dd(size_t n, lmmc_sparse_mat_t* out) {
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st;
    size_t i;

    st = lmmc_sparse_builder_create(n, n, 3 * n, &builder);
    if (st != LMMC_STATUS_OK) return st;

    for (i = 0; i < n; ++i) {
        st = lmmc_sparse_builder_add(builder, i, i, 5.0);
        if (st != LMMC_STATUS_OK) goto fail;
        if (i > 0) {
            st = lmmc_sparse_builder_add(builder, i, i - 1, -1.0);
            if (st != LMMC_STATUS_OK) goto fail;
        }
        if (i + 1 < n) {
            st = lmmc_sparse_builder_add(builder, i, i + 1, -2.0);
            if (st != LMMC_STATUS_OK) goto fail;
        }
    }

    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, out);
fail:
    lmmc_sparse_builder_destroy(builder);
    return st;
}

int main(void) {
    int rc = 0;
    lmmc_status_t st;


    {
        lmmc_sparse_mat_t A = {0};
        lmmc_precond_t jacobi = {0};
        lmmc_vec_t v = {0}, out = {0};
        size_t i;

        st = build_spd_tridiag(5, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_precond_create_jacobi(&A, &jacobi);
        if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&A); rc = 1; goto done; }

        st = lmmc_vec_create(5, &v);
        if (st != LMMC_STATUS_OK) { lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done; }
        st = lmmc_vec_create(5, &out);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done; }

        for (i = 0; i < 5; ++i) {
            v.data[i] = (lmmc_real_t)(i + 1);
        }

        st = lmmc_precond_apply(&jacobi, &v, &out);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&out); lmmc_vec_destroy(&v); lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done; }


        for (i = 0; i < 5; ++i) {
            lmmc_real_t expected = (lmmc_real_t)(i + 1) / 4.0;
            if (!lmmc_test_nearly_equal(out.data[i], expected, 1e-12)) {
                rc = 1; lmmc_vec_destroy(&out); lmmc_vec_destroy(&v); lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); goto done;
            }
        }

        lmmc_vec_destroy(&out);
        lmmc_vec_destroy(&v);
        lmmc_precond_destroy(&jacobi);
        lmmc_sparse_destroy(&A);
    }


    {
        lmmc_sparse_mat_t A = {0};
        lmmc_precond_t ilu0 = {0};
        lmmc_vec_t v = {0}, out_ilu = {0}, Ax = {0};
        size_t i;
        lmmc_real_t residual_norm = 0.0;

        st = build_spd_tridiag(5, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_precond_create_ilu0(&A, &ilu0);
        if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&A); rc = 1; goto done; }

        st = lmmc_vec_create(5, &v);
        if (st != LMMC_STATUS_OK) { lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); rc = 1; goto done; }
        st = lmmc_vec_create(5, &out_ilu);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); rc = 1; goto done; }
        st = lmmc_vec_create(5, &Ax);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&out_ilu); lmmc_vec_destroy(&v); lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); rc = 1; goto done; }

        for (i = 0; i < 5; ++i) {
            v.data[i] = (lmmc_real_t)(i + 1);
        }


        st = lmmc_precond_apply(&ilu0, &v, &out_ilu);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&Ax); lmmc_vec_destroy(&out_ilu); lmmc_vec_destroy(&v); lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); rc = 1; goto done; }


        st = lmmc_sparse_mat_vec_mul(&A, &out_ilu, &Ax);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&Ax); lmmc_vec_destroy(&out_ilu); lmmc_vec_destroy(&v); lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); rc = 1; goto done; }

        residual_norm = 0.0;
        for (i = 0; i < 5; ++i) {
            lmmc_real_t diff = Ax.data[i] - v.data[i];
            residual_norm += diff * diff;
        }
        residual_norm = sqrt(residual_norm);

        if (residual_norm > 1e-10) {
            rc = 1; lmmc_vec_destroy(&Ax); lmmc_vec_destroy(&out_ilu); lmmc_vec_destroy(&v); lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); goto done;
        }

        lmmc_vec_destroy(&Ax);
        lmmc_vec_destroy(&out_ilu);
        lmmc_vec_destroy(&v);
        lmmc_precond_destroy(&ilu0);
        lmmc_sparse_destroy(&A);
    }


    {
        lmmc_precond_t none = {0};
        lmmc_vec_t v = {0}, out = {0};
        size_t i;

        st = lmmc_precond_create_none(5, &none);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_vec_create(5, &v);
        if (st != LMMC_STATUS_OK) { lmmc_precond_destroy(&none); rc = 1; goto done; }
        st = lmmc_vec_create(5, &out);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); lmmc_precond_destroy(&none); rc = 1; goto done; }

        for (i = 0; i < 5; ++i) {
            v.data[i] = (lmmc_real_t)(i * 3.14 + 1.0);
        }

        st = lmmc_precond_apply(&none, &v, &out);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&out); lmmc_vec_destroy(&v); lmmc_precond_destroy(&none); rc = 1; goto done; }

        for (i = 0; i < 5; ++i) {
            if (!lmmc_test_nearly_equal(out.data[i], v.data[i], 1e-15)) {
                rc = 1; lmmc_vec_destroy(&out); lmmc_vec_destroy(&v); lmmc_precond_destroy(&none); goto done;
            }
        }

        lmmc_vec_destroy(&out);
        lmmc_vec_destroy(&v);
        lmmc_precond_destroy(&none);
    }


    {
        lmmc_sparse_mat_t A = {0};
        lmmc_precond_t ilut = {0};
        lmmc_vec_t v = {0}, out = {0}, Ax = {0};
        size_t i;
        lmmc_real_t residual_norm = 0.0;

        st = build_spd_tridiag(10, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_precond_create_ilut(&A, 1e-3, 10, &ilut);
        if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&A); rc = 1; goto done; }

        st = lmmc_vec_create(10, &v);
        if (st != LMMC_STATUS_OK) { lmmc_precond_destroy(&ilut); lmmc_sparse_destroy(&A); rc = 1; goto done; }
        st = lmmc_vec_create(10, &out);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); lmmc_precond_destroy(&ilut); lmmc_sparse_destroy(&A); rc = 1; goto done; }
        st = lmmc_vec_create(10, &Ax);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&out); lmmc_vec_destroy(&v); lmmc_precond_destroy(&ilut); lmmc_sparse_destroy(&A); rc = 1; goto done; }

        for (i = 0; i < 10; ++i) {
            v.data[i] = (lmmc_real_t)(i + 1);
        }


        st = lmmc_precond_apply(&ilut, &v, &out);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&Ax); lmmc_vec_destroy(&out); lmmc_vec_destroy(&v); lmmc_precond_destroy(&ilut); lmmc_sparse_destroy(&A); rc = 1; goto done; }


        st = lmmc_sparse_mat_vec_mul(&A, &out, &Ax);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&Ax); lmmc_vec_destroy(&out); lmmc_vec_destroy(&v); lmmc_precond_destroy(&ilut); lmmc_sparse_destroy(&A); rc = 1; goto done; }

        residual_norm = 0.0;
        for (i = 0; i < 10; ++i) {
            lmmc_real_t diff = Ax.data[i] - v.data[i];
            residual_norm += diff * diff;
        }
        residual_norm = sqrt(residual_norm);


        if (residual_norm > 1e-6) {
            rc = 1; lmmc_vec_destroy(&Ax); lmmc_vec_destroy(&out); lmmc_vec_destroy(&v); lmmc_precond_destroy(&ilut); lmmc_sparse_destroy(&A); goto done;
        }

        lmmc_vec_destroy(&Ax);
        lmmc_vec_destroy(&out);
        lmmc_vec_destroy(&v);
        lmmc_precond_destroy(&ilut);
        lmmc_sparse_destroy(&A);
    }


    {
        lmmc_sparse_mat_t A = {0};
        lmmc_precond_t jacobi = {0};
        lmmc_vec_t b = {0}, x_precond = {0}, x_none = {0};
        lmmc_itersolve_config_t cfg_iter = {0};
        lmmc_itersolve_result_t res_precond = {0}, res_none = {0};
        size_t i;
        size_t n = 50;

        st = build_spd_tridiag(n, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_precond_create_jacobi(&A, &jacobi);
        if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&A); rc = 1; goto done; }

        st = lmmc_vec_create(n, &b);
        if (st != LMMC_STATUS_OK) { lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done; }
        st = lmmc_vec_create(n, &x_precond);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&b); lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done; }
        st = lmmc_vec_create(n, &x_none);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&x_precond); lmmc_vec_destroy(&b); lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done; }

        for (i = 0; i < n; ++i) {
            b.data[i] = 1.0;
        }

        st = lmmc_itersolve_default_config(n, &cfg_iter);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&x_none); lmmc_vec_destroy(&x_precond); lmmc_vec_destroy(&b); lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done; }
        cfg_iter.max_iter = 200;


        st = lmmc_cg_solve(&A, &b, &jacobi, &cfg_iter, &x_precond, &res_precond);
        if (st != LMMC_STATUS_OK || res_precond.converged != 1) {
            lmmc_vec_destroy(&x_none); lmmc_vec_destroy(&x_precond); lmmc_vec_destroy(&b); lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done;
        }


        st = lmmc_cg_solve(&A, &b, NULL, &cfg_iter, &x_none, &res_none);
        if (st != LMMC_STATUS_OK || res_none.converged != 1) {
            lmmc_vec_destroy(&x_none); lmmc_vec_destroy(&x_precond); lmmc_vec_destroy(&b); lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done;
        }


        if (res_precond.num_iter > res_none.num_iter) {
            rc = 1; lmmc_vec_destroy(&x_none); lmmc_vec_destroy(&x_precond); lmmc_vec_destroy(&b); lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); goto done;
        }

        lmmc_vec_destroy(&x_none);
        lmmc_vec_destroy(&x_precond);
        lmmc_vec_destroy(&b);
        lmmc_precond_destroy(&jacobi);
        lmmc_sparse_destroy(&A);
    }


    {
        lmmc_sparse_mat_t A = {0};
        lmmc_precond_t ilu0 = {0};
        lmmc_vec_t b = {0}, x = {0}, Ax = {0};
        lmmc_itersolve_config_t cfg_iter = {0};
        lmmc_itersolve_result_t res = {0};
        size_t i;
        size_t n = 20;
        lmmc_real_t residual_norm = 0.0;

        st = build_general_dd(n, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_precond_create_ilu0(&A, &ilu0);
        if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&A); rc = 1; goto done; }

        st = lmmc_vec_create(n, &b);
        if (st != LMMC_STATUS_OK) { lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); rc = 1; goto done; }
        st = lmmc_vec_create(n, &x);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&b); lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); rc = 1; goto done; }
        st = lmmc_vec_create(n, &Ax);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); rc = 1; goto done; }

        for (i = 0; i < n; ++i) {
            b.data[i] = (lmmc_real_t)(i + 1);
        }

        st = lmmc_itersolve_default_config(n, &cfg_iter);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&Ax); lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); rc = 1; goto done; }
        cfg_iter.max_iter = 200;

        st = lmmc_bicgstab_solve(&A, &b, &ilu0, &cfg_iter, &x, &res);
        if (st != LMMC_STATUS_OK || res.converged != 1) {
            lmmc_vec_destroy(&Ax); lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); rc = 1; goto done;
        }


        st = lmmc_sparse_mat_vec_mul(&A, &x, &Ax);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&Ax); lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); rc = 1; goto done; }

        residual_norm = 0.0;
        for (i = 0; i < n; ++i) {
            lmmc_real_t diff = Ax.data[i] - b.data[i];
            residual_norm += diff * diff;
        }
        residual_norm = sqrt(residual_norm);

        if (residual_norm > 1e-8) {
            rc = 1; lmmc_vec_destroy(&Ax); lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_precond_destroy(&ilu0); lmmc_sparse_destroy(&A); goto done;
        }

        lmmc_vec_destroy(&Ax);
        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_precond_destroy(&ilu0);
        lmmc_sparse_destroy(&A);
    }


    {
        lmmc_precond_t p = {0};

        st = lmmc_precond_create_jacobi(NULL, &p);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_precond_create_ilu0(NULL, &p);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_precond_create_ilut(NULL, 1e-3, 10, &p);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }


    {
        lmmc_sparse_builder_t* builder = NULL;
        lmmc_sparse_mat_t A = {0};
        lmmc_precond_t p = {0};


        st = lmmc_sparse_builder_create(3, 3, 9, &builder);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        lmmc_sparse_builder_add(builder, 0, 0, 2.0);
        lmmc_sparse_builder_add(builder, 0, 1, 1.0);
        lmmc_sparse_builder_add(builder, 1, 0, 1.0);
        lmmc_sparse_builder_add(builder, 1, 1, 0.0);
        lmmc_sparse_builder_add(builder, 1, 2, 1.0);
        lmmc_sparse_builder_add(builder, 2, 1, 1.0);
        lmmc_sparse_builder_add(builder, 2, 2, 3.0);

        st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSR, &A);
        lmmc_sparse_builder_destroy(builder);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


        st = lmmc_precond_create_jacobi(&A, &p);
        if (st != LMMC_STATUS_SINGULAR_MATRIX) {
            lmmc_precond_destroy(&p);
            lmmc_sparse_destroy(&A);
            rc = 1; goto done;
        }

        lmmc_sparse_destroy(&A);
    }


    {
        size_t sizes[] = {1, 5, 50, 200};
        size_t num_sizes = 4;
        size_t s;

        for (s = 0; s < num_sizes; ++s) {
            size_t n = sizes[s];
            lmmc_sparse_mat_t A = {0};
            lmmc_precond_t jacobi = {0};
            lmmc_precond_t ilu0_p = {0};
            lmmc_precond_t none_p = {0};

            st = build_spd_tridiag(n, &A);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


            st = lmmc_precond_create_jacobi(&A, &jacobi);
            if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&A); rc = 1; goto done; }
            lmmc_precond_destroy(&jacobi);


            st = lmmc_precond_create_ilu0(&A, &ilu0_p);
            if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&A); rc = 1; goto done; }
            lmmc_precond_destroy(&ilu0_p);


            st = lmmc_precond_create_none(n, &none_p);
            if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&A); rc = 1; goto done; }
            lmmc_precond_destroy(&none_p);

            lmmc_sparse_destroy(&A);
        }
    }


    {
        lmmc_sparse_mat_t A = {0};
        lmmc_precond_t jacobi = {0};
        lmmc_precond_t ilu0_p = {0};
        lmmc_precond_t ilut_p = {0};
        lmmc_precond_t none_p = {0};

        st = build_spd_tridiag(10, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_precond_create_jacobi(&A, &jacobi);
        if (st != LMMC_STATUS_OK) { lmmc_sparse_destroy(&A); rc = 1; goto done; }

        st = lmmc_precond_create_ilu0(&A, &ilu0_p);
        if (st != LMMC_STATUS_OK) { lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done; }

        st = lmmc_precond_create_ilut(&A, 1e-3, 10, &ilut_p);
        if (st != LMMC_STATUS_OK) { lmmc_precond_destroy(&ilu0_p); lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done; }

        st = lmmc_precond_create_none(10, &none_p);
        if (st != LMMC_STATUS_OK) { lmmc_precond_destroy(&ilut_p); lmmc_precond_destroy(&ilu0_p); lmmc_precond_destroy(&jacobi); lmmc_sparse_destroy(&A); rc = 1; goto done; }


        lmmc_precond_destroy(&none_p);
        lmmc_precond_destroy(&ilut_p);
        lmmc_precond_destroy(&ilu0_p);
        lmmc_precond_destroy(&jacobi);


        lmmc_precond_destroy(NULL);

        lmmc_sparse_destroy(&A);
    }

done:
    if (rc != 0) {
        printf("precond test failed\n");
    }
    return rc;
}
