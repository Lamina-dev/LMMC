/**
 * @file test_memory_safety.c
 * 针对 LMMC 中 memory safety 相关接口的单元测试。
 */
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"


int main(void) {
    int rc = 0;


    {
        lmmc_vec_t v = {0};
        lmmc_status_t st = lmmc_vec_create(10, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_destroy(&v);
    }


    {
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_create(5, 5, &m);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_mat_destroy(&m);
    }


    {
        lmmc_tensor_t t = {0};
        lmmc_status_t st = lmmc_tensor3_create(3, 4, 5, &t);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_tensor_destroy(&t);
    }


    {
        lmmc_sparse_mat_t sp = {0};
        lmmc_status_t st = lmmc_sparse_create_csr(5, 5, 10, &sp);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_sparse_destroy(&sp);
    }


    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_destroy(rng);
    }


    {
        lmmc_precond_t pc = {0};
        lmmc_status_t st = lmmc_precond_create_none(10, &pc);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_precond_destroy(&pc);
    }


    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
        lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};
        lmmc_interp_cspline_t* spline = NULL;
        lmmc_status_t st = lmmc_interp_cspline_create(xs, ys, 5, &spline);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_interp_cspline_destroy(spline);
    }


    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
        lmmc_real_t ys[] = {1.0, 2.0, 5.0, 10.0};
        lmmc_interp_lagrange_t* lag = NULL;
        lmmc_status_t st = lmmc_interp_lagrange_create(xs, ys, 4, &lag);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_interp_lagrange_destroy(lag);
    }


    {
        lmmc_vec_t v = {0};
        lmmc_vec_destroy(&v);
        lmmc_vec_destroy(NULL);
    }


    {
        lmmc_mat_t m = {0};
        lmmc_mat_destroy(&m);
        lmmc_mat_destroy(NULL);
    }


    {
        lmmc_tensor_t t = {0};
        lmmc_tensor_destroy(&t);
        lmmc_tensor_destroy(NULL);
    }


    {
        lmmc_sparse_mat_t sp = {0};
        lmmc_sparse_destroy(&sp);
        lmmc_sparse_destroy(NULL);
    }


    {
        lmmc_rng_destroy(NULL);
    }


    {
        lmmc_precond_t pc = {0};
        lmmc_precond_destroy(&pc);
        lmmc_precond_destroy(NULL);
    }


    {
        lmmc_interp_cspline_destroy(NULL);
    }


    {
        lmmc_interp_lagrange_destroy(NULL);
    }


    {
        lmmc_vec_t v = {0};
        lmmc_status_t st = lmmc_vec_create(0, &v);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

    }


    {
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_create(0, 5, &m);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }


    {
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_create(5, 0, &m);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }


    {
        lmmc_tensor_t t = {0};
        lmmc_status_t st = lmmc_tensor3_create(0, 4, 5, &t);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }


    {
        lmmc_real_t xs[] = {0.0, 1.0};
        lmmc_real_t ys[] = {0.0, 1.0};
        lmmc_interp_cspline_t* spline = NULL;
        lmmc_status_t st = lmmc_interp_cspline_create(xs, ys, 2, &spline);

        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }


    {
        lmmc_interp_lagrange_t* lag = NULL;
        lmmc_status_t st = lmmc_interp_lagrange_create(NULL, NULL, 0, &lag);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }


    {
        lmmc_real_t ext_data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        lmmc_vec_t v = {0};
        lmmc_status_t st = lmmc_vec_wrap(5, ext_data, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_destroy(&v);

        if (!lmmc_test_nearly_equal(ext_data[0], 1.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[2], 3.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[4], 5.0, 1e-15)) {
            rc = 1; goto done;
        }
    }


    {
        lmmc_real_t ext_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_wrap(2, 3, 3, ext_data, &m);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_mat_destroy(&m);

        if (!lmmc_test_nearly_equal(ext_data[0], 1.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[3], 4.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[5], 6.0, 1e-15)) {
            rc = 1; goto done;
        }
    }


    {
        lmmc_real_t ext_data[24];
        for (int i = 0; i < 24; i++) ext_data[i] = (lmmc_real_t)(i + 1);
        lmmc_tensor_t t = {0};

        lmmc_status_t st = lmmc_tensor3_wrap(2, 3, 4, 12, 4, 1, ext_data, &t);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_tensor_destroy(&t);

        if (!lmmc_test_nearly_equal(ext_data[0], 1.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[12], 13.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[23], 24.0, 1e-15)) {
            rc = 1; goto done;
        }
    }


    {

        size_t row_ptr[] = {0, 1, 2, 3};
        size_t col_idx[] = {0, 1, 2};
        lmmc_real_t values[] = {1.0, 1.0, 1.0};
        lmmc_sparse_mat_t sp = {0};
        lmmc_status_t st = lmmc_sparse_wrap_csr(3, 3, 3, row_ptr, col_idx, values, &sp);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_sparse_destroy(&sp);

        if (!lmmc_test_nearly_equal(values[0], 1.0, 1e-15) ||
            !lmmc_test_nearly_equal(values[1], 1.0, 1e-15) ||
            !lmmc_test_nearly_equal(values[2], 1.0, 1e-15)) {
            rc = 1; goto done;
        }
        if (row_ptr[0] != 0 || row_ptr[3] != 3) { rc = 1; goto done; }
        if (col_idx[0] != 0 || col_idx[2] != 2) { rc = 1; goto done; }
    }


    {
        for (int i = 0; i < 1000; i++) {
            lmmc_vec_t v = {0};
            lmmc_status_t st = lmmc_vec_create(100, &v);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_vec_destroy(&v);
        }
    }


    {
        for (int i = 0; i < 1000; i++) {
            lmmc_mat_t m = {0};
            lmmc_status_t st = lmmc_mat_create(10, 10, &m);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_mat_destroy(&m);
        }
    }


    {
        for (int i = 0; i < 1000; i++) {
            lmmc_tensor_t t = {0};
            lmmc_status_t st = lmmc_tensor3_create(3, 3, 3, &t);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_tensor_destroy(&t);
        }
    }


    {
        for (int i = 0; i < 1000; i++) {
            lmmc_sparse_mat_t sp = {0};
            lmmc_status_t st = lmmc_sparse_create_csr(5, 5, 5, &sp);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_sparse_destroy(&sp);
        }
    }


    {
        for (int i = 0; i < 1000; i++) {
            lmmc_rng_t* rng = NULL;
            lmmc_status_t st = lmmc_rng_create(&rng);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_rng_destroy(rng);
        }
    }


    {
        for (int i = 0; i < 1000; i++) {
            lmmc_precond_t pc = {0};
            lmmc_status_t st = lmmc_precond_create_none(10, &pc);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_precond_destroy(&pc);
        }
    }


    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
        lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};
        for (int i = 0; i < 1000; i++) {
            lmmc_interp_cspline_t* spline = NULL;
            lmmc_status_t st = lmmc_interp_cspline_create(xs, ys, 5, &spline);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_interp_cspline_destroy(spline);
        }
    }


    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
        lmmc_real_t ys[] = {1.0, 2.0, 5.0, 10.0};
        for (int i = 0; i < 1000; i++) {
            lmmc_interp_lagrange_t* lag = NULL;
            lmmc_status_t st = lmmc_interp_lagrange_create(xs, ys, 4, &lag);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_interp_lagrange_destroy(lag);
        }
    }

done:
    if (rc != 0) {
        printf("memory_safety test failed\n");
    }
    return rc;
}
