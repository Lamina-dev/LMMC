/**
 * @file test_sparse_extended.c
 * 针对 LMMC 中 sparse extended 相关接口的单元测试。
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define TEST_EPS_TIGHT   1e-12
#define TEST_EPS_NORMAL  1e-10


static lmmc_status_t helper_build_sparse(const double* data, size_t rows, size_t cols,
                                          lmmc_sparse_mat_t* out) {
    lmmc_mat_t dense = {0};
    lmmc_status_t st = lmmc_mat_create(rows, cols, &dense);
    if (st != LMMC_STATUS_OK) return st;
    for (size_t i = 0; i < rows * cols; i++) {
        LMMC_REAL_SET_D(&dense.data[i], data[i]);
    }
    st = lmmc_sparse_from_dense(&dense, 1e-14, out);
    lmmc_mat_destroy(&dense);
    return st;
}


static lmmc_status_t helper_build_sparse_csc(const double* data, size_t rows, size_t cols,
                                              lmmc_sparse_mat_t* out) {
    lmmc_sparse_builder_t* builder = NULL;
    lmmc_status_t st = lmmc_sparse_builder_create(rows, cols, rows * cols, &builder);
    if (st != LMMC_STATUS_OK) return st;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (fabs(data[i * cols + j]) > 1e-14) {
                st = lmmc_sparse_builder_add(builder, i, j, data[i * cols + j]);
                if (st != LMMC_STATUS_OK) { lmmc_sparse_builder_destroy(builder); return st; }
            }
        }
    }
    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

int main(void) {
    int rc = 0;
    lmmc_status_t st;

    printf("Starting sparse extended tests...\n");


    printf("Test 1: CSR->CSC->CSR roundtrip\n");
    {
        double data[] = {
            4.0, 0.0, 2.0,
            0.0, 5.0, 1.0,
            3.0, 0.0, 7.0
        };
        lmmc_sparse_mat_t csr = {0}, csc = {0}, csr_back = {0};

        st = helper_build_sparse(data, 3, 3, &csr);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_sparse_to_csc(&csr, &csc);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&csr); goto done; }

        st = lmmc_sparse_to_csr(&csc, &csr_back);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&csc); lmmc_sparse_destroy(&csr); goto done; }


        if (csr_back.nnz != csr.nnz || csr_back.rows != csr.rows || csr_back.cols != csr.cols) {
            rc = 1; lmmc_sparse_destroy(&csr_back); lmmc_sparse_destroy(&csc); lmmc_sparse_destroy(&csr); goto done;
        }
        for (size_t i = 0; i <= csr.rows; i++) {
            if (csr_back.row_ptr[i] != csr.row_ptr[i]) {
                rc = 1; lmmc_sparse_destroy(&csr_back); lmmc_sparse_destroy(&csc); lmmc_sparse_destroy(&csr); goto done;
            }
        }
        for (size_t i = 0; i < csr.nnz; i++) {
            if (csr_back.col_idx[i] != csr.col_idx[i] ||
                !lmmc_test_nearly_equal(csr_back.values[i], csr.values[i], TEST_EPS_TIGHT)) {
                rc = 1; lmmc_sparse_destroy(&csr_back); lmmc_sparse_destroy(&csc); lmmc_sparse_destroy(&csr); goto done;
            }
        }
        lmmc_sparse_destroy(&csr_back);
        lmmc_sparse_destroy(&csc);
        lmmc_sparse_destroy(&csr);
    }


    printf("Test 2: Dense->Sparse->Dense roundtrip\n");
    {
        double data[] = {
            1.0, 0.0, 3.0,
            0.0, 2.0, 0.0,
            4.0, 0.0, 5.0
        };
        lmmc_mat_t dense_orig = {0}, dense_back = {0};
        lmmc_sparse_mat_t sparse = {0};

        st = lmmc_mat_create(3, 3, &dense_orig);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        for (size_t i = 0; i < 9; i++) LMMC_REAL_SET_D(&dense_orig.data[i], data[i]);

        st = lmmc_sparse_from_dense(&dense_orig, 1e-14, &sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&dense_orig); goto done; }

        st = lmmc_mat_create(3, 3, &dense_back);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense_orig); goto done; }

        st = lmmc_sparse_to_dense(&sparse, &dense_back);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&dense_back); lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense_orig); goto done; }

        for (size_t i = 0; i < 9; i++) {
            if (!lmmc_test_nearly_equal(dense_back.data[i], data[i], TEST_EPS_TIGHT)) {
                rc = 1; lmmc_mat_destroy(&dense_back); lmmc_sparse_destroy(&sparse); lmmc_mat_destroy(&dense_orig); goto done;
            }
        }
        lmmc_mat_destroy(&dense_back);
        lmmc_sparse_destroy(&sparse);
        lmmc_mat_destroy(&dense_orig);
    }


    printf("Test 3: SpMV vs dense\n");
    {
        double a_data[] = {
            2.0, 0.0, 1.0,
            0.0, 3.0, 0.0,
            4.0, 0.0, 5.0
        };
        double x_data[] = {1.0, 2.0, 3.0};
        lmmc_sparse_mat_t sparse = {0};
        lmmc_mat_t dense_a = {0};
        lmmc_vec_t x = {0}, y_sparse = {0}, y_dense = {0};

        st = helper_build_sparse(a_data, 3, 3, &sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_mat_create(3, 3, &dense_a);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&sparse); goto done; }
        for (size_t i = 0; i < 9; i++) LMMC_REAL_SET_D(&dense_a.data[i], a_data[i]);

        st = lmmc_vec_create(3, &x);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&dense_a); lmmc_sparse_destroy(&sparse); goto done; }
        for (size_t i = 0; i < 3; i++) LMMC_REAL_SET_D(&x.data[i], x_data[i]);

        st = lmmc_vec_create(3, &y_sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&x); lmmc_mat_destroy(&dense_a); lmmc_sparse_destroy(&sparse); goto done; }
        st = lmmc_vec_create(3, &y_dense);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&y_sparse); lmmc_vec_destroy(&x); lmmc_mat_destroy(&dense_a); lmmc_sparse_destroy(&sparse); goto done; }

        st = lmmc_sparse_mat_vec_mul(&sparse, &x, &y_sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test3_cleanup; }

        st = lmmc_mat_vec_mul(&dense_a, &x, &y_dense);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test3_cleanup; }

        for (size_t i = 0; i < 3; i++) {
            if (!lmmc_test_nearly_equal(y_sparse.data[i], y_dense.data[i], TEST_EPS_TIGHT)) {
                rc = 1; goto test3_cleanup;
            }
        }
    test3_cleanup:
        lmmc_vec_destroy(&y_dense);
        lmmc_vec_destroy(&y_sparse);
        lmmc_vec_destroy(&x);
        lmmc_mat_destroy(&dense_a);
        lmmc_sparse_destroy(&sparse);
        if (rc != 0) goto done;
    }


    printf("Test 4: SpGEMM vs dense\n");
    {
        double a_data[] = {
            1.0, 0.0, 2.0,
            0.0, 3.0, 0.0,
            4.0, 0.0, 5.0
        };
        double b_data[] = {
            0.0, 1.0, 0.0,
            2.0, 0.0, 3.0,
            0.0, 4.0, 0.0
        };
        lmmc_sparse_mat_t sa = {0}, sb = {0}, sc = {0};
        lmmc_mat_t da = {0}, db = {0}, dc_dense = {0}, dc_from_sparse = {0};

        st = helper_build_sparse(a_data, 3, 3, &sa);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = helper_build_sparse(b_data, 3, 3, &sb);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&sa); goto done; }

        st = lmmc_sparse_mat_mat_mul_sparse(&sa, &sb, &sc);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&sb); lmmc_sparse_destroy(&sa); goto done; }


        st = lmmc_mat_create(3, 3, &da);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test4_cleanup; }
        st = lmmc_mat_create(3, 3, &db);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test4_cleanup; }
        st = lmmc_mat_create(3, 3, &dc_dense);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test4_cleanup; }
        st = lmmc_mat_create(3, 3, &dc_from_sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test4_cleanup; }

        for (size_t i = 0; i < 9; i++) LMMC_REAL_SET_D(&da.data[i], a_data[i]);
        for (size_t i = 0; i < 9; i++) LMMC_REAL_SET_D(&db.data[i], b_data[i]);

        st = lmmc_mat_mul(&da, &db, &dc_dense);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test4_cleanup; }

        st = lmmc_sparse_to_dense(&sc, &dc_from_sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test4_cleanup; }

        for (size_t i = 0; i < 9; i++) {
            if (!lmmc_test_nearly_equal(dc_from_sparse.data[i], dc_dense.data[i], TEST_EPS_TIGHT)) {
                rc = 1; goto test4_cleanup;
            }
        }
    test4_cleanup:
        lmmc_mat_destroy(&dc_from_sparse);
        lmmc_mat_destroy(&dc_dense);
        lmmc_mat_destroy(&db);
        lmmc_mat_destroy(&da);
        lmmc_sparse_destroy(&sc);
        lmmc_sparse_destroy(&sb);
        lmmc_sparse_destroy(&sa);
        if (rc != 0) goto done;
    }


    printf("Test 5: Transpose roundtrip\n");
    {
        double data[] = {
            1.0, 0.0, 2.0,
            0.0, 3.0, 0.0,
            4.0, 0.0, 5.0
        };
        lmmc_sparse_mat_t orig = {0}, trans = {0}, trans_trans = {0};
        lmmc_mat_t d_orig = {0}, d_tt = {0};

        st = helper_build_sparse(data, 3, 3, &orig);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_sparse_transpose(&orig, &trans);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&orig); goto done; }

        st = lmmc_sparse_transpose(&trans, &trans_trans);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&trans); lmmc_sparse_destroy(&orig); goto done; }


        st = lmmc_mat_create(3, 3, &d_orig);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test5_cleanup; }
        st = lmmc_mat_create(3, 3, &d_tt);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test5_cleanup; }

        st = lmmc_sparse_to_dense(&orig, &d_orig);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test5_cleanup; }
        st = lmmc_sparse_to_dense(&trans_trans, &d_tt);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test5_cleanup; }

        for (size_t i = 0; i < 9; i++) {
            if (!lmmc_test_nearly_equal(d_orig.data[i], d_tt.data[i], TEST_EPS_TIGHT)) {
                rc = 1; goto test5_cleanup;
            }
        }
    test5_cleanup:
        lmmc_mat_destroy(&d_tt);
        lmmc_mat_destroy(&d_orig);
        lmmc_sparse_destroy(&trans_trans);
        lmmc_sparse_destroy(&trans);
        lmmc_sparse_destroy(&orig);
        if (rc != 0) goto done;
    }


    printf("Test 6: Sparse addition\n");
    {
        double a_data[] = {
            1.0, 0.0, 2.0,
            0.0, 3.0, 0.0,
            4.0, 0.0, 5.0
        };
        double b_data[] = {
            0.0, 6.0, 0.0,
            7.0, 0.0, 8.0,
            0.0, 9.0, 0.0
        };
        lmmc_sparse_mat_t sa = {0}, sb = {0}, sc = {0};
        lmmc_mat_t da = {0}, db = {0}, dc_sparse = {0};

        st = helper_build_sparse(a_data, 3, 3, &sa);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = helper_build_sparse(b_data, 3, 3, &sb);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&sa); goto done; }


        st = lmmc_sparse_add(1.0, &sa, 1.0, &sb, &sc);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&sb); lmmc_sparse_destroy(&sa); goto done; }

        st = lmmc_mat_create(3, 3, &dc_sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test6_cleanup; }
        st = lmmc_sparse_to_dense(&sc, &dc_sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test6_cleanup; }


        double expected[] = {1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0};
        for (size_t i = 0; i < 9; i++) {
            if (!lmmc_test_nearly_equal(dc_sparse.data[i], expected[i], TEST_EPS_TIGHT)) {
                rc = 1; goto test6_cleanup;
            }
        }
    test6_cleanup:
        lmmc_mat_destroy(&dc_sparse);
        lmmc_sparse_destroy(&sc);
        lmmc_sparse_destroy(&sb);
        lmmc_sparse_destroy(&sa);
        if (rc != 0) goto done;
    }


    printf("Test 7: Empty matrix operations\n");
    {
        lmmc_mat_t zero_dense = {0};
        lmmc_sparse_mat_t zero_sparse = {0};
        lmmc_vec_t x = {0}, y = {0};

        st = lmmc_mat_create(3, 3, &zero_dense);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


        st = lmmc_sparse_from_dense(&zero_dense, 0.0, &zero_sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&zero_dense); goto done; }

        if (zero_sparse.nnz != 0) { rc = 1; lmmc_sparse_destroy(&zero_sparse); lmmc_mat_destroy(&zero_dense); goto done; }


        st = lmmc_vec_create(3, &x);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&zero_sparse); lmmc_mat_destroy(&zero_dense); goto done; }
        st = lmmc_vec_fill(&x, 5.0);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test7_cleanup; }
        st = lmmc_vec_create(3, &y);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test7_cleanup; }

        st = lmmc_sparse_mat_vec_mul(&zero_sparse, &x, &y);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test7_cleanup; }

        for (size_t i = 0; i < 3; i++) {
            if (!lmmc_test_nearly_equal(y.data[i], 0.0, TEST_EPS_TIGHT)) {
                rc = 1; goto test7_cleanup;
            }
        }


        lmmc_sparse_mat_t zero_t = {0};
        st = lmmc_sparse_transpose(&zero_sparse, &zero_t);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test7_cleanup; }
        if (zero_t.nnz != 0) { rc = 1; lmmc_sparse_destroy(&zero_t); goto test7_cleanup; }
        lmmc_sparse_destroy(&zero_t);

    test7_cleanup:
        lmmc_vec_destroy(&y);
        lmmc_vec_destroy(&x);
        lmmc_sparse_destroy(&zero_sparse);
        lmmc_mat_destroy(&zero_dense);
        if (rc != 0) goto done;
    }


    printf("Test 8: Diagonal SpMV\n");
    {

        double diag_data[] = {
            2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 4.0
        };
        double x_vals[] = {1.0, 2.0, 3.0};
        lmmc_sparse_mat_t diag_sparse = {0};
        lmmc_vec_t x = {0}, y = {0};

        st = helper_build_sparse(diag_data, 3, 3, &diag_sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_vec_create(3, &x);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&diag_sparse); goto done; }
        for (size_t i = 0; i < 3; i++) LMMC_REAL_SET_D(&x.data[i], x_vals[i]);

        st = lmmc_vec_create(3, &y);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&x); lmmc_sparse_destroy(&diag_sparse); goto done; }

        st = lmmc_sparse_mat_vec_mul(&diag_sparse, &x, &y);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test8_cleanup; }


        if (!lmmc_test_nearly_equal(y.data[0], 2.0, TEST_EPS_TIGHT) ||
            !lmmc_test_nearly_equal(y.data[1], 6.0, TEST_EPS_TIGHT) ||
            !lmmc_test_nearly_equal(y.data[2], 12.0, TEST_EPS_TIGHT)) {
            rc = 1; goto test8_cleanup;
        }
    test8_cleanup:
        lmmc_vec_destroy(&y);
        lmmc_vec_destroy(&x);
        lmmc_sparse_destroy(&diag_sparse);
        if (rc != 0) goto done;
    }


    printf("Test 9: COO duplicate accumulation\n");
    {
        lmmc_sparse_coo_t coo = {0};
        lmmc_sparse_mat_t sparse = {0};
        lmmc_mat_t dense = {0};

        st = lmmc_sparse_coo_create(3, 3, 8, &coo);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


        lmmc_sparse_coo_add_entry(&coo, 0, 0, 1.0);
        lmmc_sparse_coo_add_entry(&coo, 0, 0, 2.0);
        lmmc_sparse_coo_add_entry(&coo, 0, 0, 3.0);

        lmmc_sparse_coo_add_entry(&coo, 1, 1, 5.0);

        lmmc_sparse_coo_add_entry(&coo, 2, 2, 4.0);
        lmmc_sparse_coo_add_entry(&coo, 2, 2, 6.0);

        st = lmmc_sparse_coo_to_csr(&coo, &sparse);
        lmmc_sparse_coo_destroy(&coo);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


        if (sparse.nnz != 3) { rc = 1; lmmc_sparse_destroy(&sparse); goto done; }


        st = lmmc_mat_create(3, 3, &dense);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&sparse); goto done; }
        st = lmmc_sparse_to_dense(&sparse, &dense);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&dense); lmmc_sparse_destroy(&sparse); goto done; }

        if (!lmmc_test_nearly_equal(dense.data[0], 6.0, TEST_EPS_TIGHT) ||
            !lmmc_test_nearly_equal(dense.data[4], 5.0, TEST_EPS_TIGHT) ||
            !lmmc_test_nearly_equal(dense.data[8], 10.0, TEST_EPS_TIGHT)) {
            rc = 1;
        }
        lmmc_mat_destroy(&dense);
        lmmc_sparse_destroy(&sparse);
        if (rc != 0) goto done;
    }


    printf("Test 10: COO->CSR/CSC consistency\n");
    {
        lmmc_sparse_coo_t coo = {0};
        lmmc_sparse_mat_t csr = {0}, csc = {0};
        lmmc_mat_t d_csr = {0}, d_csc = {0};

        st = lmmc_sparse_coo_create(3, 3, 8, &coo);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


        lmmc_sparse_coo_add_entry(&coo, 0, 0, 1.0);
        lmmc_sparse_coo_add_entry(&coo, 0, 2, 2.0);
        lmmc_sparse_coo_add_entry(&coo, 1, 1, 3.0);
        lmmc_sparse_coo_add_entry(&coo, 2, 0, 4.0);
        lmmc_sparse_coo_add_entry(&coo, 2, 2, 5.0);

        st = lmmc_sparse_coo_to_csr(&coo, &csr);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_coo_destroy(&coo); goto done; }

        st = lmmc_sparse_coo_to_csc(&coo, &csc);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&csr); lmmc_sparse_coo_destroy(&coo); goto done; }


        st = lmmc_mat_create(3, 3, &d_csr);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test10_cleanup; }
        st = lmmc_mat_create(3, 3, &d_csc);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test10_cleanup; }

        st = lmmc_sparse_to_dense(&csr, &d_csr);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test10_cleanup; }
        st = lmmc_sparse_to_dense(&csc, &d_csc);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test10_cleanup; }

        for (size_t i = 0; i < 9; i++) {
            if (!lmmc_test_nearly_equal(d_csr.data[i], d_csc.data[i], TEST_EPS_TIGHT)) {
                rc = 1; goto test10_cleanup;
            }
        }
    test10_cleanup:
        lmmc_mat_destroy(&d_csc);
        lmmc_mat_destroy(&d_csr);
        lmmc_sparse_destroy(&csc);
        lmmc_sparse_destroy(&csr);
        lmmc_sparse_coo_destroy(&coo);
        if (rc != 0) goto done;
    }


    printf("Test 11: Sparse LU solve\n");
    {

        double a_data[] = {
            2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 4.0
        };
        lmmc_sparse_mat_t a_csc = {0};
        lmmc_sparse_lu_t* lu = NULL;
        lmmc_vec_t b = {0}, x = {0};

        st = helper_build_sparse_csc(a_data, 3, 3, &a_csc);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_sparse_lu_symbolic(&a_csc, &lu);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&a_csc); goto done; }

        st = lmmc_sparse_lu_numeric(&a_csc, lu);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_lu_destroy(lu); lmmc_sparse_destroy(&a_csc); goto done; }

        st = lmmc_vec_create(3, &b);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_lu_destroy(lu); lmmc_sparse_destroy(&a_csc); goto done; }
        st = lmmc_vec_create(3, &x);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&b); lmmc_sparse_lu_destroy(lu); lmmc_sparse_destroy(&a_csc); goto done; }

        b.data[0] = 2.0; b.data[1] = 6.0; b.data[2] = 12.0;

        st = lmmc_sparse_lu_solve(lu, &b, &x);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test11_cleanup; }

        if (!lmmc_test_nearly_equal(x.data[0], 1.0, TEST_EPS_NORMAL) ||
            !lmmc_test_nearly_equal(x.data[1], 2.0, TEST_EPS_NORMAL) ||
            !lmmc_test_nearly_equal(x.data[2], 3.0, TEST_EPS_NORMAL)) {
            rc = 1;
        }
    test11_cleanup:
        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_sparse_lu_destroy(lu);
        lmmc_sparse_destroy(&a_csc);
        if (rc != 0) goto done;
    }


    printf("Test 12: Sparse Cholesky solve\n");
    {

        double a_data[] = {
            4.0, 1.0, 0.0,
            1.0, 4.0, 1.0,
            0.0, 1.0, 4.0
        };

        lmmc_sparse_mat_t a_csc = {0};
        lmmc_sparse_chol_t* chol = NULL;
        lmmc_vec_t b = {0}, x = {0};

        st = helper_build_sparse_csc(a_data, 3, 3, &a_csc);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_sparse_chol_symbolic(&a_csc, &chol);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&a_csc); goto done; }

        st = lmmc_sparse_chol_numeric(&a_csc, chol);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_chol_destroy(chol); lmmc_sparse_destroy(&a_csc); goto done; }

        st = lmmc_vec_create(3, &b);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_chol_destroy(chol); lmmc_sparse_destroy(&a_csc); goto done; }
        st = lmmc_vec_create(3, &x);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&b); lmmc_sparse_chol_destroy(chol); lmmc_sparse_destroy(&a_csc); goto done; }

        b.data[0] = 6.0; b.data[1] = 12.0; b.data[2] = 14.0;

        st = lmmc_sparse_chol_solve(chol, &b, &x);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test12_cleanup; }

        if (!lmmc_test_nearly_equal(x.data[0], 1.0, TEST_EPS_NORMAL) ||
            !lmmc_test_nearly_equal(x.data[1], 2.0, TEST_EPS_NORMAL) ||
            !lmmc_test_nearly_equal(x.data[2], 3.0, TEST_EPS_NORMAL)) {
            rc = 1;
        }
    test12_cleanup:
        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_sparse_chol_destroy(chol);
        lmmc_sparse_destroy(&a_csc);
        if (rc != 0) goto done;
    }


    printf("Test 13: Dimension mismatch\n");
    {
        double a_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        lmmc_sparse_mat_t sa = {0};
        lmmc_vec_t x_bad = {0}, y_bad = {0};

        st = helper_build_sparse(a_data, 2, 3, &sa);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


        st = lmmc_vec_create(2, &x_bad);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&sa); goto done; }
        st = lmmc_vec_create(2, &y_bad);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&x_bad); lmmc_sparse_destroy(&sa); goto done; }

        st = lmmc_sparse_mat_vec_mul(&sa, &x_bad, &y_bad);
        if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
            rc = 1; goto test13_cleanup;
        }


        lmmc_sparse_mat_t sb = {0}, sc = {0};
        double b_data[] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
        st = helper_build_sparse(b_data, 2, 3, &sb);
        if (st != LMMC_STATUS_OK) { rc = 1; goto test13_cleanup; }

        st = lmmc_sparse_mat_mat_mul_sparse(&sa, &sb, &sc);
        if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
            rc = 1; lmmc_sparse_destroy(&sc); lmmc_sparse_destroy(&sb); goto test13_cleanup;
        }
        lmmc_sparse_destroy(&sb);

    test13_cleanup:
        lmmc_vec_destroy(&y_bad);
        lmmc_vec_destroy(&x_bad);
        lmmc_sparse_destroy(&sa);
        if (rc != 0) goto done;
    }


    printf("Test 14: Frobenius norm\n");
    {

        double data[] = {3.0, 0.0, 0.0, 4.0};
        lmmc_sparse_mat_t sparse = {0};
        lmmc_real_t norm_sparse = 0.0;

        st = helper_build_sparse(data, 2, 2, &sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_sparse_norm_fro(&sparse, &norm_sparse);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&sparse); goto done; }

        if (!lmmc_test_nearly_equal(norm_sparse, 5.0, TEST_EPS_TIGHT)) {
            rc = 1; lmmc_sparse_destroy(&sparse); goto done;
        }

        lmmc_sparse_destroy(&sparse);


        double data2[] = {
            1.0, 2.0, 0.0,
            0.0, 3.0, 4.0,
            5.0, 0.0, 6.0
        };
        lmmc_sparse_mat_t sparse2 = {0};
        lmmc_mat_t dense2 = {0};
        lmmc_real_t norm_s2 = 0.0, norm_d2 = 0.0;

        st = helper_build_sparse(data2, 3, 3, &sparse2);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_sparse_norm_fro(&sparse2, &norm_s2);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&sparse2); goto done; }

        st = lmmc_mat_create(3, 3, &dense2);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_sparse_destroy(&sparse2); goto done; }
        for (size_t i = 0; i < 9; i++) LMMC_REAL_SET_D(&dense2.data[i], data2[i]);

        st = lmmc_mat_norm_fro(&dense2, &norm_d2);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&dense2); lmmc_sparse_destroy(&sparse2); goto done; }

        if (!lmmc_test_nearly_equal(norm_s2, norm_d2, TEST_EPS_TIGHT)) {
            rc = 1;
        }
        lmmc_mat_destroy(&dense2);
        lmmc_sparse_destroy(&sparse2);
        if (rc != 0) goto done;
    }

done:
    if (rc != 0) {
        printf("sparse extended test failed\n");
    }
    return rc;
}
