/**
 * @file test_linalg_extended.c
 * 针对 LMMC 中 linalg extended 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define TEST_EPS_NORMAL  1e-10
#define TEST_EPS_TIGHT   1e-12


static double compute_residual(const double* A, size_t n, const double* x, const double* b) {
    double max_res = 0.0;
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        double res = fabs(sum - b[i]);
        if (res > max_res) max_res = res;
    }
    return max_res;
}

int main(void) {
    int rc = 0;


    {
        lmmc_mat_t a = {0};
        lmmc_vec_t b = {0}, x = {0};
        size_t piv[2];
        double A_orig[] = {2.0, 1.0, 5.0, 7.0};
        double b_vals[] = {11.0, 13.0};

        lmmc_mat_create(2, 2, &a);
        lmmc_vec_create(2, &b);
        lmmc_vec_create(2, &x);

        for (size_t i = 0; i < 4; i++) a.data[i] = A_orig[i];
        for (size_t i = 0; i < 2; i++) b.data[i] = b_vals[i];

        lmmc_status_t st = lmmc_lu_decompose_inplace(&a, piv, NULL);
        if (st != LMMC_STATUS_OK) { printf("LU 2x2 decompose failed\n"); rc = 1; goto done; }
        st = lmmc_lu_solve(&a, piv, &b, &x);
        if (st != LMMC_STATUS_OK) { printf("LU 2x2 solve failed\n"); rc = 1; goto done; }

        double residual = compute_residual(A_orig, 2, x.data, b_vals);
        if (residual > TEST_EPS_NORMAL) {
            printf("LU 2x2 residual too large: %e\n", residual);
            rc = 1; goto done;
        }
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};
        lmmc_vec_t b = {0}, x = {0};
        size_t piv[3];
        double A_orig[] = {3.0, 2.0, -1.0, 2.0, -2.0, 4.0, -1.0, 0.5, -1.0};
        double b_vals[] = {1.0, -2.0, 0.0};

        lmmc_mat_create(3, 3, &a);
        lmmc_vec_create(3, &b);
        lmmc_vec_create(3, &x);
        for (size_t i = 0; i < 9; i++) a.data[i] = A_orig[i];
        for (size_t i = 0; i < 3; i++) b.data[i] = b_vals[i];

        lmmc_status_t st = lmmc_lu_decompose_inplace(&a, piv, NULL);
        if (st != LMMC_STATUS_OK) { printf("LU 3x3 decompose failed\n"); rc = 1; goto done; }
        st = lmmc_lu_solve(&a, piv, &b, &x);
        if (st != LMMC_STATUS_OK) { printf("LU 3x3 solve failed\n"); rc = 1; goto done; }

        double residual = compute_residual(A_orig, 3, x.data, b_vals);
        if (residual > TEST_EPS_NORMAL) {
            printf("LU 3x3 residual too large: %e\n", residual);
            rc = 1; goto done;
        }
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};
        lmmc_vec_t b = {0}, x = {0};
        size_t piv[5];
        double A_orig[] = {
            5.0, 7.0, 6.0, 5.0, 1.0,
            7.0, 10.0, 8.0, 7.0, 2.0,
            6.0, 8.0, 10.0, 9.0, 3.0,
            5.0, 7.0, 9.0, 10.0, 4.0,
            1.0, 2.0, 3.0, 4.0, 5.0
        };

        double x_true[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        double b_vals[5];
        for (size_t i = 0; i < 5; i++) {
            b_vals[i] = 0.0;
            for (size_t j = 0; j < 5; j++)
                b_vals[i] += A_orig[i * 5 + j] * x_true[j];
        }

        lmmc_mat_create(5, 5, &a);
        lmmc_vec_create(5, &b);
        lmmc_vec_create(5, &x);
        for (size_t i = 0; i < 25; i++) a.data[i] = A_orig[i];
        for (size_t i = 0; i < 5; i++) b.data[i] = b_vals[i];

        lmmc_status_t st = lmmc_lu_decompose_inplace(&a, piv, NULL);
        if (st != LMMC_STATUS_OK) { printf("LU 5x5 decompose failed\n"); rc = 1; goto done; }
        st = lmmc_lu_solve(&a, piv, &b, &x);
        if (st != LMMC_STATUS_OK) { printf("LU 5x5 solve failed\n"); rc = 1; goto done; }

        double residual = compute_residual(A_orig, 5, x.data, b_vals);
        if (residual > TEST_EPS_NORMAL) {
            printf("LU 5x5 residual too large: %e\n", residual);
            rc = 1; goto done;
        }
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};
        lmmc_vec_t b = {0}, x = {0};
        size_t piv[10];

        double A_orig[100];
        double x_true[10];
        double b_vals[10];
        for (size_t i = 0; i < 10; i++) {
            x_true[i] = (double)(i + 1);
            for (size_t j = 0; j < 10; j++) {
                if (i == j)
                    A_orig[i * 10 + j] = 20.0;
                else
                    A_orig[i * 10 + j] = 1.0 / (1.0 + (double)((i > j) ? (i - j) : (j - i)));
            }
        }
        for (size_t i = 0; i < 10; i++) {
            b_vals[i] = 0.0;
            for (size_t j = 0; j < 10; j++)
                b_vals[i] += A_orig[i * 10 + j] * x_true[j];
        }

        lmmc_mat_create(10, 10, &a);
        lmmc_vec_create(10, &b);
        lmmc_vec_create(10, &x);
        for (size_t i = 0; i < 100; i++) a.data[i] = A_orig[i];
        for (size_t i = 0; i < 10; i++) b.data[i] = b_vals[i];

        lmmc_status_t st = lmmc_lu_decompose_inplace(&a, piv, NULL);
        if (st != LMMC_STATUS_OK) { printf("LU 10x10 decompose failed\n"); rc = 1; goto done; }
        st = lmmc_lu_solve(&a, piv, &b, &x);
        if (st != LMMC_STATUS_OK) { printf("LU 10x10 solve failed\n"); rc = 1; goto done; }

        double residual = compute_residual(A_orig, 10, x.data, b_vals);
        if (residual > TEST_EPS_NORMAL) {
            printf("LU 10x10 residual too large: %e\n", residual);
            rc = 1; goto done;
        }
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};
        lmmc_vec_t b = {0}, x = {0};

        double A_orig[] = {4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0};
        double x_true[] = {1.0, 2.0, 3.0};
        double b_vals[3];
        for (size_t i = 0; i < 3; i++) {
            b_vals[i] = 0.0;
            for (size_t j = 0; j < 3; j++)
                b_vals[i] += A_orig[i * 3 + j] * x_true[j];
        }

        lmmc_mat_create(3, 3, &a);
        lmmc_vec_create(3, &b);
        lmmc_vec_create(3, &x);
        for (size_t i = 0; i < 9; i++) a.data[i] = A_orig[i];
        for (size_t i = 0; i < 3; i++) b.data[i] = b_vals[i];

        lmmc_status_t st = lmmc_cholesky_decompose_inplace(&a);
        if (st != LMMC_STATUS_OK) { printf("Cholesky 3x3 decompose failed\n"); rc = 1; goto done; }
        st = lmmc_cholesky_solve(&a, &b, &x);
        if (st != LMMC_STATUS_OK) { printf("Cholesky 3x3 solve failed\n"); rc = 1; goto done; }

        double residual = compute_residual(A_orig, 3, x.data, b_vals);
        if (residual > TEST_EPS_NORMAL) {
            printf("Cholesky 3x3 residual too large: %e\n", residual);
            rc = 1; goto done;
        }
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};
        lmmc_vec_t b = {0}, x = {0};


        double tau[2];

        lmmc_mat_create(5, 2, &a);
        lmmc_vec_create(5, &b);
        lmmc_vec_create(2, &x);

        for (size_t i = 0; i < 5; i++) {
            a.data[i * 2 + 0] = 1.0;
            a.data[i * 2 + 1] = (double)i;
        }
        b.data[0] = 1.0; b.data[1] = 3.0; b.data[2] = 5.0;
        b.data[3] = 7.0; b.data[4] = 9.0;

        lmmc_status_t st = lmmc_qr_decompose_inplace(&a, tau, 2);
        if (st != LMMC_STATUS_OK) { printf("QR overdetermined decompose failed\n"); rc = 1; goto done; }
        st = lmmc_qr_solve(&a, tau, &b, &x);
        if (st != LMMC_STATUS_OK) { printf("QR overdetermined solve failed\n"); rc = 1; goto done; }

        if (!lmmc_test_nearly_equal(x.data[0], 1.0, TEST_EPS_NORMAL) ||
            !lmmc_test_nearly_equal(x.data[1], 2.0, TEST_EPS_NORMAL)) {
            printf("QR least-squares solution incorrect: [%f, %f]\n", x.data[0], x.data[1]);
            rc = 1; goto done;
        }
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};
        size_t piv[3];

        lmmc_mat_create(3, 3, &a);

        for (size_t i = 0; i < 9; i++) a.data[i] = 0.0;
        a.data[0] = 1.0; a.data[4] = 1.0; a.data[8] = 1.0;

        lmmc_status_t st = lmmc_lu_decompose_inplace(&a, piv, NULL);
        if (st != LMMC_STATUS_OK) { printf("LU identity decompose failed\n"); rc = 1; goto done; }


        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                if (!lmmc_test_nearly_equal(a.data[i * 3 + j], expected, TEST_EPS_TIGHT)) {
                    printf("LU identity: element [%zu][%zu] = %f, expected %f\n",
                           i, j, a.data[i * 3 + j], expected);
                    rc = 1; goto done;
                }
            }
        }
        lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};
        double diag_vals[] = {4.0, 9.0, 16.0, 25.0};

        lmmc_mat_create(4, 4, &a);
        for (size_t i = 0; i < 16; i++) a.data[i] = 0.0;
        for (size_t i = 0; i < 4; i++) a.data[i * 4 + i] = diag_vals[i];

        lmmc_status_t st = lmmc_cholesky_decompose_inplace(&a);
        if (st != LMMC_STATUS_OK) { printf("Cholesky diagonal decompose failed\n"); rc = 1; goto done; }


        for (size_t i = 0; i < 4; i++) {
            double expected_diag = sqrt(diag_vals[i]);
            if (!lmmc_test_nearly_equal(a.data[i * 4 + i], expected_diag, TEST_EPS_TIGHT)) {
                printf("Cholesky diagonal: L[%zu][%zu] = %f, expected %f\n",
                       i, i, a.data[i * 4 + i], expected_diag);
                rc = 1; goto done;
            }

            for (size_t j = 0; j < 4; j++) {
                if (j != i) {
                    if (!lmmc_test_nearly_equal(a.data[i * 4 + j], 0.0, TEST_EPS_TIGHT)) {
                        printf("Cholesky diagonal: L[%zu][%zu] = %f, expected 0\n",
                               i, j, a.data[i * 4 + j]);
                        rc = 1; goto done;
                    }
                }
            }
        }
        lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};
        size_t piv[3];

        lmmc_mat_create(3, 3, &a);

        a.data[0] = 1.0; a.data[1] = 2.0; a.data[2] = 3.0;
        a.data[3] = 2.0; a.data[4] = 4.0; a.data[5] = 6.0;
        a.data[6] = 3.0; a.data[7] = 6.0; a.data[8] = 9.0;

        lmmc_status_t st = lmmc_lu_decompose_inplace(&a, piv, NULL);
        if (st != LMMC_STATUS_SINGULAR_MATRIX) {
            printf("Singular matrix: expected SINGULAR_MATRIX, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }
        lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};

        lmmc_mat_create(3, 3, &a);

        a.data[0] = 1.0;  a.data[1] = 2.0;  a.data[2] = 3.0;
        a.data[3] = 2.0;  a.data[4] = 1.0;  a.data[5] = 2.0;
        a.data[6] = 3.0;  a.data[7] = 2.0;  a.data[8] = 1.0;

        lmmc_status_t st = lmmc_cholesky_decompose_inplace(&a);
        if (st != LMMC_STATUS_NOT_POSITIVE_DEFINITE) {
            printf("Non-PD matrix: expected NOT_POSITIVE_DEFINITE, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }
        lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};
        size_t piv[4];
        size_t swap_count = 0;
        double A_orig[] = {
            2.0, 1.0, 1.0, 0.0,
            4.0, 3.0, 3.0, 1.0,
            8.0, 7.0, 9.0, 5.0,
            6.0, 7.0, 9.0, 8.0
        };

        lmmc_mat_create(4, 4, &a);
        for (size_t i = 0; i < 16; i++) a.data[i] = A_orig[i];

        lmmc_status_t st = lmmc_lu_decompose_inplace(&a, piv, &swap_count);
        if (st != LMMC_STATUS_OK) { printf("P*A=L*U decompose failed\n"); rc = 1; goto done; }


        double L[16], U[16];
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                if (i == j) {
                    L[i * 4 + j] = 1.0;
                    U[i * 4 + j] = a.data[i * 4 + j];
                } else if (i > j) {
                    L[i * 4 + j] = a.data[i * 4 + j];
                    U[i * 4 + j] = 0.0;
                } else {
                    L[i * 4 + j] = 0.0;
                    U[i * 4 + j] = a.data[i * 4 + j];
                }
            }
        }


        double LU[16];
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                LU[i * 4 + j] = 0.0;
                for (size_t k = 0; k < 4; k++)
                    LU[i * 4 + j] += L[i * 4 + k] * U[k * 4 + j];
            }
        }


        double PA[16];
        memcpy(PA, A_orig, sizeof(PA));

        for (size_t k = 0; k < 4; k++) {
            if (piv[k] != k) {
                for (size_t j = 0; j < 4; j++) {
                    double tmp = PA[k * 4 + j];
                    PA[k * 4 + j] = PA[piv[k] * 4 + j];
                    PA[piv[k] * 4 + j] = tmp;
                }
            }
        }


        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                if (!lmmc_test_nearly_equal(PA[i * 4 + j], LU[i * 4 + j], TEST_EPS_NORMAL)) {
                    printf("P*A=L*U failed at [%zu][%zu]: PA=%f, LU=%f\n",
                           i, j, PA[i * 4 + j], LU[i * 4 + j]);
                    rc = 1; goto done;
                }
            }
        }
        lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};
        lmmc_vec_t b = {0}, x = {0};
        size_t piv[5];
        size_t n = 5;

        lmmc_mat_create(n, n, &a);
        lmmc_vec_create(n, &b);
        lmmc_vec_create(n, &x);


        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                a.data[i * n + j] = 1.0 / (double)(i + j + 1);
            }
            b.data[i] = 1.0;
        }

        lmmc_status_t st = lmmc_lu_decompose_inplace(&a, piv, NULL);

        if (st != LMMC_STATUS_OK) {
            printf("Hilbert 5x5 LU decompose failed (unexpected): %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }
        st = lmmc_lu_solve(&a, piv, &b, &x);
        if (st != LMMC_STATUS_OK) {
            printf("Hilbert 5x5 LU solve failed (unexpected): %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }

        for (size_t i = 0; i < n; i++) {
            if (!isfinite(x.data[i])) {
                printf("Hilbert 5x5: non-finite result at x[%zu]\n", i);
                rc = 1; goto done;
            }
        }
        lmmc_vec_destroy(&x); lmmc_vec_destroy(&b); lmmc_mat_destroy(&a);
    }


    {
        lmmc_mat_t a = {0};
        lmmc_vec_t b = {0}, x = {0};
        size_t piv[2];
        double tau[2];


        lmmc_status_t st = lmmc_lu_decompose_inplace(NULL, piv, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("LU NULL matrix: expected INVALID_ARGUMENT, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }


        lmmc_mat_create(2, 2, &a);
        a.data[0] = 1.0; a.data[1] = 0.0; a.data[2] = 0.0; a.data[3] = 1.0;
        st = lmmc_lu_decompose_inplace(&a, NULL, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("LU NULL pivots: expected INVALID_ARGUMENT, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }
        lmmc_mat_destroy(&a);


        st = lmmc_cholesky_decompose_inplace(NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("Cholesky NULL: expected INVALID_ARGUMENT, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }


        st = lmmc_qr_decompose_inplace(NULL, tau, 2);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("QR NULL matrix: expected INVALID_ARGUMENT, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }


        lmmc_mat_create(3, 2, &a);
        for (size_t i = 0; i < 6; i++) a.data[i] = 1.0;
        st = lmmc_qr_decompose_inplace(&a, NULL, 2);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("QR NULL tau: expected INVALID_ARGUMENT, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }
        lmmc_mat_destroy(&a);


        st = lmmc_lu_solve(NULL, piv, &b, &x);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("LU solve NULL lu: expected INVALID_ARGUMENT, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }


        st = lmmc_cholesky_solve(NULL, &b, &x);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("Cholesky solve NULL l: expected INVALID_ARGUMENT, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }


        st = lmmc_qr_solve(NULL, tau, &b, &x);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("QR solve NULL qr: expected INVALID_ARGUMENT, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }


        lmmc_mat_create(2, 3, &a);
        for (size_t i = 0; i < 6; i++) a.data[i] = 1.0;
        st = lmmc_lu_decompose_inplace(&a, piv, NULL);
        if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
            printf("LU non-square: expected DIMENSION_MISMATCH, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }
        lmmc_mat_destroy(&a);


        lmmc_mat_create(2, 3, &a);
        for (size_t i = 0; i < 6; i++) a.data[i] = 1.0;
        st = lmmc_cholesky_decompose_inplace(&a);
        if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
            printf("Cholesky non-square: expected DIMENSION_MISMATCH, got %s\n", lmmc_status_string(st));
            rc = 1; goto done;
        }
        lmmc_mat_destroy(&a);
    }

done:
    if (rc != 0) {
        printf("linalg_extended test failed\n");
    } else {
        printf("linalg_extended test passed\n");
    }
    return rc;
}
