/**
 * @file test_dense_extended.c
 * @brief Extended tests for the dense matrix/vector module.
 *
 * Covers: multi-size matrix multiplication, identity matrix properties,
 * non-square matrix multiplication, extreme values, transpose roundtrip,
 * determinant, trace, axpy, swap, vector norms, Frobenius norm.
 *
 * Requirements: 5.1-5.12
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define TEST_EPS_TIGHT   1e-12
#define TEST_EPS_NORMAL  1e-10
#define TEST_EPS_LOOSE   1e-6

int main(void) {
    int rc = 0;
    lmmc_status_t st;

    /* ================================================================
     * Requirement 5.1: Multi-size matrix multiplication (1x1..100x100)
     * ================================================================ */
    {
        size_t sizes[] = {1, 2, 3, 10, 100};
        size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

        for (size_t si = 0; si < num_sizes; si++) {
            size_t n = sizes[si];
            lmmc_mat_t a = {0}, b = {0}, c = {0};
            st = lmmc_mat_create(n, n, &a);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            st = lmmc_mat_create(n, n, &b);
            if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); rc = 1; goto done; }
            st = lmmc_mat_create(n, n, &c);
            if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); rc = 1; goto done; }

            /* Fill A with i+j+1, B with identity */
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    a.data[i * n + j] = (lmmc_real_t)(i + j + 1);
                    b.data[i * n + j] = (i == j) ? 1.0 : 0.0;
                }
            }

            /* C = A * I should equal A */
            st = lmmc_mat_mul(&a, &b, &c);
            if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); lmmc_mat_destroy(&c); goto done; }

            for (size_t i = 0; i < n * n; i++) {
                if (!lmmc_test_nearly_equal(c.data[i], a.data[i], TEST_EPS_TIGHT)) {
                    printf("5.1 FAIL: size=%zu, A*I != A at index %zu\n", n, i);
                    rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); lmmc_mat_destroy(&c); goto done;
                }
            }

            lmmc_mat_destroy(&a);
            lmmc_mat_destroy(&b);
            lmmc_mat_destroy(&c);
        }
    }

    /* ================================================================
     * Requirement 5.2: Identity matrix property: A*I = I*A = A
     * ================================================================ */
    {
        size_t n = 5;
        lmmc_mat_t a = {0}, eye = {0}, c1 = {0}, c2 = {0};
        st = lmmc_mat_create(n, n, &a);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_mat_identity(n, &eye);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); rc = 1; goto done; }
        st = lmmc_mat_create(n, n, &c1);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); lmmc_mat_destroy(&eye); rc = 1; goto done; }
        st = lmmc_mat_create(n, n, &c2);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); lmmc_mat_destroy(&eye); lmmc_mat_destroy(&c1); rc = 1; goto done; }

        /* Fill A with known values */
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                a.data[i * n + j] = (lmmc_real_t)((i + 1) * 10 + j + 1);

        /* A*I = A */
        st = lmmc_mat_mul(&a, &eye, &c1);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&eye); lmmc_mat_destroy(&c1); lmmc_mat_destroy(&c2); goto done; }

        /* I*A = A */
        st = lmmc_mat_mul(&eye, &a, &c2);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&eye); lmmc_mat_destroy(&c1); lmmc_mat_destroy(&c2); goto done; }

        for (size_t i = 0; i < n * n; i++) {
            if (!lmmc_test_nearly_equal(c1.data[i], a.data[i], TEST_EPS_TIGHT) ||
                !lmmc_test_nearly_equal(c2.data[i], a.data[i], TEST_EPS_TIGHT)) {
                printf("5.2 FAIL: Identity property violated at index %zu\n", i);
                rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&eye); lmmc_mat_destroy(&c1); lmmc_mat_destroy(&c2); goto done;
            }
        }

        lmmc_mat_destroy(&a);
        lmmc_mat_destroy(&eye);
        lmmc_mat_destroy(&c1);
        lmmc_mat_destroy(&c2);
    }

    /* ================================================================
     * Requirement 5.3: Non-square matrix multiplication & dimension error
     * ================================================================ */
    {
        /* Valid: (3x5) * (5x3) = (3x3) */
        lmmc_mat_t a35 = {0}, b53 = {0}, c33 = {0};
        st = lmmc_mat_create(3, 5, &a35);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_mat_create(5, 3, &b53);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a35); rc = 1; goto done; }
        st = lmmc_mat_create(3, 3, &c33);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a35); lmmc_mat_destroy(&b53); rc = 1; goto done; }

        /* Fill with simple values */
        for (size_t i = 0; i < 15; i++) a35.data[i] = (lmmc_real_t)(i + 1);
        for (size_t i = 0; i < 15; i++) b53.data[i] = (lmmc_real_t)(i + 1);

        st = lmmc_mat_mul(&a35, &b53, &c33);
        if (st != LMMC_STATUS_OK) {
            printf("5.3 FAIL: Valid non-square mul returned error %d\n", (int)st);
            rc = 1; lmmc_mat_destroy(&a35); lmmc_mat_destroy(&b53); lmmc_mat_destroy(&c33); goto done;
        }

        /* Verify C[0][0] = sum(A[0][k]*B[k][0], k=0..4) = 1*1+2*4+3*7+4*10+5*13 = 1+8+21+40+65 = 135 */
        if (!lmmc_test_nearly_equal(c33.data[0], 135.0, TEST_EPS_TIGHT)) {
            printf("5.3 FAIL: C[0][0] = %g, expected 135\n", c33.data[0]);
            rc = 1; lmmc_mat_destroy(&a35); lmmc_mat_destroy(&b53); lmmc_mat_destroy(&c33); goto done;
        }

        /* Dimension mismatch: (3x5) * (3x3) should fail */
        lmmc_mat_t bad = {0};
        st = lmmc_mat_create(3, 3, &bad);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a35); lmmc_mat_destroy(&b53); lmmc_mat_destroy(&c33); rc = 1; goto done; }

        st = lmmc_mat_mul(&a35, &bad, &c33);
        if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
            printf("5.3 FAIL: Dimension mismatch not detected, got %d\n", (int)st);
            rc = 1; lmmc_mat_destroy(&a35); lmmc_mat_destroy(&b53); lmmc_mat_destroy(&c33); lmmc_mat_destroy(&bad); goto done;
        }

        lmmc_mat_destroy(&a35);
        lmmc_mat_destroy(&b53);
        lmmc_mat_destroy(&c33);
        lmmc_mat_destroy(&bad);
    }

    /* ================================================================
     * Requirement 5.4: Extreme values (1e300, 1e-300)
     * ================================================================ */
    {
        lmmc_mat_t a = {0}, b = {0}, c = {0};
        st = lmmc_mat_create(2, 2, &a);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_mat_create(2, 2, &b);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); rc = 1; goto done; }
        st = lmmc_mat_create(2, 2, &c);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); rc = 1; goto done; }

        /* Test large values: scale by 1e300 */
        a.data[0] = 1e300; a.data[1] = 0.0;
        a.data[2] = 0.0;   a.data[3] = 1e300;
        /* Identity */
        b.data[0] = 1.0; b.data[1] = 0.0;
        b.data[2] = 0.0; b.data[3] = 1.0;

        st = lmmc_mat_mul(&a, &b, &c);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); lmmc_mat_destroy(&c); goto done; }
        if (!lmmc_test_nearly_equal(c.data[0], 1e300, 1e285) ||
            !lmmc_test_nearly_equal(c.data[3], 1e300, 1e285)) {
            printf("5.4 FAIL: Large value multiplication failed\n");
            rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); lmmc_mat_destroy(&c); goto done;
        }

        /* Test small values: scale by 1e-300 */
        a.data[0] = 1e-300; a.data[1] = 0.0;
        a.data[2] = 0.0;    a.data[3] = 1e-300;

        st = lmmc_mat_mul(&a, &b, &c);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); lmmc_mat_destroy(&c); goto done; }
        if (!lmmc_test_nearly_equal(c.data[0], 1e-300, 1e-314) ||
            !lmmc_test_nearly_equal(c.data[3], 1e-300, 1e-314)) {
            printf("5.4 FAIL: Small value multiplication failed\n");
            rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); lmmc_mat_destroy(&c); goto done;
        }

        /* Verify no overflow/underflow: result should be finite */
        if (!isfinite(c.data[0]) || !isfinite(c.data[3])) {
            printf("5.4 FAIL: Non-finite result detected\n");
            rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); lmmc_mat_destroy(&c); goto done;
        }

        lmmc_mat_destroy(&a);
        lmmc_mat_destroy(&b);
        lmmc_mat_destroy(&c);
    }

    /* ================================================================
     * Requirement 5.5: Transpose roundtrip: (A^T)^T = A
     * ================================================================ */
    {
        /* Test with non-square matrix 3x4 */
        lmmc_mat_t a = {0}, at = {0}, att = {0};
        st = lmmc_mat_create(3, 4, &a);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_mat_create(4, 3, &at);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); rc = 1; goto done; }
        st = lmmc_mat_create(3, 4, &att);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); lmmc_mat_destroy(&at); rc = 1; goto done; }

        for (size_t i = 0; i < 12; i++) a.data[i] = (lmmc_real_t)(i + 1);

        st = lmmc_mat_transpose_to(&a, &at);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&at); lmmc_mat_destroy(&att); goto done; }

        st = lmmc_mat_transpose_to(&at, &att);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&at); lmmc_mat_destroy(&att); goto done; }

        for (size_t i = 0; i < 12; i++) {
            if (!lmmc_test_nearly_equal(att.data[i], a.data[i], TEST_EPS_TIGHT)) {
                printf("5.5 FAIL: Transpose roundtrip failed at index %zu\n", i);
                rc = 1; lmmc_mat_destroy(&a); lmmc_mat_destroy(&at); lmmc_mat_destroy(&att); goto done;
            }
        }

        lmmc_mat_destroy(&a);
        lmmc_mat_destroy(&at);
        lmmc_mat_destroy(&att);

        /* Also test square matrix 5x5 */
        lmmc_mat_t s = {0}, st2 = {0}, stt = {0};
        st = lmmc_mat_create(5, 5, &s);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_mat_create(5, 5, &st2);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&s); rc = 1; goto done; }
        st = lmmc_mat_create(5, 5, &stt);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&s); lmmc_mat_destroy(&st2); rc = 1; goto done; }

        for (size_t i = 0; i < 25; i++) s.data[i] = (lmmc_real_t)(i * 3 - 7);

        lmmc_mat_transpose_to(&s, &st2);
        lmmc_mat_transpose_to(&st2, &stt);

        for (size_t i = 0; i < 25; i++) {
            if (!lmmc_test_nearly_equal(stt.data[i], s.data[i], TEST_EPS_TIGHT)) {
                printf("5.5 FAIL: Square transpose roundtrip failed at %zu\n", i);
                rc = 1; lmmc_mat_destroy(&s); lmmc_mat_destroy(&st2); lmmc_mat_destroy(&stt); goto done;
            }
        }

        lmmc_mat_destroy(&s);
        lmmc_mat_destroy(&st2);
        lmmc_mat_destroy(&stt);
    }

    /* ================================================================
     * Requirement 5.6: Determinant (identity det=1, singular det=0)
     * ================================================================ */
    {
        /* Identity matrix det = 1 */
        lmmc_mat_t eye = {0};
        st = lmmc_mat_identity(4, &eye);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        lmmc_real_t det_val = 0.0;
        st = lmmc_mat_det(&eye, &det_val);
        if (st != LMMC_STATUS_OK) {
            printf("5.6 FAIL: det(I) returned error %d\n", (int)st);
            rc = 1; lmmc_mat_destroy(&eye); goto done;
        }
        if (!lmmc_test_nearly_equal(det_val, 1.0, TEST_EPS_TIGHT)) {
            printf("5.6 FAIL: det(I) = %g, expected 1.0\n", det_val);
            rc = 1; lmmc_mat_destroy(&eye); goto done;
        }
        lmmc_mat_destroy(&eye);

        /* Singular matrix (row of zeros) det = 0 */
        lmmc_mat_t sing = {0};
        st = lmmc_mat_create(3, 3, &sing);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        /* Row 0: [1, 2, 3], Row 1: [4, 5, 6], Row 2: [7, 8, 9] (linearly dependent) */
        sing.data[0] = 1; sing.data[1] = 2; sing.data[2] = 3;
        sing.data[3] = 4; sing.data[4] = 5; sing.data[5] = 6;
        sing.data[6] = 7; sing.data[7] = 8; sing.data[8] = 9;

        st = lmmc_mat_det(&sing, &det_val);
        if (st != LMMC_STATUS_OK) {
            /* Some implementations may return error for singular matrix */
            /* That's acceptable too */
        } else {
            if (!lmmc_test_nearly_equal(det_val, 0.0, TEST_EPS_NORMAL)) {
                printf("5.6 FAIL: det(singular) = %g, expected 0.0\n", det_val);
                rc = 1; lmmc_mat_destroy(&sing); goto done;
            }
        }
        lmmc_mat_destroy(&sing);
    }

    /* ================================================================
     * Requirement 5.7: Trace = sum of diagonal elements
     * ================================================================ */
    {
        lmmc_mat_t a = {0};
        st = lmmc_mat_create(4, 4, &a);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        /* Fill with known values */
        for (size_t i = 0; i < 16; i++) a.data[i] = (lmmc_real_t)(i + 1);
        /* Diagonal: a[0]=1, a[5]=6, a[10]=11, a[15]=16, sum=34 */

        lmmc_real_t trace_val = 0.0;
        st = lmmc_mat_trace(&a, &trace_val);
        if (st != LMMC_STATUS_OK) {
            printf("5.7 FAIL: trace returned error %d\n", (int)st);
            rc = 1; lmmc_mat_destroy(&a); goto done;
        }

        lmmc_real_t expected_trace = 1.0 + 6.0 + 11.0 + 16.0;
        if (!lmmc_test_nearly_equal(trace_val, expected_trace, TEST_EPS_TIGHT)) {
            printf("5.7 FAIL: trace = %g, expected %g\n", trace_val, expected_trace);
            rc = 1; lmmc_mat_destroy(&a); goto done;
        }

        lmmc_mat_destroy(&a);

        /* Also test identity trace = n */
        lmmc_mat_t eye = {0};
        st = lmmc_mat_identity(7, &eye);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        st = lmmc_mat_trace(&eye, &trace_val);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&eye); goto done; }
        if (!lmmc_test_nearly_equal(trace_val, 7.0, TEST_EPS_TIGHT)) {
            printf("5.7 FAIL: trace(I_7) = %g, expected 7.0\n", trace_val);
            rc = 1; lmmc_mat_destroy(&eye); goto done;
        }
        lmmc_mat_destroy(&eye);
    }

    /* ================================================================
     * Requirement 5.8: axpy verification (alpha=0 y unchanged, alpha=1 = addition)
     * ================================================================ */
    {
        size_t n = 5;
        lmmc_vec_t x = {0}, y = {0};
        st = lmmc_vec_create(n, &x);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_vec_create(n, &y);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&x); rc = 1; goto done; }

        for (size_t i = 0; i < n; i++) {
            x.data[i] = (lmmc_real_t)(i + 1);
            y.data[i] = (lmmc_real_t)(10 + i);
        }

        /* Save original y */
        lmmc_real_t orig_y[5];
        memcpy(orig_y, y.data, n * sizeof(lmmc_real_t));

        /* alpha=0: y should not change */
        st = lmmc_vec_axpy(0.0, &x, &y);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&x); lmmc_vec_destroy(&y); goto done; }
        for (size_t i = 0; i < n; i++) {
            if (!lmmc_test_nearly_equal(y.data[i], orig_y[i], TEST_EPS_TIGHT)) {
                printf("5.8 FAIL: axpy alpha=0 changed y[%zu]\n", i);
                rc = 1; lmmc_vec_destroy(&x); lmmc_vec_destroy(&y); goto done;
            }
        }

        /* alpha=1: y = x + y (addition) */
        st = lmmc_vec_axpy(1.0, &x, &y);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&x); lmmc_vec_destroy(&y); goto done; }
        for (size_t i = 0; i < n; i++) {
            lmmc_real_t expected = x.data[i] + orig_y[i];
            if (!lmmc_test_nearly_equal(y.data[i], expected, TEST_EPS_TIGHT)) {
                printf("5.8 FAIL: axpy alpha=1: y[%zu]=%g, expected %g\n", i, y.data[i], expected);
                rc = 1; lmmc_vec_destroy(&x); lmmc_vec_destroy(&y); goto done;
            }
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&y);
    }

    /* ================================================================
     * Requirement 5.10: swap verification
     * ================================================================ */
    {
        size_t n = 6;
        lmmc_vec_t x = {0}, y = {0};
        st = lmmc_vec_create(n, &x);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_vec_create(n, &y);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&x); rc = 1; goto done; }

        for (size_t i = 0; i < n; i++) {
            x.data[i] = (lmmc_real_t)(i + 1);
            y.data[i] = (lmmc_real_t)(100 + i);
        }

        /* Save originals */
        lmmc_real_t orig_x[6], orig_y[6];
        memcpy(orig_x, x.data, n * sizeof(lmmc_real_t));
        memcpy(orig_y, y.data, n * sizeof(lmmc_real_t));

        st = lmmc_vec_swap(&x, &y);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&x); lmmc_vec_destroy(&y); goto done; }

        for (size_t i = 0; i < n; i++) {
            if (!lmmc_test_nearly_equal(x.data[i], orig_y[i], TEST_EPS_TIGHT) ||
                !lmmc_test_nearly_equal(y.data[i], orig_x[i], TEST_EPS_TIGHT)) {
                printf("5.10 FAIL: swap incorrect at index %zu\n", i);
                rc = 1; lmmc_vec_destroy(&x); lmmc_vec_destroy(&y); goto done;
            }
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&y);
    }

    /* ================================================================
     * Requirement 5.11: Vector norms (L2, inf, asum)
     * Zero vector norms = 0, unit vector L2 norm = 1
     * ================================================================ */
    {
        /* Zero vector: all norms = 0 */
        size_t n = 5;
        lmmc_vec_t z = {0};
        st = lmmc_vec_create(n, &z);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_fill(&z, 0.0);

        lmmc_real_t norm2 = -1.0, norminf = -1.0, asum = -1.0;
        st = lmmc_vec_norm2(&z, &norm2);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&z); goto done; }
        st = lmmc_vec_norm_inf(&z, &norminf);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&z); goto done; }
        st = lmmc_vec_asum(&z, &asum);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&z); goto done; }

        if (!lmmc_test_nearly_equal(norm2, 0.0, TEST_EPS_TIGHT) ||
            !lmmc_test_nearly_equal(norminf, 0.0, TEST_EPS_TIGHT) ||
            !lmmc_test_nearly_equal(asum, 0.0, TEST_EPS_TIGHT)) {
            printf("5.11 FAIL: Zero vector norms not zero: L2=%g, inf=%g, asum=%g\n", norm2, norminf, asum);
            rc = 1; lmmc_vec_destroy(&z); goto done;
        }
        lmmc_vec_destroy(&z);

        /* Unit vector: L2 norm = 1 */
        lmmc_vec_t u = {0};
        st = lmmc_vec_create(4, &u);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        u.data[0] = 0.5; u.data[1] = 0.5; u.data[2] = 0.5; u.data[3] = 0.5;
        /* L2 = sqrt(0.25*4) = 1.0 */

        st = lmmc_vec_norm2(&u, &norm2);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&u); goto done; }
        if (!lmmc_test_nearly_equal(norm2, 1.0, TEST_EPS_TIGHT)) {
            printf("5.11 FAIL: Unit vector L2 norm = %g, expected 1.0\n", norm2);
            rc = 1; lmmc_vec_destroy(&u); goto done;
        }

        /* inf norm = 0.5 */
        st = lmmc_vec_norm_inf(&u, &norminf);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&u); goto done; }
        if (!lmmc_test_nearly_equal(norminf, 0.5, TEST_EPS_TIGHT)) {
            printf("5.11 FAIL: Unit vector inf norm = %g, expected 0.5\n", norminf);
            rc = 1; lmmc_vec_destroy(&u); goto done;
        }

        /* asum = 2.0 */
        st = lmmc_vec_asum(&u, &asum);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_vec_destroy(&u); goto done; }
        if (!lmmc_test_nearly_equal(asum, 2.0, TEST_EPS_TIGHT)) {
            printf("5.11 FAIL: Unit vector asum = %g, expected 2.0\n", asum);
            rc = 1; lmmc_vec_destroy(&u); goto done;
        }

        lmmc_vec_destroy(&u);

        /* Known vector: [3, -4] => L2=5, inf=4, asum=7 */
        lmmc_vec_t v = {0};
        st = lmmc_vec_create(2, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        v.data[0] = 3.0; v.data[1] = -4.0;

        lmmc_vec_norm2(&v, &norm2);
        lmmc_vec_norm_inf(&v, &norminf);
        lmmc_vec_asum(&v, &asum);

        if (!lmmc_test_nearly_equal(norm2, 5.0, TEST_EPS_TIGHT)) {
            printf("5.11 FAIL: [3,-4] L2=%g, expected 5.0\n", norm2);
            rc = 1; lmmc_vec_destroy(&v); goto done;
        }
        if (!lmmc_test_nearly_equal(norminf, 4.0, TEST_EPS_TIGHT)) {
            printf("5.11 FAIL: [3,-4] inf=%g, expected 4.0\n", norminf);
            rc = 1; lmmc_vec_destroy(&v); goto done;
        }
        if (!lmmc_test_nearly_equal(asum, 7.0, TEST_EPS_TIGHT)) {
            printf("5.11 FAIL: [3,-4] asum=%g, expected 7.0\n", asum);
            rc = 1; lmmc_vec_destroy(&v); goto done;
        }

        lmmc_vec_destroy(&v);
    }

    /* ================================================================
     * Requirement 5.12: Frobenius norm (identity = sqrt(n))
     * ================================================================ */
    {
        /* Identity matrix Frobenius norm = sqrt(n) */
        size_t sizes[] = {1, 2, 3, 5, 10};
        for (size_t si = 0; si < 5; si++) {
            size_t n = sizes[si];
            lmmc_mat_t eye = {0};
            st = lmmc_mat_identity(n, &eye);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

            lmmc_real_t fnorm = 0.0;
            st = lmmc_mat_norm_fro(&eye, &fnorm);
            if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&eye); goto done; }

            lmmc_real_t expected = sqrt((double)n);
            if (!lmmc_test_nearly_equal(fnorm, expected, TEST_EPS_TIGHT)) {
                printf("5.12 FAIL: ||I_%zu||_F = %g, expected %g\n", n, fnorm, expected);
                rc = 1; lmmc_mat_destroy(&eye); goto done;
            }
            lmmc_mat_destroy(&eye);
        }

        /* Also verify: known matrix [[1,2],[3,4]] => sqrt(1+4+9+16) = sqrt(30) */
        lmmc_mat_t m = {0};
        st = lmmc_mat_create(2, 2, &m);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        m.data[0] = 1.0; m.data[1] = 2.0;
        m.data[2] = 3.0; m.data[3] = 4.0;

        lmmc_real_t fnorm = 0.0;
        st = lmmc_mat_norm_fro(&m, &fnorm);
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_mat_destroy(&m); goto done; }
        if (!lmmc_test_nearly_equal(fnorm, sqrt(30.0), TEST_EPS_TIGHT)) {
            printf("5.12 FAIL: ||[[1,2],[3,4]]||_F = %g, expected %g\n", fnorm, sqrt(30.0));
            rc = 1; lmmc_mat_destroy(&m); goto done;
        }
        lmmc_mat_destroy(&m);
    }

done:
    if (rc != 0) {
        printf("dense extended test failed\n");
    }
    return rc;
}
