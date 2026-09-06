/**
 * @file test_interp_linear.c
 * 针对 LMMC 中 interp linear 相关接口的单元测试。
 */
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t out_y = 0.0;
    int rc = 0;


    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        lmmc_real_t ys[] = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0};
        size_t n = 6;
        lmmc_real_t queries[] = {0.5, 1.5, 2.5, 3.5, 4.5};
        size_t nq = 5;

        for (size_t i = 0; i < nq; i++) {
            lmmc_real_t expected = 2.0 * queries[i] + 1.0;
            st = lmmc_interp_linear(xs, ys, n, queries[i], &out_y);
            if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, expected, 1e-14)) {
                printf("interp_linear test failed: linear function at x=%f, got %f expected %f\n",
                       queries[i], out_y, expected);
                rc = 1;
                goto done;
            }
        }
    }


    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
        lmmc_real_t ys[] = {10.0, 20.0, 30.0, 40.0, 50.0};
        size_t n = 5;

        for (size_t i = 0; i < n; i++) {
            st = lmmc_interp_linear(xs, ys, n, xs[i], &out_y);
            if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, ys[i], 1e-14)) {
                printf("interp_linear test failed: node exact at i=%zu, got %f expected %f\n",
                       i, out_y, ys[i]);
                rc = 1;
                goto done;
            }
        }
    }


    {
        lmmc_real_t xs[] = {1.0, 3.0};
        lmmc_real_t ys[] = {5.0, 11.0};
        size_t n = 2;
        lmmc_real_t query = 2.0;
        lmmc_real_t expected = 8.0;

        st = lmmc_interp_linear(xs, ys, n, query, &out_y);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, expected, 1e-14)) {
            printf("interp_linear test failed: two-point, got %f expected %f\n", out_y, expected);
            rc = 1;
            goto done;
        }


        st = lmmc_interp_linear(xs, ys, n, 1.0, &out_y);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, 5.0, 1e-14)) {
            printf("interp_linear test failed: two-point left endpoint\n");
            rc = 1;
            goto done;
        }

        st = lmmc_interp_linear(xs, ys, n, 3.0, &out_y);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, 11.0, 1e-14)) {
            printf("interp_linear test failed: two-point right endpoint\n");
            rc = 1;
            goto done;
        }
    }

    {
        const lmmc_real_t xs[] = {-DBL_MAX, DBL_MAX};
        const lmmc_real_t ys[] = {-DBL_MAX, DBL_MAX};

        st = lmmc_interp_linear(xs, ys, 2, 0.0, &out_y);
        if (st != LMMC_STATUS_OK || out_y != 0.0) {
            printf("interp_linear test failed: finite extreme midpoint got %.17g\n",
                   out_y);
            rc = 1;
            goto done;
        }
    }


    {
        lmmc_real_t xs[100];
        lmmc_real_t ys[100];
        size_t n = 100;


        for (size_t i = 0; i < n; i++) {
            xs[i] = (lmmc_real_t)i;
            ys[i] = 2.0 * xs[i] + 1.0;
        }


        for (size_t i = 0; i < n - 1; i++) {
            lmmc_real_t qx = xs[i] + 0.5;
            lmmc_real_t expected = 2.0 * qx + 1.0;
            st = lmmc_interp_linear(xs, ys, n, qx, &out_y);
            if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, expected, 1e-12)) {
                printf("interp_linear test failed: 100-node at x=%f, got %f expected %f\n",
                       qx, out_y, expected);
                rc = 1;
                goto done;
            }
        }


        for (size_t i = 0; i < n; i++) {
            xs[i] = (lmmc_real_t)i / (lmmc_real_t)(n - 1);
            ys[i] = xs[i] * xs[i];
        }


        for (size_t i = 0; i < n - 1; i++) {
            lmmc_real_t qx = (xs[i] + xs[i + 1]) / 2.0;
            lmmc_real_t interp_expected = (ys[i] + ys[i + 1]) / 2.0;
            st = lmmc_interp_linear(xs, ys, n, qx, &out_y);
            if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, interp_expected, 1e-12)) {
                printf("interp_linear test failed: 100-node x^2 at x=%f\n", qx);
                rc = 1;
                goto done;
            }
        }
    }


    {
        lmmc_real_t xs[] = {1.0};
        lmmc_real_t ys[] = {2.0};

        st = lmmc_interp_linear(xs, ys, 1, 1.0, &out_y);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("interp_linear test failed: n=1 should return INVALID_ARGUMENT, got %d\n", (int)st);
            rc = 1;
            goto done;
        }

        st = lmmc_interp_linear(xs, ys, 0, 1.0, &out_y);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("interp_linear test failed: n=0 should return INVALID_ARGUMENT, got %d\n", (int)st);
            rc = 1;
            goto done;
        }
    }


    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0};
        lmmc_real_t ys[] = {0.0, 1.0, 2.0};

        st = lmmc_interp_linear(NULL, ys, 3, 1.0, &out_y);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("interp_linear test failed: NULL xs should return INVALID_ARGUMENT\n");
            rc = 1;
            goto done;
        }

        st = lmmc_interp_linear(xs, NULL, 3, 1.0, &out_y);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("interp_linear test failed: NULL ys should return INVALID_ARGUMENT\n");
            rc = 1;
            goto done;
        }

        st = lmmc_interp_linear(xs, ys, 3, 1.0, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("interp_linear test failed: NULL out_y should return INVALID_ARGUMENT\n");
            rc = 1;
            goto done;
        }
    }


    {
        lmmc_real_t xs[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        lmmc_real_t ys[] = {2.0, 4.0, 6.0, 8.0, 10.0};


        st = lmmc_interp_linear(xs, ys, 5, 0.5, &out_y);
        if (st != LMMC_STATUS_OUT_OF_RANGE) {
            printf("interp_linear test failed: below range should return OUT_OF_RANGE, got %d\n", (int)st);
            rc = 1;
            goto done;
        }


        st = lmmc_interp_linear(xs, ys, 5, 5.5, &out_y);
        if (st != LMMC_STATUS_OUT_OF_RANGE) {
            printf("interp_linear test failed: above range should return OUT_OF_RANGE, got %d\n", (int)st);
            rc = 1;
            goto done;
        }
    }


    {
        lmmc_real_t xs_dup[] = {1.0, 2.0, 2.0, 3.0};
        lmmc_real_t ys_dup[] = {1.0, 2.0, 3.0, 4.0};
        lmmc_real_t xs_dec[] = {3.0, 2.0, 1.0};
        lmmc_real_t ys_dec[] = {6.0, 4.0, 2.0};
        lmmc_real_t xs_valid[] = {1.0, 2.0, 3.0};
        lmmc_real_t xs_nan[] = {1.0, NAN, 3.0};
        lmmc_real_t ys_nan[] = {1.0, NAN, 3.0};

        st = lmmc_interp_linear(xs_dup, ys_dup, 4, 2.5, &out_y);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("interp_linear test failed: duplicate xs should return INVALID_ARGUMENT\n");
            rc = 1;
            goto done;
        }

        st = lmmc_interp_linear(xs_dec, ys_dec, 3, 2.0, &out_y);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("interp_linear test failed: decreasing xs should return INVALID_ARGUMENT\n");
            rc = 1;
            goto done;
        }

        st = lmmc_interp_linear(xs_nan, ys_dup, 3, 2.0, &out_y);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("interp_linear test failed: NaN xs should return INVALID_ARGUMENT\n");
            rc = 1;
            goto done;
        }

        st = lmmc_interp_linear(xs_valid, ys_nan, 3, 2.0, &out_y);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("interp_linear test failed: NaN ys should return INVALID_ARGUMENT\n");
            rc = 1;
            goto done;
        }

        st = lmmc_interp_linear(xs_dup, ys_dup, 4, NAN, &out_y);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("interp_linear test failed: NaN query should return INVALID_ARGUMENT\n");
            rc = 1;
            goto done;
        }
    }


    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
        lmmc_real_t ys[] = {0.0, 10.0, 4.0, 7.0};


        st = lmmc_interp_linear(xs, ys, 4, 0.3, &out_y);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, 3.0, 1e-14)) {
            printf("interp_linear test failed: linear combination at x=0.3, got %f expected 3.0\n", out_y);
            rc = 1;
            goto done;
        }


        st = lmmc_interp_linear(xs, ys, 4, 1.5, &out_y);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, 7.0, 1e-14)) {
            printf("interp_linear test failed: linear combination at x=1.5, got %f expected 7.0\n", out_y);
            rc = 1;
            goto done;
        }


        st = lmmc_interp_linear(xs, ys, 4, 2.7, &out_y);
        if (st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(out_y, 6.1, 1e-14)) {
            printf("interp_linear test failed: linear combination at x=2.7, got %f expected 6.1\n", out_y);
            rc = 1;
            goto done;
        }
    }

done:
    if (rc != 0) {
        printf("interp_linear test failed\n");
    }
    return rc;
}
