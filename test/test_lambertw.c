/**
 * @file test_lambertw.c
 * @brief Unit tests for Lambert W₀ domain fix and W₋₁ branch.
 *
 * Validates Requirements 2.1–2.6.
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"

static int nearly_equal(double a, double b, double tol) {
    return fabs(a - b) <= tol * (1.0 + fabs(b));
}

int main(void) {
    lmmc_status_t st;
    lmmc_real_t res = 0.0;
    int rc = 0;

    /* --- W₀ basic tests --- */

    /* W₀(0) = 0 */
    st = lmmc_lambertw(0.0, &res);
    if (st != LMMC_STATUS_OK || res != 0.0) {
        printf("FAIL: W0(0) expected 0, got %g (st=%d)\n", res, st);
        rc = 1;
    }

    /* W₀(e) = 1 */
    st = lmmc_lambertw(exp(1.0), &res);
    if (st != LMMC_STATUS_OK || !nearly_equal(res, 1.0, 1e-10)) {
        printf("FAIL: W0(e) expected 1, got %.17g (st=%d)\n", res, st);
        rc = 1;
    }

    /* W₀(1) ≈ 0.5671432904097838 */
    st = lmmc_lambertw(1.0, &res);
    if (st != LMMC_STATUS_OK || !nearly_equal(res, 0.5671432904097838, 1e-10)) {
        printf("FAIL: W0(1) expected 0.5671..., got %.17g (st=%d)\n", res, st);
        rc = 1;
    }

    /* W₀(-1/e) = -1 (branch point) */
    st = lmmc_lambertw(-LMMC_INV_E, &res);
    if (st != LMMC_STATUS_OK || !nearly_equal(res, -1.0, 1e-10)) {
        printf("FAIL: W0(-1/e) expected -1, got %.17g (st=%d)\n", res, st);
        rc = 1;
    }

    /* W₀ near branch point: z = -0.36 (valid, > -1/e) */
    st = lmmc_lambertw(-0.36, &res);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL: W0(-0.36) should succeed (st=%d)\n", st);
        rc = 1;
    } else {
        /* Verify identity: W(z)*exp(W(z)) == z */
        double check = res * exp(res);
        if (!nearly_equal(check, -0.36, 1e-10)) {
            printf("FAIL: W0(-0.36) identity check: w*exp(w)=%.17g, expected -0.36\n", check);
            rc = 1;
        }
    }

    /* W₀ out of domain: z < -1/e */
    st = lmmc_lambertw(-0.5, &res);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: W0(-0.5) should return INVALID_ARGUMENT, got st=%d\n", st);
        rc = 1;
    }

    st = lmmc_lambertw(-1.0, &res);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: W0(-1.0) should return INVALID_ARGUMENT, got st=%d\n", st);
        rc = 1;
    }

    /* W₀ NULL output */
    st = lmmc_lambertw(1.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: W0 with NULL should return INVALID_ARGUMENT, got st=%d\n", st);
        rc = 1;
    }

    /* --- W₋₁ tests --- */

    /* W₋₁(-1/e) = -1 (branch point) */
    st = lmmc_lambertw_wm1(-LMMC_INV_E, &res);
    if (st != LMMC_STATUS_OK || !nearly_equal(res, -1.0, 1e-10)) {
        printf("FAIL: Wm1(-1/e) expected -1, got %.17g (st=%d)\n", res, st);
        rc = 1;
    }

    /* W₋₁(-0.1): verify identity */
    st = lmmc_lambertw_wm1(-0.1, &res);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL: Wm1(-0.1) should succeed (st=%d)\n", st);
        rc = 1;
    } else {
        double check = res * exp(res);
        if (!nearly_equal(check, -0.1, 1e-10)) {
            printf("FAIL: Wm1(-0.1) identity: w*exp(w)=%.17g, expected -0.1\n", check);
            rc = 1;
        }
        /* W₋₁ should be <= -1 */
        if (res > -1.0) {
            printf("FAIL: Wm1(-0.1) should be <= -1, got %.17g\n", res);
            rc = 1;
        }
    }

    /* W₋₁(-0.01): verify identity */
    st = lmmc_lambertw_wm1(-0.01, &res);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL: Wm1(-0.01) should succeed (st=%d)\n", st);
        rc = 1;
    } else {
        double check = res * exp(res);
        if (!nearly_equal(check, -0.01, 1e-10)) {
            printf("FAIL: Wm1(-0.01) identity: w*exp(w)=%.17g, expected -0.01\n", check);
            rc = 1;
        }
        if (res > -1.0) {
            printf("FAIL: Wm1(-0.01) should be <= -1, got %.17g\n", res);
            rc = 1;
        }
    }

    /* W₋₁ out of domain: z >= 0 */
    st = lmmc_lambertw_wm1(0.0, &res);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: Wm1(0) should return INVALID_ARGUMENT, got st=%d\n", st);
        rc = 1;
    }

    st = lmmc_lambertw_wm1(1.0, &res);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: Wm1(1) should return INVALID_ARGUMENT, got st=%d\n", st);
        rc = 1;
    }

    /* W₋₁ out of domain: z < -1/e */
    st = lmmc_lambertw_wm1(-0.5, &res);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: Wm1(-0.5) should return INVALID_ARGUMENT, got st=%d\n", st);
        rc = 1;
    }

    /* W₋₁ NULL output */
    st = lmmc_lambertw_wm1(-0.1, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: Wm1 with NULL should return INVALID_ARGUMENT, got st=%d\n", st);
        rc = 1;
    }

    /* --- Convergence criterion test --- */
    /* Verify the identity |w*exp(w) - z| <= tol * (1 + |z|) for various inputs */
    {
        double test_vals[] = {0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0,
                              -0.1, -0.2, -0.3, -0.35, -0.367};
        size_t n = sizeof(test_vals) / sizeof(test_vals[0]);
        for (size_t i = 0; i < n; i++) {
            double z = test_vals[i];
            double w = 0.0;
            st = lmmc_lambertw(z, &w);
            if (st != LMMC_STATUS_OK) {
                printf("FAIL: W0(%g) returned st=%d\n", z, st);
                rc = 1;
                continue;
            }
            double residual = fabs(w * exp(w) - z);
            double threshold = 1e-10 * (1.0 + fabs(z));
            if (residual > threshold) {
                printf("FAIL: W0(%g) residual %.17g > threshold %.17g\n", z, residual, threshold);
                rc = 1;
            }
        }
    }

    /* W₋₁ convergence criterion */
    {
        double test_vals[] = {-0.001, -0.01, -0.05, -0.1, -0.2, -0.3, -0.35, -0.367};
        size_t n = sizeof(test_vals) / sizeof(test_vals[0]);
        for (size_t i = 0; i < n; i++) {
            double z = test_vals[i];
            double w = 0.0;
            st = lmmc_lambertw_wm1(z, &w);
            if (st != LMMC_STATUS_OK) {
                printf("FAIL: Wm1(%g) returned st=%d\n", z, st);
                rc = 1;
                continue;
            }
            double residual = fabs(w * exp(w) - z);
            double threshold = 1e-10 * (1.0 + fabs(z));
            if (residual > threshold) {
                printf("FAIL: Wm1(%g) residual %.17g > threshold %.17g\n", z, residual, threshold);
                rc = 1;
            }
            if (w > -1.0) {
                printf("FAIL: Wm1(%g) = %g should be <= -1\n", z, w);
                rc = 1;
            }
        }
    }

    if (rc == 0) {
        printf("All Lambert W tests passed.\n");
    }
    return rc;
}
