/**
 * @file test_complex_transcendental.c
 * @brief 复数超越函数单元测试：exp, log, sqrt, sin, cos, pow。
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int nearly_equal(double a, double b, double tol) {
    return fabs(a - b) <= tol * (1.0 + fabs(b));
}

int main(void)
{
    lmmc_status_t st;
    lmmc_complex_t z, result;
    int rc = 0;

    /* exp(0+0i) = 1+0i */
    lmmc_complex_create(0.0, 0.0, &z);
    st = lmmc_complex_exp(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: exp(0+0i) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* exp(1+0i) = e+0i */
    lmmc_complex_create(1.0, 0.0, &z);
    st = lmmc_complex_exp(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, exp(1.0), 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: exp(1+0i) expected e+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* exp(0+pi*i) = -1+0i (Euler's identity) */
    lmmc_complex_create(0.0, M_PI, &z);
    st = lmmc_complex_exp(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, -1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: exp(i*pi) expected -1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* exp(NULL) -> INVALID_ARGUMENT */
    st = lmmc_complex_exp(NULL, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: exp(NULL) expected INVALID_ARGUMENT, got %d\n", st);
        rc = 1;
    }

    /* log(1+0i) = 0+0i */
    lmmc_complex_create(1.0, 0.0, &z);
    st = lmmc_complex_log(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 0.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: log(1+0i) expected 0+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* log(e+0i) = 1+0i */
    lmmc_complex_create(exp(1.0), 0.0, &z);
    st = lmmc_complex_log(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: log(e+0i) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* log(-1+0i) = 0+pi*i */
    lmmc_complex_create(-1.0, 0.0, &z);
    st = lmmc_complex_log(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 0.0, 1e-12) || !nearly_equal(result.imag, M_PI, 1e-12)) {
        printf("FAIL: log(-1+0i) expected 0+pi*i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* log(0+0i) -> OUT_OF_RANGE */
    lmmc_complex_create(0.0, 0.0, &z);
    st = lmmc_complex_log(&z, &result);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        printf("FAIL: log(0+0i) expected OUT_OF_RANGE, got %d\n", st);
        rc = 1;
    }

    /* log(NULL) -> INVALID_ARGUMENT */
    st = lmmc_complex_log(NULL, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("FAIL: log(NULL) expected INVALID_ARGUMENT, got %d\n", st);
        rc = 1;
    }

    /* sqrt(1+0i) = 1+0i */
    lmmc_complex_create(1.0, 0.0, &z);
    st = lmmc_complex_sqrt(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: sqrt(1+0i) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* sqrt(4+0i) = 2+0i */
    lmmc_complex_create(4.0, 0.0, &z);
    st = lmmc_complex_sqrt(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 2.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: sqrt(4+0i) expected 2+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* sqrt(-1+0i) = 0+1i */
    lmmc_complex_create(-1.0, 0.0, &z);
    st = lmmc_complex_sqrt(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 0.0, 1e-12) || !nearly_equal(result.imag, 1.0, 1e-12)) {
        printf("FAIL: sqrt(-1+0i) expected 0+1i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* sqrt(0+0i) = 0+0i */
    lmmc_complex_create(0.0, 0.0, &z);
    st = lmmc_complex_sqrt(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 0.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: sqrt(0+0i) expected 0+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* sin(0+0i) = 0+0i */
    lmmc_complex_create(0.0, 0.0, &z);
    st = lmmc_complex_sin(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 0.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: sin(0+0i) expected 0+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* sin(pi/2 + 0i) = 1+0i */
    lmmc_complex_create(M_PI / 2.0, 0.0, &z);
    st = lmmc_complex_sin(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: sin(pi/2+0i) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* cos(0+0i) = 1+0i */
    lmmc_complex_create(0.0, 0.0, &z);
    st = lmmc_complex_cos(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: cos(0+0i) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* cos(pi+0i) = -1+0i */
    lmmc_complex_create(M_PI, 0.0, &z);
    st = lmmc_complex_cos(&z, &result);
    if (st != LMMC_STATUS_OK || !nearly_equal(result.real, -1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
        printf("FAIL: cos(pi+0i) expected -1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
        rc = 1;
    }

    /* sin^2(z) + cos^2(z) = 1 */
    {
        lmmc_complex_t sin_z, cos_z, sin2, cos2, sum;
        lmmc_complex_create(1.5, 2.3, &z);
        st = lmmc_complex_sin(&z, &sin_z);
        if (st != LMMC_STATUS_OK) {
            printf("FAIL: sin(1.5+2.3i) returned st=%d\n", st);
            rc = 1;
        }
        st = lmmc_complex_cos(&z, &cos_z);
        if (st != LMMC_STATUS_OK) {
            printf("FAIL: cos(1.5+2.3i) returned st=%d\n", st);
            rc = 1;
        }
        st = lmmc_complex_mul(&sin_z, &sin_z, &sin2);
        if (st != LMMC_STATUS_OK) {
            printf("FAIL: sin^2 mul returned st=%d\n", st);
            rc = 1;
        }
        st = lmmc_complex_mul(&cos_z, &cos_z, &cos2);
        if (st != LMMC_STATUS_OK) {
            printf("FAIL: cos^2 mul returned st=%d\n", st);
            rc = 1;
        }
        st = lmmc_complex_add(&sin2, &cos2, &sum);
        if (st != LMMC_STATUS_OK) {
            printf("FAIL: sin^2+cos^2 add returned st=%d\n", st);
            rc = 1;
        }
        if (!nearly_equal(sum.real, 1.0, 1e-10) || !nearly_equal(sum.imag, 0.0, 1e-10)) {
            printf("FAIL: sin^2(1.5+2.3i)+cos^2(1.5+2.3i) expected 1+0i, got %g+%gi\n", sum.real, sum.imag);
            rc = 1;
        }
    }

    /* pow(2+0i, 3+0i) = 8+0i */
    {
        lmmc_complex_t base, exponent;
        lmmc_complex_create(2.0, 0.0, &base);
        lmmc_complex_create(3.0, 0.0, &exponent);
        st = lmmc_complex_pow(&base, &exponent, &result);
        if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 8.0, 1e-10) || !nearly_equal(result.imag, 0.0, 1e-10)) {
            printf("FAIL: pow(2,3) expected 8+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
            rc = 1;
        }
    }

    /* pow(0+0i, -1+0i) -> OUT_OF_RANGE */
    {
        lmmc_complex_t base, exponent;
        lmmc_complex_create(0.0, 0.0, &base);
        lmmc_complex_create(-1.0, 0.0, &exponent);
        st = lmmc_complex_pow(&base, &exponent, &result);
        if (st != LMMC_STATUS_OUT_OF_RANGE) {
            printf("FAIL: pow(0, -1) expected OUT_OF_RANGE, got %d\n", st);
            rc = 1;
        }
    }

    /* pow(0+0i, 2+0i) = 0+0i */
    {
        lmmc_complex_t base, exponent;
        lmmc_complex_create(0.0, 0.0, &base);
        lmmc_complex_create(2.0, 0.0, &exponent);
        st = lmmc_complex_pow(&base, &exponent, &result);
        if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 0.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
            printf("FAIL: pow(0,2) expected 0+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
            rc = 1;
        }
    }

    /* pow(0+0i, 0+0i) = 1+0i (cpow convention) */
    {
        lmmc_complex_t base, exponent;
        lmmc_complex_create(0.0, 0.0, &base);
        lmmc_complex_create(0.0, 0.0, &exponent);
        st = lmmc_complex_pow(&base, &exponent, &result);
        if (st != LMMC_STATUS_OK || !nearly_equal(result.real, 1.0, 1e-12) || !nearly_equal(result.imag, 0.0, 1e-12)) {
            printf("FAIL: pow(0,0) expected 1+0i, got %g+%gi (st=%d)\n", result.real, result.imag, st);
            rc = 1;
        }
    }

    /* pow(NULL, ...) -> INVALID_ARGUMENT */
    {
        lmmc_complex_t exponent;
        lmmc_complex_create(1.0, 0.0, &exponent);
        st = lmmc_complex_pow(NULL, &exponent, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("FAIL: pow(NULL,...) expected INVALID_ARGUMENT, got %d\n", st);
            rc = 1;
        }
    }

    /* exp(log(z)) ≈ z */
    {
        lmmc_complex_t log_z, exp_log_z;
        lmmc_complex_create(3.0, 4.0, &z);
        st = lmmc_complex_log(&z, &log_z);
        if (st != LMMC_STATUS_OK) {
            printf("FAIL: log(3+4i) returned st=%d\n", st);
            rc = 1;
        }
        st = lmmc_complex_exp(&log_z, &exp_log_z);
        if (st != LMMC_STATUS_OK) {
            printf("FAIL: exp(log(3+4i)) returned st=%d\n", st);
            rc = 1;
        }
        if (!nearly_equal(exp_log_z.real, 3.0, 1e-10) || !nearly_equal(exp_log_z.imag, 4.0, 1e-10)) {
            printf("FAIL: exp(log(3+4i)) expected 3+4i, got %g+%gi\n", exp_log_z.real, exp_log_z.imag);
            rc = 1;
        }
    }

    if (rc == 0) {
        printf("All complex transcendental tests passed.\n");
    }
    return rc;
}
