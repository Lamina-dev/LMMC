/**
 * @file test_special_functions.c
 * 针对 LMMC 特殊函数（erf, erfc, lgamma, tgamma, beta, digamma）的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include <float.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    int rc = 0;
    lmmc_status_t st = LMMC_STATUS_OK;
    int test_section = 0;

    /* ===== Section 1: erf 基本值测试 ===== */
    test_section = 1;
    {
        lmmc_real_t res = 0.0;

        /* erf(0) = 0 */
        st = lmmc_erf(0.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res) > 1e-15) { rc = 1; goto done; }

        /* erf(1) ≈ 0.8427007929497149 */
        st = lmmc_erf(1.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 0.8427007929497149) > 1e-12) { rc = 1; goto done; }

        /* erf(-1) = -erf(1) (奇函数) */
        st = lmmc_erf(-1.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res + 0.8427007929497149) > 1e-12) { rc = 1; goto done; }

        /* erf(0.5) ≈ 0.5204998778130465 */
        st = lmmc_erf(0.5, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 0.5204998778130465) > 1e-12) { rc = 1; goto done; }

        /* erf(2) ≈ 0.9953222650189527 */
        st = lmmc_erf(2.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 0.9953222650189527) > 1e-12) { rc = 1; goto done; }

        /* erf(5) ≈ 1.0 (接近 1) */
        st = lmmc_erf(5.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 1.0) > 1e-10) { rc = 1; goto done; }

        /* NULL 检查 */
        st = lmmc_erf(1.0, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ===== Section 2: erfc 基本值测试 ===== */
    test_section = 2;
    {
        lmmc_real_t res = 0.0;

        /* erfc(0) = 1 */
        st = lmmc_erfc(0.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 1.0) > 1e-15) { rc = 1; goto done; }

        /* erfc(1) ≈ 0.1572992070502851 */
        st = lmmc_erfc(1.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 0.1572992070502851) > 1e-12) { rc = 1; goto done; }

        /* erfc(2) ≈ 0.004677734981047266 */
        st = lmmc_erfc(2.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 0.004677734981047266) > 1e-12) { rc = 1; goto done; }

        /* erfc(-1) = 2 - erfc(1) ≈ 1.8427007929497149 */
        st = lmmc_erfc(-1.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 1.8427007929497149) > 1e-12) { rc = 1; goto done; }

        /* erfc(大值) → 0 */
        st = lmmc_erfc(10.0, &res);
        if (st != LMMC_STATUS_OK || res < 0.0 || res > 1e-40) { rc = 1; goto done; }

        /* erf + erfc = 1 一致性检查 */
        {
            double test_x[] = {0.1, 0.3, 0.7, 1.5, 3.0};
            for (int i = 0; i < 5; i++) {
                lmmc_real_t erf_val, erfc_val;
                st = lmmc_erf(test_x[i], &erf_val);
                if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
                st = lmmc_erfc(test_x[i], &erfc_val);
                if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
                if (fabs(erf_val + erfc_val - 1.0) > 1e-12) { rc = 1; goto done; }
            }
        }
    }

    /* ===== Section 3: erf 与 C 标准库 erf 对比 ===== */
    test_section = 3;
    {
        double test_x[] = {-5.0, -3.0, -2.0, -1.0, -0.5, -0.1, 0.0,
                           0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0};
        int n = sizeof(test_x) / sizeof(test_x[0]);
        for (int i = 0; i < n; i++) {
            lmmc_real_t res;
            st = lmmc_erf(test_x[i], &res);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            double ref = erf(test_x[i]);
            if (fabs(res - ref) > 1e-12) {
                printf("erf(%g): got %.17g, expected %.17g, diff=%.3e\n",
                       test_x[i], res, ref, fabs(res - ref));
                rc = 1; goto done;
            }
        }
    }

    /* ===== Section 4: lgamma 基本值测试 ===== */
    test_section = 4;
    {
        lmmc_real_t res = 0.0;

        /* lgamma(1) = 0 */
        st = lmmc_lgamma(1.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res) > 1e-12) { rc = 1; goto done; }

        /* lgamma(2) = 0 (因为 Gamma(2) = 1! = 1) */
        st = lmmc_lgamma(2.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res) > 1e-12) { rc = 1; goto done; }

        /* lgamma(0.5) = ln(sqrt(pi)) ≈ 0.5723649429247001 */
        st = lmmc_lgamma(0.5, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 0.5723649429247001) > 1e-10) { rc = 1; goto done; }

        /* lgamma(5) = ln(4!) = ln(24) ≈ 3.1780538303479458 */
        st = lmmc_lgamma(5.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - log(24.0)) > 1e-10) { rc = 1; goto done; }

        /* lgamma(10) = ln(9!) = ln(362880) */
        st = lmmc_lgamma(10.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - log(362880.0)) > 1e-10) { rc = 1; goto done; }

        /* 无效参数 */
        st = lmmc_lgamma(0.0, &res);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_lgamma(-1.0, &res);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_lgamma(1.0, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ===== Section 5: tgamma 基本值测试 ===== */
    test_section = 5;
    {
        lmmc_real_t res = 0.0;

        /* Gamma(1) = 1 */
        st = lmmc_tgamma(1.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 1.0) > 1e-12) { rc = 1; goto done; }

        /* Gamma(2) = 1 */
        st = lmmc_tgamma(2.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 1.0) > 1e-12) { rc = 1; goto done; }

        /* Gamma(5) = 4! = 24 */
        st = lmmc_tgamma(5.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 24.0) > 1e-10) { rc = 1; goto done; }

        /* Gamma(0.5) = sqrt(pi) ≈ 1.7724538509055159 */
        st = lmmc_tgamma(0.5, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 1.7724538509055159) > 1e-10) { rc = 1; goto done; }

        /* Gamma(10) = 9! = 362880 */
        st = lmmc_tgamma(10.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 362880.0) > 1e-5) { rc = 1; goto done; }

        /* 无效参数 */
        st = lmmc_tgamma(0.0, &res);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_tgamma(-1.0, &res);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ===== Section 6: beta 函数测试 ===== */
    test_section = 6;
    {
        lmmc_real_t res = 0.0;

        /* B(1,1) = 1 */
        st = lmmc_beta(1.0, 1.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 1.0) > 1e-12) { rc = 1; goto done; }

        /* B(1,2) = 1/2 */
        st = lmmc_beta(1.0, 2.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 0.5) > 1e-12) { rc = 1; goto done; }

        /* B(2,2) = 1/6 */
        st = lmmc_beta(2.0, 2.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 1.0/6.0) > 1e-12) { rc = 1; goto done; }

        /* B(0.5, 0.5) = pi */
        st = lmmc_beta(0.5, 0.5, &res);
        if (st != LMMC_STATUS_OK || fabs(res - LMMC_PI) > 1e-10) { rc = 1; goto done; }

        /* B(3, 4) = 2!*3! / 6! = 2*6/720 = 1/60 */
        st = lmmc_beta(3.0, 4.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 1.0/60.0) > 1e-12) { rc = 1; goto done; }

        /* 对称性: B(a,b) = B(b,a) */
        {
            lmmc_real_t r1, r2;
            st = lmmc_beta(2.5, 3.7, &r1);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            st = lmmc_beta(3.7, 2.5, &r2);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            if (fabs(r1 - r2) > 1e-14) { rc = 1; goto done; }
        }

        /* 无效参数 */
        st = lmmc_beta(0.0, 1.0, &res);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_beta(1.0, -1.0, &res);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_beta(1.0, 1.0, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ===== Section 7: digamma 函数测试 ===== */
    test_section = 7;
    {
        lmmc_real_t res = 0.0;

        /* psi(1) = -gamma (Euler-Mascheroni) ≈ -0.5772156649015329 */
        st = lmmc_digamma(1.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - (-0.5772156649015329)) > 1e-10) { rc = 1; goto done; }

        /* psi(2) = 1 - gamma ≈ 0.4227843350984671 */
        st = lmmc_digamma(2.0, &res);
        if (st != LMMC_STATUS_OK || fabs(res - 0.4227843350984671) > 1e-10) { rc = 1; goto done; }

        /* psi(0.5) = -gamma - 2*ln(2) ≈ -1.9635100260214235 */
        st = lmmc_digamma(0.5, &res);
        if (st != LMMC_STATUS_OK || fabs(res - (-1.9635100260214235)) > 1e-10) { rc = 1; goto done; }

        /* 递推关系: psi(x+1) = psi(x) + 1/x */
        {
            lmmc_real_t psi_x, psi_x1;
            double x = 3.7;
            st = lmmc_digamma(x, &psi_x);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            st = lmmc_digamma(x + 1.0, &psi_x1);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            if (fabs(psi_x1 - psi_x - 1.0/x) > 1e-10) { rc = 1; goto done; }
        }

        /* 大参数渐近: psi(x) ~ ln(x) - 1/(2x) */
        {
            lmmc_real_t psi_val;
            double x = 100.0;
            st = lmmc_digamma(x, &psi_val);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            double approx = log(x) - 0.5 / x;
            if (fabs(psi_val - approx) > 1e-5) { rc = 1; goto done; }
        }

        /* 无效参数 */
        st = lmmc_digamma(0.0, &res);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_digamma(-1.0, &res);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }

        st = lmmc_digamma(1.0, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ===== Section 8: tgamma 递推关系 Gamma(x+1) = x * Gamma(x) ===== */
    test_section = 8;
    {
        double test_x[] = {0.5, 1.5, 2.5, 3.5, 4.5, 0.1, 0.9, 1.7, 5.3};
        int n = sizeof(test_x) / sizeof(test_x[0]);
        for (int i = 0; i < n; i++) {
            lmmc_real_t gx, gx1;
            st = lmmc_tgamma(test_x[i], &gx);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            st = lmmc_tgamma(test_x[i] + 1.0, &gx1);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            double rel_err = fabs(gx1 - test_x[i] * gx) / (fabs(gx1) + 1e-300);
            if (rel_err > 1e-10) {
                printf("tgamma recurrence failed at x=%g: Gamma(x+1)=%.17g, x*Gamma(x)=%.17g\n",
                       test_x[i], gx1, test_x[i] * gx);
                rc = 1; goto done;
            }
        }
    }

done:
    if (rc != 0) {
        printf("special functions test FAILED at section %d\n", test_section);
    } else {
        printf("special functions test PASSED\n");
    }
    return rc;
}
