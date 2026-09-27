/**
 * @file test_special_functions.c
 * 针对 LMMC 特殊函数（erf, erfc, lgamma, tgamma, beta, digamma）的单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static void test_erf_symmetry_values(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t res = 0.0;
    st = lmmc_erf(0.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res) <= 1e-15));

    st = lmmc_erf(1.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 0.8427007929497149) <= 1e-12));

    st = lmmc_erf(-1.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res + 0.8427007929497149) <= 1e-12));
}

static void test_erf_positive_tail(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t res = 0.0;
    st = lmmc_erf(0.5, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 0.5204998778130465) <= 1e-12));

    st = lmmc_erf(2.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 0.9953222650189527) <= 1e-12));

    st = lmmc_erf(5.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 1.0) <= 1e-10));

    st = lmmc_erf(1.0, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_erfc_values(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t res = 0.0;
    st = lmmc_erfc(0.0, &res);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(fabs(res - 1.0) <= 1e-15);

    st = lmmc_erfc(1.0, &res);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(fabs(res - 0.1572992070502851) <= 1e-12);

    st = lmmc_erfc(2.0, &res);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(fabs(res - 0.004677734981047266) <= 1e-12);

    st = lmmc_erfc(-1.0, &res);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(fabs(res - 1.8427007929497149) <= 1e-12);

    st = lmmc_erfc(10.0, &res);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(res >= 0.0);
    assert_true(res <= 1e-40);
}

static void test_erf_complement(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    {
        double test_x[] = {0.1, 0.3, 0.7, 1.5, 3.0};
        for (int i = 0; i < 5; i++) {
            lmmc_real_t erf_val, erfc_val;
            st = lmmc_erf(test_x[i], &erf_val);
            assert_false(st != LMMC_STATUS_OK);
            st = lmmc_erfc(test_x[i], &erfc_val);
            assert_false(st != LMMC_STATUS_OK);
            assert_true((fabs(erf_val + erfc_val - 1.0) <= 1e-12));
        }
    }
}

static void test_erf_libm_agreement(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    double test_x[] = {-5.0, -3.0, -2.0, -1.0, -0.5, -0.1, 0.0,
                       0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0};
    int n = sizeof(test_x) / sizeof(test_x[0]);
    for (int i = 0; i < n; i++) {
        lmmc_real_t res;
        st = lmmc_erf(test_x[i], &res);
        assert_false(st != LMMC_STATUS_OK);
        double ref = erf(test_x[i]);
        if (!(fabs(res - ref) <= 1e-12)) {
            fail_msg("erf(%g): got %.17g, expected %.17g, diff=%.3e\n", test_x[i], res, ref, fabs(res - ref));
        }
    }
}

static void test_lgamma_values(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t res = 0.0;
    st = lmmc_lgamma(1.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res) <= 1e-12));

    st = lmmc_lgamma(2.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res) <= 1e-12));

    st = lmmc_lgamma(0.5, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 0.5723649429247001) <= 1e-10));

    st = lmmc_lgamma(5.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - log(24.0)) <= 1e-10));

    st = lmmc_lgamma(10.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - log(362880.0)) <= 1e-10));
}

static void test_lgamma_domain(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t res = 0.0;
    st = lmmc_lgamma(0.0, &res);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_lgamma(-1.0, &res);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_lgamma(1.0, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_tgamma_values(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t res = 0.0;
    st = lmmc_tgamma(1.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 1.0) <= 1e-12));

    st = lmmc_tgamma(2.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 1.0) <= 1e-12));

    st = lmmc_tgamma(5.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 24.0) <= 1e-10));

    st = lmmc_tgamma(0.5, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 1.7724538509055159) <= 1e-10));

    st = lmmc_tgamma(10.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 362880.0) <= 1e-5));
}

static void test_tgamma_domain(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t res = 0.0;
    st = lmmc_tgamma(0.0, &res);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tgamma(-1.0, &res);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_beta_elementary_values(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t res = 0.0;
    st = lmmc_beta(1.0, 1.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 1.0) <= 1e-12));

    st = lmmc_beta(1.0, 2.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 0.5) <= 1e-12));

    st = lmmc_beta(2.0, 2.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 1.0 / 6.0) <= 1e-12));

    st = lmmc_beta(0.5, 0.5, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - LMMC_PI) <= 1e-10));

    st = lmmc_beta(3.0, 4.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 1.0 / 60.0) <= 1e-12));
}

static void test_beta_extreme_reference_values(void **state) {
    (void)state;
    lmmc_real_t res = 0.0;
    {
        const struct {
            double a, b, expected;
        } cases[] = {
            {16, 0.5, 0.44658827748481361},
            {16, 15.5, 2.9767703484331325e-10},
            {1e4, 0.5, 0.017724760067171167},
            {1e4, 15.5, 3.3109878293174557e-51},
            {1e16, 0.5, 1.7724538509055161e-08},
            {1e16, 15.5, 3.3483860987355269e-237},
            {1e100, 0.5, 1.772453850905516e-50},
            {1e100, 15.5, 0.0},
            {DBL_MAX, 0.5, 1.3219564750381269e-154},
            {DBL_MAX, 15.5, 0.0}};
        size_t i;
        for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
            double swapped;
            assert_false(lmmc_beta(cases[i].a, cases[i].b, &res) != LMMC_STATUS_OK || !isfinite(res) || (cases[i].expected == 0.0 ? res != 0.0 : fabs(res / cases[i].expected - 1.0) > 2e-12));
            assert_false(lmmc_beta(cases[i].b, cases[i].a, &swapped) != LMMC_STATUS_OK || swapped != res);
        }
    }
}

static void check_beta_integer_shape(double a, unsigned b, lmmc_real_t *res) {
    double expected = 1.0 / a;
    unsigned k;
    for (k = 1; k < b; ++k) {
        expected *= (double)k / (a + (double)k);
    }
    assert_false(lmmc_beta(a, (double)b, res) != LMMC_STATUS_OK || !isfinite(*res) || (expected == 0.0 ? *res != 0.0 : fabs(*res / expected - 1.0) > 2e-12));
}

static void test_beta_integer_shape_values(void **state) {
    (void)state;
    lmmc_real_t res = 0.0;
    {
        const double large[] = {nextafter(16.0, 0.0), 16.0, nextafter(16.0, INFINITY),
                                1e4, 1e16, 1e100, DBL_MAX};
        size_t i;
        for (i = 0; i < sizeof(large) / sizeof(large[0]); ++i) {
            const double reference = 1.0 / large[i];
            assert_false(lmmc_beta(large[i], 1.0, &res) != LMMC_STATUS_OK || !isfinite(res) || !(fabs(res / reference - 1.0) <= 2e-12));
            assert_false(lmmc_beta(1.0, large[i], &res) != LMMC_STATUS_OK || !isfinite(res) || !(fabs(res / reference - 1.0) <= 2e-12));
            if (large[i] <= 1e100) {
                unsigned b;
                for (b = 2; b <= 8; b += 6) {
                    check_beta_integer_shape(large[i], b, &res);
                }
            }
        }
    }
}

static void test_beta_limit_values(void **state) {
    (void)state;
    lmmc_real_t res = 0.0;
    assert_int_equal(lmmc_beta(1e-300, 1.0, &res), LMMC_STATUS_OK);
    assert_true(isfinite(res) && fabs(res / 1e300 - 1.0) <= 2e-12);
    assert_int_equal(lmmc_beta(DBL_MAX, DBL_MAX, &res), LMMC_STATUS_OK);
    assert_true(res == 0.0);
    res = 123.0;
    assert_int_equal(lmmc_beta(nextafter(0.0, 1.0), 1.0, &res),
                     LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(res == 123.0);
}

static void test_beta_symmetry(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    {
        lmmc_real_t r1, r2;
        st = lmmc_beta(2.5, 3.7, &r1);
        assert_false(st != LMMC_STATUS_OK);
        st = lmmc_beta(3.7, 2.5, &r2);
        assert_false(st != LMMC_STATUS_OK);
        assert_true((fabs(r1 - r2) <= 1e-14));
    }
}

static void test_beta_domain(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t res = 0.0;
    st = lmmc_beta(0.0, 1.0, &res);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_beta(1.0, -1.0, &res);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_beta(1.0, 1.0, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    {
        const double invalid[] = {0.0, -1.0, NAN, INFINITY, -INFINITY};
        size_t i;
        for (i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
            res = 123.0;
            assert_false(lmmc_beta(invalid[i], 1.0, &res) != LMMC_STATUS_INVALID_ARGUMENT || res != 123.0);
            assert_false(lmmc_beta(1.0, invalid[i], &res) != LMMC_STATUS_INVALID_ARGUMENT || res != 123.0);
        }
    }
}

static void test_digamma_values(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t res = 0.0;
    st = lmmc_digamma(1.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - (-0.5772156649015329)) <= 1e-10));

    st = lmmc_digamma(2.0, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - 0.4227843350984671) <= 1e-10));

    st = lmmc_digamma(0.5, &res);
    assert_false(st != LMMC_STATUS_OK || !(fabs(res - (-1.9635100260214235)) <= 1e-10));
}

static void test_digamma_recurrence(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    {
        lmmc_real_t psi_x, psi_x1;
        double x = 3.7;
        st = lmmc_digamma(x, &psi_x);
        assert_false(st != LMMC_STATUS_OK);
        st = lmmc_digamma(x + 1.0, &psi_x1);
        assert_false(st != LMMC_STATUS_OK);
        assert_true((fabs(psi_x1 - psi_x - 1.0 / x) <= 1e-10));
    }
}

static void test_digamma_asymptotic(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    {
        lmmc_real_t psi_val;
        double x = 100.0;
        st = lmmc_digamma(x, &psi_val);
        assert_false(st != LMMC_STATUS_OK);
        double approx = log(x) - 0.5 / x;
        assert_true((fabs(psi_val - approx) <= 1e-5));
    }
}

static void test_digamma_domain(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t res = 0.0;
    st = lmmc_digamma(0.0, &res);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_digamma(-1.0, &res);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_digamma(1.0, NULL);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_tgamma_recurrence(void **state) {
    (void)state;
    lmmc_status_t st = LMMC_STATUS_OK;
    double test_x[] = {0.5, 1.5, 2.5, 3.5, 4.5, 0.1, 0.9, 1.7, 5.3};
    int n = sizeof(test_x) / sizeof(test_x[0]);
    for (int i = 0; i < n; i++) {
        lmmc_real_t gx, gx1;
        st = lmmc_tgamma(test_x[i], &gx);
        assert_false(st != LMMC_STATUS_OK);
        st = lmmc_tgamma(test_x[i] + 1.0, &gx1);
        assert_false(st != LMMC_STATUS_OK);
        double rel_err = fabs(gx1 - test_x[i] * gx) / (fabs(gx1) + 1e-300);
        if (!(rel_err <= 1e-10)) {
            fail_msg("tgamma recurrence failed at x=%g: Gamma(x+1)=%.17g, x*Gamma(x)=%.17g\n", test_x[i], gx1, test_x[i] * gx);
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_erf_symmetry_values),
        cmocka_unit_test(test_erf_positive_tail),
        cmocka_unit_test(test_erfc_values),
        cmocka_unit_test(test_erf_complement),
        cmocka_unit_test(test_erf_libm_agreement),
        cmocka_unit_test(test_lgamma_values),
        cmocka_unit_test(test_lgamma_domain),
        cmocka_unit_test(test_tgamma_values),
        cmocka_unit_test(test_tgamma_domain),
        cmocka_unit_test(test_beta_elementary_values),
        cmocka_unit_test(test_beta_extreme_reference_values),
        cmocka_unit_test(test_beta_integer_shape_values),
        cmocka_unit_test(test_beta_limit_values),
        cmocka_unit_test(test_beta_symmetry),
        cmocka_unit_test(test_beta_domain),
        cmocka_unit_test(test_digamma_values),
        cmocka_unit_test(test_digamma_recurrence),
        cmocka_unit_test(test_digamma_asymptotic),
        cmocka_unit_test(test_digamma_domain),
        cmocka_unit_test(test_tgamma_recurrence),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
