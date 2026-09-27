/**
 * @file test_quadrature.c
 * 针对 LMMC 中 quadrature 相关接口的单元测试。
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static lmmc_real_t fn_const(lmmc_real_t x, void *ud) {
    (void)x;
    (void)ud;
    return 1.0;
}

static lmmc_real_t fn_scaled_const(lmmc_real_t x, void *ud) {
    (void)x;
    return *(const lmmc_real_t *)ud;
}

static lmmc_real_t fn_x1(lmmc_real_t x, void *ud) {
    (void)ud;
    return x;
}

static lmmc_real_t fn_x2(lmmc_real_t x, void *ud) {
    (void)ud;
    return x * x;
}

static lmmc_real_t fn_x3(lmmc_real_t x, void *ud) {
    (void)ud;
    return x * x * x;
}

static lmmc_real_t fn_x4(lmmc_real_t x, void *ud) {
    (void)ud;
    return x * x * x * x;
}

static lmmc_real_t fn_sin(lmmc_real_t x, void *ud) {
    (void)ud;
    return sin(x);
}

static lmmc_real_t fn_exp(lmmc_real_t x, void *ud) {
    (void)ud;
    return exp(x);
}

static lmmc_real_t fn_inv_sqrt(lmmc_real_t x, void *ud) {
    (void)ud;
    if (x <= 0.0)
        return 0.0;
    return 1.0 / sqrt(x);
}

static void test_trapezoid_accuracy(void **state) {
    (void)state;
    lmmc_status_t st;
    {
        lmmc_quad_func_t poly_funcs[] = {fn_const, fn_x1, fn_x2, fn_x3, fn_x4};
        double exact_vals[] = {1.0, 0.5, 1.0 / 3.0, 0.25, 0.2};

        for (int p = 0; p < 5; p++) {

            lmmc_real_t res_10, res_100;
            st = lmmc_quad_trapezoid(poly_funcs[p], NULL, 0.0, 1.0, 10, &res_10);
            assert_false(st != LMMC_STATUS_OK);
            st = lmmc_quad_trapezoid(poly_funcs[p], NULL, 0.0, 1.0, 100, &res_100);
            assert_false(st != LMMC_STATUS_OK);

            double err_10 = fabs(res_10 - exact_vals[p]);
            double err_100 = fabs(res_100 - exact_vals[p]);

            if (p <= 1) {
                assert_true(lmmc_test_nearly_equal(res_10, exact_vals[p], 1e-12));
            } else {

                assert_false(err_100 >= err_10);
            }
        }
    }
}

static void test_simpson_accuracy(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t result;
    {
        lmmc_quad_func_t poly_funcs[] = {fn_const, fn_x1, fn_x2, fn_x3};
        double exact_vals[] = {1.0, 0.5, 1.0 / 3.0, 0.25};

        for (int p = 0; p < 4; p++) {

            st = lmmc_quad_simpson(poly_funcs[p], NULL, 0.0, 1.0, 2, &result);
            assert_false(st != LMMC_STATUS_OK);
            assert_true(lmmc_test_nearly_equal(result, exact_vals[p], 1e-12));

            st = lmmc_quad_simpson(poly_funcs[p], NULL, 0.0, 1.0, 4, &result);
            assert_false(st != LMMC_STATUS_OK);
            assert_true(lmmc_test_nearly_equal(result, exact_vals[p], 1e-12));
        }
    }
}

static void test_legendre_convergence(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t result;
    {
        double exact_sin = 2.0;

        for (size_t order = 7; order <= 10; order++) {
            st = lmmc_quad_gauss_legendre(fn_sin, NULL, 0.0, LMMC_CONST_PI, order, &result);
            assert_false(st != LMMC_STATUS_OK);
            assert_true(lmmc_test_nearly_equal(result, exact_sin, 1e-10));
        }

        lmmc_real_t res4, res8;
        st = lmmc_quad_gauss_legendre(fn_sin, NULL, 0.0, LMMC_CONST_PI, 4, &res4);
        assert_false(st != LMMC_STATUS_OK);
        st = lmmc_quad_gauss_legendre(fn_sin, NULL, 0.0, LMMC_CONST_PI, 8, &res8);
        assert_false(st != LMMC_STATUS_OK);
        assert_false(fabs(res8 - exact_sin) >= fabs(res4 - exact_sin));
    }
}

static void test_legendre_polynomials(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t result;
    {

        st = lmmc_quad_gauss_legendre(fn_x3, NULL, 0.0, 1.0, 2, &result);
        assert_false(st != LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, 0.25, 1e-11));

        st = lmmc_quad_gauss_legendre(fn_x4, NULL, 0.0, 1.0, 3, &result);
        assert_false(st != LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, 0.2, 1e-11));

        st = lmmc_quad_gauss_legendre(fn_x4, NULL, 0.0, 1.0, 5, &result);
        assert_false(st != LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(result, 0.2, 1e-11));
    }
}

static void test_adaptive_smooth(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_quad_result_t adaptive_result;
    {
        double exact_exp = exp(1.0) - 1.0;

        st = lmmc_quad_adaptive(fn_exp, NULL, 0.0, 1.0, 1e-10, 1e-10, 50, &adaptive_result);
        assert_false(st != LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(adaptive_result.value, exact_exp, 1e-10));
    }
}

static void test_adaptive_singular(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_quad_result_t adaptive_result;
    {
        double exact_inv_sqrt = 2.0;

        st = lmmc_quad_adaptive(fn_inv_sqrt, NULL, 1e-10, 1.0, 1e-4, 1e-4, 50, &adaptive_result);

        assert_false(st != LMMC_STATUS_OK && st != LMMC_STATUS_WARNING_MAX_DEPTH);

        assert_true(lmmc_test_nearly_equal(adaptive_result.value, exact_inv_sqrt, 1e-2));
    }
}

static void test_trapezoid_large_interval(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t result;
    {
        const lmmc_real_t scale = 1.0 / DBL_MAX;
        st = lmmc_quad_trapezoid(
            fn_scaled_const, (void *)&scale, -DBL_MAX, DBL_MAX, 8, &result);
        assert_false(st != LMMC_STATUS_OK ||
                     !lmmc_test_nearly_equal(result, 2.0, 1e-12));
    }
}

static void test_null_callbacks(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t result;
    lmmc_quad_result_t adaptive_result;
    {
        st = lmmc_quad_trapezoid(NULL, NULL, 0.0, 1.0, 10, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_simpson(NULL, NULL, 0.0, 1.0, 2, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_gauss_legendre(NULL, NULL, 0.0, 1.0, 5, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_adaptive(NULL, NULL, 0.0, 1.0, 1e-10, 1e-10, 50, &adaptive_result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }
}

static void test_zero_panels(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t result;
    {
        st = lmmc_quad_trapezoid(fn_const, NULL, 0.0, 1.0, 0, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_simpson(fn_const, NULL, 0.0, 1.0, 0, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }
}

static void test_equal_endpoints(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t result;
    lmmc_quad_result_t adaptive_result;
    {
        st = lmmc_quad_trapezoid(fn_const, NULL, 1.0, 1.0, 10, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_simpson(fn_const, NULL, 1.0, 1.0, 2, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_gauss_legendre(fn_const, NULL, 1.0, 1.0, 5, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_adaptive(fn_const, NULL, 1.0, 1.0, 1e-10, 1e-10, 50, &adaptive_result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }
}

static void test_reversed_endpoints(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t result;
    lmmc_quad_result_t adaptive_result;
    {
        st = lmmc_quad_trapezoid(fn_const, NULL, 2.0, 1.0, 10, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_simpson(fn_const, NULL, 2.0, 1.0, 2, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_gauss_legendre(fn_const, NULL, 2.0, 1.0, 5, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_adaptive(fn_const, NULL, 2.0, 1.0, 1e-10, 1e-10, 50, &adaptive_result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }
}

static void test_null_outputs(void **state) {
    (void)state;
    lmmc_status_t st;
    {
        st = lmmc_quad_trapezoid(fn_const, NULL, 0.0, 1.0, 10, NULL);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_simpson(fn_const, NULL, 0.0, 1.0, 2, NULL);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_gauss_legendre(fn_const, NULL, 0.0, 1.0, 5, NULL);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_adaptive(fn_const, NULL, 0.0, 1.0, 1e-10, 1e-10, 50, NULL);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }
}

static void test_simpson_odd_panels(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t result;
    {
        st = lmmc_quad_simpson(fn_const, NULL, 0.0, 1.0, 3, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }
}

static void test_legendre_orders(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_real_t result;
    {
        st = lmmc_quad_gauss_legendre(
            fn_const, NULL, 0.0, 1.0, 1, &result);
        assert_false(st != LMMC_STATUS_OK ||
                     !lmmc_test_nearly_equal(result, 1.0, 1e-12));

        st = lmmc_quad_gauss_legendre(fn_const, NULL, 0.0, 1.0, 21, &result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }
}

static void test_adaptive_tolerances(void **state) {
    (void)state;
    lmmc_status_t st;
    lmmc_quad_result_t adaptive_result;
    {
        st = lmmc_quad_adaptive(fn_const, NULL, 0.0, 1.0, -1.0, 1e-10, 50, &adaptive_result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

        st = lmmc_quad_adaptive(fn_const, NULL, 0.0, 1.0, 1e-10, -1.0, 50, &adaptive_result);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_trapezoid_accuracy),
        cmocka_unit_test(test_simpson_accuracy),
        cmocka_unit_test(test_legendre_convergence),
        cmocka_unit_test(test_legendre_polynomials),
        cmocka_unit_test(test_adaptive_smooth),
        cmocka_unit_test(test_adaptive_singular),
        cmocka_unit_test(test_trapezoid_large_interval),
        cmocka_unit_test(test_null_callbacks),
        cmocka_unit_test(test_zero_panels),
        cmocka_unit_test(test_equal_endpoints),
        cmocka_unit_test(test_reversed_endpoints),
        cmocka_unit_test(test_null_outputs),
        cmocka_unit_test(test_simpson_odd_panels),
        cmocka_unit_test(test_legendre_orders),
        cmocka_unit_test(test_adaptive_tolerances),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
