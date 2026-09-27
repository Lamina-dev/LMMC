/**
 * @file test_complex_status.c
 * @brief 复数接口的参数校验与错误状态测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define PBT_ITERATIONS 100
#define TEST_PI LMMC_CONST_PI

struct test_fixture {
    lmmc_cvec_t vec;
    lmmc_cmat_t mat;
    lmmc_status_t init_status;
};

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t init_status = fixture->init_status;
    lmmc_cvec_destroy(&fixture->vec);
    lmmc_cmat_destroy(&fixture->mat);
    free(fixture);
    *state = NULL;
    if (init_status == LMMC_STATUS_OK) {
        assert_int_equal(lmmc_deinit(), LMMC_STATUS_OK);
    }
    return 0;
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    lmmc_status_t st = lmmc_init();
    fixture->init_status = st;
    if (st != LMMC_STATUS_OK) {
        teardown(state);
        return st;
    }
    return 0;
}

static void test_null_construction(void **state) {
    (void)state;
    lmmc_status_t st;
    st = lmmc_complex_create(1.0, 2.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    complex_create(NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    st = lmmc_complex_from_polar(1.0, 0.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    from_polar(NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }
}

static void test_null_arithmetic(void **state) {
    (void)state;
    lmmc_complex_t z = {1.0, 2.0}, out_c;
    lmmc_status_t st;
    st = lmmc_complex_add(NULL, &z, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    add(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    st = lmmc_complex_mul(&z, NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    mul(...,NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    st = lmmc_complex_div(&z, &z, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    div(...,NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }
}

static void test_null_polar(void **state) {
    (void)state;
    lmmc_complex_t z = {1.0, 2.0};
    lmmc_real_t out_r;
    lmmc_status_t st;
    st = lmmc_complex_modulus(NULL, &out_r);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    modulus(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    st = lmmc_complex_arg(&z, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    arg(...,NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }
}

static void test_null_transcendental(void **state) {
    (void)state;
    lmmc_complex_t z = {1.0, 2.0}, out_c;
    lmmc_status_t st;
    st = lmmc_complex_exp(NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    exp(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    st = lmmc_complex_log(NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    log(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    st = lmmc_complex_sqrt(NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    sqrt(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    st = lmmc_complex_sin(NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    sin(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    st = lmmc_complex_cos(NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    cos(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    st = lmmc_complex_pow(NULL, &z, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    pow(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }
}

static void test_null_containers(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;
    st = lmmc_cvec_create(5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    cvec_create(5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    {
        st = lmmc_cvec_create(0, &fixture->vec);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            fail_msg("    cvec_create(0,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        }
    }

    st = lmmc_cmat_create(3, 3, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    cmat_create(3,3,NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
    }

    {
        st = lmmc_cmat_create(0, 3, &fixture->mat);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            fail_msg("    cmat_create(0,3,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        }
    }

    {
        st = lmmc_cmat_create(3, 0, &fixture->mat);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            fail_msg("    cmat_create(3,0,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        }
    }
}
static void test_unit_zero_divisor(void **state) {
    (void)state;
    lmmc_complex_t a, zero_b, out;
    lmmc_status_t st;

    a.real = 3.0;
    a.imag = 4.0;
    zero_b.real = 0.0;
    zero_b.imag = 0.0;

    st = lmmc_complex_div(&a, &zero_b, &out);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        fail_msg("    div by zero expected NUMERICAL_FAILURE, got %d\n", (int)st);
    }
}

static void test_unit_log_zero(void **state) {
    (void)state;
    lmmc_complex_t zero_z, out;
    lmmc_status_t st;

    zero_z.real = 0.0;
    zero_z.imag = 0.0;

    st = lmmc_complex_log(&zero_z, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        fail_msg("    log(0) expected OUT_OF_RANGE, got %d\n", (int)st);
    }
}
static void test_unit_nonfinite_status_contract(void **state) {
    (void)state;
    lmmc_complex_t out;
    lmmc_status_t st;
    const lmmc_complex_t max_value = {DBL_MAX, 0.0};
    const lmmc_complex_t two = {2.0, 0.0};
    const lmmc_complex_t large_real = {1000.0, 0.0};
    const lmmc_complex_t large_imag = {0.0, 1000.0};
    const lmmc_complex_t infinite = {INFINITY, 0.0};

    st = lmmc_complex_create(NAN, 0.0, &out);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        fail_msg("    non-finite constructor input returned %d\n", (int)st);
    }
    st = lmmc_complex_add(&max_value, &max_value, &out);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        fail_msg("    overflowing addition returned %d\n", (int)st);
    }
    out.real = 7.0;
    out.imag = 9.0;
    st = lmmc_complex_mul(&max_value, &two, &out);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || out.real != 7.0 || out.imag != 9.0) {
        fail_msg("    overflowing multiplication returned %d\n", (int)st);
    }
    st = lmmc_complex_exp(&large_real, &out);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        fail_msg("    overflowing exponential returned %d\n", (int)st);
    }
    st = lmmc_complex_sin(&large_imag, &out);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        fail_msg("    overflowing sine returned %d\n", (int)st);
    }
    st = lmmc_complex_cos(&large_imag, &out);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        fail_msg("    overflowing cosine returned %d\n", (int)st);
    }
    st = lmmc_complex_arg(&infinite, &out.real);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        fail_msg("    non-finite argument input returned %d\n", (int)st);
    }
}
static void test_unit_polar_zero_r(void **state) {
    (void)state;
    lmmc_complex_t z;
    lmmc_status_t st;

    st = lmmc_complex_from_polar(0.0, 1.234, &z);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    from_polar(0, theta) failed\n");
    }
    if (z.real != 0.0 || z.imag != 0.0) {
        fail_msg("    from_polar(0, theta) expected (0,0), got (%g,%g)\n", z.real, z.imag);
    }
}

static void test_unit_destroy_null(void **state) {
    (void)state;
    lmmc_cvec_destroy(NULL);
    lmmc_cmat_destroy(NULL);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_null_construction, setup, teardown),
        cmocka_unit_test_setup_teardown(test_null_arithmetic, setup, teardown),
        cmocka_unit_test_setup_teardown(test_null_polar, setup, teardown),
        cmocka_unit_test_setup_teardown(test_null_transcendental, setup, teardown),
        cmocka_unit_test_setup_teardown(test_null_containers, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_zero_divisor, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_log_zero, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_nonfinite_status_contract, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_polar_zero_r, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_destroy_null, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
