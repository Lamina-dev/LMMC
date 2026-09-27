/**
 * @file test_complex_properties.c
 * @brief 复数运算的代数性质与容器初始化测试。
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
    srand(12345);
    return 0;
}

static double rand_double(double lo, double hi) {
    return ((double)rand() / RAND_MAX) * (hi - lo) + lo;
}

static double normalize_angle(double theta) {
    while (theta > TEST_PI)
        theta -= 2.0 * TEST_PI;
    while (theta <= -TEST_PI)
        theta += 2.0 * TEST_PI;
    return theta;
}

static int complex_nearly_equal(const lmmc_complex_t *a, const lmmc_complex_t *b, double eps) {
    return lmmc_test_nearly_equal(a->real, b->real, eps) &&
           lmmc_test_nearly_equal(a->imag, b->imag, eps);
}
static void test_property1_polar_cartesian_roundtrip(void **state) {
    (void)state;
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        double r = rand_double(0.01, 10.0);
        double theta = rand_double(-TEST_PI, TEST_PI);
        lmmc_complex_t z;
        lmmc_real_t mod_out, arg_out;
        lmmc_status_t st;

        st = lmmc_complex_from_polar(r, theta, &z);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    from_polar failed at iteration %d\n", i);
        }

        st = lmmc_complex_modulus(&z, &mod_out);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    modulus failed at iteration %d\n", i);
        }

        st = lmmc_complex_arg(&z, &arg_out);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    arg failed at iteration %d\n", i);
        }

        if (!lmmc_test_nearly_equal(mod_out, r, eps)) {
            fail_msg("    Modulus mismatch at iter %d: got %g, expected %g\n", i, mod_out, r);
        }

        double expected_theta = normalize_angle(theta);
        if (!lmmc_test_nearly_equal(arg_out, expected_theta, eps)) {
            fail_msg("    Arg mismatch at iter %d: got %g, expected %g\n", i, arg_out, expected_theta);
        }
    }
}

static void test_property2_mul_div_roundtrip(void **state) {
    (void)state;
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_complex_t a, b, product, recovered;
        lmmc_status_t st;

        a.real = rand_double(-10.0, 10.0);
        a.imag = rand_double(-10.0, 10.0);

        do {
            b.real = rand_double(-10.0, 10.0);
            b.imag = rand_double(-10.0, 10.0);
        } while (fabs(b.real) < 1e-6 && fabs(b.imag) < 1e-6);

        st = lmmc_complex_mul(&a, &b, &product);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    mul failed at iteration %d\n", i);
        }

        st = lmmc_complex_div(&product, &b, &recovered);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    div failed at iteration %d\n", i);
        }

        if (!complex_nearly_equal(&recovered, &a, eps)) {
            fail_msg("    Round-trip failed at iter %d: a=(%g,%g), recovered=(%g,%g)\n", i, a.real, a.imag, recovered.real, recovered.imag);
        }
    }
}

static void test_property3_exp_log_roundtrip(void **state) {
    (void)state;
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_complex_t z, log_z, recovered;
        lmmc_status_t st;

        do {
            z.real = rand_double(-5.0, 5.0);
            z.imag = rand_double(-5.0, 5.0);
        } while (fabs(z.real) < 1e-6 && fabs(z.imag) < 1e-6);

        st = lmmc_complex_log(&z, &log_z);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    log failed at iteration %d\n", i);
        }

        st = lmmc_complex_exp(&log_z, &recovered);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    exp failed at iteration %d\n", i);
        }

        if (!complex_nearly_equal(&recovered, &z, eps)) {
            fail_msg("    exp-log round-trip failed at iter %d: z=(%g,%g), recovered=(%g,%g)\n", i, z.real, z.imag, recovered.real, recovered.imag);
        }
    }
}

static void test_property4_pythagorean_identity(void **state) {
    (void)state;
    int i;
    double eps = 1e-9;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_complex_t z, sin_z, cos_z, sin2, cos2, sum;
        lmmc_complex_t one;
        lmmc_status_t st;

        z.real = rand_double(-3.0, 3.0);
        z.imag = rand_double(-3.0, 3.0);

        st = lmmc_complex_sin(&z, &sin_z);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    sin failed at iteration %d\n", i);
        }

        st = lmmc_complex_cos(&z, &cos_z);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    cos failed at iteration %d\n", i);
        }

        st = lmmc_complex_mul(&sin_z, &sin_z, &sin2);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    mul(sin,sin) failed at iteration %d\n", i);
        }

        st = lmmc_complex_mul(&cos_z, &cos_z, &cos2);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    mul(cos,cos) failed at iteration %d\n", i);
        }

        st = lmmc_complex_add(&sin2, &cos2, &sum);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    add failed at iteration %d\n", i);
        }

        one.real = 1.0;
        one.imag = 0.0;

        if (!complex_nearly_equal(&sum, &one, eps)) {
            fail_msg("    Pythagorean identity failed at iter %d: sum=(%g,%g)\n", i, sum.real, sum.imag);
        }
    }
}

static void test_property5_sqrt_roundtrip(void **state) {
    (void)state;
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_complex_t z, sqrt_z, squared;
        lmmc_status_t st;

        z.real = rand_double(-10.0, 10.0);
        z.imag = rand_double(-10.0, 10.0);

        st = lmmc_complex_sqrt(&z, &sqrt_z);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    sqrt failed at iteration %d\n", i);
        }

        st = lmmc_complex_mul(&sqrt_z, &sqrt_z, &squared);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    mul(sqrt,sqrt) failed at iteration %d\n", i);
        }

        if (!complex_nearly_equal(&squared, &z, eps)) {
            fail_msg("    sqrt round-trip failed at iter %d: z=(%g,%g), squared=(%g,%g)\n", i, z.real, z.imag, squared.real, squared.imag);
        }
    }
}

static void test_property6_zero_initialization(void **state) {
    struct test_fixture *fixture = *state;
    int i;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        size_t vec_size = (size_t)(rand() % 50) + 1;
        size_t mat_rows = (size_t)(rand() % 20) + 1;
        size_t mat_cols = (size_t)(rand() % 20) + 1;
        size_t j, r, c;

        lmmc_status_t st;

        st = lmmc_cvec_create(vec_size, &fixture->vec);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    cvec_create failed at iteration %d\n", i);
        }

        for (j = 0; j < vec_size; j++) {
            if (fixture->vec.data[j].real != 0.0 || fixture->vec.data[j].imag != 0.0) {
                fail_msg("    Vector element [%" PRIuMAX "] not zero at iter %d\n", (uintmax_t)(j), i);
            }
        }
        lmmc_cvec_destroy(&fixture->vec);

        st = lmmc_cmat_create(mat_rows, mat_cols, &fixture->mat);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    cmat_create failed at iteration %d\n", i);
        }

        for (r = 0; r < mat_rows; r++) {
            for (c = 0; c < mat_cols; c++) {
                lmmc_complex_t *elem = &fixture->mat.data[r * fixture->mat.stride + c];
                if (elem->real != 0.0 || elem->imag != 0.0) {
                    fail_msg("    Matrix element [%" PRIuMAX "][%" PRIuMAX "] not zero at iter %d\n", (uintmax_t)(r), (uintmax_t)(c), i);
                }
            }
        }
        lmmc_cmat_destroy(&fixture->mat);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_property1_polar_cartesian_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property2_mul_div_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property3_exp_log_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property4_pythagorean_identity, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property5_sqrt_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property6_zero_initialization, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
