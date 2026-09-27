/**
 * @file test_vectorized.c
 * @brief Property-based and unit tests for vectorized apply pattern.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define PBT_ITERATIONS 100

static double rand_double(double lo, double hi) {
    return ((double)rand() / RAND_MAX) * (hi - lo) + lo;
}

struct apply_fixture {
    lmmc_vec_t input, intermediate, output;
};

static int setup_property(void **state) {
    struct apply_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown_property(void **state) {
    struct apply_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->input);
    lmmc_vec_destroy(&fixture->intermediate);
    lmmc_vec_destroy(&fixture->output);
    free(fixture);
    *state = NULL;
    return 0;
}

static void test_property16_vec_apply_exp_log_roundtrip(void **state) {
    struct apply_fixture *fixture = *state;
    lmmc_vec_t *v = &fixture->input, *after_exp = &fixture->intermediate;
    lmmc_vec_t *recovered = &fixture->output;
    const double eps = 1e-10;

    for (int i = 0; i < PBT_ITERATIONS; i++) {
        size_t size = (size_t)(rand() % 20) + 1;
        assert_int_equal(lmmc_vec_create(size, v), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(size, after_exp), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(size, recovered), LMMC_STATUS_OK);
        for (size_t j = 0; j < size; j++) {
            v->data[j] = rand_double(0.1, 10.0);
        }
        assert_int_equal(lmmc_vec_apply_exp(v, after_exp), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_apply_log(after_exp, recovered), LMMC_STATUS_OK);
        for (size_t j = 0; j < size; j++) {
            if (!lmmc_test_nearly_equal(recovered->data[j], v->data[j], eps)) {
                fail_msg("Exp-log round-trip failed at iter %d, elem %" PRIuMAX ": got %g, expected %g", i, (uintmax_t)(j), recovered->data[j], v->data[j]);
            }
        }
        lmmc_vec_destroy(v);
        lmmc_vec_destroy(after_exp);
        lmmc_vec_destroy(recovered);
    }
}

static void test_property17_vec_apply_inplace_equivalence(void **state) {
    struct apply_fixture *fixture = *state;
    lmmc_vec_t *v_inplace = &fixture->input, *v_copy = &fixture->intermediate;
    lmmc_vec_t *out_of_place = &fixture->output;
    const double eps = 1e-15;

    for (int i = 0; i < PBT_ITERATIONS; i++) {
        size_t size = (size_t)(rand() % 20) + 1;
        assert_int_equal(lmmc_vec_create(size, v_inplace), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(size, v_copy), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(size, out_of_place), LMMC_STATUS_OK);
        for (size_t j = 0; j < size; j++) {
            double val = rand_double(-10.0, 10.0);
            v_inplace->data[j] = val;
            v_copy->data[j] = val;
        }
        assert_int_equal(lmmc_vec_apply(v_inplace, sin, v_inplace), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_apply(v_copy, sin, out_of_place), LMMC_STATUS_OK);
        for (size_t j = 0; j < size; j++) {
            if (!lmmc_test_nearly_equal(v_inplace->data[j], out_of_place->data[j], eps)) {
                fail_msg("In-place mismatch at iter %d, elem %" PRIuMAX ": in-place=%g, out-of-place=%g", i, (uintmax_t)(j), v_inplace->data[j], out_of_place->data[j]);
            }
        }
        lmmc_vec_destroy(v_inplace);
        lmmc_vec_destroy(v_copy);
        lmmc_vec_destroy(out_of_place);
    }
}

/* ---- Unit Tests ---- */

/**
 * Unit test: NULL func should return LMMC_STATUS_INVALID_ARGUMENT.
 */
static void test_unit_null_func(void **state) {
    (void)state;
    lmmc_vec_t v, out;
    lmmc_status_t st;

    st = lmmc_vec_create(3, &v);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_vec_create(3, &out);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&v);
        fail();
    }

    v.data[0] = 1.0;
    v.data[1] = 2.0;
    v.data[2] = 3.0;

    st = lmmc_vec_apply(&v, NULL, &out);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        print_error("    Expected LMMC_STATUS_INVALID_ARGUMENT for NULL func, got %d\n", (int)st);
        lmmc_vec_destroy(&v);
        lmmc_vec_destroy(&out);
        fail();
    }

    lmmc_vec_destroy(&v);
    lmmc_vec_destroy(&out);
}

/**
 * Unit test: Dimension mismatch between input and output vectors.
 */
static void test_unit_dimension_mismatch(void **state) {
    (void)state;
    lmmc_vec_t v, out;
    lmmc_status_t st;

    st = lmmc_vec_create(3, &v);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_vec_create(5, &out);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&v);
        fail();
    }

    v.data[0] = 1.0;
    v.data[1] = 2.0;
    v.data[2] = 3.0;

    st = lmmc_vec_apply(&v, sin, &out);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        print_error("    Expected LMMC_STATUS_DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_vec_destroy(&v);
        lmmc_vec_destroy(&out);
        fail();
    }

    lmmc_vec_destroy(&v);
    lmmc_vec_destroy(&out);
}

/**
 * Unit test: Known values — apply sin to [0, pi/2, pi] should give [0, 1, 0].
 */
static void test_unit_known_values_sin(void **state) {
    (void)state;
    lmmc_vec_t v, out;
    lmmc_status_t st;
    double eps = 1e-12;

    st = lmmc_vec_create(3, &v);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_vec_create(3, &out);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&v);
        fail();
    }

    v.data[0] = 0.0;
    v.data[1] = LMMC_CONST_PI / 2.0;
    v.data[2] = LMMC_CONST_PI;

    st = lmmc_vec_apply_sin(&v, &out);
    if (st != LMMC_STATUS_OK) {
        print_error("    vec_apply_sin failed, status=%d\n", (int)st);
        lmmc_vec_destroy(&v);
        lmmc_vec_destroy(&out);
        fail();
    }

    /* sin(0) = 0 */
    if (!lmmc_test_nearly_equal(out.data[0], 0.0, eps)) {
        print_error("    sin(0) = %g, expected 0\n", out.data[0]);
        lmmc_vec_destroy(&v);
        lmmc_vec_destroy(&out);
        fail();
    }

    /* sin(pi/2) = 1 */
    if (!lmmc_test_nearly_equal(out.data[1], 1.0, eps)) {
        print_error("    sin(pi/2) = %g, expected 1\n", out.data[1]);
        lmmc_vec_destroy(&v);
        lmmc_vec_destroy(&out);
        fail();
    }

    /* sin(pi) = 0 */
    if (!lmmc_test_nearly_equal(out.data[2], 0.0, eps)) {
        print_error("    sin(pi) = %g, expected 0\n", out.data[2]);
        lmmc_vec_destroy(&v);
        lmmc_vec_destroy(&out);
        fail();
    }

    lmmc_vec_destroy(&v);
    lmmc_vec_destroy(&out);
}

/* ---- Main ---- */

static int setup(void **state) {
    (void)state;
    srand(12345);
    assert_int_equal(lmmc_init(), LMMC_STATUS_OK);
    return 0;
}

static int teardown(void **state) {
    (void)state;
    assert_int_equal(lmmc_deinit(), LMMC_STATUS_OK);
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_property16_vec_apply_exp_log_roundtrip, setup_property, teardown_property),
        cmocka_unit_test_setup_teardown(test_property17_vec_apply_inplace_equivalence, setup_property, teardown_property),
        cmocka_unit_test(test_unit_null_func),
        cmocka_unit_test(test_unit_dimension_mismatch),
        cmocka_unit_test(test_unit_known_values_sin),
    };
    return cmocka_run_group_tests(tests, setup, teardown);
}
