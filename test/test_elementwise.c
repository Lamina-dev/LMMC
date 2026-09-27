/**
 * @file test_elementwise.c
 * @brief Property-based and unit tests for element-wise vector/matrix operations.
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

/**
 * Generate a random non-zero double in [lo, hi] with |value| >= min_abs.
 */
static double rand_nonzero(double lo, double hi, double min_abs) {
    double v;
    do {
        v = rand_double(lo, hi);
    } while (fabs(v) < min_abs);
    return v;
}

struct elementwise_fixture {
    lmmc_vec_t a, b, product, recovered;
    lmmc_mat_t mat_a, mat_b, mat_product, mat_recovered;
    int *gt_out, *le_out;
};

static int setup_property(void **state) {
    struct elementwise_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown_property(void **state) {
    struct elementwise_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->a);
    lmmc_vec_destroy(&fixture->b);
    lmmc_vec_destroy(&fixture->product);
    lmmc_vec_destroy(&fixture->recovered);
    lmmc_mat_destroy(&fixture->mat_a);
    lmmc_mat_destroy(&fixture->mat_b);
    lmmc_mat_destroy(&fixture->mat_product);
    lmmc_mat_destroy(&fixture->mat_recovered);
    free(fixture->gt_out);
    free(fixture->le_out);
    free(fixture);
    *state = NULL;
    return 0;
}

static void test_property7_vec_hadamard_div_roundtrip(void **state) {
    struct elementwise_fixture *fixture = *state;
    lmmc_vec_t *a = &fixture->a, *b = &fixture->b;
    lmmc_vec_t *product = &fixture->product, *recovered = &fixture->recovered;
    const double eps = 1e-10;

    for (int i = 0; i < PBT_ITERATIONS; i++) {
        size_t size = (size_t)(rand() % 20) + 1;
        assert_int_equal(lmmc_vec_create(size, a), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(size, b), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(size, product), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(size, recovered), LMMC_STATUS_OK);

        for (size_t j = 0; j < size; j++) {
            a->data[j] = rand_double(-10.0, 10.0);
        }
        for (size_t j = 0; j < size; j++) {
            b->data[j] = rand_nonzero(0.1, 10.0, 0.1);
        }
        assert_int_equal(lmmc_vec_hadamard(a, b, product), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_elementwise_div(product, b, recovered), LMMC_STATUS_OK);
        for (size_t j = 0; j < size; j++) {
            if (!lmmc_test_nearly_equal(recovered->data[j], a->data[j], eps)) {
                fail_msg("Round-trip failed at iter %d, elem %" PRIuMAX ": got %g, expected %g", i, (uintmax_t)(j), recovered->data[j], a->data[j]);
            }
        }
        lmmc_vec_destroy(a);
        lmmc_vec_destroy(b);
        lmmc_vec_destroy(product);
        lmmc_vec_destroy(recovered);
    }
}

static void fill_matrix_hadamard_inputs(lmmc_mat_t *a, lmmc_mat_t *b) {
    for (size_t r = 0; r < a->rows; r++) {
        for (size_t c = 0; c < a->cols; c++) {
            a->data[r * a->stride + c] = rand_double(-10.0, 10.0);
        }
    }
    for (size_t r = 0; r < b->rows; r++) {
        for (size_t c = 0; c < b->cols; c++) {
            b->data[r * b->stride + c] = rand_nonzero(0.1, 10.0, 0.1);
        }
    }
}

static int check_matrix_hadamard_roundtrip(
    int iteration, double eps, const lmmc_mat_t *a, const lmmc_mat_t *recovered) {
    for (size_t r = 0; r < a->rows; r++) {
        for (size_t c = 0; c < a->cols; c++) {
            double got = recovered->data[r * recovered->stride + c];
            double expected = a->data[r * a->stride + c];
            if (!lmmc_test_nearly_equal(got, expected, eps)) {
                print_error("    Round-trip failed at iter %d, elem [%" PRIuMAX "][%" PRIuMAX "]: got %g, expected %g\n", iteration, (uintmax_t)(r), (uintmax_t)(c), got, expected);
                return 1;
            }
        }
    }
    return 0;
}

static void test_property8_mat_hadamard_div_roundtrip(void **state) {
    struct elementwise_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->mat_a, *b = &fixture->mat_b;
    lmmc_mat_t *product = &fixture->mat_product, *recovered = &fixture->mat_recovered;
    const double eps = 1e-10;

    for (int i = 0; i < PBT_ITERATIONS; i++) {
        size_t rows = (size_t)(rand() % 10) + 1;
        size_t cols = (size_t)(rand() % 10) + 1;
        assert_int_equal(lmmc_mat_create(rows, cols, a), LMMC_STATUS_OK);
        assert_int_equal(lmmc_mat_create(rows, cols, b), LMMC_STATUS_OK);
        assert_int_equal(lmmc_mat_create(rows, cols, product), LMMC_STATUS_OK);
        assert_int_equal(lmmc_mat_create(rows, cols, recovered), LMMC_STATUS_OK);
        fill_matrix_hadamard_inputs(a, b);
        assert_int_equal(lmmc_mat_hadamard(a, b, product), LMMC_STATUS_OK);
        assert_int_equal(lmmc_mat_elementwise_div(product, b, recovered), LMMC_STATUS_OK);
        assert_int_equal(check_matrix_hadamard_roundtrip(i, eps, a, recovered), 0);
        lmmc_mat_destroy(a);
        lmmc_mat_destroy(b);
        lmmc_mat_destroy(product);
        lmmc_mat_destroy(recovered);
    }
}

static void test_property9_vec_cmp_complementarity(void **state) {
    struct elementwise_fixture *fixture = *state;
    lmmc_vec_t *a = &fixture->a, *b = &fixture->b;

    for (int i = 0; i < PBT_ITERATIONS; i++) {
        size_t size = (size_t)(rand() % 20) + 1;
        assert_int_equal(lmmc_vec_create(size, a), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_create(size, b), LMMC_STATUS_OK);
        for (size_t j = 0; j < size; j++) {
            a->data[j] = rand_double(-10.0, 10.0);
            b->data[j] = rand_double(-10.0, 10.0);
        }
        fixture->gt_out = malloc(size * sizeof(*fixture->gt_out));
        fixture->le_out = malloc(size * sizeof(*fixture->le_out));
        assert_non_null(fixture->gt_out);
        assert_non_null(fixture->le_out);
        assert_int_equal(lmmc_vec_cmp_gt(a, b, fixture->gt_out), LMMC_STATUS_OK);
        assert_int_equal(lmmc_vec_cmp_le(a, b, fixture->le_out), LMMC_STATUS_OK);
        for (size_t j = 0; j < size; j++) {
            if (fixture->gt_out[j] + fixture->le_out[j] != 1) {
                fail_msg("Complementarity failed at iter %d, elem %" PRIuMAX ": gt=%d, le=%d (a=%g, b=%g)", i, (uintmax_t)(j), fixture->gt_out[j], fixture->le_out[j], a->data[j], b->data[j]);
            }
        }
        free(fixture->gt_out);
        free(fixture->le_out);
        fixture->gt_out = NULL;
        fixture->le_out = NULL;
        lmmc_vec_destroy(a);
        lmmc_vec_destroy(b);
    }
}

/* ---- Unit Tests ---- */

/* Unit test: zero divisor scan for vector elementwise_div */
static void test_unit_zero_divisor_vec(void **state) {
    (void)state;
    lmmc_vec_t a, b, c;
    lmmc_status_t st;

    st = lmmc_vec_create(4, &a);
    if (st != LMMC_STATUS_OK) {
        print_error("    vec_create(a) failed\n");
        fail();
    }
    st = lmmc_vec_create(4, &b);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&a);
        print_error("    vec_create(b) failed\n");
        fail();
    }
    st = lmmc_vec_create(4, &c);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&b);
        print_error("    vec_create(c) failed\n");
        fail();
    }

    /* Fill a with non-zero values */
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;

    /* b has a zero element at index 2 */
    b.data[0] = 1.0;
    b.data[1] = 2.0;
    b.data[2] = 0.0;
    b.data[3] = 4.0;

    st = lmmc_vec_elementwise_div(&a, &b, &c);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        print_error("    Expected NUMERICAL_FAILURE for zero divisor, got %d\n", (int)st);
        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&b);
        lmmc_vec_destroy(&c);
        fail();
    }

    lmmc_vec_destroy(&a);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&c);
}

/* Unit test: zero divisor scan for matrix elementwise_div */
static void test_unit_zero_divisor_mat(void **state) {
    (void)state;
    lmmc_mat_t a, b, c;
    lmmc_status_t st;

    st = lmmc_mat_create(2, 3, &a);
    if (st != LMMC_STATUS_OK) {
        print_error("    mat_create(a) failed\n");
        fail();
    }
    st = lmmc_mat_create(2, 3, &b);
    if (st != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&a);
        print_error("    mat_create(b) failed\n");
        fail();
    }
    st = lmmc_mat_create(2, 3, &c);
    if (st != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&a);
        lmmc_mat_destroy(&b);
        print_error("    mat_create(c) failed\n");
        fail();
    }

    /* Fill a with values */
    lmmc_mat_fill(&a, 5.0);

    /* Fill b with non-zero values, then set one to zero */
    lmmc_mat_fill(&b, 2.0);
    b.data[1 * b.stride + 1] = 0.0; /* zero at (1,1) */

    st = lmmc_mat_elementwise_div(&a, &b, &c);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        print_error("    Expected NUMERICAL_FAILURE for zero divisor in matrix, got %d\n", (int)st);
        lmmc_mat_destroy(&a);
        lmmc_mat_destroy(&b);
        lmmc_mat_destroy(&c);
        fail();
    }

    lmmc_mat_destroy(&a);
    lmmc_mat_destroy(&b);
    lmmc_mat_destroy(&c);
}

/* Unit test: dimension mismatch for vector operations */
static void test_unit_dimension_mismatch_vec(void **state) {
    (void)state;
    lmmc_vec_t a, b, c;
    int out_cmp[5];
    lmmc_status_t st;

    st = lmmc_vec_create(3, &a);
    if (st != LMMC_STATUS_OK) {
        print_error("    vec_create(a) failed\n");
        fail();
    }
    st = lmmc_vec_create(5, &b);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&a);
        print_error("    vec_create(b) failed\n");
        fail();
    }
    st = lmmc_vec_create(3, &c);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&b);
        print_error("    vec_create(c) failed\n");
        fail();
    }

    /* Hadamard with mismatched sizes */
    st = lmmc_vec_hadamard(&a, &b, &c);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        print_error("    vec_hadamard expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&b);
        lmmc_vec_destroy(&c);
        fail();
    }

    /* elementwise_div with mismatched sizes */
    st = lmmc_vec_elementwise_div(&a, &b, &c);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        print_error("    vec_elementwise_div expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&b);
        lmmc_vec_destroy(&c);
        fail();
    }

    /* cmp_gt with mismatched sizes */
    st = lmmc_vec_cmp_gt(&a, &b, out_cmp);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        print_error("    vec_cmp_gt expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&b);
        lmmc_vec_destroy(&c);
        fail();
    }

    /* cmp_le with mismatched sizes */
    st = lmmc_vec_cmp_le(&a, &b, out_cmp);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        print_error("    vec_cmp_le expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&b);
        lmmc_vec_destroy(&c);
        fail();
    }

    lmmc_vec_destroy(&a);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&c);
}

/* Unit test: dimension mismatch for matrix operations */
static void test_unit_dimension_mismatch_mat(void **state) {
    (void)state;
    lmmc_mat_t a, b, c;
    lmmc_status_t st;

    st = lmmc_mat_create(2, 3, &a);
    if (st != LMMC_STATUS_OK) {
        print_error("    mat_create(a) failed\n");
        fail();
    }
    st = lmmc_mat_create(3, 2, &b);
    if (st != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&a);
        print_error("    mat_create(b) failed\n");
        fail();
    }
    st = lmmc_mat_create(2, 3, &c);
    if (st != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&a);
        lmmc_mat_destroy(&b);
        print_error("    mat_create(c) failed\n");
        fail();
    }

    /* Hadamard with mismatched dimensions */
    st = lmmc_mat_hadamard(&a, &b, &c);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        print_error("    mat_hadamard expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_mat_destroy(&a);
        lmmc_mat_destroy(&b);
        lmmc_mat_destroy(&c);
        fail();
    }

    /* elementwise_div with mismatched dimensions */
    st = lmmc_mat_elementwise_div(&a, &b, &c);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        print_error("    mat_elementwise_div expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_mat_destroy(&a);
        lmmc_mat_destroy(&b);
        lmmc_mat_destroy(&c);
        fail();
    }

    lmmc_mat_destroy(&a);
    lmmc_mat_destroy(&b);
    lmmc_mat_destroy(&c);
}

/* Unit test: aliasing — hadamard(a, b, a) overwrites input a correctly */
static void test_unit_aliasing(void **state) {
    (void)state;
    lmmc_vec_t a, b;
    lmmc_status_t st;
    double eps = 1e-15;

    st = lmmc_vec_create(4, &a);
    if (st != LMMC_STATUS_OK) {
        print_error("    vec_create(a) failed\n");
        fail();
    }
    st = lmmc_vec_create(4, &b);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&a);
        print_error("    vec_create(b) failed\n");
        fail();
    }

    /* Set known values */
    a.data[0] = 2.0;
    a.data[1] = 3.0;
    a.data[2] = 4.0;
    a.data[3] = 5.0;
    b.data[0] = 1.5;
    b.data[1] = 2.5;
    b.data[2] = 3.5;
    b.data[3] = 4.5;

    /* Expected results: a[i] * b[i] */
    double expected[4];
    expected[0] = 2.0 * 1.5;
    expected[1] = 3.0 * 2.5;
    expected[2] = 4.0 * 3.5;
    expected[3] = 5.0 * 4.5;

    /* hadamard(a, b, a) — output overwrites input a */
    st = lmmc_vec_hadamard(&a, &b, &a);
    if (st != LMMC_STATUS_OK) {
        print_error("    vec_hadamard aliasing failed, status=%d\n", (int)st);
        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&b);
        fail();
    }

    /* Verify results */
    {
        size_t j;
        for (j = 0; j < 4; j++) {
            if (!lmmc_test_nearly_equal(a.data[j], expected[j], eps)) {
                print_error("    Aliasing result mismatch at elem %" PRIuMAX ": got %g, expected %g\n", (uintmax_t)(j), a.data[j], expected[j]);
                lmmc_vec_destroy(&a);
                lmmc_vec_destroy(&b);
                fail();
            }
        }
    }

    lmmc_vec_destroy(&a);
    lmmc_vec_destroy(&b);
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
        cmocka_unit_test_setup_teardown(test_property7_vec_hadamard_div_roundtrip, setup_property, teardown_property),
        cmocka_unit_test_setup_teardown(test_property8_mat_hadamard_div_roundtrip, setup_property, teardown_property),
        cmocka_unit_test_setup_teardown(test_property9_vec_cmp_complementarity, setup_property, teardown_property),
        cmocka_unit_test(test_unit_zero_divisor_vec),
        cmocka_unit_test(test_unit_zero_divisor_mat),
        cmocka_unit_test(test_unit_dimension_mismatch_vec),
        cmocka_unit_test(test_unit_dimension_mismatch_mat),
        cmocka_unit_test(test_unit_aliasing),
    };
    return cmocka_run_group_tests(tests, setup, teardown);
}
