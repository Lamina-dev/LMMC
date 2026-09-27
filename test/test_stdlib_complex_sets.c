#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"
#include "internal_test_hooks.h"

static int complex_set_has(const lmmc_std_complex_set_t *set,
                           lmmc_real_t real,
                           lmmc_real_t imag) {
    int contains = 0;
    lmmc_complex_t value = {real, imag};
    return lmmc_std_complex_set_contains(set, &value, &contains) ==
               LMMC_STATUS_OK &&
           contains;
}

struct test_fixture {
    lmmc_std_complex_set_t complex_set_a;
    lmmc_std_complex_set_t complex_set_empty;
    lmmc_std_complex_set_t complex_set_b;
    lmmc_std_complex_set_t complex_set_union;
    lmmc_std_complex_set_t complex_set_intersection;
    lmmc_std_complex_set_t complex_set_difference;
    lmmc_std_complex_set_t complex_set_xor;
    lmmc_std_complex_set_t result;
};

static int setup(void **state) {
    *state = test_calloc(1, sizeof(struct test_fixture));
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_memory_fail_reset_for_test();
    lmmc_std_complex_set_destroy(&fixture->result);
    lmmc_std_complex_set_destroy(&fixture->complex_set_xor);
    lmmc_std_complex_set_destroy(&fixture->complex_set_difference);
    lmmc_std_complex_set_destroy(&fixture->complex_set_intersection);
    lmmc_std_complex_set_destroy(&fixture->complex_set_union);
    lmmc_std_complex_set_destroy(&fixture->complex_set_b);
    lmmc_std_complex_set_destroy(&fixture->complex_set_empty);
    lmmc_std_complex_set_destroy(&fixture->complex_set_a);
    test_free(fixture);
    return 0;
}

static void test_construction_membership(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_complex_t complex_set_a_values[] = {
        {1, 2}, {1, 2}, {-0.0, 0.0}, {3, 4}};
    int set_subset = 0;

    assert_true(lmmc_std_complex_set_make(complex_set_a_values, 4, &fixture->complex_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_complex_set_make(NULL, 0, &fixture->complex_set_empty) == LMMC_STATUS_OK);
    assert_true(fixture->complex_set_a.size == 3);
    assert_true(complex_set_has(&fixture->complex_set_a, 1, 2));
    assert_true(complex_set_has(&fixture->complex_set_a, 0, 0));
    assert_true(complex_set_has(&fixture->complex_set_a, 3, 4));
    assert_true(!(complex_set_has(&fixture->complex_set_a, 5, 6)));
    assert_true(fixture->complex_set_empty.size == 0);
    assert_true(lmmc_std_complex_set_subset(&fixture->complex_set_empty, &fixture->complex_set_a, &set_subset) ==
                LMMC_STATUS_OK);
    assert_true(set_subset);

    lmmc_std_complex_set_destroy(&fixture->complex_set_empty);
    lmmc_std_complex_set_destroy(&fixture->complex_set_a);
}

static void test_union_intersection_subset(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_complex_t complex_set_a_values[] = {
        {1, 2}, {1, 2}, {-0.0, 0.0}, {3, 4}};
    lmmc_complex_t complex_set_b_values[] = {{3, 4}, {5, 6}, {0.0, -0.0}};
    int set_subset = 0;

    assert_true(lmmc_std_complex_set_make(complex_set_a_values, 4, &fixture->complex_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_complex_set_make(complex_set_b_values, 3, &fixture->complex_set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_complex_set_union(&fixture->complex_set_a, &fixture->complex_set_b, &fixture->complex_set_union) ==
                LMMC_STATUS_OK);
    assert_true(fixture->complex_set_union.size == 4);
    assert_true(complex_set_has(&fixture->complex_set_union, 1, 2));
    assert_true(complex_set_has(&fixture->complex_set_union, 3, 4));
    assert_true(complex_set_has(&fixture->complex_set_union, 5, 6));
    assert_true(complex_set_has(&fixture->complex_set_union, 0, 0));
    assert_true(lmmc_std_complex_set_intersection(
                    &fixture->complex_set_a,
                    &fixture->complex_set_b,
                    &fixture->complex_set_intersection) ==
                LMMC_STATUS_OK);
    assert_true(fixture->complex_set_intersection.size == 2);
    assert_true(complex_set_has(&fixture->complex_set_intersection, 3, 4));
    assert_true(complex_set_has(&fixture->complex_set_intersection, 0, 0));
    assert_true(lmmc_std_complex_set_subset(
                    &fixture->complex_set_intersection,
                    &fixture->complex_set_union,
                    &set_subset) ==
                LMMC_STATUS_OK);
    assert_true(set_subset);
    assert_true(lmmc_std_complex_set_subset(
                    &fixture->complex_set_union,
                    &fixture->complex_set_intersection,
                    &set_subset) ==
                LMMC_STATUS_OK);
    assert_true(!set_subset);

    lmmc_std_complex_set_destroy(&fixture->complex_set_intersection);
    lmmc_std_complex_set_destroy(&fixture->complex_set_union);
    lmmc_std_complex_set_destroy(&fixture->complex_set_b);
    lmmc_std_complex_set_destroy(&fixture->complex_set_a);
}

static void test_difference_symmetric_difference(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_complex_t complex_set_a_values[] = {
        {1, 2}, {1, 2}, {-0.0, 0.0}, {3, 4}};
    lmmc_complex_t complex_set_b_values[] = {{3, 4}, {5, 6}, {0.0, -0.0}};

    assert_true(lmmc_std_complex_set_make(complex_set_a_values, 4, &fixture->complex_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_complex_set_make(complex_set_b_values, 3, &fixture->complex_set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_complex_set_difference(
                    &fixture->complex_set_a,
                    &fixture->complex_set_b,
                    &fixture->complex_set_difference) ==
                LMMC_STATUS_OK);
    assert_true(fixture->complex_set_difference.size == 1);
    assert_true(complex_set_has(&fixture->complex_set_difference, 1, 2));
    assert_true(lmmc_std_complex_set_symmetric_difference(
                    &fixture->complex_set_a,
                    &fixture->complex_set_b,
                    &fixture->complex_set_xor) ==
                LMMC_STATUS_OK);
    assert_true(fixture->complex_set_xor.size == 2);
    assert_true(complex_set_has(&fixture->complex_set_xor, 1, 2));
    assert_true(complex_set_has(&fixture->complex_set_xor, 5, 6));

    lmmc_std_complex_set_destroy(&fixture->complex_set_xor);
    lmmc_std_complex_set_destroy(&fixture->complex_set_difference);
    lmmc_std_complex_set_destroy(&fixture->complex_set_b);
    lmmc_std_complex_set_destroy(&fixture->complex_set_a);
}

static void test_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_complex_t complex_set_a_values[] = {
        {1, 2}, {1, 2}, {-0.0, 0.0}, {3, 4}};
    lmmc_complex_t complex_set_b_values[] = {{3, 4}, {5, 6}, {0.0, -0.0}};
    lmmc_complex_t complex_set_bad_values[] = {{1, 2}, {NAN, 0}};
    lmmc_complex_t z;
    int set_contains = 0;
    int set_subset = 0;

    assert_true(lmmc_std_complex_set_make(complex_set_a_values, 4, &fixture->complex_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_complex_set_make(complex_set_b_values, 3, &fixture->complex_set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_math_complex(3, 4, &z) == LMMC_STATUS_OK);
    assert_true(lmmc_std_complex_set_make(NULL, 1, &fixture->complex_set_xor) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_complex_set_make(complex_set_bad_values, 2, &fixture->complex_set_xor) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_complex_set_contains(NULL, &z, &set_contains) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_complex_set_contains(&fixture->complex_set_a, NULL, &set_contains) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_complex_set_contains(&fixture->complex_set_a, &z, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_complex_set_subset(NULL, &fixture->complex_set_b, &set_subset) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_complex_set_subset(&fixture->complex_set_a, &fixture->complex_set_b, NULL) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_complex_set_union(NULL, &fixture->complex_set_b, &fixture->complex_set_xor) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_complex_set_union(&fixture->complex_set_a, &fixture->complex_set_b, &fixture->complex_set_a) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_std_complex_set_destroy(&fixture->complex_set_xor);
    lmmc_std_complex_set_destroy(&fixture->complex_set_b);
    lmmc_std_complex_set_destroy(&fixture->complex_set_a);
    lmmc_std_complex_set_destroy(NULL);
}

static void test_allocation_before_nonfinite(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_complex_t values[] = {{NAN, 0}};
    lmmc_status_t status;

    lmmc_memory_fail_after_for_test(0);
    status = lmmc_std_complex_set_make(values, 1, &fixture->result);
    lmmc_memory_fail_reset_for_test();

    assert_true(status == LMMC_STATUS_ALLOCATION_FAILED);
    lmmc_std_complex_set_destroy(&fixture->result);
}

static void test_complex_arithmetic(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_complex_t z;
    lmmc_complex_t w;

    assert_true(lmmc_std_math_complex(3, 4, &z) == LMMC_STATUS_OK);

    assert_true(lmmc_std_math_real(&z, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 3));

    assert_true(lmmc_std_math_imag(&z, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 4));

    assert_true(lmmc_std_math_conj(&z, &w) == LMMC_STATUS_OK);
    assert_true(close_real(w.real, 3));
    assert_true(close_real(w.imag, -4));

    assert_true(lmmc_std_math_complex_abs(&z, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 5));
}

static void test_key_equality_hashing(void **state) {
    (void)state;
    lmmc_complex_t z;
    lmmc_complex_t w;
    uint64_t complex_hash = 0;
    uint64_t complex_hash_same = 0;
    uint64_t complex_hash_other = 0;
    int complex_equal = 0;

    assert_true(lmmc_std_math_complex(3, 4, &z) == LMMC_STATUS_OK);
    assert_true(lmmc_std_math_complex(3, 4, &w) == LMMC_STATUS_OK);
    assert_true(lmmc_std_math_complex_equal(&z, &w, &complex_equal) == LMMC_STATUS_OK);
    assert_true(complex_equal);
    assert_true(lmmc_std_math_complex_hash(&z, &complex_hash) == LMMC_STATUS_OK);
    assert_true(lmmc_std_math_complex_hash(&w, &complex_hash_same) == LMMC_STATUS_OK);
    assert_true(complex_hash == complex_hash_same);
    assert_true(lmmc_std_math_complex(3, 5, &w) == LMMC_STATUS_OK);
    assert_true(lmmc_std_math_complex_equal(&z, &w, &complex_equal) == LMMC_STATUS_OK);
    assert_true(!complex_equal);
    assert_true(lmmc_std_math_complex_hash(&w, &complex_hash_other) == LMMC_STATUS_OK);
    assert_true(complex_hash != complex_hash_other);
}

static void test_signed_zero_keys(void **state) {
    (void)state;
    lmmc_complex_t z;
    lmmc_complex_t w;
    uint64_t complex_hash = 0;
    uint64_t complex_hash_same = 0;
    int complex_equal = 0;

    z.real = 0.0;
    z.imag = -0.0;
    w.real = -0.0;
    w.imag = 0.0;
    assert_true(lmmc_std_math_complex_equal(&z, &w, &complex_equal) == LMMC_STATUS_OK);
    assert_true(complex_equal);
    assert_true(lmmc_std_math_complex_hash(&z, &complex_hash) == LMMC_STATUS_OK);
    assert_true(lmmc_std_math_complex_hash(&w, &complex_hash_same) == LMMC_STATUS_OK);
    assert_true(complex_hash == complex_hash_same);
}

static void test_key_invalid_arguments(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_complex_t z;
    lmmc_complex_t w;
    uint64_t complex_hash = 0;
    int complex_equal = 0;

    assert_true(lmmc_std_math_complex(3, 4, &z) == LMMC_STATUS_OK);
    assert_true(lmmc_std_math_complex(3, 4, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_real(NULL, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_real(&z, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_imag(NULL, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_imag(&z, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_conj(NULL, &w) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_conj(&z, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_complex_abs(NULL, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_complex_abs(&z, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_complex_equal(NULL, &z, &complex_equal) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_complex_equal(&z, NULL, &complex_equal) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_complex_equal(&z, &z, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_complex_hash(NULL, &complex_hash) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_complex_hash(&z, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_key_nonfinite_inputs(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_complex_t z;
    lmmc_complex_t w;
    uint64_t complex_hash = 0;
    int complex_equal = 0;

    z.real = NAN;
    z.imag = 1;
    assert_true(lmmc_std_math_complex(NAN, 1, &w) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_real(&z, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_imag(&z, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_conj(&z, &w) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_complex_abs(&z, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_complex_equal(&z, &z, &complex_equal) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_complex_hash(&z, &complex_hash) == LMMC_STATUS_NUMERICAL_FAILURE);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_construction_membership, setup, teardown),
        cmocka_unit_test_setup_teardown(test_union_intersection_subset, setup, teardown),
        cmocka_unit_test_setup_teardown(test_difference_symmetric_difference, setup, teardown),
        cmocka_unit_test_setup_teardown(test_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_allocation_before_nonfinite, setup, teardown),
        cmocka_unit_test(test_complex_arithmetic),
        cmocka_unit_test(test_key_equality_hashing),
        cmocka_unit_test(test_signed_zero_keys),
        cmocka_unit_test(test_key_invalid_arguments),
        cmocka_unit_test(test_key_nonfinite_inputs),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
