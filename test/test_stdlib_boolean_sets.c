#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"

static int bool_set_has(const lmmc_std_bool_set_t *set, int value) {
    int contains = 0;
    return lmmc_std_bool_set_contains(set, value, &contains) ==
               LMMC_STATUS_OK &&
           contains;
}

struct test_fixture {
    lmmc_std_bool_set_t bool_set_a;
    lmmc_std_bool_set_t bool_set_empty;
    lmmc_std_bool_set_t bool_set_b;
    lmmc_std_bool_set_t bool_set_union;
    lmmc_std_bool_set_t bool_set_intersection;
    lmmc_std_bool_set_t bool_set_difference;
    lmmc_std_bool_set_t bool_set_xor;
};

static int setup(void **state) {
    *state = test_calloc(1, sizeof(struct test_fixture));
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_std_bool_set_destroy(&fixture->bool_set_xor);
    lmmc_std_bool_set_destroy(&fixture->bool_set_difference);
    lmmc_std_bool_set_destroy(&fixture->bool_set_intersection);
    lmmc_std_bool_set_destroy(&fixture->bool_set_union);
    lmmc_std_bool_set_destroy(&fixture->bool_set_b);
    lmmc_std_bool_set_destroy(&fixture->bool_set_empty);
    lmmc_std_bool_set_destroy(&fixture->bool_set_a);
    test_free(fixture);
    return 0;
}

static void test_construction_membership(void **state) {
    struct test_fixture *fixture = *state;
    int bool_set_a_values[] = {1, 1, 0};
    int set_subset = 0;

    assert_true(lmmc_std_bool_set_make(bool_set_a_values, 3, &fixture->bool_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_bool_set_make(NULL, 0, &fixture->bool_set_empty) == LMMC_STATUS_OK);
    assert_true(fixture->bool_set_a.size == 2);
    assert_true(bool_set_has(&fixture->bool_set_a, 1));
    assert_true(bool_set_has(&fixture->bool_set_a, 0));
    assert_true(fixture->bool_set_empty.size == 0);
    assert_true(lmmc_std_bool_set_subset(&fixture->bool_set_empty, &fixture->bool_set_a, &set_subset) == LMMC_STATUS_OK);
    assert_true(set_subset);

    lmmc_std_bool_set_destroy(&fixture->bool_set_empty);
    lmmc_std_bool_set_destroy(&fixture->bool_set_a);
}

static void test_union_intersection_subset(void **state) {
    struct test_fixture *fixture = *state;
    int bool_set_a_values[] = {1, 1, 0};
    int bool_set_b_values[] = {0};
    int set_subset = 0;

    assert_true(lmmc_std_bool_set_make(bool_set_a_values, 3, &fixture->bool_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_bool_set_make(bool_set_b_values, 1, &fixture->bool_set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_bool_set_union(&fixture->bool_set_a, &fixture->bool_set_b, &fixture->bool_set_union) == LMMC_STATUS_OK);
    assert_true(fixture->bool_set_union.size == 2);
    assert_true(bool_set_has(&fixture->bool_set_union, 1));
    assert_true(bool_set_has(&fixture->bool_set_union, 0));
    assert_true(lmmc_std_bool_set_intersection(&fixture->bool_set_a, &fixture->bool_set_b, &fixture->bool_set_intersection) ==
                LMMC_STATUS_OK);
    assert_true(fixture->bool_set_intersection.size == 1);
    assert_true(bool_set_has(&fixture->bool_set_intersection, 0));
    assert_true(lmmc_std_bool_set_subset(&fixture->bool_set_intersection, &fixture->bool_set_union, &set_subset) ==
                LMMC_STATUS_OK);
    assert_true(set_subset);
    assert_true(lmmc_std_bool_set_subset(&fixture->bool_set_union, &fixture->bool_set_intersection, &set_subset) ==
                LMMC_STATUS_OK);
    assert_true(!set_subset);

    lmmc_std_bool_set_destroy(&fixture->bool_set_intersection);
    lmmc_std_bool_set_destroy(&fixture->bool_set_union);
    lmmc_std_bool_set_destroy(&fixture->bool_set_b);
    lmmc_std_bool_set_destroy(&fixture->bool_set_a);
}

static void test_difference_symmetric_difference(void **state) {
    struct test_fixture *fixture = *state;
    int bool_set_a_values[] = {1, 1, 0};
    int bool_set_b_values[] = {0};

    assert_true(lmmc_std_bool_set_make(bool_set_a_values, 3, &fixture->bool_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_bool_set_make(bool_set_b_values, 1, &fixture->bool_set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_bool_set_difference(&fixture->bool_set_a, &fixture->bool_set_b, &fixture->bool_set_difference) ==
                LMMC_STATUS_OK);
    assert_true(fixture->bool_set_difference.size == 1);
    assert_true(bool_set_has(&fixture->bool_set_difference, 1));
    assert_true(lmmc_std_bool_set_symmetric_difference(&fixture->bool_set_a, &fixture->bool_set_b, &fixture->bool_set_xor) ==
                LMMC_STATUS_OK);
    assert_true(fixture->bool_set_xor.size == 1);
    assert_true(bool_set_has(&fixture->bool_set_xor, 1));

    lmmc_std_bool_set_destroy(&fixture->bool_set_xor);
    lmmc_std_bool_set_destroy(&fixture->bool_set_difference);
    lmmc_std_bool_set_destroy(&fixture->bool_set_b);
    lmmc_std_bool_set_destroy(&fixture->bool_set_a);
}

static void test_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    int bool_set_a_values[] = {1, 1, 0};
    int bool_set_b_values[] = {0};
    int bool_set_bad_values[] = {1, 2};
    int set_contains = 0;
    int set_subset = 0;

    assert_true(lmmc_std_bool_set_make(bool_set_a_values, 3, &fixture->bool_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_bool_set_make(bool_set_b_values, 1, &fixture->bool_set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_bool_set_make(NULL, 1, &fixture->bool_set_xor) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_bool_set_make(bool_set_bad_values, 2, &fixture->bool_set_xor) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_bool_set_contains(NULL, 1, &set_contains) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_bool_set_contains(&fixture->bool_set_a, 2, &set_contains) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_bool_set_contains(&fixture->bool_set_a, 1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_bool_set_subset(NULL, &fixture->bool_set_b, &set_subset) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_bool_set_subset(&fixture->bool_set_a, &fixture->bool_set_b, NULL) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_bool_set_union(NULL, &fixture->bool_set_b, &fixture->bool_set_xor) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_bool_set_union(&fixture->bool_set_a, &fixture->bool_set_b, &fixture->bool_set_a) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_std_bool_set_destroy(&fixture->bool_set_xor);
    lmmc_std_bool_set_destroy(&fixture->bool_set_b);
    lmmc_std_bool_set_destroy(&fixture->bool_set_a);
    lmmc_std_bool_set_destroy(NULL);
}

static void test_key_equality_hashing(void **state) {
    (void)state;
    uint64_t bool_hash = 0;
    uint64_t bool_hash_same = 0;
    uint64_t bool_hash_other = 0;
    int bool_equal = 0;

    assert_true(lmmc_std_bool_equal(1, 1, &bool_equal) == LMMC_STATUS_OK);
    assert_true(bool_equal);
    assert_true(lmmc_std_bool_hash(1, &bool_hash) == LMMC_STATUS_OK);
    assert_true(lmmc_std_bool_hash(1, &bool_hash_same) == LMMC_STATUS_OK);
    assert_true(bool_hash == bool_hash_same);
    assert_true(lmmc_std_bool_equal(1, 0, &bool_equal) == LMMC_STATUS_OK);
    assert_true(!bool_equal);
    assert_true(lmmc_std_bool_hash(0, &bool_hash_other) == LMMC_STATUS_OK);
    assert_true(bool_hash != bool_hash_other);
}

static void test_key_invalid_inputs(void **state) {
    (void)state;
    uint64_t bool_hash = 0;
    int bool_equal = 0;

    assert_true(lmmc_std_bool_equal(2, 1, &bool_equal) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_bool_equal(1, 1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_bool_hash(2, &bool_hash) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_bool_hash(1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_construction_membership, setup, teardown),
        cmocka_unit_test_setup_teardown(test_union_intersection_subset, setup, teardown),
        cmocka_unit_test_setup_teardown(test_difference_symmetric_difference, setup, teardown),
        cmocka_unit_test_setup_teardown(test_invalid_arguments, setup, teardown),
        cmocka_unit_test(test_key_equality_hashing),
        cmocka_unit_test(test_key_invalid_inputs),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
