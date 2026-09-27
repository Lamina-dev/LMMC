#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"
#include "internal_test_hooks.h"

static int num_set_has(const lmmc_std_num_set_t *set, lmmc_real_t value) {
    int contains = 0;
    return lmmc_std_num_set_contains(set, value, &contains) ==
               LMMC_STATUS_OK &&
           contains;
}

struct test_fixture {
    lmmc_std_num_set_t set_a;
    lmmc_std_num_set_t set_empty;
    lmmc_std_num_set_t set_b;
    lmmc_std_num_set_t set_union;
    lmmc_std_num_set_t set_intersection;
    lmmc_std_num_set_t set_difference;
    lmmc_std_num_set_t set_xor;
    lmmc_std_num_set_t result;
};

static int setup(void **state) {
    *state = test_calloc(1, sizeof(struct test_fixture));
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_memory_fail_reset_for_test();
    lmmc_std_num_set_destroy(&fixture->result);
    lmmc_std_num_set_destroy(&fixture->set_xor);
    lmmc_std_num_set_destroy(&fixture->set_difference);
    lmmc_std_num_set_destroy(&fixture->set_intersection);
    lmmc_std_num_set_destroy(&fixture->set_union);
    lmmc_std_num_set_destroy(&fixture->set_b);
    lmmc_std_num_set_destroy(&fixture->set_empty);
    lmmc_std_num_set_destroy(&fixture->set_a);
    test_free(fixture);
    return 0;
}

static void test_construction_membership(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t set_a_values[] = {1, 2, 2, -0.0, 0.0};
    int set_contains = 0;
    int set_subset = 0;

    assert_true(lmmc_std_num_set_make(set_a_values, 5, &fixture->set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_num_set_make(NULL, 0, &fixture->set_empty) == LMMC_STATUS_OK);
    assert_true(fixture->set_a.size == 3);
    assert_true(num_set_has(&fixture->set_a, 1));
    assert_true(num_set_has(&fixture->set_a, 2));
    assert_true(num_set_has(&fixture->set_a, 0));
    assert_true(!(num_set_has(&fixture->set_a, 3)));
    assert_true(lmmc_std_num_set_contains(&fixture->set_a, -0.0, &set_contains) == LMMC_STATUS_OK);
    assert_true(set_contains);
    assert_true(fixture->set_empty.size == 0);
    assert_true(lmmc_std_num_set_subset(&fixture->set_empty, &fixture->set_a, &set_subset) == LMMC_STATUS_OK);
    assert_true(set_subset);

    lmmc_std_num_set_destroy(&fixture->set_empty);
    lmmc_std_num_set_destroy(&fixture->set_a);
}

static void test_union_intersection_subset(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t set_a_values[] = {1, 2, 2, -0.0, 0.0};
    lmmc_real_t set_b_values[] = {2, 3, 0};
    int set_subset = 0;

    assert_true(lmmc_std_num_set_make(set_a_values, 5, &fixture->set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_num_set_make(set_b_values, 3, &fixture->set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_num_set_union(&fixture->set_a, &fixture->set_b, &fixture->set_union) == LMMC_STATUS_OK);
    assert_true(fixture->set_union.size == 4);
    assert_true(num_set_has(&fixture->set_union, 1));
    assert_true(num_set_has(&fixture->set_union, 2));
    assert_true(num_set_has(&fixture->set_union, 3));
    assert_true(num_set_has(&fixture->set_union, 0));
    assert_true(lmmc_std_num_set_intersection(&fixture->set_a, &fixture->set_b, &fixture->set_intersection) == LMMC_STATUS_OK);
    assert_true(fixture->set_intersection.size == 2);
    assert_true(num_set_has(&fixture->set_intersection, 2));
    assert_true(num_set_has(&fixture->set_intersection, 0));
    assert_true(lmmc_std_num_set_subset(&fixture->set_intersection, &fixture->set_union, &set_subset) == LMMC_STATUS_OK);
    assert_true(set_subset);
    assert_true(lmmc_std_num_set_subset(&fixture->set_union, &fixture->set_intersection, &set_subset) == LMMC_STATUS_OK);
    assert_true(!set_subset);

    lmmc_std_num_set_destroy(&fixture->set_intersection);
    lmmc_std_num_set_destroy(&fixture->set_union);
    lmmc_std_num_set_destroy(&fixture->set_b);
    lmmc_std_num_set_destroy(&fixture->set_a);
}

static void test_difference_symmetric_difference(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t set_a_values[] = {1, 2, 2, -0.0, 0.0};
    lmmc_real_t set_b_values[] = {2, 3, 0};

    assert_true(lmmc_std_num_set_make(set_a_values, 5, &fixture->set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_num_set_make(set_b_values, 3, &fixture->set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_num_set_difference(&fixture->set_a, &fixture->set_b, &fixture->set_difference) == LMMC_STATUS_OK);
    assert_true(fixture->set_difference.size == 1);
    assert_true(num_set_has(&fixture->set_difference, 1));
    assert_true(lmmc_std_num_set_symmetric_difference(&fixture->set_a, &fixture->set_b, &fixture->set_xor) == LMMC_STATUS_OK);
    assert_true(fixture->set_xor.size == 2);
    assert_true(num_set_has(&fixture->set_xor, 1));
    assert_true(num_set_has(&fixture->set_xor, 3));

    lmmc_std_num_set_destroy(&fixture->set_xor);
    lmmc_std_num_set_destroy(&fixture->set_difference);
    lmmc_std_num_set_destroy(&fixture->set_b);
    lmmc_std_num_set_destroy(&fixture->set_a);
}

static void test_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t set_a_values[] = {1, 2, 2, -0.0, 0.0};
    lmmc_real_t set_b_values[] = {2, 3, 0};
    lmmc_real_t set_bad_values[] = {1, NAN};
    int set_contains = 0;
    int set_subset = 0;

    assert_true(lmmc_std_num_set_make(set_a_values, 5, &fixture->set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_num_set_make(set_b_values, 3, &fixture->set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_num_set_symmetric_difference(&fixture->set_a, &fixture->set_b, &fixture->set_xor) == LMMC_STATUS_OK);
    assert_true(lmmc_std_num_set_make(NULL, 1, &fixture->set_xor) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_num_set_make(set_bad_values, 2, &fixture->set_xor) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_num_set_contains(NULL, 1, &set_contains) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_num_set_contains(&fixture->set_a, NAN, &set_contains) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_num_set_contains(&fixture->set_a, 1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_num_set_subset(NULL, &fixture->set_b, &set_subset) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_num_set_subset(&fixture->set_a, &fixture->set_b, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_num_set_union(NULL, &fixture->set_b, &fixture->set_xor) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_num_set_union(&fixture->set_a, &fixture->set_b, &fixture->set_a) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_std_num_set_destroy(&fixture->set_xor);
    lmmc_std_num_set_destroy(&fixture->set_b);
    lmmc_std_num_set_destroy(&fixture->set_a);
    lmmc_std_num_set_destroy(NULL);
}

static void test_nonfinite_before_allocation(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t invalid_values[] = {NAN};
    lmmc_real_t finite_values[] = {1};
    lmmc_status_t invalid_status;
    lmmc_status_t allocation_status;

    lmmc_memory_fail_after_for_test(0);
    invalid_status = lmmc_std_num_set_make(invalid_values, 1, &fixture->result);
    allocation_status = lmmc_std_num_set_make(finite_values, 1, &fixture->result);
    lmmc_memory_fail_reset_for_test();

    assert_true(invalid_status == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(allocation_status == LMMC_STATUS_ALLOCATION_FAILED);
    lmmc_std_num_set_destroy(&fixture->result);
}

static void test_key_equality_hashing(void **state) {
    (void)state;
    uint64_t num_hash = 0;
    uint64_t num_hash_same = 0;
    uint64_t num_hash_other = 0;
    int num_equal = 0;

    assert_true(lmmc_std_num_equal(1.5, 1.5, &num_equal) == LMMC_STATUS_OK);
    assert_true(num_equal);
    assert_true(lmmc_std_num_hash(1.5, &num_hash) == LMMC_STATUS_OK);
    assert_true(lmmc_std_num_hash(1.5, &num_hash_same) == LMMC_STATUS_OK);
    assert_true(num_hash == num_hash_same);
    assert_true(lmmc_std_num_equal(1.5, 2.5, &num_equal) == LMMC_STATUS_OK);
    assert_true(!num_equal);
    assert_true(lmmc_std_num_hash(2.5, &num_hash_other) == LMMC_STATUS_OK);
    assert_true(num_hash != num_hash_other);
    assert_true(lmmc_std_num_equal(0.0, -0.0, &num_equal) == LMMC_STATUS_OK);
    assert_true(num_equal);
    assert_true(lmmc_std_num_hash(0.0, &num_hash) == LMMC_STATUS_OK);
    assert_true(lmmc_std_num_hash(-0.0, &num_hash_same) == LMMC_STATUS_OK);
    assert_true(num_hash == num_hash_same);
}

static void test_key_invalid_inputs(void **state) {
    (void)state;
    uint64_t num_hash = 0;
    int num_equal = 0;

    assert_true(lmmc_std_num_equal(1.0, 1.0, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_num_hash(1.0, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_num_equal(NAN, 1.0, &num_equal) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_num_equal(1.0, INFINITY, &num_equal) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_num_hash(NAN, &num_hash) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_num_hash(INFINITY, &num_hash) == LMMC_STATUS_NUMERICAL_FAILURE);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_construction_membership, setup, teardown),
        cmocka_unit_test_setup_teardown(test_union_intersection_subset, setup, teardown),
        cmocka_unit_test_setup_teardown(test_difference_symmetric_difference, setup, teardown),
        cmocka_unit_test_setup_teardown(test_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_nonfinite_before_allocation, setup, teardown),
        cmocka_unit_test(test_key_equality_hashing),
        cmocka_unit_test(test_key_invalid_inputs),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
