#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"
#include "internal_test_hooks.h"

static int text_set_has(const lmmc_std_text_set_t *set, const char *value) {
    int contains = 0;
    return lmmc_std_text_set_contains(set, value, &contains) ==
               LMMC_STATUS_OK &&
           contains;
}

struct test_fixture {
    lmmc_std_text_set_t text_set_a;
    lmmc_std_text_set_t text_set_empty;
    lmmc_std_text_set_t text_set_b;
    lmmc_std_text_set_t text_set_union;
    lmmc_std_text_set_t text_set_intersection;
    lmmc_std_text_set_t text_set_difference;
    lmmc_std_text_set_t text_set_xor;
    lmmc_std_text_set_t result;
};

static int setup(void **state) {
    *state = test_calloc(1, sizeof(struct test_fixture));
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_memory_fail_reset_for_test();
    lmmc_std_text_set_destroy(&fixture->result);
    lmmc_std_text_set_destroy(&fixture->text_set_xor);
    lmmc_std_text_set_destroy(&fixture->text_set_difference);
    lmmc_std_text_set_destroy(&fixture->text_set_intersection);
    lmmc_std_text_set_destroy(&fixture->text_set_union);
    lmmc_std_text_set_destroy(&fixture->text_set_b);
    lmmc_std_text_set_destroy(&fixture->text_set_empty);
    lmmc_std_text_set_destroy(&fixture->text_set_a);
    test_free(fixture);
    return 0;
}

static void test_construction_membership(void **state) {
    struct test_fixture *fixture = *state;
    const char *text_set_a_values[] = {"alpha", "beta", "alpha"};
    int set_subset = 0;

    assert_true(lmmc_std_text_set_make(text_set_a_values, 3, &fixture->text_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_text_set_make(NULL, 0, &fixture->text_set_empty) == LMMC_STATUS_OK);
    assert_true(fixture->text_set_a.size == 2);
    assert_true(text_set_has(&fixture->text_set_a, "alpha"));
    assert_true(text_set_has(&fixture->text_set_a, "beta"));
    assert_true(!(text_set_has(&fixture->text_set_a, "gamma")));
    assert_true(fixture->text_set_empty.size == 0);
    assert_true(lmmc_std_text_set_subset(&fixture->text_set_empty, &fixture->text_set_a, &set_subset) == LMMC_STATUS_OK);
    assert_true(set_subset);

    lmmc_std_text_set_destroy(&fixture->text_set_empty);
    lmmc_std_text_set_destroy(&fixture->text_set_a);
}

static void test_union_intersection_subset(void **state) {
    struct test_fixture *fixture = *state;
    const char *text_set_a_values[] = {"alpha", "beta", "alpha"};
    const char *text_set_b_values[] = {"beta", "gamma"};
    int set_subset = 0;

    assert_true(lmmc_std_text_set_make(text_set_a_values, 3, &fixture->text_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_text_set_make(text_set_b_values, 2, &fixture->text_set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_text_set_union(&fixture->text_set_a, &fixture->text_set_b, &fixture->text_set_union) == LMMC_STATUS_OK);
    assert_true(fixture->text_set_union.size == 3);
    assert_true(text_set_has(&fixture->text_set_union, "alpha"));
    assert_true(text_set_has(&fixture->text_set_union, "beta"));
    assert_true(text_set_has(&fixture->text_set_union, "gamma"));
    assert_true(lmmc_std_text_set_intersection(&fixture->text_set_a, &fixture->text_set_b, &fixture->text_set_intersection) ==
                LMMC_STATUS_OK);
    assert_true(fixture->text_set_intersection.size == 1);
    assert_true(text_set_has(&fixture->text_set_intersection, "beta"));
    assert_true(lmmc_std_text_set_subset(&fixture->text_set_intersection, &fixture->text_set_union, &set_subset) ==
                LMMC_STATUS_OK);
    assert_true(set_subset);
    assert_true(lmmc_std_text_set_subset(&fixture->text_set_union, &fixture->text_set_intersection, &set_subset) ==
                LMMC_STATUS_OK);
    assert_true(!set_subset);

    lmmc_std_text_set_destroy(&fixture->text_set_intersection);
    lmmc_std_text_set_destroy(&fixture->text_set_union);
    lmmc_std_text_set_destroy(&fixture->text_set_b);
    lmmc_std_text_set_destroy(&fixture->text_set_a);
}

static void test_difference_symmetric_difference(void **state) {
    struct test_fixture *fixture = *state;
    const char *text_set_a_values[] = {"alpha", "beta", "alpha"};
    const char *text_set_b_values[] = {"beta", "gamma"};

    assert_true(lmmc_std_text_set_make(text_set_a_values, 3, &fixture->text_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_text_set_make(text_set_b_values, 2, &fixture->text_set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_text_set_difference(&fixture->text_set_a, &fixture->text_set_b, &fixture->text_set_difference) ==
                LMMC_STATUS_OK);
    assert_true(fixture->text_set_difference.size == 1);
    assert_true(text_set_has(&fixture->text_set_difference, "alpha"));
    assert_true(lmmc_std_text_set_symmetric_difference(&fixture->text_set_a, &fixture->text_set_b, &fixture->text_set_xor) ==
                LMMC_STATUS_OK);
    assert_true(fixture->text_set_xor.size == 2);
    assert_true(text_set_has(&fixture->text_set_xor, "alpha"));
    assert_true(text_set_has(&fixture->text_set_xor, "gamma"));

    lmmc_std_text_set_destroy(&fixture->text_set_xor);
    lmmc_std_text_set_destroy(&fixture->text_set_difference);
    lmmc_std_text_set_destroy(&fixture->text_set_b);
    lmmc_std_text_set_destroy(&fixture->text_set_a);
}

static void test_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    const char *text_set_a_values[] = {"alpha", "beta", "alpha"};
    const char *text_set_b_values[] = {"beta", "gamma"};
    const char *text_set_bad_values[] = {"alpha", NULL};
    int set_contains = 0;
    int set_subset = 0;

    assert_true(lmmc_std_text_set_make(text_set_a_values, 3, &fixture->text_set_a) == LMMC_STATUS_OK);
    assert_true(lmmc_std_text_set_make(text_set_b_values, 2, &fixture->text_set_b) == LMMC_STATUS_OK);
    assert_true(lmmc_std_text_set_make(NULL, 1, &fixture->text_set_xor) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_set_make(text_set_bad_values, 2, &fixture->text_set_xor) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_set_contains(NULL, "alpha", &set_contains) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_set_contains(&fixture->text_set_a, NULL, &set_contains) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_set_contains(&fixture->text_set_a, "alpha", NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_set_subset(NULL, &fixture->text_set_b, &set_subset) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_set_subset(&fixture->text_set_a, &fixture->text_set_b, NULL) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_set_union(NULL, &fixture->text_set_b, &fixture->text_set_xor) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_set_union(&fixture->text_set_a, &fixture->text_set_b, &fixture->text_set_a) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_std_text_set_destroy(&fixture->text_set_xor);
    lmmc_std_text_set_destroy(&fixture->text_set_b);
    lmmc_std_text_set_destroy(&fixture->text_set_a);
    lmmc_std_text_set_destroy(NULL);
}

static void test_allocation_size_boundaries(void **state) {
    struct test_fixture *fixture = *state;
    const char *values[] = {"alpha"};
    const size_t max_count = SIZE_MAX / sizeof(char *);
    lmmc_status_t overflow_status;
    lmmc_status_t allocation_status;

    lmmc_memory_fail_after_for_test(0);
    overflow_status = lmmc_std_text_set_make(values, max_count + 1, &fixture->result);
    allocation_status = lmmc_std_text_set_make(values, 1, &fixture->result);
    lmmc_memory_fail_reset_for_test();

    assert_true(overflow_status == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(allocation_status == LMMC_STATUS_ALLOCATION_FAILED);
    lmmc_std_text_set_destroy(&fixture->result);

    lmmc_memory_fail_after_for_test(0);
    allocation_status = lmmc_std_text_set_make(values, max_count, &fixture->result);
    lmmc_memory_fail_reset_for_test();

    assert_true(allocation_status == LMMC_STATUS_ALLOCATION_FAILED);
    lmmc_std_text_set_destroy(&fixture->result);
}

static void test_key_equality_hashing(void **state) {
    (void)state;
    uint64_t text_hash = 0;
    uint64_t text_hash_same = 0;
    uint64_t text_hash_other = 0;
    int text_equal = 0;

    assert_true(lmmc_std_text_equal("alpha", "alpha", &text_equal) == LMMC_STATUS_OK);
    assert_true(text_equal);
    assert_true(lmmc_std_text_hash("alpha", &text_hash) == LMMC_STATUS_OK);
    assert_true(lmmc_std_text_hash("alpha", &text_hash_same) == LMMC_STATUS_OK);
    assert_true(text_hash == text_hash_same);
    assert_true(lmmc_std_text_equal("alpha", "beta", &text_equal) == LMMC_STATUS_OK);
    assert_true(!text_equal);
    assert_true(lmmc_std_text_hash("beta", &text_hash_other) == LMMC_STATUS_OK);
    assert_true(text_hash != text_hash_other);
}

static void test_key_invalid_inputs(void **state) {
    (void)state;
    uint64_t text_hash = 0;
    int text_equal = 0;

    assert_true(lmmc_std_text_equal(NULL, "alpha", &text_equal) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_equal("alpha", NULL, &text_equal) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_equal("alpha", "alpha", NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_hash(NULL, &text_hash) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_text_hash("alpha", NULL) == LMMC_STATUS_INVALID_ARGUMENT);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_construction_membership, setup, teardown),
        cmocka_unit_test_setup_teardown(test_union_intersection_subset, setup, teardown),
        cmocka_unit_test_setup_teardown(test_difference_symmetric_difference, setup, teardown),
        cmocka_unit_test_setup_teardown(test_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_allocation_size_boundaries, setup, teardown),
        cmocka_unit_test(test_key_equality_hashing),
        cmocka_unit_test(test_key_invalid_inputs),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
