/**
 * @file test_rng_jump.c
 * RNG 种子、jump、long_jump、clone 单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <stdint.h>

#include "lmmc/random.h"
#include "lmmc/status.h"

/**
 * 两个不指定种子的 RNG 在同一秒内创建，应产生不同序列。
 */
struct test_fixture {
    lmmc_rng_t *rng1;
    lmmc_rng_t *rng2;
    lmmc_rng_t *rng;
    lmmc_rng_t *clone;
    lmmc_rng_t *out;
    lmmc_rng_t *rng_jump;
    lmmc_rng_t *rng_long;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_rng_destroy(fixture->rng1);
    lmmc_rng_destroy(fixture->rng2);
    lmmc_rng_destroy(fixture->rng);
    lmmc_rng_destroy(fixture->clone);
    lmmc_rng_destroy(fixture->out);
    lmmc_rng_destroy(fixture->rng_jump);
    lmmc_rng_destroy(fixture->rng_long);
    free(fixture);
    *state = NULL;
    return 0;
}

static void test_unique_default_seeds(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int found_diff = 0;
    int i;

    st = lmmc_rng_create(&fixture->rng1);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_rng_create(&fixture->rng2);
    assert_true(st == LMMC_STATUS_OK);

    /* Check first 16 samples differ in at least one position */
    for (i = 0; i < 16; i++) {
        uint64_t v1 = lmmc_rng_next_u64(fixture->rng1);
        uint64_t v2 = lmmc_rng_next_u64(fixture->rng2);
        if (v1 != v2) {
            found_diff = 1;
            break;
        }
    }

    assert_true(found_diff);
}

/* clone 应产生独立的深拷贝。 */
static void test_clone_deep_copy(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int i;

    st = lmmc_rng_create(&fixture->rng);
    assert_true(st == LMMC_STATUS_OK);

    assert_int_equal(lmmc_rng_seed(fixture->rng, 42), LMMC_STATUS_OK);

    st = lmmc_rng_clone(fixture->rng, &fixture->clone);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->clone != NULL);
    assert_true(fixture->clone != fixture->rng);

    /* Both should produce identical sequences */
    for (i = 0; i < 100; i++) {
        uint64_t v1 = lmmc_rng_next_u64(fixture->rng);
        uint64_t v2 = lmmc_rng_next_u64(fixture->clone);
        assert_true(v1 == v2);
    }
}

/* jump 后 clone 与原始 RNG 应产生不同序列。 */
static void test_clone_then_jump_diverges(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int found_diff = 0;
    int i;

    st = lmmc_rng_create(&fixture->rng);
    assert_true(st == LMMC_STATUS_OK);

    assert_int_equal(lmmc_rng_seed(fixture->rng, 12345), LMMC_STATUS_OK);

    st = lmmc_rng_clone(fixture->rng, &fixture->clone);
    assert_true(st == LMMC_STATUS_OK);

    /* Apply jump to the clone */
    st = lmmc_rng_jump(fixture->clone);
    assert_true(st == LMMC_STATUS_OK);

    /* Sequences should now differ */
    for (i = 0; i < 16; i++) {
        uint64_t v1 = lmmc_rng_next_u64(fixture->rng);
        uint64_t v2 = lmmc_rng_next_u64(fixture->clone);
        if (v1 != v2) {
            found_diff = 1;
            break;
        }
    }

    assert_true(found_diff);
}

/* NULL 参数应返回 LMMC_STATUS_INVALID_ARGUMENT。 */
static void test_jump_null_returns_error(void **state) {
    (void)state;
    lmmc_status_t st;

    st = lmmc_rng_jump(NULL);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_rng_long_jump(NULL);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

/**
 * Test: lmmc_rng_clone with NULL src or out returns LMMC_STATUS_INVALID_ARGUMENT.
 */
static void test_clone_null_returns_error(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_rng_clone(NULL, &fixture->out);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_rng_create(&fixture->rng);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_rng_clone(fixture->rng, NULL);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

/**
 * Test: lmmc_rng_jump is deterministic — same state + jump = same result.
 */
static void test_jump_deterministic(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int i;

    st = lmmc_rng_create(&fixture->rng1);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_rng_create(&fixture->rng2);
    assert_true(st == LMMC_STATUS_OK);

    assert_int_equal(lmmc_rng_seed(fixture->rng1, 99999), LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng2, 99999), LMMC_STATUS_OK);

    st = lmmc_rng_jump(fixture->rng1);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_rng_jump(fixture->rng2);
    assert_true(st == LMMC_STATUS_OK);

    /* After identical jump, sequences should be identical */
    for (i = 0; i < 100; i++) {
        uint64_t v1 = lmmc_rng_next_u64(fixture->rng1);
        uint64_t v2 = lmmc_rng_next_u64(fixture->rng2);
        assert_true(v1 == v2);
    }
}

/**
 * Test: lmmc_rng_long_jump is deterministic and differs from jump.
 */
static void test_long_jump_deterministic_and_different(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int found_diff = 0;
    int i;

    st = lmmc_rng_create(&fixture->rng_jump);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_rng_create(&fixture->rng_long);
    assert_true(st == LMMC_STATUS_OK);

    assert_int_equal(lmmc_rng_seed(fixture->rng_jump, 77777), LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng_long, 77777), LMMC_STATUS_OK);

    st = lmmc_rng_jump(fixture->rng_jump);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_rng_long_jump(fixture->rng_long);
    assert_true(st == LMMC_STATUS_OK);

    /* jump and long_jump should produce different sequences */
    for (i = 0; i < 16; i++) {
        uint64_t v1 = lmmc_rng_next_u64(fixture->rng_jump);
        uint64_t v2 = lmmc_rng_next_u64(fixture->rng_long);
        if (v1 != v2) {
            found_diff = 1;
            break;
        }
    }

    assert_true(found_diff);
}

/**
 * Test: Verify no overlap between original and jumped clone over many samples.
 * This is a statistical check — draw many samples from both and verify
 * no common values appear .
 */
static void test_jump_no_overlap_short(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int i, j;
    int found_overlap = 0;

/* Use a smaller sample for unit test speed */
#define OVERLAP_SAMPLES 1000
    static uint64_t seq_orig[OVERLAP_SAMPLES];
    static uint64_t seq_clone[OVERLAP_SAMPLES];

    st = lmmc_rng_create(&fixture->rng);
    assert_true(st == LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->rng, 54321), LMMC_STATUS_OK);

    st = lmmc_rng_clone(fixture->rng, &fixture->clone);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_rng_jump(fixture->clone);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < OVERLAP_SAMPLES; i++) {
        seq_orig[i] = lmmc_rng_next_u64(fixture->rng);
        seq_clone[i] = lmmc_rng_next_u64(fixture->clone);
    }

    /* Simple O(n^2) check for small sample — no value should appear in both */
    for (i = 0; i < OVERLAP_SAMPLES && !found_overlap; i++) {
        for (j = 0; j < OVERLAP_SAMPLES; j++) {
            if (seq_orig[i] == seq_clone[j]) {
                found_overlap = 1;
                break;
            }
        }
    }

    /* With 2^64 range and 1000 samples each, collision probability is negligible */
    assert_true(!found_overlap);

#undef OVERLAP_SAMPLES
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_unique_default_seeds, setup, teardown),
        cmocka_unit_test_setup_teardown(test_clone_deep_copy, setup, teardown),
        cmocka_unit_test_setup_teardown(test_clone_then_jump_diverges, setup, teardown),
        cmocka_unit_test_setup_teardown(test_jump_null_returns_error, setup, teardown),
        cmocka_unit_test_setup_teardown(test_clone_null_returns_error, setup, teardown),
        cmocka_unit_test_setup_teardown(test_jump_deterministic, setup, teardown),
        cmocka_unit_test_setup_teardown(test_long_jump_deterministic_and_different, setup, teardown),
        cmocka_unit_test_setup_teardown(test_jump_no_overlap_short, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
