/**
 * @file test_random_rng.c
 * 针对 LMMC 中 random rng 相关接口的单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <stdint.h>

#include "lmmc/random.h"
#include "lmmc/status.h"

#define SEQUENCE_LENGTH 200

#define NUM_ITERATIONS 100

struct test_fixture {
    lmmc_rng_t *rng1;
    lmmc_rng_t *rng2;
    lmmc_rng_t *rng;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    srand(0x52524E47u);
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_rng_destroy(fixture->rng1);
    lmmc_rng_destroy(fixture->rng2);
    lmmc_rng_destroy(fixture->rng);
    free(fixture);
    *state = NULL;
    return 0;
}

static uint64_t rand_u64(void) {
    uint64_t val = 0;
    size_t i;
    for (i = 0; i < sizeof(uint64_t); i++) {
        val = (val << 8) | (uint64_t)(rand() & 0xFF);
    }
    return val;
}

static void check_same_seed_same_sequence(struct test_fixture *fixture, uint64_t seed) {
    lmmc_status_t st;
    int i;

    st = lmmc_rng_create(&fixture->rng1);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_rng_create(&fixture->rng2);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_rng_seed(fixture->rng1, seed);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_rng_seed(fixture->rng2, seed);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < SEQUENCE_LENGTH; i++) {
        uint64_t v1 = lmmc_rng_next_u64(fixture->rng1);
        uint64_t v2 = lmmc_rng_next_u64(fixture->rng2);
        assert_true(v1 == v2);
    }

    lmmc_rng_destroy(fixture->rng1);
    fixture->rng1 = NULL;
    lmmc_rng_destroy(fixture->rng2);
    fixture->rng2 = NULL;
}

static void test_different_seeds_different_sequences(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    int i;
    int found_diff;
    uint64_t seed1 = 12345;
    uint64_t seed2 = 67890;

    st = lmmc_rng_create(&fixture->rng1);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_rng_create(&fixture->rng2);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_rng_seed(fixture->rng1, seed1);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_rng_seed(fixture->rng2, seed2);
    assert_true(st == LMMC_STATUS_OK);

    found_diff = 0;
    for (i = 0; i < SEQUENCE_LENGTH; i++) {
        uint64_t v1 = lmmc_rng_next_u64(fixture->rng1);
        uint64_t v2 = lmmc_rng_next_u64(fixture->rng2);
        if (v1 != v2) {
            found_diff = 1;
            break;
        }
    }

    assert_true(found_diff);
}

static void test_different_seeds_random_pairs(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        lmmc_status_t st;
        uint64_t seed1 = rand_u64();
        uint64_t seed2 = rand_u64();
        int i;
        int found_diff;

        if (seed1 == seed2)
            continue;

        st = lmmc_rng_create(&fixture->rng1);
        assert_true(st == LMMC_STATUS_OK);

        st = lmmc_rng_create(&fixture->rng2);
        assert_true(st == LMMC_STATUS_OK);

        st = lmmc_rng_seed(fixture->rng1, seed1);
        assert_true(st == LMMC_STATUS_OK);

        st = lmmc_rng_seed(fixture->rng2, seed2);
        assert_true(st == LMMC_STATUS_OK);

        found_diff = 0;
        for (i = 0; i < SEQUENCE_LENGTH; i++) {
            uint64_t v1 = lmmc_rng_next_u64(fixture->rng1);
            uint64_t v2 = lmmc_rng_next_u64(fixture->rng2);
            if (v1 != v2) {
                found_diff = 1;
                break;
            }
        }

        assert_true(found_diff);

        lmmc_rng_destroy(fixture->rng1);
        fixture->rng1 = NULL;
        lmmc_rng_destroy(fixture->rng2);
        fixture->rng2 = NULL;
    }
}

static void test_reseed_resets_state(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    uint64_t first_run[SEQUENCE_LENGTH];
    uint64_t second_run[SEQUENCE_LENGTH];
    int i;
    uint64_t seed = UINT64_C(0xABCDEF0123456789);

    st = lmmc_rng_create(&fixture->rng);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_rng_seed(fixture->rng, seed);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < SEQUENCE_LENGTH; i++) {
        first_run[i] = lmmc_rng_next_u64(fixture->rng);
    }

    for (i = 0; i < 1000; i++) {
        (void)lmmc_rng_next_u64(fixture->rng);
    }

    st = lmmc_rng_seed(fixture->rng, seed);
    assert_true(st == LMMC_STATUS_OK);

    for (i = 0; i < SEQUENCE_LENGTH; i++) {
        second_run[i] = lmmc_rng_next_u64(fixture->rng);
    }

    for (i = 0; i < SEQUENCE_LENGTH; i++) {
        assert_true(first_run[i] == second_run[i]);
    }
}

static void test_multiple_specific_seeds(void **state) {
    struct test_fixture *fixture = *state;
    uint64_t seeds[] = {
        0,
        1,
        UINT64_MAX,
        UINT64_C(0x0000000000000002),
        UINT64_C(0x7FFFFFFFFFFFFFFF),
        UINT64_C(0x8000000000000000),
        UINT64_C(0xDEADBEEFCAFEBABE),
        UINT64_C(0x0123456789ABCDEF)};
    size_t num_seeds = sizeof(seeds) / sizeof(seeds[0]);
    size_t s;

    for (s = 0; s < num_seeds; s++) {
        check_same_seed_same_sequence(fixture, seeds[s]);
    }
}

static void test_random_seeds_reproducibility(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        uint64_t seed = rand_u64();
        check_same_seed_same_sequence(fixture, seed);
    }
}

static void test_raw_xoshiro_fixed_seed(void **state) {
    struct test_fixture *fixture = *state;
    static const uint64_t expected_u64[] = {
        UINT64_C(0xa2c2a42038d4ec3d),
        UINT64_C(0x05fc25d0738e7b0f),
        UINT64_C(0x625e7bff938e701e),
        UINT64_C(0x1ba4ddc6fe2b5726)};
    const uint64_t seed = UINT64_C(0x0123456789abcdef);

    size_t i;

    assert_true(lmmc_rng_create(&fixture->rng) == LMMC_STATUS_OK);
    assert_true(lmmc_rng_seed(fixture->rng, seed) == LMMC_STATUS_OK);
    for (i = 0; i < sizeof(expected_u64) / sizeof(expected_u64[0]); ++i) {
        assert_true(lmmc_rng_next_u64(fixture->rng) == expected_u64[i]);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_different_seeds_different_sequences, setup, teardown),
        cmocka_unit_test_setup_teardown(test_different_seeds_random_pairs, setup, teardown),
        cmocka_unit_test_setup_teardown(test_reseed_resets_state, setup, teardown),
        cmocka_unit_test_setup_teardown(test_multiple_specific_seeds, setup, teardown),
        cmocka_unit_test_setup_teardown(test_random_seeds_reproducibility, setup, teardown),
        cmocka_unit_test_setup_teardown(test_raw_xoshiro_fixed_seed, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
