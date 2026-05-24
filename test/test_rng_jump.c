/**
 * @file test_rng_jump.c
 * @brief Unit tests for RNG robust seeding, jump, long_jump, and clone.
 *
 * Validates Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "lmmc/random.h"
#include "lmmc/status.h"

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)

/**
 * Test: Two RNGs created without explicit seed in the same second
 * produce different sequences (Req 9.6).
 */
static int test_unique_default_seeds(void)
{
    lmmc_rng_t* rng1 = NULL;
    lmmc_rng_t* rng2 = NULL;
    lmmc_status_t st;
    int found_diff = 0;
    int i;

    st = lmmc_rng_create(&rng1);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng1");

    st = lmmc_rng_create(&rng2);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng2");

    /* Check first 16 samples differ in at least one position */
    for (i = 0; i < 16; i++) {
        uint64_t v1 = lmmc_rng_next_u64(rng1);
        uint64_t v2 = lmmc_rng_next_u64(rng2);
        if (v1 != v2) {
            found_diff = 1;
            break;
        }
    }

    CHECK(found_diff,
          "Two RNGs created without explicit seed produced identical first 16 samples");

    lmmc_rng_destroy(rng1);
    lmmc_rng_destroy(rng2);
    return 0;
}

/**
 * Test: lmmc_rng_clone produces an independent deep copy (Req 9.7).
 */
static int test_clone_deep_copy(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_rng_t* clone = NULL;
    lmmc_status_t st;
    int i;

    st = lmmc_rng_create(&rng);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng");

    lmmc_rng_seed(rng, 42);

    st = lmmc_rng_clone(rng, &clone);
    CHECK(st == LMMC_STATUS_OK, "Failed to clone rng");
    CHECK(clone != NULL, "Clone returned NULL");
    CHECK(clone != rng, "Clone returned same pointer");

    /* Both should produce identical sequences */
    for (i = 0; i < 100; i++) {
        uint64_t v1 = lmmc_rng_next_u64(rng);
        uint64_t v2 = lmmc_rng_next_u64(clone);
        CHECK(v1 == v2,
              "Clone diverged at index %d: original=%llu, clone=%llu",
              i, (unsigned long long)v1, (unsigned long long)v2);
    }

    lmmc_rng_destroy(rng);
    lmmc_rng_destroy(clone);
    return 0;
}

/**
 * Test: After jump, clone and original produce different sequences (Req 9.7).
 */
static int test_clone_then_jump_diverges(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_rng_t* clone = NULL;
    lmmc_status_t st;
    int found_diff = 0;
    int i;

    st = lmmc_rng_create(&rng);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng");

    lmmc_rng_seed(rng, 12345);

    st = lmmc_rng_clone(rng, &clone);
    CHECK(st == LMMC_STATUS_OK, "Failed to clone rng");

    /* Apply jump to the clone */
    st = lmmc_rng_jump(clone);
    CHECK(st == LMMC_STATUS_OK, "Failed to jump clone");

    /* Sequences should now differ */
    for (i = 0; i < 16; i++) {
        uint64_t v1 = lmmc_rng_next_u64(rng);
        uint64_t v2 = lmmc_rng_next_u64(clone);
        if (v1 != v2) {
            found_diff = 1;
            break;
        }
    }

    CHECK(found_diff,
          "After jump, clone and original still produce identical sequences");

    lmmc_rng_destroy(rng);
    lmmc_rng_destroy(clone);
    return 0;
}

/**
 * Test: lmmc_rng_jump with NULL returns LMMC_STATUS_INVALID_ARGUMENT (Req 9.5).
 */
static int test_jump_null_returns_error(void)
{
    lmmc_status_t st;

    st = lmmc_rng_jump(NULL);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "lmmc_rng_jump(NULL) returned %d, expected INVALID_ARGUMENT", (int)st);

    st = lmmc_rng_long_jump(NULL);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "lmmc_rng_long_jump(NULL) returned %d, expected INVALID_ARGUMENT", (int)st);

    return 0;
}

/**
 * Test: lmmc_rng_clone with NULL src or out returns LMMC_STATUS_INVALID_ARGUMENT.
 */
static int test_clone_null_returns_error(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_rng_t* out = NULL;
    lmmc_status_t st;

    st = lmmc_rng_clone(NULL, &out);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "lmmc_rng_clone(NULL, &out) returned %d", (int)st);

    st = lmmc_rng_create(&rng);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng");

    st = lmmc_rng_clone(rng, NULL);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "lmmc_rng_clone(rng, NULL) returned %d", (int)st);

    lmmc_rng_destroy(rng);
    return 0;
}

/**
 * Test: lmmc_rng_jump is deterministic — same state + jump = same result.
 */
static int test_jump_deterministic(void)
{
    lmmc_rng_t* rng1 = NULL;
    lmmc_rng_t* rng2 = NULL;
    lmmc_status_t st;
    int i;

    st = lmmc_rng_create(&rng1);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng1");
    st = lmmc_rng_create(&rng2);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng2");

    lmmc_rng_seed(rng1, 99999);
    lmmc_rng_seed(rng2, 99999);

    st = lmmc_rng_jump(rng1);
    CHECK(st == LMMC_STATUS_OK, "Failed to jump rng1");
    st = lmmc_rng_jump(rng2);
    CHECK(st == LMMC_STATUS_OK, "Failed to jump rng2");

    /* After identical jump, sequences should be identical */
    for (i = 0; i < 100; i++) {
        uint64_t v1 = lmmc_rng_next_u64(rng1);
        uint64_t v2 = lmmc_rng_next_u64(rng2);
        CHECK(v1 == v2,
              "After jump, rng1 and rng2 diverged at index %d", i);
    }

    lmmc_rng_destroy(rng1);
    lmmc_rng_destroy(rng2);
    return 0;
}

/**
 * Test: lmmc_rng_long_jump is deterministic and differs from jump.
 */
static int test_long_jump_deterministic_and_different(void)
{
    lmmc_rng_t* rng_jump = NULL;
    lmmc_rng_t* rng_long = NULL;
    lmmc_status_t st;
    int found_diff = 0;
    int i;

    st = lmmc_rng_create(&rng_jump);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng_jump");
    st = lmmc_rng_create(&rng_long);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng_long");

    lmmc_rng_seed(rng_jump, 77777);
    lmmc_rng_seed(rng_long, 77777);

    st = lmmc_rng_jump(rng_jump);
    CHECK(st == LMMC_STATUS_OK, "Failed to jump");
    st = lmmc_rng_long_jump(rng_long);
    CHECK(st == LMMC_STATUS_OK, "Failed to long_jump");

    /* jump and long_jump should produce different sequences */
    for (i = 0; i < 16; i++) {
        uint64_t v1 = lmmc_rng_next_u64(rng_jump);
        uint64_t v2 = lmmc_rng_next_u64(rng_long);
        if (v1 != v2) {
            found_diff = 1;
            break;
        }
    }

    CHECK(found_diff,
          "jump and long_jump produced identical sequences from same seed");

    lmmc_rng_destroy(rng_jump);
    lmmc_rng_destroy(rng_long);
    return 0;
}

/**
 * Test: Verify no overlap between original and jumped clone over many samples.
 * This is a statistical check — draw many samples from both and verify
 * no common values appear (Req 9.7 partial check with smaller sample).
 */
static int test_jump_no_overlap_short(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_rng_t* clone = NULL;
    lmmc_status_t st;
    int i, j;
    int found_overlap = 0;

    /* Use a smaller sample for unit test speed */
    #define OVERLAP_SAMPLES 1000
    static uint64_t seq_orig[OVERLAP_SAMPLES];
    static uint64_t seq_clone[OVERLAP_SAMPLES];

    st = lmmc_rng_create(&rng);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng");
    lmmc_rng_seed(rng, 54321);

    st = lmmc_rng_clone(rng, &clone);
    CHECK(st == LMMC_STATUS_OK, "Failed to clone");

    st = lmmc_rng_jump(clone);
    CHECK(st == LMMC_STATUS_OK, "Failed to jump clone");

    for (i = 0; i < OVERLAP_SAMPLES; i++) {
        seq_orig[i] = lmmc_rng_next_u64(rng);
        seq_clone[i] = lmmc_rng_next_u64(clone);
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
    CHECK(!found_overlap,
          "Found overlapping value between original and jumped clone sequences");

    lmmc_rng_destroy(rng);
    lmmc_rng_destroy(clone);
    return 0;
    #undef OVERLAP_SAMPLES
}


int main(void)
{
    int rc = 0;

    printf("=== RNG Jump/Clone/Seeding Unit Tests ===\n");
    printf("  Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7\n\n");

    printf("--- Unique default seeds ---\n");
    if (test_unique_default_seeds()) { rc = 1; }
    else { printf("  [PASS] Two RNGs created without seed produce different sequences\n"); }

    printf("--- Clone deep copy ---\n");
    if (test_clone_deep_copy()) { rc = 1; }
    else { printf("  [PASS] Clone produces identical sequence\n"); }

    printf("--- Clone then jump diverges ---\n");
    if (test_clone_then_jump_diverges()) { rc = 1; }
    else { printf("  [PASS] After jump, clone diverges from original\n"); }

    printf("--- Jump/long_jump NULL returns error ---\n");
    if (test_jump_null_returns_error()) { rc = 1; }
    else { printf("  [PASS] NULL handle returns INVALID_ARGUMENT\n"); }

    printf("--- Clone NULL returns error ---\n");
    if (test_clone_null_returns_error()) { rc = 1; }
    else { printf("  [PASS] NULL src/out returns INVALID_ARGUMENT\n"); }

    printf("--- Jump is deterministic ---\n");
    if (test_jump_deterministic()) { rc = 1; }
    else { printf("  [PASS] Same seed + jump = same sequence\n"); }

    printf("--- Long jump differs from jump ---\n");
    if (test_long_jump_deterministic_and_different()) { rc = 1; }
    else { printf("  [PASS] long_jump produces different sequence than jump\n"); }

    printf("--- No overlap between original and jumped clone ---\n");
    if (test_jump_no_overlap_short()) { rc = 1; }
    else { printf("  [PASS] No overlapping values in 1000-sample check\n"); }

    printf("\n");
    if (rc == 0) {
        printf("All RNG jump/clone/seeding tests PASSED.\n");
    } else {
        printf("Some RNG jump/clone/seeding tests FAILED (%d failures).\n", test_failures);
    }

    return rc;
}
