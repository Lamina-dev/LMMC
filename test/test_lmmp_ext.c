/**
 * @file test_lmmp_ext.c
 * 针对 LMMC 中 LMMP 扩展接口的单元测试。
 */
#include <lmmp.h>
#include <lmmpn.h>
#include <mprand.h>
#include <numth.h>
#include <secret.h>

#include "test_common.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int encode_lmmp_digits(mp_byte_t *buffer, mp_size_t wrote, int base) {
    for (mp_size_t i = 0; i < wrote; ++i) {
        mp_byte_t d = buffer[i];
        if (d >= (mp_byte_t)base) {
            lmmp_free(buffer);
            return 1;
        }
        buffer[i] = (d < 10) ? (mp_byte_t)('0' + d) : (mp_byte_t)('a' + (d - 10));
    }
    return 0;
}

static int lmmp_bigint_to_text(mp_srcptr limbs, mp_size_t limb_count, int base, char **out_text) {
    mp_size_t need = 0;
    mp_size_t wrote = 0;
    mp_size_t i = 0;
    mp_byte_t *buffer = NULL;

    if (out_text == NULL || base < 2 || base > 36) {
        return 1;
    }
    *out_text = NULL;

    if (limbs == NULL || limb_count == 0) {
        buffer = (mp_byte_t *)lmmp_alloc(2);
        if (buffer == NULL) {
            return 1;
        }
        buffer[0] = '0';
        buffer[1] = '\0';
        *out_text = (char *)buffer;
        return 0;
    }

    need = lmmp_to_str_len_(limbs, limb_count, base);
    buffer = (mp_byte_t *)lmmp_alloc((size_t)need + 1);
    if (buffer == NULL) {
        return 1;
    }

    wrote = lmmp_to_str_(buffer, limbs, limb_count, base);
    if (wrote == 0) {
        buffer[0] = '0';
        wrote = 1;
    } else {
        for (i = 0; i < wrote / 2; ++i) {
            mp_byte_t t = buffer[i];
            buffer[i] = buffer[wrote - 1 - i];
            buffer[wrote - 1 - i] = t;
        }

        if (encode_lmmp_digits(buffer, wrote, base) != 0) {
            return 1;
        }
    }
    buffer[wrote] = '\0';
    *out_text = (char *)buffer;
    return 0;
}

static void test_modular_arithmetic(void **state) {
    (void)state;

    mp_limb_t inv = 0;
    ulong q = 0;
    ulong rem = 0;
    assert_true(lmmp_gcd_11_(48, 18) == 6);

    rem = lmmp_mulmod_ulong_(4, 6, 13, &q);
    assert_true(rem == 11 && q == 1);

    assert_true(lmmp_powmod_ulong_odd_(7, 128, 13) == 3);

    assert_true(lmmp_is_prime_ulong_(97) && !lmmp_is_prime_ulong_(100));

    inv = lmmp_binvert_ulong_(3);
    assert_true(((mp_limb_t)3 * inv) == 1);
}

static void test_random_reproducibility(void **state) {
    (void)state;

    mp_limb_t limb_a = 0;
    mp_limb_t limb_b = 0;
    lmmp_global_rng_init_(12345, 1);
    assert_true(lmmp_random_(&limb_a, 1) != 0);
    lmmp_global_rng_init_(12345, 1);
    assert_true(lmmp_random_(&limb_b, 1) != 0 && limb_a == limb_b);

    assert_true(lmmp_seed_random_(&limb_a, 1, 987654321u, 0) != 0);
    assert_true(lmmp_seed_random_(&limb_b, 1, 987654321u, 0) != 0 && limb_a == limb_b);
}

static void test_hash_reproducibility(void **state) {
    (void)state;

    mp_limb_t words[4] = {11, 22, 33, 44};
    uint64_t sip_key[2] = {UINT64_C(0x0123456789abcdef), UINT64_C(0xfedcba9876543210)};
    uint64_t h0 = 0;
    uint64_t h1 = 0;
    h0 = lmmp_xxhash_(words, 4, (const uint64_t[1]){UINT64_C(0x9e3779b97f4a7c15)});
    h1 = lmmp_xxhash_(words, 4, (const uint64_t[1]){UINT64_C(0x9e3779b97f4a7c15)});
    assert_true(h0 == h1);

    h0 = lmmp_siphash24_(words, 4, sip_key);
    h1 = lmmp_siphash24_(words, 4, sip_key);
    assert_true(h0 == h1);
}

typedef struct {
    mp_ptr limbs;
    char *text;
    lmmp_strong_rng_t *rng;
} lmmp_fixture_t;

static int setup(void **state) {
    lmmp_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    lmmp_fixture_t *fixture = *state;
    lmmp_free(fixture->limbs);
    lmmp_free(fixture->text);
    if (fixture->rng != NULL)
        lmmp_strong_rng_free_(fixture->rng);
    free(fixture);
    return 0;
}

static int group_setup(void **state) {
    (void)state;
    lmmp_global_init();
    return 0;
}

static int group_teardown(void **state) {
    (void)state;
    lmmp_global_deinit();
    return 0;
}

static void test_factorial(void **state) {
    lmmp_fixture_t *fixture = *state;
    mp_bitcnt_t bits = 0;
    mp_size_t rn = lmmp_factorial_size_(10, &bits);
    assert_true(rn > 0);
    fixture->limbs = (mp_ptr)lmmp_alloc((size_t)rn * sizeof(mp_limb_t));
    assert_non_null(fixture->limbs);
    mp_size_t an = lmmp_factorial_(fixture->limbs, bits, rn, 10);
    assert_int_equal(lmmp_bigint_to_text(fixture->limbs, an, 10, &fixture->text), 0);
    assert_string_equal(fixture->text, "3628800");
}

static void test_permutations(void **state) {
    lmmp_fixture_t *fixture = *state;
    mp_bitcnt_t bits = 0;
    mp_size_t rn = lmmp_nPr_size_(10, 3, &bits);
    fixture->limbs = (mp_ptr)lmmp_alloc((size_t)rn * sizeof(mp_limb_t));
    assert_non_null(fixture->limbs);
    mp_size_t an = lmmp_nPr_(fixture->limbs, bits, rn, 10, 3);
    assert_int_equal(lmmp_bigint_to_text(fixture->limbs, an, 10, &fixture->text), 0);
    assert_string_equal(fixture->text, "720");
}

static void test_combinations(void **state) {
    lmmp_fixture_t *fixture = *state;
    mp_bitcnt_t bits = 0;
    mp_size_t rn = lmmp_nCr_size_(10, 3, &bits);
    fixture->limbs = (mp_ptr)lmmp_alloc((size_t)rn * sizeof(mp_limb_t));
    assert_non_null(fixture->limbs);
    mp_size_t an = lmmp_nCr_(fixture->limbs, bits, rn, 10, 3);
    assert_int_equal(lmmp_bigint_to_text(fixture->limbs, an, 10, &fixture->text), 0);
    assert_string_equal(fixture->text, "120");
}

static void test_strong_random_sample(void **state) {
    lmmp_fixture_t *fixture = *state;
    uint64_t values[4] = {0, 0, 0, 0};
    fixture->rng = lmmp_strong_rng_init_(4, 2026);
    assert_non_null(fixture->rng);
    mp_size_t actual = lmmp_strong_random_((mp_ptr)values, 4, fixture->rng);
    assert_true(actual <= 4);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_modular_arithmetic),
        cmocka_unit_test(test_random_reproducibility),
        cmocka_unit_test(test_hash_reproducibility),
        cmocka_unit_test_setup_teardown(test_factorial, setup, teardown),
        cmocka_unit_test_setup_teardown(test_permutations, setup, teardown),
        cmocka_unit_test_setup_teardown(test_combinations, setup, teardown),
        cmocka_unit_test_setup_teardown(test_strong_random_sample, setup, teardown),
    };
    return cmocka_run_group_tests(tests, group_setup, group_teardown);
}
