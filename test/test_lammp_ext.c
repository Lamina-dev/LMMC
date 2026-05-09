#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "lammp/lmmp.h"
#include "lammp/lmmpn.h"
#include "lammp/mprand.h"
#include "lammp/numth.h"
#include "lammp/secret.h"
#include "lammp/version.h"

#define CHECK_OR_FAIL(label, cond) \
    do { \
        if (!(cond)) { \
            printf("lammp_ext fail: %s\n", (label)); \
            rc = 1; \
            goto cleanup; \
        } \
    } while (0)

static int lmmp_bigint_to_text(mp_srcptr limbs, mp_size_t limb_count, int base, char** out_text) {
    mp_size_t need = 0;
    mp_size_t wrote = 0;
    mp_size_t i = 0;
    mp_byte_t* buffer = NULL;

    if (out_text == NULL || base < 2 || base > 36) {
        return 1;
    }
    *out_text = NULL;

    if (limbs == NULL || limb_count == 0) {
        buffer = (mp_byte_t*)lmmp_alloc(2);
        if (buffer == NULL) {
            return 1;
        }
        buffer[0] = '0';
        buffer[1] = '\0';
        *out_text = (char*)buffer;
        return 0;
    }

    need = lmmp_to_str_len_(limbs, limb_count, base);
    buffer = (mp_byte_t*)lmmp_alloc((size_t)need + 1);
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
        // lmmp_to_str_ emits digits as numeric values; map them to ASCII.
        for (i = 0; i < wrote; ++i) {
            mp_byte_t d = buffer[i];
            if (d >= (mp_byte_t)base) {
                lmmp_free(buffer);
                return 1;
            }
            buffer[i] = (d < 10) ? (mp_byte_t)('0' + d) : (mp_byte_t)('a' + (d - 10));
        }
    }
    buffer[wrote] = '\0';
    *out_text = (char*)buffer;
    return 0;
}

int main(void) {
    mp_limb_t limb_a = 0;
    mp_limb_t limb_b = 0;
    mp_limb_t inv = 0;
    mp_limb_t words[4] = {11, 22, 33, 44};
    uint64_t sip_key[2] = {0x0123456789abcdefull, 0xfedcba9876543210ull};
    uint64_t strong_values[4] = {0, 0, 0, 0};
    uint64_t h0 = 0;
    uint64_t h1 = 0;
    ulong q = 0;
    ulong rem = 0;
    mp_size_t an = 0;
    mp_size_t rn = 0;
    mp_size_t strong_actual = 0;
    mp_ptr limbs = NULL;
    mp_ptr product = NULL;
    lmmp_strong_rng_t* strong_rng = NULL;
    char* factorial_text = NULL;
    char* npr_text = NULL;
    char* ncr_text = NULL;
    char* limb_mul_text = NULL;
    int rc = 0;

    lmmp_stack_init();

    CHECK_OR_FAIL("gcd", lmmp_gcd_11_(48, 18) == 6);

    rem = lmmp_mulmod_ulong_(4, 6, 13, &q);
    CHECK_OR_FAIL("mulmod", rem == 11 && q == 1);

    CHECK_OR_FAIL("powmod", lmmp_powmod_ulong_(7, 128, 13) == 3);

    CHECK_OR_FAIL("is_prime", lmmp_is_prime_ulong_(97) && !lmmp_is_prime_ulong_(100));

    inv = lmmp_binvert_ulong_(3);
    CHECK_OR_FAIL("binvert", ((mp_limb_t)3 * inv) == 1);

    lmmp_global_rng_init_(12345, 1);
    CHECK_OR_FAIL("random_1", lmmp_random_(&limb_a, 1) != 0);
    lmmp_global_rng_init_(12345, 1);
    CHECK_OR_FAIL("random_repro", lmmp_random_(&limb_b, 1) != 0 && limb_a == limb_b);

    CHECK_OR_FAIL("seed_random_1", lmmp_seed_random_(&limb_a, 1, 987654321u, 0) != 0);
    CHECK_OR_FAIL("seed_random_repro", lmmp_seed_random_(&limb_b, 1, 987654321u, 0) != 0 && limb_a == limb_b);

    h0 = lmmp_xxhash_(words, 4, (const uint64_t[1]){0x9e3779b97f4a7c15ull});
    h1 = lmmp_xxhash_(words, 4, (const uint64_t[1]){0x9e3779b97f4a7c15ull});
    CHECK_OR_FAIL("xxhash", h0 == h1);

    h0 = lmmp_siphash24_(words, 4, sip_key);
    h1 = lmmp_siphash24_(words, 4, sip_key);
    CHECK_OR_FAIL("siphash", h0 == h1);

    CHECK_OR_FAIL("version", LAMMP_VERSION[0] != '\0' && LAMMP_COMPILER[0] != '\0');

    mp_bitcnt_t fac_bits = 0;
    rn = lmmp_factorial_size_(10, &fac_bits);
    assert(rn > 0);
    limbs = (mp_ptr)lmmp_alloc((size_t)rn * sizeof(mp_limb_t));
    CHECK_OR_FAIL("factorial_alloc", limbs != NULL);
    an = lmmp_factorial_(limbs, fac_bits, rn, 10);
    CHECK_OR_FAIL("factorial_to_text", lmmp_bigint_to_text(limbs, an, 10, &factorial_text) == 0);
    CHECK_OR_FAIL("factorial_text", strcmp(factorial_text, "3628800") == 0);
    lmmp_free(limbs);
    limbs = NULL;

    mp_bitcnt_t npr_bits = 0;
    rn = lmmp_nPr_size_(10, 3, &npr_bits);
    limbs = (mp_ptr)lmmp_alloc((size_t)rn * sizeof(mp_limb_t));
    CHECK_OR_FAIL("npr_alloc", limbs != NULL);
    an = lmmp_nPr_(limbs, npr_bits, rn, 10, 3);
    CHECK_OR_FAIL("npr_text", lmmp_bigint_to_text(limbs, an, 10, &npr_text) == 0 && strcmp(npr_text, "720") == 0);
    lmmp_free(limbs);
    limbs = NULL;

    mp_bitcnt_t ncr_bits = 0;
    rn = lmmp_nCr_size_(10, 3, &ncr_bits);
    limbs = (mp_ptr)lmmp_alloc((size_t)rn * sizeof(mp_limb_t));
    CHECK_OR_FAIL("ncr_alloc", limbs != NULL);
    an = lmmp_nCr_(limbs, ncr_bits, rn, 10, 3);
    CHECK_OR_FAIL("ncr_text", lmmp_bigint_to_text(limbs, an, 10, &ncr_text) == 0 && strcmp(ncr_text, "120") == 0);
    lmmp_free(limbs);
    limbs = NULL;

    an = lmmp_limb_elem_mul_(&product, words, 4);
    CHECK_OR_FAIL("limb_elem_mul", lmmp_bigint_to_text(product, an, 10, &limb_mul_text) == 0 && strcmp(limb_mul_text, "351384") == 0);

    strong_rng = lmmp_strong_rng_init_(4, 2026);
    CHECK_OR_FAIL("strong_rng_init", strong_rng != NULL);
    strong_actual = lmmp_strong_random_((mp_ptr)strong_values, 4, strong_rng);
    CHECK_OR_FAIL("strong_rng_sample", strong_actual <= 4);

cleanup:
    if (strong_rng != NULL) {
        lmmp_strong_rng_free_(strong_rng);
    }
    if (product != NULL) {
        lmmp_free(product);
    }
    if (limbs != NULL) {
        lmmp_free(limbs);
    }
    lmmp_free(limb_mul_text);
    lmmp_free(ncr_text);
    lmmp_free(npr_text);
    lmmp_free(factorial_text);
    lmmp_global_deinit();

    if (rc != 0) {
        printf("lammp_ext test failed\n");
    }
    return rc;
}