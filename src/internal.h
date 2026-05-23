#ifndef LMMC_INTERNAL_H
#define LMMC_INTERNAL_H

/**
 * @file internal.h
 * @brief Internal utility functions shared across LMMC source files.
 *
 * This header is NOT part of the public API and should NOT be installed.
 * It provides integer overflow checks, finite-number validation, and
 * common real-number utility functions used by multiple modules.
 */

#include "lmmc/config.h"
#include <stddef.h>
#include <stdint.h>
#include <math.h>

/* ========================================================================
 * Integer overflow safe arithmetic
 * ======================================================================== */

/**
 * @brief Safe multiplication of two size_t values with overflow detection.
 *
 * @param a First operand.
 * @param b Second operand.
 * @param result Pointer to store the product if no overflow occurs.
 * @return 1 on success (no overflow), 0 on overflow.
 */
static inline int lmmc_safe_mul_size(size_t a, size_t b, size_t *result)
{
    if (a == 0 || b == 0) {
        *result = 0;
        return 1;
    }
    if (a > SIZE_MAX / b) {
        return 0;
    }
    *result = a * b;
    return 1;
}

/**
 * @brief Safe addition of two size_t values with overflow detection.
 *
 * @param a First operand.
 * @param b Second operand.
 * @param result Pointer to store the sum if no overflow occurs.
 * @return 1 on success (no overflow), 0 on overflow.
 */
static inline int lmmc_safe_add_size(size_t a, size_t b, size_t *result)
{
    if (a > SIZE_MAX - b) {
        return 0;
    }
    *result = a + b;
    return 1;
}

/* ========================================================================
 * Finite number check (pointer version)
 * ======================================================================== */

/**
 * @brief Check whether a real number is finite (not NaN, not Inf).
 *
 * @param x Pointer to the value to check.
 * @return Non-zero (true) if the value is finite, 0 (false) otherwise.
 */
static inline int lmmc_is_finite(const lmmc_real_t *x)
{
    return isfinite(*x) != 0;
}

/* ========================================================================
 * Real-number utility functions
 * ======================================================================== */

/**
 * @brief Absolute value of a real number.
 */
static inline lmmc_real_t lmmc_abs(lmmc_real_t x)
{
    return fabs(x);
}

/**
 * @brief Maximum of two real numbers.
 */
static inline lmmc_real_t lmmc_max(lmmc_real_t a, lmmc_real_t b)
{
    return (a >= b) ? a : b;
}

/**
 * @brief Minimum of two real numbers.
 */
static inline lmmc_real_t lmmc_min(lmmc_real_t a, lmmc_real_t b)
{
    return (a <= b) ? a : b;
}

/**
 * @brief Clamp a value to the range [lo, hi].
 *
 * @param x  Value to clamp.
 * @param lo Lower bound.
 * @param hi Upper bound.
 * @return The clamped value, guaranteed to be in [lo, hi].
 */
static inline lmmc_real_t lmmc_clamp(lmmc_real_t x, lmmc_real_t lo, lmmc_real_t hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/**
 * @brief Swap two real numbers in place.
 *
 * @param a Pointer to the first value.
 * @param b Pointer to the second value.
 */
static inline void lmmc_swap(lmmc_real_t *a, lmmc_real_t *b)
{
    lmmc_real_t tmp = *a;
    *a = *b;
    *b = tmp;
}

#endif /* LMMC_INTERNAL_H */
