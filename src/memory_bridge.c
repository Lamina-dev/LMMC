#include "memory_bridge.h"

#include <stdint.h>
#include <stdlib.h>

#if defined(LMMC_BUILD_TESTS) || defined(LMMC_DEBUG_LEAKS)
#include <stdatomic.h>
#endif

#ifdef LMMC_BUILD_TESTS
static _Atomic size_t lmmc_memory_fail_after = SIZE_MAX;

void lmmc_memory_fail_after_for_test(size_t successful_allocations) {
    atomic_store_explicit(
        &lmmc_memory_fail_after, successful_allocations, memory_order_relaxed);
}

static int lmmc_memory_should_fail(void) {
    size_t remaining = atomic_load_explicit(
        &lmmc_memory_fail_after, memory_order_relaxed);
    while (remaining != SIZE_MAX) {
        if (remaining == 0) {
            if (atomic_compare_exchange_weak_explicit(
                    &lmmc_memory_fail_after, &remaining, SIZE_MAX,
                    memory_order_relaxed, memory_order_relaxed)) {
                return 1;
            }
        } else if (atomic_compare_exchange_weak_explicit(
                       &lmmc_memory_fail_after, &remaining, remaining - 1,
                       memory_order_relaxed, memory_order_relaxed)) {
            return 0;
        }
    }
    return 0;
}
#else
static int lmmc_memory_should_fail(void) { return 0; }
#endif

void* lmmc_memory_alloc(size_t size) {
    if (lmmc_memory_should_fail()) return NULL;
    return malloc(size);
}

void lmmc_memory_free(void* pointer) {
    free(pointer);
}

void* lmmc_memory_realloc(void* pointer, size_t size) {
    if (pointer == NULL) return lmmc_memory_alloc(size);
    if (size == 0) {
        free(pointer);
        return NULL;
    }
    if (lmmc_memory_should_fail()) return NULL;
    return realloc(pointer, size);
}

#ifdef LMMC_DEBUG_LEAKS
static _Atomic size_t lmmc_debug_allocation_count = 0;

void lmmc_debug_leaks_alloc(void) {
    atomic_fetch_add_explicit(
        &lmmc_debug_allocation_count, 1, memory_order_relaxed);
}

void lmmc_debug_leaks_free(void) {
    size_t count = atomic_load_explicit(
        &lmmc_debug_allocation_count, memory_order_relaxed);
    while (count != 0 && !atomic_compare_exchange_weak_explicit(
               &lmmc_debug_allocation_count, &count, count - 1,
               memory_order_relaxed, memory_order_relaxed)) {}
}

long long lmmc_debug_leaks_get_count(void) {
    const size_t count = atomic_load_explicit(
        &lmmc_debug_allocation_count, memory_order_relaxed);
    return count > (size_t)INT64_MAX ? INT64_MAX : (long long)count;
}
#endif
