/**
 * @file memory_bridge.h
 * @brief Private recoverable allocator for persistent LMMC ownership.
 *
 * Persistent objects use the C heap so ownership may cross initialized
 * threads. LMMP temporary allocation remains an independent fail-fast layer.
 *
 * @internal
 */
#ifndef LMMC_MEMORY_BRIDGE_H
#define LMMC_MEMORY_BRIDGE_H

#include <stddef.h>

void* lmmc_memory_alloc(size_t size);
void* lmmc_memory_alloc_array(size_t count, size_t element_size);
void* lmmc_memory_alloc_array_plus(
    size_t count, size_t extra, size_t element_size);
void* lmmc_memory_alloc_array_2d(
    size_t rows, size_t cols, size_t element_size);
void lmmc_memory_free(void* pointer);
void* lmmc_memory_realloc(void* pointer, size_t size);

/** @brief Allocate persistent storage through the recoverable bridge. */
#define lmmc_alloc(sz) lmmc_memory_alloc((sz))
/** @brief Allocate a checked one-dimensional array. */
#define lmmc_alloc_array(count, element_size) \
    lmmc_memory_alloc_array((count), (element_size))
/** @brief Allocate an array after checked addition to its element count. */
#define lmmc_alloc_array_plus(count, extra, element_size) \
    lmmc_memory_alloc_array_plus((count), (extra), (element_size))
/** @brief Allocate a checked two-dimensional dense array. */
#define lmmc_alloc_array_2d(rows, cols, element_size) \
    lmmc_memory_alloc_array_2d((rows), (cols), (element_size))
/** @brief Release persistent storage on any externally synchronized thread. */
#define lmmc_free(ptr) lmmc_memory_free((ptr))
/** @brief Resize persistent storage without involving LMMP temporary state. */
#define lmmc_realloc(ptr, sz) lmmc_memory_realloc((ptr), (sz))

#endif
