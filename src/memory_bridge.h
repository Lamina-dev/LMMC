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

#endif
