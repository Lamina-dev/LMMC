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
void lmmc_memory_free(void* pointer);
void* lmmc_memory_realloc(void* pointer, size_t size);

/** @brief Allocate persistent storage through the recoverable bridge. */
#define lmmc_alloc(sz) lmmc_memory_alloc((sz))
/** @brief Release persistent storage on any externally synchronized thread. */
#define lmmc_free(ptr) lmmc_memory_free((ptr))
/** @brief Resize persistent storage without involving LMMP temporary state. */
#define lmmc_realloc(ptr, sz) lmmc_memory_realloc((ptr), (sz))

#endif
