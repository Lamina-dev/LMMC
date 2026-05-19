#ifndef LMMC_INIT_H
#define LMMC_INIT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize LMMC and underlying LAMMP global resources.
 *
 * Must be called once before any LMMC or LAMMP operation.
 * Thread-safe with reference counting — each init() must be paired
 * with a matching deinit().
 *
 * In the current LAMMP version this initialises the internal
 * operation stack.  When LAMMP is updated to provide explicit
 * lmmp_global_init_() / lmmp_global_deinit(), this function will
 * call those instead.
 */
void lmmc_init(void);

/**
 * @brief Deinitialize LMMC and release LAMMP global resources.
 *
 * Paired with lmmc_init().  When the reference count reaches zero
 * the underlying LAMMP resources are freed.
 */
void lmmc_deinit(void);

/**
 * @brief Reset / trim the LAMMP internal stack back to the given
 *        size, releasing any excess memory.
 *
 * This is a thin wrapper around lmmp_stack_reset().  When the new
 * LAMMP ships with proper stack management inside
 * lmmp_global_init_() / lmmp_global_deinit(), this wrapper may
 * become a no-op and should be removed from callers.
 *
 * @param size  Target stack size in bytes (e.g. 327680 for 320 KiB).
 *              Pass 0 to free the entire stack.
 */
void lmmc_stack_reset(size_t size);

#ifdef __cplusplus
}
#endif

#endif /* LMMC_INIT_H */
