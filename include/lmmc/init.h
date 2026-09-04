/**
 * @file init.h
 * @brief LMMC 当前线程的初始化与底层栈管理接口.
 *
 * LMMC 在内部使用 LMMP 的线程局部临时内存.每个调用线程必须独立获取
 * 生命周期租约,并在同一线程中成对释放.
 */
#ifndef LMMC_INIT_H
#define LMMC_INIT_H

#include <stddef.h>

#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 获取当前线程的 LMMC 生命周期租约.
 *
 * 调用支持嵌套;首次获取时初始化当前线程的 LMMP 资源.
 */
lmmc_status_t lmmc_init(void);

/**
 * @brief Release one lifecycle lease owned by the current thread.
 *
 * A final matching call releases only the current thread's LMMP temporary
 * stack state. Persistent LMMC objects use the recoverable heap bridge and may
 * outlive deinitialization; concurrent mutation still requires external
 * synchronization.
 *
 * @return @c LMMC_STATUS_OK on success or
 *         @c LMMC_STATUS_NOT_INITIALIZED when no lease is active.
 */
lmmc_status_t lmmc_deinit(void);

/**
 * @brief Reset the current thread's LMMP temporary-stack capacity.
 *
 * The reset is rejected only while this thread holds nested lifecycle leases.
 * Persistent vectors, matrices, tensors, interpolation objects, and RNG
 * handles do not block a temporary-stack reset.
 *
 * @param pool_size Requested LMMP temporary-pool size in bytes.
 * @return @c LMMC_STATUS_OK on success, @c LMMC_STATUS_NOT_INITIALIZED
 *         without an active lease, or @c LMMC_STATUS_BUSY for a nested lease.
 */
lmmc_status_t lmmc_stack_reset(size_t pool_size);

#ifdef LMMC_DEBUG_LEAKS
/**
 * @brief Get the number of persistent allocations currently owned by LMMC.
 *
 * Available only when compiled with @c LMMC_DEBUG_LEAKS. The process-wide
 * atomic count follows bridge allocations and remains correct when an object
 * is destroyed on a different thread.
 *
 * @return Current live bridge-allocation count.
 */
long long lmmc_debug_leaks_get_count(void);
#endif /* LMMC_DEBUG_LEAKS */

#ifdef __cplusplus
}
#endif

#endif
