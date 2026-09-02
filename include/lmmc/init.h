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
 * @brief 释放当前线程的一层 LMMC 生命周期租约.
 *
 * 最后一层租约仅在当前线程没有活动 LMMC 分配时释放;否则返回
 * ::LMMC_STATUS_BUSY 且保留租约和资源。
 */
lmmc_status_t lmmc_deinit(void);

/**
 * @brief 重置当前线程的 LMMP 临时栈容量.
 *
 * 当前线程必须恰好持有一层生命周期租约且没有活动 LMMC 分配;
 * 存在嵌套租约或活动对象时返回 ::LMMC_STATUS_BUSY.
 */
lmmc_status_t lmmc_stack_reset(size_t size);

#ifdef LMMC_DEBUG_LEAKS
/**
 * @brief 记录一次分配（调试泄漏追踪）。
 *
 * 仅在 LMMC_DEBUG_LEAKS 编译宏启用时可用。
 * 计数按线程隔离，并与 LMMC 内部分配桥接共享。
 */
void lmmc_debug_leaks_alloc(void);

/**
 * @brief 记录一次释放（调试泄漏追踪）。
 *
 * 仅在 LMMC_DEBUG_LEAKS 编译宏启用时可用。
 */
void lmmc_debug_leaks_free(void);

/**
 * @brief 获取当前未释放的分配计数。
 *
 * @return 当前未释放的分配数量。
 */
long long lmmc_debug_leaks_get_count(void);
#endif /* LMMC_DEBUG_LEAKS */

#ifdef __cplusplus
}
#endif

#endif
