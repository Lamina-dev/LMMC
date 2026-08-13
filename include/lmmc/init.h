/**
 * @file init.h
 * @brief LMMC 库初始化与底层栈管理接口。
 *
 * LMMC 在内部使用 LAMMP 提供的栈分配器以减少堆分配开销。
 * 调用任何 LMMC 接口前必须先 ::lmmc_init ，使用完毕后调用
 * ::lmmc_deinit 释放。两者支持嵌套调用计数（原子引用计数）。
 */
#ifndef LMMC_INIT_H
#define LMMC_INIT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化 LMMC 库。
 *
 * 内部维护原子引用计数，多次调用是安全的；只有首次调用（0→1 转换）
 * 真正完成初始化（调用 lmmp_global_init）。
 * 必须在调用其他 LMMC 接口之前完成至少一次。
 */
void lmmc_init(void);

/**
 * @brief 反初始化 LMMC 库。
 *
 * 与 ::lmmc_init 配对使用，原子引用计数归零时（1→0 转换）执行真正的清理：
 * 释放栈分配器并调用 lmmp_global_deinit。
 *
 * 当编译时定义了 LMMC_DEBUG_LEAKS 宏，可在反初始化前通过
 * ::lmmc_debug_leaks_get_count 查询未释放的分配。
 */
void lmmc_deinit(void);

/**
 * @brief 重置内部栈分配器到指定容量。
 *
 * 通常用于在大规模运算之间释放栈上残留分配，或调整可用栈尺寸。
 *
 * @param size 新的栈容量（字节数）。
 */
void lmmc_stack_reset(size_t size);

#ifdef LMMC_DEBUG_LEAKS
/**
 * @brief 记录一次分配（调试泄漏追踪）。
 *
 * 仅在 LMMC_DEBUG_LEAKS 编译宏启用时可用。
 * 内部使用原子计数器追踪分配数量。
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
