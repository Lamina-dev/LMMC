/**
 * @file init.h
 * @brief LMMC 库初始化与底层栈管理接口。
 *
 * LMMC 在内部使用 LAMMP 提供的栈分配器以减少堆分配开销。
 * 调用任何 LMMC 接口前必须先 ::lmmc_init ，使用完毕后调用
 * ::lmmc_deinit 释放。两者支持嵌套调用计数。
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
 * 内部维护引用计数，多次调用是安全的；只有首次调用真正完成初始化。
 * 必须在调用其他 LMMC 接口之前完成至少一次。
 */
void lmmc_init(void);

/**
 * @brief 反初始化 LMMC 库。
 *
 * 与 ::lmmc_init 配对使用，引用计数归零时执行真正的清理。
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

#ifdef __cplusplus
}
#endif

#endif
