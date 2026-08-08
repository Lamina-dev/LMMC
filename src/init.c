/**
 * @file init.c
 * @brief LMMC 库初始化与栈分配器桥接实现。
 *
 * 使用原子引用计数保证多线程安全的初始化/反初始化。
 * 当定义 LMMC_DEBUG_LEAKS 编译宏时，追踪分配计数并在最终反初始化时输出摘要。
 */
#include "lmmc/init.h"

#include <stdatomic.h>
#include <stdio.h>

#include "lammp/lmmp.h"

/* ─── 原子引用计数 ─── */
static _Atomic int lmmc_ref_count = 0;

/* ─── 调试泄漏追踪 ─── */
#ifdef LMMC_DEBUG_LEAKS
static _Atomic long long lmmc_alloc_count = 0;

void lmmc_debug_leaks_alloc(void) {
    atomic_fetch_add_explicit(&lmmc_alloc_count, 1, memory_order_relaxed);
}

void lmmc_debug_leaks_free(void) {
    atomic_fetch_sub_explicit(&lmmc_alloc_count, 1, memory_order_relaxed);
}

long long lmmc_debug_leaks_get_count(void) {
    return atomic_load_explicit(&lmmc_alloc_count, memory_order_relaxed);
}
#endif /* LMMC_DEBUG_LEAKS */

void lmmc_init(void) {
    int prev = atomic_fetch_add_explicit(&lmmc_ref_count, 1, memory_order_acq_rel);
    if (prev == 0) {
        /* 0→1 转换：执行真正的全局初始化 */
        lmmp_global_init();
    }
}

void lmmc_deinit(void) {
    int prev = atomic_fetch_sub_explicit(&lmmc_ref_count, 1, memory_order_acq_rel);
    if (prev == 1) {
        /* 1→0 转换：执行真正的全局清理 */

#ifdef LMMC_DEBUG_LEAKS
        long long remaining = atomic_load_explicit(&lmmc_alloc_count, memory_order_relaxed);
        if (remaining != 0) {
            fprintf(stderr,
                    "[LMMC_DEBUG_LEAKS] Final deinit: %lld allocation(s) still outstanding.\n",
                    remaining);
        }
        /* 重置计数器 */
        atomic_store_explicit(&lmmc_alloc_count, 0, memory_order_relaxed);
#endif /* LMMC_DEBUG_LEAKS */

        /* 释放全局堆资源 */
        lmmp_global_deinit();
    }
}

void lmmc_stack_reset(size_t size) {
    lmmp_stack_deinit();
    lmmp_stack_init(size);
}
