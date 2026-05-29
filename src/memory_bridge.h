/**
 * @file memory_bridge.h
 * @brief 内存分配抽象层，可在标准库 malloc 与 LAMMP 栈分配器之间切换。
 *
 * 当定义 @c LMMC_USE_LAMMP_ALLOC 时，全部分配 / 释放路由到 LAMMP；
 * 否则使用 @c malloc / @c free 。
 *
 * @internal
 */
#ifndef LMMC_MEMORY_BRIDGE_H
#define LMMC_MEMORY_BRIDGE_H

#include <stdlib.h>

#if defined(LMMC_USE_LAMMP_ALLOC)
#include "lammp/lmmp.h"
/** @brief 分配 @c sz 字节内存。 */
#define lmmc_alloc(sz) lmmp_alloc((sz))
/** @brief 释放 ::lmmc_alloc 返回的指针。 */
#define lmmc_free(ptr) lmmp_free((ptr))
/** @brief 重新分配内存。 */
#define lmmc_realloc(ptr, sz) lmmp_realloc((ptr), (sz))
#else
#define lmmc_alloc(sz) malloc((sz))
#define lmmc_free(ptr) free((ptr))
#define lmmc_realloc(ptr, sz) realloc((ptr), (sz))
#endif

#endif
