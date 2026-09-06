/**
 * @file init.c
 * @brief LMMC 当前线程生命周期与 LMMP 临时栈桥接实现.
 */
#include "lmmc/init.h"

#include <lmmp.h>

static _Thread_local size_t lmmc_lifecycle_depth = 0;

lmmc_status_t lmmc_init(void) {
    if (lmmc_lifecycle_depth == SIZE_MAX) {
        return LMMC_STATUS_REFERENCE_LIMIT;
    }
    if (lmmc_lifecycle_depth == 0) {
        lmmp_global_init();
    }
    ++lmmc_lifecycle_depth;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_deinit(void) {
    if (lmmc_lifecycle_depth == 0) {
        return LMMC_STATUS_NOT_INITIALIZED;
    }
    --lmmc_lifecycle_depth;
    if (lmmc_lifecycle_depth == 0) {
        lmmp_global_deinit();
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_stack_reset(size_t size) {
    if (lmmc_lifecycle_depth == 0) {
        return LMMC_STATUS_NOT_INITIALIZED;
    }
    if (lmmc_lifecycle_depth != 1) {
        return LMMC_STATUS_BUSY;
    }
    if (lmmp_stack_deinit() != 0) {
        return LMMC_STATUS_NOT_INITIALIZED;
    }
    if (lmmp_stack_init(size) != 0) {
        return LMMC_STATUS_NOT_INITIALIZED;
    }
    return LMMC_STATUS_OK;
}
