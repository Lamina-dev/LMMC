/**
 * @file diagnostic.h
 * @brief Unified diagnostic events for LMMC operations.
 */
#ifndef LMMC_DIAGNOSTIC_H
#define LMMC_DIAGNOSTIC_H

#include <stddef.h>

#include "lmmc/numeric.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LMMC_DIAGNOSTIC_TRACE = 0,
    LMMC_DIAGNOSTIC_INFO = 1,
    LMMC_DIAGNOSTIC_WARNING = 2,
    LMMC_DIAGNOSTIC_ERROR = 3
} lmmc_diagnostic_level_t;

typedef struct {
    lmmc_diagnostic_level_t level;
    const char* operation;
    const char* message;
    size_t iteration;
    const lmmc_real_t* values;
    size_t value_count;
} lmmc_diagnostic_t;

typedef void (*lmmc_diagnostic_callback_t)(
    const lmmc_diagnostic_t* diagnostic,
    void* user_data
);

typedef struct {
    lmmc_diagnostic_callback_t callback;
    void* user_data;
    lmmc_diagnostic_level_t minimum_level;
} lmmc_diagnostic_sink_t;

/** @brief 通过调用方持有的接收器发送一个诊断事件;空接收器直接完成. */
void lmmc_diagnostic_emit(
    const lmmc_diagnostic_sink_t* sink,
    const lmmc_diagnostic_t* diagnostic
);

#ifdef __cplusplus
}
#endif

#endif
