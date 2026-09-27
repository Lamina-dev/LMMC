/**
 * @file status.c
 * @brief 状态码到字符串的映射实现。
 */
#include "lmmc/config.h"
#include "lmmc/status.h"

const char* lmmc_status_string(lmmc_status_t status) {
    static const char* const names[] = {
        [LMMC_STATUS_OK] = "ok",
        [LMMC_STATUS_INVALID_ARGUMENT] = "invalid argument",
        [LMMC_STATUS_DIMENSION_MISMATCH] = "dimension mismatch",
        [LMMC_STATUS_ALLOCATION_FAILED] = "allocation failed",
        [LMMC_STATUS_SINGULAR_MATRIX] = "singular matrix",
        [LMMC_STATUS_NOT_IMPLEMENTED] = "not implemented",
        [LMMC_STATUS_NUMERICAL_FAILURE] = "numerical failure",
        [LMMC_STATUS_NOT_POSITIVE_DEFINITE] = "not positive definite",
        [LMMC_STATUS_CONVERGENCE_FAILED] = "convergence failed",
        [LMMC_STATUS_OUT_OF_RANGE] = "out of range",
        [LMMC_STATUS_INDEX_OUT_OF_BOUNDS] = "index out of bounds",
        [LMMC_STATUS_WARNING_MAX_DEPTH] = "warning: max depth reached",
        [LMMC_STATUS_EMPTY_INPUT] = "empty input",
        [LMMC_STATUS_UNIT_STRIP_TYPE_MISMATCH] = "unit strip type mismatch",
        [LMMC_STATUS_UNIT_STRIP_OVERFLOW] = "unit strip overflow",
        [LMMC_STATUS_UNIT_STRIP_INVALID] = "unit strip invalid",
        [LMMC_STATUS_UNIT_STRIP_LEGACY_SYNTAX] = "unit strip legacy syntax",
        [LMMC_STATUS_NOT_INITIALIZED] = "not initialized",
        [LMMC_STATUS_BUSY] = "busy",
        [LMMC_STATUS_REFERENCE_LIMIT] = "reference limit"
    };
    if ((unsigned int)status >= sizeof(names) / sizeof(names[0])) return "unknown";
    return names[status];
}
