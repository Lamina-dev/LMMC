/**
 * @file status.c
 * @brief 状态码到字符串的映射实现。
 */
#include "lmmc/config.h"
#include "lmmc/status.h"

const char* lmmc_status_string(lmmc_status_t status) {
    switch (status) {
        case LMMC_STATUS_OK:
            return "ok";
        case LMMC_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case LMMC_STATUS_DIMENSION_MISMATCH:
            return "dimension mismatch";
        case LMMC_STATUS_ALLOCATION_FAILED:
            return "allocation failed";
        case LMMC_STATUS_SINGULAR_MATRIX:
            return "singular matrix";
        case LMMC_STATUS_NOT_IMPLEMENTED:
            return "not implemented";
        case LMMC_STATUS_NUMERICAL_FAILURE:
            return "numerical failure";
        case LMMC_STATUS_NOT_POSITIVE_DEFINITE:
            return "not positive definite";
        case LMMC_STATUS_CONVERGENCE_FAILED:
            return "convergence failed";
        case LMMC_STATUS_OUT_OF_RANGE:
            return "out of range";
        case LMMC_STATUS_INDEX_OUT_OF_BOUNDS:
            return "index out of bounds";
        case LMMC_STATUS_WARNING_MAX_DEPTH:
            return "warning: max depth reached";
        default:
            return "unknown";
    }
}
