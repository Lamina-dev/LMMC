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
        default:
            return "unknown";
    }
}
