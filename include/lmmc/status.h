#ifndef LMMC_STATUS_H
#define LMMC_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LMMC_STATUS_OK = 0,
    LMMC_STATUS_INVALID_ARGUMENT = 1,
    LMMC_STATUS_DIMENSION_MISMATCH = 2,
    LMMC_STATUS_ALLOCATION_FAILED = 3,
    LMMC_STATUS_SINGULAR_MATRIX = 4,
    LMMC_STATUS_NOT_IMPLEMENTED = 5,
    LMMC_STATUS_NUMERICAL_FAILURE = 6
} lmmc_status_t;

const char* lmmc_status_string(lmmc_status_t status);

#ifdef __cplusplus
}
#endif

#endif
