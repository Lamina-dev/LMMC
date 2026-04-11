#ifndef LMMC_NUMERIC_H
#define LMMC_NUMERIC_H

#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LMMC_DEFAULT_ABS_TOL 1e-12
#define LMMC_DEFAULT_REL_TOL 1e-10

lmmc_status_t lmmc_double_nearly_equal_tol(
    double a,
    double b,
    double abs_tol,
    double rel_tol,
    int* out_equal
);

lmmc_status_t lmmc_double_nearly_equal(double a, double b, int* out_equal);

#ifdef __cplusplus
}
#endif

#endif
