#include <math.h>
#include "lmmc/numeric.h"

static double lmmc_absd(double x) {
    return x < 0.0 ? -x : x;
}

static double lmmc_maxd(double a, double b) {
    return a > b ? a : b;
}

lmmc_status_t lmmc_double_nearly_equal_tol(
    double a,
    double b,
    double abs_tol,
    double rel_tol,
    int* out_equal
) {
    double diff = 0.0;
    double scale = 0.0;
    double threshold = 0.0;

    if (out_equal == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!(abs_tol >= 0.0) || !(rel_tol >= 0.0) || !isfinite(abs_tol) || !isfinite(rel_tol)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (isnan(a) || isnan(b)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (isinf(a) || isinf(b)) {
        *out_equal = (a == b) ? 1 : 0;
        return LMMC_STATUS_OK;
    }

    diff = lmmc_absd(a - b);
    scale = lmmc_maxd(lmmc_absd(a), lmmc_absd(b));
    threshold = abs_tol + rel_tol * scale;

    if (isnan(diff) || isnan(threshold)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (isinf(threshold)) {
        *out_equal = 1;
        return LMMC_STATUS_OK;
    }

    *out_equal = (diff <= threshold) ? 1 : 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_double_nearly_equal(double a, double b, int* out_equal) {
    return lmmc_double_nearly_equal_tol(a, b, LMMC_DEFAULT_ABS_TOL, LMMC_DEFAULT_REL_TOL, out_equal);
}
