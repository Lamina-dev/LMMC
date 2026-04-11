#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"

int main(void) {
    lmmc_status_t st = LMMC_STATUS_OK;
    int equal = 0;
    int rc = 0;

    st = lmmc_double_nearly_equal(0.1 + 0.2, 0.3, &equal);
    if (st != LMMC_STATUS_OK || equal != 1) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal(1.0, 1.0 + 1e-6, &equal);
    if (st != LMMC_STATUS_OK || equal != 0) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal_tol(1.0, 1.0 + 1e-6, 1e-9, 1e-5, &equal);
    if (st != LMMC_STATUS_OK || equal != 1) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal_tol(1.0, 1.0, -1.0, 1e-9, &equal);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal_tol(1.0, 1.0, 1e-9, 1e-9, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal_tol(NAN, 1.0, 1e-9, 1e-9, &equal);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        rc = 1;
        goto done;
    }

    st = lmmc_double_nearly_equal_tol(INFINITY, INFINITY, 1e-9, 1e-9, &equal);
    if (st != LMMC_STATUS_OK || equal != 1) {
        rc = 1;
        goto done;
    }

done:
    if (rc != 0) {
        printf("numeric test failed\n");
    }
    return rc;
}
