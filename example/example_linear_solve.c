#include <stdio.h>
#include "lmmc/lmmc.h"

int main(void) {
    lmmc_mat_t a = {0};
    lmmc_vec_t b = {0};
    lmmc_vec_t x = {0};
    size_t piv[3] = {0};
    lmmc_status_t st = LMMC_STATUS_OK;

    st = lmmc_mat_create(3, 3, &a);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create failed: %s\n", lmmc_status_string(st));
        return 1;
    }
    st = lmmc_vec_create(3, &b);
    if (st != LMMC_STATUS_OK) {
        printf("vec_create b failed: %s\n", lmmc_status_string(st));
        lmmc_mat_destroy(&a);
        return 1;
    }
    st = lmmc_vec_create(3, &x);
    if (st != LMMC_STATUS_OK) {
        printf("vec_create x failed: %s\n", lmmc_status_string(st));
        lmmc_vec_destroy(&b);
        lmmc_mat_destroy(&a);
        return 1;
    }

    a.data[0] = 3.0; a.data[1] = 2.0; a.data[2] = -1.0;
    a.data[3] = 2.0; a.data[4] = -2.0; a.data[5] = 4.0;
    a.data[6] = -1.0; a.data[7] = 0.5; a.data[8] = -1.0;

    b.data[0] = 1.0;
    b.data[1] = -2.0;
    b.data[2] = 0.0;

    st = lmmc_lu_decompose_inplace(&a, piv, NULL);
    if (st != LMMC_STATUS_OK) {
        printf("lu_decompose failed: %s\n", lmmc_status_string(st));
        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_mat_destroy(&a);
        return 1;
    }

    st = lmmc_lu_solve(&a, piv, &b, &x);
    if (st != LMMC_STATUS_OK) {
        printf("lu_solve failed: %s\n", lmmc_status_string(st));
        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_mat_destroy(&a);
        return 1;
    }

    printf("solution x = [%.6f, %.6f, %.6f]\n", x.data[0], x.data[1], x.data[2]);

    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&b);
    lmmc_mat_destroy(&a);
    return 0;
}
