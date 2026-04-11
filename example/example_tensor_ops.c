#include <stdio.h>
#include "lmmc/lmmc.h"

static void print_mat_2d(const char* name, const lmmc_mat_t* mat) {
    size_t i = 0;
    size_t j = 0;
    printf("%s =\n", name);
    for (i = 0; i < mat->rows; ++i) {
        printf("  [");
        for (j = 0; j < mat->cols; ++j) {
            printf("%8.3f", mat->data[i * mat->stride + j]);
            if (j + 1 < mat->cols) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

int main(void) {
    lmmc_tensor_t a = {0};
    lmmc_tensor_t b = {0};
    lmmc_tensor_t out = {0};
    lmmc_tensor_t reshaped = {0};
    lmmc_tensor_t sliced = {0};
    lmmc_mat_t sum_axis0 = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_tensor3_create(2, 2, 2, &a);
    if (st != LMMC_STATUS_OK) {
        printf("tensor3_create(a) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor3_create(2, 2, 2, &b);
    if (st != LMMC_STATUS_OK) {
        printf("tensor3_create(b) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    st = lmmc_tensor3_create(2, 2, 2, &out);
    if (st != LMMC_STATUS_OK) {
        printf("tensor3_create(out) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_set(&a, 0, 0, 0, 1.0);
    st = (st == LMMC_STATUS_OK) ? lmmc_tensor_set(&a, 0, 0, 1, 2.0) : st;
    st = (st == LMMC_STATUS_OK) ? lmmc_tensor_set(&a, 0, 1, 0, 3.0) : st;
    st = (st == LMMC_STATUS_OK) ? lmmc_tensor_set(&a, 0, 1, 1, 4.0) : st;
    st = (st == LMMC_STATUS_OK) ? lmmc_tensor_set(&a, 1, 0, 0, 5.0) : st;
    st = (st == LMMC_STATUS_OK) ? lmmc_tensor_set(&a, 1, 0, 1, 6.0) : st;
    st = (st == LMMC_STATUS_OK) ? lmmc_tensor_set(&a, 1, 1, 0, 7.0) : st;
    st = (st == LMMC_STATUS_OK) ? lmmc_tensor_set(&a, 1, 1, 1, 8.0) : st;
    if (st != LMMC_STATUS_OK) {
        printf("fill a failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_fill(&b, 1.0);
    if (st != LMMC_STATUS_OK) {
        printf("tensor_fill(b) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_add(&a, &b, &out);
    if (st != LMMC_STATUS_OK) {
        printf("tensor_add failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    {
        double v = 0.0;
        st = lmmc_tensor_get(&out, 1, 1, 1, &v);
        if (st != LMMC_STATUS_OK) {
            printf("tensor_get(out) failed: %s\n", lmmc_status_string(st));
            rc = 1;
            goto cleanup;
        }
        printf("out(1,1,1) = %.3f (expected 9.000)\n", v);
    }

    st = lmmc_mat_create(2, 2, &sum_axis0);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create(sum_axis0) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_sum_axis(&a, 0, &sum_axis0);
    if (st != LMMC_STATUS_OK) {
        printf("tensor_sum_axis(axis=0) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }
    print_mat_2d("sum_axis0", &sum_axis0);

    st = lmmc_tensor_reshape_view(&a, 1, 4, 2, &reshaped);
    if (st != LMMC_STATUS_OK) {
        printf("tensor_reshape_view failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_set(&reshaped, 0, 3, 1, 99.0);
    if (st != LMMC_STATUS_OK) {
        printf("set on reshaped view failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    {
        double v = 0.0;
        st = lmmc_tensor_get(&a, 1, 1, 1, &v);
        if (st != LMMC_STATUS_OK) {
            printf("tensor_get(a) after reshape write failed: %s\n", lmmc_status_string(st));
            rc = 1;
            goto cleanup;
        }
        printf("a(1,1,1) after reshape view write = %.3f\n", v);
    }

    st = lmmc_tensor_slice_view(&a, 0, 2, 0, 1, 0, 2, &sliced);
    if (st != LMMC_STATUS_OK) {
        printf("tensor_slice_view failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor_set(&sliced, 0, 0, 0, 42.0);
    if (st != LMMC_STATUS_OK) {
        printf("set on sliced view failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    {
        double v = 0.0;
        st = lmmc_tensor_get(&a, 0, 0, 0, &v);
        if (st != LMMC_STATUS_OK) {
            printf("tensor_get(a) after slice write failed: %s\n", lmmc_status_string(st));
            rc = 1;
            goto cleanup;
        }
        printf("a(0,0,0) after slice view write = %.3f\n", v);
    }

cleanup:
    lmmc_mat_destroy(&sum_axis0);
    lmmc_tensor_destroy(&sliced);
    lmmc_tensor_destroy(&reshaped);
    lmmc_tensor_destroy(&out);
    lmmc_tensor_destroy(&b);
    lmmc_tensor_destroy(&a);
    return rc;
}
