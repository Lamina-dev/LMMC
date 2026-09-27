/**
 * @file example_tensor_ops.c
 * @brief 演示 LMMC 中 tensor ops 相关接口的使用。
 */
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

static int create_tensors(lmmc_tensor3_t* a, lmmc_tensor3_t* b, lmmc_tensor3_t* out) {
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_tensor3_create(2, 2, 2, a);
    if (st != LMMC_STATUS_OK) {
        printf("tensor3_create(a) failed: %s\n", lmmc_status_string(st));
        return 1;
    }
    st = lmmc_tensor3_create(2, 2, 2, b);
    if (st != LMMC_STATUS_OK) {
        printf("tensor3_create(b) failed: %s\n", lmmc_status_string(st));
        return 1;
    }
    st = lmmc_tensor3_create(2, 2, 2, out);
    if (st != LMMC_STATUS_OK) {
        printf("tensor3_create(out) failed: %s\n", lmmc_status_string(st));
        return 1;
    }
    return 0;
}

static int initialize_tensor_values(lmmc_tensor3_t* a) {
    for (size_t i = 0; i < 8; ++i) {
        lmmc_status_t st = lmmc_tensor3_set(a, i / 4, (i / 2) % 2, i % 2, (double)(i + 1));
        if (st != LMMC_STATUS_OK) {
            printf("fill a failed: %s\n", lmmc_status_string(st));
            return 1;
        }
    }
    return 0;
}

static int demonstrate_tensor_add(const lmmc_tensor3_t* a, lmmc_tensor3_t* b, lmmc_tensor3_t* out) {
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_tensor3_fill(b, 1.0);
    if (st != LMMC_STATUS_OK) {
        printf("tensor_fill(b) failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_tensor3_add(a, b, out);
    if (st != LMMC_STATUS_OK) {
        printf("tensor_add failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    {
        double v = 0.0;
        st = lmmc_tensor3_get(out, 1, 1, 1, &v);
        if (st != LMMC_STATUS_OK) {
            printf("tensor_get(out) failed: %s\n", lmmc_status_string(st));
            return 1;
        }
        printf("out(1,1,1) = %.3f (expected 9.000)\n", v);
    }
    return 0;
}

static int demonstrate_axis_sum(const lmmc_tensor3_t* a, lmmc_mat_t* sum_axis0) {
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_mat_create(2, 2, sum_axis0);
    if (st != LMMC_STATUS_OK) {
        printf("mat_create(sum_axis0) failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_tensor3_sum_axis(a, 0, sum_axis0);
    if (st != LMMC_STATUS_OK) {
        printf("tensor_sum_axis(axis=0) failed: %s\n", lmmc_status_string(st));
        return 1;
    }
    print_mat_2d("sum_axis0", sum_axis0);
    return 0;
}

static int demonstrate_reshape_alias(lmmc_tensor3_t* a, lmmc_tensor3_t* reshaped) {
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_tensor3_reshape_view(a, 1, 4, 2, reshaped);
    if (st != LMMC_STATUS_OK) {
        printf("tensor_reshape_view failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_tensor3_set(reshaped, 0, 3, 1, 99.0);
    if (st != LMMC_STATUS_OK) {
        printf("set on reshaped view failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    {
        double v = 0.0;
        st = lmmc_tensor3_get(a, 1, 1, 1, &v);
        if (st != LMMC_STATUS_OK) {
            printf("tensor_get(a) after reshape write failed: %s\n", lmmc_status_string(st));
            return 1;
        }
        printf("a(1,1,1) after reshape view write = %.3f\n", v);
    }
    return 0;
}

static int demonstrate_slice_alias(lmmc_tensor3_t* a, lmmc_tensor3_t* sliced) {
    lmmc_status_t st = LMMC_STATUS_OK;
    st = lmmc_tensor3_slice_view(a, 0, 2, 0, 1, 0, 2, sliced);
    if (st != LMMC_STATUS_OK) {
        printf("tensor_slice_view failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    st = lmmc_tensor3_set(sliced, 0, 0, 0, 42.0);
    if (st != LMMC_STATUS_OK) {
        printf("set on sliced view failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    {
        double v = 0.0;
        st = lmmc_tensor3_get(a, 0, 0, 0, &v);
        if (st != LMMC_STATUS_OK) {
            printf("tensor_get(a) after slice write failed: %s\n", lmmc_status_string(st));
            return 1;
        }
        printf("a(0,0,0) after slice view write = %.3f\n", v);
    }
    return 0;
}

int main(void) {
    lmmc_tensor3_t a = {0};
    lmmc_tensor3_t b = {0};
    lmmc_tensor3_t out = {0};
    lmmc_tensor3_t reshaped = {0};
    lmmc_tensor3_t sliced = {0};
    lmmc_mat_t sum_axis0 = {0};
    int rc = 0;

    rc = create_tensors(&a, &b, &out);
    if (rc != 0) {
            goto cleanup;
        }
    rc = initialize_tensor_values(&a);
    if (rc != 0) {
            goto cleanup;
        }
    rc = demonstrate_tensor_add(&a, &b, &out);
    if (rc != 0) {
            goto cleanup;
        }
    rc = demonstrate_axis_sum(&a, &sum_axis0);
    if (rc != 0) {
            goto cleanup;
        }
    rc = demonstrate_reshape_alias(&a, &reshaped);
    if (rc != 0) {
            goto cleanup;
        }
    rc = demonstrate_slice_alias(&a, &sliced);
    if (rc != 0) {
            goto cleanup;
        }

cleanup:
    lmmc_mat_destroy(&sum_axis0);
    lmmc_tensor3_destroy(&sliced);
    lmmc_tensor3_destroy(&reshaped);
    lmmc_tensor3_destroy(&out);
    lmmc_tensor3_destroy(&b);
    lmmc_tensor3_destroy(&a);
    return rc;
}
