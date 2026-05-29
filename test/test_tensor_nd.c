/**
 * @file test_tensor_nd.c
 * Unit tests for the N-D tensor API (lmmc_tensor_nd_t).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define ASSERT_OK(expr) do { \
    lmmc_status_t _st = (expr); \
    if (_st != LMMC_STATUS_OK) { \
        printf("FAIL at %s:%d: %s returned %d\n", __FILE__, __LINE__, #expr, (int)_st); \
        return 1; \
    } \
} while(0)

#define ASSERT_ERR(expr, expected) do { \
    lmmc_status_t _st = (expr); \
    if (_st != (expected)) { \
        printf("FAIL at %s:%d: %s returned %d, expected %d\n", __FILE__, __LINE__, #expr, (int)_st, (int)(expected)); \
        return 1; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        printf("FAIL at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while(0)

static int test_create_basic(void) {
    lmmc_tensor_nd_t t = {0};
    size_t dims[] = {2, 3, 4};

    ASSERT_OK(lmmc_tensor_create(3, dims, &t));
    ASSERT_TRUE(t.ndim == 3);
    ASSERT_TRUE(t.dims[0] == 2);
    ASSERT_TRUE(t.dims[1] == 3);
    ASSERT_TRUE(t.dims[2] == 4);
    ASSERT_TRUE(t.strides[0] == 12);
    ASSERT_TRUE(t.strides[1] == 4);
    ASSERT_TRUE(t.strides[2] == 1);
    ASSERT_TRUE(t.owns_data == 1);
    ASSERT_TRUE(t.data != NULL);

    lmmc_tensor_nd_destroy(&t);
    return 0;
}

static int test_create_1d(void) {
    lmmc_tensor_nd_t t = {0};
    size_t dims[] = {10};

    ASSERT_OK(lmmc_tensor_create(1, dims, &t));
    ASSERT_TRUE(t.ndim == 1);
    ASSERT_TRUE(t.dims[0] == 10);
    ASSERT_TRUE(t.strides[0] == 1);
    ASSERT_TRUE(t.owns_data == 1);

    lmmc_tensor_nd_destroy(&t);
    return 0;
}

static int test_create_max_ndim(void) {
    lmmc_tensor_nd_t t = {0};
    size_t dims[] = {2, 2, 2, 2, 2, 2, 2, 2};

    ASSERT_OK(lmmc_tensor_create(LMMC_TENSOR_MAX_NDIM, dims, &t));
    ASSERT_TRUE(t.ndim == 8);
    ASSERT_TRUE(t.strides[0] == 128);
    ASSERT_TRUE(t.strides[7] == 1);

    lmmc_tensor_nd_destroy(&t);
    return 0;
}

static int test_create_errors(void) {
    lmmc_tensor_nd_t t = {0};
    size_t dims[] = {2, 3, 4};
    size_t dims_zero[] = {2, 0, 4};

    /* ndim == 0 */
    ASSERT_ERR(lmmc_tensor_create(0, dims, &t), LMMC_STATUS_INVALID_ARGUMENT);
    /* ndim > MAX */
    ASSERT_ERR(lmmc_tensor_create(9, dims, &t), LMMC_STATUS_INVALID_ARGUMENT);
    /* NULL dims */
    ASSERT_ERR(lmmc_tensor_create(3, NULL, &t), LMMC_STATUS_INVALID_ARGUMENT);
    /* NULL out */
    ASSERT_ERR(lmmc_tensor_create(3, dims, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    /* Zero dimension */
    ASSERT_ERR(lmmc_tensor_create(3, dims_zero, &t), LMMC_STATUS_INVALID_ARGUMENT);

    return 0;
}

static int test_get_set(void) {
    lmmc_tensor_nd_t t = {0};
    size_t dims[] = {2, 3, 4};
    lmmc_real_t val = 0.0;

    ASSERT_OK(lmmc_tensor_create(3, dims, &t));

    /* Set and get */
    size_t idx[] = {1, 2, 3};
    ASSERT_OK(lmmc_tensor_set_nd(&t, idx, 42.0));
    ASSERT_OK(lmmc_tensor_get_nd(&t, idx, &val));
    ASSERT_TRUE(lmmc_test_nearly_equal(val, 42.0, 1e-15));

    /* Check zero-initialized */
    size_t idx2[] = {0, 0, 0};
    ASSERT_OK(lmmc_tensor_get_nd(&t, idx2, &val));
    ASSERT_TRUE(lmmc_test_nearly_equal(val, 0.0, 1e-15));

    lmmc_tensor_nd_destroy(&t);
    return 0;
}

static int test_get_set_errors(void) {
    lmmc_tensor_nd_t t = {0};
    size_t dims[] = {2, 3, 4};
    lmmc_real_t val = 0.0;

    ASSERT_OK(lmmc_tensor_create(3, dims, &t));

    /* Out of bounds */
    size_t idx_oob[] = {2, 0, 0};
    ASSERT_ERR(lmmc_tensor_get_nd(&t, idx_oob, &val), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_ERR(lmmc_tensor_set_nd(&t, idx_oob, 1.0), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL tensor */
    size_t idx[] = {0, 0, 0};
    ASSERT_ERR(lmmc_tensor_get_nd(NULL, idx, &val), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_ERR(lmmc_tensor_set_nd(NULL, idx, 1.0), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL idx */
    ASSERT_ERR(lmmc_tensor_get_nd(&t, NULL, &val), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_ERR(lmmc_tensor_set_nd(&t, NULL, 1.0), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL out */
    ASSERT_ERR(lmmc_tensor_get_nd(&t, idx, NULL), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&t);
    return 0;
}

static int test_permute_2d(void) {
    lmmc_tensor_nd_t t = {0};
    lmmc_tensor_nd_t p = {0};
    size_t dims[] = {2, 3};
    lmmc_real_t val = 0.0;

    ASSERT_OK(lmmc_tensor_create(2, dims, &t));

    /* Fill: t[i,j] = i*3 + j */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            size_t idx[] = {i, j};
            ASSERT_OK(lmmc_tensor_set_nd(&t, idx, (lmmc_real_t)(i * 3 + j)));
        }
    }

    /* Transpose: perm = {1, 0} */
    size_t perm[] = {1, 0};
    ASSERT_OK(lmmc_tensor_permute(&t, perm, &p));

    ASSERT_TRUE(p.ndim == 2);
    ASSERT_TRUE(p.dims[0] == 3);
    ASSERT_TRUE(p.dims[1] == 2);

    /* Check p[j,i] == t[i,j] */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            size_t idx_p[] = {j, i};
            ASSERT_OK(lmmc_tensor_get_nd(&p, idx_p, &val));
            ASSERT_TRUE(lmmc_test_nearly_equal(val, (lmmc_real_t)(i * 3 + j), 1e-15));
        }
    }

    lmmc_tensor_nd_destroy(&t);
    lmmc_tensor_nd_destroy(&p);
    return 0;
}

static int test_permute_errors(void) {
    lmmc_tensor_nd_t t = {0};
    lmmc_tensor_nd_t out = {0};
    size_t dims[] = {2, 3, 4};

    ASSERT_OK(lmmc_tensor_create(3, dims, &t));

    /* Invalid permutation: repeated value */
    size_t perm_bad[] = {0, 0, 1};
    ASSERT_ERR(lmmc_tensor_permute(&t, perm_bad, &out), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_TRUE(out.data == NULL);
    ASSERT_TRUE(out.owns_data == 0);

    /* Invalid permutation: out of range */
    size_t perm_oob[] = {0, 1, 3};
    ASSERT_ERR(lmmc_tensor_permute(&t, perm_oob, &out), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL args */
    size_t perm_ok[] = {2, 1, 0};
    ASSERT_ERR(lmmc_tensor_permute(NULL, perm_ok, &out), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_ERR(lmmc_tensor_permute(&t, NULL, &out), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_ERR(lmmc_tensor_permute(&t, perm_ok, NULL), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&t);
    return 0;
}

static int test_contract_matmul(void) {
    /* Contract two 2-D tensors along one axis = matrix multiply */
    lmmc_tensor_nd_t a = {0}, b = {0}, c = {0};
    size_t dims_a[] = {2, 3};
    size_t dims_b[] = {3, 4};
    lmmc_real_t val = 0.0;

    ASSERT_OK(lmmc_tensor_create(2, dims_a, &a));
    ASSERT_OK(lmmc_tensor_create(2, dims_b, &b));

    /* Fill a[i,j] = i*3+j+1, b[j,k] = j*4+k+1 */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            size_t idx[] = {i, j};
            ASSERT_OK(lmmc_tensor_set_nd(&a, idx, (lmmc_real_t)(i * 3 + j + 1)));
        }
    }
    for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 4; ++k) {
            size_t idx[] = {j, k};
            ASSERT_OK(lmmc_tensor_set_nd(&b, idx, (lmmc_real_t)(j * 4 + k + 1)));
        }
    }

    /* Contract: c[i,k] = sum_j a[i,j] * b[j,k] */
    size_t axes_a[] = {1};
    size_t axes_b[] = {0};
    ASSERT_OK(lmmc_tensor_contract(&a, &b, axes_a, axes_b, 1, &c));

    ASSERT_TRUE(c.ndim == 2);
    ASSERT_TRUE(c.dims[0] == 2);
    ASSERT_TRUE(c.dims[1] == 4);

    /* Verify against manual computation */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            lmmc_real_t expected = 0.0;
            for (size_t j = 0; j < 3; ++j) {
                expected += (lmmc_real_t)(i * 3 + j + 1) * (lmmc_real_t)(j * 4 + k + 1);
            }
            size_t idx[] = {i, k};
            ASSERT_OK(lmmc_tensor_get_nd(&c, idx, &val));
            ASSERT_TRUE(lmmc_test_nearly_equal(val, expected, 1e-10));
        }
    }

    lmmc_tensor_nd_destroy(&a);
    lmmc_tensor_nd_destroy(&b);
    lmmc_tensor_nd_destroy(&c);
    return 0;
}

static int test_contract_errors(void) {
    lmmc_tensor_nd_t a = {0}, b = {0}, out = {0};
    size_t dims_a[] = {2, 3};
    size_t dims_b[] = {4, 3};

    ASSERT_OK(lmmc_tensor_create(2, dims_a, &a));
    ASSERT_OK(lmmc_tensor_create(2, dims_b, &b));

    /* Dimension mismatch on contracted axes */
    size_t axes_a[] = {0};
    size_t axes_b[] = {0};
    ASSERT_ERR(lmmc_tensor_contract(&a, &b, axes_a, axes_b, 1, &out), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_TRUE(out.data == NULL);

    /* NULL args */
    ASSERT_ERR(lmmc_tensor_contract(NULL, &b, axes_a, axes_b, 1, &out), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_ERR(lmmc_tensor_contract(&a, NULL, axes_a, axes_b, 1, &out), LMMC_STATUS_INVALID_ARGUMENT);

    /* Axes out of range */
    size_t axes_oob[] = {5};
    ASSERT_ERR(lmmc_tensor_contract(&a, &b, axes_oob, axes_b, 1, &out), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&a);
    lmmc_tensor_nd_destroy(&b);
    return 0;
}

static int test_mode_n_product(void) {
    /* 3-D tensor mode-1 product with a matrix */
    lmmc_tensor_nd_t t = {0}, out = {0};
    lmmc_mat_t mat = {0};
    size_t dims[] = {2, 3, 4};
    lmmc_real_t val = 0.0;

    ASSERT_OK(lmmc_tensor_create(3, dims, &t));
    ASSERT_OK(lmmc_mat_create(5, 3, &mat));

    /* Fill tensor: t[i,j,k] = i*12 + j*4 + k + 1 */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                size_t idx[] = {i, j, k};
                ASSERT_OK(lmmc_tensor_set_nd(&t, idx, (lmmc_real_t)(i * 12 + j * 4 + k + 1)));
            }
        }
    }

    /* Fill matrix: mat[r,c] = r*3 + c + 1 */
    for (size_t r = 0; r < 5; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            mat.data[r * mat.stride + c] = (lmmc_real_t)(r * 3 + c + 1);
        }
    }

    /* Mode-1 product: out[i,r,k] = sum_j mat[r,j] * t[i,j,k] */
    ASSERT_OK(lmmc_tensor_mode_n_product(&t, &mat, 1, &out));

    ASSERT_TRUE(out.ndim == 3);
    ASSERT_TRUE(out.dims[0] == 2);
    ASSERT_TRUE(out.dims[1] == 5);
    ASSERT_TRUE(out.dims[2] == 4);

    /* Verify a few elements */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t r = 0; r < 5; ++r) {
            for (size_t k = 0; k < 4; ++k) {
                lmmc_real_t expected = 0.0;
                for (size_t j = 0; j < 3; ++j) {
                    expected += (lmmc_real_t)(r * 3 + j + 1) * (lmmc_real_t)(i * 12 + j * 4 + k + 1);
                }
                size_t idx[] = {i, r, k};
                ASSERT_OK(lmmc_tensor_get_nd(&out, idx, &val));
                ASSERT_TRUE(lmmc_test_nearly_equal(val, expected, 1e-10));
            }
        }
    }

    lmmc_tensor_nd_destroy(&t);
    lmmc_tensor_nd_destroy(&out);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_mode_n_product_errors(void) {
    lmmc_tensor_nd_t t = {0}, out = {0};
    lmmc_mat_t mat = {0};
    size_t dims[] = {2, 3, 4};

    ASSERT_OK(lmmc_tensor_create(3, dims, &t));
    ASSERT_OK(lmmc_mat_create(5, 7, &mat)); /* cols != dims[mode] */

    /* Mode out of range */
    ASSERT_ERR(lmmc_tensor_mode_n_product(&t, &mat, 3, &out), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_TRUE(out.data == NULL);

    /* Column mismatch */
    ASSERT_ERR(lmmc_tensor_mode_n_product(&t, &mat, 0, &out), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL args */
    ASSERT_ERR(lmmc_tensor_mode_n_product(NULL, &mat, 0, &out), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_ERR(lmmc_tensor_mode_n_product(&t, NULL, 0, &out), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&t);
    lmmc_mat_destroy(&mat);
    return 0;
}

static int test_reshape_view(void) {
    lmmc_tensor_nd_t t = {0}, view = {0};
    size_t dims[] = {2, 3, 4};
    lmmc_real_t val = 0.0;

    ASSERT_OK(lmmc_tensor_create(3, dims, &t));

    /* Fill with sequential values */
    for (size_t i = 0; i < 24; ++i) {
        t.data[i] = (lmmc_real_t)i;
    }

    /* Reshape to 4x6 */
    size_t new_dims[] = {4, 6};
    ASSERT_OK(lmmc_tensor_nd_reshape_view(&t, 2, new_dims, &view));

    ASSERT_TRUE(view.ndim == 2);
    ASSERT_TRUE(view.dims[0] == 4);
    ASSERT_TRUE(view.dims[1] == 6);
    ASSERT_TRUE(view.owns_data == 0);
    ASSERT_TRUE(view.data == t.data);

    /* Check element access */
    size_t idx[] = {0, 0};
    ASSERT_OK(lmmc_tensor_get_nd(&view, idx, &val));
    ASSERT_TRUE(lmmc_test_nearly_equal(val, 0.0, 1e-15));

    size_t idx2[] = {3, 5};
    ASSERT_OK(lmmc_tensor_get_nd(&view, idx2, &val));
    ASSERT_TRUE(lmmc_test_nearly_equal(val, 23.0, 1e-15));

    lmmc_tensor_nd_destroy(&t);
    /* view doesn't own data, destroy is safe */
    lmmc_tensor_nd_destroy(&view);
    return 0;
}

static int test_reshape_view_errors(void) {
    lmmc_tensor_nd_t t = {0}, view = {0};
    size_t dims[] = {2, 3, 4};

    ASSERT_OK(lmmc_tensor_create(3, dims, &t));

    /* Wrong total size */
    size_t bad_dims[] = {5, 5};
    ASSERT_ERR(lmmc_tensor_nd_reshape_view(&t, 2, bad_dims, &view), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL args */
    size_t good_dims[] = {4, 6};
    ASSERT_ERR(lmmc_tensor_nd_reshape_view(NULL, 2, good_dims, &view), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_ERR(lmmc_tensor_nd_reshape_view(&t, 2, NULL, &view), LMMC_STATUS_INVALID_ARGUMENT);
    ASSERT_ERR(lmmc_tensor_nd_reshape_view(&t, 2, good_dims, NULL), LMMC_STATUS_INVALID_ARGUMENT);

    /* ndim == 0 */
    ASSERT_ERR(lmmc_tensor_nd_reshape_view(&t, 0, good_dims, &view), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&t);
    return 0;
}

static int test_destroy_null_safe(void) {
    /* Should not crash */
    lmmc_tensor_nd_destroy(NULL);

    lmmc_tensor_nd_t t = {0};
    lmmc_tensor_nd_destroy(&t); /* zero-initialized, no data */
    return 0;
}

static int test_contract_scalar(void) {
    /* Full contraction of two vectors = dot product */
    lmmc_tensor_nd_t a = {0}, b = {0}, c = {0};
    size_t dims[] = {4};
    lmmc_real_t val = 0.0;

    ASSERT_OK(lmmc_tensor_create(1, dims, &a));
    ASSERT_OK(lmmc_tensor_create(1, dims, &b));

    /* a = [1, 2, 3, 4], b = [5, 6, 7, 8] */
    for (size_t i = 0; i < 4; ++i) {
        size_t idx[] = {i};
        ASSERT_OK(lmmc_tensor_set_nd(&a, idx, (lmmc_real_t)(i + 1)));
        ASSERT_OK(lmmc_tensor_set_nd(&b, idx, (lmmc_real_t)(i + 5)));
    }

    size_t axes_a[] = {0};
    size_t axes_b[] = {0};
    ASSERT_OK(lmmc_tensor_contract(&a, &b, axes_a, axes_b, 1, &c));

    /* Result should be scalar: 1*5 + 2*6 + 3*7 + 4*8 = 70 */
    ASSERT_TRUE(c.ndim == 1);
    ASSERT_TRUE(c.dims[0] == 1);
    size_t idx[] = {0};
    ASSERT_OK(lmmc_tensor_get_nd(&c, idx, &val));
    ASSERT_TRUE(lmmc_test_nearly_equal(val, 70.0, 1e-10));

    lmmc_tensor_nd_destroy(&a);
    lmmc_tensor_nd_destroy(&b);
    lmmc_tensor_nd_destroy(&c);
    return 0;
}

int main(void) {
    int failures = 0;

    lmmc_init();
    printf("=== N-D Tensor Unit Tests ===\n");

#define RUN_TEST(fn) do { \
    printf("  %s ... ", #fn); \
    if (fn() == 0) { printf("PASS\n"); } \
    else { printf("FAIL\n"); failures++; } \
} while(0)

    RUN_TEST(test_create_basic);
    RUN_TEST(test_create_1d);
    RUN_TEST(test_create_max_ndim);
    RUN_TEST(test_create_errors);
    RUN_TEST(test_get_set);
    RUN_TEST(test_get_set_errors);
    RUN_TEST(test_permute_2d);
    RUN_TEST(test_permute_errors);
    RUN_TEST(test_contract_matmul);
    RUN_TEST(test_contract_errors);
    RUN_TEST(test_contract_scalar);
    RUN_TEST(test_mode_n_product);
    RUN_TEST(test_mode_n_product_errors);
    RUN_TEST(test_reshape_view);
    RUN_TEST(test_reshape_view_errors);
    RUN_TEST(test_destroy_null_safe);

#undef RUN_TEST

    printf("\n%d test(s) failed.\n", failures);
    lmmc_deinit();
    return failures > 0 ? 1 : 0;
}
