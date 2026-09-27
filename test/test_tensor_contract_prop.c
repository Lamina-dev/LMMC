/**
 * @file test_tensor_contract_prop.c
 * 张量缩并精度属性测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/**
 * @brief Compute the Frobenius norm of an N-D tensor.
 */
static double tensor_frobenius_norm(const lmmc_tensor_nd_t *t) {
    size_t total = 1;
    for (size_t i = 0; i < t->ndim; ++i) {
        total *= t->dims[i];
    }
    double sum = 0.0;
    /* Iterate over all elements using multi-index */
    size_t idx[LMMC_TENSOR_MAX_NDIM] = {0};
    for (size_t flat = 0; flat < total; ++flat) {
        /* Compute linear offset from idx and strides */
        size_t offset = 0;
        for (size_t d = 0; d < t->ndim; ++d) {
            offset += idx[d] * t->strides[d];
        }
        double v = t->data[offset];
        sum += v * v;

        /* Increment multi-index */
        for (size_t d = t->ndim; d > 0; --d) {
            idx[d - 1]++;
            if (idx[d - 1] < t->dims[d - 1]) {
                break;
            }
            idx[d - 1] = 0;
        }
    }
    return sqrt(sum);
}

/**
 * @brief Fill a tensor with random values in [-1, 1].
 */
static void fill_random(lmmc_tensor_nd_t *t, lmmc_rng_t *rng) {
    size_t total = 1;
    for (size_t i = 0; i < t->ndim; ++i) {
        total *= t->dims[i];
    }
    /* For row-major contiguous tensors, data is sequential */
    for (size_t i = 0; i < total; ++i) {
        lmmc_real_t val;
        assert_int_equal(lmmc_rng_uniform(rng, -1.0, 1.0, &val), LMMC_STATUS_OK);
        t->data[i] = val;
    }
}

/**
 * @brief Compute the reference contraction using explicit nested loops.
 *
 * Given tensors a and b, contracted along axes_a/axes_b (naxes pairs),
 * compute the output tensor element by element using a brute-force approach.
 *
 * The output has dimensions: free dims of a (in order) followed by free dims of b (in order).
 *
 * @param a       First input tensor.
 * @param b       Second input tensor.
 * @param axes_a  Axes of a to contract.
 * @param axes_b  Axes of b to contract.
 * @param naxes   Number of contracted axis pairs.
 * @param ref_out Pre-allocated output tensor (same shape as expected contraction result).
 */
static size_t collect_free_axes(size_t ndim, const size_t *axes, size_t naxes,
                                size_t *free_axes) {
    int contracted[LMMC_TENSOR_MAX_NDIM] = {0};
    size_t count = 0;
    for (size_t i = 0; i < naxes; ++i) {
        contracted[axes[i]] = 1;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (!contracted[i]) {
            free_axes[count++] = i;
        }
    }
    return count;
}

static size_t contraction_offset(const lmmc_tensor_nd_t *tensor,
                                 const size_t *free_axes, size_t nfree,
                                 const size_t *free_index, const size_t *axes,
                                 size_t naxes, const size_t *contract_index) {
    size_t index[LMMC_TENSOR_MAX_NDIM] = {0};
    for (size_t i = 0; i < nfree; ++i) {
        index[free_axes[i]] = free_index[i];
    }
    for (size_t i = 0; i < naxes; ++i) {
        index[axes[i]] = contract_index[i];
    }
    size_t offset = 0;
    for (size_t d = 0; d < tensor->ndim; ++d) {
        offset += index[d] * tensor->strides[d];
    }
    return offset;
}

static void reference_contraction(
    const lmmc_tensor_nd_t *a, const lmmc_tensor_nd_t *b,
    const size_t *axes_a, const size_t *axes_b, size_t naxes,
    lmmc_tensor_nd_t *ref_out) {
    size_t a_free[LMMC_TENSOR_MAX_NDIM];
    size_t b_free[LMMC_TENSOR_MAX_NDIM];
    size_t n_a_free = collect_free_axes(a->ndim, axes_a, naxes, a_free);
    size_t n_b_free = collect_free_axes(b->ndim, axes_b, naxes, b_free);

    /* Compute contraction size */
    size_t contract_size = 1;
    for (size_t i = 0; i < naxes; ++i) {
        contract_size *= a->dims[axes_a[i]];
    }

    /* Total output elements */
    size_t out_total = 1;
    for (size_t i = 0; i < ref_out->ndim; ++i) {
        out_total *= ref_out->dims[i];
    }

    /* Iterate over all output elements */
    size_t out_idx[LMMC_TENSOR_MAX_NDIM] = {0};
    for (size_t flat_out = 0; flat_out < out_total; ++flat_out) {
        double sum = 0.0;

        /* Iterate over contracted indices */
        size_t contract_idx[LMMC_TENSOR_MAX_NDIM] = {0};
        for (size_t ck = 0; ck < contract_size; ++ck) {
            size_t a_offset = contraction_offset(a, a_free, n_a_free, out_idx,
                                                 axes_a, naxes, contract_idx);
            size_t b_offset = contraction_offset(b, b_free, n_b_free,
                                                 out_idx + n_a_free, axes_b,
                                                 naxes, contract_idx);

            sum += a->data[a_offset] * b->data[b_offset];

            /* Increment contract_idx */
            if (naxes > 0) {
                size_t ci = naxes;
                while (ci > 0) {
                    --ci;
                    contract_idx[ci]++;
                    if (contract_idx[ci] < a->dims[axes_a[ci]]) {
                        break;
                    }
                    contract_idx[ci] = 0;
                }
            }
        }

        /* Write to output */
        size_t out_offset = 0;
        for (size_t d = 0; d < ref_out->ndim; ++d) {
            out_offset += out_idx[d] * ref_out->strides[d];
        }
        ref_out->data[out_offset] = sum;

        /* Increment out_idx */
        for (size_t d = ref_out->ndim; d > 0; --d) {
            out_idx[d - 1]++;
            if (out_idx[d - 1] < ref_out->dims[d - 1]) {
                break;
            }
            out_idx[d - 1] = 0;
        }
    }
}

/**
 * @brief Compare two tensors element-wise, returning the max absolute difference.
 */
static double max_abs_diff(const lmmc_tensor_nd_t *x, const lmmc_tensor_nd_t *y) {
    size_t total = 1;
    for (size_t i = 0; i < x->ndim; ++i) {
        total *= x->dims[i];
    }
    double max_diff = 0.0;
    size_t idx[LMMC_TENSOR_MAX_NDIM] = {0};
    for (size_t flat = 0; flat < total; ++flat) {
        size_t x_offset = 0, y_offset = 0;
        for (size_t d = 0; d < x->ndim; ++d) {
            x_offset += idx[d] * x->strides[d];
            y_offset += idx[d] * y->strides[d];
        }
        double diff = fabs(x->data[x_offset] - y->data[y_offset]);
        if (diff > max_diff) {
            max_diff = diff;
        }

        for (size_t d = x->ndim; d > 0; --d) {
            idx[d - 1]++;
            if (idx[d - 1] < x->dims[d - 1]) {
                break;
            }
            idx[d - 1] = 0;
        }
    }
    return max_diff;
}

/**
 * @brief Test configuration for a single contraction trial.
 */
typedef struct {
    size_t a_ndim;
    size_t a_dims[LMMC_TENSOR_MAX_NDIM];
    size_t b_ndim;
    size_t b_dims[LMMC_TENSOR_MAX_NDIM];
    size_t naxes;
    size_t axes_a[LMMC_TENSOR_MAX_NDIM];
    size_t axes_b[LMMC_TENSOR_MAX_NDIM];
    const char *description;
} contraction_case_t;

/**
 * @brief Run a single contraction property test.
 *
 */

/* Define test cases covering 2-D, 3-D, and 4-D tensors */
static const contraction_case_t cases[] = {
    /* 2-D x 2-D: matrix multiply (contract axis 1 of a with axis 0 of b) */
    {
        .a_ndim = 2, .a_dims = {3, 4}, .b_ndim = 2, .b_dims = {4, 5}, .naxes = 1, .axes_a = {1}, .axes_b = {0}, .description = "2D*2D matmul (3x4)*(4x5)"},
    /* 2-D x 2-D: larger matrix multiply */
    {
        .a_ndim = 2, .a_dims = {5, 6}, .b_ndim = 2, .b_dims = {6, 7}, .naxes = 1, .axes_a = {1}, .axes_b = {0}, .description = "2D*2D matmul (5x6)*(6x7)"},
    /* 2-D x 2-D: dot product (full contraction) */
    {
        .a_ndim = 2, .a_dims = {3, 4}, .b_ndim = 2, .b_dims = {3, 4}, .naxes = 2, .axes_a = {0, 1}, .axes_b = {0, 1}, .description = "2D*2D full contraction (3x4)*(3x4)"},
    /* 3-D x 2-D: contract one axis */
    {
        .a_ndim = 3, .a_dims = {2, 3, 4}, .b_ndim = 2, .b_dims = {4, 5}, .naxes = 1, .axes_a = {2}, .axes_b = {0}, .description = "3D*2D contract axis2-axis0 (2x3x4)*(4x5)"},
    /* 3-D x 3-D: contract one axis */
    {
        .a_ndim = 3, .a_dims = {2, 3, 4}, .b_ndim = 3, .b_dims = {5, 4, 3}, .naxes = 1, .axes_a = {2}, .axes_b = {1}, .description = "3D*3D contract axis2-axis1 (2x3x4)*(5x4x3)"},
    /* 3-D x 3-D: contract two axes */
    {
        .a_ndim = 3, .a_dims = {2, 3, 4}, .b_ndim = 3, .b_dims = {4, 5, 3}, .naxes = 2, .axes_a = {1, 2}, .axes_b = {2, 0}, .description = "3D*3D contract 2 axes (2x3x4)*(4x5x3)"},
    /* 4-D x 2-D: contract one axis */
    {
        .a_ndim = 4, .a_dims = {2, 3, 4, 2}, .b_ndim = 2, .b_dims = {4, 3}, .naxes = 1, .axes_a = {2}, .axes_b = {0}, .description = "4D*2D contract axis2-axis0 (2x3x4x2)*(4x3)"},
    /* 4-D x 3-D: contract two axes */
    {
        .a_ndim = 4, .a_dims = {2, 3, 4, 2}, .b_ndim = 3, .b_dims = {4, 2, 5}, .naxes = 2, .axes_a = {2, 3}, .axes_b = {0, 1}, .description = "4D*3D contract 2 axes (2x3x4x2)*(4x2x5)"},
    /* 4-D x 4-D: contract one axis */
    {
        .a_ndim = 4, .a_dims = {2, 3, 2, 4}, .b_ndim = 4, .b_dims = {4, 2, 3, 2}, .naxes = 1, .axes_a = {3}, .axes_b = {0}, .description = "4D*4D contract axis3-axis0 (2x3x2x4)*(4x2x3x2)"},
    /* 4-D x 4-D: contract two axes */
    {
        .a_ndim = 4, .a_dims = {2, 3, 4, 2}, .b_ndim = 4, .b_dims = {4, 2, 3, 2}, .naxes = 2, .axes_a = {2, 3}, .axes_b = {0, 1}, .description = "4D*4D contract 2 axes (2x3x4x2)*(4x2x3x2)"},
    /* 3-D x 3-D: full contraction (scalar result) */
    {
        .a_ndim = 3, .a_dims = {2, 3, 4}, .b_ndim = 3, .b_dims = {2, 3, 4}, .naxes = 3, .axes_a = {0, 1, 2}, .axes_b = {0, 1, 2}, .description = "3D*3D full contraction (2x3x4)*(2x3x4)"},
    /* 2-D x 2-D: contract non-adjacent axes */
    {
        .a_ndim = 2, .a_dims = {4, 5}, .b_ndim = 2, .b_dims = {6, 4}, .naxes = 1, .axes_a = {0}, .axes_b = {1}, .description = "2D*2D contract axis0-axis1 (4x5)*(6x4)"},
};

typedef struct {
    const contraction_case_t *test_case;
    lmmc_rng_t *rng;
    lmmc_tensor_nd_t a;
    lmmc_tensor_nd_t b;
    lmmc_tensor_nd_t result;
    lmmc_tensor_nd_t reference;
} contraction_fixture_t;

static int teardown(void **state) {
    contraction_fixture_t *fixture = *state;
    lmmc_tensor_nd_destroy(&fixture->reference);
    lmmc_tensor_nd_destroy(&fixture->result);
    lmmc_tensor_nd_destroy(&fixture->b);
    lmmc_tensor_nd_destroy(&fixture->a);
    lmmc_rng_destroy(fixture->rng);
    free(fixture);
    *state = NULL;
    return 0;
}

static int setup(void **state) {
    const contraction_case_t *test_case = *state;
    contraction_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    fixture->test_case = test_case;
    *state = fixture;
    lmmc_status_t st = lmmc_rng_create(&fixture->rng);
    if (st == LMMC_STATUS_OK) {
        st = lmmc_rng_seed(fixture->rng, UINT64_C(0x54434F4E));
    }
    if (st != LMMC_STATUS_OK) {
        teardown(state);
        return st;
    }
    return 0;
}

static int group_setup(void **state) {
    (void)state;
    assert_int_equal(lmmc_init(), LMMC_STATUS_OK);
    return 0;
}

static int group_teardown(void **state) {
    (void)state;
    assert_int_equal(lmmc_deinit(), LMMC_STATUS_OK);
    return 0;
}

static void test_contraction(void **state) {
    contraction_fixture_t *fixture = *state;
    const contraction_case_t *tc = fixture->test_case;
    for (int trial = 0; trial < 10; ++trial) {
        assert_int_equal(lmmc_tensor_nd_create(tc->a_ndim, tc->a_dims, &fixture->a), LMMC_STATUS_OK);
        assert_int_equal(lmmc_tensor_nd_create(tc->b_ndim, tc->b_dims, &fixture->b), LMMC_STATUS_OK);
        fill_random(&fixture->a, fixture->rng);
        fill_random(&fixture->b, fixture->rng);
        assert_int_equal(lmmc_tensor_nd_contract(&fixture->a, &fixture->b,
                                                 tc->axes_a, tc->axes_b, tc->naxes, &fixture->result),
                         LMMC_STATUS_OK);
        assert_int_equal(lmmc_tensor_nd_create(fixture->result.ndim,
                                               fixture->result.dims, &fixture->reference),
                         LMMC_STATUS_OK);
        reference_contraction(&fixture->a, &fixture->b, tc->axes_a,
                              tc->axes_b, tc->naxes, &fixture->reference);
        const double norm_a = tensor_frobenius_norm(&fixture->a);
        const double norm_b = tensor_frobenius_norm(&fixture->b);
        const double tolerance = 1e-10 * (1.0 + norm_a * norm_b);
        const double diff = max_abs_diff(&fixture->result, &fixture->reference);
        assert_true(diff <= tolerance);
        lmmc_tensor_nd_destroy(&fixture->reference);
        lmmc_tensor_nd_destroy(&fixture->result);
        lmmc_tensor_nd_destroy(&fixture->b);
        lmmc_tensor_nd_destroy(&fixture->a);
    }
}

int main(void) {
    struct CMUnitTest tests[sizeof(cases) / sizeof(cases[0])];
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        tests[i] = (struct CMUnitTest){
            cases[i].description, test_contraction, setup, teardown, (void *)&cases[i]};
    }
    return cmocka_run_group_tests(tests, group_setup, group_teardown);
}
