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

/* ---------- Helpers ---------- */

/**
 * @brief Compute the Frobenius norm of an N-D tensor.
 */
static double tensor_frobenius_norm(const lmmc_tensor_nd_t* t) {
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
            if (idx[d - 1] < t->dims[d - 1]) break;
            idx[d - 1] = 0;
        }
    }
    return sqrt(sum);
}

/**
 * @brief Fill a tensor with random values in [-1, 1].
 */
static void fill_random(lmmc_tensor_nd_t* t, lmmc_rng_t* rng) {
    size_t total = 1;
    for (size_t i = 0; i < t->ndim; ++i) {
        total *= t->dims[i];
    }
    /* For row-major contiguous tensors, data is sequential */
    for (size_t i = 0; i < total; ++i) {
        lmmc_real_t val;
        lmmc_rng_uniform(rng, -1.0, 1.0, &val);
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
static void reference_contraction(
    const lmmc_tensor_nd_t* a, const lmmc_tensor_nd_t* b,
    const size_t* axes_a, const size_t* axes_b, size_t naxes,
    lmmc_tensor_nd_t* ref_out)
{
    /* Identify free axes */
    int a_contracted[LMMC_TENSOR_MAX_NDIM] = {0};
    int b_contracted[LMMC_TENSOR_MAX_NDIM] = {0};
    size_t a_free[LMMC_TENSOR_MAX_NDIM];
    size_t b_free[LMMC_TENSOR_MAX_NDIM];
    size_t n_a_free = 0, n_b_free = 0;

    for (size_t i = 0; i < naxes; ++i) {
        a_contracted[axes_a[i]] = 1;
        b_contracted[axes_b[i]] = 1;
    }
    for (size_t i = 0; i < a->ndim; ++i) {
        if (!a_contracted[i]) a_free[n_a_free++] = i;
    }
    for (size_t i = 0; i < b->ndim; ++i) {
        if (!b_contracted[i]) b_free[n_b_free++] = i;
    }

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
            /* Build full index for a */
            size_t a_idx[LMMC_TENSOR_MAX_NDIM] = {0};
            for (size_t i = 0; i < n_a_free; ++i) {
                a_idx[a_free[i]] = out_idx[i];
            }
            for (size_t i = 0; i < naxes; ++i) {
                a_idx[axes_a[i]] = contract_idx[i];
            }

            /* Build full index for b */
            size_t b_idx[LMMC_TENSOR_MAX_NDIM] = {0};
            for (size_t i = 0; i < n_b_free; ++i) {
                b_idx[b_free[i]] = out_idx[n_a_free + i];
            }
            for (size_t i = 0; i < naxes; ++i) {
                b_idx[axes_b[i]] = contract_idx[i];
            }

            /* Compute linear offsets */
            size_t a_offset = 0, b_offset = 0;
            for (size_t d = 0; d < a->ndim; ++d) {
                a_offset += a_idx[d] * a->strides[d];
            }
            for (size_t d = 0; d < b->ndim; ++d) {
                b_offset += b_idx[d] * b->strides[d];
            }

            sum += a->data[a_offset] * b->data[b_offset];

            /* Increment contract_idx */
            if (naxes > 0) {
                size_t ci = naxes;
                while (ci > 0) {
                    --ci;
                    contract_idx[ci]++;
                    if (contract_idx[ci] < a->dims[axes_a[ci]]) break;
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
            if (out_idx[d - 1] < ref_out->dims[d - 1]) break;
            out_idx[d - 1] = 0;
        }
    }
}

/**
 * @brief Compare two tensors element-wise, returning the max absolute difference.
 */
static double max_abs_diff(const lmmc_tensor_nd_t* x, const lmmc_tensor_nd_t* y) {
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
        if (diff > max_diff) max_diff = diff;

        for (size_t d = x->ndim; d > 0; --d) {
            idx[d - 1]++;
            if (idx[d - 1] < x->dims[d - 1]) break;
            idx[d - 1] = 0;
        }
    }
    return max_diff;
}

/* ---------- Test Cases ---------- */

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
    const char* description;
} contraction_case_t;

/**
 * @brief Run a single contraction property test.
 *
 * @return 0 on success, 1 on failure.
 */
static int run_contraction_trial(lmmc_rng_t* rng, const contraction_case_t* tc, int trial) {
    lmmc_tensor_nd_t a = {0}, b = {0}, result = {0}, ref = {0};
    lmmc_status_t st;
    int rc = 0;

    /* Create tensors */
    st = lmmc_tensor_create(tc->a_ndim, tc->a_dims, &a);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] %s trial %d: create a failed (%d)\n", tc->description, trial, (int)st);
        return 1;
    }
    st = lmmc_tensor_create(tc->b_ndim, tc->b_dims, &b);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] %s trial %d: create b failed (%d)\n", tc->description, trial, (int)st);
        lmmc_tensor_nd_destroy(&a);
        return 1;
    }

    /* Fill with random data */
    fill_random(&a, rng);
    fill_random(&b, rng);

    /* Perform contraction */
    st = lmmc_tensor_contract(&a, &b, tc->axes_a, tc->axes_b, tc->naxes, &result);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] %s trial %d: tensor_contract failed (%d)\n", tc->description, trial, (int)st);
        rc = 1;
        goto cleanup;
    }

    /* Create reference tensor with same shape as result */
    st = lmmc_tensor_create(result.ndim, result.dims, &ref);
    if (st != LMMC_STATUS_OK) {
        printf("  [FAIL] %s trial %d: create ref failed (%d)\n", tc->description, trial, (int)st);
        rc = 1;
        goto cleanup;
    }

    /* Compute reference via nested loops */
    reference_contraction(&a, &b, tc->axes_a, tc->axes_b, tc->naxes, &ref);

    /* Compute norms */
    double norm_a = tensor_frobenius_norm(&a);
    double norm_b = tensor_frobenius_norm(&b);
    double tolerance = 1e-10 * (1.0 + norm_a * norm_b);

    /* Compare */
    double diff = max_abs_diff(&result, &ref);
    if (diff > tolerance) {
        printf("  [FAIL] %s trial %d: max|diff| = %.6e > tol = %.6e (||a||=%.3e, ||b||=%.3e)\n",
               tc->description, trial, diff, tolerance, norm_a, norm_b);
        rc = 1;
        goto cleanup;
    }

cleanup:
    lmmc_tensor_nd_destroy(&ref);
    lmmc_tensor_nd_destroy(&result);
    lmmc_tensor_nd_destroy(&b);
    lmmc_tensor_nd_destroy(&a);
    return rc;
}

int main(void) {
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    int failures = 0;
    int total_trials = 0;

    lmmc_init();

    printf("=== Property Test: Tensor Contraction ===\n");
    printf("output matches unrolled nested-loop reference\n");
    printf("Tolerance: 1e-10 * (1 + ||a||_F * ||b||_F)\n");

    /* 固定种子保证属性测试可复现。 */
    const uint64_t seed = UINT64_C(0x54434F4E);
    st = lmmc_rng_create(&rng);
    if (st != LMMC_STATUS_OK) {
        printf("FATAL: Failed to create RNG\n");
        lmmc_deinit();
        return 1;
    }
    lmmc_rng_seed(rng, seed);

    /* Define test cases covering 2-D, 3-D, and 4-D tensors */
    contraction_case_t cases[] = {
        /* 2-D x 2-D: matrix multiply (contract axis 1 of a with axis 0 of b) */
        {
            .a_ndim = 2, .a_dims = {3, 4},
            .b_ndim = 2, .b_dims = {4, 5},
            .naxes = 1, .axes_a = {1}, .axes_b = {0},
            .description = "2D*2D matmul (3x4)*(4x5)"
        },
        /* 2-D x 2-D: larger matrix multiply */
        {
            .a_ndim = 2, .a_dims = {5, 6},
            .b_ndim = 2, .b_dims = {6, 7},
            .naxes = 1, .axes_a = {1}, .axes_b = {0},
            .description = "2D*2D matmul (5x6)*(6x7)"
        },
        /* 2-D x 2-D: dot product (full contraction) */
        {
            .a_ndim = 2, .a_dims = {3, 4},
            .b_ndim = 2, .b_dims = {3, 4},
            .naxes = 2, .axes_a = {0, 1}, .axes_b = {0, 1},
            .description = "2D*2D full contraction (3x4)*(3x4)"
        },
        /* 3-D x 2-D: contract one axis */
        {
            .a_ndim = 3, .a_dims = {2, 3, 4},
            .b_ndim = 2, .b_dims = {4, 5},
            .naxes = 1, .axes_a = {2}, .axes_b = {0},
            .description = "3D*2D contract axis2-axis0 (2x3x4)*(4x5)"
        },
        /* 3-D x 3-D: contract one axis */
        {
            .a_ndim = 3, .a_dims = {2, 3, 4},
            .b_ndim = 3, .b_dims = {5, 4, 3},
            .naxes = 1, .axes_a = {2}, .axes_b = {1},
            .description = "3D*3D contract axis2-axis1 (2x3x4)*(5x4x3)"
        },
        /* 3-D x 3-D: contract two axes */
        {
            .a_ndim = 3, .a_dims = {2, 3, 4},
            .b_ndim = 3, .b_dims = {4, 5, 3},
            .naxes = 2, .axes_a = {1, 2}, .axes_b = {2, 0},
            .description = "3D*3D contract 2 axes (2x3x4)*(4x5x3)"
        },
        /* 4-D x 2-D: contract one axis */
        {
            .a_ndim = 4, .a_dims = {2, 3, 4, 2},
            .b_ndim = 2, .b_dims = {4, 3},
            .naxes = 1, .axes_a = {2}, .axes_b = {0},
            .description = "4D*2D contract axis2-axis0 (2x3x4x2)*(4x3)"
        },
        /* 4-D x 3-D: contract two axes */
        {
            .a_ndim = 4, .a_dims = {2, 3, 4, 2},
            .b_ndim = 3, .b_dims = {4, 2, 5},
            .naxes = 2, .axes_a = {2, 3}, .axes_b = {0, 1},
            .description = "4D*3D contract 2 axes (2x3x4x2)*(4x2x5)"
        },
        /* 4-D x 4-D: contract one axis */
        {
            .a_ndim = 4, .a_dims = {2, 3, 2, 4},
            .b_ndim = 4, .b_dims = {4, 2, 3, 2},
            .naxes = 1, .axes_a = {3}, .axes_b = {0},
            .description = "4D*4D contract axis3-axis0 (2x3x2x4)*(4x2x3x2)"
        },
        /* 4-D x 4-D: contract two axes */
        {
            .a_ndim = 4, .a_dims = {2, 3, 4, 2},
            .b_ndim = 4, .b_dims = {4, 2, 3, 2},
            .naxes = 2, .axes_a = {2, 3}, .axes_b = {0, 1},
            .description = "4D*4D contract 2 axes (2x3x4x2)*(4x2x3x2)"
        },
        /* 3-D x 3-D: full contraction (scalar result) */
        {
            .a_ndim = 3, .a_dims = {2, 3, 4},
            .b_ndim = 3, .b_dims = {2, 3, 4},
            .naxes = 3, .axes_a = {0, 1, 2}, .axes_b = {0, 1, 2},
            .description = "3D*3D full contraction (2x3x4)*(2x3x4)"
        },
        /* 2-D x 2-D: contract non-adjacent axes */
        {
            .a_ndim = 2, .a_dims = {4, 5},
            .b_ndim = 2, .b_dims = {6, 4},
            .naxes = 1, .axes_a = {0}, .axes_b = {1},
            .description = "2D*2D contract axis0-axis1 (4x5)*(6x4)"
        },
    };

    size_t num_cases = sizeof(cases) / sizeof(cases[0]);
    int trials_per_case = 10;

    for (size_t ci = 0; ci < num_cases; ++ci) {
        printf("Testing: %s (%d trials)...\n", cases[ci].description, trials_per_case);
        for (int t = 0; t < trials_per_case; ++t) {
            total_trials++;
            if (run_contraction_trial(rng, &cases[ci], t + 1) != 0) {
                failures++;
            }
        }
    }

    printf("\n=== Results ===\n");
    printf("Total trials: %d\n", total_trials);
    printf("Passed: %d\n", total_trials - failures);
    printf("Failed: %d\n", failures);

    if (failures == 0) {
        printf("\nProperty test PASSED: Tensor contraction matches reference.\n");
    } else {
        printf("\nProperty test FAILED: %d/%d trials violated the property.\n", failures, total_trials);
    }

    lmmc_rng_destroy(rng);
    lmmc_deinit();
    return (failures == 0) ? 0 : 1;
}
