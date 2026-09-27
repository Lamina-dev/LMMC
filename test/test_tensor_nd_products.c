#include <stdlib.h>
/**
 * @file test_tensor_nd_products.c
 * N-D tensor permutation and product tests.
 */
#include <stdint.h>
#include "lmmc/lmmc.h"
#include "test_common.h"
#include "internal_test_hooks.h"

typedef struct {
    lmmc_tensor_nd_t tensor_nd_t;
    lmmc_tensor_nd_t tensor_nd_p;
    lmmc_tensor_nd_t tensor_nd_out;
    lmmc_tensor_nd_t tensor_nd_a;
    lmmc_tensor_nd_t tensor_nd_b;
    lmmc_tensor_nd_t tensor_nd_c;
    lmmc_mat_t mat_mat;
} test_fixture_t;

static int setup(void **state) {
    test_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    lmmc_memory_fail_reset_for_test();
    test_fixture_t *fixture = *state;
    lmmc_mat_destroy(&fixture->mat_mat);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_c);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_b);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_a);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_out);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_p);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
    free(fixture);
    return 0;
}

static void assert_owning_tensor_unchanged(
    const lmmc_tensor_nd_t* tensor, lmmc_real_t* data, size_t ndim,
    const size_t* dims, const lmmc_real_t* values, size_t value_count) {
    assert_true(tensor->data == data);
    assert_int_equal(tensor->owns_data, 1);
    assert_int_equal(tensor->ndim, ndim);
    for (size_t i = 0; i < ndim; ++i) {
        assert_int_equal(tensor->dims[i], dims[i]);
    }
    for (size_t i = 0; i < value_count; ++i) {
        assert_true(lmmc_test_nearly_equal(tensor->data[i], values[i], 1e-15));
    }
}

static void test_permute_2d(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {2, 3};
    lmmc_real_t val = 0.0;

    assert_int_equal(lmmc_tensor_nd_create(2, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);

    /* Fill: t[i,j] = i*3 + j */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            size_t idx[] = {i, j};
            assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_t, idx, (lmmc_real_t)(i * 3 + j)), LMMC_STATUS_OK);
        }
    }

    /* Transpose: perm = {1, 0} */
    size_t perm[] = {1, 0};
    assert_int_equal(lmmc_tensor_nd_permute(&fixture->tensor_nd_t, perm, &fixture->tensor_nd_p), LMMC_STATUS_OK);

    assert_true(fixture->tensor_nd_p.ndim == 2);
    assert_true(fixture->tensor_nd_p.dims[0] == 3);
    assert_true(fixture->tensor_nd_p.dims[1] == 2);

    /* Check p[j,i] == t[i,j] */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            size_t idx_p[] = {j, i};
            assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_p, idx_p, &val), LMMC_STATUS_OK);
            assert_true(lmmc_test_nearly_equal(val, (lmmc_real_t)(i * 3 + j), 1e-15));
        }
    }

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_p);
}

static void test_permute_errors(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {2, 3, 4};

    assert_int_equal(lmmc_tensor_nd_create(3, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);

    /* Invalid permutation: repeated value */
    size_t perm_bad[] = {0, 0, 1};
    assert_int_equal(lmmc_tensor_nd_permute(&fixture->tensor_nd_t, perm_bad, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(fixture->tensor_nd_out.data == NULL);
    assert_true(fixture->tensor_nd_out.owns_data == 0);

    /* Invalid permutation: out of range */
    size_t perm_oob[] = {0, 1, 3};
    assert_int_equal(lmmc_tensor_nd_permute(&fixture->tensor_nd_t, perm_oob, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL args */
    size_t perm_ok[] = {2, 1, 0};
    assert_int_equal(lmmc_tensor_nd_permute(NULL, perm_ok, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_tensor_nd_permute(&fixture->tensor_nd_t, NULL, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_tensor_nd_permute(&fixture->tensor_nd_t, perm_ok, NULL), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
}

static void test_permute_rejects_owning_self_alias(void **state) {
    test_fixture_t *fixture = *state;
    const size_t dims[] = {2, 3};
    const size_t perm[] = {1, 0};
    const lmmc_real_t values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    assert_int_equal(
        lmmc_tensor_nd_create(2, dims, &fixture->tensor_nd_t),
        LMMC_STATUS_OK);
    for (size_t i = 0; i < 6; ++i) {
        fixture->tensor_nd_t.data[i] = values[i];
    }
    lmmc_real_t* const data = fixture->tensor_nd_t.data;

    assert_int_equal(
        lmmc_tensor_nd_permute(
            &fixture->tensor_nd_t, perm, &fixture->tensor_nd_t),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_owning_tensor_unchanged(
        &fixture->tensor_nd_t, data, 2, dims, values, 6);
}

static void test_permute_overflow_preserves_output(void **state) {
    (void)state;
    const size_t perm[] = {0, 1};
    lmmc_real_t input_values[] = {2.0, 3.0};
    lmmc_tensor_nd_t input = {
        .ndim = 2,
        .dims = {SIZE_MAX / 2 + 1, 2},
        .strides = {0, 1},
        .data = input_values,
        .owns_data = 0,
    };
    lmmc_real_t sentinel[] = {19.0};
    lmmc_tensor_nd_t out = {
        .ndim = 1,
        .dims = {1},
        .strides = {1},
        .data = sentinel,
        .owns_data = 0,
    };

    lmmc_memory_fail_after_for_test(0);
    assert_int_equal(
        lmmc_tensor_nd_permute(&input, perm, &out),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(out.data == sentinel);
    assert_int_equal(out.owns_data, 0);
    assert_int_equal(out.ndim, 1);
    assert_int_equal(out.dims[0], 1);
    assert_true(lmmc_test_nearly_equal(sentinel[0], 19.0, 1e-15));
}

static void test_contract_matmul(void **state) {
    test_fixture_t *fixture = *state;
    /* Contract two 2-D tensors along one axis = matrix multiply */

    size_t dims_a[] = {2, 3};
    size_t dims_b[] = {3, 4};
    lmmc_real_t val = 0.0;

    assert_int_equal(lmmc_tensor_nd_create(2, dims_a, &fixture->tensor_nd_a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_create(2, dims_b, &fixture->tensor_nd_b), LMMC_STATUS_OK);

    /* Fill a[i,j] = i*3+j+1, b[j,k] = j*4+k+1 */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            size_t idx[] = {i, j};
            assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_a, idx, (lmmc_real_t)(i * 3 + j + 1)), LMMC_STATUS_OK);
        }
    }
    for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 4; ++k) {
            size_t idx[] = {j, k};
            assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_b, idx, (lmmc_real_t)(j * 4 + k + 1)), LMMC_STATUS_OK);
        }
    }

    /* Contract: c[i,k] = sum_j a[i,j] * b[j,k] */
    size_t axes_a[] = {1};
    size_t axes_b[] = {0};
    assert_int_equal(lmmc_tensor_nd_contract(&fixture->tensor_nd_a, &fixture->tensor_nd_b, axes_a, axes_b, 1, &fixture->tensor_nd_c), LMMC_STATUS_OK);

    assert_true(fixture->tensor_nd_c.ndim == 2);
    assert_true(fixture->tensor_nd_c.dims[0] == 2);
    assert_true(fixture->tensor_nd_c.dims[1] == 4);

    /* Verify against manual computation */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            lmmc_real_t expected = 0.0;
            for (size_t j = 0; j < 3; ++j) {
                expected += (lmmc_real_t)(i * 3 + j + 1) * (lmmc_real_t)(j * 4 + k + 1);
            }
            size_t idx[] = {i, k};
            assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_c, idx, &val), LMMC_STATUS_OK);
            assert_true(lmmc_test_nearly_equal(val, expected, 1e-10));
        }
    }

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_a);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_b);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_c);
}

static void test_contract_errors(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims_a[] = {2, 3};
    size_t dims_b[] = {4, 3};

    assert_int_equal(lmmc_tensor_nd_create(2, dims_a, &fixture->tensor_nd_a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_create(2, dims_b, &fixture->tensor_nd_b), LMMC_STATUS_OK);

    /* Dimension mismatch on contracted axes */
    size_t axes_a[] = {0};
    size_t axes_b[] = {0};
    assert_int_equal(lmmc_tensor_nd_contract(&fixture->tensor_nd_a, &fixture->tensor_nd_b, axes_a, axes_b, 1, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(fixture->tensor_nd_out.data == NULL);

    /* NULL args */
    assert_int_equal(lmmc_tensor_nd_contract(NULL, &fixture->tensor_nd_b, axes_a, axes_b, 1, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_tensor_nd_contract(&fixture->tensor_nd_a, NULL, axes_a, axes_b, 1, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);

    /* Axes out of range */
    size_t axes_oob[] = {5};
    assert_int_equal(lmmc_tensor_nd_contract(&fixture->tensor_nd_a, &fixture->tensor_nd_b, axes_oob, axes_b, 1, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_a);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_b);
}

static void test_contract_rejects_input_aliases(void **state) {
    test_fixture_t *fixture = *state;
    const size_t dims[] = {2, 2};
    const size_t axes_a[] = {1};
    const size_t axes_b[] = {0};
    const lmmc_real_t values_a[] = {1.0, 2.0, 3.0, 4.0};
    const lmmc_real_t values_b[] = {5.0, 6.0, 7.0, 8.0};

    assert_int_equal(
        lmmc_tensor_nd_create(2, dims, &fixture->tensor_nd_a),
        LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_tensor_nd_create(2, dims, &fixture->tensor_nd_b),
        LMMC_STATUS_OK);
    for (size_t i = 0; i < 4; ++i) {
        fixture->tensor_nd_a.data[i] = values_a[i];
        fixture->tensor_nd_b.data[i] = values_b[i];
    }
    lmmc_real_t* const data_a = fixture->tensor_nd_a.data;
    lmmc_real_t* const data_b = fixture->tensor_nd_b.data;

    assert_int_equal(
        lmmc_tensor_nd_contract(
            &fixture->tensor_nd_a, &fixture->tensor_nd_b,
            axes_a, axes_b, 1, &fixture->tensor_nd_a),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_owning_tensor_unchanged(
        &fixture->tensor_nd_a, data_a, 2, dims, values_a, 4);

    assert_int_equal(
        lmmc_tensor_nd_contract(
            &fixture->tensor_nd_a, &fixture->tensor_nd_b,
            axes_a, axes_b, 1, &fixture->tensor_nd_b),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_owning_tensor_unchanged(
        &fixture->tensor_nd_b, data_b, 2, dims, values_b, 4);
}

static void test_contract_overflow_precedes_allocation_failure(void **state) {
    test_fixture_t *fixture = *state;
    const size_t axes[] = {0, 1};
    lmmc_real_t a_values[] = {2.0, 3.0};
    lmmc_real_t b_values[] = {5.0, 7.0};
    lmmc_tensor_nd_t overflow_a = {
        .ndim = 2,
        .dims = {SIZE_MAX / 2 + 1, 2},
        .strides = {0, 1},
        .data = a_values,
        .owns_data = 0,
    };
    lmmc_tensor_nd_t overflow_b = {
        .ndim = 2,
        .dims = {SIZE_MAX / 2 + 1, 2},
        .strides = {0, 1},
        .data = b_values,
        .owns_data = 0,
    };
    lmmc_real_t sentinel[] = {17.0};
    lmmc_tensor_nd_t out = {
        .ndim = 1,
        .dims = {1},
        .strides = {1},
        .data = sentinel,
        .owns_data = 0,
    };

    lmmc_memory_fail_after_for_test(0);
    assert_int_equal(
        lmmc_tensor_nd_contract(
            &overflow_a, &overflow_b, axes, axes, 2, &out),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(out.data == sentinel);
    assert_int_equal(out.owns_data, 0);
    assert_true(lmmc_test_nearly_equal(sentinel[0], 17.0, 1e-15));

    lmmc_memory_fail_reset_for_test();
    const size_t valid_dims[] = {2, 2};
    assert_int_equal(
        lmmc_tensor_nd_create(2, valid_dims, &fixture->tensor_nd_a),
        LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_tensor_nd_create(2, valid_dims, &fixture->tensor_nd_b),
        LMMC_STATUS_OK);
    lmmc_memory_fail_after_for_test(0);
    assert_int_equal(
        lmmc_tensor_nd_contract(
            &fixture->tensor_nd_a, &fixture->tensor_nd_b,
            axes, axes, 2, &out),
        LMMC_STATUS_ALLOCATION_FAILED);
    assert_true(out.data == sentinel);
    assert_int_equal(out.owns_data, 0);
    assert_int_equal(out.ndim, 1);
    assert_int_equal(out.dims[0], 1);
    assert_true(lmmc_test_nearly_equal(sentinel[0], 17.0, 1e-15));
}

static void test_mode_n_product(void **state) {
    test_fixture_t *fixture = *state;
    /* 3-D tensor mode-1 product with a matrix */

    size_t dims[] = {2, 3, 4};
    lmmc_real_t val = 0.0;

    assert_int_equal(lmmc_tensor_nd_create(3, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(5, 3, &fixture->mat_mat), LMMC_STATUS_OK);

    /* Fill tensor: t[i,j,k] = i*12 + j*4 + k + 1 */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                size_t idx[] = {i, j, k};
                assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_t, idx, (lmmc_real_t)(i * 12 + j * 4 + k + 1)), LMMC_STATUS_OK);
            }
        }
    }

    /* Fill matrix: mat[r,c] = r*3 + c + 1 */
    for (size_t r = 0; r < 5; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            fixture->mat_mat.data[r * fixture->mat_mat.stride + c] = (lmmc_real_t)(r * 3 + c + 1);
        }
    }

    /* Mode-1 product: out[i,r,k] = sum_j mat[r,j] * t[i,j,k] */
    assert_int_equal(lmmc_tensor_nd_mode_n_product(&fixture->tensor_nd_t, &fixture->mat_mat, 1, &fixture->tensor_nd_out), LMMC_STATUS_OK);

    assert_true(fixture->tensor_nd_out.ndim == 3);
    assert_true(fixture->tensor_nd_out.dims[0] == 2);
    assert_true(fixture->tensor_nd_out.dims[1] == 5);
    assert_true(fixture->tensor_nd_out.dims[2] == 4);

    /* Verify a few elements */
    for (size_t i = 0; i < 2; ++i) {
        for (size_t r = 0; r < 5; ++r) {
            for (size_t k = 0; k < 4; ++k) {
                lmmc_real_t expected = 0.0;
                for (size_t j = 0; j < 3; ++j) {
                    expected += (lmmc_real_t)(r * 3 + j + 1) * (lmmc_real_t)(i * 12 + j * 4 + k + 1);
                }
                size_t idx[] = {i, r, k};
                assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_out, idx, &val), LMMC_STATUS_OK);
                assert_true(lmmc_test_nearly_equal(val, expected, 1e-10));
            }
        }
    }

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_out);
    lmmc_mat_destroy(&fixture->mat_mat);
}

static void test_mode_n_product_errors(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {2, 3, 4};

    assert_int_equal(lmmc_tensor_nd_create(3, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(5, 7, &fixture->mat_mat), LMMC_STATUS_OK); /* cols != dims[mode] */

    /* Mode out of range */
    assert_int_equal(lmmc_tensor_nd_mode_n_product(&fixture->tensor_nd_t, &fixture->mat_mat, 3, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(fixture->tensor_nd_out.data == NULL);

    /* Column mismatch */
    assert_int_equal(lmmc_tensor_nd_mode_n_product(&fixture->tensor_nd_t, &fixture->mat_mat, 0, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL args */
    assert_int_equal(lmmc_tensor_nd_mode_n_product(NULL, &fixture->mat_mat, 0, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_tensor_nd_mode_n_product(&fixture->tensor_nd_t, NULL, 0, &fixture->tensor_nd_out), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
    lmmc_mat_destroy(&fixture->mat_mat);
}

static void test_mode_n_product_rejects_owning_self_alias(void **state) {
    test_fixture_t *fixture = *state;
    const size_t dims[] = {2, 3};
    const lmmc_real_t values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    assert_int_equal(
        lmmc_tensor_nd_create(2, dims, &fixture->tensor_nd_t),
        LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_mat_create(2, 2, &fixture->mat_mat),
        LMMC_STATUS_OK);
    for (size_t i = 0; i < 6; ++i) {
        fixture->tensor_nd_t.data[i] = values[i];
    }
    fixture->mat_mat.data[0] = 1.0;
    fixture->mat_mat.data[1] = 0.0;
    fixture->mat_mat.data[fixture->mat_mat.stride] = 0.0;
    fixture->mat_mat.data[fixture->mat_mat.stride + 1] = 1.0;
    lmmc_real_t* const data = fixture->tensor_nd_t.data;

    assert_int_equal(
        lmmc_tensor_nd_mode_n_product(
            &fixture->tensor_nd_t, &fixture->mat_mat, 0,
            &fixture->tensor_nd_t),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_owning_tensor_unchanged(
        &fixture->tensor_nd_t, data, 2, dims, values, 6);
}

static void test_contract_scalar(void **state) {
    test_fixture_t *fixture = *state;
    /* Full contraction of two vectors = dot product */

    size_t dims[] = {4};
    lmmc_real_t val = 0.0;

    assert_int_equal(lmmc_tensor_nd_create(1, dims, &fixture->tensor_nd_a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_create(1, dims, &fixture->tensor_nd_b), LMMC_STATUS_OK);

    /* a = [1, 2, 3, 4], b = [5, 6, 7, 8] */
    for (size_t i = 0; i < 4; ++i) {
        size_t idx[] = {i};
        assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_a, idx, (lmmc_real_t)(i + 1)), LMMC_STATUS_OK);
        assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_b, idx, (lmmc_real_t)(i + 5)), LMMC_STATUS_OK);
    }

    size_t axes_a[] = {0};
    size_t axes_b[] = {0};
    assert_int_equal(lmmc_tensor_nd_contract(&fixture->tensor_nd_a, &fixture->tensor_nd_b, axes_a, axes_b, 1, &fixture->tensor_nd_c), LMMC_STATUS_OK);

    /* Result should be scalar: 1*5 + 2*6 + 3*7 + 4*8 = 70 */
    assert_true(fixture->tensor_nd_c.ndim == 1);
    assert_true(fixture->tensor_nd_c.dims[0] == 1);
    size_t idx[] = {0};
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_c, idx, &val), LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 70.0, 1e-10));

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_a);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_b);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_c);
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

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_permute_2d, setup, teardown),
        cmocka_unit_test_setup_teardown(test_permute_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_permute_rejects_owning_self_alias, setup, teardown),
        cmocka_unit_test_setup_teardown(test_permute_overflow_preserves_output, setup, teardown),
        cmocka_unit_test_setup_teardown(test_contract_matmul, setup, teardown),
        cmocka_unit_test_setup_teardown(test_contract_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_contract_rejects_input_aliases, setup, teardown),
        cmocka_unit_test_setup_teardown(test_contract_overflow_precedes_allocation_failure, setup, teardown),
        cmocka_unit_test_setup_teardown(test_mode_n_product, setup, teardown),
        cmocka_unit_test_setup_teardown(test_mode_n_product_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_mode_n_product_rejects_owning_self_alias, setup, teardown),
        cmocka_unit_test_setup_teardown(test_contract_scalar, setup, teardown),
    };
    return cmocka_run_group_tests(tests, group_setup, group_teardown);
}
