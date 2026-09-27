#include <stdlib.h>
/**
 * @file test_tensor_nd.c
 * Unit tests for the N-D tensor API (lmmc_tensor_nd_t).
 */
#include <stdint.h>
#include "lmmc/lmmc.h"
#include "test_common.h"
#include "internal_test_hooks.h"

typedef struct {
    lmmc_tensor_nd_t tensor_nd_t;
    lmmc_tensor_nd_t tensor_nd_view;
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
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_view);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
    free(fixture);
    return 0;
}


static void test_create_basic(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {2, 3, 4};

    assert_int_equal(lmmc_tensor_nd_create(3, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);
    assert_true(fixture->tensor_nd_t.ndim == 3);
    assert_true(fixture->tensor_nd_t.dims[0] == 2);
    assert_true(fixture->tensor_nd_t.dims[1] == 3);
    assert_true(fixture->tensor_nd_t.dims[2] == 4);
    assert_true(fixture->tensor_nd_t.strides[0] == 12);
    assert_true(fixture->tensor_nd_t.strides[1] == 4);
    assert_true(fixture->tensor_nd_t.strides[2] == 1);
    assert_true(fixture->tensor_nd_t.owns_data == 1);
    assert_true(fixture->tensor_nd_t.data != NULL);

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
}

static void test_create_1d(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {10};

    assert_int_equal(lmmc_tensor_nd_create(1, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);
    assert_true(fixture->tensor_nd_t.ndim == 1);
    assert_true(fixture->tensor_nd_t.dims[0] == 10);
    assert_true(fixture->tensor_nd_t.strides[0] == 1);
    assert_true(fixture->tensor_nd_t.owns_data == 1);

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
}

static void test_create_max_ndim(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {2, 2, 2, 2, 2, 2, 2, 2};

    assert_int_equal(lmmc_tensor_nd_create(LMMC_TENSOR_MAX_NDIM, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);
    assert_true(fixture->tensor_nd_t.ndim == 8);
    assert_true(fixture->tensor_nd_t.strides[0] == 128);
    assert_true(fixture->tensor_nd_t.strides[7] == 1);

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
}

static void test_create_errors(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {2, 3, 4};
    size_t dims_zero[] = {2, 0, 4};

    /* ndim == 0 */
    assert_int_equal(lmmc_tensor_nd_create(0, dims, &fixture->tensor_nd_t), LMMC_STATUS_INVALID_ARGUMENT);
    /* ndim > MAX */
    assert_int_equal(lmmc_tensor_nd_create(9, dims, &fixture->tensor_nd_t), LMMC_STATUS_INVALID_ARGUMENT);
    /* NULL dims */
    assert_int_equal(lmmc_tensor_nd_create(3, NULL, &fixture->tensor_nd_t), LMMC_STATUS_INVALID_ARGUMENT);
    /* NULL out */
    assert_int_equal(lmmc_tensor_nd_create(3, dims, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    /* Zero dimension */
    assert_int_equal(lmmc_tensor_nd_create(3, dims_zero, &fixture->tensor_nd_t), LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_create_overflow_precedes_allocation_failure(void **state) {
    (void)state;
    size_t overflow_dims[] = {SIZE_MAX / 2 + 1, 2};
    size_t valid_dims[] = {2, 2};
    lmmc_real_t sentinel[] = {11.0, 13.0};
    lmmc_tensor_nd_t out = {
        .ndim = 1,
        .dims = {2},
        .strides = {1},
        .data = sentinel,
        .owns_data = 0,
    };

    lmmc_memory_fail_after_for_test(0);
    assert_int_equal(
        lmmc_tensor_nd_create(2, overflow_dims, &out),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(out.data == sentinel);
    assert_int_equal(out.owns_data, 0);
    assert_int_equal(out.ndim, 1);
    assert_int_equal(out.dims[0], 2);
    assert_true(lmmc_test_nearly_equal(sentinel[0], 11.0, 1e-15));
    assert_true(lmmc_test_nearly_equal(sentinel[1], 13.0, 1e-15));

    assert_int_equal(
        lmmc_tensor_nd_create(2, valid_dims, &out),
        LMMC_STATUS_ALLOCATION_FAILED);
    assert_true(out.data == sentinel);
    assert_int_equal(out.owns_data, 0);
    assert_int_equal(out.ndim, 1);
    assert_int_equal(out.dims[0], 2);
    assert_true(lmmc_test_nearly_equal(sentinel[0], 11.0, 1e-15));
    assert_true(lmmc_test_nearly_equal(sentinel[1], 13.0, 1e-15));
}

static void test_create_aliased_dims(void **state) {
    test_fixture_t *fixture = *state;

    size_t idx[] = {1, 2};
    size_t out_of_bounds[] = {1, 3};
    lmmc_real_t val = 0.0;

    fixture->tensor_nd_t.dims[0] = 2;
    fixture->tensor_nd_t.dims[1] = 3;
    assert_int_equal(lmmc_tensor_nd_create(2, fixture->tensor_nd_t.dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);
    assert_true(fixture->tensor_nd_t.ndim == 2 && fixture->tensor_nd_t.dims[0] == 2 && fixture->tensor_nd_t.dims[1] == 3);
    assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_t, idx, 42.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_t, idx, &val), LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 42.0, 1e-15));
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_t, out_of_bounds, &val), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
}

static void test_get_set(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {2, 3, 4};
    lmmc_real_t val = 0.0;

    assert_int_equal(lmmc_tensor_nd_create(3, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);

    /* Set and get */
    size_t idx[] = {1, 2, 3};
    assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_t, idx, 42.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_t, idx, &val), LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 42.0, 1e-15));

    /* Check zero-initialized */
    size_t idx2[] = {0, 0, 0};
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_t, idx2, &val), LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.0, 1e-15));

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
}

static void test_get_set_errors(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {2, 3, 4};
    lmmc_real_t val = 0.0;

    assert_int_equal(lmmc_tensor_nd_create(3, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);

    /* Out of bounds */
    size_t idx_oob[] = {2, 0, 0};
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_t, idx_oob, &val), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_t, idx_oob, 1.0), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL tensor */
    size_t idx[] = {0, 0, 0};
    assert_int_equal(lmmc_tensor_nd_get(NULL, idx, &val), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_tensor_nd_set(NULL, idx, 1.0), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL idx */
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_t, NULL, &val), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_t, NULL, 1.0), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL out */
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_t, idx, NULL), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
}




static void test_reshape_view(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {2, 3, 4};
    lmmc_real_t val = 0.0;

    assert_int_equal(lmmc_tensor_nd_create(3, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);

    /* Fill with sequential values */
    for (size_t i = 0; i < 24; ++i) {
        fixture->tensor_nd_t.data[i] = (lmmc_real_t)i;
    }

    /* Reshape to 4x6 */
    size_t new_dims[] = {4, 6};
    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_t, 2, new_dims, &fixture->tensor_nd_view), LMMC_STATUS_OK);

    assert_true(fixture->tensor_nd_view.ndim == 2);
    assert_true(fixture->tensor_nd_view.dims[0] == 4);
    assert_true(fixture->tensor_nd_view.dims[1] == 6);
    assert_true(fixture->tensor_nd_view.owns_data == 0);
    assert_true(fixture->tensor_nd_view.data == fixture->tensor_nd_t.data);

    /* Check element access */
    size_t idx[] = {0, 0};
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_view, idx, &val), LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 0.0, 1e-15));

    size_t idx2[] = {3, 5};
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_view, idx2, &val), LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 23.0, 1e-15));

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
    /* view doesn't own data, destroy is safe */
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_view);
}

static void test_reshape_view_errors(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {2, 3, 4};

    assert_int_equal(lmmc_tensor_nd_create(3, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);

    /* Wrong total size */
    size_t bad_dims[] = {5, 5};
    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_t, 2, bad_dims, &fixture->tensor_nd_view), LMMC_STATUS_INVALID_ARGUMENT);

    /* NULL args */
    size_t good_dims[] = {4, 6};
    assert_int_equal(lmmc_tensor_nd_reshape_view(NULL, 2, good_dims, &fixture->tensor_nd_view), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_t, 2, NULL, &fixture->tensor_nd_view), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_t, 2, good_dims, NULL), LMMC_STATUS_INVALID_ARGUMENT);

    /* ndim == 0 */
    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_t, 0, good_dims, &fixture->tensor_nd_view), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
}

static void test_reshape_view_overflow(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {4};
    size_t overflow_dims[] = {SIZE_MAX / 2 + 3, 2};
    size_t zero_dims[] = {4, 0};
    size_t good_dims[] = {2, 2};
    size_t source_idx[] = {3};
    size_t view_idx[] = {1, 1};
    size_t out_of_bounds[] = {3, 0};
    lmmc_real_t val = 0.0;

    assert_int_equal(lmmc_tensor_nd_create(1, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_t, source_idx, 17.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_t, 2, overflow_dims, &fixture->tensor_nd_view), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_t, 2, zero_dims, &fixture->tensor_nd_view), LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_t, 2, good_dims, &fixture->tensor_nd_view), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_view, view_idx, &val), LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 17.0, 1e-15));
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_view, out_of_bounds, &val), LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_view);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
}

static void test_reshape_view_aliases(void **state) {
    test_fixture_t *fixture = *state;

    size_t dims[] = {4};
    size_t source_idx[] = {3};
    size_t view_idx[] = {1, 1};
    lmmc_real_t val = 0.0;

    assert_int_equal(lmmc_tensor_nd_create(1, dims, &fixture->tensor_nd_t), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_t, source_idx, 23.0), LMMC_STATUS_OK);
    fixture->tensor_nd_view.dims[0] = 2;
    fixture->tensor_nd_view.dims[1] = 2;
    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_t, 2, fixture->tensor_nd_view.dims, &fixture->tensor_nd_view), LMMC_STATUS_OK);
    assert_true(fixture->tensor_nd_view.ndim == 2 && fixture->tensor_nd_view.dims[0] == 2 && fixture->tensor_nd_view.dims[1] == 2);
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_view, view_idx, &val), LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 23.0, 1e-15));
    assert_int_equal(lmmc_tensor_nd_set(&fixture->tensor_nd_view, view_idx, 31.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_t, source_idx, &val), LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 31.0, 1e-15));

    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_view, 2, fixture->tensor_nd_view.dims, &fixture->tensor_nd_view), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_view, 1, dims, &fixture->tensor_nd_view), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_view, source_idx, &val), LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 31.0, 1e-15));

    assert_int_equal(lmmc_tensor_nd_reshape_view(&fixture->tensor_nd_t, 1, fixture->tensor_nd_t.dims, &fixture->tensor_nd_t), LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(fixture->tensor_nd_t.owns_data == 1);
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_view);
    assert_int_equal(lmmc_tensor_nd_get(&fixture->tensor_nd_t, source_idx, &val), LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(val, 31.0, 1e-15));
    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t);
}

static void test_destroy_null_safe(void **state) {
    test_fixture_t *fixture = *state;
    /* Should not crash */
    lmmc_tensor_nd_destroy(NULL);

    lmmc_tensor_nd_destroy(&fixture->tensor_nd_t); /* zero-initialized, no data */
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
        cmocka_unit_test_setup_teardown(test_create_basic, setup, teardown),
        cmocka_unit_test_setup_teardown(test_create_1d, setup, teardown),
        cmocka_unit_test_setup_teardown(test_create_max_ndim, setup, teardown),
        cmocka_unit_test_setup_teardown(test_create_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_create_overflow_precedes_allocation_failure, setup, teardown),
        cmocka_unit_test_setup_teardown(test_create_aliased_dims, setup, teardown),
        cmocka_unit_test_setup_teardown(test_get_set, setup, teardown),
        cmocka_unit_test_setup_teardown(test_get_set_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_reshape_view, setup, teardown),
        cmocka_unit_test_setup_teardown(test_reshape_view_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_reshape_view_overflow, setup, teardown),
        cmocka_unit_test_setup_teardown(test_reshape_view_aliases, setup, teardown),
        cmocka_unit_test_setup_teardown(test_destroy_null_safe, setup, teardown),
    };
    return cmocka_run_group_tests(tests, group_setup, group_teardown);
}
