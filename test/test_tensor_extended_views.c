#include <stdlib.h>
/**
 * @file test_tensor_extended_views.c
 * Tensor3 view and reshape tests.
 */
#include <stdint.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

typedef struct {
    lmmc_tensor3_t a;
    lmmc_tensor3_t sliced;
    lmmc_tensor3_t reshaped;
    lmmc_tensor3_t tensor3_bad_slice;
    lmmc_tensor3_t tensor3_reshaped2;
    lmmc_tensor3_t tensor3_bad_reshape;
} test_fixture_t;

static int teardown(void **state);

static int setup(void **state) {
    test_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;

    lmmc_status_t status = lmmc_tensor3_create(2, 3, 4, &fixture->a);
    if (status != LMMC_STATUS_OK) {
        goto error;
    }
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                status = lmmc_tensor3_set(
                    &fixture->a, i, j, k,
                    (lmmc_real_t)(i * 12 + j * 4 + k + 1));
                if (status != LMMC_STATUS_OK) {
                    goto error;
                }
            }
        }
    }
    return 0;

error:
    teardown(state);
    return status;
}

static int teardown(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_destroy(&fixture->tensor3_bad_reshape);
    lmmc_tensor3_destroy(&fixture->tensor3_reshaped2);
    lmmc_tensor3_destroy(&fixture->tensor3_bad_slice);
    lmmc_tensor3_destroy(&fixture->reshaped);
    lmmc_tensor3_destroy(&fixture->sliced);
    lmmc_tensor3_destroy(&fixture->a);
    free(fixture);
    *state = NULL;
    return 0;
}

static void assert_owning_source_unchanged(
    const lmmc_tensor3_t* tensor, lmmc_real_t* data) {
    assert_true(tensor->data == data);
    assert_int_equal(tensor->owns_data, 1);
    assert_int_equal(tensor->dim0, 2);
    assert_int_equal(tensor->dim1, 3);
    assert_int_equal(tensor->dim2, 4);
    assert_int_equal(tensor->stride0, 12);
    assert_int_equal(tensor->stride1, 4);
    assert_int_equal(tensor->stride2, 1);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                lmmc_real_t value = 0.0;
                assert_int_equal(
                    lmmc_tensor3_get(tensor, i, j, k, &value),
                    LMMC_STATUS_OK);
                assert_true(lmmc_test_nearly_equal(
                    value, (lmmc_real_t)(i * 12 + j * 4 + k + 1), 1e-15));
            }
        }
    }
}

static void test_tensor_slice_values(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *a = &fixture->a;
    lmmc_tensor3_t *sliced = &fixture->sliced;
    lmmc_status_t st;

    st = lmmc_tensor3_slice_view(a, 0, 2, 1, 3, 0, 2, sliced);
    assert_false(st != LMMC_STATUS_OK);
    assert_false(sliced->dim0 != 2 || sliced->dim1 != 2 || sliced->dim2 != 2);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                double vs, va;
                lmmc_tensor3_get(sliced, i, j, k, &vs);
                lmmc_tensor3_get(a, i, j + 1, k, &va);
                assert_true(lmmc_test_nearly_equal(vs, va, 1e-12));
            }
        }
    }

    {

        st = lmmc_tensor3_slice_view(a, 0, 5, 0, 1, 0, 1, &fixture->tensor3_bad_slice);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }
}

static void test_tensor_slice_rejects_owning_self_alias(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t* tensor = &fixture->a;
    lmmc_real_t* const data = tensor->data;

    assert_int_equal(
        lmmc_tensor3_slice_view(
            tensor, 0, 2, 1, 3, 0, 4, tensor),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_owning_source_unchanged(tensor, data);
}

static void test_tensor_slice_allows_nonowning_self_alias(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t* const source = &fixture->a;
    lmmc_tensor3_t* const view = &fixture->sliced;
    lmmc_real_t* const source_data = source->data;
    lmmc_real_t value = 0.0;

    assert_int_equal(
        lmmc_tensor3_slice_view(
            source, 0, 2, 0, 3, 0, 4, view),
        LMMC_STATUS_OK);
    assert_true(view->data == source_data);
    assert_int_equal(view->owns_data, 0);

    assert_int_equal(
        lmmc_tensor3_slice_view(
            view, 1, 2, 1, 3, 1, 4, view),
        LMMC_STATUS_OK);
    assert_true(view->data == source_data + 17);
    assert_int_equal(view->owns_data, 0);
    assert_int_equal(view->dim0, 1);
    assert_int_equal(view->dim1, 2);
    assert_int_equal(view->dim2, 3);
    assert_int_equal(view->stride0, 12);
    assert_int_equal(view->stride1, 4);
    assert_int_equal(view->stride2, 1);
    assert_int_equal(
        lmmc_tensor3_get(view, 0, 0, 0, &value),
        LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(value, 18.0, 1e-15));

    assert_int_equal(
        lmmc_tensor3_set(view, 0, 0, 0, 91.0),
        LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_tensor3_get(source, 1, 1, 1, &value),
        LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(value, 91.0, 1e-15));
    assert_true(source->data == source_data);
    assert_int_equal(source->owns_data, 1);
    assert_int_equal(source->dim0, 2);
    assert_int_equal(source->dim1, 3);
    assert_int_equal(source->dim2, 4);
}

static void test_tensor_views(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *a = &fixture->a;
    lmmc_tensor3_t *reshaped = &fixture->reshaped;
    lmmc_status_t st;

    st = lmmc_tensor3_reshape_view(a, 4, 6, 1, reshaped);
    assert_false(st != LMMC_STATUS_OK);

    assert_false(reshaped->dim0 * reshaped->dim1 * reshaped->dim2 != a->dim0 * a->dim1 * a->dim2);

    assert_false(reshaped->owns_data != 0);
    assert_false(reshaped->data != a->data);

    {

        st = lmmc_tensor3_reshape_view(a, 1, 24, 1, &fixture->tensor3_reshaped2);
        assert_false(st != LMMC_STATUS_OK);
        assert_false(fixture->tensor3_reshaped2.dim0 * fixture->tensor3_reshaped2.dim1 * fixture->tensor3_reshaped2.dim2 != 24);
        lmmc_tensor3_destroy(&fixture->tensor3_reshaped2);
    }
}

static void test_tensor_reshape_errors(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *a = &fixture->a;
    lmmc_status_t st;

    st = lmmc_tensor3_reshape_view(a, 2, 3, 5, &fixture->tensor3_bad_reshape);
    assert_false(st != LMMC_STATUS_DIMENSION_MISMATCH);

    st = lmmc_tensor3_reshape_view(a, 5, 5, 1, &fixture->tensor3_bad_reshape);
    assert_false(st != LMMC_STATUS_DIMENSION_MISMATCH);

    st = lmmc_tensor3_reshape_view(a, 1, 1, 1, &fixture->tensor3_bad_reshape);
    assert_false(st != LMMC_STATUS_DIMENSION_MISMATCH);

    st = lmmc_tensor3_reshape_view(a, SIZE_MAX / 2 + 13, 2, 1, &fixture->tensor3_bad_reshape);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    st = lmmc_tensor3_reshape_view(a, SIZE_MAX, 2, 1, &fixture->tensor3_bad_reshape);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);

    st = lmmc_tensor3_reshape_view(a, 4, 6, 1, a);
    assert_false(st != LMMC_STATUS_INVALID_ARGUMENT || a->owns_data != 1);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_tensor_slice_values, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_slice_rejects_owning_self_alias, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_slice_allows_nonowning_self_alias, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_views, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_reshape_errors, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
