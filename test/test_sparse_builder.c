/**
 * @file test_sparse_builder.c
 * 针对 LMMC 中 sparse builder 相关接口的单元测试。
 */
#include <stdio.h>
#include "test_common.h"
#include "lmmc/lmmc.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_sparse_builder_t *builder;
    lmmc_sparse_mat_t a;
    lmmc_sparse_mat_t converted;
    lmmc_sparse_mat_t roundtrip;
    lmmc_sparse_mat_t transposed;
    lmmc_sparse_mat_t twice;
    lmmc_mat_t dense;
    lmmc_vec_t x;
    lmmc_vec_t y;
    lmmc_vec_t diagonal;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    return fixture ? 0 : -1;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->diagonal);
    lmmc_vec_destroy(&fixture->y);
    lmmc_vec_destroy(&fixture->x);
    lmmc_mat_destroy(&fixture->dense);
    lmmc_sparse_destroy(&fixture->twice);
    lmmc_sparse_destroy(&fixture->transposed);
    lmmc_sparse_destroy(&fixture->roundtrip);
    lmmc_sparse_destroy(&fixture->converted);
    lmmc_sparse_destroy(&fixture->a);
    lmmc_sparse_builder_destroy(fixture->builder);
    free(fixture);
    *state = NULL;
    return 0;
}

static lmmc_status_t add_builder_entries(lmmc_sparse_builder_t *builder) {
    lmmc_status_t st;
    st = lmmc_sparse_builder_add(builder, 2, 2, 5.0);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_sparse_builder_add(builder, 0, 0, 1.0);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_sparse_builder_add(builder, 1, 1, 3.0);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_sparse_builder_add(builder, 0, 2, 2.0);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_sparse_builder_add(builder, 2, 0, 4.0);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return st;
}

static void test_duplicate_observers(void **state) {
    struct test_fixture *fixture = *state;

    assert_true(lmmc_sparse_builder_create(1, 1, 1, &fixture->builder) == LMMC_STATUS_OK);
    assert_true(lmmc_sparse_builder_add(fixture->builder, 0, 0, 2.0) == LMMC_STATUS_OK);
    assert_true(lmmc_sparse_builder_add(fixture->builder, 0, 0, 3.0) == LMMC_STATUS_OK);

    assert_true(lmmc_sparse_builder_build(fixture->builder, LMMC_SPARSE_CSR, &fixture->converted) == LMMC_STATUS_OK);
    for (int cancelled = 0; cancelled < 2; ++cancelled) {
        const double expected = cancelled ? 0.0 : 5.0;
        if (cancelled) {
            assert_true(lmmc_sparse_builder_add(fixture->builder, 0, 0, -5.0) == LMMC_STATUS_OK);
        }
        for (int format = 0; format < 2; ++format) {

            double norm = -1.0;
            assert_true(lmmc_sparse_builder_build(fixture->builder, (lmmc_sparse_format_t)format, &fixture->a) == LMMC_STATUS_OK);
            assert_true(fixture->a.nnz == 1 && fixture->a.values[0] == expected);
            assert_true(lmmc_mat_create(1, 1, &fixture->dense) == LMMC_STATUS_OK);
            assert_true(lmmc_vec_create(1, &fixture->x) == LMMC_STATUS_OK);
            assert_true(lmmc_vec_create(1, &fixture->y) == LMMC_STATUS_OK);
            assert_true(lmmc_vec_create(1, &fixture->diagonal) == LMMC_STATUS_OK);
            fixture->x.data[0] = 1.0;
            assert_true(lmmc_sparse_to_dense(&fixture->a, &fixture->dense) == LMMC_STATUS_OK && fixture->dense.data[0] == expected);
            assert_true(lmmc_sparse_diag(&fixture->a, &fixture->diagonal) == LMMC_STATUS_OK && fixture->diagonal.data[0] == expected);
            assert_true(lmmc_sparse_norm_fro(&fixture->a, &norm) == LMMC_STATUS_OK && norm == expected);
            assert_true(lmmc_sparse_mat_vec_mul(&fixture->a, &fixture->x, &fixture->y) == LMMC_STATUS_OK && fixture->y.data[0] == expected);
            lmmc_vec_destroy(&fixture->diagonal);
            lmmc_vec_destroy(&fixture->y);
            lmmc_vec_destroy(&fixture->x);
            lmmc_mat_destroy(&fixture->dense);
            lmmc_sparse_destroy(&fixture->a);
        }
    }
    assert_true(fixture->converted.nnz == 1 && fixture->converted.values[0] == 5.0);
}

static void test_rectangular_rebuild(void **state) {
    struct test_fixture *fixture = *state;

    const size_t rows[] = {2, 0, 2, 0, 2};
    const size_t cols[] = {3, 2, 1, 2, 3};
    const double values[] = {2, 4, 3, -4, 5};
    const double expected[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 7};
    assert_true(lmmc_sparse_builder_create(3, 4, 1, &fixture->builder) == LMMC_STATUS_OK);
    for (int format = 0; format < 2; ++format) {
        assert_true(lmmc_sparse_builder_build(fixture->builder, (lmmc_sparse_format_t)format, &fixture->a) == LMMC_STATUS_OK);
        assert_true(fixture->a.nnz == 0 && fixture->a.row_ptr[format ? 4 : 3] == 0);
        lmmc_sparse_destroy(&fixture->a);
    }
    assert_true(lmmc_sparse_builder_build(fixture->builder, (lmmc_sparse_format_t)9, &fixture->a) == LMMC_STATUS_INVALID_ARGUMENT);
    for (size_t i = 0; i < 5; ++i) {
        assert_true(lmmc_sparse_builder_add(fixture->builder, rows[i], cols[i], values[i]) == LMMC_STATUS_OK);
    }
    assert_true(lmmc_sparse_builder_build(fixture->builder, LMMC_SPARSE_CSR, &fixture->a) == LMMC_STATUS_OK);
    assert_true(lmmc_sparse_builder_build(fixture->builder, LMMC_SPARSE_CSC, &fixture->converted) == LMMC_STATUS_OK);
    assert_true(lmmc_sparse_to_csr(&fixture->converted, &fixture->roundtrip) == LMMC_STATUS_OK);
    assert_true(lmmc_sparse_transpose(&fixture->a, &fixture->transposed) == LMMC_STATUS_OK);
    assert_true(lmmc_sparse_transpose(&fixture->transposed, &fixture->twice) == LMMC_STATUS_OK);
    assert_true(lmmc_mat_create(3, 4, &fixture->dense) == LMMC_STATUS_OK);
    const lmmc_sparse_mat_t *matrices[] = {&fixture->a, &fixture->converted, &fixture->roundtrip, &fixture->twice};
    for (size_t k = 0; k < 4; ++k) {
        assert_true(matrices[k]->nnz == 3);
        assert_true(lmmc_sparse_to_dense(matrices[k], &fixture->dense) == LMMC_STATUS_OK);
        for (size_t r = 0; r < 3; ++r) {
            for (size_t c = 0; c < 4; ++c) {
                assert_true(fixture->dense.data[r * fixture->dense.stride + c] == expected[r * 4 + c]);
            }
        }
    }
}

static void test_builder_entries(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_builder_create(3, 3, 4, &fixture->builder);
    assert_true(st == LMMC_STATUS_OK);

    st = add_builder_entries(fixture->builder);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_builder_build(fixture->builder, LMMC_SPARSE_CSR, &fixture->a);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(fixture->a.nnz == 5);

    st = lmmc_mat_create(3, 3, &fixture->dense);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_to_dense(&fixture->a, &fixture->dense);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(fixture->dense.data[0 * fixture->dense.stride + 0] == 1.0);
    assert_true(fixture->dense.data[0 * fixture->dense.stride + 1] == 0.0);
    assert_true(fixture->dense.data[0 * fixture->dense.stride + 2] == 2.0);
    assert_true(fixture->dense.data[1 * fixture->dense.stride + 0] == 0.0);
    assert_true(fixture->dense.data[1 * fixture->dense.stride + 1] == 3.0);
    assert_true(fixture->dense.data[1 * fixture->dense.stride + 2] == 0.0);
    assert_true(fixture->dense.data[2 * fixture->dense.stride + 0] == 4.0);
    assert_true(fixture->dense.data[2 * fixture->dense.stride + 1] == 0.0);
    assert_true(fixture->dense.data[2 * fixture->dense.stride + 2] == 5.0);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_duplicate_observers, setup, teardown),
        cmocka_unit_test_setup_teardown(test_rectangular_rebuild, setup, teardown),
        cmocka_unit_test_setup_teardown(test_builder_entries, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
