/**
 * @file test_memory_safety.c
 * 针对 LMMC 中 memory safety 相关接口的单元测试。
 */
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static void test_dense_allocation_lifecycle(void **state) {
    (void)state;
    {
        lmmc_vec_t v = {0};
        lmmc_status_t st = lmmc_vec_create(10, &v);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_vec_destroy(&v);
    }

    {
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_create(5, 5, &m);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_mat_destroy(&m);
    }

    {
        lmmc_tensor3_t t = {0};
        lmmc_status_t st = lmmc_tensor3_create(3, 4, 5, &t);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_tensor3_destroy(&t);
    }

    {
        lmmc_sparse_mat_t sp = {0};
        lmmc_status_t st = lmmc_sparse_create_csr(5, 5, 10, &sp);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_sparse_destroy(&sp);
    }
}

static void test_workspace_allocation_lifecycle(void **state) {
    (void)state;
    {
        lmmc_rng_t *rng = NULL;
        lmmc_status_t st = lmmc_rng_create(&rng);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_rng_destroy(rng);
    }

    {
        lmmc_precond_t pc = {0};
        lmmc_status_t st = lmmc_precond_create_none(10, &pc);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_precond_destroy(&pc);
    }
}

static void test_interpolator_allocation_lifecycle(void **state) {
    (void)state;
    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
        lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};
        lmmc_interp_cspline_t *spline = NULL;
        lmmc_status_t st = lmmc_interp_cspline_create(xs, ys, 5, &spline);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_interp_cspline_destroy(spline);
    }

    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
        lmmc_real_t ys[] = {1.0, 2.0, 5.0, 10.0};
        lmmc_interp_lagrange_t *lag = NULL;
        lmmc_status_t st = lmmc_interp_lagrange_create(xs, ys, 4, &lag);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_interp_lagrange_destroy(lag);
    }
}

static void test_zero_shape_rejection(void **state) {
    (void)state;
    {
        lmmc_vec_t v = {0};
        lmmc_status_t st = lmmc_vec_create(0, &v);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }

    {
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_create(0, 5, &m);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }

    {
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_create(5, 0, &m);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }

    {
        lmmc_tensor3_t t = {0};
        lmmc_status_t st = lmmc_tensor3_create(0, 4, 5, &t);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }
}

static void test_invalid_interpolator_storage(void **state) {
    (void)state;
    {
        lmmc_real_t xs[] = {0.0, 1.0};
        lmmc_real_t ys[] = {0.0, 1.0};
        lmmc_interp_cspline_t *spline = NULL;
        lmmc_status_t st = lmmc_interp_cspline_create(xs, ys, 2, &spline);

        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }

    {
        lmmc_interp_lagrange_t *lag = NULL;
        lmmc_status_t st = lmmc_interp_lagrange_create(NULL, NULL, 0, &lag);
        assert_false(st != LMMC_STATUS_INVALID_ARGUMENT);
    }
}

static void test_borrowed_vector_storage(void **state) {
    (void)state;
    lmmc_real_t ext_data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    lmmc_vec_t v = {0};
    lmmc_status_t st = lmmc_vec_wrap(5, ext_data, &v);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_vec_destroy(&v);

    assert_false(!lmmc_test_nearly_equal(ext_data[0], 1.0, 1e-15) ||
                 !lmmc_test_nearly_equal(ext_data[2], 3.0, 1e-15) ||
                 !lmmc_test_nearly_equal(ext_data[4], 5.0, 1e-15));
}

static void test_borrowed_matrix_storage(void **state) {
    (void)state;
    lmmc_real_t ext_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    lmmc_mat_t m = {0};
    lmmc_status_t st = lmmc_mat_wrap(2, 3, 3, ext_data, &m);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_mat_destroy(&m);

    assert_false(!lmmc_test_nearly_equal(ext_data[0], 1.0, 1e-15) ||
                 !lmmc_test_nearly_equal(ext_data[3], 4.0, 1e-15) ||
                 !lmmc_test_nearly_equal(ext_data[5], 6.0, 1e-15));
}

static void test_borrowed_tensor_storage(void **state) {
    (void)state;
    lmmc_real_t ext_data[24];
    for (int i = 0; i < 24; i++)
        ext_data[i] = (lmmc_real_t)(i + 1);
    lmmc_tensor3_t t = {0};

    lmmc_status_t st = lmmc_tensor3_wrap(2, 3, 4, 12, 4, 1, ext_data, &t);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_tensor3_destroy(&t);

    assert_false(!lmmc_test_nearly_equal(ext_data[0], 1.0, 1e-15) ||
                 !lmmc_test_nearly_equal(ext_data[12], 13.0, 1e-15) ||
                 !lmmc_test_nearly_equal(ext_data[23], 24.0, 1e-15));
}

static void test_borrowed_sparse_storage(void **state) {
    (void)state;
    size_t row_ptr[] = {0, 1, 2, 3};
    size_t col_idx[] = {0, 1, 2};
    lmmc_real_t values[] = {1.0, 1.0, 1.0};
    lmmc_sparse_mat_t sp = {0};
    lmmc_status_t st = lmmc_sparse_wrap_csr(3, 3, 3, row_ptr, col_idx, values, &sp);
    assert_int_equal(st, LMMC_STATUS_OK);
    lmmc_sparse_destroy(&sp);

    assert_false(!lmmc_test_nearly_equal(values[0], 1.0, 1e-15) ||
                 !lmmc_test_nearly_equal(values[1], 1.0, 1e-15) ||
                 !lmmc_test_nearly_equal(values[2], 1.0, 1e-15));
    assert_false(row_ptr[0] != 0 || row_ptr[3] != 3);
    assert_false(col_idx[0] != 0 || col_idx[2] != 2);
}

static void test_repeated_vector_lifecycle(void **state) {
    (void)state;
    for (int i = 0; i < 1000; i++) {
        lmmc_vec_t v = {0};
        lmmc_status_t st = lmmc_vec_create(100, &v);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_vec_destroy(&v);
    }
}

static void test_repeated_matrix_lifecycle(void **state) {
    (void)state;
    for (int i = 0; i < 1000; i++) {
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_create(10, 10, &m);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_mat_destroy(&m);
    }
}

static void test_repeated_tensor_lifecycle(void **state) {
    (void)state;
    for (int i = 0; i < 1000; i++) {
        lmmc_tensor3_t t = {0};
        lmmc_status_t st = lmmc_tensor3_create(3, 3, 3, &t);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_tensor3_destroy(&t);
    }
}

static void test_repeated_sparse_lifecycle(void **state) {
    (void)state;
    for (int i = 0; i < 1000; i++) {
        lmmc_sparse_mat_t sp = {0};
        lmmc_status_t st = lmmc_sparse_create_csr(5, 5, 5, &sp);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_sparse_destroy(&sp);
    }
}

static void test_repeated_rng_lifecycle(void **state) {
    (void)state;
    for (int i = 0; i < 1000; i++) {
        lmmc_rng_t *rng = NULL;
        lmmc_status_t st = lmmc_rng_create(&rng);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_rng_destroy(rng);
    }
}

static void test_repeated_preconditioner_lifecycle(void **state) {
    (void)state;
    for (int i = 0; i < 1000; i++) {
        lmmc_precond_t pc = {0};
        lmmc_status_t st = lmmc_precond_create_none(10, &pc);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_precond_destroy(&pc);
    }
}

static void test_repeated_spline_lifecycle(void **state) {
    (void)state;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};
    for (int i = 0; i < 1000; i++) {
        lmmc_interp_cspline_t *spline = NULL;
        lmmc_status_t st = lmmc_interp_cspline_create(xs, ys, 5, &spline);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_interp_cspline_destroy(spline);
    }
}

static void test_repeated_lagrange_lifecycle(void **state) {
    (void)state;
    lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
    lmmc_real_t ys[] = {1.0, 2.0, 5.0, 10.0};
    for (int i = 0; i < 1000; i++) {
        lmmc_interp_lagrange_t *lag = NULL;
        lmmc_status_t st = lmmc_interp_lagrange_create(xs, ys, 4, &lag);
        assert_int_equal(st, LMMC_STATUS_OK);
        lmmc_interp_lagrange_destroy(lag);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_dense_allocation_lifecycle),
        cmocka_unit_test(test_workspace_allocation_lifecycle),
        cmocka_unit_test(test_interpolator_allocation_lifecycle),
        cmocka_unit_test(test_zero_shape_rejection),
        cmocka_unit_test(test_invalid_interpolator_storage),
        cmocka_unit_test(test_borrowed_vector_storage),
        cmocka_unit_test(test_borrowed_matrix_storage),
        cmocka_unit_test(test_borrowed_tensor_storage),
        cmocka_unit_test(test_borrowed_sparse_storage),
        cmocka_unit_test(test_repeated_vector_lifecycle),
        cmocka_unit_test(test_repeated_matrix_lifecycle),
        cmocka_unit_test(test_repeated_tensor_lifecycle),
        cmocka_unit_test(test_repeated_sparse_lifecycle),
        cmocka_unit_test(test_repeated_rng_lifecycle),
        cmocka_unit_test(test_repeated_preconditioner_lifecycle),
        cmocka_unit_test(test_repeated_spline_lifecycle),
        cmocka_unit_test(test_repeated_lagrange_lifecycle),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
