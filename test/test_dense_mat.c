/**
 * @file test_dense_mat.c
 * 针对 LMMC 中 dense mat 相关接口的单元测试。
 */
#include <inttypes.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"

#define NUM_ITERATIONS 100

#define TOLERANCE 1e-12

struct test_fixture {
    lmmc_mat_t A;
    lmmc_mat_t B;
    lmmc_mat_t C;
    lmmc_mat_t D;
};

static double rand_double(double range) {
    return ((double)rand() / (double)RAND_MAX) * 2.0 * range - range;
}

static size_t rand_size(size_t min_size, size_t max_size) {
    return min_size + (size_t)(rand() % (int)(max_size - min_size + 1));
}

static void fill_random_matrix(lmmc_mat_t *mat, double range) {
    for (size_t i = 0; i < mat->rows; ++i) {
        for (size_t j = 0; j < mat->cols; ++j) {
            lmmc_real_t val = rand_double(range);
            LMMC_REAL_SET(&mat->data[i * mat->stride + j], &val);
        }
    }
}

static void test_mat_add_sub_inverse(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t rows = rand_size(2, 8);
        size_t cols = rand_size(2, 8);

        lmmc_status_t status;

        status = lmmc_mat_create(rows, cols, &fixture->A);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: failed to create A (%" PRIuMAX "x%" PRIuMAX ")", iter, (uintmax_t)(rows), (uintmax_t)(cols));
        }

        status = lmmc_mat_create(rows, cols, &fixture->B);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: failed to create B (%" PRIuMAX "x%" PRIuMAX ")", iter, (uintmax_t)(rows), (uintmax_t)(cols));
        }

        status = lmmc_mat_create(rows, cols, &fixture->C);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: failed to create C (%" PRIuMAX "x%" PRIuMAX ")", iter, (uintmax_t)(rows), (uintmax_t)(cols));
        }

        status = lmmc_mat_create(rows, cols, &fixture->D);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: failed to create D (%" PRIuMAX "x%" PRIuMAX ")", iter, (uintmax_t)(rows), (uintmax_t)(cols));
        }

        fill_random_matrix(&fixture->A, 100.0);
        fill_random_matrix(&fixture->B, 100.0);

        status = lmmc_mat_add(&fixture->A, &fixture->B, &fixture->C);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: mat_add failed", iter);
        }

        status = lmmc_mat_sub(&fixture->C, &fixture->B, &fixture->D);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: mat_sub failed", iter);
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                lmmc_real_t d_val = fixture->D.data[i * fixture->D.stride + j];
                lmmc_real_t a_val = fixture->A.data[i * fixture->A.stride + j];
                double diff = fabs(d_val - a_val);
                if (!(diff <= TOLERANCE)) {
                    fail_msg("iter %d: D[%" PRIuMAX "][%" PRIuMAX "] = %.15g != A[%" PRIuMAX "][%" PRIuMAX "] = %.15g (diff = %.2e)", iter, (uintmax_t)(i), (uintmax_t)(j), d_val, (uintmax_t)(i), (uintmax_t)(j), a_val, diff);
                }
            }
        }

        lmmc_mat_destroy(&fixture->A);
        lmmc_mat_destroy(&fixture->B);
        lmmc_mat_destroy(&fixture->C);
        lmmc_mat_destroy(&fixture->D);
    }
}

static void test_mat_sub_add_inverse(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t rows = rand_size(2, 8);
        size_t cols = rand_size(2, 8);

        lmmc_status_t status;

        status = lmmc_mat_create(rows, cols, &fixture->A);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: failed to create A", iter);
        }

        status = lmmc_mat_create(rows, cols, &fixture->B);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: failed to create B", iter);
        }

        status = lmmc_mat_create(rows, cols, &fixture->C);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: failed to create C", iter);
        }

        status = lmmc_mat_create(rows, cols, &fixture->D);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: failed to create D", iter);
        }

        fill_random_matrix(&fixture->A, 100.0);
        fill_random_matrix(&fixture->B, 100.0);

        status = lmmc_mat_sub(&fixture->A, &fixture->B, &fixture->C);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: mat_sub failed", iter);
        }

        status = lmmc_mat_add(&fixture->C, &fixture->B, &fixture->D);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: mat_add failed", iter);
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                lmmc_real_t d_val = fixture->D.data[i * fixture->D.stride + j];
                lmmc_real_t a_val = fixture->A.data[i * fixture->A.stride + j];
                double diff = fabs(d_val - a_val);
                if (!(diff <= TOLERANCE)) {
                    fail_msg("iter %d: D[%" PRIuMAX "][%" PRIuMAX "] = %.15g != A[%" PRIuMAX "][%" PRIuMAX "] = %.15g (diff = %.2e)", iter, (uintmax_t)(i), (uintmax_t)(j), d_val, (uintmax_t)(i), (uintmax_t)(j), a_val, diff);
                }
            }
        }

        lmmc_mat_destroy(&fixture->A);
        lmmc_mat_destroy(&fixture->B);
        lmmc_mat_destroy(&fixture->C);
        lmmc_mat_destroy(&fixture->D);
    }
}

static void test_identity_structure(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_size(2, 8);

        lmmc_status_t status;

        status = lmmc_mat_identity(n, &fixture->A);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: mat_identity(%" PRIuMAX ") failed", iter, (uintmax_t)(n));
        }

        if (!(fixture->A.rows == n)) {
            fail_msg("iter %d: identity rows = %" PRIuMAX ", expected %" PRIuMAX, iter, (uintmax_t)(fixture->A.rows), (uintmax_t)(n));
        }
        if (!(fixture->A.cols == n)) {
            fail_msg("iter %d: identity cols = %" PRIuMAX ", expected %" PRIuMAX, iter, (uintmax_t)(fixture->A.cols), (uintmax_t)(n));
        }

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                lmmc_real_t val = fixture->A.data[i * fixture->A.stride + j];
                if (i == j) {
                    if (!(fabs(val - 1.0) <= TOLERANCE)) {
                        fail_msg("iter %d: I[%" PRIuMAX "][%" PRIuMAX "] = %.15g, expected 1.0", iter, (uintmax_t)(i), (uintmax_t)(j), val);
                    }
                } else {
                    if (!(fabs(val) <= TOLERANCE)) {
                        fail_msg("iter %d: I[%" PRIuMAX "][%" PRIuMAX "] = %.15g, expected 0.0", iter, (uintmax_t)(i), (uintmax_t)(j), val);
                    }
                }
            }
        }

        lmmc_mat_destroy(&fixture->A);
    }
}

static void test_identity_trace(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_size(2, 8);

        lmmc_real_t trace;
        lmmc_status_t status;

        status = lmmc_mat_identity(n, &fixture->A);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: mat_identity(%" PRIuMAX ") failed", iter, (uintmax_t)(n));
        }

        status = lmmc_mat_trace(&fixture->A, &trace);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: mat_trace failed", iter);
        }

        if (!(fabs(trace - (double)n) <= TOLERANCE)) {
            fail_msg("iter %d: trace(I_%" PRIuMAX ") = %.15g, expected %" PRIuMAX, iter, (uintmax_t)(n), trace, (uintmax_t)(n));
        }

        lmmc_mat_destroy(&fixture->A);
    }
}

static void test_identity_det(void **state) {
    struct test_fixture *fixture = *state;
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_size(2, 8);

        lmmc_real_t det;
        lmmc_status_t status;

        status = lmmc_mat_identity(n, &fixture->A);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: mat_identity(%" PRIuMAX ") failed", iter, (uintmax_t)(n));
        }

        status = lmmc_mat_det(&fixture->A, &det);
        if (!(status == LMMC_STATUS_OK)) {
            fail_msg("iter %d: mat_det failed", iter);
        }

        if (!(fabs(det - 1.0) <= TOLERANCE)) {
            fail_msg("iter %d: det(I_%" PRIuMAX ") = %.15g, expected 1.0", iter, (uintmax_t)(n), det);
        }

        lmmc_mat_destroy(&fixture->A);
    }
}

static void test_invalid_descriptors_are_rejected(void **state) {
    (void)state;
    lmmc_real_t storage[4] = {1.0, 2.0, 3.0, 4.0};
    lmmc_mat_t valid = {2, 2, 2, storage, 0};
    lmmc_mat_t invalid_stride = valid;
    lmmc_mat_t overflowing = valid;
    lmmc_mat_t wrapped;
    lmmc_real_t scalar = 0.0;

    invalid_stride.stride = 1;
    if (!(lmmc_mat_fill(&invalid_stride, 0.0) ==
          LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("mat_fill accepted stride smaller than column count");
    }
    if (!(lmmc_mat_add(&invalid_stride, &valid, &valid) ==
          LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("mat_add accepted an invalid matrix descriptor");
    }
    if (!(lmmc_mat_gemm(1.0, &valid, 0, &valid, 0, 0.0,
                        &invalid_stride) ==
          LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("mat_gemm accepted an invalid output descriptor");
    }
    if (!(lmmc_mat_det(&invalid_stride, &scalar) ==
          LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("mat_det accepted an invalid matrix descriptor");
    }

    overflowing.stride = SIZE_MAX;
    if (!(lmmc_mat_fill(&overflowing, 0.0) ==
          LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("mat_fill accepted overflowing address arithmetic");
    }
    if (!(lmmc_mat_wrap(2, 2, SIZE_MAX, storage, &wrapped) ==
          LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("mat_wrap accepted overflowing address arithmetic");
    }
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    assert_non_null(fixture);
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->A);
    lmmc_mat_destroy(&fixture->B);
    lmmc_mat_destroy(&fixture->C);
    lmmc_mat_destroy(&fixture->D);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const unsigned int seed = 0x444D4154u;
    srand(seed);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_mat_add_sub_inverse, setup, teardown),
        cmocka_unit_test_setup_teardown(test_mat_sub_add_inverse, setup, teardown),
        cmocka_unit_test_setup_teardown(test_identity_structure, setup, teardown),
        cmocka_unit_test_setup_teardown(test_identity_trace, setup, teardown),
        cmocka_unit_test_setup_teardown(test_identity_det, setup, teardown),
        cmocka_unit_test(test_invalid_descriptors_are_rejected),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
