/**
 * @file test_eigen_extended_pseudoinverse.c
 * @brief 满秩与秩亏矩阵的 Moore-Penrose 恒等式测试。
 */
#include <inttypes.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include "lmmc/dense.h"
#include "lmmc/eigen.h"

#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])

static void product(const lmmc_mat_t *left, const lmmc_mat_t *right,
                    lmmc_mat_t *out) {
    for (size_t i = 0; i < out->rows; ++i) {
        for (size_t j = 0; j < out->cols; ++j) {
            lmmc_real_t sum = 0.0;
            for (size_t k = 0; k < left->cols; ++k) {
                sum += MAT_ELEM(left, i, k) * MAT_ELEM(right, k, j);
            }
            MAT_ELEM(out, i, j) = sum;
        }
    }
}

static void check_equal(const lmmc_mat_t *actual, const lmmc_mat_t *expected) {
    for (size_t i = 0; i < actual->rows; ++i) {
        for (size_t j = 0; j < actual->cols; ++j) {
            lmmc_real_t value = MAT_ELEM(actual, i, j);
            lmmc_real_t target = MAT_ELEM(expected, i, j);
            if (!(isfinite(value) && isfinite(target) && fabs(value - target) <= 1e-8)) {
                fail_msg("identity mismatch at (%" PRIuMAX ",%" PRIuMAX "): %.17g vs %.17g", (uintmax_t)(i), (uintmax_t)(j), value, target);
            }
        }
    }
}

static void check_symmetric(const lmmc_mat_t *matrix) {
    for (size_t i = 0; i < matrix->rows; ++i) {
        for (size_t j = 0; j < matrix->cols; ++j) {
            lmmc_real_t value = MAT_ELEM(matrix, i, j);
            lmmc_real_t transpose = MAT_ELEM(matrix, j, i);
            if (!(isfinite(value) && isfinite(transpose) && fabs(value - transpose) <= 1e-8)) {
                fail_msg("projector is not finite symmetric at (%" PRIuMAX ",%" PRIuMAX ")", (uintmax_t)(i), (uintmax_t)(j));
            }
        }
    }
}

static void check_moore_penrose(lmmc_real_t *data, size_t m, size_t n,
                                lmmc_real_t tolerance) {
    lmmc_real_t inverse[16], left[16], right[16], recovered[16];
    lmmc_mat_t a, pinv, ap, pa, result;
    if (!(lmmc_mat_wrap(m, n, n, data, &a) == LMMC_STATUS_OK)) {
        fail_msg("wrap input");
    }
    if (!(lmmc_mat_wrap(n, m, m, inverse, &pinv) == LMMC_STATUS_OK)) {
        fail_msg("wrap inverse");
    }
    if (!(lmmc_mat_wrap(m, m, m, left, &ap) == LMMC_STATUS_OK)) {
        fail_msg("wrap A*P");
    }
    if (!(lmmc_mat_wrap(n, n, n, right, &pa) == LMMC_STATUS_OK)) {
        fail_msg("wrap P*A");
    }
    if (!(lmmc_pinv(&a, tolerance, &pinv) == LMMC_STATUS_OK)) {
        fail_msg("pseudoinverse failed");
    }
    product(&a, &pinv, &ap);
    product(&pinv, &a, &pa);
    check_symmetric(&ap);
    check_symmetric(&pa);
    if (!(lmmc_mat_wrap(m, n, n, recovered, &result) == LMMC_STATUS_OK)) {
        fail_msg("wrap A*P*A");
    }
    product(&ap, &a, &result);
    check_equal(&result, &a);
    if (!(lmmc_mat_wrap(n, m, m, recovered, &result) == LMMC_STATUS_OK)) {
        fail_msg("wrap P*A*P");
    }
    product(&pa, &pinv, &result);
    check_equal(&result, &pinv);
}

static void test_pinv_moore_penrose(void **state) {
    (void)state;
    lmmc_real_t data[] = {1, 2, 3, 0, 1, 4, 5, 6, 0};
    check_moore_penrose(data, 3, 3, 0.0);
}

static void test_pinv_rank_deficient(void **state) {
    (void)state;
    lmmc_real_t data[] = {1, 2, 3, 4, 5, 6, 5, 7, 9};
    check_moore_penrose(data, 3, 3, 1e-10);
}

static void test_pinv_tall(void **state) {
    (void)state;
    lmmc_real_t data[] = {1, 2, 3, 4, 5, 6};
    check_moore_penrose(data, 3, 2, 0.0);
}

static void test_pinv_wide(void **state) {
    (void)state;
    lmmc_real_t data[] = {1, 2, 3, 4, 5, 6};
    check_moore_penrose(data, 2, 3, 0.0);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_pinv_moore_penrose),
        cmocka_unit_test(test_pinv_rank_deficient),
        cmocka_unit_test(test_pinv_tall),
        cmocka_unit_test(test_pinv_wide),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
