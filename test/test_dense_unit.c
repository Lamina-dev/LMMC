/**
 * @file test_dense_unit.c
 * @brief 针对 LMMC 中 dense unit 相关接口的单元测试。
 *
 * @internal
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/status.h"

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)

#define TOL 1e-12


static int test_vec_norm2_null(void)
{
    lmmc_real_t result;
    lmmc_status_t s = lmmc_vec_norm2(NULL, &result);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "vec_norm2(NULL, ...) should return INVALID_ARGUMENT, got %d", (int)s);
    return 0;
}


static int test_vec_axpy_dimension_mismatch(void)
{
    lmmc_vec_t x, y;
    lmmc_vec_create(3, &x);
    lmmc_vec_create(5, &y);

    lmmc_real_t alpha = 1.0;
    lmmc_status_t s = lmmc_vec_axpy(alpha, &x, &y);
    CHECK(s == LMMC_STATUS_DIMENSION_MISMATCH,
          "vec_axpy with mismatched sizes should return DIMENSION_MISMATCH, got %d", (int)s);

    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&y);
    return 0;
}


static int test_vec_copy_dimension_mismatch(void)
{
    lmmc_vec_t src, dst;
    lmmc_vec_create(4, &src);
    lmmc_vec_create(2, &dst);

    lmmc_status_t s = lmmc_vec_copy(&src, &dst);
    CHECK(s == LMMC_STATUS_DIMENSION_MISMATCH,
          "vec_copy with mismatched sizes should return DIMENSION_MISMATCH, got %d", (int)s);

    lmmc_vec_destroy(&src);
    lmmc_vec_destroy(&dst);
    return 0;
}


static int test_vec_swap_dimension_mismatch(void)
{
    lmmc_vec_t x, y;
    lmmc_vec_create(6, &x);
    lmmc_vec_create(3, &y);

    lmmc_status_t s = lmmc_vec_swap(&x, &y);
    CHECK(s == LMMC_STATUS_DIMENSION_MISMATCH,
          "vec_swap with mismatched sizes should return DIMENSION_MISMATCH, got %d", (int)s);

    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&y);
    return 0;
}


static int test_mat_add_dimension_mismatch(void)
{
    lmmc_mat_t a, b, c;
    lmmc_mat_create(2, 3, &a);
    lmmc_mat_create(3, 2, &b);
    lmmc_mat_create(2, 3, &c);

    lmmc_status_t s = lmmc_mat_add(&a, &b, &c);
    CHECK(s == LMMC_STATUS_DIMENSION_MISMATCH,
          "mat_add with mismatched dims should return DIMENSION_MISMATCH, got %d", (int)s);

    lmmc_mat_destroy(&a);
    lmmc_mat_destroy(&b);
    lmmc_mat_destroy(&c);
    return 0;
}


static int test_mat_trace_non_square(void)
{
    lmmc_mat_t a;
    lmmc_mat_create(2, 3, &a);

    lmmc_real_t trace;
    lmmc_status_t s = lmmc_mat_trace(&a, &trace);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "mat_trace on non-square should return INVALID_ARGUMENT, got %d", (int)s);

    lmmc_mat_destroy(&a);
    return 0;
}


static int test_mat_det_non_square(void)
{
    lmmc_mat_t a;
    lmmc_mat_create(3, 2, &a);

    lmmc_real_t det;
    lmmc_status_t s = lmmc_mat_det(&a, &det);
    CHECK(s == LMMC_STATUS_INVALID_ARGUMENT,
          "mat_det on non-square should return INVALID_ARGUMENT, got %d", (int)s);

    lmmc_mat_destroy(&a);
    return 0;
}


static int test_vec_norm2_known(void)
{
    lmmc_vec_t v;
    lmmc_vec_create(2, &v);
    v.data[0] = 3.0;
    v.data[1] = 4.0;

    lmmc_real_t norm;
    lmmc_status_t s = lmmc_vec_norm2(&v, &norm);
    CHECK(s == LMMC_STATUS_OK, "norm2 returned error %d", (int)s);
    CHECK(fabs(norm - 5.0) < TOL,
          "norm2([3,4]) should be 5.0, got %g", norm);

    lmmc_vec_destroy(&v);
    return 0;
}


static int test_vec_asum_known(void)
{
    lmmc_vec_t v;
    lmmc_vec_create(3, &v);
    v.data[0] = -1.0;
    v.data[1] = 2.0;
    v.data[2] = -3.0;

    lmmc_real_t asum;
    lmmc_status_t s = lmmc_vec_asum(&v, &asum);
    CHECK(s == LMMC_STATUS_OK, "asum returned error %d", (int)s);
    CHECK(fabs(asum - 6.0) < TOL,
          "asum([-1,2,-3]) should be 6.0, got %g", asum);

    lmmc_vec_destroy(&v);
    return 0;
}


static int test_vec_iamax_known(void)
{
    lmmc_vec_t v;
    lmmc_vec_create(3, &v);
    v.data[0] = 1.0;
    v.data[1] = -5.0;
    v.data[2] = 3.0;

    size_t idx;
    lmmc_status_t s = lmmc_vec_iamax(&v, &idx);
    CHECK(s == LMMC_STATUS_OK, "iamax returned error %d", (int)s);
    CHECK(idx == 1,
          "iamax([1,-5,3]) should be 1, got %zu", idx);

    lmmc_vec_destroy(&v);
    return 0;
}


static int test_mat_trace_known(void)
{
    lmmc_mat_t m;
    lmmc_mat_create(2, 2, &m);
    m.data[0 * m.stride + 0] = 1.0;
    m.data[0 * m.stride + 1] = 2.0;
    m.data[1 * m.stride + 0] = 3.0;
    m.data[1 * m.stride + 1] = 4.0;

    lmmc_real_t trace;
    lmmc_status_t s = lmmc_mat_trace(&m, &trace);
    CHECK(s == LMMC_STATUS_OK, "trace returned error %d", (int)s);
    CHECK(fabs(trace - 5.0) < TOL,
          "trace([[1,2],[3,4]]) should be 5.0, got %g", trace);

    lmmc_mat_destroy(&m);
    return 0;
}


static int test_mat_det_known(void)
{
    lmmc_mat_t m;
    lmmc_mat_create(2, 2, &m);
    m.data[0 * m.stride + 0] = 1.0;
    m.data[0 * m.stride + 1] = 2.0;
    m.data[1 * m.stride + 0] = 3.0;
    m.data[1 * m.stride + 1] = 4.0;

    lmmc_real_t det;
    lmmc_status_t s = lmmc_mat_det(&m, &det);
    CHECK(s == LMMC_STATUS_OK, "det returned error %d", (int)s);
    CHECK(fabs(det - (-2.0)) < TOL,
          "det([[1,2],[3,4]]) should be -2.0, got %g", det);

    lmmc_mat_destroy(&m);
    return 0;
}


int main(void)
{
    int rc = 0;

    printf("=== Dense Module Unit Tests ===\n");
    printf("  Validates: Requirements 2.9, 2.10, 3.7, 3.8\n\n");


    printf("--- Vector Error Cases ---\n");

    if (test_vec_norm2_null()) { rc = 1; printf("  [FAIL] vec_norm2 NULL\n"); }
    else { printf("  [PASS] vec_norm2 NULL returns INVALID_ARGUMENT\n"); }

    if (test_vec_axpy_dimension_mismatch()) { rc = 1; printf("  [FAIL] vec_axpy dimension mismatch\n"); }
    else { printf("  [PASS] vec_axpy dimension mismatch returns DIMENSION_MISMATCH\n"); }

    if (test_vec_copy_dimension_mismatch()) { rc = 1; printf("  [FAIL] vec_copy dimension mismatch\n"); }
    else { printf("  [PASS] vec_copy dimension mismatch returns DIMENSION_MISMATCH\n"); }

    if (test_vec_swap_dimension_mismatch()) { rc = 1; printf("  [FAIL] vec_swap dimension mismatch\n"); }
    else { printf("  [PASS] vec_swap dimension mismatch returns DIMENSION_MISMATCH\n"); }


    printf("\n--- Matrix Error Cases ---\n");

    if (test_mat_add_dimension_mismatch()) { rc = 1; printf("  [FAIL] mat_add dimension mismatch\n"); }
    else { printf("  [PASS] mat_add dimension mismatch returns DIMENSION_MISMATCH\n"); }

    if (test_mat_trace_non_square()) { rc = 1; printf("  [FAIL] mat_trace non-square\n"); }
    else { printf("  [PASS] mat_trace non-square returns INVALID_ARGUMENT\n"); }

    if (test_mat_det_non_square()) { rc = 1; printf("  [FAIL] mat_det non-square\n"); }
    else { printf("  [PASS] mat_det non-square returns INVALID_ARGUMENT\n"); }


    printf("\n--- Known-Result Tests ---\n");

    if (test_vec_norm2_known()) { rc = 1; printf("  [FAIL] norm2 known result\n"); }
    else { printf("  [PASS] norm2([3,4]) = 5\n"); }

    if (test_vec_asum_known()) { rc = 1; printf("  [FAIL] asum known result\n"); }
    else { printf("  [PASS] asum([-1,2,-3]) = 6\n"); }

    if (test_vec_iamax_known()) { rc = 1; printf("  [FAIL] iamax known result\n"); }
    else { printf("  [PASS] iamax([1,-5,3]) = 1\n"); }

    if (test_mat_trace_known()) { rc = 1; printf("  [FAIL] trace known result\n"); }
    else { printf("  [PASS] trace([[1,2],[3,4]]) = 5\n"); }

    if (test_mat_det_known()) { rc = 1; printf("  [FAIL] det known result\n"); }
    else { printf("  [PASS] det([[1,2],[3,4]]) = -2\n"); }


    printf("\n");
    if (rc == 0) {
        printf("All dense module unit tests PASSED.\n");
    } else {
        printf("Some dense module unit tests FAILED (%d failures).\n", test_failures);
    }

    return rc;
}
