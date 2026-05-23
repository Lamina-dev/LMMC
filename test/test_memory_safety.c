#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/*
 * test_memory_safety.c
 * 内存安全与资源管理测试
 * Validates: Requirements 19.1-19.5
 */

int main(void) {
    int rc = 0;

    /* ===================================================================
     * 19.1: 创建后立即销毁验证 — 不产生内存泄漏
     * =================================================================== */

    /* vec create/destroy */
    {
        lmmc_vec_t v = {0};
        lmmc_status_t st = lmmc_vec_create(10, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_destroy(&v);
    }

    /* mat create/destroy */
    {
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_create(5, 5, &m);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_mat_destroy(&m);
    }

    /* tensor create/destroy */
    {
        lmmc_tensor_t t = {0};
        lmmc_status_t st = lmmc_tensor3_create(3, 4, 5, &t);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_tensor_destroy(&t);
    }

    /* sparse CSR create/destroy */
    {
        lmmc_sparse_mat_t sp = {0};
        lmmc_status_t st = lmmc_sparse_create_csr(5, 5, 10, &sp);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_sparse_destroy(&sp);
    }

    /* rng create/destroy */
    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_destroy(rng);
    }

    /* precond create_none/destroy */
    {
        lmmc_precond_t pc = {0};
        lmmc_status_t st = lmmc_precond_create_none(10, &pc);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_precond_destroy(&pc);
    }

    /* interp cspline create/destroy */
    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
        lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};
        lmmc_interp_cspline_t* spline = NULL;
        lmmc_status_t st = lmmc_interp_cspline_create(xs, ys, 5, &spline);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_interp_cspline_destroy(spline);
    }

    /* interp lagrange create/destroy */
    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
        lmmc_real_t ys[] = {1.0, 2.0, 5.0, 10.0};
        lmmc_interp_lagrange_t* lag = NULL;
        lmmc_status_t st = lmmc_interp_lagrange_create(xs, ys, 4, &lag);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_interp_lagrange_destroy(lag);
    }

    /* ===================================================================
     * 19.2: NULL 指针 destroy 安全验证 — 所有 destroy 函数传 NULL 不崩溃
     * =================================================================== */

    /* vec destroy NULL */
    {
        lmmc_vec_t v = {0};
        lmmc_vec_destroy(&v);  /* zero-initialized, data is NULL */
        lmmc_vec_destroy(NULL);
    }

    /* mat destroy NULL */
    {
        lmmc_mat_t m = {0};
        lmmc_mat_destroy(&m);
        lmmc_mat_destroy(NULL);
    }

    /* tensor destroy NULL */
    {
        lmmc_tensor_t t = {0};
        lmmc_tensor_destroy(&t);
        lmmc_tensor_destroy(NULL);
    }

    /* sparse destroy NULL */
    {
        lmmc_sparse_mat_t sp = {0};
        lmmc_sparse_destroy(&sp);
        lmmc_sparse_destroy(NULL);
    }

    /* rng destroy NULL */
    {
        lmmc_rng_destroy(NULL);
    }

    /* precond destroy NULL */
    {
        lmmc_precond_t pc = {0};
        lmmc_precond_destroy(&pc);
        lmmc_precond_destroy(NULL);
    }

    /* interp cspline destroy NULL */
    {
        lmmc_interp_cspline_destroy(NULL);
    }

    /* interp lagrange destroy NULL */
    {
        lmmc_interp_lagrange_destroy(NULL);
    }

    /* ===================================================================
     * 19.3: 错误路径资源释放验证 — 触发错误后无泄漏
     * =================================================================== */

    /* vec_create with size 0 should fail without leaking */
    {
        lmmc_vec_t v = {0};
        lmmc_status_t st = lmmc_vec_create(0, &v);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
        /* No cleanup needed — function should not have allocated */
    }

    /* mat_create with 0 rows should fail without leaking */
    {
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_create(0, 5, &m);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* mat_create with 0 cols should fail without leaking */
    {
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_create(5, 0, &m);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* tensor create with 0 dimension should fail */
    {
        lmmc_tensor_t t = {0};
        lmmc_status_t st = lmmc_tensor3_create(0, 4, 5, &t);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* interp cspline with too few points should fail */
    {
        lmmc_real_t xs[] = {0.0, 1.0};
        lmmc_real_t ys[] = {0.0, 1.0};
        lmmc_interp_cspline_t* spline = NULL;
        lmmc_status_t st = lmmc_interp_cspline_create(xs, ys, 2, &spline);
        /* cspline requires at least 3 points */
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* interp lagrange with 0 points should fail */
    {
        lmmc_interp_lagrange_t* lag = NULL;
        lmmc_status_t st = lmmc_interp_lagrange_create(NULL, NULL, 0, &lag);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

    /* ===================================================================
     * 19.4: wrap 对象 destroy 不释放外部数据验证
     * =================================================================== */

    /* vec wrap: destroy wrapper, verify external data still accessible */
    {
        lmmc_real_t ext_data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        lmmc_vec_t v = {0};
        lmmc_status_t st = lmmc_vec_wrap(5, ext_data, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_vec_destroy(&v);
        /* External data must still be valid */
        if (!lmmc_test_nearly_equal(ext_data[0], 1.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[2], 3.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[4], 5.0, 1e-15)) {
            rc = 1; goto done;
        }
    }

    /* mat wrap: destroy wrapper, verify external data still accessible */
    {
        lmmc_real_t ext_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        lmmc_mat_t m = {0};
        lmmc_status_t st = lmmc_mat_wrap(2, 3, 3, ext_data, &m);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_mat_destroy(&m);
        /* External data must still be valid */
        if (!lmmc_test_nearly_equal(ext_data[0], 1.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[3], 4.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[5], 6.0, 1e-15)) {
            rc = 1; goto done;
        }
    }

    /* tensor wrap: destroy wrapper, verify external data still accessible */
    {
        lmmc_real_t ext_data[24];
        for (int i = 0; i < 24; i++) ext_data[i] = (lmmc_real_t)(i + 1);
        lmmc_tensor_t t = {0};
        /* 2x3x4 tensor, row-major strides: stride0=12, stride1=4, stride2=1 */
        lmmc_status_t st = lmmc_tensor3_wrap(2, 3, 4, 12, 4, 1, ext_data, &t);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_tensor_destroy(&t);
        /* External data must still be valid */
        if (!lmmc_test_nearly_equal(ext_data[0], 1.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[12], 13.0, 1e-15) ||
            !lmmc_test_nearly_equal(ext_data[23], 24.0, 1e-15)) {
            rc = 1; goto done;
        }
    }

    /* sparse wrap CSR: destroy wrapper, verify external data still accessible */
    {
        /* Simple 3x3 identity matrix in CSR */
        size_t row_ptr[] = {0, 1, 2, 3};
        size_t col_idx[] = {0, 1, 2};
        lmmc_real_t values[] = {1.0, 1.0, 1.0};
        lmmc_sparse_mat_t sp = {0};
        lmmc_status_t st = lmmc_sparse_wrap_csr(3, 3, 3, row_ptr, col_idx, values, &sp);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_sparse_destroy(&sp);
        /* External data must still be valid */
        if (!lmmc_test_nearly_equal(values[0], 1.0, 1e-15) ||
            !lmmc_test_nearly_equal(values[1], 1.0, 1e-15) ||
            !lmmc_test_nearly_equal(values[2], 1.0, 1e-15)) {
            rc = 1; goto done;
        }
        if (row_ptr[0] != 0 || row_ptr[3] != 3) { rc = 1; goto done; }
        if (col_idx[0] != 0 || col_idx[2] != 2) { rc = 1; goto done; }
    }

    /* ===================================================================
     * 19.5: 循环创建/销毁 1000 次压力测试
     * =================================================================== */

    /* vec stress test */
    {
        for (int i = 0; i < 1000; i++) {
            lmmc_vec_t v = {0};
            lmmc_status_t st = lmmc_vec_create(100, &v);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_vec_destroy(&v);
        }
    }

    /* mat stress test */
    {
        for (int i = 0; i < 1000; i++) {
            lmmc_mat_t m = {0};
            lmmc_status_t st = lmmc_mat_create(10, 10, &m);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_mat_destroy(&m);
        }
    }

    /* tensor stress test */
    {
        for (int i = 0; i < 1000; i++) {
            lmmc_tensor_t t = {0};
            lmmc_status_t st = lmmc_tensor3_create(3, 3, 3, &t);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_tensor_destroy(&t);
        }
    }

    /* sparse stress test */
    {
        for (int i = 0; i < 1000; i++) {
            lmmc_sparse_mat_t sp = {0};
            lmmc_status_t st = lmmc_sparse_create_csr(5, 5, 5, &sp);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_sparse_destroy(&sp);
        }
    }

    /* rng stress test */
    {
        for (int i = 0; i < 1000; i++) {
            lmmc_rng_t* rng = NULL;
            lmmc_status_t st = lmmc_rng_create(&rng);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_rng_destroy(rng);
        }
    }

    /* precond stress test */
    {
        for (int i = 0; i < 1000; i++) {
            lmmc_precond_t pc = {0};
            lmmc_status_t st = lmmc_precond_create_none(10, &pc);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_precond_destroy(&pc);
        }
    }

    /* interp cspline stress test */
    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
        lmmc_real_t ys[] = {0.0, 1.0, 4.0, 9.0, 16.0};
        for (int i = 0; i < 1000; i++) {
            lmmc_interp_cspline_t* spline = NULL;
            lmmc_status_t st = lmmc_interp_cspline_create(xs, ys, 5, &spline);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_interp_cspline_destroy(spline);
        }
    }

    /* interp lagrange stress test */
    {
        lmmc_real_t xs[] = {0.0, 1.0, 2.0, 3.0};
        lmmc_real_t ys[] = {1.0, 2.0, 5.0, 10.0};
        for (int i = 0; i < 1000; i++) {
            lmmc_interp_lagrange_t* lag = NULL;
            lmmc_status_t st = lmmc_interp_lagrange_create(xs, ys, 4, &lag);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            lmmc_interp_lagrange_destroy(lag);
        }
    }

done:
    if (rc != 0) {
        printf("memory_safety test failed\n");
    }
    return rc;
}
