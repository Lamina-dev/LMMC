/**
 * @file test_init.c
 * 针对 LMMC 中 init 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    int rc = 0;


    {
        lmmc_init();
        lmmc_deinit();
    }


    {
        lmmc_init();
        lmmc_init();
        lmmc_init();

        lmmc_deinit();
        lmmc_deinit();
        lmmc_deinit();
    }


    {
        lmmc_vec_t v = {0};
        lmmc_real_t dot_result = 0.0;
        lmmc_status_t st;

        lmmc_init();


        st = lmmc_vec_create(3, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        v.data[0] = 1.0;
        v.data[1] = 2.0;
        v.data[2] = 3.0;

        st = lmmc_vec_dot(&v, &v, &dot_result);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


        if (!lmmc_test_nearly_equal(dot_result, 14.0, 1e-12)) { rc = 1; goto done; }

        lmmc_vec_destroy(&v);


        {
            lmmc_mat_t a = {0}, b = {0}, c = {0};

            st = lmmc_mat_create(2, 2, &a);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            st = lmmc_mat_create(2, 2, &b);
            if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); rc = 1; goto done; }
            st = lmmc_mat_create(2, 2, &c);
            if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); rc = 1; goto done; }


            a.data[0] = 1.0; a.data[1] = 0.0;
            a.data[2] = 0.0; a.data[3] = 1.0;


            b.data[0] = 2.0; b.data[1] = 3.0;
            b.data[2] = 4.0; b.data[3] = 5.0;

            st = lmmc_mat_mul(&a, &b, &c);
            if (st != LMMC_STATUS_OK) {
                lmmc_mat_destroy(&a);
                lmmc_mat_destroy(&b);
                lmmc_mat_destroy(&c);
                rc = 1; goto done;
            }


            if (!lmmc_test_nearly_equal(c.data[0], 2.0, 1e-12) ||
                !lmmc_test_nearly_equal(c.data[1], 3.0, 1e-12) ||
                !lmmc_test_nearly_equal(c.data[2], 4.0, 1e-12) ||
                !lmmc_test_nearly_equal(c.data[3], 5.0, 1e-12)) {
                lmmc_mat_destroy(&a);
                lmmc_mat_destroy(&b);
                lmmc_mat_destroy(&c);
                rc = 1; goto done;
            }

            lmmc_mat_destroy(&a);
            lmmc_mat_destroy(&b);
            lmmc_mat_destroy(&c);
        }

        lmmc_deinit();
    }


    {
        lmmc_vec_t v = {0};
        lmmc_real_t norm = 0.0;
        lmmc_status_t st;

        lmmc_init();


        lmmc_stack_reset(327680);


        st = lmmc_vec_create(4, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        v.data[0] = 3.0;
        v.data[1] = 4.0;
        v.data[2] = 0.0;
        v.data[3] = 0.0;

        st = lmmc_vec_norm2(&v, &norm);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); rc = 1; goto done; }


        if (!lmmc_test_nearly_equal(norm, 5.0, 1e-12)) {
            lmmc_vec_destroy(&v);
            rc = 1; goto done;
        }

        lmmc_vec_destroy(&v);


        lmmc_stack_reset(0);


        st = lmmc_vec_create(2, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        v.data[0] = 1.0;
        v.data[1] = 1.0;

        st = lmmc_vec_norm2(&v, &norm);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); rc = 1; goto done; }


        if (!lmmc_test_nearly_equal(norm, sqrt(2.0), 1e-12)) {
            lmmc_vec_destroy(&v);
            rc = 1; goto done;
        }

        lmmc_vec_destroy(&v);
        lmmc_deinit();
    }


    {
        lmmc_vec_t v = {0};
        lmmc_real_t dot_result = 0.0;
        lmmc_status_t st;


        lmmc_init();
        lmmc_deinit();


        lmmc_init();

        st = lmmc_vec_create(2, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        v.data[0] = 5.0;
        v.data[1] = 12.0;

        st = lmmc_vec_dot(&v, &v, &dot_result);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); rc = 1; goto done; }


        if (!lmmc_test_nearly_equal(dot_result, 169.0, 1e-12)) {
            lmmc_vec_destroy(&v);
            rc = 1; goto done;
        }

        lmmc_vec_destroy(&v);
        lmmc_deinit();
    }

done:
    if (rc != 0) {
        printf("init test failed\n");
    }
    return rc;
}
