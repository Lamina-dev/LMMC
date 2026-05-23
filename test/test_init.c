#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

int main(void) {
    int rc = 0;

    /* ===== Test 1: Basic init/deinit flow (Requirement 3.1, 3.2) ===== */
    /* lmmc_init() and lmmc_deinit() are void functions.
     * We verify they don't crash and that the library is usable after init. */
    {
        lmmc_init();
        lmmc_deinit();
    }

    /* ===== Test 2: Repeated initialization safety (Requirement 3.3) ===== */
    /* Reference-counted: multiple init calls should not cause errors.
     * Each init must be paired with a deinit. */
    {
        lmmc_init();
        lmmc_init();
        lmmc_init();
        /* Three inits — deinit three times */
        lmmc_deinit();
        lmmc_deinit();
        lmmc_deinit();
    }

    /* ===== Test 3: Basic numeric operations after init (Requirement 3.4) ===== */
    /* Verify that after initialization, basic vector/matrix operations work. */
    {
        lmmc_vec_t v = {0};
        lmmc_real_t dot_result = 0.0;
        lmmc_status_t st;

        lmmc_init();

        /* Create a vector and perform dot product */
        st = lmmc_vec_create(3, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        v.data[0] = 1.0;
        v.data[1] = 2.0;
        v.data[2] = 3.0;

        st = lmmc_vec_dot(&v, &v, &dot_result);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        /* dot(v, v) = 1^2 + 2^2 + 3^2 = 14 */
        if (!lmmc_test_nearly_equal(dot_result, 14.0, 1e-12)) { rc = 1; goto done; }

        lmmc_vec_destroy(&v);

        /* Test matrix creation and multiplication */
        {
            lmmc_mat_t a = {0}, b = {0}, c = {0};

            st = lmmc_mat_create(2, 2, &a);
            if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
            st = lmmc_mat_create(2, 2, &b);
            if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); rc = 1; goto done; }
            st = lmmc_mat_create(2, 2, &c);
            if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); rc = 1; goto done; }

            /* a = identity */
            a.data[0] = 1.0; a.data[1] = 0.0;
            a.data[2] = 0.0; a.data[3] = 1.0;

            /* b = [[2, 3], [4, 5]] */
            b.data[0] = 2.0; b.data[1] = 3.0;
            b.data[2] = 4.0; b.data[3] = 5.0;

            st = lmmc_mat_mul(&a, &b, &c);
            if (st != LMMC_STATUS_OK) {
                lmmc_mat_destroy(&a);
                lmmc_mat_destroy(&b);
                lmmc_mat_destroy(&c);
                rc = 1; goto done;
            }

            /* I * B = B */
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

    /* ===== Test 4: stack_reset functionality (Requirement 3.4) ===== */
    /* Verify lmmc_stack_reset does not crash and operations still work after. */
    {
        lmmc_vec_t v = {0};
        lmmc_real_t norm = 0.0;
        lmmc_status_t st;

        lmmc_init();

        /* Reset stack to a specific size */
        lmmc_stack_reset(327680); /* 320 KiB */

        /* Verify operations still work after stack reset */
        st = lmmc_vec_create(4, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        v.data[0] = 3.0;
        v.data[1] = 4.0;
        v.data[2] = 0.0;
        v.data[3] = 0.0;

        st = lmmc_vec_norm2(&v, &norm);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); rc = 1; goto done; }

        /* norm2([3, 4, 0, 0]) = 5.0 */
        if (!lmmc_test_nearly_equal(norm, 5.0, 1e-12)) {
            lmmc_vec_destroy(&v);
            rc = 1; goto done;
        }

        lmmc_vec_destroy(&v);

        /* Reset stack to 0 (free entire stack) */
        lmmc_stack_reset(0);

        /* Operations should still work after freeing the stack
         * (the library should re-allocate as needed) */
        st = lmmc_vec_create(2, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        v.data[0] = 1.0;
        v.data[1] = 1.0;

        st = lmmc_vec_norm2(&v, &norm);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); rc = 1; goto done; }

        /* norm2([1, 1]) = sqrt(2) */
        if (!lmmc_test_nearly_equal(norm, sqrt(2.0), 1e-12)) {
            lmmc_vec_destroy(&v);
            rc = 1; goto done;
        }

        lmmc_vec_destroy(&v);
        lmmc_deinit();
    }

    /* ===== Test 5: Init after full deinit cycle (Requirement 3.3) ===== */
    /* Verify that re-initializing after a complete init/deinit cycle works. */
    {
        lmmc_vec_t v = {0};
        lmmc_real_t dot_result = 0.0;
        lmmc_status_t st;

        /* First cycle */
        lmmc_init();
        lmmc_deinit();

        /* Second cycle — should work fine */
        lmmc_init();

        st = lmmc_vec_create(2, &v);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }

        v.data[0] = 5.0;
        v.data[1] = 12.0;

        st = lmmc_vec_dot(&v, &v, &dot_result);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); rc = 1; goto done; }

        /* dot([5,12], [5,12]) = 25 + 144 = 169 */
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
