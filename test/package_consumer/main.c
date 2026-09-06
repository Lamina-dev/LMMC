#include <lmmc/lmmc.h>
#include <lmmp/lmmpn.h>

#include <math.h>
#include <stdio.h>

int main(void)
{
    int result = 1;
    lmmc_mat_t product = {0};
    lmmc_vec_t cross = {0};
    lmmc_std_num_set_t set = {0};
    lmmc_real_t matrix_values[] = {1.0, 2.0, 3.0, 4.0};
    lmmc_real_t a_values[] = {1.0, 2.0, 3.0};
    lmmc_real_t b_values[] = {4.0, 5.0, 6.0};
    lmmc_real_t set_values[] = {1.0, 2.0, 2.0, -0.0};
    lmmc_mat_t matrix = {2, 2, 2, matrix_values, 0};
    lmmc_vec_t a = {3, a_values, 0};
    lmmc_vec_t b = {3, b_values, 0};
    lmmc_real_t dot = 0.0;
    lmmc_real_t norm = 0.0;
    int contains = 0;
    const mp_limb_t limb_a[] = {LIMB_MAX, 2};
    const mp_limb_t limb_b[] = {1, 3};
    mp_limb_t limb_sum[2] = {0};

    if (lmmc_init() != LMMC_STATUS_OK) {
        fputs("installed LMMC initialization failed\n", stderr);
        return 1;
    }

    /* A direct LMMP call requires the package's transitive link dependency. */
    if (lmmp_add_n_(limb_sum, limb_a, limb_b, 2) != 0 ||
        limb_sum[0] != 0 || limb_sum[1] != 6) {
        fputs("installed LMMP addition failed\n", stderr);
        goto cleanup;
    }

    if (lmmc_mat_create(2, 2, &product) != LMMC_STATUS_OK ||
        lmmc_mat_gemm(1.0, &matrix, 0, &matrix, 0, 0.0, &product) !=
            LMMC_STATUS_OK ||
        product.rows != 2 || product.cols != 2 || !product.data ||
        product.data[0] != 7.0 || product.data[1] != 10.0 ||
        product.data[product.stride] != 15.0 ||
        product.data[product.stride + 1] != 22.0 ||
        lmmc_vec_dot(&a, &b, &dot) != LMMC_STATUS_OK || dot != 32.0) {
        fputs("installed LMMC dense arithmetic failed\n", stderr);
        goto cleanup;
    }

    if (lmmc_std_linalg_norm(&a, &norm) != LMMC_STATUS_OK ||
        !(fabs(norm - sqrt(14.0)) <= 1e-12) ||
        lmmc_std_linalg_cross(&a, &b, &cross) != LMMC_STATUS_OK ||
        cross.size != 3 || !cross.data || cross.data[0] != -3.0 ||
        cross.data[1] != 6.0 || cross.data[2] != -3.0) {
        fputs("installed LMMC standard-library linear algebra failed\n", stderr);
        goto cleanup;
    }

    if (lmmc_std_num_set_make(set_values, 4, &set) != LMMC_STATUS_OK ||
        set.size != 3 ||
        lmmc_std_num_set_contains(&set, 0.0, &contains) != LMMC_STATUS_OK ||
        !contains ||
        lmmc_std_num_set_contains(&set, 3.0, &contains) != LMMC_STATUS_OK ||
        contains) {
        fputs("installed LMMC standard-library numeric set failed\n", stderr);
        goto cleanup;
    }

    result = 0;
cleanup:
    lmmc_std_num_set_destroy(&set);
    lmmc_vec_destroy(&cross);
    lmmc_mat_destroy(&product);
    if (lmmc_deinit() != LMMC_STATUS_OK) {
        fputs("installed LMMC deinitialization failed\n", stderr);
        result = 1;
    }
    if (result == 0) {
        puts("LMMC installed package checks passed");
    }
    return result;
}
