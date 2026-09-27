#include <lmmc/lmmc.h>
#include <lmmp/lmmpn.h>

#include <math.h>
#include <stdio.h>

static int check_transitive_lmmp(void)
{
    const mp_limb_t limb_a[] = {LIMB_MAX, 2};
    const mp_limb_t limb_b[] = {1, 3};
    mp_limb_t limb_sum[2] = {0};

    return lmmp_add_n_(limb_sum, limb_a, limb_b, 2) != 0 ||
           limb_sum[0] != 0 || limb_sum[1] != 6;
}

static int check_dense_arithmetic(lmmc_mat_t* product,
                                  const lmmc_vec_t* a, const lmmc_vec_t* b)
{
    lmmc_real_t matrix_values[] = {1.0, 2.0, 3.0, 4.0};
    lmmc_mat_t matrix = {2, 2, 2, matrix_values, 0};
    lmmc_real_t dot = 0.0;
    if (lmmc_mat_create(2, 2, product) != LMMC_STATUS_OK ||
        lmmc_mat_gemm(1.0, &matrix, 0, &matrix, 0, 0.0, product) !=
            LMMC_STATUS_OK) {
        return 1;
    }
    if (product->rows != 2 || product->cols != 2 || !product->data) {
        return 1;
    }
    if (product->data[0] != 7.0 || product->data[1] != 10.0 ||
        product->data[product->stride] != 15.0 ||
        product->data[product->stride + 1] != 22.0) {
        return 1;
    }
    return lmmc_vec_dot(a, b, &dot) != LMMC_STATUS_OK || dot != 32.0;
}

static int check_standard_linear_algebra(const lmmc_vec_t* a,
                                         const lmmc_vec_t* b, lmmc_vec_t* cross)
{
    lmmc_real_t norm = 0.0;
    if (lmmc_std_linalg_norm(a, &norm) != LMMC_STATUS_OK ||
        !(fabs(norm - sqrt(14.0)) <= 1e-12)) {
        return 1;
    }
    if (lmmc_std_linalg_cross(a, b, cross) != LMMC_STATUS_OK) {
        return 1;
    }
    if (cross->size != 3 || !cross->data) {
        return 1;
    }
    return cross->data[0] != -3.0 || cross->data[1] != 6.0 ||
           cross->data[2] != -3.0;
}

static int check_numeric_set(lmmc_std_num_set_t* set)
{
    lmmc_real_t set_values[] = {1.0, 2.0, 2.0, -0.0};
    int contains = 0;
    return lmmc_std_num_set_make(set_values, 4, set) != LMMC_STATUS_OK ||
           set->size != 3 ||
           lmmc_std_num_set_contains(set, 0.0, &contains) != LMMC_STATUS_OK ||
           !contains ||
           lmmc_std_num_set_contains(set, 3.0, &contains) != LMMC_STATUS_OK ||
           contains;
}

static lmmc_status_t lsqr_forward(const lmmc_vec_t* x, lmmc_vec_t* y,
                                  void* user_data)
{
    (void)user_data;
    if (x->size != 2 || y->size != 3) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    y->data[0] = x->data[0];
    y->data[1] = x->data[1];
    y->data[2] = x->data[0] + x->data[1];
    return LMMC_STATUS_OK;
}

static lmmc_status_t lsqr_transpose(const lmmc_vec_t* x, lmmc_vec_t* y,
                                    void* user_data)
{
    (void)user_data;
    if (x->size != 3 || y->size != 2) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    y->data[0] = x->data[0] + x->data[2];
    y->data[1] = x->data[1] + x->data[2];
    return LMMC_STATUS_OK;
}

static int check_matrix_free_lsqr(void)
{
    lmmc_real_t rhs_values[] = {1.0, 2.0, 3.0};
    lmmc_real_t solution_values[] = {0.0, 0.0};
    lmmc_vec_t rhs = {3, rhs_values, 0};
    lmmc_vec_t solution = {2, solution_values, 0};
    lmmc_itersolve_config_t config = {0};
    lmmc_itersolve_result_t solve_result = {0};

    if (lmmc_itersolve_default_config(2, &config) != LMMC_STATUS_OK) {
        return 1;
    }
    config.abs_tol = 1e-12;
    config.rel_tol = 1e-12;
    config.apply_op = lsqr_forward;
    config.apply_transpose_op = lsqr_transpose;
    if (lmmc_lsqr_solve(NULL, &rhs, &config, &solution, &solve_result) !=
            LMMC_STATUS_OK ||
        !solve_result.converged) {
        return 1;
    }
    return fabs(solution.data[0] - 1.0) > 1e-10 ||
           fabs(solution.data[1] - 2.0) > 1e-10;
}

int main(void)
{
    int result = 1;
    lmmc_mat_t product = {0};
    lmmc_vec_t cross = {0};
    lmmc_std_num_set_t set = {0};
    lmmc_real_t a_values[] = {1.0, 2.0, 3.0};
    lmmc_real_t b_values[] = {4.0, 5.0, 6.0};
    lmmc_vec_t a = {3, a_values, 0};
    lmmc_vec_t b = {3, b_values, 0};

    if (lmmc_init() != LMMC_STATUS_OK) {
        fputs("installed LMMC initialization failed\n", stderr);
        return 1;
    }
    if (check_transitive_lmmp() != 0) {
        fputs("installed LMMP addition failed\n", stderr);
        goto cleanup;
    }
    if (check_dense_arithmetic(&product, &a, &b) != 0) {
        fputs("installed LMMC dense arithmetic failed\n", stderr);
        goto cleanup;
    }
    if (check_standard_linear_algebra(&a, &b, &cross) != 0) {
        fputs("installed LMMC standard-library linear algebra failed\n", stderr);
        goto cleanup;
    }
    if (check_numeric_set(&set) != 0) {
        fputs("installed LMMC standard-library numeric set failed\n", stderr);
        goto cleanup;
    }
    if (check_matrix_free_lsqr() != 0) {
        fputs("installed LMMC matrix-free LSQR failed\n", stderr);
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
