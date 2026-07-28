#include "lmmc/lmmc.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int close_real(lmmc_real_t a, lmmc_real_t b)
{
    return fabs((double)(a - b)) <= 1e-12;
}

static void set_mat_values(lmmc_mat_t* mat, const lmmc_real_t* values)
{
    size_t k = 0;
    for (size_t i = 0; i < mat->rows; ++i) {
        for (size_t j = 0; j < mat->cols; ++j) {
            mat->data[i * mat->stride + j] = values[k++];
        }
    }
}

static int mat_close_at(const lmmc_mat_t* mat, size_t row, size_t col,
                        lmmc_real_t expected)
{
    return close_real(mat->data[row * mat->stride + col], expected);
}

int main(void)
{
    lmmc_real_t out = 0;
    lmmc_real_t out2 = 0;
    lmmc_real_t values[] = {1, 2, 3, 4};
    lmmc_real_t scaled_values[] = {2, 4, 6, 8};
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    lmmc_complex_t z;
    lmmc_complex_t w;
    lmmc_rng_t* rng = NULL;
    lmmc_mat_t mat = {0};
    lmmc_mat_t rhs = {0};
    lmmc_mat_t rectangular = {0};
    lmmc_mat_t result = {0};
    lmmc_eigen_gen_full_result_t eig_result = {0};
    lmmc_svd_result_t svd_result = {0};
    lmmc_lsr_eig_table_t eig_table = {0};
    lmmc_lsr_svd_table_t svd_table = {0};
    const lmmc_mat_t* named = NULL;
    size_t rows = 0;
    size_t cols = 0;
    size_t rank = 0;
    int64_t randint_out = 0;

    if (lmmc_lsr_math_pi(&out) != LMMC_STATUS_OK ||
        !close_real(out, LMMC_CONST_PI)) {
        fprintf(stderr, "std.math.pi mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_i(&z) != LMMC_STATUS_OK ||
        !close_real(z.real, 0) || !close_real(z.imag, 1)) {
        fprintf(stderr, "std.math.i mismatch\n");
        return 1;
    }

    if (lmmc_lsr_constants_count() < 20 ||
        lmmc_lsr_constants_name(0) == NULL ||
        lmmc_lsr_constants_name(lmmc_lsr_constants_count()) != NULL) {
        fprintf(stderr, "std.constants enumeration mismatch\n");
        return 1;
    }

    if (lmmc_lsr_constants_get("EARTH_GRAVITY", &out) != LMMC_STATUS_OK ||
        !close_real(out, 9.80665) ||
        strcmp(lmmc_lsr_constants_unit("EARTH_GRAVITY"), "m*s^-2") != 0) {
        fprintf(stderr, "std.constants.EARTH_GRAVITY mismatch\n");
        return 1;
    }

    if (lmmc_lsr_constants_get("C", &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.99792458e8) ||
        strcmp(lmmc_lsr_constants_unit("C"), "m*s^-1") != 0) {
        fprintf(stderr, "std.constants.C mismatch\n");
        return 1;
    }

    if (lmmc_lsr_constants_get("AVOGADRO", &out) != LMMC_STATUS_OK ||
        !close_real(out, 6.02214076e23) ||
        strcmp(lmmc_lsr_constants_unit("AVOGADRO"), "mol^-1") != 0) {
        fprintf(stderr, "std.constants.AVOGADRO mismatch\n");
        return 1;
    }

    if (lmmc_lsr_constants_get("NO_SUCH_CONSTANT", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_constants_unit("NO_SUCH_CONSTANT") != NULL) {
        fprintf(stderr, "std.constants unknown name mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_I(&w) != LMMC_STATUS_OK ||
        !close_real(w.real, z.real) || !close_real(w.imag, z.imag)) {
        fprintf(stderr, "std.math.I alias mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_complex(3, 4, &z) != LMMC_STATUS_OK) {
        fprintf(stderr, "std.math.complex failed\n");
        return 1;
    }

    if (lmmc_lsr_math_real(&z, &out) != LMMC_STATUS_OK ||
        !close_real(out, 3)) {
        fprintf(stderr, "std.math.real mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_imag(&z, &out) != LMMC_STATUS_OK ||
        !close_real(out, 4)) {
        fprintf(stderr, "std.math.imag mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_conj(&z, &w) != LMMC_STATUS_OK ||
        !close_real(w.real, 3) || !close_real(w.imag, -4)) {
        fprintf(stderr, "std.math.conj mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_complex_abs(&z, &out) != LMMC_STATUS_OK ||
        !close_real(out, 5)) {
        fprintf(stderr, "std.math.abs complex mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_sqrt(-1, &out) != LMMC_STATUS_OUT_OF_RANGE) {
        fprintf(stderr, "std.math.sqrt domain error not reported\n");
        return 1;
    }
    if (lmmc_lsr_error_name(LMMC_STATUS_OUT_OF_RANGE) == NULL ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_OUT_OF_RANGE),
               "DomainError") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_DIMENSION_MISMATCH),
               "DimensionMismatch") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_ALLOCATION_FAILED),
               "ResourceLimit") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_SINGULAR_MATRIX),
               "SingularMatrix") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_NUMERICAL_FAILURE),
               "NumericFailure") != 0) {
        fprintf(stderr, "LSR diagnostic status mapping mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_log(8, 2, &out) != LMMC_STATUS_OK ||
        !close_real(out, 3)) {
        fprintf(stderr, "std.math.log mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_log10(1000, &out) != LMMC_STATUS_OK ||
        !close_real(out, 3)) {
        fprintf(stderr, "std.math.log10 mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_pow(2, 3, &out) != LMMC_STATUS_OK ||
        !close_real(out, 8)) {
        fprintf(stderr, "std.math.pow mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_clamp(5, 1, 3, &out) != LMMC_STATUS_OK ||
        !close_real(out, 3)) {
        fprintf(stderr, "std.math.clamp mismatch\n");
        return 1;
    }

    int dimensionless = 0;
    if (lmmc_lsr_units_convert(36, "km/h", "m/s", &out) != LMMC_STATUS_OK ||
        !close_real(out, 10) ||
        lmmc_lsr_units_convert(10, "m", "s", &out) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_units_strip(12.5, &out) != LMMC_STATUS_OK ||
        !close_real(out, 12.5) ||
        lmmc_lsr_units_is_dimensionless("m/m", &dimensionless) !=
            LMMC_STATUS_OK ||
        !dimensionless ||
        lmmc_lsr_units_is_dimensionless("m*s^-1", &dimensionless) !=
            LMMC_STATUS_OK ||
        dimensionless ||
        lmmc_lsr_units_convert(1, "unknown", "m", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.units adapter mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_mean(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.5)) {
        fprintf(stderr, "std.stats.mean mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_median(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.5)) {
        fprintf(stderr, "std.stats.median mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_var(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, 5.0 / 3.0)) {
        fprintf(stderr, "std.stats.var mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_std(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, sqrt(5.0 / 3.0))) {
        fprintf(stderr, "std.stats.std mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_quantile(values, 4, 0.5, &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.5)) {
        fprintf(stderr, "std.stats.quantile mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_cov(values, scaled_values, 4, &out) !=
            LMMC_STATUS_OK ||
        !close_real(out, 10.0 / 3.0)) {
        fprintf(stderr, "std.stats.cov mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_corr(values, scaled_values, 4, &out) !=
            LMMC_STATUS_OK ||
        !close_real(out, 1.0)) {
        fprintf(stderr, "std.stats.corr mismatch\n");
        return 1;
    }

    if (lmmc_rng_create(&rng) != LMMC_STATUS_OK) {
        fprintf(stderr, "rng create failed\n");
        return 1;
    }

    if (lmmc_lsr_random_seed(rng, 42) != LMMC_STATUS_OK ||
        lmmc_lsr_random_rand(rng, &out) != LMMC_STATUS_OK ||
        lmmc_lsr_random_seed(rng, 42) != LMMC_STATUS_OK ||
        lmmc_lsr_random_rand(rng, &out2) != LMMC_STATUS_OK ||
        !close_real(out, out2)) {
        fprintf(stderr, "std.random fixed seed is not reproducible\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    if (lmmc_lsr_random_randint(rng, 1, 3, &randint_out) != LMMC_STATUS_OK ||
        randint_out < 1 || randint_out > 3) {
        fprintf(stderr, "std.random.randint range mismatch\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    if (lmmc_lsr_random_choice(rng, values, 4, &out) != LMMC_STATUS_OK ||
        out < 1 || out > 4) {
        fprintf(stderr, "std.random.choice mismatch\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    lmmc_rng_destroy(rng);

    if (lmmc_mat_create(2, 2, &mat) != LMMC_STATUS_OK ||
        lmmc_mat_create(2, 2, &rhs) != LMMC_STATUS_OK ||
        lmmc_mat_create(3, 2, &rectangular) != LMMC_STATUS_OK) {
        fprintf(stderr, "matrix allocation failed\n");
        return 1;
    }
    set_mat_values(&mat, matrix_values);
    set_mat_values(&rhs, rhs_values);
    set_mat_values(&rectangular, rectangular_values);

    if (lmmc_lsr_linalg_shape(&mat, &rows, &cols) != LMMC_STATUS_OK ||
        rows != 2 || cols != 2) {
        fprintf(stderr, "std.linalg.shape mismatch\n");
        return 1;
    }

    if (lmmc_lsr_linalg_transpose(&mat, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 1, 3) ||
        !mat_close_at(&result, 1, 0, 2)) {
        fprintf(stderr, "std.linalg.transpose mismatch\n");
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_adjoint(&mat, &result) != LMMC_STATUS_OK ||
        !mat_close_at(&result, 0, 1, 3) ||
        !mat_close_at(&result, 1, 0, 2)) {
        fprintf(stderr, "std.linalg.adjoint real mismatch\n");
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_det(&mat, &out) != LMMC_STATUS_OK ||
        !close_real(out, -2)) {
        fprintf(stderr, "std.linalg.det mismatch\n");
        return 1;
    }

    if (lmmc_lsr_linalg_trace(&mat, &out) != LMMC_STATUS_OK ||
        !close_real(out, 5)) {
        fprintf(stderr, "std.linalg.trace mismatch\n");
        return 1;
    }

    if (lmmc_lsr_linalg_inv(&mat, &result) != LMMC_STATUS_OK ||
        !mat_close_at(&result, 0, 0, -2) ||
        !mat_close_at(&result, 0, 1, 1) ||
        !mat_close_at(&result, 1, 0, 1.5) ||
        !mat_close_at(&result, 1, 1, -0.5)) {
        fprintf(stderr, "std.linalg.inv mismatch\n");
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_rank(&rectangular, &rank) != LMMC_STATUS_OK ||
        rank != 1) {
        fprintf(stderr, "std.linalg.rank mismatch\n");
        return 1;
    }

    if (lmmc_lsr_linalg_solve_left(&mat, &rhs, &result) != LMMC_STATUS_OK ||
        !mat_close_at(&result, 0, 0, -3) ||
        !mat_close_at(&result, 0, 1, -4) ||
        !mat_close_at(&result, 1, 0, 4) ||
        !mat_close_at(&result, 1, 1, 5)) {
        fprintf(stderr, "std.linalg.solve_left mismatch\n");
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_solve_right(&rhs, &mat, &result) != LMMC_STATUS_OK ||
        !mat_close_at(&result, 0, 0, -1) ||
        !mat_close_at(&result, 0, 1, 2) ||
        !mat_close_at(&result, 1, 0, -2) ||
        !mat_close_at(&result, 1, 1, 3)) {
        fprintf(stderr, "std.linalg.solve_right mismatch\n");
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_det(&rectangular, &out) !=
        LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.linalg.det rectangular error mismatch\n");
        return 1;
    }

    if (lmmc_lsr_linalg_eig(&mat, &eig_result) != LMMC_STATUS_OK ||
        eig_result.real_parts.size != 2 ||
        eig_result.imag_parts.size != 2 ||
        !close_real(eig_result.imag_parts.data[0], 0) ||
        !close_real(eig_result.imag_parts.data[1], 0)) {
        fprintf(stderr, "std.linalg.eig mismatch\n");
        return 1;
    }
    lmmc_eigen_gen_full_result_destroy(&eig_result);

    if (lmmc_lsr_linalg_eig_table(&mat, &eig_table) != LMMC_STATUS_OK) {
        fprintf(stderr, "std.linalg.eig table failed\n");
        return 1;
    }
    named = lmmc_lsr_eig_table_get(&eig_table, "values_real");
    if (!named || named->rows != 2 || named->cols != 1 ||
        !isfinite((double)named->data[0]) ||
        !isfinite((double)named->data[named->stride]) ||
        lmmc_lsr_eig_table_get(&eig_table, "missing") != NULL) {
        fprintf(stderr, "std.linalg.eig table mapping mismatch\n");
        return 1;
    }
    named = lmmc_lsr_eig_table_get(&eig_table, "vectors_real");
    if (!named || named->rows != 2 || named->cols != 2) {
        fprintf(stderr, "std.linalg.eig table vectors mismatch\n");
        return 1;
    }
    lmmc_lsr_eig_table_destroy(&eig_table);

    if (lmmc_lsr_linalg_svd(&mat, &svd_result) != LMMC_STATUS_OK ||
        svd_result.sigma.size != 2 ||
        svd_result.U.rows != 2 ||
        svd_result.Vt.cols != 2 ||
        svd_result.sigma.data[0] < svd_result.sigma.data[1]) {
        fprintf(stderr, "std.linalg.svd mismatch\n");
        return 1;
    }
    lmmc_svd_result_destroy(&svd_result);

    if (lmmc_lsr_linalg_svd_table(&mat, &svd_table) != LMMC_STATUS_OK) {
        fprintf(stderr, "std.linalg.svd table failed\n");
        return 1;
    }
    named = lmmc_lsr_svd_table_get(&svd_table, "S");
    if (!named || named->rows != 2 || named->cols != 2 ||
        named->data[0] < named->data[named->stride + 1] ||
        lmmc_lsr_svd_table_get(&svd_table, "sigma") != NULL) {
        fprintf(stderr, "std.linalg.svd table mapping mismatch\n");
        return 1;
    }
    named = lmmc_lsr_svd_table_get(&svd_table, "U");
    if (!named || named->rows != 2 || named->cols != 2) {
        fprintf(stderr, "std.linalg.svd table U mismatch\n");
        return 1;
    }
    lmmc_lsr_svd_table_destroy(&svd_table);

    if (lmmc_lsr_linalg_eig(&rectangular, &eig_result) !=
        LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.linalg.eig rectangular error mismatch\n");
        return 1;
    }

    lmmc_mat_destroy(&rectangular);
    lmmc_mat_destroy(&rhs);
    lmmc_mat_destroy(&mat);

    return 0;
}
