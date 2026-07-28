#include "lmmc/lsr_stdlib.h"

#include <math.h>
#include <stdlib.h>

#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/linear_algebra.h"
#include "lmmc/numeric.h"
#include "lmmc/random.h"
#include "lmmc/stats.h"

static lmmc_status_t lmmc_lsr_store_real(lmmc_real_t value, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = value;
    return LMMC_STATUS_OK;
}

const char* lmmc_lsr_error_name(lmmc_status_t status)
{
    switch (status) {
    case LMMC_STATUS_OK:
        return "Ok";
    case LMMC_STATUS_INVALID_ARGUMENT:
        return "InvalidArgument";
    case LMMC_STATUS_DIMENSION_MISMATCH:
        return "DimensionMismatch";
    case LMMC_STATUS_ALLOCATION_FAILED:
        return "ResourceLimit";
    case LMMC_STATUS_SINGULAR_MATRIX:
        return "DomainError";
    case LMMC_STATUS_NOT_IMPLEMENTED:
        return "UnsupportedExpression";
    case LMMC_STATUS_NUMERICAL_FAILURE:
        return "NumericFailure";
    case LMMC_STATUS_NOT_POSITIVE_DEFINITE:
        return "DomainError";
    case LMMC_STATUS_CONVERGENCE_FAILED:
        return "NumericFailure";
    case LMMC_STATUS_OUT_OF_RANGE:
        return "DomainError";
    case LMMC_STATUS_INDEX_OUT_OF_BOUNDS:
        return "InvalidArgument";
    case LMMC_STATUS_WARNING_MAX_DEPTH:
        return "ResourceLimit";
    }
    return "InternalInvariant";
}

lmmc_status_t lmmc_lsr_math_pi(lmmc_real_t* out)
{
    return lmmc_lsr_store_real(LMMC_CONST_PI, out);
}

lmmc_status_t lmmc_lsr_math_e(lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)2.71828182845904523536, out);
}

lmmc_status_t lmmc_lsr_math_phi(lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)1.61803398874989484820, out);
}

lmmc_status_t lmmc_lsr_math_i(lmmc_complex_t* out)
{
    return lmmc_complex_create((lmmc_real_t)0, (lmmc_real_t)1, out);
}

lmmc_status_t lmmc_lsr_math_I(lmmc_complex_t* out)
{
    return lmmc_lsr_math_i(out);
}

lmmc_status_t lmmc_lsr_math_complex(lmmc_real_t real,
                                    lmmc_real_t imag,
                                    lmmc_complex_t* out)
{
    return lmmc_complex_create(real, imag, out);
}

lmmc_status_t lmmc_lsr_math_real(const lmmc_complex_t* z, lmmc_real_t* out)
{
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = z->real;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_imag(const lmmc_complex_t* z, lmmc_real_t* out)
{
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = z->imag;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_conj(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    return lmmc_complex_conj(z, out);
}

lmmc_status_t lmmc_lsr_math_complex_abs(const lmmc_complex_t* z,
                                        lmmc_real_t* out)
{
    return lmmc_complex_modulus(z, out);
}

lmmc_status_t lmmc_lsr_math_sin(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)sin((double)x), out);
}

lmmc_status_t lmmc_lsr_math_cos(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)cos((double)x), out);
}

lmmc_status_t lmmc_lsr_math_tan(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)tan((double)x), out);
}

lmmc_status_t lmmc_lsr_math_asin(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < (lmmc_real_t)-1 || x > (lmmc_real_t)1) return LMMC_STATUS_OUT_OF_RANGE;
    *out = (lmmc_real_t)asin((double)x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_acos(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < (lmmc_real_t)-1 || x > (lmmc_real_t)1) return LMMC_STATUS_OUT_OF_RANGE;
    *out = (lmmc_real_t)acos((double)x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_atan(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)atan((double)x), out);
}

lmmc_status_t lmmc_lsr_math_sqrt(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < (lmmc_real_t)0) return LMMC_STATUS_OUT_OF_RANGE;
    *out = (lmmc_real_t)sqrt((double)x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_exp(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)exp((double)x), out);
}

lmmc_status_t lmmc_lsr_math_ln(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= (lmmc_real_t)0) return LMMC_STATUS_OUT_OF_RANGE;
    *out = (lmmc_real_t)log((double)x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_log(lmmc_real_t x, lmmc_real_t base,
                                lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= (lmmc_real_t)0 || base <= (lmmc_real_t)0 ||
        base == (lmmc_real_t)1) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    *out = (lmmc_real_t)(log((double)x) / log((double)base));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_abs(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)fabs((double)x), out);
}

lmmc_status_t lmmc_lsr_math_floor(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)floor((double)x), out);
}

lmmc_status_t lmmc_lsr_math_ceil(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)ceil((double)x), out);
}

lmmc_status_t lmmc_lsr_math_round(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)round((double)x), out);
}

lmmc_status_t lmmc_lsr_math_clamp(lmmc_real_t x, lmmc_real_t lo,
                                  lmmc_real_t hi, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lo > hi) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < lo) *out = lo;
    else if (x > hi) *out = hi;
    else *out = x;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_wrap_const_vec(const lmmc_real_t* values,
                                             size_t count,
                                             lmmc_vec_t* out)
{
    if (!values || !out || count == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    out->size = count;
    out->data = (lmmc_real_t*)values;
    out->owns_data = 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_stats_mean(const lmmc_real_t* values, size_t count,
                                  lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_mean(&view, out);
}

lmmc_status_t lmmc_lsr_stats_median(const lmmc_real_t* values, size_t count,
                                    lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_median(&view, out);
}

lmmc_status_t lmmc_lsr_stats_var(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_variance_sample(&view, out);
}

lmmc_status_t lmmc_lsr_stats_std(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_stddev_sample(&view, out);
}

lmmc_status_t lmmc_lsr_stats_quantile(const lmmc_real_t* values, size_t count,
                                      lmmc_real_t q, lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_quantile(&view, q, out);
}

lmmc_status_t lmmc_lsr_random_seed(lmmc_rng_t* rng, uint64_t seed)
{
    return lmmc_rng_seed(rng, seed);
}

lmmc_status_t lmmc_lsr_random_rand(lmmc_rng_t* rng, lmmc_real_t* out)
{
    return lmmc_rng_uniform(rng, (lmmc_real_t)0, (lmmc_real_t)1, out);
}

lmmc_status_t lmmc_lsr_random_randint(lmmc_rng_t* rng, int64_t lo,
                                      int64_t hi, int64_t* out)
{
    return lmmc_rng_int_uniform(rng, lo, hi, out);
}

lmmc_status_t lmmc_lsr_random_normal(lmmc_rng_t* rng, lmmc_real_t mean,
                                     lmmc_real_t stddev, lmmc_real_t* out)
{
    return lmmc_rng_normal(rng, mean, stddev, out);
}

lmmc_status_t lmmc_lsr_random_choice(lmmc_rng_t* rng,
                                     const lmmc_real_t* values,
                                     size_t count,
                                     lmmc_real_t* out)
{
    int64_t index = 0;
    lmmc_status_t status;
    if (!values || !out || count == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (count > (size_t)INT64_MAX + 1u) return LMMC_STATUS_OUT_OF_RANGE;
    status = lmmc_rng_int_uniform(rng, 0, (int64_t)count - 1, &index);
    if (status != LMMC_STATUS_OK) return status;
    *out = values[index];
    return LMMC_STATUS_OK;
}

static int lmmc_lsr_mat_valid(const lmmc_mat_t* a)
{
    return a && a->data && a->rows > 0 && a->cols > 0 && a->stride >= a->cols;
}

lmmc_status_t lmmc_lsr_linalg_shape(const lmmc_mat_t* a,
                                    size_t* out_rows,
                                    size_t* out_cols)
{
    if (!lmmc_lsr_mat_valid(a) || !out_rows || !out_cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out_rows = a->rows;
    *out_cols = a->cols;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_transpose(const lmmc_mat_t* a,
                                        lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_mat_create(a->cols, a->rows, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_transpose_to(a, out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_adjoint(const lmmc_mat_t* a,
                                      lmmc_mat_t* out)
{
    return lmmc_lsr_linalg_transpose(a, out);
}

lmmc_status_t lmmc_lsr_linalg_det(const lmmc_mat_t* a,
                                  lmmc_real_t* out)
{
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_mat_det(a, out);
}

lmmc_status_t lmmc_lsr_linalg_inv(const lmmc_mat_t* a,
                                  lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_inv(a, out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_rank(const lmmc_mat_t* a,
                                   size_t* out_rank)
{
    size_t rows, cols, count, rank, col;
    lmmc_real_t* work;
    lmmc_real_t scale = (lmmc_real_t)0;
    lmmc_real_t tol;

    if (!lmmc_lsr_mat_valid(a) || !out_rank) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != 0 && a->cols > ((size_t)-1) / a->rows) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    rows = a->rows;
    cols = a->cols;
    count = rows * cols;
    work = (lmmc_real_t*)malloc(count * sizeof(lmmc_real_t));
    if (!work) return LMMC_STATUS_ALLOCATION_FAILED;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            lmmc_real_t value = a->data[i * a->stride + j];
            work[i * cols + j] = value;
            if (fabs((double)value) > fabs((double)scale)) scale = value;
        }
    }

    if (scale == (lmmc_real_t)0) {
        free(work);
        *out_rank = 0;
        return LMMC_STATUS_OK;
    }
    tol = (lmmc_real_t)(1e-12 * fabs((double)scale) *
                        (double)(rows > cols ? rows : cols));

    rank = 0;
    for (col = 0; col < cols && rank < rows; ++col) {
        size_t pivot = rank;
        lmmc_real_t pivot_abs =
            (lmmc_real_t)fabs((double)work[pivot * cols + col]);
        for (size_t row = rank + 1; row < rows; ++row) {
            lmmc_real_t candidate =
                (lmmc_real_t)fabs((double)work[row * cols + col]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= tol) continue;

        if (pivot != rank) {
            for (size_t j = col; j < cols; ++j) {
                lmmc_real_t tmp = work[rank * cols + j];
                work[rank * cols + j] = work[pivot * cols + j];
                work[pivot * cols + j] = tmp;
            }
        }

        for (size_t row = rank + 1; row < rows; ++row) {
            lmmc_real_t factor = work[row * cols + col] /
                                 work[rank * cols + col];
            work[row * cols + col] = (lmmc_real_t)0;
            for (size_t j = col + 1; j < cols; ++j) {
                work[row * cols + j] -= factor * work[rank * cols + j];
            }
        }
        ++rank;
    }

    free(work);
    *out_rank = rank;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_trace(const lmmc_mat_t* a,
                                    lmmc_real_t* out)
{
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_mat_trace(a, out);
}

lmmc_status_t lmmc_lsr_linalg_solve_left(const lmmc_mat_t* a,
                                         const lmmc_mat_t* b,
                                         lmmc_mat_t* out)
{
    lmmc_mat_t lu = {0};
    lmmc_vec_t rhs = {0};
    lmmc_vec_t sol = {0};
    size_t* pivots = NULL;
    lmmc_status_t status;

    if (!lmmc_lsr_mat_valid(a) || !lmmc_lsr_mat_valid(b) || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols || b->rows != a->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    status = lmmc_mat_create(a->rows, a->cols, &lu);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_copy(a, &lu);
    if (status != LMMC_STATUS_OK) goto cleanup;

    pivots = (size_t*)malloc(a->rows * sizeof(size_t));
    if (!pivots) {
        status = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    status = lmmc_lu_decompose_inplace(&lu, pivots, NULL);
    if (status != LMMC_STATUS_OK) goto cleanup;

    status = lmmc_mat_create(a->cols, b->cols, out);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_vec_create(b->rows, &rhs);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(out);
        goto cleanup;
    }
    status = lmmc_vec_create(a->cols, &sol);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(out);
        goto cleanup;
    }

    for (size_t col = 0; col < b->cols; ++col) {
        for (size_t row = 0; row < b->rows; ++row) {
            rhs.data[row] = b->data[row * b->stride + col];
        }
        status = lmmc_lu_solve(&lu, pivots, &rhs, &sol);
        if (status != LMMC_STATUS_OK) {
            lmmc_mat_destroy(out);
            goto cleanup;
        }
        for (size_t row = 0; row < a->cols; ++row) {
            out->data[row * out->stride + col] = sol.data[row];
        }
    }

cleanup:
    lmmc_vec_destroy(&sol);
    lmmc_vec_destroy(&rhs);
    if (pivots) free(pivots);
    lmmc_mat_destroy(&lu);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_solve_right(const lmmc_mat_t* b,
                                          const lmmc_mat_t* a,
                                          lmmc_mat_t* out)
{
    lmmc_mat_t at = {0};
    lmmc_mat_t bt = {0};
    lmmc_mat_t xt = {0};
    lmmc_status_t status;

    if (!lmmc_lsr_mat_valid(a) || !lmmc_lsr_mat_valid(b) || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols || b->cols != a->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    status = lmmc_lsr_linalg_transpose(a, &at);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_linalg_transpose(b, &bt);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_linalg_solve_left(&at, &bt, &xt);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_linalg_transpose(&xt, out);

cleanup:
    lmmc_mat_destroy(&xt);
    lmmc_mat_destroy(&bt);
    lmmc_mat_destroy(&at);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_eig(const lmmc_mat_t* a,
                                  lmmc_eigen_gen_full_result_t* out)
{
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_eigen_general_full(a, out);
}

lmmc_status_t lmmc_lsr_linalg_svd(const lmmc_mat_t* a,
                                  lmmc_svd_result_t* out)
{
    if (!lmmc_lsr_mat_valid(a) || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    return lmmc_svd(a, out);
}
