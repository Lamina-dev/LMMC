/**
 * @file dense_solvers.c
 * @brief 稠密矩阵求解器实现（行列式、逆、三角求解、右除、秩、幂）。
 */

#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/linear_algebra.h"
#include "lmmc/eigen.h"

static size_t lmmc_det_pivot(const lmmc_real_t* lu, size_t n, size_t k) {
    size_t pivot_row = k;
    lmmc_real_t abs_val;
    lmmc_real_t max_val;
    lmmc_real_t zero = 0.0;
    LMMC_REAL_ABS(&max_val, &lu[k * n + k]);
    for (size_t i = k + 1; i < n; ++i) {
        LMMC_REAL_ABS(&abs_val, &lu[i * n + k]);
        if (LMMC_REAL_CMP(&abs_val, &max_val) > 0) {
            LMMC_REAL_SET(&max_val, &abs_val);
            pivot_row = i;
        }
    }
    return LMMC_REAL_CMP(&max_val, &zero) == 0 ? n : pivot_row;
}

static void lmmc_det_eliminate(lmmc_real_t* lu, size_t n, size_t k) {
    lmmc_real_t factor;
    lmmc_real_t tmp_mul;
    lmmc_real_t tmp_sub;
        for (size_t i = k + 1; i < n; ++i) {
            LMMC_REAL_DIV(&factor, &lu[i * n + k], &lu[k * n + k]);
            LMMC_REAL_SET(&lu[i * n + k], &factor);

            for (size_t j = k + 1; j < n; ++j) {
                LMMC_REAL_MUL(&tmp_mul, &factor, &lu[k * n + j]);
                LMMC_REAL_SUB(&tmp_sub, &lu[i * n + j], &tmp_mul);
                LMMC_REAL_SET(&lu[i * n + j], &tmp_sub);
            }
        }
}

static void lmmc_det_lu(lmmc_real_t* lu, size_t n, lmmc_real_t* out_det) {
    int sign = 1;
    for (size_t k = 0; k < n; ++k) {
        size_t pivot_row = lmmc_det_pivot(lu, n, k);
        if (pivot_row == n) {
            LMMC_REAL_SET_D(out_det, 0.0);
            return;
        }
        if (pivot_row != k) {
            for (size_t j = 0; j < n; ++j) {
                lmmc_real_t tmp = lu[k * n + j];
                lu[k * n + j] = lu[pivot_row * n + j];
                lu[pivot_row * n + j] = tmp;
            }
            sign = -sign;
        }
        lmmc_det_eliminate(lu, n, k);
    }
    lmmc_real_t det; LMMC_REAL_INIT(&det);
    LMMC_REAL_SET_D(&det, (double)sign);

    lmmc_real_t tmp_prod; LMMC_REAL_INIT(&tmp_prod);
    for (size_t i = 0; i < n; ++i) {
        LMMC_REAL_MUL(&tmp_prod, &det, &lu[i * n + i]);
        LMMC_REAL_SET(&det, &tmp_prod);
    }
    LMMC_REAL_SET(out_det, &det);

    LMMC_REAL_CLEAR(&det);
    LMMC_REAL_CLEAR(&tmp_prod);
}

static void lmmc_det_two(const lmmc_mat_t* a, lmmc_real_t* out_det) {
        lmmc_real_t ad; LMMC_REAL_INIT(&ad);
        lmmc_real_t bc; LMMC_REAL_INIT(&bc);
        lmmc_real_t result; LMMC_REAL_INIT(&result);

        LMMC_REAL_MUL(&ad, &a->data[0 * a->stride + 0], &a->data[1 * a->stride + 1]);
        LMMC_REAL_MUL(&bc, &a->data[0 * a->stride + 1], &a->data[1 * a->stride + 0]);
        LMMC_REAL_SUB(&result, &ad, &bc);
        LMMC_REAL_SET(out_det, &result);

        LMMC_REAL_CLEAR(&ad);
        LMMC_REAL_CLEAR(&bc);
        LMMC_REAL_CLEAR(&result);
}

lmmc_status_t lmmc_mat_det(const lmmc_mat_t* a, lmmc_real_t* out_det) {
    if (!lmmc_mat_descriptor_is_valid(a) || out_det == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    size_t n = a->rows;


    if (n == 1) {
        LMMC_REAL_SET(out_det, &a->data[0]);
        return LMMC_STATUS_OK;
    }
    if (n == 2) {
        lmmc_det_two(a, out_det);
        return LMMC_STATUS_OK;
    }
    size_t n_elem;
    size_t n_bytes;
    if (!lmmc_safe_mul_size(n, n, &n_elem) ||
        !lmmc_safe_mul_size(n_elem, sizeof(lmmc_real_t), &n_bytes)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    lmmc_real_t* lu = (lmmc_real_t*)lmmc_memory_alloc(n_bytes);
    if (lu == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }


    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            lu[i * n + j] = a->data[i * a->stride + j];
        }
    }
    lmmc_det_lu(lu, n, out_det);
    lmmc_memory_free(lu);
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_inverse_columns(const lmmc_mat_t* lu,
    const size_t* pivots, lmmc_mat_t* inverse) {
    size_t n = lu->rows;
    lmmc_vec_t ei;
    lmmc_vec_t col;
    lmmc_status_t status = lmmc_vec_create(n, &ei);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_vec_create(n, &col);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&ei);
        return status;
    }
    /** @brief 每次基向量求解成功后写入逆矩阵的一整列。 */
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            LMMC_REAL_SET_D(&ei.data[j], 0.0);
        }
        LMMC_REAL_SET_D(&ei.data[i], 1.0);
        status = lmmc_lu_solve(lu, pivots, &ei, &col);
        if (status != LMMC_STATUS_OK) {
            break;
        }
        for (size_t j = 0; j < n; ++j) {
            LMMC_REAL_SET(&inverse->data[j * inverse->stride + i], &col.data[j]);
        }
    }
    lmmc_vec_destroy(&col);
    lmmc_vec_destroy(&ei);
    return status;
}

lmmc_status_t lmmc_mat_inv(const lmmc_mat_t* A, lmmc_mat_t* A_inv) {
    lmmc_mat_t lu_mat;
    size_t* pivots = NULL;
    size_t n;
    lmmc_status_t status;
    if (!lmmc_mat_descriptor_is_valid(A) ||
        !lmmc_mat_descriptor_is_valid(A_inv)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (A->rows != A->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (A_inv->rows != A->rows || A_inv->cols != A->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    n = A->rows;

    /* Copy A into workspace for LU decomposition */
    status = lmmc_mat_create(n, n, &lu_mat);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_mat_copy(A, &lu_mat);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&lu_mat);
        return status;
    }

    /* Allocate pivot array */
    pivots = (size_t*)lmmc_memory_alloc_array(n, sizeof(size_t));
    if (pivots == NULL) {
        lmmc_mat_destroy(&lu_mat);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* LU decompose with partial pivoting */
    status = lmmc_lu_decompose_inplace(&lu_mat, pivots, NULL);
    if (status != LMMC_STATUS_OK) {
        /* Singular or other failure — leave A_inv unmodified */
        lmmc_memory_free(pivots);
        lmmc_mat_destroy(&lu_mat);
        return status;
    }
    status = lmmc_inverse_columns(&lu_mat, pivots, A_inv);
    lmmc_memory_free(pivots);
    lmmc_mat_destroy(&lu_mat);
    return status;
}

static lmmc_status_t lmmc_triangular_row(const lmmc_mat_t* T,
    int upper, int diag_unit, const lmmc_vec_t* b, lmmc_vec_t* x, size_t i) {
    lmmc_real_t sum;
    lmmc_real_t tmp;
    lmmc_real_t diag_val;
    lmmc_real_t abs_diag;
    lmmc_real_t eps = 1e-15;
    size_t begin = upper ? i + 1 : 0;
    size_t end = upper ? T->rows : i;
    LMMC_REAL_SET(&sum, &b->data[i]);
    for (size_t j = begin; j < end; ++j) {
        LMMC_REAL_MUL(&tmp, &T->data[i * T->stride + j], &x->data[j]);
        LMMC_REAL_SUB(&sum, &sum, &tmp);
    }
    if (diag_unit) {
        LMMC_REAL_SET(&x->data[i], &sum);
        return LMMC_STATUS_OK;
    }
    LMMC_REAL_SET(&diag_val, &T->data[i * T->stride + i]);
    LMMC_REAL_ABS(&abs_diag, &diag_val);
    if (LMMC_REAL_CMP(&abs_diag, &eps) <= 0) {
        return LMMC_STATUS_SINGULAR_MATRIX;
    }
    LMMC_REAL_DIV(&x->data[i], &sum, &diag_val);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_solve_triangular(const lmmc_mat_t* T, int upper, int diag_unit,
    const lmmc_vec_t* b, lmmc_vec_t* x) {

    if (!lmmc_mat_descriptor_is_valid(T) ||
        !lmmc_vec_descriptor_is_valid(b) ||
        !lmmc_vec_descriptor_is_valid(x)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (T->rows != T->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (b->size != T->rows || x->size != T->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    for (size_t step = 0; step < T->rows; ++step) {
        size_t i = upper ? T->rows - 1 - step : step;
        lmmc_status_t status = lmmc_triangular_row(T, upper, diag_unit, b, x, i);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_rank(const lmmc_mat_t* a, lmmc_real_t tol, size_t* out_rank)
{
    lmmc_svd_result_t svd;
    lmmc_status_t status;
    size_t i, count;
    size_t max_dim;
    lmmc_real_t effective_tol;
    lmmc_real_t eps_val;
    lmmc_real_t dim_val;
    lmmc_real_t tmp;

    if (!lmmc_mat_descriptor_is_valid(a) || out_rank == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Compute SVD */
    status = lmmc_svd(a, &svd);
    if (status != LMMC_STATUS_OK) {
        return status;
    }

    /* Determine effective tolerance */
    LMMC_REAL_INIT(&effective_tol);
    if (tol <= 0.0) {
        /* Default: max(rows, cols) * LMMC_REAL_EPSILON * sigma[0] */
        max_dim = (a->rows > a->cols) ? a->rows : a->cols;

        LMMC_REAL_INIT(&eps_val);
        LMMC_REAL_SET_D(&eps_val, LMMC_REAL_EPSILON);

        LMMC_REAL_INIT(&dim_val);
        LMMC_REAL_SET_D(&dim_val, (double)max_dim);

        LMMC_REAL_INIT(&tmp);
        LMMC_REAL_MUL(&tmp, &dim_val, &eps_val);
        LMMC_REAL_MUL(&effective_tol, &tmp, &svd.sigma.data[0]);

        LMMC_REAL_CLEAR(&eps_val);
        LMMC_REAL_CLEAR(&dim_val);
        LMMC_REAL_CLEAR(&tmp);
    } else {
        LMMC_REAL_SET_D(&effective_tol, tol);
    }

    /* Count singular values > tol */
    count = 0;
    for (i = 0; i < svd.sigma.size; ++i) {
        if (LMMC_REAL_CMP(&svd.sigma.data[i], &effective_tol) > 0) {
            count++;
        }
    }

    *out_rank = count;

    LMMC_REAL_CLEAR(&effective_tol);
    lmmc_svd_result_destroy(&svd);
    return LMMC_STATUS_OK;
}


static lmmc_status_t lmmc_power_workspace(const lmmc_mat_t* a, int n,
    lmmc_mat_t* base, lmmc_mat_t* temp) {
    lmmc_mat_t inverse = {0};
    const lmmc_mat_t* source = a;
    lmmc_status_t status;
    if (n < 0) {
        status = lmmc_mat_create(a->rows, a->cols, &inverse);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
        status = lmmc_mat_inv(a, &inverse);
        if (status != LMMC_STATUS_OK) {
            lmmc_mat_destroy(&inverse);
            return status;
        }
        source = &inverse;
    }
    status = lmmc_mat_create(a->rows, a->cols, base);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_mat_create(a->rows, a->cols, temp);
        if (status != LMMC_STATUS_OK) {
            lmmc_mat_destroy(base);
        }
    }
    if (status == LMMC_STATUS_OK) {
        status = lmmc_mat_copy(source, base);
        if (status != LMMC_STATUS_OK) {
            lmmc_mat_destroy(temp);
            lmmc_mat_destroy(base);
        }
    }
    if (n < 0) {
        lmmc_mat_destroy(&inverse);
    }
    return status;
}

static lmmc_status_t lmmc_power_multiply(const lmmc_mat_t* left,
    const lmmc_mat_t* right, lmmc_mat_t* temp, lmmc_mat_t* out) {
    lmmc_status_t status = lmmc_mat_mul(left, right, temp);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    return lmmc_mat_copy(temp, out);
}

static lmmc_status_t lmmc_power_accumulate(lmmc_mat_t* base,
    lmmc_mat_t* temp, unsigned int exp, lmmc_mat_t* out) {
    lmmc_fill_identity_unchecked(out);
    while (exp > 0) {
        if (exp & 1u) {
            lmmc_status_t status = lmmc_power_multiply(out, base, temp, out);
            if (status != LMMC_STATUS_OK) {
                return status;
            }
        }
        exp >>= 1u;
        if (exp > 0) {
            lmmc_status_t status = lmmc_power_multiply(base, base, temp, base);
            if (status != LMMC_STATUS_OK) {
                return status;
            }
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_pow(const lmmc_mat_t* a, int n, lmmc_mat_t* out) {
    lmmc_mat_t base;
    lmmc_mat_t temp;
    size_t dim;
    size_t total;
    if (!lmmc_mat_descriptor_is_valid(a) ||
        !lmmc_mat_descriptor_is_valid(out)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    dim = a->rows;
    if (out->rows != dim || out->cols != dim) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n == 0) {
        lmmc_fill_identity_unchecked(out);
        return LMMC_STATUS_OK;
    }
    if (n == 1) {
        return lmmc_mat_copy(a, out);
    }
    if (!lmmc_safe_mul_size(dim, dim, &total)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    lmmc_status_t status = lmmc_power_workspace(a, n, &base, &temp);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    /** @brief 无符号幅值可表示 INT_MIN 的绝对值。 */
    unsigned int exp = n < 0 ? (unsigned int)(-(n + 1)) + 1u : (unsigned int)n;
    status = lmmc_power_accumulate(&base, &temp, exp, out);
    lmmc_mat_destroy(&temp);
    lmmc_mat_destroy(&base);
    return status;
}

static lmmc_status_t lmmc_rdiv_rows(const lmmc_mat_t* lu,
    const size_t* pivots, const lmmc_mat_t* B, lmmc_mat_t* X) {
    lmmc_vec_t rhs_vec;
    lmmc_vec_t sol_vec;
    size_t nn = lu->rows;
    lmmc_status_t status = lmmc_vec_create(nn, &rhs_vec);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_vec_create(nn, &sol_vec);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&rhs_vec);
        return status;
    }
    /** @brief 求解 A^T x = B[i,:]^T 成功后写入 X 的第 i 行。 */
    for (size_t i = 0; i < B->rows; ++i) {
        for (size_t j = 0; j < nn; ++j) {
            LMMC_REAL_SET(&rhs_vec.data[j], &B->data[i * B->stride + j]);
        }
        status = lmmc_lu_solve(lu, pivots, &rhs_vec, &sol_vec);
        if (status != LMMC_STATUS_OK) {
            break;
        }
        for (size_t j = 0; j < nn; ++j) {
            LMMC_REAL_SET(&X->data[i * X->stride + j], &sol_vec.data[j]);
        }
    }
    lmmc_vec_destroy(&sol_vec);
    lmmc_vec_destroy(&rhs_vec);
    return status;
}

lmmc_status_t lmmc_mat_rdiv(const lmmc_mat_t* B, const lmmc_mat_t* A, lmmc_mat_t* X)
{
    lmmc_mat_t at_mat;
    size_t* pivots = NULL;
    lmmc_status_t status;
    size_t nn, m;
    size_t pivot_bytes;
    if (!lmmc_mat_descriptor_is_valid(B) ||
        !lmmc_mat_descriptor_is_valid(A) ||
        !lmmc_mat_descriptor_is_valid(X)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    /* A must be square */
    if (A->rows != A->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    nn = A->rows;
    m = B->rows;

    /* B must have nn columns */
    if (B->cols != nn) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    /* X must be m × nn */
    if (X->rows != m || X->cols != nn) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Compute A^T into at_mat */
    status = lmmc_mat_create(nn, nn, &at_mat);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_mat_transpose_to(A, &at_mat);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&at_mat);
        return status;
    }

    /* Allocate pivot array */
    if (!lmmc_safe_mul_size(nn, sizeof(size_t), &pivot_bytes)) {
        lmmc_mat_destroy(&at_mat);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    pivots = (size_t*)lmmc_memory_alloc(pivot_bytes);
    if (pivots == NULL) {
        lmmc_mat_destroy(&at_mat);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* LU decompose A^T in-place */
    status = lmmc_lu_decompose_inplace(&at_mat, pivots, NULL);
    if (status != LMMC_STATUS_OK) {
        lmmc_memory_free(pivots);
        lmmc_mat_destroy(&at_mat);
        return status;
    }
    status = lmmc_rdiv_rows(&at_mat, pivots, B, X);
    lmmc_memory_free(pivots);
    lmmc_mat_destroy(&at_mat);
    return status;
}
