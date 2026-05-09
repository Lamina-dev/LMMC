#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/linear_algebra.h"

static void lmmc_abs_to(lmmc_real_t* res, const lmmc_real_t* x) {
    lmmc_real_t zero;
    LMMC_REAL_INIT(&zero);
    LMMC_REAL_SET_D(&zero, 0.0);
    if (LMMC_REAL_CMP(x, &zero) < 0) {
        LMMC_REAL_NEG(res, x);
    } else {
        LMMC_REAL_SET(res, x);
    }
    LMMC_REAL_CLEAR(&zero);
}

static void lmmc_swap_rows(lmmc_mat_t* a, size_t r0, size_t r1) {
    size_t j = 0;
    if (r0 == r1) {
        return;
    }
    for (j = 0; j < a->cols; ++j) {
        lmmc_real_t tmp;
        LMMC_REAL_INIT(&tmp);
        LMMC_REAL_SET(&tmp, &a->data[r0 * a->stride + j]);
        LMMC_REAL_SET(&a->data[r0 * a->stride + j], &a->data[r1 * a->stride + j]);
        LMMC_REAL_SET(&a->data[r1 * a->stride + j], &tmp);
        LMMC_REAL_CLEAR(&tmp);
    }
}

lmmc_status_t lmmc_lu_decompose_inplace(lmmc_mat_t* a, size_t* pivots, size_t* out_swap_count) {
    size_t n = 0;
    size_t k = 0;
    size_t swap_count = 0;
    lmmc_real_t eps;

    if (a == NULL || pivots == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    LMMC_REAL_INIT(&eps);
    LMMC_REAL_SET_D(&eps, 1e-15);

    n = a->rows;
    for (k = 0; k < n; ++k) {
        size_t i = 0;
        size_t p = k;
        lmmc_real_t maxv;
        LMMC_REAL_INIT(&maxv);
        lmmc_abs_to(&maxv, &a->data[k * a->stride + k]);

        for (i = k + 1; i < n; ++i) {
            lmmc_real_t v;
            LMMC_REAL_INIT(&v);
            lmmc_abs_to(&v, &a->data[i * a->stride + k]);
            if (LMMC_REAL_CMP(&v, &maxv) > 0) {
                LMMC_REAL_SET(&maxv, &v);
                p = i;
            }
            LMMC_REAL_CLEAR(&v);
        }

        if (LMMC_REAL_CMP(&maxv, &eps) <= 0) {
            LMMC_REAL_CLEAR(&maxv);
            LMMC_REAL_CLEAR(&eps);
            return LMMC_STATUS_SINGULAR_MATRIX;
        }
        LMMC_REAL_CLEAR(&maxv);

        pivots[k] = p;
        if (p != k) {
            lmmc_swap_rows(a, p, k);
            ++swap_count;
        }

        {
            lmmc_real_t akk;
            LMMC_REAL_INIT(&akk);
            LMMC_REAL_SET(&akk, &a->data[k * a->stride + k]);
            for (i = k + 1; i < n; ++i) {
                size_t j = 0;
                LMMC_REAL_DIV(&a->data[i * a->stride + k], &a->data[i * a->stride + k], &akk);
                for (j = k + 1; j < n; ++j) {
                    lmmc_real_t tmp;
                    LMMC_REAL_INIT(&tmp);
                    LMMC_REAL_MUL(&tmp, &a->data[i * a->stride + k], &a->data[k * a->stride + j]);
                    LMMC_REAL_SUB(&a->data[i * a->stride + j], &a->data[i * a->stride + j], &tmp);
                    LMMC_REAL_CLEAR(&tmp);
                }
            }
            LMMC_REAL_CLEAR(&akk);
        }
    }

    if (out_swap_count != NULL) {
        *out_swap_count = swap_count;
    }
    LMMC_REAL_CLEAR(&eps);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lu_solve(const lmmc_mat_t* lu, const size_t* pivots, const lmmc_vec_t* b, lmmc_vec_t* x) {
    size_t n = 0;
    size_t i = 0;

    if (lu == NULL || pivots == NULL || b == NULL || x == NULL || lu->data == NULL || b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lu->rows != lu->cols || b->size != lu->rows || x->size != lu->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    n = lu->rows;
    for (i = 0; i < n; ++i) {
        LMMC_REAL_SET(&x->data[i], &b->data[i]);
    }

    for (i = 0; i < n; ++i) {
        size_t p = pivots[i];
        if (p != i) {
            lmmc_real_t tmp;
            LMMC_REAL_INIT(&tmp);
            LMMC_REAL_SET(&tmp, &x->data[i]);
            LMMC_REAL_SET(&x->data[i], &x->data[p]);
            LMMC_REAL_SET(&x->data[p], &tmp);
            LMMC_REAL_CLEAR(&tmp);
        }
    }

    for (i = 0; i < n; ++i) {
        size_t j = 0;
        for (j = 0; j < i; ++j) {
            lmmc_real_t tmp;
            LMMC_REAL_INIT(&tmp);
            LMMC_REAL_MUL(&tmp, &lu->data[i * lu->stride + j], &x->data[j]);
            LMMC_REAL_SUB(&x->data[i], &x->data[i], &tmp);
            LMMC_REAL_CLEAR(&tmp);
        }
    }

    lmmc_real_t eps;
    lmmc_real_t abs_lu;
    LMMC_REAL_INIT(&eps);
    LMMC_REAL_SET_D(&eps, 1e-15);
    LMMC_REAL_INIT(&abs_lu);

    for (i = n; i-- > 0;) {
        size_t j = 0;
        for (j = i + 1; j < n; ++j) {
            lmmc_real_t tmp;
            LMMC_REAL_INIT(&tmp);
            LMMC_REAL_MUL(&tmp, &lu->data[i * lu->stride + j], &x->data[j]);
            LMMC_REAL_SUB(&x->data[i], &x->data[i], &tmp);
            LMMC_REAL_CLEAR(&tmp);
        }
        
        lmmc_abs_to(&abs_lu, &lu->data[i * lu->stride + i]);
        
        if (LMMC_REAL_CMP(&abs_lu, &eps) <= 0) {
            LMMC_REAL_CLEAR(&abs_lu);
            LMMC_REAL_CLEAR(&eps);
            return LMMC_STATUS_SINGULAR_MATRIX;
        }
        
        LMMC_REAL_DIV(&x->data[i], &x->data[i], &lu->data[i * lu->stride + i]);
    }
    
    LMMC_REAL_CLEAR(&abs_lu);
    LMMC_REAL_CLEAR(&eps);

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_cholesky_decompose_inplace(lmmc_mat_t* a) {
    size_t n = 0;
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;

    if (a == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    n = a->rows;
    for (i = 0; i < n; ++i) {
        for (j = 0; j <= i; ++j) {
            lmmc_real_t sum;
            LMMC_REAL_INIT(&sum);
            LMMC_REAL_SET(&sum, &a->data[i * a->stride + j]);
            for (k = 0; k < j; ++k) {
                lmmc_real_t tmp;
                LMMC_REAL_INIT(&tmp);
                LMMC_REAL_MUL(&tmp, &a->data[i * a->stride + k], &a->data[j * a->stride + k]);
                LMMC_REAL_SUB(&sum, &sum, &tmp);
                LMMC_REAL_CLEAR(&tmp);
            }

            if (i == j) {
                lmmc_real_t zero;
                LMMC_REAL_INIT(&zero);
                LMMC_REAL_SET_D(&zero, 0.0);
                if (LMMC_REAL_CMP(&sum, &zero) <= 0) {
                    LMMC_REAL_CLEAR(&zero);
                    LMMC_REAL_CLEAR(&sum);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
                LMMC_REAL_CLEAR(&zero);
                LMMC_REAL_SQRT(&a->data[i * a->stride + i], &sum);
            } else {
                LMMC_REAL_DIV(&a->data[i * a->stride + j], &sum, &a->data[j * a->stride + j]);
            }
            LMMC_REAL_CLEAR(&sum);
        }
        for (j = i + 1; j < n; ++j) {
            lmmc_real_t zero;
            LMMC_REAL_INIT(&zero);
            LMMC_REAL_SET_D(&zero, 0.0);
            LMMC_REAL_SET(&a->data[i * a->stride + j], &zero);
            LMMC_REAL_CLEAR(&zero);
        }
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_cholesky_solve(const lmmc_mat_t* l, const lmmc_vec_t* b, lmmc_vec_t* x) {
    size_t n = 0;
    size_t i = 0;
    size_t j = 0;

    if (l == NULL || b == NULL || x == NULL || l->data == NULL || b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (l->rows != l->cols || b->size != l->rows || x->size != l->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    n = l->rows;

    for (i = 0; i < n; ++i) {
        lmmc_real_t sum;
        LMMC_REAL_INIT(&sum);
        LMMC_REAL_SET(&sum, &b->data[i]);
        for (j = 0; j < i; ++j) {
            lmmc_real_t tmp;
            LMMC_REAL_INIT(&tmp);
            LMMC_REAL_MUL(&tmp, &l->data[i * l->stride + j], &x->data[j]);
            LMMC_REAL_SUB(&sum, &sum, &tmp);
            LMMC_REAL_CLEAR(&tmp);
        }
        LMMC_REAL_DIV(&x->data[i], &sum, &l->data[i * l->stride + i]);
        LMMC_REAL_CLEAR(&sum);
    }

    for (i = n; i-- > 0;) {
        lmmc_real_t sum;
        LMMC_REAL_INIT(&sum);
        LMMC_REAL_SET(&sum, &x->data[i]);
        for (j = i + 1; j < n; ++j) {
            lmmc_real_t tmp;
            LMMC_REAL_INIT(&tmp);
            LMMC_REAL_MUL(&tmp, &l->data[j * l->stride + i], &x->data[j]);
            LMMC_REAL_SUB(&sum, &sum, &tmp);
            LMMC_REAL_CLEAR(&tmp);
        }
        LMMC_REAL_DIV(&x->data[i], &sum, &l->data[i * l->stride + i]);
        LMMC_REAL_CLEAR(&sum);
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_qr_decompose_inplace(lmmc_mat_t* a, lmmc_real_t* tau, size_t tau_size) {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;

    if (a == NULL || tau == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    m = a->rows;
    n = a->cols;
    if (m == 0 || n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (tau_size < (m < n ? m : n)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (k = 0; k < (m < n ? m : n); ++k) {
        size_t i = 0;
        size_t j = 0;
        lmmc_real_t sigma;
        lmmc_real_t alpha;
        lmmc_real_t zero;
        
        LMMC_REAL_INIT(&sigma);
        LMMC_REAL_SET_D(&sigma, 0.0);
        LMMC_REAL_INIT(&alpha);
        LMMC_REAL_SET(&alpha, &a->data[k * a->stride + k]);

        for (i = k + 1; i < m; ++i) {
            lmmc_real_t v;
            lmmc_real_t v_sq;
            LMMC_REAL_INIT(&v);
            LMMC_REAL_SET(&v, &a->data[i * a->stride + k]);
            LMMC_REAL_INIT(&v_sq);
            LMMC_REAL_MUL(&v_sq, &v, &v);
            LMMC_REAL_ADD(&sigma, &sigma, &v_sq);
            LMMC_REAL_CLEAR(&v_sq);
            LMMC_REAL_CLEAR(&v);
        }

        LMMC_REAL_INIT(&zero);
        LMMC_REAL_SET_D(&zero, 0.0);
        if (LMMC_REAL_CMP(&sigma, &zero) == 0) {
            LMMC_REAL_SET(&tau[k], &zero);
            LMMC_REAL_CLEAR(&zero);
            LMMC_REAL_CLEAR(&alpha);
            LMMC_REAL_CLEAR(&sigma);
            continue;
        }

        {
            lmmc_real_t norm;
            lmmc_real_t beta;
            lmmc_real_t v0;
            lmmc_real_t alpha_sq;
            lmmc_real_t sum;
            
            LMMC_REAL_INIT(&norm);
            LMMC_REAL_INIT(&beta);
            LMMC_REAL_INIT(&v0);
            LMMC_REAL_INIT(&alpha_sq);
            LMMC_REAL_INIT(&sum);

            LMMC_REAL_MUL(&alpha_sq, &alpha, &alpha);
            LMMC_REAL_ADD(&sum, &alpha_sq, &sigma);
            LMMC_REAL_SQRT(&norm, &sum);
            
            if (LMMC_REAL_CMP(&alpha, &zero) <= 0) {
                LMMC_REAL_SET(&beta, &norm);
            } else {
                LMMC_REAL_NEG(&beta, &norm);
            }

            LMMC_REAL_SUB(&v0, &alpha, &beta);
            LMMC_REAL_SET(&a->data[k * a->stride + k], &beta);

            for (i = k + 1; i < m; ++i) {
                LMMC_REAL_DIV(&a->data[i * a->stride + k], &a->data[i * a->stride + k], &v0);
            }

            LMMC_REAL_SUB(&tau[k], &beta, &alpha);
            LMMC_REAL_DIV(&tau[k], &tau[k], &beta);

            for (j = k + 1; j < n; ++j) {
                lmmc_real_t dot;
                LMMC_REAL_INIT(&dot);
                LMMC_REAL_SET(&dot, &a->data[k * a->stride + j]);
                for (i = k + 1; i < m; ++i) {
                    lmmc_real_t tmp;
                    LMMC_REAL_INIT(&tmp);
                    LMMC_REAL_MUL(&tmp, &a->data[i * a->stride + k], &a->data[i * a->stride + j]);
                    LMMC_REAL_ADD(&dot, &dot, &tmp);
                    LMMC_REAL_CLEAR(&tmp);
                }
                LMMC_REAL_MUL(&dot, &dot, &tau[k]);

                LMMC_REAL_SUB(&a->data[k * a->stride + j], &a->data[k * a->stride + j], &dot);
                for (i = k + 1; i < m; ++i) {
                    lmmc_real_t tmp;
                    LMMC_REAL_INIT(&tmp);
                    LMMC_REAL_MUL(&tmp, &a->data[i * a->stride + k], &dot);
                    LMMC_REAL_SUB(&a->data[i * a->stride + j], &a->data[i * a->stride + j], &tmp);
                    LMMC_REAL_CLEAR(&tmp);
                }
                LMMC_REAL_CLEAR(&dot);
            }
            
            LMMC_REAL_CLEAR(&sum);
            LMMC_REAL_CLEAR(&alpha_sq);
            LMMC_REAL_CLEAR(&v0);
            LMMC_REAL_CLEAR(&beta);
            LMMC_REAL_CLEAR(&norm);
        }
        LMMC_REAL_CLEAR(&zero);
        LMMC_REAL_CLEAR(&alpha);
        LMMC_REAL_CLEAR(&sigma);
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_qr_solve(const lmmc_mat_t* qr, const lmmc_real_t* tau, const lmmc_vec_t* b, lmmc_vec_t* x) {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    lmmc_real_t* y = NULL;

    if (qr == NULL || tau == NULL || b == NULL || x == NULL || qr->data == NULL || b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    m = qr->rows;
    n = qr->cols;
    if (m < n || b->size != m || x->size != n) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    y = (lmmc_real_t*)lmmc_alloc(m * sizeof(lmmc_real_t));
    if (y == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    
    for (k = 0; k < m; ++k) {
        LMMC_REAL_INIT(&y[k]);
        LMMC_REAL_SET(&y[k], &b->data[k]);
    }

    for (k = 0; k < n; ++k) {
        size_t i = 0;
        lmmc_real_t dot;
        LMMC_REAL_INIT(&dot);
        LMMC_REAL_SET(&dot, &y[k]);
        for (i = k + 1; i < m; ++i) {
            lmmc_real_t tmp;
            LMMC_REAL_INIT(&tmp);
            LMMC_REAL_MUL(&tmp, &qr->data[i * qr->stride + k], &y[i]);
            LMMC_REAL_ADD(&dot, &dot, &tmp);
            LMMC_REAL_CLEAR(&tmp);
        }
        LMMC_REAL_MUL(&dot, &dot, &tau[k]);
        LMMC_REAL_SUB(&y[k], &y[k], &dot);
        for (i = k + 1; i < m; ++i) {
            lmmc_real_t tmp;
            LMMC_REAL_INIT(&tmp);
            LMMC_REAL_MUL(&tmp, &qr->data[i * qr->stride + k], &dot);
            LMMC_REAL_SUB(&y[i], &y[i], &tmp);
            LMMC_REAL_CLEAR(&tmp);
        }
        LMMC_REAL_CLEAR(&dot);
    }

    lmmc_real_t eps;
    lmmc_real_t abs_qr;
    LMMC_REAL_INIT(&eps);
    LMMC_REAL_SET_D(&eps, 1e-15);
    LMMC_REAL_INIT(&abs_qr);

    for (k = n; k-- > 0;) {
        size_t j = 0;
        lmmc_real_t rhs;
        
        LMMC_REAL_INIT(&rhs);
        LMMC_REAL_SET(&rhs, &y[k]);
        for (j = k + 1; j < n; ++j) {
            lmmc_real_t tmp;
            LMMC_REAL_INIT(&tmp);
            LMMC_REAL_MUL(&tmp, &qr->data[k * qr->stride + j], &x->data[j]);
            LMMC_REAL_SUB(&rhs, &rhs, &tmp);
            LMMC_REAL_CLEAR(&tmp);
        }
        
        lmmc_abs_to(&abs_qr, &qr->data[k * qr->stride + k]);
        
        if (LMMC_REAL_CMP(&abs_qr, &eps) <= 0) {
            size_t c = 0;
            LMMC_REAL_CLEAR(&abs_qr);
            LMMC_REAL_CLEAR(&eps);
            LMMC_REAL_CLEAR(&rhs);
            for (c = 0; c < m; ++c) {
                LMMC_REAL_CLEAR(&y[c]);
            }
            lmmc_free(y);
            return LMMC_STATUS_SINGULAR_MATRIX;
        }
        
        LMMC_REAL_DIV(&x->data[k], &rhs, &qr->data[k * qr->stride + k]);
        LMMC_REAL_CLEAR(&rhs);
    }
    
    LMMC_REAL_CLEAR(&abs_qr);
    LMMC_REAL_CLEAR(&eps);

    for (k = 0; k < m; ++k) {
        LMMC_REAL_CLEAR(&y[k]);
    }
    lmmc_free(y);
    return LMMC_STATUS_OK;
}