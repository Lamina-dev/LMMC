#include <math.h>
#include <string.h>
#include <stdio.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/itersolve.h"

static int lmmc_is_finite_number(lmmc_real_t v) {
    // Note: Assuming isfinite still works or is replaced via macro in user code.
    return isfinite(v) ? 1 : 0;
}

static lmmc_status_t lmmc_vec_copy(const lmmc_vec_t* src, lmmc_vec_t* dst) {
    if (src == NULL || dst == NULL || src->data == NULL || dst->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (src->size != dst->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    size_t i = 0;
    for (i = 0; i < src->size; ++i) {
        LMMC_REAL_SET(&dst->data[i], &src->data[i]);
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_norm2_checked(const lmmc_vec_t* v, lmmc_real_t* out_norm) {
    size_t i = 0;
    lmmc_real_t sum;
    LMMC_REAL_INIT(&sum);
    LMMC_REAL_SET_D(&sum, 0.0);

    lmmc_real_t tmp_mul;
    LMMC_REAL_INIT(&tmp_mul);

    lmmc_real_t tmp_add;
    LMMC_REAL_INIT(&tmp_add);

    if (v == NULL || out_norm == NULL || v->data == NULL || v->size == 0) {
        LMMC_REAL_CLEAR(&tmp_add);
        LMMC_REAL_CLEAR(&tmp_mul);
        LMMC_REAL_CLEAR(&sum);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < v->size; ++i) {
        lmmc_real_t x;
        LMMC_REAL_INIT(&x);
        LMMC_REAL_SET(&x, &v->data[i]);
        if (!lmmc_is_finite_number(x)) {
            LMMC_REAL_CLEAR(&x);
            LMMC_REAL_CLEAR(&tmp_add);
            LMMC_REAL_CLEAR(&tmp_mul);
            LMMC_REAL_CLEAR(&sum);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        LMMC_REAL_MUL(&tmp_mul, &x, &x);
        LMMC_REAL_ADD(&tmp_add, &sum, &tmp_mul);
        LMMC_REAL_SET(&sum, &tmp_add);
        LMMC_REAL_CLEAR(&x);

        if (!lmmc_is_finite_number(sum)) {
            LMMC_REAL_CLEAR(&tmp_add);
            LMMC_REAL_CLEAR(&tmp_mul);
            LMMC_REAL_CLEAR(&sum);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    LMMC_REAL_SQRT(out_norm, &sum);

    LMMC_REAL_CLEAR(&tmp_add);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&sum);

    if (!lmmc_is_finite_number(*out_norm)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_vec_dot_checked(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t* out_dot) {
    lmmc_status_t st = lmmc_vec_dot(a, b, out_dot);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    if (!lmmc_is_finite_number(*out_dot)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_apply_precond_or_identity(
    const lmmc_precond_t* precond,
    const lmmc_vec_t* rhs,
    lmmc_vec_t* out
) {
    if (precond == NULL) {
        return lmmc_vec_copy(rhs, out);
    }
    return lmmc_precond_apply(precond, rhs, out);
}

static lmmc_status_t lmmc_gmres_back_substitute(
    const lmmc_real_t* h,
    size_t ld,
    const lmmc_real_t* rhs,
    size_t dim,
    lmmc_real_t* out_y
) {
    size_t ii = 0;
    lmmc_real_t sum;
    lmmc_real_t diag;
    lmmc_real_t tmp_mul;
    lmmc_real_t tmp_sub;
    lmmc_real_t abs_diag;
    lmmc_real_t eps_30;

    LMMC_REAL_INIT(&sum);
    LMMC_REAL_INIT(&diag);
    LMMC_REAL_INIT(&tmp_mul);
    LMMC_REAL_INIT(&tmp_sub);
    LMMC_REAL_INIT(&abs_diag);
    LMMC_REAL_INIT(&eps_30);

    LMMC_REAL_SET_D(&eps_30, 1e-30);

    if (h == NULL || rhs == NULL || out_y == NULL || ld == 0 || dim == 0) {
        LMMC_REAL_CLEAR(&eps_30);
        LMMC_REAL_CLEAR(&abs_diag);
        LMMC_REAL_CLEAR(&tmp_sub);
        LMMC_REAL_CLEAR(&tmp_mul);
        LMMC_REAL_CLEAR(&diag);
        LMMC_REAL_CLEAR(&sum);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (ii = dim; ii > 0; --ii) {
        size_t i = ii - 1;
        size_t j = 0;
        
        LMMC_REAL_SET(&sum, &rhs[i]);
        LMMC_REAL_SET_D(&diag, 0.0);

        for (j = i + 1; j < dim; ++j) {
            LMMC_REAL_MUL(&tmp_mul, &h[i * ld + j], &out_y[j]);
            LMMC_REAL_SUB(&tmp_sub, &sum, &tmp_mul);
            LMMC_REAL_SET(&sum, &tmp_sub);
        }

        LMMC_REAL_SET(&diag, &h[i * ld + i]);
        LMMC_REAL_ABS(&abs_diag, &diag);

        if (!lmmc_is_finite_number(sum) || !lmmc_is_finite_number(diag) || LMMC_REAL_CMP(&abs_diag, &eps_30) <= 0) {
            LMMC_REAL_CLEAR(&eps_30);
            LMMC_REAL_CLEAR(&abs_diag);
            LMMC_REAL_CLEAR(&tmp_sub);
            LMMC_REAL_CLEAR(&tmp_mul);
            LMMC_REAL_CLEAR(&diag);
            LMMC_REAL_CLEAR(&sum);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        LMMC_REAL_DIV(&out_y[i], &sum, &diag);
        if (!lmmc_is_finite_number(out_y[i])) {
            LMMC_REAL_CLEAR(&eps_30);
            LMMC_REAL_CLEAR(&abs_diag);
            LMMC_REAL_CLEAR(&tmp_sub);
            LMMC_REAL_CLEAR(&tmp_mul);
            LMMC_REAL_CLEAR(&diag);
            LMMC_REAL_CLEAR(&sum);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    LMMC_REAL_CLEAR(&eps_30);
    LMMC_REAL_CLEAR(&abs_diag);
    LMMC_REAL_CLEAR(&tmp_sub);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&diag);
    LMMC_REAL_CLEAR(&sum);
    return LMMC_STATUS_OK;
}

static void lmmc_itersolve_do_log(const lmmc_itersolve_config_t* cfg, size_t iter, lmmc_real_t residual_norm) {
    if (cfg->log_cb != NULL) {
        cfg->log_cb(iter, residual_norm, cfg->log_user_data);
    } else if (cfg->verbose) {
        // Warning: printf with %.10e may break if lmmc_real_t is a struct.
        // Assuming user will replace printf or handle it properly.
        printf("Iteration %zu: residual norm = %.10e\n", iter, residual_norm);
    }
}

lmmc_status_t lmmc_itersolve_default_config(size_t problem_size, lmmc_itersolve_config_t* out_cfg) {
    size_t max_iter = 0;

    if (out_cfg == NULL || problem_size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (problem_size > ((size_t)-1) / 20) {
        max_iter = 1000;
    } else {
        max_iter = problem_size * 20;
        if (max_iter < 100) {
            max_iter = 100;
        }
    }

    // Since these are struct members, we set them directly or use LMMC_REAL_SET_D if they are changed to lmmc_real_t.
    // LMMC_DEFAULT_ABS_TOL might be a macro. Let's use LMMC_REAL_SET if it's lmmc_real_t.
    LMMC_REAL_INIT(&out_cfg->abs_tol);
    LMMC_REAL_INIT(&out_cfg->rel_tol);
    LMMC_REAL_SET_D(&out_cfg->abs_tol, LMMC_DEFAULT_ABS_TOL);
    LMMC_REAL_SET_D(&out_cfg->rel_tol, 1e-8);
    
    out_cfg->max_iter = max_iter;
    out_cfg->restart = (problem_size < 30) ? problem_size : 30;
    out_cfg->verbose = 0;
    out_cfg->log_cb = NULL;
    out_cfg->log_user_data = NULL;
    return LMMC_STATUS_OK;
}

static lmmc_status_t validate_and_init_itersolve_config(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_config_t* out_cfg,
    lmmc_itersolve_result_t* out_result
) {
    lmmc_real_t zero; LMMC_REAL_INIT(&zero); LMMC_REAL_SET_D(&zero, 0.0);
    lmmc_status_t st = LMMC_STATUS_OK;

    if (a == NULL || b == NULL || x == NULL || a->row_ptr == NULL || b->data == NULL || x->data == NULL) {
        st = LMMC_STATUS_INVALID_ARGUMENT;
        goto cleanup;
    }
    if (a->rows == 0 || a->cols == 0 || b->size == 0 || x->size == 0) {
        st = LMMC_STATUS_INVALID_ARGUMENT;
        goto cleanup;
    }
    if (a->rows != a->cols || b->size != a->rows || x->size != a->cols) {
        st = LMMC_STATUS_DIMENSION_MISMATCH;
        goto cleanup;
    }

    if (cfg != NULL) {
        *out_cfg = *cfg;
    } else {
        st = lmmc_itersolve_default_config(b->size, out_cfg);
        if (st != LMMC_STATUS_OK) goto cleanup;
    }

    if (!lmmc_is_finite_number(out_cfg->abs_tol) || !lmmc_is_finite_number(out_cfg->rel_tol) ||
        LMMC_REAL_CMP(&out_cfg->abs_tol, &zero) < 0 || LMMC_REAL_CMP(&out_cfg->rel_tol, &zero) < 0 || out_cfg->max_iter == 0) {
        st = LMMC_STATUS_INVALID_ARGUMENT;
        goto cleanup;
    }

    if (precond != NULL && precond->size != b->size) {
        st = LMMC_STATUS_DIMENSION_MISMATCH;
        goto cleanup;
    }

    if (out_result != NULL) {
        out_result->converged = 0;
        out_result->num_iter = 0;
        LMMC_REAL_INIT(&out_result->initial_residual_norm);
        LMMC_REAL_INIT(&out_result->final_residual_norm);
        LMMC_REAL_SET_D(&out_result->initial_residual_norm, 0.0);
        LMMC_REAL_SET_D(&out_result->final_residual_norm, 0.0);
    }

cleanup:
    LMMC_REAL_CLEAR(&zero);
    return st;
}

static lmmc_status_t lmmc_gmres_arnoldi_step(
    const lmmc_sparse_mat_t* a, const lmmc_precond_t* precond,
    lmmc_vec_t* basis, lmmc_real_t* h, size_t j, size_t restart,
    lmmc_vec_t* ax, lmmc_vec_t* w, lmmc_real_t* out_h_next
) {
    size_t ii = 0;
    size_t i = 0;
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sub; LMMC_REAL_INIT(&tmp_sub);

    lmmc_status_t st = lmmc_sparse_mat_vec_mul(a, &basis[j], ax);
    if (st != LMMC_STATUS_OK) goto end;
    
    st = lmmc_apply_precond_or_identity(precond, ax, w);
    if (st != LMMC_STATUS_OK) goto end;

    for (ii = 0; ii <= j; ++ii) {
        lmmc_real_t hij; LMMC_REAL_INIT(&hij); LMMC_REAL_SET_D(&hij, 0.0);
        st = lmmc_vec_dot_checked(&basis[ii], w, &hij);
        if (st != LMMC_STATUS_OK) {
            LMMC_REAL_CLEAR(&hij); goto end;
        }
        LMMC_REAL_SET(&h[ii * restart + j], &hij);

        for (i = 0; i < w->size; ++i) {
            LMMC_REAL_MUL(&tmp_mul, &hij, &basis[ii].data[i]);
            LMMC_REAL_SUB(&tmp_sub, &w->data[i], &tmp_mul);
            LMMC_REAL_SET(&w->data[i], &tmp_sub);
        }
        LMMC_REAL_CLEAR(&hij);
    }

    st = lmmc_vec_norm2_checked(w, out_h_next);
    if (st == LMMC_STATUS_OK) {
        LMMC_REAL_SET(&h[(j + 1) * restart + j], out_h_next);
    }

end:
    LMMC_REAL_CLEAR(&tmp_sub);
    LMMC_REAL_CLEAR(&tmp_mul);
    return st;
}

static lmmc_status_t lmmc_gmres_apply_givens(
    lmmc_real_t* h, size_t j, size_t restart,
    lmmc_real_t* cs, lmmc_real_t* sn, lmmc_real_t* g,
    lmmc_real_t eps_30, int* out_breakdown
) {
    size_t ii = 0;
    *out_breakdown = 0;
    for (ii = 0; ii < j; ++ii) {
        lmmc_real_t h0; LMMC_REAL_INIT(&h0); LMMC_REAL_SET(&h0, &h[ii * restart + j]);
        lmmc_real_t h1; LMMC_REAL_INIT(&h1); LMMC_REAL_SET(&h1, &h[(ii + 1) * restart + j]);
        
        lmmc_real_t m1, m2; LMMC_REAL_INIT(&m1); LMMC_REAL_INIT(&m2);
        LMMC_REAL_MUL(&m1, &cs[ii], &h0);
        LMMC_REAL_MUL(&m2, &sn[ii], &h1);
        LMMC_REAL_ADD(&h[ii * restart + j], &m1, &m2);
        
        lmmc_real_t neg_sn; LMMC_REAL_INIT(&neg_sn);
        LMMC_REAL_NEG(&neg_sn, &sn[ii]);
        LMMC_REAL_MUL(&m1, &neg_sn, &h0);
        LMMC_REAL_MUL(&m2, &cs[ii], &h1);
        LMMC_REAL_ADD(&h[(ii + 1) * restart + j], &m1, &m2);
        
        LMMC_REAL_CLEAR(&neg_sn);
        LMMC_REAL_CLEAR(&m2); LMMC_REAL_CLEAR(&m1);
        LMMC_REAL_CLEAR(&h1); LMMC_REAL_CLEAR(&h0);
    }

    {
        lmmc_real_t hj; LMMC_REAL_INIT(&hj); LMMC_REAL_SET(&hj, &h[j * restart + j]);
        lmmc_real_t hsub; LMMC_REAL_INIT(&hsub); LMMC_REAL_SET(&hsub, &h[(j + 1) * restart + j]);
        
        lmmc_real_t denom_loc; LMMC_REAL_INIT(&denom_loc);
        lmmc_real_t m1, m2, a_sum;
        LMMC_REAL_INIT(&m1); LMMC_REAL_INIT(&m2); LMMC_REAL_INIT(&a_sum);
        LMMC_REAL_MUL(&m1, &hj, &hj);
        LMMC_REAL_MUL(&m2, &hsub, &hsub);
        LMMC_REAL_ADD(&a_sum, &m1, &m2);
        LMMC_REAL_SQRT(&denom_loc, &a_sum);
        
        lmmc_real_t gtmp; LMMC_REAL_INIT(&gtmp); LMMC_REAL_SET_D(&gtmp, 0.0);

        if (!lmmc_is_finite_number(denom_loc)) {
            LMMC_REAL_CLEAR(&a_sum); LMMC_REAL_CLEAR(&m2); LMMC_REAL_CLEAR(&m1);
            LMMC_REAL_CLEAR(&gtmp); LMMC_REAL_CLEAR(&denom_loc);
            LMMC_REAL_CLEAR(&hsub); LMMC_REAL_CLEAR(&hj);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (LMMC_REAL_CMP(&denom_loc, &eps_30) <= 0) {
            LMMC_REAL_CLEAR(&a_sum); LMMC_REAL_CLEAR(&m2); LMMC_REAL_CLEAR(&m1);
            LMMC_REAL_CLEAR(&gtmp); LMMC_REAL_CLEAR(&denom_loc);
            LMMC_REAL_CLEAR(&hsub); LMMC_REAL_CLEAR(&hj);
            *out_breakdown = 1;
            return LMMC_STATUS_OK;
        }

        LMMC_REAL_DIV(&cs[j], &hj, &denom_loc);
        LMMC_REAL_DIV(&sn[j], &hsub, &denom_loc);
        if (!lmmc_is_finite_number(cs[j]) || !lmmc_is_finite_number(sn[j])) {
            LMMC_REAL_CLEAR(&a_sum); LMMC_REAL_CLEAR(&m2); LMMC_REAL_CLEAR(&m1);
            LMMC_REAL_CLEAR(&gtmp); LMMC_REAL_CLEAR(&denom_loc);
            LMMC_REAL_CLEAR(&hsub); LMMC_REAL_CLEAR(&hj);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        LMMC_REAL_MUL(&m1, &cs[j], &hj);
        LMMC_REAL_MUL(&m2, &sn[j], &hsub);
        LMMC_REAL_ADD(&h[j * restart + j], &m1, &m2);
        LMMC_REAL_SET_D(&h[(j + 1) * restart + j], 0.0);

        LMMC_REAL_MUL(&gtmp, &cs[j], &g[j]);
        
        lmmc_real_t neg_sn; LMMC_REAL_INIT(&neg_sn);
        LMMC_REAL_NEG(&neg_sn, &sn[j]);
        LMMC_REAL_MUL(&g[j + 1], &neg_sn, &g[j]);
        LMMC_REAL_CLEAR(&neg_sn);
        
        LMMC_REAL_SET(&g[j], &gtmp);

        LMMC_REAL_CLEAR(&a_sum); LMMC_REAL_CLEAR(&m2); LMMC_REAL_CLEAR(&m1);
        LMMC_REAL_CLEAR(&gtmp); LMMC_REAL_CLEAR(&denom_loc);
        LMMC_REAL_CLEAR(&hsub); LMMC_REAL_CLEAR(&hj);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_cg_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
) {
    lmmc_itersolve_config_t local_cfg = {0};
    lmmc_vec_t r = {0};
    lmmc_vec_t z = {0};
    lmmc_vec_t p = {0};
    lmmc_vec_t ap = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    
    lmmc_real_t norm_b; LMMC_REAL_INIT(&norm_b); LMMC_REAL_SET_D(&norm_b, 0.0);
    lmmc_real_t norm_r; LMMC_REAL_INIT(&norm_r); LMMC_REAL_SET_D(&norm_r, 0.0);
    lmmc_real_t threshold; LMMC_REAL_INIT(&threshold); LMMC_REAL_SET_D(&threshold, 0.0);
    lmmc_real_t rho; LMMC_REAL_INIT(&rho); LMMC_REAL_SET_D(&rho, 0.0);
    lmmc_real_t denom; LMMC_REAL_INIT(&denom); LMMC_REAL_SET_D(&denom, 0.0);
    lmmc_real_t alpha; LMMC_REAL_INIT(&alpha); LMMC_REAL_SET_D(&alpha, 0.0);
    lmmc_real_t rho_new; LMMC_REAL_INIT(&rho_new); LMMC_REAL_SET_D(&rho_new, 0.0);
    lmmc_real_t beta; LMMC_REAL_INIT(&beta); LMMC_REAL_SET_D(&beta, 0.0);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul); LMMC_REAL_SET_D(&tmp_mul, 0.0);
    lmmc_real_t tmp_add; LMMC_REAL_INIT(&tmp_add); LMMC_REAL_SET_D(&tmp_add, 0.0);
    lmmc_real_t zero; LMMC_REAL_INIT(&zero); LMMC_REAL_SET_D(&zero, 0.0);
    lmmc_real_t eps_30; LMMC_REAL_INIT(&eps_30); LMMC_REAL_SET_D(&eps_30, 1e-30);
    lmmc_real_t abs_val; LMMC_REAL_INIT(&abs_val); LMMC_REAL_SET_D(&abs_val, 0.0);

    size_t iter_count = 0;
    int converged = 0;

    st = validate_and_init_itersolve_config(a, b, precond, cfg, x, &local_cfg, out_result);
    if (st != LMMC_STATUS_OK) goto cleanup;

    st = lmmc_vec_create(b->size, &r);
    if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(b->size, &z);
    if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(b->size, &p);
    if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(b->size, &ap);
    if (st != LMMC_STATUS_OK) goto cleanup;

    st = lmmc_sparse_mat_vec_mul(a, x, &ap);
    if (st != LMMC_STATUS_OK) goto cleanup;

    {
        size_t i = 0;
        for (i = 0; i < b->size; ++i) {
            LMMC_REAL_SUB(&r.data[i], &b->data[i], &ap.data[i]);
        }
    }

    st = lmmc_vec_norm2_checked(b, &norm_b);
    if (st != LMMC_STATUS_OK) goto cleanup;

    st = lmmc_vec_norm2_checked(&r, &norm_r);
    if (st != LMMC_STATUS_OK) goto cleanup;

    LMMC_REAL_MUL(&tmp_mul, &local_cfg.rel_tol, &norm_b);
    LMMC_REAL_ADD(&threshold, &local_cfg.abs_tol, &tmp_mul);
    
    if (!lmmc_is_finite_number(threshold)) {
        st = LMMC_STATUS_NUMERICAL_FAILURE;
        goto cleanup;
    }

    if (out_result != NULL) {
        LMMC_REAL_SET(&out_result->initial_residual_norm, &norm_r);
        LMMC_REAL_SET(&out_result->final_residual_norm, &norm_r);
    }

    lmmc_itersolve_do_log(&local_cfg, 0, norm_r);

    if (LMMC_REAL_CMP(&norm_r, &threshold) <= 0) {
        converged = 1;
        iter_count = 0;
        goto cleanup;
    }

    st = lmmc_apply_precond_or_identity(precond, &r, &z);
    if (st != LMMC_STATUS_OK) goto cleanup;

    st = lmmc_vec_dot_checked(&r, &z, &rho);
    if (st != LMMC_STATUS_OK) goto cleanup;
    
    LMMC_REAL_ABS(&abs_val, &rho);
    if (LMMC_REAL_CMP(&abs_val, &eps_30) <= 0) {
        st = LMMC_STATUS_NUMERICAL_FAILURE;
        goto cleanup;
    }

    st = lmmc_vec_copy(&z, &p);
    if (st != LMMC_STATUS_OK) goto cleanup;

    {
        size_t iter = 0;
        for (iter = 0; iter < local_cfg.max_iter; ++iter) {
            size_t i = 0;

            st = lmmc_sparse_mat_vec_mul(a, &p, &ap);
            if (st != LMMC_STATUS_OK) goto cleanup;

            st = lmmc_vec_dot_checked(&p, &ap, &denom);
            if (st != LMMC_STATUS_OK) goto cleanup;
            
            LMMC_REAL_ABS(&abs_val, &denom);
            if (LMMC_REAL_CMP(&abs_val, &eps_30) <= 0) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            LMMC_REAL_DIV(&alpha, &rho, &denom);
            if (!lmmc_is_finite_number(alpha)) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            for (i = 0; i < x->size; ++i) {
                LMMC_REAL_MUL(&tmp_mul, &alpha, &p.data[i]);
                LMMC_REAL_ADD(&tmp_add, &x->data[i], &tmp_mul);
                LMMC_REAL_SET(&x->data[i], &tmp_add);
                if (!lmmc_is_finite_number(x->data[i])) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }
            }

            for (i = 0; i < r.size; ++i) {
                LMMC_REAL_MUL(&tmp_mul, &alpha, &ap.data[i]);
                lmmc_real_t tmp_sub;
                LMMC_REAL_INIT(&tmp_sub);
                LMMC_REAL_SUB(&tmp_sub, &r.data[i], &tmp_mul);
                LMMC_REAL_SET(&r.data[i], &tmp_sub);
                LMMC_REAL_CLEAR(&tmp_sub);
            }

            st = lmmc_vec_norm2_checked(&r, &norm_r);
            if (st != LMMC_STATUS_OK) goto cleanup;

            iter_count = iter + 1;
            if (out_result != NULL) {
                LMMC_REAL_SET(&out_result->final_residual_norm, &norm_r);
            }

            lmmc_itersolve_do_log(&local_cfg, iter_count, norm_r);

            if (LMMC_REAL_CMP(&norm_r, &threshold) <= 0) {
                converged = 1;
                break;
            }

            st = lmmc_apply_precond_or_identity(precond, &r, &z);
            if (st != LMMC_STATUS_OK) goto cleanup;

            st = lmmc_vec_dot_checked(&r, &z, &rho_new);
            if (st != LMMC_STATUS_OK) goto cleanup;
            
            LMMC_REAL_ABS(&abs_val, &rho_new);
            if (LMMC_REAL_CMP(&abs_val, &eps_30) <= 0) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            LMMC_REAL_DIV(&beta, &rho_new, &rho);
            if (!lmmc_is_finite_number(beta)) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }

            for (i = 0; i < p.size; ++i) {
                LMMC_REAL_MUL(&tmp_mul, &beta, &p.data[i]);
                LMMC_REAL_ADD(&tmp_add, &z.data[i], &tmp_mul);
                LMMC_REAL_SET(&p.data[i], &tmp_add);
                if (!lmmc_is_finite_number(p.data[i])) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto cleanup;
                }
            }

            LMMC_REAL_SET(&rho, &rho_new);
            LMMC_REAL_ABS(&abs_val, &rho);
            if (LMMC_REAL_CMP(&abs_val, &eps_30) <= 0) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto cleanup;
            }
        }
    }

cleanup:
    if (out_result != NULL) {
        out_result->converged = converged;
        out_result->num_iter = iter_count;
        if (lmmc_is_finite_number(norm_r)) {
            LMMC_REAL_SET(&out_result->final_residual_norm, &norm_r);
        }
    }

    LMMC_REAL_CLEAR(&abs_val);
    LMMC_REAL_CLEAR(&eps_30);
    LMMC_REAL_CLEAR(&zero);
    LMMC_REAL_CLEAR(&tmp_add);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&beta);
    LMMC_REAL_CLEAR(&rho_new);
    LMMC_REAL_CLEAR(&alpha);
    LMMC_REAL_CLEAR(&denom);
    LMMC_REAL_CLEAR(&rho);
    LMMC_REAL_CLEAR(&threshold);
    LMMC_REAL_CLEAR(&norm_r);
    LMMC_REAL_CLEAR(&norm_b);

    lmmc_vec_destroy(&ap);
    lmmc_vec_destroy(&p);
    lmmc_vec_destroy(&z);
    lmmc_vec_destroy(&r);

    return st;
}

static lmmc_status_t lmmc_bicgstab_compute_alpha(
    const lmmc_sparse_mat_t* a, const lmmc_precond_t* precond,
    const lmmc_vec_t* p, lmmc_vec_t* y, lmmc_vec_t* v,
    const lmmc_vec_t* r_hat, lmmc_real_t rho_hat, lmmc_real_t eps_30,
    lmmc_real_t* out_alpha
) {
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t denom; LMMC_REAL_INIT(&denom);
    lmmc_real_t abs_val; LMMC_REAL_INIT(&abs_val);

    st = lmmc_apply_precond_or_identity(precond, p, y);
    if (st != LMMC_STATUS_OK) goto end;

    st = lmmc_sparse_mat_vec_mul(a, y, v);
    if (st != LMMC_STATUS_OK) goto end;

    st = lmmc_vec_dot_checked(r_hat, v, &denom);
    if (st != LMMC_STATUS_OK) goto end;
    
    LMMC_REAL_ABS(&abs_val, &denom);
    if (LMMC_REAL_CMP(&abs_val, &eps_30) <= 0) {
        st = LMMC_STATUS_NUMERICAL_FAILURE; goto end;
    }

    LMMC_REAL_DIV(out_alpha, &rho_hat, &denom);
    if (!lmmc_is_finite_number(*out_alpha)) {
        st = LMMC_STATUS_NUMERICAL_FAILURE; goto end;
    }

end:
    LMMC_REAL_CLEAR(&abs_val);
    LMMC_REAL_CLEAR(&denom);
    return st;
}

static lmmc_status_t lmmc_bicgstab_compute_omega(
    const lmmc_sparse_mat_t* a, const lmmc_precond_t* precond,
    const lmmc_vec_t* s, lmmc_vec_t* z, lmmc_vec_t* t,
    lmmc_real_t eps_30, lmmc_real_t* out_omega
) {
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t ts, tt; LMMC_REAL_INIT(&ts); LMMC_REAL_INIT(&tt);
    lmmc_real_t abs_val; LMMC_REAL_INIT(&abs_val);

    st = lmmc_apply_precond_or_identity(precond, s, z);
    if (st != LMMC_STATUS_OK) goto end;

    st = lmmc_sparse_mat_vec_mul(a, z, t);
    if (st != LMMC_STATUS_OK) goto end;

    st = lmmc_vec_dot_checked(t, s, &ts);
    if (st != LMMC_STATUS_OK) goto end;
    
    st = lmmc_vec_dot_checked(t, t, &tt);
    if (st != LMMC_STATUS_OK) goto end;
    
    LMMC_REAL_ABS(&abs_val, &tt);
    if (LMMC_REAL_CMP(&abs_val, &eps_30) <= 0) {
        st = LMMC_STATUS_NUMERICAL_FAILURE; goto end;
    }

    LMMC_REAL_DIV(out_omega, &ts, &tt);
    if (!lmmc_is_finite_number(*out_omega)) {
        st = LMMC_STATUS_NUMERICAL_FAILURE; goto end;
    }

end:
    LMMC_REAL_CLEAR(&abs_val);
    LMMC_REAL_CLEAR(&tt);
    LMMC_REAL_CLEAR(&ts);
    return st;
}

lmmc_status_t lmmc_bicgstab_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
) {
    lmmc_itersolve_config_t local_cfg = {0};
    lmmc_vec_t r = {0}, r_hat = {0}, p = {0}, v = {0}, s = {0}, t = {0}, y = {0}, z = {0}, ax = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    
    lmmc_real_t norm_b; LMMC_REAL_INIT(&norm_b); LMMC_REAL_SET_D(&norm_b, 0.0);
    lmmc_real_t norm_r; LMMC_REAL_INIT(&norm_r); LMMC_REAL_SET_D(&norm_r, 0.0);
    lmmc_real_t threshold; LMMC_REAL_INIT(&threshold); LMMC_REAL_SET_D(&threshold, 0.0);
    lmmc_real_t rho; LMMC_REAL_INIT(&rho); LMMC_REAL_SET_D(&rho, 1.0);
    lmmc_real_t alpha; LMMC_REAL_INIT(&alpha); LMMC_REAL_SET_D(&alpha, 1.0);
    lmmc_real_t omega; LMMC_REAL_INIT(&omega); LMMC_REAL_SET_D(&omega, 1.0);
    lmmc_real_t rho_hat; LMMC_REAL_INIT(&rho_hat); LMMC_REAL_SET_D(&rho_hat, 0.0);
    lmmc_real_t beta; LMMC_REAL_INIT(&beta); LMMC_REAL_SET_D(&beta, 0.0);
    lmmc_real_t denom; LMMC_REAL_INIT(&denom); LMMC_REAL_SET_D(&denom, 0.0);
    lmmc_real_t ts; LMMC_REAL_INIT(&ts); LMMC_REAL_SET_D(&ts, 0.0);
    lmmc_real_t tt; LMMC_REAL_INIT(&tt); LMMC_REAL_SET_D(&tt, 0.0);
    lmmc_real_t norm_s; LMMC_REAL_INIT(&norm_s); LMMC_REAL_SET_D(&norm_s, 0.0);

    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul); LMMC_REAL_SET_D(&tmp_mul, 0.0);
    lmmc_real_t tmp_add; LMMC_REAL_INIT(&tmp_add); LMMC_REAL_SET_D(&tmp_add, 0.0);
    lmmc_real_t tmp_sub; LMMC_REAL_INIT(&tmp_sub); LMMC_REAL_SET_D(&tmp_sub, 0.0);
    lmmc_real_t zero; LMMC_REAL_INIT(&zero); LMMC_REAL_SET_D(&zero, 0.0);
    lmmc_real_t eps_30; LMMC_REAL_INIT(&eps_30); LMMC_REAL_SET_D(&eps_30, 1e-30);
    lmmc_real_t abs_val; LMMC_REAL_INIT(&abs_val); LMMC_REAL_SET_D(&abs_val, 0.0);

    size_t iter_count = 0;
    int converged = 0;

    st = validate_and_init_itersolve_config(a, b, precond, cfg, x, &local_cfg, out_result);
    if (st != LMMC_STATUS_OK) goto cleanup;

    st = lmmc_vec_create(b->size, &r); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(b->size, &r_hat); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(b->size, &p); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(b->size, &v); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(b->size, &s); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(b->size, &t); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(b->size, &y); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(b->size, &z); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(b->size, &ax); if (st != LMMC_STATUS_OK) goto cleanup;

    st = lmmc_sparse_mat_vec_mul(a, x, &ax);
    if (st != LMMC_STATUS_OK) goto cleanup;

    {
        size_t i = 0;
        for (i = 0; i < b->size; ++i) {
            LMMC_REAL_SUB(&r.data[i], &b->data[i], &ax.data[i]);
            LMMC_REAL_SET_D(&p.data[i], 0.0);
            LMMC_REAL_SET_D(&v.data[i], 0.0);
        }
    }

    st = lmmc_vec_copy(&r, &r_hat);
    if (st != LMMC_STATUS_OK) goto cleanup;

    st = lmmc_vec_norm2_checked(b, &norm_b);
    if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_norm2_checked(&r, &norm_r);
    if (st != LMMC_STATUS_OK) goto cleanup;

    LMMC_REAL_MUL(&tmp_mul, &local_cfg.rel_tol, &norm_b);
    LMMC_REAL_ADD(&threshold, &local_cfg.abs_tol, &tmp_mul);

    if (!lmmc_is_finite_number(threshold)) {
        st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
    }

    if (out_result != NULL) {
        LMMC_REAL_SET(&out_result->initial_residual_norm, &norm_r);
        LMMC_REAL_SET(&out_result->final_residual_norm, &norm_r);
    }

    lmmc_itersolve_do_log(&local_cfg, 0, norm_r);

    if (LMMC_REAL_CMP(&norm_r, &threshold) <= 0) {
        converged = 1;
        iter_count = 0;
        goto cleanup;
    }

    {
        size_t iter = 0;
        for (iter = 0; iter < local_cfg.max_iter; ++iter) {
            size_t i = 0;

            st = lmmc_vec_dot_checked(&r_hat, &r, &rho_hat);
            if (st != LMMC_STATUS_OK) goto cleanup;
            
            LMMC_REAL_ABS(&abs_val, &rho_hat);
            if (LMMC_REAL_CMP(&abs_val, &eps_30) <= 0) {
                st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
            }

            if (iter == 0) {
                st = lmmc_vec_copy(&r, &p);
                if (st != LMMC_STATUS_OK) goto cleanup;
            } else {
                LMMC_REAL_ABS(&abs_val, &omega);
                if (LMMC_REAL_CMP(&abs_val, &eps_30) <= 0) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
                }
                
                lmmc_real_t tmp_div1; LMMC_REAL_INIT(&tmp_div1);
                lmmc_real_t tmp_div2; LMMC_REAL_INIT(&tmp_div2);
                LMMC_REAL_DIV(&tmp_div1, &rho_hat, &rho);
                LMMC_REAL_DIV(&tmp_div2, &alpha, &omega);
                LMMC_REAL_MUL(&beta, &tmp_div1, &tmp_div2);
                LMMC_REAL_CLEAR(&tmp_div2);
                LMMC_REAL_CLEAR(&tmp_div1);

                if (!lmmc_is_finite_number(beta)) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
                }

                for (i = 0; i < p.size; ++i) {
                    LMMC_REAL_MUL(&tmp_mul, &omega, &v.data[i]);
                    LMMC_REAL_SUB(&tmp_sub, &p.data[i], &tmp_mul);
                    LMMC_REAL_MUL(&tmp_mul, &beta, &tmp_sub);
                    LMMC_REAL_ADD(&tmp_add, &r.data[i], &tmp_mul);
                    LMMC_REAL_SET(&p.data[i], &tmp_add);

                    if (!lmmc_is_finite_number(p.data[i])) {
                        st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
                    }
                }
            }

            st = lmmc_bicgstab_compute_alpha(a, precond, &p, &y, &v, &r_hat, rho_hat, eps_30, &alpha);
            if (st != LMMC_STATUS_OK) goto cleanup;

            for (i = 0; i < s.size; ++i) {
                LMMC_REAL_MUL(&tmp_mul, &alpha, &v.data[i]);
                LMMC_REAL_SUB(&s.data[i], &r.data[i], &tmp_mul);
            }

            st = lmmc_vec_norm2_checked(&s, &norm_s);
            if (st != LMMC_STATUS_OK) goto cleanup;

            if (LMMC_REAL_CMP(&norm_s, &threshold) <= 0) {
                for (i = 0; i < x->size; ++i) {
                    LMMC_REAL_MUL(&tmp_mul, &alpha, &y.data[i]);
                    LMMC_REAL_ADD(&tmp_add, &x->data[i], &tmp_mul);
                    LMMC_REAL_SET(&x->data[i], &tmp_add);
                    if (!lmmc_is_finite_number(x->data[i])) {
                        st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
                    }
                }
                LMMC_REAL_SET(&norm_r, &norm_s);
                converged = 1;
                iter_count = iter + 1;
                if (out_result != NULL) {
                    LMMC_REAL_SET(&out_result->final_residual_norm, &norm_r);
                }
                lmmc_itersolve_do_log(&local_cfg, iter_count, norm_r);
                break;
            }

            st = lmmc_bicgstab_compute_omega(a, precond, &s, &z, &t, eps_30, &omega);
            if (st != LMMC_STATUS_OK) goto cleanup;

            for (i = 0; i < x->size; ++i) {
                LMMC_REAL_MUL(&tmp_mul, &alpha, &y.data[i]);
                lmmc_real_t tmp_mul2; LMMC_REAL_INIT(&tmp_mul2);
                LMMC_REAL_MUL(&tmp_mul2, &omega, &z.data[i]);
                
                LMMC_REAL_ADD(&tmp_add, &tmp_mul, &tmp_mul2);
                lmmc_real_t tmp_add2; LMMC_REAL_INIT(&tmp_add2);
                LMMC_REAL_ADD(&tmp_add2, &x->data[i], &tmp_add);
                LMMC_REAL_SET(&x->data[i], &tmp_add2);
                LMMC_REAL_CLEAR(&tmp_add2);
                LMMC_REAL_CLEAR(&tmp_mul2);

                if (!lmmc_is_finite_number(x->data[i])) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
                }
            }

            for (i = 0; i < r.size; ++i) {
                LMMC_REAL_MUL(&tmp_mul, &omega, &t.data[i]);
                LMMC_REAL_SUB(&r.data[i], &s.data[i], &tmp_mul);
            }

            st = lmmc_vec_norm2_checked(&r, &norm_r);
            if (st != LMMC_STATUS_OK) goto cleanup;

            iter_count = iter + 1;
            if (out_result != NULL) {
                LMMC_REAL_SET(&out_result->final_residual_norm, &norm_r);
            }

            lmmc_itersolve_do_log(&local_cfg, iter_count, norm_r);

            if (LMMC_REAL_CMP(&norm_r, &threshold) <= 0) {
                converged = 1;
                break;
            }

            LMMC_REAL_ABS(&abs_val, &omega);
            if (LMMC_REAL_CMP(&abs_val, &eps_30) <= 0) {
                st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
            }

            LMMC_REAL_SET(&rho, &rho_hat);
        }
    }

cleanup:
    if (out_result != NULL) {
        out_result->converged = converged;
        out_result->num_iter = iter_count;
        if (lmmc_is_finite_number(norm_r)) {
            LMMC_REAL_SET(&out_result->final_residual_norm, &norm_r);
        }
    }

    LMMC_REAL_CLEAR(&abs_val);
    LMMC_REAL_CLEAR(&eps_30);
    LMMC_REAL_CLEAR(&zero);
    LMMC_REAL_CLEAR(&tmp_sub);
    LMMC_REAL_CLEAR(&tmp_add);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&norm_s);
    LMMC_REAL_CLEAR(&tt);
    LMMC_REAL_CLEAR(&ts);
    LMMC_REAL_CLEAR(&denom);
    LMMC_REAL_CLEAR(&beta);
    LMMC_REAL_CLEAR(&rho_hat);
    LMMC_REAL_CLEAR(&omega);
    LMMC_REAL_CLEAR(&alpha);
    LMMC_REAL_CLEAR(&rho);
    LMMC_REAL_CLEAR(&threshold);
    LMMC_REAL_CLEAR(&norm_r);
    LMMC_REAL_CLEAR(&norm_b);

    lmmc_vec_destroy(&ax);
    lmmc_vec_destroy(&z);
    lmmc_vec_destroy(&y);
    lmmc_vec_destroy(&t);
    lmmc_vec_destroy(&s);
    lmmc_vec_destroy(&v);
    lmmc_vec_destroy(&p);
    lmmc_vec_destroy(&r_hat);
    lmmc_vec_destroy(&r);

    return st;
}

lmmc_status_t lmmc_gmres_solve(
    const lmmc_sparse_mat_t* a,
    const lmmc_vec_t* b,
    const lmmc_precond_t* precond,
    const lmmc_itersolve_config_t* cfg,
    lmmc_vec_t* x,
    lmmc_itersolve_result_t* out_result
) {
    lmmc_itersolve_config_t local_cfg = {0};
    lmmc_vec_t r = {0}, z = {0}, w = {0}, ax = {0}, x_base = {0}, x_trial = {0};
    lmmc_vec_t* basis = NULL;
    lmmc_real_t* h = NULL;
    lmmc_real_t* cs = NULL;
    lmmc_real_t* sn = NULL;
    lmmc_real_t* g = NULL;
    lmmc_real_t* y = NULL;
    size_t basis_count = 0, restart = 0, iter_count = 0, n = 0, i = 0;
    lmmc_status_t st = LMMC_STATUS_OK;
    int converged = 0;

    lmmc_real_t norm_b; LMMC_REAL_INIT(&norm_b); LMMC_REAL_SET_D(&norm_b, 0.0);
    lmmc_real_t norm_r; LMMC_REAL_INIT(&norm_r); LMMC_REAL_SET_D(&norm_r, 0.0);
    lmmc_real_t threshold; LMMC_REAL_INIT(&threshold); LMMC_REAL_SET_D(&threshold, 0.0);
    lmmc_real_t beta; LMMC_REAL_INIT(&beta); LMMC_REAL_SET_D(&beta, 0.0);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul); LMMC_REAL_SET_D(&tmp_mul, 0.0);
    lmmc_real_t tmp_add; LMMC_REAL_INIT(&tmp_add); LMMC_REAL_SET_D(&tmp_add, 0.0);
    lmmc_real_t tmp_sub; LMMC_REAL_INIT(&tmp_sub); LMMC_REAL_SET_D(&tmp_sub, 0.0);
    lmmc_real_t zero; LMMC_REAL_INIT(&zero); LMMC_REAL_SET_D(&zero, 0.0);
    lmmc_real_t eps_30; LMMC_REAL_INIT(&eps_30); LMMC_REAL_SET_D(&eps_30, 1e-30);

    st = validate_and_init_itersolve_config(a, b, precond, cfg, x, &local_cfg, out_result);
    if (st != LMMC_STATUS_OK) goto cleanup;

    n = b->size;
    restart = local_cfg.restart;
    if (restart == 0) restart = (n < 30) ? n : 30;
    if (restart > n) restart = n;
    if (restart == 0) { st = LMMC_STATUS_INVALID_ARGUMENT; goto cleanup; }

    if (out_result != NULL) {
        out_result->converged = 0;
        out_result->num_iter = 0;
        LMMC_REAL_INIT(&out_result->initial_residual_norm);
        LMMC_REAL_INIT(&out_result->final_residual_norm);
        LMMC_REAL_SET_D(&out_result->initial_residual_norm, 0.0);
        LMMC_REAL_SET_D(&out_result->final_residual_norm, 0.0);
    }

    st = lmmc_vec_create(n, &r); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(n, &z); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(n, &w); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(n, &ax); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(n, &x_base); if (st != LMMC_STATUS_OK) goto cleanup;
    st = lmmc_vec_create(n, &x_trial); if (st != LMMC_STATUS_OK) goto cleanup;

    basis = (lmmc_vec_t*)lmmc_alloc((restart + 1) * sizeof(lmmc_vec_t));
    if (basis == NULL) { st = LMMC_STATUS_ALLOCATION_FAILED; goto cleanup; }
    memset(basis, 0, (restart + 1) * sizeof(lmmc_vec_t));

    for (i = 0; i < restart + 1; ++i) {
        st = lmmc_vec_create(n, &basis[i]);
        if (st != LMMC_STATUS_OK) goto cleanup;
        ++basis_count;
    }

    h = (lmmc_real_t*)lmmc_alloc((restart + 1) * restart * sizeof(lmmc_real_t));
    cs = (lmmc_real_t*)lmmc_alloc(restart * sizeof(lmmc_real_t));
    sn = (lmmc_real_t*)lmmc_alloc(restart * sizeof(lmmc_real_t));
    g = (lmmc_real_t*)lmmc_alloc((restart + 1) * sizeof(lmmc_real_t));
    y = (lmmc_real_t*)lmmc_alloc(restart * sizeof(lmmc_real_t));

    if (h == NULL || cs == NULL || sn == NULL || g == NULL || y == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED; goto cleanup;
    }

    // Initialize all h, cs, sn, g, y elements
    for(size_t k = 0; k < (restart+1)*restart; k++) { LMMC_REAL_INIT(&h[k]); }
    for(size_t k = 0; k < restart; k++) { LMMC_REAL_INIT(&cs[k]); }
    for(size_t k = 0; k < restart; k++) { LMMC_REAL_INIT(&sn[k]); }
    for(size_t k = 0; k < restart+1; k++) { LMMC_REAL_INIT(&g[k]); }
    for(size_t k = 0; k < restart; k++) { LMMC_REAL_INIT(&y[k]); }

    st = lmmc_vec_norm2_checked(b, &norm_b);
    if (st != LMMC_STATUS_OK) goto cleanup;

    st = lmmc_sparse_mat_vec_mul(a, x, &ax);
    if (st != LMMC_STATUS_OK) goto cleanup;
    for (i = 0; i < n; ++i) {
        LMMC_REAL_SUB(&r.data[i], &b->data[i], &ax.data[i]);
    }

    st = lmmc_vec_norm2_checked(&r, &norm_r);
    if (st != LMMC_STATUS_OK) goto cleanup;

    LMMC_REAL_MUL(&tmp_mul, &local_cfg.rel_tol, &norm_b);
    LMMC_REAL_ADD(&threshold, &local_cfg.abs_tol, &tmp_mul);

    if (!lmmc_is_finite_number(threshold)) {
        st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
    }

    if (out_result != NULL) {
        LMMC_REAL_SET(&out_result->initial_residual_norm, &norm_r);
        LMMC_REAL_SET(&out_result->final_residual_norm, &norm_r);
    }

    lmmc_itersolve_do_log(&local_cfg, 0, norm_r);

    if (LMMC_REAL_CMP(&norm_r, &threshold) <= 0) {
        converged = 1;
        iter_count = 0;
        goto cleanup;
    }

    while (iter_count < local_cfg.max_iter && !converged) {
        size_t inner_limit = restart;
        size_t j = 0;
        int breakdown = 0;

        if (inner_limit > local_cfg.max_iter - iter_count) {
            inner_limit = local_cfg.max_iter - iter_count;
        }

        st = lmmc_vec_copy(x, &x_base);
        if (st != LMMC_STATUS_OK) goto cleanup;

        st = lmmc_sparse_mat_vec_mul(a, &x_base, &ax);
        if (st != LMMC_STATUS_OK) goto cleanup;
        for (i = 0; i < n; ++i) {
            LMMC_REAL_SUB(&r.data[i], &b->data[i], &ax.data[i]);
        }

        st = lmmc_apply_precond_or_identity(precond, &r, &z);
        if (st != LMMC_STATUS_OK) goto cleanup;
        
        st = lmmc_vec_norm2_checked(&z, &beta);
        if (st != LMMC_STATUS_OK) goto cleanup;

        if (LMMC_REAL_CMP(&beta, &eps_30) <= 0) {
            st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
        }

        for(size_t k = 0; k < (restart+1)*restart; k++) { LMMC_REAL_SET_D(&h[k], 0.0); }
        for(size_t k = 0; k < restart; k++) { LMMC_REAL_SET_D(&cs[k], 0.0); }
        for(size_t k = 0; k < restart; k++) { LMMC_REAL_SET_D(&sn[k], 0.0); }
        for(size_t k = 0; k < restart+1; k++) { LMMC_REAL_SET_D(&g[k], 0.0); }
        for(size_t k = 0; k < restart; k++) { LMMC_REAL_SET_D(&y[k], 0.0); }

        for (i = 0; i < n; ++i) {
            LMMC_REAL_DIV(&basis[0].data[i], &z.data[i], &beta);
            if (!lmmc_is_finite_number(basis[0].data[i])) {
                st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
            }
        }
        LMMC_REAL_SET(&g[0], &beta);

        for (j = 0; j < inner_limit; ++j) {
            lmmc_real_t h_next; LMMC_REAL_INIT(&h_next); LMMC_REAL_SET_D(&h_next, 0.0);

            st = lmmc_gmres_arnoldi_step(a, precond, basis, h, j, restart, &ax, &w, &h_next);
            if (st != LMMC_STATUS_OK) { LMMC_REAL_CLEAR(&h_next); goto cleanup; }

            st = lmmc_gmres_apply_givens(h, j, restart, cs, sn, g, eps_30, &breakdown);
            if (st != LMMC_STATUS_OK) { LMMC_REAL_CLEAR(&h_next); goto cleanup; }
            if (breakdown) {
                break;
            }

            st = lmmc_gmres_back_substitute(h, restart, g, j + 1, y);
            if (st != LMMC_STATUS_OK) { LMMC_REAL_CLEAR(&h_next); goto cleanup; }

            for (i = 0; i < n; ++i) {
                lmmc_real_t val; LMMC_REAL_INIT(&val); LMMC_REAL_SET(&val, &x_base.data[i]);
                size_t k = 0;
                for (k = 0; k < j + 1; ++k) {
                    LMMC_REAL_MUL(&tmp_mul, &y[k], &basis[k].data[i]);
                    LMMC_REAL_ADD(&tmp_add, &val, &tmp_mul);
                    LMMC_REAL_SET(&val, &tmp_add);
                }
                if (!lmmc_is_finite_number(val)) {
                    LMMC_REAL_CLEAR(&val); LMMC_REAL_CLEAR(&h_next);
                    st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
                }
                LMMC_REAL_SET(&x_trial.data[i], &val);
                LMMC_REAL_CLEAR(&val);
            }

            st = lmmc_sparse_mat_vec_mul(a, &x_trial, &ax);
            if (st != LMMC_STATUS_OK) { LMMC_REAL_CLEAR(&h_next); goto cleanup; }
            for (i = 0; i < n; ++i) {
                LMMC_REAL_SUB(&r.data[i], &b->data[i], &ax.data[i]);
            }
            st = lmmc_vec_norm2_checked(&r, &norm_r);
            if (st != LMMC_STATUS_OK) { LMMC_REAL_CLEAR(&h_next); goto cleanup; }

            iter_count += 1;
            if (out_result != NULL) {
                LMMC_REAL_SET(&out_result->final_residual_norm, &norm_r);
            }

            lmmc_itersolve_do_log(&local_cfg, iter_count, norm_r);

            st = lmmc_vec_copy(&x_trial, x);
            if (st != LMMC_STATUS_OK) { LMMC_REAL_CLEAR(&h_next); goto cleanup; }

            if (LMMC_REAL_CMP(&norm_r, &threshold) <= 0) {
                converged = 1;
                LMMC_REAL_CLEAR(&h_next);
                break;
            }

            if (LMMC_REAL_CMP(&h_next, &eps_30) <= 0) {
                breakdown = 1;
                LMMC_REAL_CLEAR(&h_next);
                break;
            }

            for (i = 0; i < n; ++i) {
                LMMC_REAL_DIV(&basis[j + 1].data[i], &w.data[i], &h_next);
                if (!lmmc_is_finite_number(basis[j + 1].data[i])) {
                    LMMC_REAL_CLEAR(&h_next);
                    st = LMMC_STATUS_NUMERICAL_FAILURE; goto cleanup;
                }
            }
            LMMC_REAL_CLEAR(&h_next);
        }

        if (converged) {
            break;
        }

        if (breakdown) {
            if (LMMC_REAL_CMP(&norm_r, &threshold) <= 0) {
                converged = 1;
                break;
            }
            st = LMMC_STATUS_NUMERICAL_FAILURE;
            goto cleanup;
        }
    }

cleanup:
    if (out_result != NULL) {
        out_result->converged = converged;
        out_result->num_iter = iter_count;
        if (lmmc_is_finite_number(norm_r)) {
            LMMC_REAL_SET(&out_result->final_residual_norm, &norm_r);
        }
    }

    LMMC_REAL_CLEAR(&eps_30);
    LMMC_REAL_CLEAR(&zero);
    LMMC_REAL_CLEAR(&tmp_sub);
    LMMC_REAL_CLEAR(&tmp_add);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&beta);
    LMMC_REAL_CLEAR(&threshold);
    LMMC_REAL_CLEAR(&norm_r);
    LMMC_REAL_CLEAR(&norm_b);

    if (basis != NULL) {
        for (i = 0; i < basis_count; ++i) {
            lmmc_vec_destroy(&basis[i]);
        }
        lmmc_free(basis);
    }

    if (h != NULL) {
        for(size_t k = 0; k < (restart+1)*restart; k++) { LMMC_REAL_CLEAR(&h[k]); }
        lmmc_free(h);
    }
    if (cs != NULL) {
        for(size_t k = 0; k < restart; k++) { LMMC_REAL_CLEAR(&cs[k]); }
        lmmc_free(cs);
    }
    if (sn != NULL) {
        for(size_t k = 0; k < restart; k++) { LMMC_REAL_CLEAR(&sn[k]); }
        lmmc_free(sn);
    }
    if (g != NULL) {
        for(size_t k = 0; k < restart+1; k++) { LMMC_REAL_CLEAR(&g[k]); }
        lmmc_free(g);
    }
    if (y != NULL) {
        for(size_t k = 0; k < restart; k++) { LMMC_REAL_CLEAR(&y[k]); }
        lmmc_free(y);
    }

    lmmc_vec_destroy(&x_trial);
    lmmc_vec_destroy(&x_base);
    lmmc_vec_destroy(&ax);
    lmmc_vec_destroy(&w);
    lmmc_vec_destroy(&z);
    lmmc_vec_destroy(&r);

    return st;
}
