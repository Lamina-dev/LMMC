/**
 * @file complex.c
 * @brief 复数类型构造,算术运算与复数向量/矩阵生命周期管理.
 */
#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/complex.h"

lmmc_status_t lmmc_complex_create(lmmc_real_t real, lmmc_real_t imag, lmmc_complex_t* out)
{
    if (out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_SET(&out->real, &real);
    LMMC_REAL_SET(&out->imag, &imag);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_from_polar(lmmc_real_t r, lmmc_real_t theta, lmmc_complex_t* out)
{
    lmmc_real_t zero;
    lmmc_real_t cos_theta, sin_theta;

    if (out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_SET_D(&zero, 0.0);
    if (LMMC_REAL_CMP(&r, &zero) == 0) {
        LMMC_REAL_SET_D(&out->real, 0.0);
        LMMC_REAL_SET_D(&out->imag, 0.0);
        return LMMC_STATUS_OK;
    }

    LMMC_REAL_COS(&cos_theta, &theta);
    LMMC_REAL_SIN(&sin_theta, &theta);
    LMMC_REAL_MUL(&out->real, &r, &cos_theta);
    LMMC_REAL_MUL(&out->imag, &r, &sin_theta);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_add(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_ADD(&out->real, &a->real, &b->real);
    LMMC_REAL_ADD(&out->imag, &a->imag, &b->imag);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_sub(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_SUB(&out->real, &a->real, &b->real);
    LMMC_REAL_SUB(&out->imag, &a->imag, &b->imag);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_mul(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out)
{
    lmmc_real_t ac, bd, ad, bc;

    if (a == NULL || b == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* (a.real * b.real - a.imag * b.imag) + i*(a.real * b.imag + a.imag * b.real) */
    LMMC_REAL_MUL(&ac, &a->real, &b->real);
    LMMC_REAL_MUL(&bd, &a->imag, &b->imag);
    LMMC_REAL_MUL(&ad, &a->real, &b->imag);
    LMMC_REAL_MUL(&bc, &a->imag, &b->real);

    LMMC_REAL_SUB(&out->real, &ac, &bd);
    LMMC_REAL_ADD(&out->imag, &ad, &bc);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_div(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out)
{
    lmmc_real_t zero;
    lmmc_real_t abs_c, abs_d;
    lmmc_real_t r, denom, tmp1, tmp2;

    if (a == NULL || b == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /** 除数实部与虚部同时为零时返回数值失败. */
    LMMC_REAL_SET_D(&zero, 0.0);
    if (LMMC_REAL_CMP(&b->real, &zero) == 0 && LMMC_REAL_CMP(&b->imag, &zero) == 0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    /** Smith 复数除法令除数 b = c + di,并按主导分量缩放. */
    LMMC_REAL_ABS(&abs_c, &b->real);
    LMMC_REAL_ABS(&abs_d, &b->imag);

    if (LMMC_REAL_CMP(&abs_d, &abs_c) <= 0) {
        /** |d| <= |c| 时取 r = d/c,denom = c + d*r. */
        LMMC_REAL_DIV(&r, &b->imag, &b->real);
        LMMC_REAL_MUL(&tmp1, &b->imag, &r);
        LMMC_REAL_ADD(&denom, &b->real, &tmp1);

        /** out.real = (a.real + a.imag * r) / denom. */
        LMMC_REAL_MUL(&tmp1, &a->imag, &r);
        LMMC_REAL_ADD(&tmp2, &a->real, &tmp1);
        LMMC_REAL_DIV(&out->real, &tmp2, &denom);

        /** out.imag = (a.imag - a.real * r) / denom. */
        LMMC_REAL_MUL(&tmp1, &a->real, &r);
        LMMC_REAL_SUB(&tmp2, &a->imag, &tmp1);
        LMMC_REAL_DIV(&out->imag, &tmp2, &denom);
    } else {
        /** |d| > |c| 时取 r = c/d,denom = d + c*r. */
        LMMC_REAL_DIV(&r, &b->real, &b->imag);
        LMMC_REAL_MUL(&tmp1, &b->real, &r);
        LMMC_REAL_ADD(&denom, &b->imag, &tmp1);

        /** out.real = (a.real * r + a.imag) / denom. */
        LMMC_REAL_MUL(&tmp1, &a->real, &r);
        LMMC_REAL_ADD(&tmp2, &tmp1, &a->imag);
        LMMC_REAL_DIV(&out->real, &tmp2, &denom);

        /** out.imag = (a.imag * r - a.real) / denom. */
        LMMC_REAL_MUL(&tmp1, &a->imag, &r);
        LMMC_REAL_SUB(&tmp2, &tmp1, &a->real);
        LMMC_REAL_DIV(&out->imag, &tmp2, &denom);
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_conj(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_SET(&out->real, &z->real);
    LMMC_REAL_NEG(&out->imag, &z->imag);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_modulus(const lmmc_complex_t* z, lmmc_real_t* out)
{
    lmmc_real_t r2, i2, sum;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_MUL(&r2, &z->real, &z->real);
    LMMC_REAL_MUL(&i2, &z->imag, &z->imag);
    LMMC_REAL_ADD(&sum, &r2, &i2);
    LMMC_REAL_SQRT(out, &sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_arg(const lmmc_complex_t* z, lmmc_real_t* out)
{
    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_ATAN2(out, &z->imag, &z->real);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_exp(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    lmmc_real_t ex, cos_y, sin_y;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /** z = x + iy 时,e^z = e^x * (cos(y) + i*sin(y)). */
    LMMC_REAL_EXP(&ex, &z->real);
    LMMC_REAL_COS(&cos_y, &z->imag);
    LMMC_REAL_SIN(&sin_y, &z->imag);
    LMMC_REAL_MUL(&out->real, &ex, &cos_y);
    LMMC_REAL_MUL(&out->imag, &ex, &sin_y);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_log(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    lmmc_real_t zero, modulus, r2, i2, sum;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /** z = 0+0i 时对数位于定义域之外. */
    LMMC_REAL_SET_D(&zero, 0.0);
    if (LMMC_REAL_CMP(&z->real, &zero) == 0 && LMMC_REAL_CMP(&z->imag, &zero) == 0) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }

    /** 使用 ln(z) = ln|z| + i*arg(z),先计算 |z|. */
    LMMC_REAL_MUL(&r2, &z->real, &z->real);
    LMMC_REAL_MUL(&i2, &z->imag, &z->imag);
    LMMC_REAL_ADD(&sum, &r2, &i2);
    LMMC_REAL_SQRT(&modulus, &sum);

    /** 实部为 ln|z|. */
    LMMC_REAL_LOG(&out->real, &modulus);

    /** 虚部为 arg(z) = atan2(imag, real). */
    LMMC_REAL_ATAN2(&out->imag, &z->imag, &z->real);

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_sqrt(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    lmmc_real_t modulus, r2, i2, sum, sqrt_mod, arg_z, half_arg;
    lmmc_real_t two, cos_ha, sin_ha;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /** 使用 sqrt(z) = sqrt(|z|) * (cos(arg(z)/2) + i*sin(arg(z)/2)),
     * 先计算 |z|.
     */
    LMMC_REAL_MUL(&r2, &z->real, &z->real);
    LMMC_REAL_MUL(&i2, &z->imag, &z->imag);
    LMMC_REAL_ADD(&sum, &r2, &i2);
    LMMC_REAL_SQRT(&modulus, &sum);

    /** 计算 sqrt(|z|). */
    LMMC_REAL_SQRT(&sqrt_mod, &modulus);

    /** 计算 arg(z) / 2. */
    LMMC_REAL_ATAN2(&arg_z, &z->imag, &z->real);
    LMMC_REAL_SET_D(&two, 2.0);
    LMMC_REAL_DIV(&half_arg, &arg_z, &two);

    /** 计算 cos(arg/2) 与 sin(arg/2). */
    LMMC_REAL_COS(&cos_ha, &half_arg);
    LMMC_REAL_SIN(&sin_ha, &half_arg);

    /** 合成 sqrt(|z|) * (cos(arg/2) + i*sin(arg/2)). */
    LMMC_REAL_MUL(&out->real, &sqrt_mod, &cos_ha);
    LMMC_REAL_MUL(&out->imag, &sqrt_mod, &sin_ha);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_sin(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    lmmc_real_t sin_x, cos_x, cosh_y, sinh_y;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /** z = x + iy 时,sin(z) = sin(x)*cosh(y) + i*cos(x)*sinh(y). */
    LMMC_REAL_SIN(&sin_x, &z->real);
    LMMC_REAL_COS(&cos_x, &z->real);
    LMMC_REAL_COSH(&cosh_y, &z->imag);
    LMMC_REAL_SINH(&sinh_y, &z->imag);

    LMMC_REAL_MUL(&out->real, &sin_x, &cosh_y);
    LMMC_REAL_MUL(&out->imag, &cos_x, &sinh_y);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_cos(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    lmmc_real_t cos_x, sin_x, cosh_y, sinh_y, neg_sin_x;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /** z = x + iy 时,cos(z) = cos(x)*cosh(y) - i*sin(x)*sinh(y). */
    LMMC_REAL_COS(&cos_x, &z->real);
    LMMC_REAL_SIN(&sin_x, &z->real);
    LMMC_REAL_COSH(&cosh_y, &z->imag);
    LMMC_REAL_SINH(&sinh_y, &z->imag);

    LMMC_REAL_MUL(&out->real, &cos_x, &cosh_y);
    LMMC_REAL_NEG(&neg_sin_x, &sin_x);
    LMMC_REAL_MUL(&out->imag, &neg_sin_x, &sinh_y);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_pow(const lmmc_complex_t* base, const lmmc_complex_t* exp, lmmc_complex_t* out)
{
    lmmc_real_t zero;
    lmmc_complex_t log_base, exp_times_log;
    lmmc_status_t st;

    if (base == NULL || exp == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Check if base == 0+0i and exp.real < 0 */
    LMMC_REAL_SET_D(&zero, 0.0);
    if (LMMC_REAL_CMP(&base->real, &zero) == 0 && LMMC_REAL_CMP(&base->imag, &zero) == 0) {
        if (LMMC_REAL_CMP(&exp->real, &zero) < 0) {
            return LMMC_STATUS_OUT_OF_RANGE;
        }
        if (LMMC_REAL_CMP(&exp->real, &zero) == 0 && LMMC_REAL_CMP(&exp->imag, &zero) == 0) {
            /* 0^0 = 1+0i (cpow convention) */
            LMMC_REAL_SET_D(&out->real, 1.0);
            LMMC_REAL_SET_D(&out->imag, 0.0);
            return LMMC_STATUS_OK;
        }
        /* 0^(positive) = 0+0i */
        LMMC_REAL_SET_D(&out->real, 0.0);
        LMMC_REAL_SET_D(&out->imag, 0.0);
        return LMMC_STATUS_OK;
    }

    /* pow(base, exp) = exp(exp * log(base)) */
    st = lmmc_complex_log(base, &log_base);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    st = lmmc_complex_mul(exp, &log_base, &exp_times_log);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    return lmmc_complex_exp(&exp_times_log, out);
}

lmmc_status_t lmmc_cvec_create(size_t size, lmmc_cvec_t* out)
{
    size_t n_bytes;
    lmmc_complex_t* data;

    if (out == NULL || size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_safe_mul_size(size, sizeof(lmmc_complex_t), &n_bytes)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    data = (lmmc_complex_t*)lmmc_alloc(n_bytes);
    if (data == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, n_bytes);

    out->size = size;
    out->data = data;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_cmat_create(size_t rows, size_t cols, lmmc_cmat_t* out)
{
    size_t n_elem, n_bytes;
    lmmc_complex_t* data;

    if (out == NULL || rows == 0 || cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_safe_mul_size(rows, cols, &n_elem)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    if (!lmmc_safe_mul_size(n_elem, sizeof(lmmc_complex_t), &n_bytes)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    data = (lmmc_complex_t*)lmmc_alloc(n_bytes);
    if (data == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, n_bytes);

    out->rows = rows;
    out->cols = cols;
    out->stride = cols;
    out->data = data;
    out->owns_data = 1;
    return LMMC_STATUS_OK;
}

void lmmc_cvec_destroy(lmmc_cvec_t* vec)
{
    if (vec == NULL) {
        return;
    }
    if (vec->owns_data && vec->data != NULL) {
        lmmc_free(vec->data);
    }
    vec->data = NULL;
    vec->size = 0;
    vec->owns_data = 0;
}

void lmmc_cmat_destroy(lmmc_cmat_t* mat)
{
    if (mat == NULL) {
        return;
    }
    if (mat->owns_data && mat->data != NULL) {
        lmmc_free(mat->data);
    }
    mat->data = NULL;
    mat->rows = 0;
    mat->cols = 0;
    mat->stride = 0;
    mat->owns_data = 0;
}
