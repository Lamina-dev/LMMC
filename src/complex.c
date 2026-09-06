/**
 * @file complex.c
 * @brief 复数类型构造,算术运算与复数向量/矩阵生命周期管理.
 */
#include <float.h>
#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/complex.h"

static int lmmc_complex_is_finite(const lmmc_complex_t* z)
{
    return isfinite(z->real) && isfinite(z->imag);
}

static lmmc_status_t lmmc_complex_finite_result(const lmmc_complex_t* z)
{
    return lmmc_complex_is_finite(z)
               ? LMMC_STATUS_OK
               : LMMC_STATUS_NUMERICAL_FAILURE;
}

lmmc_status_t lmmc_complex_create(lmmc_real_t real, lmmc_real_t imag, lmmc_complex_t* out)
{
    if (out == NULL || !isfinite(real) || !isfinite(imag)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    LMMC_REAL_SET(&out->real, &real);
    LMMC_REAL_SET(&out->imag, &imag);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_from_polar(lmmc_real_t r, lmmc_real_t theta, lmmc_complex_t* out)
{
    lmmc_real_t cos_theta, sin_theta;

    if (out == NULL || !isfinite(r) || !isfinite(theta) || r < 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (r == 0.0) {
        out->real = 0.0;
        out->imag = 0.0;
        return LMMC_STATUS_OK;
    }

    cos_theta = cos(theta);
    sin_theta = sin(theta);
    out->real = r * cos_theta;
    out->imag = r * sin_theta;
    return isfinite(out->real) && isfinite(out->imag)
               ? LMMC_STATUS_OK
               : LMMC_STATUS_NUMERICAL_FAILURE;
}

lmmc_status_t lmmc_complex_add(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(a) || !lmmc_complex_is_finite(b)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_ADD(&out->real, &a->real, &b->real);
    LMMC_REAL_ADD(&out->imag, &a->imag, &b->imag);
    return lmmc_complex_finite_result(out);
}

lmmc_status_t lmmc_complex_sub(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out)
{
    if (a == NULL || b == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(a) || !lmmc_complex_is_finite(b)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_SUB(&out->real, &a->real, &b->real);
    LMMC_REAL_SUB(&out->imag, &a->imag, &b->imag);
    return lmmc_complex_finite_result(out);
}

/**
 * @brief Sum products of significands while retaining their binary scale.
 *
 * Zero products do not determine the shared exponent. Product residuals
 * preserve low-order contributions when the aligned products cancel.
 * @see Ogita, Rump, Oishi, "Accurate Sum and Dot Product" (2005),
 * Algorithm 3.5 (TwoProductFMA). https://doi.org/10.1137/030601818
 */
static double lmmc_scaled_sum_products(double a, double b, int ep,
                                       double c, double d, int eq, int* scale)
{
    const double p = a * b;
    const double q = c * d;
    *scale = p == 0.0 ? eq : q == 0.0 ? ep : ep > eq ? ep : eq;
    const double sum = scalbn(p, ep - *scale) + scalbn(q, eq - *scale);
    const double residual = scalbn(fma(a, b, -p), ep - *scale) +
                            scalbn(fma(c, d, -q), eq - *scale);
    return sum + residual;
}

/**
 * @brief Evaluate a two-product sum with a shared binary exponent.
 *
 * Normal-range sums use FMA and the second product's residual. Products or
 * sums at the exponent boundaries are formed from individual significands,
 * aligned, and summed before the final scale. Each component has its own scale.
 * @see David Goldberg, "What Every Computer Scientist Should Know About
 * Floating-Point Arithmetic" (1991), Floating-point Formats and Theorem 7.
 * https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
 * @see Fred J. Tydeman, WG14 N1399 (2009), complex arithmetic range cases.
 * https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1399.htm
 * @see Ogita, Rump, Oishi, "Accurate Sum and Dot Product" (2005),
 * Algorithm 3.5 (TwoProductFMA). https://doi.org/10.1137/030601818
 */
static double lmmc_sum_products(double a, double b, double c, double d)
{
    const double p = a * b;
    const double q = c * d;
    int ea, eb, ec, ed;
    double ma, mb, mc, md;
    int scale;

    if (a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0) {
        return p + q;
    }
    if (isfinite(p) && isfinite(q) &&
        fabs(p) >= DBL_MIN && fabs(q) >= DBL_MIN) {
        const double sum = fma(a, b, q) + fma(c, d, -q);
        if (isfinite(sum) && fabs(sum) >= DBL_MIN) {
            return sum;
        }
    }
    ma = frexp(a, &ea);
    mb = frexp(b, &eb);
    mc = frexp(c, &ec);
    md = frexp(d, &ed);
    const double sum = lmmc_scaled_sum_products(ma, mb, ea + eb,
                                                mc, md, ec + ed, &scale);
    return scalbn(sum, scale);
}

lmmc_status_t lmmc_complex_mul(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out)
{
    lmmc_complex_t result;

    if (a == NULL || b == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(a) || !lmmc_complex_is_finite(b)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    result.real = lmmc_sum_products(a->real, b->real, -a->imag, b->imag);
    result.imag = lmmc_sum_products(a->real, b->imag, a->imag, b->real);
    if (!lmmc_complex_is_finite(&result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_div(const lmmc_complex_t* a, const lmmc_complex_t* b, lmmc_complex_t* out)
{
    int ea, eb, ec, ed;
    int denominator_scale, real_scale, imag_scale;
    lmmc_complex_t result;

    if (a == NULL || b == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(a) || !lmmc_complex_is_finite(b)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (b->real == 0.0 && b->imag == 0.0) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    const double ma = frexp(a->real, &ea);
    const double mb = frexp(a->imag, &eb);
    const double mc = frexp(b->real, &ec);
    const double md = frexp(b->imag, &ed);
    const double denominator = lmmc_scaled_sum_products(
        mc, mc, ec + ec, md, md, ed + ed, &denominator_scale);
    const double real = lmmc_scaled_sum_products(
        ma, mc, ea + ec, mb, md, eb + ed, &real_scale);
    const double imag = lmmc_scaled_sum_products(
        mb, mc, eb + ec, -ma, md, ea + ed, &imag_scale);
    result.real = scalbn(real / denominator, real_scale - denominator_scale);
    result.imag = scalbn(imag / denominator, imag_scale - denominator_scale);
    if (!lmmc_complex_is_finite(&result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_conj(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(z)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_SET(&out->real, &z->real);
    LMMC_REAL_NEG(&out->imag, &z->imag);
    return lmmc_complex_finite_result(out);
}

lmmc_status_t lmmc_complex_modulus(const lmmc_complex_t* z, lmmc_real_t* out)
{
    lmmc_real_t modulus;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(z)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    modulus = hypot(z->real, z->imag);
    if (!isfinite(modulus)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = modulus;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_arg(const lmmc_complex_t* z, lmmc_real_t* out)
{
    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(z)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    LMMC_REAL_ATAN2(out, &z->imag, &z->real);
    return isfinite(*out) ? LMMC_STATUS_OK : LMMC_STATUS_NUMERICAL_FAILURE;
}

lmmc_status_t lmmc_complex_exp(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    lmmc_real_t ex, cos_y, sin_y;
    lmmc_complex_t result;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(z)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    /** z = x + iy 时,e^z = e^x * (cos(y) + i*sin(y)). */
    LMMC_REAL_EXP(&ex, &z->real);
    LMMC_REAL_COS(&cos_y, &z->imag);
    LMMC_REAL_SIN(&sin_y, &z->imag);
    if (!isfinite(ex) || ex < DBL_MIN) {
        /* e^(x/2) factors carry the scale through each trigonometric product. */
        ex = exp(z->real * 0.5);
        result.real = (ex * cos_y) * ex;
        result.imag = (ex * sin_y) * ex;
    } else {
        LMMC_REAL_MUL(&result.real, &ex, &cos_y);
        LMMC_REAL_MUL(&result.imag, &ex, &sin_y);
    }
    if (!lmmc_complex_is_finite(&result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = result;
    return LMMC_STATUS_OK;
}

/**
 * @brief Return a rounded sum and its floating-point residual.
 * @see Ogita, Rump, Oishi, "Accurate Sum and Dot Product" (2005),
 * Algorithm 3.1 (TwoSum). https://doi.org/10.1137/030601818
 */
static double lmmc_two_sum(double a, double b, double* residual)
{
    const double sum = a + b;
    const double virtual_b = sum - a;
    *residual = (a - (sum - virtual_b)) + (b - virtual_b);
    return sum;
}

lmmc_status_t lmmc_complex_log(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    lmmc_real_t major;
    lmmc_real_t minor;
    lmmc_real_t ratio;
    lmmc_complex_t result;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(z)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (z->real == 0.0 && z->imag == 0.0) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }

    major = fmax(fabs(z->real), fabs(z->imag));
    minor = fmin(fabs(z->real), fabs(z->imag));
    if (major >= 0.5 && major <= 1.0) {
        /**
         * Compensated squared modulus supplies the small argument of log1p.
         * Ordered squares use FastTwoSum; FMA supplies product residuals.
         * TwoSum carries the low-order terms through residual cancellation.
         * @see Ogita, Rump, Oishi, "Accurate Sum and Dot Product" (2005),
         * Algorithms 1.1 (FastTwoSum) and 3.5 (TwoProductFMA).
         * https://doi.org/10.1137/030601818
         */
        const double major_squared = major * major;
        const double minor_squared = minor * minor;
        const double sum = major_squared + minor_squared;
        const double sum_error = (major_squared - sum) + minor_squared;
        const double major_error = fma(major, major, -major_squared);
        const double minor_error = fma(minor, minor, -minor_squared);
        double tail_a, tail_b, tail_c;
        double correction = lmmc_two_sum(sum_error, major_error, &tail_a);
        correction = lmmc_two_sum(correction, minor_error, &tail_b);
        const double delta = lmmc_two_sum(sum - 1.0, correction, &tail_c);
        result.real = 0.5 * log1p(delta + ((tail_a + tail_b) + tail_c));
    } else {
        ratio = minor / major;
        result.real = log(major) + 0.5 * log1p(ratio * ratio);
    }
    result.imag = atan2(z->imag, z->real);
    if (!lmmc_complex_is_finite(&result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_sqrt(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    lmmc_real_t scale;
    lmmc_real_t scaled_magnitude;
    lmmc_real_t component;
    lmmc_complex_t result;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(z)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (z->imag == 0.0) {
        if (z->real < 0.0) {
            result.real = 0.0;
            result.imag = copysign(sqrt(-z->real), z->imag);
        } else {
            result.real = sqrt(fabs(z->real));
            result.imag = copysign(0.0, z->imag);
        }
        *out = result;
        return LMMC_STATUS_OK;
    }

    scale = fmax(fabs(z->real), fabs(z->imag));
    scaled_magnitude = hypot(z->real / scale, z->imag / scale);
    if (z->real >= 0.0) {
        component = sqrt(scale) *
                    sqrt(0.5 * (scaled_magnitude + z->real / scale));
        result.real = component;
        result.imag = z->imag / (2.0 * component);
    } else {
        component = sqrt(scale) *
                    sqrt(0.5 * (scaled_magnitude - z->real / scale));
        result.real = fabs(z->imag) / (2.0 * component);
        result.imag = copysign(component, z->imag);
    }
    if (!lmmc_complex_is_finite(&result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = result;
    return LMMC_STATUS_OK;
}

/**
 * @brief Form even*cosh(y) and odd*sinh(y) as Cartesian components.
 *
 * Finite hyperbolic factors use direct products. At hyperbolic overflow,
 * the dominant exponential is split into two exp(|y|/2) factors; the relative
 * contribution exp(-2|y|) is below double precision in this range.
 * @see NIST DLMF, 4.28.1 and 4.28.2, exponential definitions of sinh and cosh.
 * https://dlmf.nist.gov/4.28
 */
static lmmc_complex_t lmmc_complex_hyperbolic_products(double even, double odd, double y)
{
    const double cosh_y = cosh(y);
    const double sinh_y = sinh(y);
    lmmc_complex_t result;

    if (isfinite(cosh_y) && isfinite(sinh_y)) {
        result.real = even * cosh_y;
        result.imag = odd * sinh_y;
    } else {
        const double scale = exp(fabs(y) * 0.5);
        const double half_scale = scale * 0.5;
        result.real = (even * half_scale) * scale;
        result.imag = (odd * copysign(half_scale, y)) * scale;
    }
    return result;
}

lmmc_status_t lmmc_complex_sin(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    lmmc_real_t sin_x, cos_x;
    lmmc_complex_t result;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(z)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    /** z = x + iy 时,sin(z) = sin(x)*cosh(y) + i*cos(x)*sinh(y). */
    LMMC_REAL_SIN(&sin_x, &z->real);
    LMMC_REAL_COS(&cos_x, &z->real);

    result = lmmc_complex_hyperbolic_products(sin_x, cos_x, z->imag);
    if (!lmmc_complex_is_finite(&result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_cos(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    lmmc_real_t cos_x, sin_x;
    lmmc_complex_t result;

    if (z == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_complex_is_finite(z)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    /** z = x + iy 时,cos(z) = cos(x)*cosh(y) - i*sin(x)*sinh(y). */
    LMMC_REAL_COS(&cos_x, &z->real);
    LMMC_REAL_SIN(&sin_x, &z->real);

    result = lmmc_complex_hyperbolic_products(cos_x, -sin_x, z->imag);
    if (!lmmc_complex_is_finite(&result)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out = result;
    return LMMC_STATUS_OK;
}

/**
 * @brief Evaluate a finite real integer power of a nonzero finite base.
 *
 * Negative powers start from the reciprocal. Halving integral doubles keeps
 * the exponent bits available without conversion to a fixed-width integer.
 * The first selected factor initializes the product directly.
 * @see Menezes, van Oorschot, Vanstone, Handbook of Applied Cryptography,
 * Algorithm 14.76 (right-to-left binary exponentiation).
 * https://cacr.uwaterloo.ca/hac/about/chap14.pdf
 */
static lmmc_status_t lmmc_complex_integer_power(const lmmc_complex_t* base,
                                               double exponent, lmmc_complex_t* out)
{
    const lmmc_complex_t one = {1.0, 0.0};
    lmmc_complex_t factor = *base;
    lmmc_complex_t result = one;
    lmmc_status_t st;
    double remaining = fabs(exponent);
    int has_product = 0;

    if (exponent < 0.0) {
        st = lmmc_complex_div(&one, base, &factor);
        if (st != LMMC_STATUS_OK) return st;
    }
    while (remaining != 0.0) {
        if (fmod(remaining, 2.0) != 0.0) {
            if (!has_product) {
                result = factor;
                has_product = 1;
            } else {
                st = lmmc_complex_mul(&result, &factor, &result);
                if (st != LMMC_STATUS_OK) return st;
            }
        }
        remaining = floor(remaining * 0.5);
        if (remaining != 0.0) {
            st = lmmc_complex_mul(&factor, &factor, &factor);
            if (st != LMMC_STATUS_OK) return st;
        }
    }
    *out = result;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_pow(const lmmc_complex_t* base, const lmmc_complex_t* exp, lmmc_complex_t* out)
{
    lmmc_complex_t log_base, exp_times_log, result;
    lmmc_status_t st;

    if (base == NULL || exp == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_complex_is_finite(base) || !lmmc_complex_is_finite(exp)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (base->real == 0.0 && base->imag == 0.0) {
        if (exp->real == 0.0 && exp->imag == 0.0) {
            result.real = 1.0;
        } else if (exp->real > 0.0) {
            result.real = 0.0;
        } else {
            return LMMC_STATUS_OUT_OF_RANGE;
        }
        result.imag = 0.0;
        *out = result;
        return LMMC_STATUS_OK;
    }

    if (exp->imag == 0.0 && trunc(exp->real) == exp->real) {
        return lmmc_complex_integer_power(base, exp->real, out);
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

    st = lmmc_complex_exp(&exp_times_log, &result);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    *out = result;
    return LMMC_STATUS_OK;
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
