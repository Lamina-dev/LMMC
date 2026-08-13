/**
 * @file test_complex.c
 * @brief Property-based and unit tests for the complex number module.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define PBT_ITERATIONS 100
#define TEST_PI LMMC_CONST_PI

static int test_failures = 0;
static int test_count = 0;

#define REPORT(name, result) do { \
    test_count++; \
    if (result) { \
        printf("  FAIL: %s\n", name); \
        test_failures++; \
    } else { \
        printf("  PASS: %s\n", name); \
    } \
} while (0)

static double rand_double(double lo, double hi)
{
    return ((double)rand() / RAND_MAX) * (hi - lo) + lo;
}

/**
 * Normalize angle to (-pi, pi].
 */
static double normalize_angle(double theta)
{
    while (theta > TEST_PI) theta -= 2.0 * TEST_PI;
    while (theta <= -TEST_PI) theta += 2.0 * TEST_PI;
    return theta;
}

/**
 * Compare two complex numbers with tolerance.
 */
static int complex_nearly_equal(const lmmc_complex_t* a, const lmmc_complex_t* b, double eps)
{
    return lmmc_test_nearly_equal(a->real, b->real, eps) &&
           lmmc_test_nearly_equal(a->imag, b->imag, eps);
}

/* ---- Property Tests ---- */

static int test_property1_polar_cartesian_roundtrip(void)
{
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        double r = rand_double(0.01, 10.0);
        double theta = rand_double(-TEST_PI, TEST_PI);
        lmmc_complex_t z;
        lmmc_real_t mod_out, arg_out;
        lmmc_status_t st;

        st = lmmc_complex_from_polar(r, theta, &z);
        if (st != LMMC_STATUS_OK) {
            printf("    from_polar failed at iteration %d\n", i);
            return 1;
        }

        st = lmmc_complex_modulus(&z, &mod_out);
        if (st != LMMC_STATUS_OK) {
            printf("    modulus failed at iteration %d\n", i);
            return 1;
        }

        st = lmmc_complex_arg(&z, &arg_out);
        if (st != LMMC_STATUS_OK) {
            printf("    arg failed at iteration %d\n", i);
            return 1;
        }

        /* Check modulus recovery */
        if (!lmmc_test_nearly_equal(mod_out, r, eps)) {
            printf("    Modulus mismatch at iter %d: got %g, expected %g\n",
                   i, mod_out, r);
            return 1;
        }

        /* Check arg recovery (normalize theta) */
        double expected_theta = normalize_angle(theta);
        if (!lmmc_test_nearly_equal(arg_out, expected_theta, eps)) {
            printf("    Arg mismatch at iter %d: got %g, expected %g\n",
                   i, arg_out, expected_theta);
            return 1;
        }
    }
    return 0;
}

static int test_property2_mul_div_roundtrip(void)
{
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_complex_t a, b, product, recovered;
        lmmc_status_t st;

        a.real = rand_double(-10.0, 10.0);
        a.imag = rand_double(-10.0, 10.0);

        /* Ensure b is non-zero */
        do {
            b.real = rand_double(-10.0, 10.0);
            b.imag = rand_double(-10.0, 10.0);
        } while (fabs(b.real) < 1e-6 && fabs(b.imag) < 1e-6);

        st = lmmc_complex_mul(&a, &b, &product);
        if (st != LMMC_STATUS_OK) {
            printf("    mul failed at iteration %d\n", i);
            return 1;
        }

        st = lmmc_complex_div(&product, &b, &recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    div failed at iteration %d\n", i);
            return 1;
        }

        if (!complex_nearly_equal(&recovered, &a, eps)) {
            printf("    Round-trip failed at iter %d: a=(%g,%g), recovered=(%g,%g)\n",
                   i, a.real, a.imag, recovered.real, recovered.imag);
            return 1;
        }
    }
    return 0;
}

static int test_property3_exp_log_roundtrip(void)
{
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_complex_t z, log_z, recovered;
        lmmc_status_t st;

        /* Generate z with non-zero modulus */
        do {
            z.real = rand_double(-5.0, 5.0);
            z.imag = rand_double(-5.0, 5.0);
        } while (fabs(z.real) < 1e-6 && fabs(z.imag) < 1e-6);

        st = lmmc_complex_log(&z, &log_z);
        if (st != LMMC_STATUS_OK) {
            printf("    log failed at iteration %d\n", i);
            return 1;
        }

        st = lmmc_complex_exp(&log_z, &recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    exp failed at iteration %d\n", i);
            return 1;
        }

        if (!complex_nearly_equal(&recovered, &z, eps)) {
            printf("    exp-log round-trip failed at iter %d: z=(%g,%g), recovered=(%g,%g)\n",
                   i, z.real, z.imag, recovered.real, recovered.imag);
            return 1;
        }
    }
    return 0;
}

static int test_property4_pythagorean_identity(void)
{
    int i;
    double eps = 1e-9;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_complex_t z, sin_z, cos_z, sin2, cos2, sum;
        lmmc_complex_t one;
        lmmc_status_t st;

        z.real = rand_double(-3.0, 3.0);
        z.imag = rand_double(-3.0, 3.0);

        st = lmmc_complex_sin(&z, &sin_z);
        if (st != LMMC_STATUS_OK) {
            printf("    sin failed at iteration %d\n", i);
            return 1;
        }

        st = lmmc_complex_cos(&z, &cos_z);
        if (st != LMMC_STATUS_OK) {
            printf("    cos failed at iteration %d\n", i);
            return 1;
        }

        /* sin(z)^2 */
        st = lmmc_complex_mul(&sin_z, &sin_z, &sin2);
        if (st != LMMC_STATUS_OK) {
            printf("    mul(sin,sin) failed at iteration %d\n", i);
            return 1;
        }

        /* cos(z)^2 */
        st = lmmc_complex_mul(&cos_z, &cos_z, &cos2);
        if (st != LMMC_STATUS_OK) {
            printf("    mul(cos,cos) failed at iteration %d\n", i);
            return 1;
        }

        /* sin^2 + cos^2 */
        st = lmmc_complex_add(&sin2, &cos2, &sum);
        if (st != LMMC_STATUS_OK) {
            printf("    add failed at iteration %d\n", i);
            return 1;
        }

        one.real = 1.0;
        one.imag = 0.0;

        if (!complex_nearly_equal(&sum, &one, eps)) {
            printf("    Pythagorean identity failed at iter %d: sum=(%g,%g)\n",
                   i, sum.real, sum.imag);
            return 1;
        }
    }
    return 0;
}

static int test_property5_sqrt_roundtrip(void)
{
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_complex_t z, sqrt_z, squared;
        lmmc_status_t st;

        z.real = rand_double(-10.0, 10.0);
        z.imag = rand_double(-10.0, 10.0);

        st = lmmc_complex_sqrt(&z, &sqrt_z);
        if (st != LMMC_STATUS_OK) {
            printf("    sqrt failed at iteration %d\n", i);
            return 1;
        }

        st = lmmc_complex_mul(&sqrt_z, &sqrt_z, &squared);
        if (st != LMMC_STATUS_OK) {
            printf("    mul(sqrt,sqrt) failed at iteration %d\n", i);
            return 1;
        }

        if (!complex_nearly_equal(&squared, &z, eps)) {
            printf("    sqrt round-trip failed at iter %d: z=(%g,%g), squared=(%g,%g)\n",
                   i, z.real, z.imag, squared.real, squared.imag);
            return 1;
        }
    }
    return 0;
}

static int test_property6_zero_initialization(void)
{
    int i;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        /* Random size between 1 and 50 */
        size_t vec_size = (size_t)(rand() % 50) + 1;
        size_t mat_rows = (size_t)(rand() % 20) + 1;
        size_t mat_cols = (size_t)(rand() % 20) + 1;
        size_t j, r, c;
        lmmc_cvec_t vec;
        lmmc_cmat_t mat;
        lmmc_status_t st;

        /* Test vector zero-initialization */
        st = lmmc_cvec_create(vec_size, &vec);
        if (st != LMMC_STATUS_OK) {
            printf("    cvec_create failed at iteration %d\n", i);
            return 1;
        }

        for (j = 0; j < vec_size; j++) {
            if (vec.data[j].real != 0.0 || vec.data[j].imag != 0.0) {
                printf("    Vector element [%zu] not zero at iter %d\n", j, i);
                lmmc_cvec_destroy(&vec);
                return 1;
            }
        }
        lmmc_cvec_destroy(&vec);

        /* Test matrix zero-initialization */
        st = lmmc_cmat_create(mat_rows, mat_cols, &mat);
        if (st != LMMC_STATUS_OK) {
            printf("    cmat_create failed at iteration %d\n", i);
            return 1;
        }

        for (r = 0; r < mat_rows; r++) {
            for (c = 0; c < mat_cols; c++) {
                lmmc_complex_t* elem = &mat.data[r * mat.stride + c];
                if (elem->real != 0.0 || elem->imag != 0.0) {
                    printf("    Matrix element [%zu][%zu] not zero at iter %d\n", r, c, i);
                    lmmc_cmat_destroy(&mat);
                    return 1;
                }
            }
        }
        lmmc_cmat_destroy(&mat);
    }
    return 0;
}

/* ---- Unit Tests ---- */

/* NULL pointer checks */
static int test_unit_null_checks(void)
{
    lmmc_complex_t z, out_c;
    lmmc_real_t out_r;
    lmmc_status_t st;

    z.real = 1.0;
    z.imag = 2.0;

    /* create with NULL out */
    st = lmmc_complex_create(1.0, 2.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    complex_create(NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* from_polar with NULL out */
    st = lmmc_complex_from_polar(1.0, 0.0, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    from_polar(NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* add with NULL */
    st = lmmc_complex_add(NULL, &z, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    add(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* mul with NULL */
    st = lmmc_complex_mul(&z, NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    mul(...,NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* div with NULL */
    st = lmmc_complex_div(&z, &z, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    div(...,NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* modulus with NULL */
    st = lmmc_complex_modulus(NULL, &out_r);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    modulus(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* arg with NULL */
    st = lmmc_complex_arg(&z, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    arg(...,NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* exp with NULL */
    st = lmmc_complex_exp(NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    exp(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* log with NULL */
    st = lmmc_complex_log(NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    log(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* sqrt with NULL */
    st = lmmc_complex_sqrt(NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    sqrt(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* sin with NULL */
    st = lmmc_complex_sin(NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    sin(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* cos with NULL */
    st = lmmc_complex_cos(NULL, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    cos(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* pow with NULL */
    st = lmmc_complex_pow(NULL, &z, &out_c);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    pow(NULL,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* cvec_create with NULL out */
    st = lmmc_cvec_create(5, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    cvec_create(5, NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* cvec_create with size 0 */
    {
        lmmc_cvec_t vec;
        st = lmmc_cvec_create(0, &vec);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("    cvec_create(0,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
            return 1;
        }
    }

    /* cmat_create with NULL out */
    st = lmmc_cmat_create(3, 3, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    cmat_create(3,3,NULL) expected INVALID_ARGUMENT, got %d\n", (int)st);
        return 1;
    }

    /* cmat_create with rows=0 */
    {
        lmmc_cmat_t mat;
        st = lmmc_cmat_create(0, 3, &mat);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("    cmat_create(0,3,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
            return 1;
        }
    }

    /* cmat_create with cols=0 */
    {
        lmmc_cmat_t mat;
        st = lmmc_cmat_create(3, 0, &mat);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            printf("    cmat_create(3,0,...) expected INVALID_ARGUMENT, got %d\n", (int)st);
            return 1;
        }
    }

    return 0;
}

/* Zero divisor test */
static int test_unit_zero_divisor(void)
{
    lmmc_complex_t a, zero_b, out;
    lmmc_status_t st;

    a.real = 3.0;
    a.imag = 4.0;
    zero_b.real = 0.0;
    zero_b.imag = 0.0;

    st = lmmc_complex_div(&a, &zero_b, &out);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        printf("    div by zero expected NUMERICAL_FAILURE, got %d\n", (int)st);
        return 1;
    }

    return 0;
}

/* log(0) test */
static int test_unit_log_zero(void)
{
    lmmc_complex_t zero_z, out;
    lmmc_status_t st;

    zero_z.real = 0.0;
    zero_z.imag = 0.0;

    st = lmmc_complex_log(&zero_z, &out);
    if (st != LMMC_STATUS_OUT_OF_RANGE) {
        printf("    log(0) expected OUT_OF_RANGE, got %d\n", (int)st);
        return 1;
    }

    return 0;
}

/* Struct layout verification */
static int test_unit_struct_layout(void)
{
    lmmc_complex_t z;

    /* Verify lmmc_complex_t has real and imag fields of type lmmc_real_t */
    z.real = 1.5;
    z.imag = -2.5;

    if (z.real != 1.5 || z.imag != -2.5) {
        printf("    Struct layout: field assignment failed\n");
        return 1;
    }

    /* Verify sizeof is as expected (two doubles) */
    if (sizeof(lmmc_complex_t) < 2 * sizeof(lmmc_real_t)) {
        printf("    Struct layout: unexpected size %zu (expected >= %zu)\n",
               sizeof(lmmc_complex_t), 2 * sizeof(lmmc_real_t));
        return 1;
    }

    /* Verify cvec_t layout */
    {
        lmmc_cvec_t vec;
        lmmc_status_t st = lmmc_cvec_create(3, &vec);
        if (st != LMMC_STATUS_OK) {
            printf("    cvec_create for layout check failed\n");
            return 1;
        }
        if (vec.size != 3 || vec.data == NULL || vec.owns_data != 1) {
            printf("    cvec_t layout: unexpected field values\n");
            lmmc_cvec_destroy(&vec);
            return 1;
        }
        lmmc_cvec_destroy(&vec);
    }

    /* Verify cmat_t layout */
    {
        lmmc_cmat_t mat;
        lmmc_status_t st = lmmc_cmat_create(2, 4, &mat);
        if (st != LMMC_STATUS_OK) {
            printf("    cmat_create for layout check failed\n");
            return 1;
        }
        if (mat.rows != 2 || mat.cols != 4 || mat.stride < 4 ||
            mat.data == NULL || mat.owns_data != 1) {
            printf("    cmat_t layout: unexpected field values\n");
            lmmc_cmat_destroy(&mat);
            return 1;
        }
        lmmc_cmat_destroy(&mat);
    }

    return 0;
}

/* from_polar with r=0 should produce 0+0i regardless of theta */
static int test_unit_polar_zero_r(void)
{
    lmmc_complex_t z;
    lmmc_status_t st;

    st = lmmc_complex_from_polar(0.0, 1.234, &z);
    if (st != LMMC_STATUS_OK) {
        printf("    from_polar(0, theta) failed\n");
        return 1;
    }
    if (z.real != 0.0 || z.imag != 0.0) {
        printf("    from_polar(0, theta) expected (0,0), got (%g,%g)\n", z.real, z.imag);
        return 1;
    }

    return 0;
}

/* Destroy NULL is safe */
static int test_unit_destroy_null(void)
{
    /* These should not crash */
    lmmc_cvec_destroy(NULL);
    lmmc_cmat_destroy(NULL);
    return 0;
}

/* ---- Main ---- */

int main(void)
{
    srand(12345);
    lmmc_init();

    printf("=== Complex Module Property Tests ===\n");
    REPORT("Property 1: Polar-Cartesian round-trip",
           test_property1_polar_cartesian_roundtrip());
    REPORT("Property 2: Multiplication-division round-trip",
           test_property2_mul_div_roundtrip());
    REPORT("Property 3: Exp-log round-trip",
           test_property3_exp_log_roundtrip());
    REPORT("Property 4: Pythagorean identity",
           test_property4_pythagorean_identity());
    REPORT("Property 5: Sqrt round-trip",
           test_property5_sqrt_roundtrip());
    REPORT("Property 6: Vector/matrix zero-initialization",
           test_property6_zero_initialization());

    printf("\n=== Complex Module Unit Tests ===\n");
    REPORT("Unit: NULL pointer checks",
           test_unit_null_checks());
    REPORT("Unit: Zero divisor",
           test_unit_zero_divisor());
    REPORT("Unit: log(0) domain error",
           test_unit_log_zero());
    REPORT("Unit: Struct layout verification",
           test_unit_struct_layout());
    REPORT("Unit: from_polar with r=0",
           test_unit_polar_zero_r());
    REPORT("Unit: Destroy NULL safety",
           test_unit_destroy_null());

    printf("\n=== Results: %d/%d passed ===\n",
           test_count - test_failures, test_count);

    lmmc_deinit();
    return test_failures > 0 ? 1 : 0;
}
