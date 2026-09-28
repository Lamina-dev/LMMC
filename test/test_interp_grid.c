#include <float.h>
#include <math.h>
#include <stdio.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

static void test_nonuniform_affine_regression(void **state) {
    (void)state;
    const lmmc_real_t xs[] = {0.0, 1.0, 2.0, 4.0};
    const lmmc_real_t ys[] = {0.0, 1.0, 2.0, 3.0};
    lmmc_real_t zs[16], result = 42.0;
    size_t i, j;
    for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j)
            zs[i * 4 + j] = xs[i];
    }
    assert_true(lmmc_interp_bicubic(xs, 4, ys, 4, zs, 1.5, 1.5, &result) ==
                LMMC_STATUS_OK);
    assert_true(isfinite(result) && fabs(result - 1.5) <= 1e-12);
}

static lmmc_real_t constant_surface(lmmc_real_t x, lmmc_real_t y) {
    (void)x;
    (void)y;
    return 7.0;
}

static lmmc_real_t affine_surface(lmmc_real_t x, lmmc_real_t y) {
    return 2.0 * x - 3.0 * y + 5.0;
}

static lmmc_real_t cubic_surface(lmmc_real_t x, lmmc_real_t y) {
    return (x * x * x - 2.0 * x * x + x + 3.0) *
           (0.5 * y * y * y + y * y - 3.0 * y + 1.0);
}

static void check_surface_reproduction(lmmc_real_t (*surface)(lmmc_real_t, lmmc_real_t)) {
    const lmmc_real_t xs[] = {-2.0, -0.5, 0.25, 2.0, 4.0, 7.0};
    const lmmc_real_t ys[] = {-3.0, -1.0, 0.5, 1.25, 3.0, 6.0};
    const lmmc_real_t queries[][2] = {
        {-1.5, -2.0}, {6.5, 5.5}, {1.0, 1.0}, {-1.5, 5.5}, {6.5, -2.0}, {-2.0, -3.0}, {7.0, 6.0}, {0.25, 1.25}};
    lmmc_real_t zs[36];
    size_t i, j;
    for (i = 0; i < 6; ++i) {
        for (j = 0; j < 6; ++j)
            zs[i * 6 + j] = surface(xs[i], ys[j]);
    }
    for (i = 0; i < sizeof(queries) / sizeof(queries[0]); ++i) {
        lmmc_real_t result = 42.0;
        lmmc_real_t expected = surface(queries[i][0], queries[i][1]);
        assert_true(lmmc_interp_bicubic(xs, 6, ys, 6, zs,
                                        queries[i][0], queries[i][1], &result) == LMMC_STATUS_OK);
        assert_true(isfinite(result) && fabs(result - expected) <= 1e-10 * fmax(1.0, fabs(expected)));
    }
}

static void test_constant_reproduction(void **state) {
    (void)state;
    check_surface_reproduction(constant_surface);
}

static void test_affine_reproduction(void **state) {
    (void)state;
    check_surface_reproduction(affine_surface);
}

static void test_bicubic_reproduction(void **state) {
    (void)state;
    check_surface_reproduction(cubic_surface);
}

static void test_bilinear_extreme_span(void **state) {
    (void)state;
    const lmmc_real_t xs[] = {-DBL_MAX, DBL_MAX};
    const lmmc_real_t ys[] = {0.0, 1.0};
    const lmmc_real_t zs[] = {0.0, 0.0, 2.0, 2.0};
    lmmc_real_t result = 42.0;
    assert_int_equal(lmmc_interp_bilinear(xs, 2, ys, 2, zs, 0.0, 0.5, &result),
                     LMMC_STATUS_OK);
    assert_true(fabs(result - 1.0) <= 1e-12);
}

static void test_exact_nodes_at_extreme_spacing(void **state) {
    (void)state;
    const lmmc_real_t large[] = {0.0, 1e150, 2e150, 3e150};
    const lmmc_real_t tiny[] = {0.0, 1e-150, 2e-150, 3e-150};
    const lmmc_real_t *axes[][2] = {
        {large, large}, {tiny, tiny}, {large, tiny}};
    lmmc_real_t zs[16];
    size_t i;
    for (i = 0; i < 16; ++i) zs[i] = 7.0;
    for (i = 0; i < sizeof(axes) / sizeof(axes[0]); ++i) {
        lmmc_real_t result = 42.0;
        assert_true(lmmc_interp_bicubic(axes[i][0], 4, axes[i][1], 4, zs,
                    axes[i][0][1], axes[i][1][2], &result) == LMMC_STATUS_OK);
        assert_true(result == 7.0);
    }
}

static void test_boundary_stencil_locality(void **state) {
    (void)state;
    const lmmc_real_t axis[] = {0.0, 0.5, 1.5, 3.0, 5.0, 8.0};
    lmmc_real_t zs[36], result;
    size_t i;
    for (i = 0; i < 36; ++i)
        zs[i] = 2.0;
    zs[35] = NAN;
    assert_true(lmmc_interp_bicubic(axis, 6, axis, 6, zs, 0.25, 0.25, &result) ==
                LMMC_STATUS_OK);
    assert_true(isfinite(result) && fabs(result - 2.0) <= 1e-12);
    zs[35] = 2.0;
    zs[0] = NAN;
    assert_true(lmmc_interp_bicubic(axis, 6, axis, 6, zs, 7.0, 7.0, &result) ==
                LMMC_STATUS_OK);
    assert_true(isfinite(result) && fabs(result - 2.0) <= 1e-12);
}

static void test_invalid_axes_preserve_output(void **state) {
    (void)state;
    const lmmc_real_t axis[] = {0.0, 1.0, 2.0, 4.0};
    const lmmc_real_t bad_axes[][4] = {
        {0.0, 1.0, 1.0, 4.0}, {0.0, 2.0, 1.0, 4.0}, {0.0, NAN, 2.0, 4.0}, {0.0, 1.0, 2.0, INFINITY}, {-INFINITY, 1.0, 2.0, 4.0}};
    lmmc_real_t zs[16] = {0}, result = 42.0;
    size_t i;
    for (i = 0; i < sizeof(bad_axes) / sizeof(bad_axes[0]); ++i) {
        assert_true(lmmc_interp_bicubic(bad_axes[i], 4, axis, 4, zs, 1.5, 1.5, &result) ==
                    LMMC_STATUS_INVALID_ARGUMENT);
        assert_true(result == 42.0);
        assert_true(lmmc_interp_bicubic(axis, 4, bad_axes[i], 4, zs, 1.5, 1.5, &result) ==
                    LMMC_STATUS_INVALID_ARGUMENT);
        assert_true(result == 42.0);
    }
}

static void test_invalid_queries_and_values(void **state) {
    (void)state;
    const lmmc_real_t axis[] = {0.0, 1.0, 2.0, 4.0};
    lmmc_real_t zs[16] = {0}, result = 42.0;
    assert_true(lmmc_interp_bicubic(axis, 4, axis, 4, zs, NAN, 1.5, &result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(result == 42.0);
    assert_true(lmmc_interp_bicubic(axis, 4, axis, 4, zs, 1.5, INFINITY, &result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(result == 42.0);
    assert_true(lmmc_interp_bicubic(axis, 4, axis, 4, zs, -1.0, 1.5, &result) ==
                LMMC_STATUS_OUT_OF_RANGE);
    assert_true(result == 42.0);
    zs[5] = NAN;
    assert_true(lmmc_interp_bicubic(axis, 4, axis, 4, zs, 2.0, 2.0, &result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(result == 42.0);
    zs[5] = INFINITY;
    assert_true(lmmc_interp_bicubic(axis, 4, axis, 4, zs, 2.0, 2.0, &result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(result == 42.0);
}

static void test_extreme_bicubic_spacings(void **state) {
    (void)state;
    const lmmc_real_t regular[] = {0.0, 1.0, 2.0, 4.0};
    const lmmc_real_t extreme[] = {-DBL_MAX, -1.0, 1.0, DBL_MAX};
    const lmmc_real_t tiny[] = {0.0, DBL_MIN, 2.0 * DBL_MIN, 3.0 * DBL_MIN};
    lmmc_real_t zs[16], result = 42.0;
    size_t i;
    for (i = 0; i < 16; ++i) zs[i] = 7.0;
    assert_int_equal(lmmc_interp_bicubic(extreme, 4, regular, 4, zs, 0.0, 1.5, &result),
                     LMMC_STATUS_OK);
    assert_true(fabs(result - 7.0) <= 1e-12);
    result = 42.0;
    assert_int_equal(lmmc_interp_bicubic(tiny, 4, regular, 4, zs, 1.5 * DBL_MIN, 1.5, &result),
                     LMMC_STATUS_OK);
    assert_true(fabs(result - 7.0) <= 1e-12);
}

static void test_output_alias_and_overflow(void **state) {
    (void)state;
    const lmmc_real_t axis[] = {0.0, 1.0, 2.0, 4.0};
    lmmc_real_t zs[16], result = 42.0;
    size_t i;
    for (i = 0; i < 16; ++i)
        zs[i] = 3.0;
    assert_true(lmmc_interp_bicubic(axis, 4, axis, 4, zs, 1.5, 1.5, &zs[5]) ==
                LMMC_STATUS_OK);
    assert_true(isfinite(zs[5]) && fabs(zs[5] - 3.0) <= 1e-12);
    for (i = 0; i < 16; ++i)
        zs[i] = DBL_MAX;
    assert_true(lmmc_interp_bicubic(axis, 4, axis, 4, zs, 1.5, 1.5, &result) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(result == 42.0);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_nonuniform_affine_regression),
        cmocka_unit_test(test_constant_reproduction),
        cmocka_unit_test(test_affine_reproduction),
        cmocka_unit_test(test_bicubic_reproduction),
        cmocka_unit_test(test_exact_nodes_at_extreme_spacing),
        cmocka_unit_test(test_bilinear_extreme_span),
        cmocka_unit_test(test_boundary_stencil_locality),
        cmocka_unit_test(test_invalid_axes_preserve_output),
        cmocka_unit_test(test_invalid_queries_and_values),
        cmocka_unit_test(test_extreme_bicubic_spacings),
        cmocka_unit_test(test_output_alias_and_overflow),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
