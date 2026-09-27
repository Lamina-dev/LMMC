#include <stdlib.h>
/**
 * @file test_tensor_extended.c
 * 针对 LMMC 中 tensor extended 相关接口的单元测试。
 */
#include <math.h>
#include <stdint.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

typedef struct {
    lmmc_tensor3_t tensor3_b_with_zero;
    lmmc_tensor3_t tensor3_ones;
    lmmc_tensor3_t tensor3_extreme;
    lmmc_tensor3_t a;
    lmmc_tensor3_t b;
    lmmc_tensor3_t out;
    lmmc_tensor3_t scaled;
    lmmc_mat_t sum_ax1;
    lmmc_mat_t sum_ax0;
    lmmc_mat_t sum_ax2;
    lmmc_tensor3_t t1x1x1;
    lmmc_tensor3_t t1_b;
    lmmc_tensor3_t t1_out;
    lmmc_tensor3_t zeros;
} test_fixture_t;

static int teardown(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_destroy(&fixture->tensor3_extreme);
    lmmc_tensor3_destroy(&fixture->tensor3_ones);
    lmmc_tensor3_destroy(&fixture->tensor3_b_with_zero);
    lmmc_tensor3_destroy(&fixture->zeros);
    lmmc_tensor3_destroy(&fixture->t1_out);
    lmmc_tensor3_destroy(&fixture->t1_b);
    lmmc_tensor3_destroy(&fixture->t1x1x1);
    lmmc_mat_destroy(&fixture->sum_ax2);
    lmmc_mat_destroy(&fixture->sum_ax0);
    lmmc_mat_destroy(&fixture->sum_ax1);
    lmmc_tensor3_destroy(&fixture->scaled);
    lmmc_tensor3_destroy(&fixture->out);
    lmmc_tensor3_destroy(&fixture->b);
    lmmc_tensor3_destroy(&fixture->a);
    free(fixture);
    *state = NULL;
    return 0;
}

static lmmc_status_t create_tensor_operands(lmmc_tensor3_t *a, lmmc_tensor3_t *b, lmmc_tensor3_t *out) {
    lmmc_status_t st = lmmc_tensor3_create(2, 3, 4, a);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_tensor3_create(2, 3, 4, b);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_tensor3_create(2, 3, 4, out);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                double val = (double)(i * 12 + j * 4 + k + 1);
                st = lmmc_tensor3_set(a, i, j, k, val);
                if (st != LMMC_STATUS_OK) {
                    return st;
                }
                st = lmmc_tensor3_set(b, i, j, k, val * 2.0);
                if (st != LMMC_STATUS_OK) {
                    return st;
                }
            }
        }
    }
    return LMMC_STATUS_OK;
}


static void test_tensor_addition_and_product(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *a = &fixture->a;
    lmmc_tensor3_t *b = &fixture->b;
    lmmc_tensor3_t *out = &fixture->out;
    lmmc_status_t st;

    st = lmmc_tensor3_add(a, b, out);
    assert_false(st != LMMC_STATUS_OK);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                double va, vb, vo;
                lmmc_tensor3_get(a, i, j, k, &va);
                lmmc_tensor3_get(b, i, j, k, &vb);
                lmmc_tensor3_get(out, i, j, k, &vo);
                assert_true(lmmc_test_nearly_equal(vo, va + vb, 1e-12));
            }
        }
    }

    st = lmmc_tensor3_mul(a, b, out);
    assert_false(st != LMMC_STATUS_OK);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                double va, vb, vo;
                lmmc_tensor3_get(a, i, j, k, &va);
                lmmc_tensor3_get(b, i, j, k, &vb);
                lmmc_tensor3_get(out, i, j, k, &vo);
                assert_true(lmmc_test_nearly_equal(vo, va * vb, 1e-12));
            }
        }
    }
}

static void test_tensor_zero_divisor(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *a = &fixture->a;
    lmmc_tensor3_t *out = &fixture->out;
    lmmc_status_t st;

    st = lmmc_tensor3_create(2, 3, 4, &fixture->tensor3_b_with_zero);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_fill(&fixture->tensor3_b_with_zero, 2.0);
    assert_false(st != LMMC_STATUS_OK);

    st = lmmc_tensor3_set(&fixture->tensor3_b_with_zero, 0, 0, 0, 0.0);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_div(a, &fixture->tensor3_b_with_zero, out);
    assert_false(st != LMMC_STATUS_NUMERICAL_FAILURE);
    lmmc_tensor3_destroy(&fixture->tensor3_b_with_zero);
}

static void test_tensor_scaling_identities(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *a = &fixture->a;
    lmmc_tensor3_t *scaled = &fixture->scaled;
    lmmc_status_t st;

    st = lmmc_tensor3_create(2, 3, 4, scaled);
    assert_false(st != LMMC_STATUS_OK);

    st = lmmc_tensor3_scale(a, 0.0, scaled);
    assert_false(st != LMMC_STATUS_OK);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                double vo;
                lmmc_tensor3_get(scaled, i, j, k, &vo);
                assert_true(lmmc_test_nearly_equal(vo, 0.0, 1e-12));
            }
        }
    }

    st = lmmc_tensor3_scale(a, 1.0, scaled);
    assert_false(st != LMMC_STATUS_OK);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                double va, vo;
                lmmc_tensor3_get(a, i, j, k, &va);
                lmmc_tensor3_get(scaled, i, j, k, &vo);
                assert_true(lmmc_test_nearly_equal(vo, va, 1e-12));
            }
        }
    }
}

static void test_tensor_middle_axis_sum(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *a = &fixture->a;
    lmmc_mat_t *sum_ax1 = &fixture->sum_ax1;
    lmmc_status_t st;

    st = lmmc_tensor3_sum_axis(a, 1, sum_ax1);
    assert_false(st != LMMC_STATUS_OK);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            double expected = 0.0;
            for (size_t j = 0; j < 3; ++j) {
                double v;
                lmmc_tensor3_get(a, i, j, k, &v);
                expected += v;
            }
            double actual = sum_ax1->data[i * sum_ax1->stride + k];
            assert_true(lmmc_test_nearly_equal(actual, expected, 1e-12));
        }
    }
}

static void test_tensor_first_two_axis_sums(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *a = &fixture->a;
    lmmc_mat_t *sum_ax0 = &fixture->sum_ax0;
    lmmc_status_t st;

    st = lmmc_tensor3_sum_axis(a, 0, sum_ax0);
    assert_false(st != LMMC_STATUS_OK);

    for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 4; ++k) {
            double v0, v1;
            lmmc_tensor3_get(a, 0, j, k, &v0);
            lmmc_tensor3_get(a, 1, j, k, &v1);
            double expected = v0 + v1;
            double actual = sum_ax0->data[j * sum_ax0->stride + k];
            assert_true(lmmc_test_nearly_equal(actual, expected, 1e-12));
        }
    }
}

static void test_tensor_last_axis_sum(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *a = &fixture->a;
    lmmc_mat_t *sum_ax2 = &fixture->sum_ax2;
    lmmc_status_t st;

    st = lmmc_tensor3_sum_axis(a, 2, sum_ax2);
    assert_false(st != LMMC_STATUS_OK);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < 4; ++k) {
                double v;
                lmmc_tensor3_get(a, i, j, k, &v);
                expected += v;
            }
            double actual = sum_ax2->data[i * sum_ax2->stride + j];
            assert_true(lmmc_test_nearly_equal(actual, expected, 1e-12));
        }
    }
}



static void test_scalar_tensor_arithmetic(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *t1x1x1 = &fixture->t1x1x1;
    lmmc_tensor3_t *t1_b = &fixture->t1_b;
    lmmc_tensor3_t *t1_out = &fixture->t1_out;
    lmmc_status_t st;

    st = lmmc_tensor3_create(1, 1, 1, t1_b);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_set(t1_b, 0, 0, 0, 8.0);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_create(1, 1, 1, t1_out);
    assert_false(st != LMMC_STATUS_OK);

    st = lmmc_tensor3_add(t1x1x1, t1_b, t1_out);
    assert_false(st != LMMC_STATUS_OK);
    {
        double v;
        lmmc_tensor3_get(t1_out, 0, 0, 0, &v);
        assert_true(lmmc_test_nearly_equal(v, 50.0, 1e-12));
    }

    st = lmmc_tensor3_mul(t1x1x1, t1_b, t1_out);
    assert_false(st != LMMC_STATUS_OK);
    {
        double v;
        lmmc_tensor3_get(t1_out, 0, 0, 0, &v);
        assert_true(lmmc_test_nearly_equal(v, 336.0, 1e-12));
    }

    st = lmmc_tensor3_scale(t1x1x1, 3.0, t1_out);
    assert_false(st != LMMC_STATUS_OK);
    {
        double v;
        lmmc_tensor3_get(t1_out, 0, 0, 0, &v);
        assert_true(lmmc_test_nearly_equal(v, 126.0, 1e-12));
    }
}

static void test_zero_tensor_reduction(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *zeros = &fixture->zeros;
    lmmc_status_t st;

    st = lmmc_tensor3_create(2, 3, 4, zeros);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_tensor3_fill(zeros, 0.0);
    assert_false(st != LMMC_STATUS_OK);
    {
        double norm;
        st = lmmc_tensor3_norm_fro(zeros, &norm);
        assert_false(st != LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(norm, 0.0, 1e-12));
    }
}

static void test_scalar_and_zero_tensor_reductions(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *t1x1x1 = &fixture->t1x1x1;
    lmmc_status_t st;

    {
        double s, mx, mn;
        st = lmmc_tensor3_sum(t1x1x1, &s);
        assert_false(st != LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(s, 42.0, 1e-12));
        st = lmmc_tensor3_max(t1x1x1, &mx);
        assert_false(st != LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(mx, 42.0, 1e-12));
        st = lmmc_tensor3_min(t1x1x1, &mn);
        assert_false(st != LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(mn, 42.0, 1e-12));
    }

    {
        double norm;
        st = lmmc_tensor3_norm_fro(t1x1x1, &norm);
        assert_false(st != LMMC_STATUS_OK);
        assert_true(lmmc_test_nearly_equal(norm, 42.0, 1e-12));
    }
}

static void test_tensor_norm_scaling(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_status_t st;

    {

        st = lmmc_tensor3_create(3, 4, 5, &fixture->tensor3_ones);
        assert_false(st != LMMC_STATUS_OK);
        st = lmmc_tensor3_fill(&fixture->tensor3_ones, 1.0);
        assert_false(st != LMMC_STATUS_OK);
        double norm;
        st = lmmc_tensor3_norm_fro(&fixture->tensor3_ones, &norm);
        assert_false(st != LMMC_STATUS_OK);
        double expected_norm = sqrt(3.0 * 4.0 * 5.0);
        assert_true(lmmc_test_nearly_equal(norm, expected_norm, 1e-12));
        lmmc_tensor3_destroy(&fixture->tensor3_ones);
    }

    {

        double norm = -1.0;

        st = lmmc_tensor3_create(1, 1, 2, &fixture->tensor3_extreme);
        assert_false(st != LMMC_STATUS_OK);
        fixture->tensor3_extreme.data[0] = 1.0e308;
        fixture->tensor3_extreme.data[1] = 1.0e308;
        st = lmmc_tensor3_norm_fro(&fixture->tensor3_extreme, &norm);
        assert_false(st != LMMC_STATUS_OK || !isfinite(norm) ||
                     fabs(norm / 1.0e308 - sqrt(2.0)) > 1.0e-15);
        fixture->tensor3_extreme.data[0] = 1.0e-300;
        fixture->tensor3_extreme.data[1] = 1.0e-300;
        st = lmmc_tensor3_norm_fro(&fixture->tensor3_extreme, &norm);
        assert_false(st != LMMC_STATUS_OK ||
                     fabs(norm / 1.0e-300 - sqrt(2.0)) > 1.0e-15);
        lmmc_tensor3_destroy(&fixture->tensor3_extreme);
    }
}

static void test_tensor_extrema_and_sum(void **state) {
    test_fixture_t *fixture = *state;
    lmmc_tensor3_t *a = &fixture->a;
    lmmc_status_t st;

    double sum_val, max_val, min_val;

    double expected_sum = 0.0;
    double expected_max = -1e300;
    double expected_min = 1e300;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                double v;
                lmmc_tensor3_get(a, i, j, k, &v);
                expected_sum += v;
                if (v > expected_max) {
                    expected_max = v;
                }
                if (v < expected_min) {
                    expected_min = v;
                }
            }
        }
    }

    st = lmmc_tensor3_sum(a, &sum_val);
    assert_false(st != LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(sum_val, expected_sum, 1e-10));

    st = lmmc_tensor3_max(a, &max_val);
    assert_false(st != LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(max_val, expected_max, 1e-12));

    st = lmmc_tensor3_min(a, &min_val);
    assert_false(st != LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(min_val, expected_min, 1e-12));
}


static int setup(void **state) {
    test_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    lmmc_status_t status = create_tensor_operands(&fixture->a, &fixture->b, &fixture->out);
    if (status != LMMC_STATUS_OK) {
        goto error;
    }
    status = lmmc_mat_create(3, 4, &fixture->sum_ax0);
    if (status != LMMC_STATUS_OK) {
        goto error;
    }
    status = lmmc_mat_create(2, 4, &fixture->sum_ax1);
    if (status != LMMC_STATUS_OK) {
        goto error;
    }
    status = lmmc_mat_create(2, 3, &fixture->sum_ax2);
    if (status != LMMC_STATUS_OK) {
        goto error;
    }
    status = lmmc_tensor3_create(1, 1, 1, &fixture->t1x1x1);
    if (status != LMMC_STATUS_OK) {
        goto error;
    }
    status = lmmc_tensor3_set(&fixture->t1x1x1, 0, 0, 0, 42.0);
    if (status != LMMC_STATUS_OK) {
        goto error;
    }
    return 0;

error:
    teardown(state);
    return status;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_tensor_addition_and_product, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_zero_divisor, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_scaling_identities, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_middle_axis_sum, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_first_two_axis_sums, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_last_axis_sum, setup, teardown),
        cmocka_unit_test_setup_teardown(test_scalar_tensor_arithmetic, setup, teardown),
        cmocka_unit_test_setup_teardown(test_zero_tensor_reduction, setup, teardown),
        cmocka_unit_test_setup_teardown(test_scalar_and_zero_tensor_reductions, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_norm_scaling, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_extrema_and_sum, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
