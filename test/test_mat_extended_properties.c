/**
 * @file test_mat_extended_properties.c
 * @brief 矩阵扩展运算的代数性质测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define PBT_ITERATIONS 100

struct test_fixture {
    struct test_property10_cross_product_orthogonality_resources {
        lmmc_vec_t a;
        lmmc_vec_t b;
        lmmc_vec_t c;
    } test_property10_cross_product_orthogonality;
    struct test_property11_cross_product_self_annihilation_resources {
        lmmc_vec_t a;
        lmmc_vec_t c;
    } test_property11_cross_product_self_annihilation;
    struct check_power_addition_resources {
        lmmc_mat_t matrices[5];
    } check_power_addition;
    struct check_right_division_resources {
        lmmc_mat_t a;
        lmmc_mat_t matrices[3];
    } check_right_division;
    struct test_property14_mat_norm_transpose_duality_resources {
        lmmc_mat_t A;
        lmmc_mat_t AT;
    } test_property14_mat_norm_transpose_duality;
    struct test_property15_mat_rank_upper_bound_resources {
        lmmc_mat_t A;
    } test_property15_mat_rank_upper_bound;
};

static double rand_double(double lo, double hi) {
    return ((double)rand() / RAND_MAX) * (hi - lo) + lo;
}

static void test_property10_cross_product_orthogonality(void **state) {
    struct test_fixture *fixture = *state;
    struct test_property10_cross_product_orthogonality_resources *resources = &fixture->test_property10_cross_product_orthogonality;
    int i;
    double eps = 1e-9;

    for (i = 0; i < PBT_ITERATIONS; i++) {

        lmmc_status_t st;
        double dot_ca, dot_cb;
        size_t j;

        st = lmmc_vec_create(3, &resources->a);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    vec_create(a) failed at iter %d\n", i);
        }
        st = lmmc_vec_create(3, &resources->b);
        assert_int_equal(st, LMMC_STATUS_OK);
        st = lmmc_vec_create(3, &resources->c);
        assert_int_equal(st, LMMC_STATUS_OK);

        for (j = 0; j < 3; j++) {
            resources->a.data[j] = rand_double(-10.0, 10.0);
            resources->b.data[j] = rand_double(-10.0, 10.0);
        }

        st = lmmc_vec_cross(&resources->a, &resources->b, &resources->c);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    vec_cross failed at iter %d, status=%d\n", i, (int)st);
        }

        st = lmmc_vec_dot(&resources->c, &resources->a, &dot_ca);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    vec_dot(c,a) failed at iter %d\n", i);
        }

        st = lmmc_vec_dot(&resources->c, &resources->b, &dot_cb);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    vec_dot(c,b) failed at iter %d\n", i);
        }

        if (!lmmc_test_nearly_equal(dot_ca, 0.0, eps)) {
            fail_msg("    dot(cross(a,b), a) != 0 at iter %d: got %g\n", i, dot_ca);
        }

        if (!lmmc_test_nearly_equal(dot_cb, 0.0, eps)) {
            fail_msg("    dot(cross(a,b), b) != 0 at iter %d: got %g\n", i, dot_cb);
        }

        lmmc_vec_destroy(&resources->a);
        lmmc_vec_destroy(&resources->b);
        lmmc_vec_destroy(&resources->c);
    }
}

static void test_property11_cross_product_self_annihilation(void **state) {
    struct test_fixture *fixture = *state;
    struct test_property11_cross_product_self_annihilation_resources *resources = &fixture->test_property11_cross_product_self_annihilation;
    int i;
    double eps = 1e-15;

    for (i = 0; i < PBT_ITERATIONS; i++) {

        lmmc_status_t st;
        size_t j;

        st = lmmc_vec_create(3, &resources->a);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    vec_create(a) failed at iter %d\n", i);
        }
        st = lmmc_vec_create(3, &resources->c);
        assert_int_equal(st, LMMC_STATUS_OK);

        for (j = 0; j < 3; j++) {
            resources->a.data[j] = rand_double(-10.0, 10.0);
        }

        st = lmmc_vec_cross(&resources->a, &resources->a, &resources->c);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    vec_cross(a,a) failed at iter %d, status=%d\n", i, (int)st);
        }

        for (j = 0; j < 3; j++) {
            if (!lmmc_test_nearly_equal(resources->c.data[j], 0.0, eps)) {
                fail_msg("    cross(a,a)[%" PRIuMAX "] != 0 at iter %d: got %g\n", (uintmax_t)(j), i, resources->c.data[j]);
            }
        }

        lmmc_vec_destroy(&resources->a);
        lmmc_vec_destroy(&resources->c);
    }
}
static void create_matrix_group(lmmc_mat_t *matrices, size_t count,
                                size_t rows, size_t cols) {
    for (size_t i = 0; i < count; ++i) {
        assert_int_equal(lmmc_mat_create(rows, cols, &matrices[i]), LMMC_STATUS_OK);
    }
}

static void destroy_matrix_group(lmmc_mat_t *matrices, size_t count) {
    for (size_t i = 0; i < count; ++i)
        lmmc_mat_destroy(&matrices[i]);
}

static void fill_random_matrix(lmmc_mat_t *matrix, double lo, double hi,
                               double diagonal_shift) {
    for (size_t r = 0; r < matrix->rows; ++r) {
        for (size_t c = 0; c < matrix->cols; ++c) {
            matrix->data[r * matrix->stride + c] = rand_double(lo, hi);
        }
        if (diagonal_shift != 0.0) {
            matrix->data[r * matrix->stride + r] += diagonal_shift;
        }
    }
}

static void check_relative_matrix(const lmmc_mat_t *actual,
                                  const lmmc_mat_t *expected, double eps) {
    for (size_t r = 0; r < actual->rows; ++r) {
        for (size_t c = 0; c < actual->cols; ++c) {
            double got = actual->data[r * actual->stride + c];
            double want = expected->data[r * expected->stride + c];
            if (!lmmc_test_nearly_equal(got, want, eps * fabs(want) + eps)) {
                fail_msg("matrix mismatch [%" PRIuMAX "][%" PRIuMAX "]: got %g, expected %g\n", (uintmax_t)(r), (uintmax_t)(c), got, want);
            }
        }
    }
}

static void check_power_addition(struct test_fixture *fixture, int m, int n) {
    struct check_power_addition_resources *resources = &fixture->check_power_addition;

    lmmc_mat_t *a = &resources->matrices[0];
    lmmc_mat_t *pow_mn = &resources->matrices[1];
    lmmc_mat_t *pow_m = &resources->matrices[2];
    lmmc_mat_t *pow_n = &resources->matrices[3];
    lmmc_mat_t *product = &resources->matrices[4];

    create_matrix_group(resources->matrices, 5, 2, 2);
    fill_random_matrix(a, -1.0, 1.0, 5.0);
    assert_int_equal(lmmc_mat_pow(a, m + n, pow_mn), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_pow(a, m, pow_m), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_pow(a, n, pow_n), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_mul(pow_m, pow_n, product), LMMC_STATUS_OK);
    check_relative_matrix(pow_mn, product, 1e-6);
    destroy_matrix_group(resources->matrices, 5);
}

static void test_property12_mat_pow_exponent_addition(void **state) {
    struct test_fixture *fixture = *state;
    for (int i = 0; i < PBT_ITERATIONS; ++i) {
        int m = (rand() % 3) + 1;
        int n = (rand() % 3) + 1;
        check_power_addition(fixture, m, n);
    }
}

static void check_right_division(struct test_fixture *fixture, size_t rows) {
    struct check_right_division_resources *resources = &fixture->check_right_division;

    lmmc_mat_t *b = &resources->matrices[0];
    lmmc_mat_t *x = &resources->matrices[1];
    lmmc_mat_t *recovered = &resources->matrices[2];

    assert_int_equal(lmmc_mat_create(2, 2, &resources->a), LMMC_STATUS_OK);
    create_matrix_group(resources->matrices, 3, rows, 2);
    fill_random_matrix(&resources->a, -1.0, 1.0, 5.0);
    fill_random_matrix(b, -5.0, 5.0, 0.0);
    assert_int_equal(lmmc_mat_rdiv(b, &resources->a, x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_mul(x, &resources->a, recovered), LMMC_STATUS_OK);
    check_relative_matrix(recovered, b, 1e-7);
    destroy_matrix_group(resources->matrices, 3);
    lmmc_mat_destroy(&resources->a);
}

static void test_property13_mat_rdiv_roundtrip(void **state) {
    struct test_fixture *fixture = *state;
    for (int i = 0; i < PBT_ITERATIONS; ++i) {
        size_t rows = (size_t)(rand() % 3) + 1;
        check_right_division(fixture, rows);
    }
}
static void test_property14_mat_norm_transpose_duality(void **state) {
    struct test_fixture *fixture = *state;
    struct test_property14_mat_norm_transpose_duality_resources *resources = &fixture->test_property14_mat_norm_transpose_duality;
    int i;
    double eps = 1e-12;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        size_t rows = (size_t)(rand() % 5) + 1;
        size_t cols = (size_t)(rand() % 5) + 1;

        lmmc_status_t st;
        double norm1_A, norm_inf_AT;
        size_t r, c;

        st = lmmc_mat_create(rows, cols, &resources->A);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    mat_create(A) failed at iter %d\n", i);
        }
        st = lmmc_mat_create(cols, rows, &resources->AT);
        assert_int_equal(st, LMMC_STATUS_OK);

        for (r = 0; r < rows; r++) {
            for (c = 0; c < cols; c++) {
                resources->A.data[r * resources->A.stride + c] = rand_double(-10.0, 10.0);
            }
        }

        st = lmmc_mat_transpose_to(&resources->A, &resources->AT);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    mat_transpose_to failed at iter %d, status=%d\n", i, (int)st);
        }

        st = lmmc_mat_norm1(&resources->A, &norm1_A);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    mat_norm1 failed at iter %d, status=%d\n", i, (int)st);
        }

        st = lmmc_mat_norm_inf(&resources->AT, &norm_inf_AT);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    mat_norm_inf failed at iter %d, status=%d\n", i, (int)st);
        }

        if (!lmmc_test_nearly_equal(norm1_A, norm_inf_AT, eps)) {
            fail_msg("    norm1(A) != norm_inf(A^T) at iter %d: %g vs %g\n",
                     i, norm1_A, norm_inf_AT);
        }

        if (norm1_A < 0.0 || norm_inf_AT < 0.0) {
            fail_msg("    Negative norm at iter %d: norm1=%g, norm_inf=%g\n",
                     i, norm1_A, norm_inf_AT);
        }

        lmmc_mat_destroy(&resources->A);
        lmmc_mat_destroy(&resources->AT);
    }
}

static void test_property15_mat_rank_upper_bound(void **state) {
    struct test_fixture *fixture = *state;
    struct test_property15_mat_rank_upper_bound_resources *resources = &fixture->test_property15_mat_rank_upper_bound;
    int i;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        size_t rows = (size_t)(rand() % 5) + 1;
        size_t cols = (size_t)(rand() % 5) + 1;
        size_t min_dim = rows < cols ? rows : cols;

        lmmc_status_t st;
        size_t rank;
        size_t r, c;

        st = lmmc_mat_create(rows, cols, &resources->A);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    mat_create(A) failed at iter %d\n", i);
        }

        for (r = 0; r < rows; r++) {
            for (c = 0; c < cols; c++) {
                resources->A.data[r * resources->A.stride + c] = rand_double(-10.0, 10.0);
            }
        }

        st = lmmc_mat_rank(&resources->A, -1.0, &rank);
        if (st != LMMC_STATUS_OK) {
            fail_msg("    mat_rank failed at iter %d, status=%d\n", i, (int)st);
        }

        if (rank > min_dim) {
            fail_msg("    rank(%" PRIuMAX " x %" PRIuMAX ") = %" PRIuMAX " > min(%" PRIuMAX ", %" PRIuMAX ") = %" PRIuMAX " at iter %d\n", (uintmax_t)(rows), (uintmax_t)(cols), (uintmax_t)(rank), (uintmax_t)(rows), (uintmax_t)(cols), (uintmax_t)(min_dim), i);
        }

        lmmc_mat_destroy(&resources->A);
    }
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->test_property10_cross_product_orthogonality.a);
    lmmc_vec_destroy(&fixture->test_property10_cross_product_orthogonality.b);
    lmmc_vec_destroy(&fixture->test_property10_cross_product_orthogonality.c);
    lmmc_vec_destroy(&fixture->test_property11_cross_product_self_annihilation.a);
    lmmc_vec_destroy(&fixture->test_property11_cross_product_self_annihilation.c);
    for (size_t i = 0; i < 5; ++i)
        lmmc_mat_destroy(&fixture->check_power_addition.matrices[i]);
    lmmc_mat_destroy(&fixture->check_right_division.a);
    for (size_t i = 0; i < 3; ++i)
        lmmc_mat_destroy(&fixture->check_right_division.matrices[i]);
    lmmc_mat_destroy(&fixture->test_property14_mat_norm_transpose_duality.A);
    lmmc_mat_destroy(&fixture->test_property14_mat_norm_transpose_duality.AT);
    lmmc_mat_destroy(&fixture->test_property15_mat_rank_upper_bound.A);
    free(fixture);
    *state = NULL;
    return 0;
}

static int group_setup(void **state) {
    (void)state;
    assert_int_equal(lmmc_init(), LMMC_STATUS_OK);
    return 0;
}

static int group_teardown(void **state) {
    (void)state;
    assert_int_equal(lmmc_deinit(), LMMC_STATUS_OK);
    return 0;
}

int main(void) {
    srand(12345);
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_property10_cross_product_orthogonality, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property11_cross_product_self_annihilation, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property12_mat_pow_exponent_addition, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property13_mat_rdiv_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property14_mat_norm_transpose_duality, setup, teardown),
        cmocka_unit_test_setup_teardown(test_property15_mat_rank_upper_bound, setup, teardown),
    };
    return cmocka_run_group_tests(tests, group_setup, group_teardown);
}
