#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"

struct test_fixture {
    lmmc_rng_t *rng;
};

static int setup(void **state) {
    *state = test_calloc(1, sizeof(struct test_fixture));
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_rng_destroy(fixture->rng);
    test_free(fixture);
    return 0;
}

static int teardown_default(void **state) {
    (void)state;
    lmmc_std_random_default_deinit();
    return 0;
}

static void test_uniform_reproducibility(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t out2 = 0;
    assert_true(lmmc_rng_create(&fixture->rng) == LMMC_STATUS_OK);

    assert_true(lmmc_std_random_seed(fixture->rng, 42) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_rand(fixture->rng, &out) == LMMC_STATUS_OK);
    assert_true(isfinite((double)out));
    assert_true(out >= (lmmc_real_t)0);
    assert_true(out < (lmmc_real_t)1);
    assert_true(lmmc_std_random_seed(fixture->rng, 42) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_rand(fixture->rng, &out2) == LMMC_STATUS_OK);
    assert_true(isfinite((double)out2));
    assert_true(out2 >= (lmmc_real_t)0);
    assert_true(out2 < (lmmc_real_t)1);
    assert_true(close_real(out, out2));
}

static void test_invalid_arguments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t values[] = {1, 2, 3, 4};
    int64_t randint_out = 0;
    assert_true(lmmc_rng_create(&fixture->rng) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_seed(fixture->rng, 42) == LMMC_STATUS_OK);

    assert_true(lmmc_std_random_seed(NULL, 42) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_rand(NULL, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_rand(fixture->rng, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_randint(NULL, 1, 3, &randint_out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_randint(fixture->rng, 1, 3, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_randint(fixture->rng, 3, 1, &randint_out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_normal(NULL, 0, 1, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_normal(fixture->rng, 0, 1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_choice(NULL, values, 4, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_choice(fixture->rng, NULL, 4, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_choice(fixture->rng, values, 4, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_integer_range_reproducibility(void **state) {
    struct test_fixture *fixture = *state;
    int64_t randint_out = 0;
    int64_t randint_out2 = 0;
    assert_true(lmmc_rng_create(&fixture->rng) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_seed(fixture->rng, 42) == LMMC_STATUS_OK);

    assert_true(lmmc_std_random_randint(fixture->rng, 1, 3, &randint_out) == LMMC_STATUS_OK);
    assert_true(randint_out >= 1);
    assert_true(randint_out <= 3);
    assert_true(lmmc_std_random_seed(fixture->rng, 2718) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_randint(fixture->rng, 1, 3, &randint_out) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_seed(fixture->rng, 2718) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_randint(fixture->rng, 1, 3, &randint_out2) == LMMC_STATUS_OK);
    assert_true(randint_out == randint_out2);
}

static void test_normal_reproducibility_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t out2 = 0;
    assert_true(lmmc_rng_create(&fixture->rng) == LMMC_STATUS_OK);

    assert_true(lmmc_std_random_seed(fixture->rng, 314) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_normal(fixture->rng, 10, 2, &out) == LMMC_STATUS_OK);
    assert_true(isfinite((double)out));
    assert_true(lmmc_std_random_seed(fixture->rng, 314) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_normal(fixture->rng, 10, 2, &out2) == LMMC_STATUS_OK);
    assert_true(close_real(out, out2));
    assert_true(lmmc_std_random_normal(fixture->rng, 0, 0, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_normal(fixture->rng, 0, -1, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_normal(fixture->rng, INFINITY, 1, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_random_normal(fixture->rng, 0, NAN, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_choice_reproducibility_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t out2 = 0;
    lmmc_real_t values[] = {1, 2, 3, 4};
    lmmc_real_t nonfinite_values[] = {1, NAN, 3};
    assert_true(lmmc_rng_create(&fixture->rng) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_seed(fixture->rng, 314) == LMMC_STATUS_OK);

    assert_true(lmmc_std_random_choice(fixture->rng, values, 4, &out) == LMMC_STATUS_OK);
    assert_true(out >= 1);
    assert_true(out <= 4);
    assert_true(lmmc_std_random_seed(fixture->rng, 1618) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_choice(fixture->rng, values, 4, &out) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_seed(fixture->rng, 1618) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_choice(fixture->rng, values, 4, &out2) == LMMC_STATUS_OK);
    assert_true(close_real(out, out2));
    assert_true(lmmc_std_random_choice(fixture->rng, nonfinite_values, 3, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_random_choice(fixture->rng, values, 0, &out) == LMMC_STATUS_EMPTY_INPUT);
}

static void test_default_diagnostics(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_real_t values[] = {1, 2, 3, 4};
    lmmc_real_t nonfinite_values[] = {1, NAN, 3};
    int64_t randint_out = 0;

    assert_true(lmmc_std_random_default_rand(NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_default_randint(4, 1, &randint_out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_default_normal(0, 0, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_random_default_choice(values, 0, &out) == LMMC_STATUS_EMPTY_INPUT);
    assert_true(lmmc_std_random_default_choice(nonfinite_values, 3, &out) ==
                LMMC_STATUS_NUMERICAL_FAILURE);

    lmmc_std_random_default_deinit();
}

static void test_default_reproducibility(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_real_t out2 = 0;

    assert_true(lmmc_std_random_default_seed(2718) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_default_rand(&out) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_default_seed(2718) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_default_rand(&out2) == LMMC_STATUS_OK);
    assert_true(close_real(out, out2));

    lmmc_std_random_default_deinit();
}

static void test_default_distribution_values(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_real_t values[] = {1, 2, 3, 4};
    int64_t randint_out = 0;

    assert_true(lmmc_std_random_default_seed(31415) == LMMC_STATUS_OK);
    assert_true(lmmc_std_random_default_randint(2, 5, &randint_out) == LMMC_STATUS_OK);
    assert_true(randint_out >= 2);
    assert_true(randint_out <= 5);
    assert_true(lmmc_std_random_default_normal(0, 1, &out) == LMMC_STATUS_OK);
    assert_true(isfinite((double)out));
    assert_true(lmmc_std_random_default_choice(values, 4, &out) == LMMC_STATUS_OK);
    assert_true(!((out != 1 && out != 2 && out != 3 && out != 4)));
    lmmc_std_random_default_deinit();
    lmmc_std_random_default_deinit();
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_uniform_reproducibility, setup, teardown),
        cmocka_unit_test_setup_teardown(test_invalid_arguments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_integer_range_reproducibility, setup, teardown),
        cmocka_unit_test_setup_teardown(test_normal_reproducibility_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_choice_reproducibility_errors, setup, teardown),
        cmocka_unit_test_teardown(test_default_diagnostics, teardown_default),
        cmocka_unit_test_teardown(test_default_reproducibility, teardown_default),
        cmocka_unit_test_teardown(test_default_distribution_values, teardown_default),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
