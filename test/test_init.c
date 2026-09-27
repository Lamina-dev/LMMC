#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "lmmc/lmmc.h"
#include "test_common.h"
#include "internal_test_hooks.h"

static int exercise_vector(void) {
    lmmc_vec_t vector = {0};
    lmmc_real_t norm = 0.0;
    if (lmmc_vec_create(2, &vector) != LMMC_STATUS_OK) {
        return 0;
    }
    vector.data[0] = 3.0;
    vector.data[1] = 4.0;
    if (lmmc_vec_norm2(&vector, &norm) != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&vector);
        return 0;
    }
    lmmc_vec_destroy(&vector);
    return lmmc_test_nearly_equal(norm, 5.0, 1e-12);
}

typedef struct {
    lmmc_vec_t vector;
    lmmc_mat_t matrix;
    lmmc_tensor3_t tensor;
    lmmc_rng_t *rng;
} cross_thread_state_t;

static int cross_thread_destroy_body(cross_thread_state_t *state) {
    if (!(lmmc_init() == LMMC_STATUS_OK)) {
        return 1;
    }
    lmmc_vec_destroy(&state->vector);
    lmmc_mat_destroy(&state->matrix);
    lmmc_tensor3_destroy(&state->tensor);
    lmmc_rng_destroy(state->rng);
    state->rng = NULL;
    if (!(lmmc_deinit() == LMMC_STATUS_OK)) {
        return 1;
    }
    return 0;
}

enum { RNG_THREAD_COUNT = 8 };

typedef struct {
    atomic_int ready;
    atomic_int go;
    int results[RNG_THREAD_COUNT];
} rng_concurrency_state_t;

typedef struct {
    rng_concurrency_state_t *state;
    int index;
} rng_concurrency_arg_t;

static int exercise_default_rng(int index) {
    lmmc_real_t uniform = 0.0;
    lmmc_real_t normal = 0.0;
    if (lmmc_std_random_default_rand(&uniform) != LMMC_STATUS_OK) {
        return 1;
    }
    if (lmmc_std_random_default_normal(0.0, 1.0, &normal) != LMMC_STATUS_OK) {
        return 1;
    }
    if (!(uniform >= 0.0 && uniform < 1.0) || !isfinite(normal)) {
        return 1;
    }
    return lmmc_std_random_default_seed(
               UINT64_C(0x123456789abcdef0) + (uint64_t)index) != LMMC_STATUS_OK;
}

static int rng_concurrency_body(rng_concurrency_arg_t *argument) {
    int failed = 0;
    if (lmmc_init() != LMMC_STATUS_OK) {
        failed = 1;
    }
    atomic_fetch_add(&argument->state->ready, 1);
    while (atomic_load(&argument->state->go) == 0) {
    }
    if (!failed) {
        failed = exercise_default_rng(argument->index);
    }
    lmmc_std_random_default_deinit();
    if (lmmc_deinit() != LMMC_STATUS_OK) {
        failed = 1;
    }
    argument->state->results[argument->index] = failed;
    return failed;
}

static int lifecycle_thread_body(void) {
    if (!(lmmc_deinit() == LMMC_STATUS_NOT_INITIALIZED)) {
        return 1;
    }
    if (!(lmmc_init() == LMMC_STATUS_OK)) {
        return 1;
    }
    if (!exercise_vector()) {
        return 1;
    }
    if (!(lmmc_deinit() == LMMC_STATUS_OK)) {
        return 1;
    }
    if (!(lmmc_deinit() == LMMC_STATUS_NOT_INITIALIZED)) {
        return 1;
    }
    return 0;
}

#ifdef _WIN32
static DWORD WINAPI lifecycle_thread(LPVOID unused) {
    (void)unused;
    return (DWORD)lifecycle_thread_body();
}

static DWORD WINAPI cross_thread_destroy(LPVOID argument) {
    return (DWORD)cross_thread_destroy_body((cross_thread_state_t *)argument);
}

static DWORD WINAPI rng_concurrency_thread(LPVOID argument) {
    return (DWORD)rng_concurrency_body((rng_concurrency_arg_t *)argument);
}
#else
static int thread_failure_token;

static void *lifecycle_thread(void *unused) {
    (void)unused;
    return lifecycle_thread_body() == 0 ? NULL : &thread_failure_token;
}

static void *cross_thread_destroy(void *argument) {
    return cross_thread_destroy_body((cross_thread_state_t *)argument) == 0
               ? NULL
               : &thread_failure_token;
}

static void *rng_concurrency_thread(void *argument) {
    return rng_concurrency_body((rng_concurrency_arg_t *)argument) == 0
               ? NULL
               : &thread_failure_token;
}
#endif

typedef struct {
    cross_thread_state_t cross_thread;
    lmmc_tensor_nd_t nd;
    lmmc_interp_cspline_t *spline;
} lifecycle_fixture_t;

static int setup(void **state) {
    lifecycle_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static void test_nested_lifecycle(void **state) {
    (void)state;
    assert_true((lmmc_deinit() == LMMC_STATUS_NOT_INITIALIZED));
    assert_true((lmmc_init() == LMMC_STATUS_OK));
    assert_true((lmmc_init() == LMMC_STATUS_OK));
    assert_true((lmmc_stack_reset(327680) == LMMC_STATUS_BUSY));
    assert_true((lmmc_deinit() == LMMC_STATUS_OK));
    assert_true((lmmc_stack_reset(327680) == LMMC_STATUS_OK));
}

static void test_allocation_survives_deinit(void **state) {
    lifecycle_fixture_t *fixture = *state;
    (void)state;
    {

        assert_int_equal(lmmc_vec_create(2, &fixture->cross_thread.vector), LMMC_STATUS_OK);
        assert_true((lmmc_stack_reset(327680) == LMMC_STATUS_OK));
        assert_true((lmmc_deinit() == LMMC_STATUS_OK));
        lmmc_vec_destroy(&fixture->cross_thread.vector);
        assert_true((lmmc_init() == LMMC_STATUS_OK));
    }
    assert_true((lmmc_stack_reset(327680) == LMMC_STATUS_OK));
    assert_true(exercise_vector());
}

static void test_thread_local_lifecycle(void **state) {
    (void)state;
#ifdef _WIN32
    HANDLE worker;
    DWORD worker_result = 1;
#else
    pthread_t worker;
    void *worker_result = &thread_failure_token;
#endif
#ifdef _WIN32
    worker = CreateThread(NULL, 0, lifecycle_thread, NULL, 0, NULL);
    if (worker == NULL || WaitForSingleObject(worker, INFINITE) != WAIT_OBJECT_0 ||
        !GetExitCodeThread(worker, &worker_result) || worker_result != 0) {
        if (worker != NULL) {
            CloseHandle(worker);
        }
        fail();
    }
    CloseHandle(worker);
#else
    assert_false(pthread_create(&worker, NULL, lifecycle_thread, NULL) != 0);
    assert_false(pthread_join(worker, &worker_result) != 0 || worker_result != NULL);
#endif
    assert_true(exercise_vector());
}

static int run_cross_thread_destruction(cross_thread_state_t *cross_thread) {
#ifdef _WIN32
    HANDLE worker;
    DWORD worker_result = 1;
#else
    pthread_t worker;
    void *worker_result = &thread_failure_token;
#endif
#ifdef _WIN32
    worker = CreateThread(NULL, 0, cross_thread_destroy, cross_thread, 0, NULL);
    if (worker == NULL || WaitForSingleObject(worker, INFINITE) != WAIT_OBJECT_0 ||
        !GetExitCodeThread(worker, &worker_result) || worker_result != 0) {
        if (worker != NULL) {
            CloseHandle(worker);
        }
        return 1;
    }
    CloseHandle(worker);
#else
    worker_result = &thread_failure_token;
    if (pthread_create(
            &worker, NULL, cross_thread_destroy, cross_thread) != 0) {
        return 1;
    }
    if (pthread_join(worker, &worker_result) != 0 || worker_result != NULL) {
        return 1;
    }
#endif
    return 0;
}

static void test_cross_thread_destruction(void **state) {
    lifecycle_fixture_t *fixture = *state;
    (void)state;

    assert_int_equal(lmmc_vec_create(2, &fixture->cross_thread.vector), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(2, 2, &fixture->cross_thread.matrix), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor3_create(2, 2, 2, &fixture->cross_thread.tensor), LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_create(&fixture->cross_thread.rng), LMMC_STATUS_OK);
    assert_int_equal(run_cross_thread_destruction(&fixture->cross_thread), 0);
}

static void test_finalize_and_reinitialize(void **state) {
    (void)state;
    assert_true((lmmc_stack_reset(0) == LMMC_STATUS_OK));
    assert_true((lmmc_deinit() == LMMC_STATUS_OK));
    assert_true((lmmc_stack_reset(0) == LMMC_STATUS_NOT_INITIALIZED));
    assert_true((lmmc_deinit() == LMMC_STATUS_NOT_INITIALIZED));

    assert_true((lmmc_init() == LMMC_STATUS_OK));
    assert_true(exercise_vector());
    assert_true((lmmc_deinit() == LMMC_STATUS_OK));
}

static void test_resources_survive_reinitialization(void **state) {
    lifecycle_fixture_t *fixture = *state;
    (void)state;

    assert_true((lmmc_init() == LMMC_STATUS_OK));
    assert_int_equal(lmmc_vec_create(2, &fixture->cross_thread.vector), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(2, 2, &fixture->cross_thread.matrix), LMMC_STATUS_OK);
    assert_int_equal(lmmc_tensor3_create(2, 2, 2, &fixture->cross_thread.tensor), LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_create(&fixture->cross_thread.rng), LMMC_STATUS_OK);
    assert_true((lmmc_deinit() == LMMC_STATUS_OK));
    assert_true((lmmc_init() == LMMC_STATUS_OK));
    lmmc_vec_destroy(&fixture->cross_thread.vector);
    lmmc_mat_destroy(&fixture->cross_thread.matrix);
    lmmc_tensor3_destroy(&fixture->cross_thread.tensor);
    lmmc_rng_destroy(fixture->cross_thread.rng);
    fixture->cross_thread.rng = NULL;
    assert_true((lmmc_deinit() == LMMC_STATUS_OK));
}

static void test_rng_allocation_failure(void **state) {
    lifecycle_fixture_t *fixture = *state;
    (void)state;

    lmmc_memory_fail_after_for_test(0);
    assert_true((lmmc_rng_create(&fixture->cross_thread.rng) == LMMC_STATUS_ALLOCATION_FAILED));
    assert_false(fixture->cross_thread.rng != NULL);
    lmmc_memory_fail_reset_for_test();
    assert_true((lmmc_rng_create(&fixture->cross_thread.rng) == LMMC_STATUS_OK));
    lmmc_rng_destroy(fixture->cross_thread.rng);
    fixture->cross_thread.rng = NULL;
}

static void test_tensor_allocation_failure(void **state) {
    lifecycle_fixture_t *fixture = *state;
    (void)state;
    const size_t dims[2] = {2, 2};

    lmmc_memory_fail_after_for_test(0);
    assert_true((lmmc_tensor_nd_create(2, dims, &fixture->nd) == LMMC_STATUS_ALLOCATION_FAILED));
    assert_false(fixture->nd.data != NULL);
    lmmc_memory_fail_reset_for_test();
    assert_true((lmmc_tensor_nd_create(2, dims, &fixture->nd) == LMMC_STATUS_OK));
    lmmc_tensor_nd_destroy(&fixture->nd);
}

static void test_spline_allocation_failure(void **state) {
    lifecycle_fixture_t *fixture = *state;
    (void)state;
    const lmmc_real_t xs[3] = {0.0, 1.0, 2.0};
    const lmmc_real_t ys[3] = {0.0, 1.0, 4.0};

    lmmc_memory_fail_after_for_test(0);
    assert_true((lmmc_interp_cspline_create(xs, ys, 3, &fixture->spline) == LMMC_STATUS_ALLOCATION_FAILED));
    assert_false(fixture->spline != NULL);
    lmmc_memory_fail_reset_for_test();
    assert_true((lmmc_interp_cspline_create(xs, ys, 3, &fixture->spline) == LMMC_STATUS_OK));
    lmmc_interp_cspline_destroy(fixture->spline);
    fixture->spline = NULL;
}

static void test_rank_allocation_failure(void **state) {
    (void)state;
    lmmc_real_t data[4] = {1.0, 0.0, 0.0, 1.0};
    lmmc_mat_t matrix = {2, 2, 2, data, 0};
    size_t rank = 0;
    lmmc_memory_fail_after_for_test(0);
    assert_true((lmmc_std_linalg_rank(&matrix, &rank) == LMMC_STATUS_ALLOCATION_FAILED));
    lmmc_memory_fail_reset_for_test();
    assert_true((lmmc_std_linalg_rank(&matrix, &rank) == LMMC_STATUS_OK));
    assert_false(rank != 2);
}

#ifdef LMMC_DEBUG_LEAKS
static void test_leak_accounting(void **state) {
    lifecycle_fixture_t *fixture = *state;
    const size_t baseline = lmmc_debug_leaks_get_count();

    assert_int_equal(lmmc_vec_create(2, &fixture->cross_thread.vector), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(2, 2, &fixture->cross_thread.matrix), LMMC_STATUS_OK);
    assert_int_equal(lmmc_debug_leaks_get_count(), baseline + 2);
    lmmc_vec_destroy(&fixture->cross_thread.vector);
    lmmc_mat_destroy(&fixture->cross_thread.matrix);
    assert_int_equal(lmmc_debug_leaks_get_count(), baseline);
}
#endif

#ifdef _WIN32
typedef HANDLE rng_worker_t;
#else
typedef pthread_t rng_worker_t;
#endif

static void test_concurrent_default_rng(void **state) {
    (void)state;
    rng_concurrency_state_t rng_state = {0};
    rng_concurrency_arg_t rng_args[RNG_THREAD_COUNT];
    rng_worker_t rng_workers[RNG_THREAD_COUNT];
    int join_status[RNG_THREAD_COUNT] = {0};
    int started = 0;

    for (; started < RNG_THREAD_COUNT; ++started) {
        rng_args[started].state = &rng_state;
        rng_args[started].index = started;
#ifdef _WIN32
        rng_workers[started] = CreateThread(
            NULL, 0, rng_concurrency_thread, &rng_args[started], 0, NULL);
        if (rng_workers[started] == NULL)
            break;
#else
        if (pthread_create(&rng_workers[started], NULL,
                           rng_concurrency_thread, &rng_args[started]) != 0)
            break;
#endif
    }
    while (atomic_load(&rng_state.ready) != started) {
    }
    atomic_store(&rng_state.go, 1);

    /* Join every launched worker before assertions can unwind its arguments. */
    for (int i = 0; i < started; ++i) {
#ifdef _WIN32
        DWORD result = 1;
        DWORD wait_status = WaitForSingleObject(rng_workers[i], INFINITE);
        BOOL exit_status = GetExitCodeThread(rng_workers[i], &result);
        join_status[i] = wait_status != WAIT_OBJECT_0 || !exit_status || result != 0;
        CloseHandle(rng_workers[i]);
#else
        void *result = &thread_failure_token;
        int status = pthread_join(rng_workers[i], &result);
        join_status[i] = status != 0 || result != NULL;
#endif
    }
    assert_int_equal(started, RNG_THREAD_COUNT);
    for (int i = 0; i < started; ++i) {
        assert_int_equal(join_status[i], 0);
        assert_int_equal(rng_state.results[i], 0);
    }
}

static int teardown(void **state) {
    lifecycle_fixture_t *fixture = *state;
    lmmc_memory_fail_reset_for_test();
    lmmc_vec_destroy(&fixture->cross_thread.vector);
    lmmc_mat_destroy(&fixture->cross_thread.matrix);
    lmmc_tensor3_destroy(&fixture->cross_thread.tensor);
    lmmc_rng_destroy(fixture->cross_thread.rng);
    lmmc_tensor_nd_destroy(&fixture->nd);
    lmmc_interp_cspline_destroy(fixture->spline);
    free(fixture);
    *state = NULL;
    lmmc_std_random_default_deinit();
    while (lmmc_deinit() == LMMC_STATUS_OK) {
    }
    return 0;
}

static int setup_initialized(void **state) {
    setup(state);
    lmmc_status_t status = lmmc_init();
    if (status != LMMC_STATUS_OK) {
        free(*state);
        *state = NULL;
    }
    return status;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_nested_lifecycle, setup, teardown),
        cmocka_unit_test_setup_teardown(test_allocation_survives_deinit, setup_initialized, teardown),
        cmocka_unit_test_setup_teardown(test_thread_local_lifecycle, setup_initialized, teardown),
        cmocka_unit_test_setup_teardown(test_cross_thread_destruction, setup_initialized, teardown),
        cmocka_unit_test_setup_teardown(test_finalize_and_reinitialize, setup_initialized, teardown),
        cmocka_unit_test_setup_teardown(test_resources_survive_reinitialization, setup, teardown),
        cmocka_unit_test_setup_teardown(test_rng_allocation_failure, setup, teardown),
        cmocka_unit_test_setup_teardown(test_tensor_allocation_failure, setup, teardown),
        cmocka_unit_test_setup_teardown(test_spline_allocation_failure, setup, teardown),
        cmocka_unit_test_setup_teardown(test_rank_allocation_failure, setup, teardown),
#ifdef LMMC_DEBUG_LEAKS
        cmocka_unit_test_setup_teardown(test_leak_accounting, setup, teardown),
#endif
        cmocka_unit_test_setup_teardown(test_concurrent_default_rng, setup_initialized, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
