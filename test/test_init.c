#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <threads.h>
#endif

#include "lmmc/lmmc.h"
#include "test_common.h"
#include "internal_test_hooks.h"

static int require_status(
    const char* label,
    lmmc_status_t actual,
    lmmc_status_t expected
) {
    if (actual == expected) return 1;
    fprintf(stderr, "%s: expected %s, got %s\n", label,
            lmmc_status_string(expected), lmmc_status_string(actual));
    return 0;
}

static int exercise_vector(void) {
    lmmc_vec_t vector = {0};
    lmmc_real_t norm = 0.0;
    if (lmmc_vec_create(2, &vector) != LMMC_STATUS_OK) return 0;
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
    lmmc_tensor_t tensor;
    lmmc_rng_t* rng;
} cross_thread_state_t;

static int cross_thread_destroy_body(cross_thread_state_t* state) {
    if (!require_status("destroyer init", lmmc_init(), LMMC_STATUS_OK)) return 1;
    lmmc_vec_destroy(&state->vector);
    lmmc_mat_destroy(&state->matrix);
    lmmc_tensor_destroy(&state->tensor);
    lmmc_rng_destroy(state->rng);
    state->rng = NULL;
    if (!require_status("destroyer deinit", lmmc_deinit(), LMMC_STATUS_OK)) return 1;
    return 0;
}

enum { RNG_THREAD_COUNT = 8 };

typedef struct {
    atomic_int ready;
    atomic_int go;
    int results[RNG_THREAD_COUNT];
} rng_concurrency_state_t;

typedef struct {
    rng_concurrency_state_t* state;
    int index;
} rng_concurrency_arg_t;

static int rng_concurrency_body(rng_concurrency_arg_t* argument) {
    lmmc_real_t uniform = 0.0;
    lmmc_real_t normal = 0.0;
    int failed = 0;
    if (lmmc_init() != LMMC_STATUS_OK) failed = 1;
    atomic_fetch_add(&argument->state->ready, 1);
    while (atomic_load(&argument->state->go) == 0) {
    }
    if (!failed &&
        lmmc_lsr_random_default_rand(&uniform) != LMMC_STATUS_OK) failed = 1;
    if (!failed &&
        lmmc_lsr_random_default_normal(0.0, 1.0, &normal) !=
            LMMC_STATUS_OK) failed = 1;
    if (!failed && (!(uniform >= 0.0 && uniform < 1.0) ||
                    !isfinite(normal))) failed = 1;
    if (!failed &&
        lmmc_lsr_random_default_seed(
            UINT64_C(0x123456789abcdef0) +
            (uint64_t)argument->index) != LMMC_STATUS_OK) failed = 1;
    lmmc_lsr_random_default_deinit();
    if (lmmc_deinit() != LMMC_STATUS_OK) failed = 1;
    argument->state->results[argument->index] = failed;
    return failed;
}

static int lifecycle_thread_body(void) {
    if (!require_status("thread initial underflow", lmmc_deinit(),
                        LMMC_STATUS_NOT_INITIALIZED)) return 1;
    if (!require_status("thread init", lmmc_init(), LMMC_STATUS_OK)) return 1;
    if (!exercise_vector()) return 1;
    if (!require_status("thread deinit", lmmc_deinit(), LMMC_STATUS_OK)) return 1;
    if (!require_status("thread final underflow", lmmc_deinit(),
                        LMMC_STATUS_NOT_INITIALIZED)) return 1;
    return 0;
}

#ifdef _WIN32
static DWORD WINAPI lifecycle_thread(LPVOID unused) {
    (void)unused;
    return (DWORD)lifecycle_thread_body();
}

static DWORD WINAPI cross_thread_destroy(LPVOID argument) {
    return (DWORD)cross_thread_destroy_body((cross_thread_state_t*)argument);
}

static DWORD WINAPI rng_concurrency_thread(LPVOID argument) {
    return (DWORD)rng_concurrency_body((rng_concurrency_arg_t*)argument);
}
#else
static int lifecycle_thread(void* unused) {
    (void)unused;
    return lifecycle_thread_body();
}

static int cross_thread_destroy(void* argument) {
    return cross_thread_destroy_body((cross_thread_state_t*)argument);
}

static int rng_concurrency_thread(void* argument) {
    return rng_concurrency_body((rng_concurrency_arg_t*)argument);
}
#endif

int main(void) {
#ifdef _WIN32
    HANDLE worker;
    DWORD worker_result = 1;
#else
    thrd_t worker;
    int worker_result = 1;
#endif
    cross_thread_state_t cross_thread = {0};
    rng_concurrency_state_t rng_state = {0};
    rng_concurrency_arg_t rng_args[RNG_THREAD_COUNT];
#ifdef _WIN32
    HANDLE rng_workers[RNG_THREAD_COUNT] = {0};
#else
    thrd_t rng_workers[RNG_THREAD_COUNT];
#endif

    if (!require_status("initial underflow", lmmc_deinit(),
                        LMMC_STATUS_NOT_INITIALIZED)) return 1;
    if (!require_status("first init", lmmc_init(), LMMC_STATUS_OK)) return 1;
    if (!require_status("nested init", lmmc_init(), LMMC_STATUS_OK)) return 1;
    if (!require_status("nested reset", lmmc_stack_reset(327680),
                        LMMC_STATUS_BUSY)) return 1;
    if (!require_status("nested deinit", lmmc_deinit(), LMMC_STATUS_OK)) return 1;
    if (!require_status("single reset", lmmc_stack_reset(327680),
                        LMMC_STATUS_OK)) return 1;
    {
        lmmc_vec_t live_vector = {0};
        if (lmmc_vec_create(2, &live_vector) != LMMC_STATUS_OK) return 1;
        if (!require_status("reset with live allocation",
                            lmmc_stack_reset(327680),
                            LMMC_STATUS_OK)) return 1;
        if (!require_status("deinit with live allocation",
                            lmmc_deinit(),
                            LMMC_STATUS_OK)) return 1;
        lmmc_vec_destroy(&live_vector);
        if (!require_status("reinit after persistent allocation",
                            lmmc_init(), LMMC_STATUS_OK)) return 1;
    }
    if (!require_status("reset after allocation release",
                        lmmc_stack_reset(327680),
                        LMMC_STATUS_OK)) return 1;
    if (!exercise_vector()) return 1;

#ifdef _WIN32
    worker = CreateThread(NULL, 0, lifecycle_thread, NULL, 0, NULL);
    if (worker == NULL || WaitForSingleObject(worker, INFINITE) != WAIT_OBJECT_0 ||
        !GetExitCodeThread(worker, &worker_result) || worker_result != 0) {
        if (worker != NULL) CloseHandle(worker);
        return 1;
    }
    CloseHandle(worker);
#else
    if (thrd_create(&worker, lifecycle_thread, NULL) != thrd_success) return 1;
    if (thrd_join(worker, &worker_result) != thrd_success || worker_result != 0) {
        return 1;
    }
#endif
    if (!exercise_vector()) return 1;

    if (lmmc_vec_create(2, &cross_thread.vector) != LMMC_STATUS_OK) return 1;
    if (lmmc_mat_create(2, 2, &cross_thread.matrix) != LMMC_STATUS_OK) return 1;
    if (lmmc_tensor3_create(2, 2, 2, &cross_thread.tensor) != LMMC_STATUS_OK) return 1;
    if (lmmc_rng_create(&cross_thread.rng) != LMMC_STATUS_OK) return 1;
#ifdef _WIN32
    worker = CreateThread(NULL, 0, cross_thread_destroy, &cross_thread, 0, NULL);
    if (worker == NULL || WaitForSingleObject(worker, INFINITE) != WAIT_OBJECT_0 ||
        !GetExitCodeThread(worker, &worker_result) || worker_result != 0) {
        if (worker != NULL) CloseHandle(worker);
        return 1;
    }
    CloseHandle(worker);
#else
    if (thrd_create(&worker, cross_thread_destroy, &cross_thread) != thrd_success) return 1;
    if (thrd_join(worker, &worker_result) != thrd_success || worker_result != 0) {
        return 1;
    }
#endif

    {
        int i;
        for (i = 0; i < RNG_THREAD_COUNT; ++i) {
            rng_args[i].state = &rng_state;
            rng_args[i].index = i;
#ifdef _WIN32
            rng_workers[i] = CreateThread(
                NULL, 0, rng_concurrency_thread, &rng_args[i], 0, NULL);
            if (rng_workers[i] == NULL) return 1;
#else
            if (thrd_create(&rng_workers[i], rng_concurrency_thread,
                            &rng_args[i]) != thrd_success) return 1;
#endif
        }
        while (atomic_load(&rng_state.ready) != RNG_THREAD_COUNT) {
        }
        atomic_store(&rng_state.go, 1);
        for (i = 0; i < RNG_THREAD_COUNT; ++i) {
#ifdef _WIN32
            DWORD result = 1;
            if (WaitForSingleObject(rng_workers[i], INFINITE) != WAIT_OBJECT_0 ||
                !GetExitCodeThread(rng_workers[i], &result) || result != 0) {
                CloseHandle(rng_workers[i]);
                return 1;
            }
            CloseHandle(rng_workers[i]);
#else
            int result = 1;
            if (thrd_join(rng_workers[i], &result) != thrd_success ||
                result != 0) return 1;
#endif
            if (rng_state.results[i] != 0) return 1;
        }
    }

    if (!require_status("zero reset", lmmc_stack_reset(0),
                        LMMC_STATUS_OK)) return 1;
    if (!require_status("final deinit", lmmc_deinit(), LMMC_STATUS_OK)) return 1;
    if (!require_status("post-final reset", lmmc_stack_reset(0),
                        LMMC_STATUS_NOT_INITIALIZED)) return 1;
    if (!require_status("post-final underflow", lmmc_deinit(),
                        LMMC_STATUS_NOT_INITIALIZED)) return 1;

    if (!require_status("reinit", lmmc_init(), LMMC_STATUS_OK)) return 1;
    if (!exercise_vector()) return 1;
    if (!require_status("reinit deinit", lmmc_deinit(), LMMC_STATUS_OK)) return 1;

    {
        lmmc_vec_t vector = {0};
        lmmc_mat_t matrix = {0};
        lmmc_tensor_t tensor = {0};
        lmmc_rng_t* rng = NULL;
        if (!require_status("survivor init", lmmc_init(), LMMC_STATUS_OK)) return 1;
        if (lmmc_vec_create(2, &vector) != LMMC_STATUS_OK) return 1;
        if (lmmc_mat_create(2, 2, &matrix) != LMMC_STATUS_OK) return 1;
        if (lmmc_tensor3_create(2, 2, 2, &tensor) != LMMC_STATUS_OK) return 1;
        if (lmmc_rng_create(&rng) != LMMC_STATUS_OK) return 1;
        if (!require_status("survivor deinit", lmmc_deinit(), LMMC_STATUS_OK)) return 1;
        if (!require_status("survivor reinit", lmmc_init(), LMMC_STATUS_OK)) return 1;
        lmmc_vec_destroy(&vector);
        lmmc_mat_destroy(&matrix);
        lmmc_tensor_destroy(&tensor);
        lmmc_rng_destroy(rng);
        if (!require_status("survivor final deinit", lmmc_deinit(),
                            LMMC_STATUS_OK)) return 1;
    }

    {
        lmmc_rng_t* rng = NULL;
        lmmc_memory_fail_after_for_test(0);
        if (!require_status("rng allocation failure",
                            lmmc_rng_create(&rng),
                            LMMC_STATUS_ALLOCATION_FAILED)) return 1;
        if (rng != NULL) return 1;
        lmmc_memory_fail_reset_for_test();
        if (!require_status("rng allocation retry",
                            lmmc_rng_create(&rng),
                            LMMC_STATUS_OK)) return 1;
        lmmc_rng_destroy(rng);
    }
    {
        const size_t dims[2] = {2, 2};
        lmmc_tensor_nd_t tensor = {0};
        lmmc_memory_fail_after_for_test(0);
        if (!require_status("tensor allocation failure",
                            lmmc_tensor_create(2, dims, &tensor),
                            LMMC_STATUS_ALLOCATION_FAILED)) return 1;
        if (tensor.data != NULL) return 1;
        lmmc_memory_fail_reset_for_test();
        if (!require_status("tensor allocation retry",
                            lmmc_tensor_create(2, dims, &tensor),
                            LMMC_STATUS_OK)) return 1;
        lmmc_tensor_nd_destroy(&tensor);
    }
    {
        const lmmc_real_t xs[3] = {0.0, 1.0, 2.0};
        const lmmc_real_t ys[3] = {0.0, 1.0, 4.0};
        lmmc_interp_cspline_t* spline = NULL;
        lmmc_memory_fail_after_for_test(0);
        if (!require_status("interpolation allocation failure",
                            lmmc_interp_cspline_create(xs, ys, 3, &spline),
                            LMMC_STATUS_ALLOCATION_FAILED)) return 1;
        if (spline != NULL) return 1;
        lmmc_memory_fail_reset_for_test();
        if (!require_status("interpolation allocation retry",
                            lmmc_interp_cspline_create(xs, ys, 3, &spline),
                            LMMC_STATUS_OK)) return 1;
        lmmc_interp_cspline_destroy(spline);
    }
    return 0;
}
