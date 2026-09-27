#include "internal/quadrature_internal.h"
#include "memory_bridge.h"

/** @brief 叶节点保存五个采样值，每个子节点复用其中三个。 */
typedef struct {
    lmmc_real_t x[5];
    lmmc_real_t f[5];
    lmmc_real_t value;
    lmmc_real_t error;
    size_t depth;
} lmmc_quad_leaf_t;

typedef struct {
    lmmc_real_t sum;
    lmmc_real_t correction;
} lmmc_quad_sum_t;

typedef struct {
    lmmc_quad_func_t func;
    void* user_data;
    lmmc_quad_leaf_t* heap;
    size_t size;
    size_t capacity;
    size_t num_evals;
    size_t max_depth;
    lmmc_quad_sum_t value;
    lmmc_quad_sum_t error;
} lmmc_quad_workspace_t;

static int lmmc_quad_sum_add(lmmc_quad_sum_t* acc, lmmc_real_t value)
{
    const lmmc_real_t sum = acc->sum + value;
    lmmc_real_t residual;
    if (!isfinite(sum)) {
        return 0;
    }
    if (fabs(acc->sum) >= fabs(value)) {
        residual = (acc->sum - sum) + value;
    } else {
        residual = (value - sum) + acc->sum;
    }
    acc->correction += residual;
    acc->sum = sum;
    return isfinite(acc->correction);
}

static int lmmc_quad_sample(
    lmmc_quad_workspace_t* work, lmmc_real_t x, lmmc_real_t* value)
{
    if (!isfinite(x) || work->num_evals == SIZE_MAX) {
        return 0;
    }
    *value = work->func(x, work->user_data);
    ++work->num_evals;
    return isfinite(*value);
}

static lmmc_real_t lmmc_quad_simpson_panel(
    const lmmc_quad_leaf_t* leaf, size_t first, size_t step)
{
    lmmc_real_t center, half_length;
    lmmc_quad_center_half_length(
        leaf->x[first], leaf->x[first + 2 * step], &center, &half_length);
    return (half_length / 3.0) *
        (leaf->f[first] + 4.0 * leaf->f[first + step] +
         leaf->f[first + 2 * step]);
}

static int lmmc_quad_leaf_estimate(lmmc_quad_leaf_t* leaf)
{
    const lmmc_real_t whole = lmmc_quad_simpson_panel(leaf, 0, 2);
    const lmmc_real_t left = lmmc_quad_simpson_panel(leaf, 0, 1);
    const lmmc_real_t right = lmmc_quad_simpson_panel(leaf, 2, 1);
    const lmmc_real_t combined = left + right;
    const lmmc_real_t delta = combined - whole;
    leaf->value = combined + delta / 15.0;
    leaf->error = fabs(delta) / 15.0;
    return isfinite(whole) && isfinite(left) && isfinite(right) &&
        isfinite(combined) && isfinite(leaf->value) && isfinite(leaf->error);
}

static int lmmc_quad_leaf_midpoints(lmmc_quad_leaf_t* leaf)
{
    leaf->x[1] = lmmc_quad_midpoint(leaf->x[0], leaf->x[2]);
    leaf->x[3] = lmmc_quad_midpoint(leaf->x[2], leaf->x[4]);
    return leaf->x[0] < leaf->x[1] && leaf->x[1] < leaf->x[2] &&
        leaf->x[2] < leaf->x[3] && leaf->x[3] < leaf->x[4];
}

static int lmmc_quad_leaf_finish(
    lmmc_quad_workspace_t* work, lmmc_quad_leaf_t* leaf)
{
    if (!lmmc_quad_sample(work, leaf->x[1], &leaf->f[1])) {
        return 0;
    }
    if (!lmmc_quad_sample(work, leaf->x[3], &leaf->f[3])) {
        return 0;
    }
    return lmmc_quad_leaf_estimate(leaf);
}

static int lmmc_quad_leaf_initial(
    lmmc_quad_workspace_t* work, lmmc_quad_leaf_t* leaf,
    lmmc_real_t a, lmmc_real_t b)
{
    leaf->depth = 0;
    leaf->x[0] = a;
    leaf->x[2] = lmmc_quad_midpoint(a, b);
    leaf->x[4] = b;
    if (!lmmc_quad_leaf_midpoints(leaf)) {
        return 0;
    }
    if (!lmmc_quad_sample(work, a, &leaf->f[0])) {
        return 0;
    }
    if (!lmmc_quad_sample(work, b, &leaf->f[4])) {
        return 0;
    }
    if (!lmmc_quad_sample(work, leaf->x[2], &leaf->f[2])) {
        return 0;
    }
    return lmmc_quad_leaf_finish(work, leaf);
}

static int lmmc_quad_leaf_can_split(
    const lmmc_quad_leaf_t* leaf, size_t max_depth)
{
    if (leaf->depth >= max_depth) {
        return 0;
    }
    /** @brief 表示精度检查同时防止 max_depth 为 SIZE_MAX 时深度溢出。 */
    for (size_t i = 0; i < 4; ++i) {
        const lmmc_real_t mid = lmmc_quad_midpoint(leaf->x[i], leaf->x[i + 1]);
        if (mid <= leaf->x[i] || mid >= leaf->x[i + 1]) {
            return 0;
        }
    }
    return 1;
}

static int lmmc_quad_leaf_child(
    lmmc_quad_workspace_t* work, const lmmc_quad_leaf_t* parent,
    size_t first, lmmc_quad_leaf_t* child)
{
    child->depth = parent->depth + 1;
    for (size_t i = 0; i < 3; ++i) {
        child->x[2 * i] = parent->x[first + i];
        child->f[2 * i] = parent->f[first + i];
    }
    if (!lmmc_quad_leaf_midpoints(child)) {
        return 0;
    }
    return lmmc_quad_leaf_finish(work, child);
}

static int lmmc_quad_leaf_precedes(
    const lmmc_quad_leaf_t* left, const lmmc_quad_leaf_t* right)
{
    if (left->error != right->error) {
        return left->error > right->error;
    }
    return left->x[0] < right->x[0];
}

static int lmmc_quad_heap_reserve(lmmc_quad_workspace_t* work, size_t extra)
{
    size_t needed, capacity, bytes;
    lmmc_quad_leaf_t* heap;
    if (!lmmc_safe_add_size(work->size, extra, &needed)) {
        return 0;
    }
    if (needed <= work->capacity) {
        return 1;
    }
    capacity = work->capacity == 0 ? 16 : work->capacity;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }
    if (!lmmc_safe_mul_size(capacity, sizeof(*heap), &bytes)) {
        return 0;
    }
    heap = (lmmc_quad_leaf_t*)lmmc_memory_realloc(work->heap, bytes);
    if (heap == NULL) {
        return 0;
    }
    work->heap = heap;
    work->capacity = capacity;
    return 1;
}

static void lmmc_quad_heap_push(
    lmmc_quad_workspace_t* work, const lmmc_quad_leaf_t* leaf)
{
    size_t pos = work->size++;
    while (pos != 0) {
        const size_t parent = (pos - 1) / 2;
        if (!lmmc_quad_leaf_precedes(leaf, &work->heap[parent])) break;
        work->heap[pos] = work->heap[parent];
        pos = parent;
    }
    work->heap[pos] = *leaf;
}

static lmmc_quad_leaf_t lmmc_quad_heap_pop(lmmc_quad_workspace_t* work)
{
    const lmmc_quad_leaf_t result = work->heap[0];
    const lmmc_quad_leaf_t last = work->heap[--work->size];
    size_t pos = 0;
    /** @brief pos < size / 2 保证 2*pos+1 可表示。 */
    while (pos < work->size / 2) {
        size_t child = 2 * pos + 1;
        if (child + 1 < work->size &&
            lmmc_quad_leaf_precedes(&work->heap[child + 1], &work->heap[child])) {
            ++child;
        }
        if (!lmmc_quad_leaf_precedes(&work->heap[child], &last)) break;
        work->heap[pos] = work->heap[child];
        pos = child;
    }
    if (work->size != 0) work->heap[pos] = last;
    return result;
}

static int lmmc_quad_totals_add(
    lmmc_quad_workspace_t* work, const lmmc_quad_leaf_t* leaf,
    lmmc_real_t sign)
{
    return lmmc_quad_sum_add(&work->value, sign * leaf->value) &&
        lmmc_quad_sum_add(&work->error, sign * leaf->error);
}

static lmmc_status_t lmmc_quad_refine(lmmc_quad_workspace_t* work)
{
    lmmc_quad_leaf_t parent = lmmc_quad_heap_pop(work);
    lmmc_quad_leaf_t left, right;
    int split_left, split_right;
    if (work->num_evals > SIZE_MAX - 4) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (!lmmc_quad_leaf_child(work, &parent, 0, &left) ||
        !lmmc_quad_leaf_child(work, &parent, 2, &right)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    split_left = lmmc_quad_leaf_can_split(&left, work->max_depth);
    split_right = lmmc_quad_leaf_can_split(&right, work->max_depth);
    if (!lmmc_quad_heap_reserve(work, (size_t)split_left + (size_t)split_right)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    if (!lmmc_quad_totals_add(work, &parent, -1.0) ||
        !lmmc_quad_totals_add(work, &left, 1.0) ||
        !lmmc_quad_totals_add(work, &right, 1.0)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    /** @brief 受深度或表示精度限制的叶节点仍计入补偿总和。 */
    if (split_left) {
        lmmc_quad_heap_push(work, &left);
    }
    if (split_right) {
        lmmc_quad_heap_push(work, &right);
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_quad_global_integral(
    lmmc_quad_workspace_t* work, const lmmc_quad_leaf_t* root,
    lmmc_real_t abs_tol, lmmc_real_t rel_tol,
    lmmc_quad_result_t* result)
{
    for (;;) {
        lmmc_real_t tolerance;
        lmmc_status_t status;
        result->value = work->value.sum + work->value.correction;
        result->error = work->error.sum + work->error.correction;
        result->num_evals = work->num_evals;
        tolerance = lmmc_max(abs_tol, rel_tol * fabs(result->value));
        if (!isfinite(result->value) || !isfinite(result->error) ||
            result->error < 0.0 || !isfinite(tolerance)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (result->error <= tolerance) {
            return LMMC_STATUS_OK;
        }
        if (root != NULL) {
            if (lmmc_quad_leaf_can_split(root, work->max_depth)) {
                if (!lmmc_quad_heap_reserve(work, 1)) {
                    return LMMC_STATUS_ALLOCATION_FAILED;
                }
                lmmc_quad_heap_push(work, root);
            }
            root = NULL;
        }
        if (work->size == 0) {
            return LMMC_STATUS_WARNING_MAX_DEPTH;
        }
        status = lmmc_quad_refine(work);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
    }
}

static int lmmc_quad_adaptive_arguments(
    lmmc_real_t a, lmmc_real_t b, lmmc_real_t abs_tol, lmmc_real_t rel_tol)
{
    if (!isfinite(a) || !isfinite(b) || a >= b) {
        return 0;
    }
    if (!isfinite(abs_tol) || !isfinite(rel_tol)) {
        return 0;
    }
    return abs_tol >= 0.0 && rel_tol >= 0.0;
}

lmmc_status_t lmmc_quad_adaptive(
    lmmc_quad_func_t func, void* user_data,
    lmmc_real_t a, lmmc_real_t b,
    lmmc_real_t abs_tol, lmmc_real_t rel_tol,
    size_t max_depth, lmmc_quad_result_t* out_result)
{
    lmmc_quad_workspace_t work = {0};
    lmmc_quad_leaf_t root;
    lmmc_quad_result_t result;
    lmmc_status_t status;
    if (func == NULL || out_result == NULL ||
        !lmmc_quad_adaptive_arguments(a, b, abs_tol, rel_tol)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    work.func = func;
    work.user_data = user_data;
    work.max_depth = max_depth;
    if (!lmmc_quad_leaf_initial(&work, &root, a, b) ||
        !lmmc_quad_totals_add(&work, &root, 1.0)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_quad_global_integral(&work, &root, abs_tol, rel_tol, &result);
    lmmc_memory_free(work.heap);
    if (status == LMMC_STATUS_OK || status == LMMC_STATUS_WARNING_MAX_DEPTH) {
        *out_result = result;
    }
    return status;
}
