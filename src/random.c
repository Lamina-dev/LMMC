/**
 * @file random.c
 * @brief 伪随机数发生器与常用分布实现.
 */
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdatomic.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/random.h"
#include "lmmc/status.h"


struct lmmc_rng_t {
    uint64_t state[4];
};


static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}


static inline uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}


/**
 * @brief Generate a robust default seed by mixing time, stack address (ASLR),
 *        and a monotonically increasing atomic counter through SplitMix64.
 */
static uint64_t generate_default_seed(void) {
    static _Atomic uint64_t seed_counter = 0;

    uint64_t t = (uint64_t)time(NULL);
    /* Use address of a local variable for ASLR entropy */
    volatile int stack_var = 0;
    uint64_t addr = (uint64_t)(uintptr_t)&stack_var;
    uint64_t counter = atomic_fetch_add(&seed_counter, 1);

    /* Mix all sources via SplitMix64 finalizer */
    uint64_t mixed = t ^ addr ^ counter;
    return splitmix64_next(&mixed);
}


static inline uint64_t xoshiro256ss_next(uint64_t* s) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
}


lmmc_status_t lmmc_rng_create(lmmc_rng_t** out_rng) {
    lmmc_rng_t* rng;

    if (out_rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    rng = (lmmc_rng_t*)lmmc_alloc(sizeof(lmmc_rng_t));
    if (rng == NULL) {
        *out_rng = NULL;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /** 将平台熵混入默认种子,为各随机数发生器建立独立数据流. */
    lmmc_rng_seed(rng, generate_default_seed());

    *out_rng = rng;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_seed(lmmc_rng_t* rng, uint64_t seed) {
    uint64_t sm_state;

    if (rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    sm_state = seed;
    rng->state[0] = splitmix64_next(&sm_state);
    rng->state[1] = splitmix64_next(&sm_state);
    rng->state[2] = splitmix64_next(&sm_state);
    rng->state[3] = splitmix64_next(&sm_state);

    return LMMC_STATUS_OK;
}

void lmmc_rng_destroy(lmmc_rng_t* rng) {
    if (rng != NULL) {
        memset(rng->state, 0, sizeof(rng->state));
        lmmc_free(rng);
    }
}


lmmc_status_t lmmc_rng_clone(const lmmc_rng_t* src, lmmc_rng_t** out_rng) {
    lmmc_rng_t* copy;

    if (src == NULL || out_rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    copy = (lmmc_rng_t*)lmmc_alloc(sizeof(lmmc_rng_t));
    if (copy == NULL) {
        *out_rng = NULL;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    memcpy(copy->state, src->state, sizeof(src->state));
    *out_rng = copy;
    return LMMC_STATUS_OK;
}


/*
 * xoshiro256** jump polynomial constants.
 * Advances the state by 2^128 steps.
 * Reference: https://prng.di.unimi.it/xoshiro256starstar.c
 */
static const uint64_t JUMP_POLY[4] = {
    UINT64_C(0x180ec6d33cfd0aba),
    UINT64_C(0xd5a61266f0c9392c),
    UINT64_C(0xa9582618e03fc9aa),
    UINT64_C(0x39abdc4529b1661c)
};

/*
 * xoshiro256** long_jump polynomial constants.
 * Advances the state by 2^192 steps.
 * Reference: https://prng.di.unimi.it/xoshiro256starstar.c
 */
static const uint64_t LONG_JUMP_POLY[4] = {
    UINT64_C(0x76e15d3efefdcbbf),
    UINT64_C(0xc5004e441c522fb3),
    UINT64_C(0x77710069854ee241),
    UINT64_C(0x39109bb02acbe635)
};


/**
 * @brief Internal helper: apply a jump polynomial to the RNG state.
 */
static void rng_apply_jump_poly(uint64_t* s, const uint64_t* poly) {
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i, b;

    for (i = 0; i < 4; i++) {
        for (b = 0; b < 64; b++) {
            if (poly[i] & (UINT64_C(1) << b)) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            xoshiro256ss_next(s);
        }
    }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}


lmmc_status_t lmmc_rng_jump(lmmc_rng_t* rng) {
    if (rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    rng_apply_jump_poly(rng->state, JUMP_POLY);
    return LMMC_STATUS_OK;
}


lmmc_status_t lmmc_rng_long_jump(lmmc_rng_t* rng) {
    if (rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    rng_apply_jump_poly(rng->state, LONG_JUMP_POLY);
    return LMMC_STATUS_OK;
}


uint64_t lmmc_rng_next_u64(lmmc_rng_t* rng) {
    if (rng == NULL) {
        return 0;
    }
    return xoshiro256ss_next(rng->state);
}


static inline double u64_to_double01(uint64_t x) {
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}


lmmc_status_t lmmc_rng_uniform(
    lmmc_rng_t* rng,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t* out_value)
{
    double u;

    if (rng == NULL || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    u = u64_to_double01(xoshiro256ss_next(rng->state));
    *out_value = a + (b - a) * u;
    return LMMC_STATUS_OK;
}

/**
 * Ziggurat normal sampler (256 rectangles)
 *
 * Reference: Marsaglia & Tsang, "The Ziggurat Method for
 * Generating Random Variables", JSS 2000.
 *
 * Tables are built for the right half of the standard normal
 * (x >= 0). The sampler generates |x| and applies a random sign.
 *
 * Table layout:
 *   zig_xtab[0] = v/f(r)  (width of base strip including tail)
 *   zig_xtab[1] = r       (tail cutoff)
 *   zig_xtab[i] for i=2..255: decreasing x-coordinates
 *   zig_xtab[256] = 0     (peak of distribution)
 *
 * For 256 rectangles:
 *   r = 3.6541528853610088
 *   v = 0.00492867323399 (area of each rectangle in the half-normal)
 */

#define ZIG_N 256
#define ZIG_R 3.6541528853610088
#define ZIG_V 0.00492867323399

static double zig_xtab[ZIG_N + 1]; /* x-coordinates of rectangle edges */
static int zig_ready = 0;

static inline double zig_pdf(double x) {
    return exp(-0.5 * x * x);
}

static void zig_setup(void) {
    int i;

    if (zig_ready) return;

    /* Build the table:
     * xtab[1] = r (the tail start)
     * xtab[i] for i=2..255: computed from equal-area property
     * xtab[256] = 0 (the peak of the distribution)
     * xtab[0] = v / f(r) (width of the base strip that includes the tail)
     */
    zig_xtab[256] = 0.0;
    zig_xtab[1] = ZIG_R;

    for (i = 2; i <= 255; i++) {
        zig_xtab[i] = sqrt(-2.0 * log(ZIG_V / zig_xtab[i - 1] + zig_pdf(zig_xtab[i - 1])));
    }
    zig_xtab[0] = ZIG_V / zig_pdf(ZIG_R);

    zig_ready = 1;
}

/**
 * @brief Marsaglia's exact tail algorithm.
 * Samples from the tail |x| > r of the standard normal.
 */
static double zig_sample_tail(uint64_t* state) {
    double x, y, u1, u2;
    for (;;) {
        do {
            u1 = u64_to_double01(xoshiro256ss_next(state));
        } while (u1 == 0.0);
        do {
            u2 = u64_to_double01(xoshiro256ss_next(state));
        } while (u2 == 0.0);

        x = -log(u1) / ZIG_R;
        y = -log(u2);

        if (2.0 * y >= x * x) {
            return x + ZIG_R;
        }
    }
}

/**
 * @brief Ziggurat standard normal sampler (256 rectangles).
 *
 * The tables are built for the right half of the normal (x >= 0).
 * We generate |x| from the half-normal and then apply a random sign.
 */
static double ziggurat_rnor(uint64_t* state) {
    uint64_t u, u2;
    int i, sign;
    double x;

    if (!zig_ready) {
        zig_setup();
    }

    for (;;) {
        u = xoshiro256ss_next(state);
        i = (int)(u & 0xFF);  /* layer index: 0..255 */
        sign = (u & 0x100) ? -1 : 1;  /* bit 8 for sign */

        /* Generate uniform x in [0, xtab[i]) using a fresh random number */
        u2 = xoshiro256ss_next(state);
        x = u64_to_double01(u2) * zig_xtab[i];

        /* Fast accept: x < xtab[i+1] */
        if (x < zig_xtab[i + 1]) {
            return sign * x;
        }

        /* Layer 0 is special: it includes the tail */
        if (i == 0) {
            /* x is in [xtab[1], xtab[0]). Check if in rectangular part or tail */
            if (x < zig_xtab[1]) {
                return sign * x;
            }
            /* Need tail sample */
            double tail = zig_sample_tail(state);
            return sign * tail;
        }

        /* Wedge test: accept with probability (f(x) - f(xtab[i])) / (f(xtab[i+1]) - f(xtab[i])) */
        {
            double f_x = zig_pdf(x);
            double f_outer = zig_pdf(zig_xtab[i]);     /* f at outer edge (smaller f value) */
            double f_inner = zig_pdf(zig_xtab[i + 1]); /* f at inner edge (larger f value) */
            double u_wedge = u64_to_double01(xoshiro256ss_next(state));

            if (u_wedge * (f_inner - f_outer) < (f_x - f_outer)) {
                return sign * x;
            }
        }
    }
}

lmmc_status_t lmmc_rng_normal(
    lmmc_rng_t* rng,
    lmmc_real_t mean,
    lmmc_real_t stddev,
    lmmc_real_t* out_value)
{
    if (rng == NULL || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (stddev <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    *out_value = mean + stddev * ziggurat_rnor(rng->state);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_exponential(
    lmmc_rng_t* rng,
    lmmc_real_t rate,
    lmmc_real_t* out_value)
{
    double u;

    if (rng == NULL || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (rate <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    do {
        u = u64_to_double01(xoshiro256ss_next(rng->state));
    } while (u == 1.0);

    *out_value = -log(1.0 - u) / rate;
    return LMMC_STATUS_OK;
}


lmmc_status_t lmmc_rng_fill_uniform(
    lmmc_rng_t* rng,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t* array,
    size_t count)
{
    size_t i;
    double u;

    if (rng == NULL || array == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < count; i++) {
        u = u64_to_double01(xoshiro256ss_next(rng->state));
        array[i] = a + (b - a) * u;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_shuffle(
    lmmc_rng_t* rng,
    void* array,
    size_t count,
    size_t elem_size)
{
    size_t i, j;
    unsigned char* arr;
    unsigned char* tmp;

    if (rng == NULL || array == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (elem_size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (count <= 1) {
        return LMMC_STATUS_OK;
    }

    arr = (unsigned char*)array;
    tmp = (unsigned char*)lmmc_alloc(elem_size);
    if (tmp == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (i = count - 1; i > 0; i--) {
        uint64_t r = xoshiro256ss_next(rng->state);
        j = (size_t)(r % (i + 1));

        if (i != j) {
            memcpy(tmp, arr + i * elem_size, elem_size);
            memcpy(arr + i * elem_size, arr + j * elem_size, elem_size);
            memcpy(arr + j * elem_size, tmp, elem_size);
        }
    }

    lmmc_free(tmp);
    return LMMC_STATUS_OK;
}

/**
 * @brief Internal: generate standard normal using Ziggurat.
 */
static inline double rng_std_normal(uint64_t* state) {
    return ziggurat_rnor(state);
}

lmmc_status_t lmmc_rng_gamma(
    lmmc_rng_t* rng,
    lmmc_real_t shape,
    lmmc_real_t scale,
    lmmc_real_t* out)
{
    double d, c, x, v, u;

    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (shape <= 0.0 || scale <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* For shape < 1, use: Gamma(shape) = Gamma(shape+1) * U^(1/shape) */
    if (shape < 1.0) {
        lmmc_real_t g;
        lmmc_status_t st = lmmc_rng_gamma(rng, shape + 1.0, 1.0, &g);
        if (st != LMMC_STATUS_OK) return st;

        u = u64_to_double01(xoshiro256ss_next(rng->state));
        while (u == 0.0) {
            u = u64_to_double01(xoshiro256ss_next(rng->state));
        }
        *out = scale * g * pow(u, 1.0 / shape);
        return LMMC_STATUS_OK;
    }

    /* Marsaglia-Tsang method for shape >= 1 */
    d = shape - 1.0 / 3.0;
    c = 1.0 / sqrt(9.0 * d);

    for (;;) {
        do {
            x = rng_std_normal(rng->state);
            v = 1.0 + c * x;
        } while (v <= 0.0);

        v = v * v * v;
        u = u64_to_double01(xoshiro256ss_next(rng->state));

        /* Squeeze test */
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
            *out = scale * d * v;
            return LMMC_STATUS_OK;
        }

        /* Full acceptance check */
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) {
            *out = scale * d * v;
            return LMMC_STATUS_OK;
        }
    }
}

lmmc_status_t lmmc_rng_beta(
    lmmc_rng_t* rng,
    lmmc_real_t alpha,
    lmmc_real_t beta_param,
    lmmc_real_t* out)
{
    lmmc_real_t x, y;
    lmmc_status_t st;

    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (alpha <= 0.0 || beta_param <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_rng_gamma(rng, alpha, 1.0, &x);
    if (st != LMMC_STATUS_OK) return st;

    st = lmmc_rng_gamma(rng, beta_param, 1.0, &y);
    if (st != LMMC_STATUS_OK) return st;

    *out = x / (x + y);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_chi_squared(
    lmmc_rng_t* rng,
    lmmc_real_t df,
    lmmc_real_t* out)
{
    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (df <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    return lmmc_rng_gamma(rng, df / 2.0, 2.0, out);
}

lmmc_status_t lmmc_rng_student_t(
    lmmc_rng_t* rng,
    lmmc_real_t df,
    lmmc_real_t* out)
{
    lmmc_real_t z, chi2;
    lmmc_status_t st;

    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (df <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    z = rng_std_normal(rng->state);

    st = lmmc_rng_chi_squared(rng, df, &chi2);
    if (st != LMMC_STATUS_OK) return st;

    *out = z / sqrt(chi2 / df);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_f(
    lmmc_rng_t* rng,
    lmmc_real_t df1,
    lmmc_real_t df2,
    lmmc_real_t* out)
{
    lmmc_real_t chi1, chi2;
    lmmc_status_t st;

    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (df1 <= 0.0 || df2 <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    st = lmmc_rng_chi_squared(rng, df1, &chi1);
    if (st != LMMC_STATUS_OK) return st;

    st = lmmc_rng_chi_squared(rng, df2, &chi2);
    if (st != LMMC_STATUS_OK) return st;

    *out = (chi1 / df1) / (chi2 / df2);
    return LMMC_STATUS_OK;
}


/**
 * Poisson distribution
 * - Inversion for lambda < 10
 * - PTRD (Hörmann) for lambda >= 10
 */

/**
 * @brief Poisson inversion method for small lambda.
 */
static size_t poisson_inversion(uint64_t* state, double lambda) {
    double L = exp(-lambda);
    double p = 1.0;
    size_t k = 0;

    do {
        k++;
        p *= u64_to_double01(xoshiro256ss_next(state));
    } while (p > L);

    return k - 1;
}

/**
 * @brief PTRD (Transformed Rejection with Decomposition) for Poisson, lambda >= 10.
 * Reference: Hörmann, "The transformed rejection method for generating Poisson
 * random variables", Insurance: Mathematics and Economics 12 (1993) 39-45.
 */
static size_t poisson_ptrd(uint64_t* state, double lambda) {
    double smu = sqrt(lambda);
    double b = 0.931 + 2.53 * smu;
    double a = -0.059 + 0.02483 * b;
    double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    double vr = 0.9277 - 3.6224 / (b - 2.0);
    double us, v, u, k_real;
    int64_t k;

    for (;;) {
        v = u64_to_double01(xoshiro256ss_next(state));
        if (v <= 0.86 * vr) {
            u = v / vr - 0.43;
            k_real = floor((2.0 * a / (0.5 - fabs(u)) + b) * u + lambda + 0.445);
            if (k_real >= 0.0) return (size_t)k_real;
        }

        if (v >= vr) {
            u = u64_to_double01(xoshiro256ss_next(state)) - 0.5;
        } else {
            u = v / vr - 0.93;
            u = ((u >= 0.0) ? 0.5 : -0.5) - u;
            v = u64_to_double01(xoshiro256ss_next(state)) * vr;
        }

        us = 0.5 - fabs(u);
        if (us < 0.013 && v > us) continue;

        k_real = floor((2.0 * a / us + b) * u + lambda + 0.445);
        k = (int64_t)k_real;
        if (k < 0) continue;

        /* Acceptance: log(v * inv_alpha / (a/(us*us) + b)) <=
         * -lambda + k*log(lambda) - lgamma(k+1) */
        v = v * inv_alpha / (a / (us * us) + b);
        {
            /* Acceptance: log(v) <= k*log(lambda) - lambda - lgamma(k+1) */
            double log_accept = (double)k * log(lambda) - lambda - lgamma((double)k + 1.0);
            if (log(v) <= log_accept) return (size_t)k;
        }
    }
}

lmmc_status_t lmmc_rng_poisson(
    lmmc_rng_t* rng,
    lmmc_real_t lambda,
    size_t* out)
{
    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lambda <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (lambda < 10.0) {
        *out = poisson_inversion(rng->state, lambda);
    } else {
        *out = poisson_ptrd(rng->state, lambda);
    }

    return LMMC_STATUS_OK;
}


/**
 * Binomial distribution
 * For small n*min(p,1-p), use direct Bernoulli trials.
 * For larger values, use the BTRD algorithm (Hörmann).
 */

lmmc_status_t lmmc_rng_binomial(
    lmmc_rng_t* rng,
    size_t n,
    lmmc_real_t p,
    size_t* out)
{
    size_t i, count;
    double pp, u;
    int flip;

    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (p < 0.0 || p > 1.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (n == 0 || p == 0.0) {
        *out = 0;
        return LMMC_STATUS_OK;
    }
    if (p == 1.0) {
        *out = n;
        return LMMC_STATUS_OK;
    }

    /* Use symmetry: work with min(p, 1-p) */
    flip = 0;
    pp = p;
    if (p > 0.5) {
        pp = 1.0 - p;
        flip = 1;
    }

    /* For small n*p, use direct Bernoulli trials */
    if ((double)n * pp < 30.0) {
        count = 0;
        for (i = 0; i < n; i++) {
            u = u64_to_double01(xoshiro256ss_next(rng->state));
            if (u < pp) {
                count++;
            }
        }
        *out = flip ? (n - count) : count;
        return LMMC_STATUS_OK;
    }

    /* For larger n*p, use normal approximation with continuity correction
     * and rejection from Poisson. This is a simplified approach using
     * the waiting-time / geometric method. */
    {
        /* Use the inverse-transform geometric method (Devroye) */
        double log_q = log(1.0 - pp);
        double sum = 0.0;
        count = 0;

        for (;;) {
            u = u64_to_double01(xoshiro256ss_next(rng->state));
            while (u == 0.0) {
                u = u64_to_double01(xoshiro256ss_next(rng->state));
            }
            sum += floor(log(u) / log_q) + 1.0;
            if (sum > (double)n) break;
            count++;
        }

        *out = flip ? (n - count) : count;
        return LMMC_STATUS_OK;
    }
}

lmmc_status_t lmmc_rng_int_uniform(
    lmmc_rng_t* rng,
    int64_t lo,
    int64_t hi,
    int64_t* out)
{
    uint64_t range, r, limit;

    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lo > hi) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (lo == hi) {
        *out = lo;
        return LMMC_STATUS_OK;
    }

    /* Unbiased rejection sampling */
    range = (uint64_t)(hi - lo) + 1;

    if (range == 0) {
        /* Full 64-bit range (overflow case: hi - lo + 1 == 2^64) */
        *out = (int64_t)xoshiro256ss_next(rng->state);
        return LMMC_STATUS_OK;
    }

    /* Reject values >= limit to avoid modulo bias */
    limit = UINT64_MAX - (UINT64_MAX % range);

    do {
        r = xoshiro256ss_next(rng->state);
    } while (r >= limit);

    *out = lo + (int64_t)(r % range);
    return LMMC_STATUS_OK;
}
