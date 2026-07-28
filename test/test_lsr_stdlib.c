#include "lmmc/lmmc.h"

#include <math.h>
#include <stdio.h>

static int close_real(lmmc_real_t a, lmmc_real_t b)
{
    return fabs((double)(a - b)) <= 1e-12;
}

int main(void)
{
    lmmc_real_t out = 0;
    lmmc_real_t out2 = 0;
    lmmc_real_t values[] = {1, 2, 3, 4};
    lmmc_complex_t z;
    lmmc_complex_t w;
    lmmc_rng_t* rng = NULL;
    int64_t randint_out = 0;

    if (lmmc_lsr_math_pi(&out) != LMMC_STATUS_OK ||
        !close_real(out, LMMC_CONST_PI)) {
        fprintf(stderr, "std.math.pi mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_i(&z) != LMMC_STATUS_OK ||
        !close_real(z.real, 0) || !close_real(z.imag, 1)) {
        fprintf(stderr, "std.math.i mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_I(&w) != LMMC_STATUS_OK ||
        !close_real(w.real, z.real) || !close_real(w.imag, z.imag)) {
        fprintf(stderr, "std.math.I alias mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_complex(3, 4, &z) != LMMC_STATUS_OK) {
        fprintf(stderr, "std.math.complex failed\n");
        return 1;
    }

    if (lmmc_lsr_math_real(&z, &out) != LMMC_STATUS_OK ||
        !close_real(out, 3)) {
        fprintf(stderr, "std.math.real mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_imag(&z, &out) != LMMC_STATUS_OK ||
        !close_real(out, 4)) {
        fprintf(stderr, "std.math.imag mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_conj(&z, &w) != LMMC_STATUS_OK ||
        !close_real(w.real, 3) || !close_real(w.imag, -4)) {
        fprintf(stderr, "std.math.conj mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_complex_abs(&z, &out) != LMMC_STATUS_OK ||
        !close_real(out, 5)) {
        fprintf(stderr, "std.math.abs complex mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_sqrt(-1, &out) != LMMC_STATUS_OUT_OF_RANGE) {
        fprintf(stderr, "std.math.sqrt domain error not reported\n");
        return 1;
    }

    if (lmmc_lsr_math_log(8, 2, &out) != LMMC_STATUS_OK ||
        !close_real(out, 3)) {
        fprintf(stderr, "std.math.log mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_clamp(5, 1, 3, &out) != LMMC_STATUS_OK ||
        !close_real(out, 3)) {
        fprintf(stderr, "std.math.clamp mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_mean(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.5)) {
        fprintf(stderr, "std.stats.mean mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_median(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.5)) {
        fprintf(stderr, "std.stats.median mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_var(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, 5.0 / 3.0)) {
        fprintf(stderr, "std.stats.var mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_std(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, sqrt(5.0 / 3.0))) {
        fprintf(stderr, "std.stats.std mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_quantile(values, 4, 0.5, &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.5)) {
        fprintf(stderr, "std.stats.quantile mismatch\n");
        return 1;
    }

    if (lmmc_rng_create(&rng) != LMMC_STATUS_OK) {
        fprintf(stderr, "rng create failed\n");
        return 1;
    }

    if (lmmc_lsr_random_seed(rng, 42) != LMMC_STATUS_OK ||
        lmmc_lsr_random_rand(rng, &out) != LMMC_STATUS_OK ||
        lmmc_lsr_random_seed(rng, 42) != LMMC_STATUS_OK ||
        lmmc_lsr_random_rand(rng, &out2) != LMMC_STATUS_OK ||
        !close_real(out, out2)) {
        fprintf(stderr, "std.random fixed seed is not reproducible\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    if (lmmc_lsr_random_randint(rng, 1, 3, &randint_out) != LMMC_STATUS_OK ||
        randint_out < 1 || randint_out > 3) {
        fprintf(stderr, "std.random.randint range mismatch\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    if (lmmc_lsr_random_choice(rng, values, 4, &out) != LMMC_STATUS_OK ||
        out < 1 || out > 4) {
        fprintf(stderr, "std.random.choice mismatch\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    lmmc_rng_destroy(rng);

    return 0;
}
