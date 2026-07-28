/**
 * @file complex.h
 * @brief Basic complex scalar support for LSR std.math bindings.
 */
#ifndef LMMC_COMPLEX_H
#define LMMC_COMPLEX_H

#include "lmmc/config.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    lmmc_real_t real;
    lmmc_real_t imag;
} lmmc_complex_t;

lmmc_status_t lmmc_complex_create(lmmc_real_t real,
                                  lmmc_real_t imag,
                                  lmmc_complex_t* out);
lmmc_status_t lmmc_complex_conj(const lmmc_complex_t* z,
                                lmmc_complex_t* out);
lmmc_status_t lmmc_complex_modulus(const lmmc_complex_t* z,
                                   lmmc_real_t* out);

#ifdef __cplusplus
}
#endif

#endif
