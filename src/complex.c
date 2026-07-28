#include "lmmc/complex.h"

#include <math.h>

lmmc_status_t lmmc_complex_create(lmmc_real_t real,
                                  lmmc_real_t imag,
                                  lmmc_complex_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    out->real = real;
    out->imag = imag;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_conj(const lmmc_complex_t* z,
                                lmmc_complex_t* out)
{
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    out->real = z->real;
    out->imag = -z->imag;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_complex_modulus(const lmmc_complex_t* z,
                                   lmmc_real_t* out)
{
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = (lmmc_real_t)hypot((double)z->real, (double)z->imag);
    return LMMC_STATUS_OK;
}
