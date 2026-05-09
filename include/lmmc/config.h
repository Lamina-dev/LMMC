#ifndef LMMC_CONFIG_H
#define LMMC_CONFIG_H

#include <math.h>
typedef double lmmc_real_t;

#define LMMC_PRI_REAL "f"
#define LMMC_SCN_REAL "lf"

#define LMMC_REAL_INIT(x)           ((void)0)
#define LMMC_REAL_CLEAR(x)          ((void)0)

#define LMMC_REAL_SET(dst, src)     (*(dst) = *(src))
#define LMMC_REAL_SET_D(dst, dbl)   (*(dst) = (dbl))

#define LMMC_REAL_ADD(res, a, b)    (*(res) = *(a) + *(b))
#define LMMC_REAL_SUB(res, a, b)    (*(res) = *(a) - *(b))
#define LMMC_REAL_MUL(res, a, b)    (*(res) = *(a) * *(b))
#define LMMC_REAL_DIV(res, a, b)    (*(res) = *(a) / *(b))

#define LMMC_REAL_NEG(res, a)       (*(res) = -*(a))
#define LMMC_REAL_SQRT(res, a)      (*(res) = sqrt(*(a)))
#define LMMC_REAL_ABS(res, a)       (*(res) = fabs(*(a)))

#define LMMC_REAL_CMP(a, b)         ((*(a) > *(b)) ? 1 : ((*(a) < *(b)) ? -1 : 0))
#define LMMC_REAL_IS_FINITE(x)      isfinite(*(x))

#define LMMC_REAL_EPSILON  ((lmmc_real_t)1e-15)
#define LMMC_CONST_PI      ((lmmc_real_t)3.14159265358979323846)

#endif
