#ifndef LMMC_MEMORY_BRIDGE_H
#define LMMC_MEMORY_BRIDGE_H

#include <stdlib.h>

#if defined(LMMC_USE_LAMMP_ALLOC)
#include "lammp/lmmp.h"
#define lmmc_alloc(sz) lmmp_alloc((sz))
#define lmmc_free(ptr) lmmp_free((ptr))
#else
#define lmmc_alloc(sz) malloc((sz))
#define lmmc_free(ptr) free((ptr))
#endif

#endif
