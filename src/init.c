#include "lmmc/init.h"

/* LAMMP — the low-level multi-precision library */
#include "lammp/lmmp.h"

/*
 * Reference-counted initialisation counter.
 * The first lmmc_init() initialises LAMMP resources; the last
 * lmmc_deinit() tears them down.
 *
 * Not yet thread-safe — planned for the multi-threaded LAMMP.
 */
static int lmmc_init_count = 0;

void lmmc_init(void) {
    if (++lmmc_init_count != 1)
        return;

    /*
     * Current LAMMP: initialise the internal operation stack.
     *
     * When the new LAMMP ships with lmmp_global_init_() /
     * lmmp_global_deinit(), replace the call below with:
     *
     *     lmmp_global_init_();   // thread-local / process-wide init
     *
     * and add to lmmc_deinit():
     *
     *     lmmp_global_deinit();  // teardown
     *
     * The {stack_init,stack_reset} hacks in LMCAS's bigint.hpp
     * should then be removed because the new LAMMP manages its
     * own internal stack without leaking.
     */
    lmmp_stack_init();
}

void lmmc_deinit(void) {
    if (--lmmc_init_count != 0)
        return;

    /*
     * Current LAMMP: release the stack allocation.
     *
     * In the new LAMMP, replace with:
     *
     *     lmmp_global_deinit();
     *
     * and remove the lmmp_stack_reset workaround from the callers.
     */

    /* Future: lmmp_global_deinit(); */
}

void lmmc_stack_reset(size_t size) {
    /*
     * Delegate to the underlying LAMMP stack reset.
     *
     * When the new LAMMP manages its own internal stack correctly
     * (i.e. it reuses / frees scratch space rather than letting it
     * accumulate), this function can become a no-op:
     *
     *     (void)size;
     *
     * and the callers in LMCAS can be removed entirely.
     */
    lmmp_stack_reset(size);
}
