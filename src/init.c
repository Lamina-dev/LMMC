/**
 * @file init.c
 * @brief LMMC 库初始化与栈分配器桥接实现。
 */
#include "lmmc/init.h"


#include "lammp/lmmp.h"


static int lmmc_init_count = 0;

void lmmc_init(void) {
    if (++lmmc_init_count != 1)
        return;


    lmmp_stack_init();
}

void lmmc_deinit(void) {
    if (--lmmc_init_count != 0)
        return;


}

void lmmc_stack_reset(size_t size) {

    lmmp_stack_reset(size);
}
