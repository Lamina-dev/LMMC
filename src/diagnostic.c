#include "lmmc/diagnostic.h"

void lmmc_diagnostic_emit(
    const lmmc_diagnostic_sink_t* sink,
    const lmmc_diagnostic_t* diagnostic
) {
    if (sink == NULL || diagnostic == NULL || sink->callback == NULL) {
        return;
    }
    if (diagnostic->level < sink->minimum_level) {
        return;
    }
    sink->callback(diagnostic, sink->user_data);
}
