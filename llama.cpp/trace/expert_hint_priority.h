#pragma once

#include "expert_prefetch_types.h"
#include <cstdint>

struct ExpertHintPriorityKey {
    uint64_t step = 0;
    int layer = -1;
    double route_score = 0.0;
    uint64_t sequence = 0;
    uint64_t deadline_ts_ns = 0;
};

// Returns true when a must be issued before b.
bool expert_hint_priority_higher(
        const ExpertHintPriorityKey & a,
        const ExpertHintPriorityKey & b,
        ExpertAsyncPriorityMode mode);
