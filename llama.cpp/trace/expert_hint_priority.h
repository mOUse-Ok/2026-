#pragma once

#include "expert_prefetch_types.h"
#include "expert_tensor_stage.h"

#include <cstdint>

struct ExpertHintPriorityKey {
    uint64_t step = 0;
    int layer = -1;
    ExpertTensorStage stage = ExpertTensorStage::Unknown;
    double route_score = 0.0;
    uint64_t sequence = 0;
    uint64_t deadline_ts_ns = 0;
};

// Returns true when a must be issued before b. StageDeadlineScore callers must
// keep known-stage and UNKNOWN tasks in separate heaps. Known tasks use the
// stage order; UNKNOWN tasks and cross-heap head arbitration use the legacy
// DeadlineScore order so UNKNOWN is neither lowered nor discarded.
bool expert_hint_priority_higher(
        const ExpertHintPriorityKey & a,
        const ExpertHintPriorityKey & b,
        ExpertAsyncPriorityMode mode);

bool expert_hint_priority_uses_legacy_partition(ExpertTensorStage stage);
