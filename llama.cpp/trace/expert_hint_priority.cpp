#include "expert_hint_priority.h"

namespace {

bool compare_score(const ExpertHintPriorityKey & a, const ExpertHintPriorityKey & b) {
    if (a.route_score != b.route_score) {
        return a.route_score > b.route_score;
    }
    return a.sequence < b.sequence;
}

bool compare_deadline(const ExpertHintPriorityKey & a, const ExpertHintPriorityKey & b) {
    if (a.deadline_ts_ns != b.deadline_ts_ns) {
        if (a.deadline_ts_ns == 0) {
            return false;
        }
        if (b.deadline_ts_ns == 0) {
            return true;
        }
        return a.deadline_ts_ns < b.deadline_ts_ns;
    }
    if (a.step != b.step) {
        return a.step < b.step;
    }
    if (a.layer != b.layer) {
        return a.layer < b.layer;
    }
    return a.sequence < b.sequence;
}

bool compare_deadline_score(const ExpertHintPriorityKey & a, const ExpertHintPriorityKey & b) {
    if (a.deadline_ts_ns != b.deadline_ts_ns) {
        if (a.deadline_ts_ns == 0) {
            return false;
        }
        if (b.deadline_ts_ns == 0) {
            return true;
        }
        return a.deadline_ts_ns < b.deadline_ts_ns;
    }
    return compare_score(a, b);
}

bool compare_known_stage(const ExpertHintPriorityKey & a, const ExpertHintPriorityKey & b) {
    if (a.step != b.step) {
        return a.step < b.step;
    }
    if (a.layer != b.layer) {
        return a.layer < b.layer;
    }
    if (a.stage != b.stage) {
        return a.stage == ExpertTensorStage::Early;
    }
    return compare_score(a, b);
}

} // namespace

bool expert_hint_priority_higher(
        const ExpertHintPriorityKey & a,
        const ExpertHintPriorityKey & b,
        ExpertAsyncPriorityMode mode) {
    switch (mode) {
        case ExpertAsyncPriorityMode::Score:
            return compare_score(a, b);
        case ExpertAsyncPriorityMode::Deadline:
            return compare_deadline(a, b);
        case ExpertAsyncPriorityMode::DeadlineScore:
            return compare_deadline_score(a, b);
        case ExpertAsyncPriorityMode::StageDeadlineScore:
            if (expert_hint_priority_uses_legacy_partition(a.stage) ||
                    expert_hint_priority_uses_legacy_partition(b.stage)) {
                return compare_deadline_score(a, b);
            }
            return compare_known_stage(a, b);
    }
    return compare_score(a, b);
}

bool expert_hint_priority_uses_legacy_partition(ExpertTensorStage stage) {
    return stage == ExpertTensorStage::Unknown;
}
