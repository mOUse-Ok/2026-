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
    if (a.deterministic != b.deterministic) {
        return a.deterministic;
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
    }
    return compare_score(a, b);
}
