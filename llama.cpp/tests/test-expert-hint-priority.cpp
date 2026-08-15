#include "expert_hint_priority.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-expert-hint-priority: %s\n", message);
        std::abort();
    }
}

static ExpertHintPriorityKey key(
        uint64_t sequence,
        uint64_t step,
        int layer,
        double score,
        uint64_t deadline = 0) {
    ExpertHintPriorityKey result;
    result.sequence = sequence;
    result.step = step;
    result.layer = layer;
    result.route_score = score;
    result.deadline_ts_ns = deadline;
    return result;
}

static void test_deadline_score_order_is_preserved() {
    const ExpertHintPriorityKey no_deadline = key(1, 0, 0, 1.0, 0);
    const ExpertHintPriorityKey later_deadline = key(2, 0, 0, 0.9, 200);
    const ExpertHintPriorityKey earlier_deadline = key(3, 0, 0, 0.1, 100);
    require(expert_hint_priority_higher(
                    earlier_deadline, no_deadline, ExpertAsyncPriorityMode::DeadlineScore),
            "nonzero deadline must precede a zero deadline");
    require(expert_hint_priority_higher(
                    earlier_deadline, later_deadline, ExpertAsyncPriorityMode::DeadlineScore),
            "earlier deadline must be selected first");

    const ExpertHintPriorityKey high_score = key(4, 0, 0, 0.9, 100);
    const ExpertHintPriorityKey low_score = key(5, 0, 0, 0.1, 100);
    require(expert_hint_priority_higher(
                    high_score, low_score, ExpertAsyncPriorityMode::DeadlineScore),
            "route score must break equal-deadline ties in descending order");

    const ExpertHintPriorityKey first = key(6, 0, 0, 0.9, 100);
    const ExpertHintPriorityKey second = key(7, 0, 0, 0.9, 100);
    require(expert_hint_priority_higher(
                    first, second, ExpertAsyncPriorityMode::DeadlineScore),
            "sequence must break equal deadline and score ties in ascending order");

    ExpertHintPriorityKey deterministic = key(8, 0, 0, 0.1, 100);
    deterministic.deterministic = true;
    require(expert_hint_priority_higher(
                    deterministic, high_score, ExpertAsyncPriorityMode::DeadlineScore),
            "deterministic requests must break equal-deadline ties");
    const ExpertHintPriorityKey strictly_earlier = key(9, 0, 0, 0.0, 99);
    require(expert_hint_priority_higher(
                    strictly_earlier, deterministic, ExpertAsyncPriorityMode::DeadlineScore),
            "deterministic requests must not override an earlier deadline");
}

static void test_other_core_modes_are_preserved() {
    const ExpertHintPriorityKey high_score = key(1, 9, 9, 0.9, 500);
    const ExpertHintPriorityKey early_deadline = key(2, 1, 1, 0.1, 100);
    require(expert_hint_priority_higher(
                    high_score, early_deadline, ExpertAsyncPriorityMode::Score),
            "score mode changed");
    require(expert_hint_priority_higher(
                    early_deadline, high_score, ExpertAsyncPriorityMode::Deadline),
            "deadline mode changed");
}

int main() {
    test_deadline_score_order_is_preserved();
    test_other_core_modes_are_preserved();
    return 0;
}
