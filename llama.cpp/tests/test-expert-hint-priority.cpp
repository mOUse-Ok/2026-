#include "expert_hint_priority.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

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
        ExpertTensorStage stage,
        double score,
        uint64_t deadline = 0) {
    ExpertHintPriorityKey result;
    result.sequence = sequence;
    result.step = step;
    result.layer = layer;
    result.stage = stage;
    result.route_score = score;
    result.deadline_ts_ns = deadline;
    return result;
}

static std::vector<uint64_t> drain_stage_queue(std::vector<ExpertHintPriorityKey> tasks) {
    std::vector<ExpertHintPriorityKey> known;
    std::vector<ExpertHintPriorityKey> legacy;
    const auto cmp = [](const ExpertHintPriorityKey & a, const ExpertHintPriorityKey & b) {
        return expert_hint_priority_higher(
                b, a, ExpertAsyncPriorityMode::StageDeadlineScore);
    };
    for (ExpertHintPriorityKey & task : tasks) {
        std::vector<ExpertHintPriorityKey> & heap =
                expert_hint_priority_uses_legacy_partition(task.stage) ? legacy : known;
        heap.push_back(std::move(task));
        std::push_heap(heap.begin(), heap.end(), cmp);
    }

    std::vector<uint64_t> result;
    while (!known.empty() || !legacy.empty()) {
        std::vector<ExpertHintPriorityKey> * heap = &known;
        if (known.empty() || (!legacy.empty() && expert_hint_priority_higher(
                legacy.front(), known.front(), ExpertAsyncPriorityMode::StageDeadlineScore))) {
            heap = &legacy;
        }
        std::pop_heap(heap->begin(), heap->end(), cmp);
        result.push_back(heap->back().sequence);
        heap->pop_back();
    }
    return result;
}

static void test_stage_order_is_step_layer_stage_score_sequence() {
    const std::vector<uint64_t> order = drain_stage_queue({
        key(6, 1, 0, ExpertTensorStage::Early, 100.0),
        key(5, 0, 1, ExpertTensorStage::Early, 100.0),
        key(4, 0, 0, ExpertTensorStage::Late, 100.0),
        key(3, 0, 0, ExpertTensorStage::Early, 0.4),
        key(2, 0, 0, ExpertTensorStage::Early, 0.8),
        key(1, 0, 0, ExpertTensorStage::Early, 0.8),
    });
    require(order == std::vector<uint64_t>({1, 2, 3, 4, 5, 6}),
            "stage queue did not follow step/layer/stage/score/sequence");
}

static void test_unknown_uses_deadline_score_fallback() {
    const std::vector<uint64_t> order = drain_stage_queue({
        key(1, 0, 0, ExpertTensorStage::Early, 0.9, 300),
        key(2, 0, 0, ExpertTensorStage::Unknown, 0.1, 100),
        key(3, 0, 0, ExpertTensorStage::Unknown, 0.8, 0),
        key(4, 0, 0, ExpertTensorStage::Late, 0.2, 200),
    });
    require(order == std::vector<uint64_t>({2, 1, 4, 3}),
            "UNKNOWN was not arbitrated with legacy deadline_score");
}

static void test_late_starvation_scope_is_bounded_by_step_and_layer() {
    const std::vector<uint64_t> same_layer = drain_stage_queue({
        key(1, 7, 3, ExpertTensorStage::Late, 1.0),
        key(2, 7, 3, ExpertTensorStage::Early, 0.1),
        key(3, 7, 3, ExpertTensorStage::Early, 0.1),
    });
    require(same_layer == std::vector<uint64_t>({2, 3, 1}),
            "same-layer EARLY tasks did not precede LATE");

    const std::vector<uint64_t> next_step = drain_stage_queue({
        key(1, 7, 3, ExpertTensorStage::Late, 0.1),
        key(2, 8, 0, ExpertTensorStage::Early, 100.0),
    });
    require(next_step == std::vector<uint64_t>({1, 2}),
            "a later step bypassed an earlier-step LATE task");
}

static void test_legacy_modes_are_preserved() {
    const ExpertHintPriorityKey high_score = key(1, 9, 9, ExpertTensorStage::Late, 0.9, 500);
    const ExpertHintPriorityKey early_deadline = key(2, 1, 1, ExpertTensorStage::Early, 0.1, 100);
    require(expert_hint_priority_higher(
                    high_score, early_deadline, ExpertAsyncPriorityMode::Score),
            "score mode changed");
    require(expert_hint_priority_higher(
                    early_deadline, high_score, ExpertAsyncPriorityMode::Deadline),
            "deadline mode changed");
    require(expert_hint_priority_higher(
                    early_deadline, high_score, ExpertAsyncPriorityMode::DeadlineScore),
            "deadline_score mode changed");
}

int main() {
    test_stage_order_is_step_layer_stage_score_sequence();
    test_unknown_uses_deadline_score_fallback();
    test_late_starvation_scope_is_bounded_by_step_and_layer();
    test_legacy_modes_are_preserved();
    return 0;
}
