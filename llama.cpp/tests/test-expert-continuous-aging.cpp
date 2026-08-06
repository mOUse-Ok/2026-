#include "expert_continuous_aging.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-expert-continuous-aging: %s\n", message);
        std::abort();
    }
}

static ExpertContinuousAgingConfig config() {
    ExpertContinuousAgingConfig result;
    result.score_epoch_ts_ns = 1'000'000'000ULL;
    return result;
}

static ExpertContinuousAgingTaskKey key(
        uint64_t task_id,
        uint64_t step,
        int layer,
        ExpertTensorStage stage,
        double score,
        uint64_t enqueue_offset_ns,
        uint64_t deadline,
        uint64_t sequence) {
    ExpertContinuousAgingTaskKey result;
    result.task_id = task_id;
    result.step = step;
    result.layer = layer;
    result.stage = stage;
    result.route_score = score;
    result.sequence = sequence;
    result.deadline_ts_ns = deadline;
    result.enqueued_ts_ns = config().score_epoch_ts_ns + enqueue_offset_ns;
    result.nbytes = 4096;
    return result;
}

static void test_static_key_matches_direct_formula() {
    const ExpertContinuousAgingConfig cfg = config();
    for (uint64_t i = 0; i < 100; ++i) {
        const auto a = key(
                i * 2 + 1, 7, 3, ExpertTensorStage::Late,
                0.001 * static_cast<double>((i * 17) % 11),
                i * 731'111, 0, i * 2);
        const auto b = key(
                i * 2 + 2, 7, 3, ExpertTensorStage::Late,
                0.001 * static_cast<double>((i * 29 + 3) % 11),
                i * 119'999 + 20'000'000, 0, i * 2 + 1);
        const uint64_t now = cfg.score_epoch_ts_ns + 200'000'000 + i;
        require(
                expert_continuous_aging_higher_static(a, b, cfg) ==
                expert_continuous_aging_higher_direct(a, b, now, cfg),
                "static key and direct formula disagree");
        require(
                expert_continuous_aging_higher_static(b, a, cfg) ==
                expert_continuous_aging_higher_direct(b, a, now, cfg),
                "reverse static key and direct formula disagree");
    }
}

static void test_waiting_never_crosses_group_boundaries() {
    const ExpertContinuousAgingConfig cfg = config();
    const auto old_later_step = key(
            1, 8, 0, ExpertTensorStage::Early, 1000.0, 0, 1, 1);
    const auto fresh_earlier_step = key(
            2, 7, 99, ExpertTensorStage::Late, -1000.0, 190'000'000, 999, 2);
    require(expert_continuous_aging_higher_static(
                    fresh_earlier_step, old_later_step, cfg),
            "waiting or score crossed Step");

    const auto old_later_layer = key(
            3, 7, 4, ExpertTensorStage::Early, 1000.0, 0, 1, 3);
    const auto fresh_earlier_layer = key(
            4, 7, 3, ExpertTensorStage::Late, -1000.0, 190'000'000, 999, 4);
    require(expert_continuous_aging_higher_static(
                    fresh_earlier_layer, old_later_layer, cfg),
            "waiting or score crossed Layer");

    const auto old_late = key(
            5, 7, 3, ExpertTensorStage::Late, 1000.0, 0, 1, 5);
    const auto fresh_early = key(
            6, 7, 3, ExpertTensorStage::Early, -1000.0, 190'000'000, 999, 6);
    require(expert_continuous_aging_higher_static(fresh_early, old_late, cfg),
            "waiting or score crossed EARLY/LATE");
}

static void test_queue_changes_only_same_group_and_drains_cleanly() {
    const ExpertContinuousAgingConfig cfg = config();
    ExpertContinuousAgingQueue queue;
    queue.reset(8, cfg);
    const auto older = key(
            1, 2, 4, ExpertTensorStage::Late, 0.0, 0,
            cfg.score_epoch_ts_ns + 500'000'000, 1);
    const auto newer_high_score = key(
            2, 2, 4, ExpertTensorStage::Late, 0.001, 50'000'000,
            cfg.score_epoch_ts_ns + 500'000'000, 2);
    queue.insert(older);
    queue.insert(newer_high_score);
    const auto first = queue.select(cfg.score_epoch_ts_ns + 100'000'000);
    require(first.legacy_head.key.task_id == 2,
            "legacy head no longer matches deadline_score feature-off order");
    require(first.selected.key.task_id == 1,
            "50ms compensation did not change a typical-gap winner");
    require(first.winner_changed_vs_legacy,
            "winner change was not recorded");
    const auto second = queue.select(cfg.score_epoch_ts_ns + 100'000'001);
    require(second.selected.key.task_id == 2, "remaining Task was not selected");

    const ExpertContinuousAgingAudit audit = queue.audit(true);
    require(audit.valid && audit.final_queue_empty,
            "queue/store/index/bytes did not drain cleanly");
    const ExpertContinuousAgingCounters counters = queue.counters();
    require(counters.stale_handle_count == 0, "stale handle observed");
    require(counters.full_store_scan_count == 0, "full-store scan observed");
    require(counters.invariant_error_count == 0, "invariant error observed");
    require(counters.insert_count == 2 && counters.erase_count == 2 &&
                    counters.selection_count == 2,
            "operation conservation failed");
}

static void test_legacy_head_preserves_deadline_score() {
    const ExpertContinuousAgingConfig cfg = config();
    ExpertContinuousAgingQueue queue;
    queue.reset(8, cfg);
    queue.insert(key(
            1, 0, 0, ExpertTensorStage::Early, 0.1, 10, 900, 1));
    queue.insert(key(
            2, 9, 9, ExpertTensorStage::Late, 100.0, 20, 800, 2));
    const auto selected = queue.select(cfg.score_epoch_ts_ns + 1000);
    require(selected.legacy_head.key.task_id == 2,
            "feature-off reference stopped using deadline_score");
    require(selected.selected.key.task_id == 1,
            "Continuous Aging did not preserve Step before Layer/stage/score");
}

int main() {
    test_static_key_matches_direct_formula();
    test_waiting_never_crosses_group_boundaries();
    test_queue_changes_only_same_group_and_drains_cleanly();
    test_legacy_head_preserves_deadline_score();
    return 0;
}
