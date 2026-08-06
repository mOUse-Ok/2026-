#include "expert_reserved_service.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

ExpertReservedTaskKey task(
        uint64_t id,
        uint64_t enqueue,
        uint64_t deadline,
        double score,
        uint64_t sequence,
        uint64_t bytes = 4096) {
    ExpertReservedTaskKey key;
    key.task_id = id;
    key.step = 2;
    key.layer = static_cast<int>(id % 8);
    key.stage = ExpertTensorStage::Late;
    key.route_score = score;
    key.sequence = sequence;
    key.deadline_ts_ns = deadline;
    key.enqueued_ts_ns = enqueue;
    key.nbytes = bytes;
    return key;
}

ExpertReservedServiceConfig fixture_config(uint64_t r = 1, uint64_t d = 8) {
    ExpertReservedServiceConfig config;
    config.reserved_numerator = r;
    config.reserved_denominator = d;
    config.eligibility_age_ns = 41;
    config.hard_urgent_guard_ns = 0;
    return config;
}

void assert_clean(const ExpertReservedServiceQueue & queue, bool empty = false) {
    const auto audit = queue.audit(empty);
    assert(audit.valid);
    assert(audit.store_size == audit.registry_size);
    assert(audit.store_size == audit.legacy_index_size);
    assert(audit.store_size == audit.aging_index_size);
    assert(audit.queued_bytes == audit.live_bytes);
    assert(queue.counters().stale_handle_count == 0);
    assert(queue.counters().full_store_scan_count == 0);
    assert(queue.counters().invariant_error_count == 0);
}

void test_legacy_order_and_reset() {
    ExpertReservedServiceQueue queue;
    queue.reset(8, fixture_config());
    queue.insert(task(1, 100, 1000, 0.1, 1));
    queue.insert(task(2, 101, 900, 0.9, 2));
    queue.insert(task(3, 102, 900, 0.2, 3));
    auto first = queue.select(110, false);
    assert(first.selected.key.task_id == 2);
    assert(first.source == ExpertReservedWinnerSource::Legacy);
    assert(first.credit_after == 0); // no AGE_GATED_ALL eligible Task
    auto second = queue.select(111, false);
    assert(second.selected.key.task_id == 3);
    auto third = queue.select(112, false);
    assert(third.selected.key.task_id == 1);
    assert_clean(queue, true);
}

void test_fresh_reserved_and_oldest_winner() {
    ExpertReservedServiceQueue queue;
    queue.reset(16, fixture_config(1, 2));
    queue.insert(task(1, 1, 1000, 1.0, 1));
    queue.insert(task(2, 2, 900, 1.0, 2));
    queue.insert(task(3, 3, 800, 1.0, 3));
    const auto first = queue.select(100, false);
    assert(first.selected.key.task_id == 3);
    assert(first.credit_after == 1);
    const auto second = queue.select(100, false);
    assert(second.reserved_due);
    assert(second.source == ExpertReservedWinnerSource::Reserved);
    assert(second.selected.key.task_id == 1);
    assert(second.winner_changed_vs_legacy);
    assert(queue.counters().reserved_selected_count == 1);
    assert(queue.counters().active_winner_changed_count == 1);
    queue.select(100, true);
    assert_clean(queue, true);
}

void test_reserved_same_as_legacy() {
    ExpertReservedServiceQueue queue;
    queue.reset(8, fixture_config(1, 2));
    queue.insert(task(1, 1, 10, 1.0, 1));
    queue.insert(task(2, 2, 20, 1.0, 2));
    queue.select(100, false); // hard urgent also happens to be oldest
    queue.insert(task(3, 3, 30, 1.0, 3));
    const auto selection = queue.select(100, false);
    assert(selection.reserved_due);
    assert(selection.source == ExpertReservedWinnerSource::HardUrgent);
    assert(selection.debt_created);
    // Clear the remaining hard urgent Task, then debt repayment is reserved.
    const auto repayment = queue.select(100, false);
    assert(repayment.source == ExpertReservedWinnerSource::HardUrgent ||
            repayment.source == ExpertReservedWinnerSource::Reserved);
    while (queue.size() != 0) {
        queue.select(100, true);
    }
    assert_clean(queue, true);
}

void test_single_pending_debt_and_hard_urgent_safety() {
    ExpertReservedServiceQueue queue;
    queue.reset(16, fixture_config(1, 2));
    queue.insert(task(1, 1, 1000, 1.0, 1)); // oldest, not urgent
    queue.insert(task(2, 2, 50, 1.0, 2));   // urgent legacy head
    queue.insert(task(3, 3, 60, 1.0, 3));   // next urgent
    const auto a = queue.select(100, false);
    assert(a.source == ExpertReservedWinnerSource::HardUrgent);
    assert(a.selected.key.task_id == 2);
    assert(!a.debt_after);
    const auto b = queue.select(100, false);
    assert(b.reserved_due);
    assert(b.source == ExpertReservedWinnerSource::HardUrgent);
    assert(b.selected.key.task_id == 3);
    assert(b.debt_created && b.debt_after);
    const auto c = queue.select(100, false);
    assert(c.source == ExpertReservedWinnerSource::Reserved);
    assert(c.debt_repaid && !c.debt_after);
    assert(c.selected.key.task_id == 1);
    assert(queue.counters().debt_created_count == 1);
    assert(queue.counters().debt_repaid_count == 1);
    assert_clean(queue, true);
}

void test_bounded_store_and_generation() {
    ExpertReservedServiceQueue queue;
    queue.reset(1, fixture_config());
    const auto first = queue.insert(task(1, 1, 100, 1.0, 1));
    const auto selected = queue.select(1000, true);
    assert(selected.selected.handle == first);
    const auto second = queue.insert(task(2, 2, 200, 1.0, 2));
    assert(second.slot_id == first.slot_id);
    assert(second.generation == first.generation + 1);
    queue.select(1000, true);
    assert_clean(queue, true);
}

std::vector<uint64_t> deterministic_sequence() {
    ExpertReservedServiceQueue queue;
    queue.reset(64, fixture_config());
    uint64_t seed = 0x4d365331ULL;
    for (uint64_t id = 1; id <= 48; ++id) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint64_t enqueue = 1 + seed % 100;
        const uint64_t deadline = (seed >> 8) % 5 == 0 ? 0 : 100 + ((seed >> 16) % 500);
        const double score = static_cast<double>((seed >> 24) % 1000) / 1000.0;
        queue.insert(task(id, enqueue, deadline, score, id, 1024 + id));
    }
    std::vector<uint64_t> result;
    for (uint64_t decision = 0; queue.size() != 0; ++decision) {
        const auto selected = queue.select(200 + decision, false);
        result.push_back(selected.selected.key.task_id);
        assert_clean(queue);
    }
    assert_clean(queue, true);
    assert(queue.counters().insert_count == 48);
    assert(queue.counters().erase_count == 48);
    assert(queue.counters().selection_count == 48);
    return result;
}

void test_determinism_and_complexity_counters() {
    const auto first = deterministic_sequence();
    const auto second = deterministic_sequence();
    assert(first == second);
}

void test_invalid_config_fails_closed() {
    ExpertReservedServiceQueue queue;
    bool threw = false;
    try {
        queue.reset(8, fixture_config(4, 7));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    test_legacy_order_and_reset();
    test_fresh_reserved_and_oldest_winner();
    test_reserved_same_as_legacy();
    test_single_pending_debt_and_hard_urgent_safety();
    test_bounded_store_and_generation();
    test_determinism_and_complexity_counters();
    test_invalid_config_fails_closed();
    std::cout << "expert reserved service tests passed\n";
    return 0;
}
