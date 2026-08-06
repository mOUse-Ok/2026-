#pragma once

#include "expert_hint_priority.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

// Estimated from the median positive adjacent route-score gap within identical
// Step/Layer/EARLY-or-LATE groups in the nine M6C-F B0 Runs:
// 0.0011134296655654907 / 50,000,000 ns.
constexpr double EXPERT_CONTINUOUS_AGING_ALPHA_PER_NS =
        2.2268593311309814e-11;

struct ExpertContinuousAgingConfig {
    double alpha_per_ns = EXPERT_CONTINUOUS_AGING_ALPHA_PER_NS;
    uint64_t score_epoch_ts_ns = 0;
};

struct ExpertContinuousAgingHandle {
    uint32_t slot_id = UINT32_MAX;
    uint64_t generation = 0;

    bool valid() const { return slot_id != UINT32_MAX && generation != 0; }
};

inline bool operator==(
        const ExpertContinuousAgingHandle & a,
        const ExpertContinuousAgingHandle & b) {
    return a.slot_id == b.slot_id && a.generation == b.generation;
}

inline bool operator!=(
        const ExpertContinuousAgingHandle & a,
        const ExpertContinuousAgingHandle & b) {
    return !(a == b);
}

struct ExpertContinuousAgingTaskKey {
    uint64_t task_id = 0;
    uint64_t step = 0;
    int layer = -1;
    ExpertTensorStage stage = ExpertTensorStage::Unknown;
    double route_score = 0.0;
    uint64_t sequence = 0;
    uint64_t deadline_ts_ns = 0;
    uint64_t enqueued_ts_ns = 0;
    uint64_t nbytes = 0;
};

struct ExpertContinuousAgingTaskRef {
    bool available = false;
    ExpertContinuousAgingHandle handle;
    ExpertContinuousAgingTaskKey key;
};

struct ExpertContinuousAgingSelection {
    bool valid = false;
    uint64_t decision_ts_ns = 0;
    ExpertContinuousAgingTaskRef selected;
    ExpertContinuousAgingTaskRef legacy_head;
    bool winner_changed_vs_legacy = false;
    bool legacy_head_hard_urgent = false;
    long double selected_static_score = 0.0L;
    long double selected_direct_adjusted_score = 0.0L;
    uint64_t size_before = 0;
    uint64_t size_after = 0;
    uint64_t queued_bytes_before = 0;
    uint64_t queued_bytes_after = 0;
};

struct ExpertContinuousAgingCounters {
    uint64_t insert_count = 0;
    uint64_t erase_count = 0;
    uint64_t selection_count = 0;
    uint64_t active_winner_changed_count = 0;
    uint64_t winner_same_as_legacy_count = 0;
    uint64_t hard_urgent_bypass_count = 0;
    uint64_t selected_after_deadline_count = 0;
    uint64_t stale_handle_count = 0;
    uint64_t duplicate_erase_count = 0;
    uint64_t generation_mismatch_count = 0;
    uint64_t invariant_error_count = 0;
    uint64_t full_store_scan_count = 0;
    uint64_t legacy_heap_sift_count = 0;
    uint64_t continuous_heap_sift_count = 0;
};

struct ExpertContinuousAgingAudit {
    bool valid = false;
    bool final_queue_empty = false;
    uint64_t store_size = 0;
    uint64_t registry_size = 0;
    uint64_t legacy_index_size = 0;
    uint64_t continuous_index_size = 0;
    uint64_t queued_bytes = 0;
    uint64_t live_bytes = 0;
};

int expert_continuous_aging_stage_rank(ExpertTensorStage stage);

long double expert_continuous_aging_static_score(
        const ExpertContinuousAgingTaskKey & key,
        const ExpertContinuousAgingConfig & config);

long double expert_continuous_aging_direct_adjusted_score(
        const ExpertContinuousAgingTaskKey & key,
        uint64_t decision_ts_ns,
        const ExpertContinuousAgingConfig & config);

bool expert_continuous_aging_higher_static(
        const ExpertContinuousAgingTaskKey & a,
        const ExpertContinuousAgingTaskKey & b,
        const ExpertContinuousAgingConfig & config);

bool expert_continuous_aging_higher_direct(
        const ExpertContinuousAgingTaskKey & a,
        const ExpertContinuousAgingTaskKey & b,
        uint64_t decision_ts_ns,
        const ExpertContinuousAgingConfig & config);

// A unique bounded key store plus two eager indexed heaps. Runtime Task
// entities remain uniquely owned by the caller at the returned stable slot.
class ExpertContinuousAgingQueue {
public:
    void reset(size_t capacity, const ExpertContinuousAgingConfig & config);
    void clear();

    ExpertContinuousAgingHandle insert(const ExpertContinuousAgingTaskKey & key);
    ExpertContinuousAgingSelection select(uint64_t decision_ts_ns);

    size_t size() const { return live_count_; }
    uint64_t queued_bytes() const { return queued_bytes_; }
    const ExpertContinuousAgingConfig & config() const { return config_; }
    const ExpertContinuousAgingCounters & counters() const { return counters_; }
    ExpertContinuousAgingAudit audit(bool require_empty) const;

private:
    enum class IndexKind { Legacy, Continuous };
    static constexpr size_t npos = static_cast<size_t>(-1);

    struct Slot {
        bool live = false;
        uint64_t generation = 0;
        ExpertContinuousAgingTaskKey key;
    };

    struct IndexedHeap {
        IndexKind kind = IndexKind::Legacy;
        std::vector<ExpertContinuousAgingHandle> heap;
        std::vector<size_t> position;
    };

    Slot * resolve(ExpertContinuousAgingHandle handle);
    const Slot * resolve_const(ExpertContinuousAgingHandle handle) const;
    bool higher(IndexKind kind, ExpertContinuousAgingHandle a,
            ExpertContinuousAgingHandle b) const;
    void heap_swap(IndexedHeap & index, size_t a, size_t b);
    void sift_up(IndexedHeap & index, size_t position);
    void sift_down(IndexedHeap & index, size_t position);
    void index_insert(IndexedHeap & index, ExpertContinuousAgingHandle handle);
    void index_erase(IndexedHeap & index, ExpertContinuousAgingHandle handle);
    ExpertContinuousAgingTaskRef head(const IndexedHeap & index) const;
    [[noreturn]] void fail_invariant(const char * message);

    ExpertContinuousAgingConfig config_;
    std::vector<Slot> slots_;
    std::vector<uint32_t> free_slots_;
    std::unordered_map<uint64_t, ExpertContinuousAgingHandle> registry_;
    IndexedHeap legacy_;
    IndexedHeap continuous_;
    size_t live_count_ = 0;
    uint64_t queued_bytes_ = 0;
    ExpertContinuousAgingCounters counters_;
};
