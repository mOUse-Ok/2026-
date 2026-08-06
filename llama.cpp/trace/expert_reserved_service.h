#pragma once

#include "expert_hint_priority.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct ExpertReservedServiceConfig {
    uint64_t reserved_numerator = 1;
    uint64_t reserved_denominator = 8;
    uint64_t eligibility_age_ns = 41000000;
    uint64_t hard_urgent_guard_ns = 0;
};

struct ExpertReservedHandle {
    uint32_t slot_id = UINT32_MAX;
    uint64_t generation = 0;

    bool valid() const { return slot_id != UINT32_MAX && generation != 0; }
};

inline bool operator==(const ExpertReservedHandle & a, const ExpertReservedHandle & b) {
    return a.slot_id == b.slot_id && a.generation == b.generation;
}

inline bool operator!=(const ExpertReservedHandle & a, const ExpertReservedHandle & b) {
    return !(a == b);
}

struct ExpertReservedTaskKey {
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

enum class ExpertReservedWinnerSource {
    Legacy,
    Reserved,
    HardUrgent,
    ShutdownDrain,
};

const char * expert_reserved_winner_source_name(ExpertReservedWinnerSource source);

struct ExpertReservedTaskRef {
    bool available = false;
    ExpertReservedHandle handle;
    ExpertReservedTaskKey key;
};

struct ExpertReservedSelection {
    bool valid = false;
    uint64_t decision_ts_ns = 0;
    ExpertReservedWinnerSource source = ExpertReservedWinnerSource::Legacy;
    ExpertReservedTaskRef selected;
    ExpertReservedTaskRef legacy_head;
    ExpertReservedTaskRef aging_head;
    bool waiting_eligible = false;
    bool hard_urgent_present = false;
    bool reserved_triggered = false;
    bool reserved_due = false;
    bool winner_changed_vs_legacy = false;
    bool reserved_same_as_legacy = false;
    bool debt_created = false;
    bool debt_repaid = false;
    uint64_t credit_before = 0;
    uint64_t credit_accrued = 0;
    uint64_t credit_after = 0;
    bool debt_before = false;
    bool debt_after = false;
    uint64_t size_before = 0;
    uint64_t size_after = 0;
    uint64_t queued_bytes_before = 0;
    uint64_t queued_bytes_after = 0;
};

struct ExpertReservedServiceCounters {
    uint64_t insert_count = 0;
    uint64_t erase_count = 0;
    uint64_t selection_count = 0;
    uint64_t reserved_trigger_count = 0;
    uint64_t reserved_due_count = 0;
    uint64_t reserved_selected_count = 0;
    uint64_t active_winner_changed_count = 0;
    uint64_t reserved_same_as_legacy_count = 0;
    uint64_t hard_urgent_override_count = 0;
    uint64_t debt_created_count = 0;
    uint64_t debt_repaid_count = 0;
    uint64_t stale_handle_count = 0;
    uint64_t duplicate_erase_count = 0;
    uint64_t generation_mismatch_count = 0;
    uint64_t invariant_error_count = 0;
    uint64_t full_store_scan_count = 0;
    uint64_t legacy_heap_sift_count = 0;
    uint64_t aging_heap_sift_count = 0;
};

struct ExpertReservedServiceAudit {
    bool valid = false;
    bool final_queue_empty = false;
    uint64_t store_size = 0;
    uint64_t registry_size = 0;
    uint64_t legacy_index_size = 0;
    uint64_t aging_index_size = 0;
    uint64_t queued_bytes = 0;
    uint64_t live_bytes = 0;
};

// Queue-global pure Reserved-Service state backed by one bounded key store and
// two eager indexed binary heaps. The runtime Task entity remains in the owner
// at the returned stable slot; neither index copies that entity.
class ExpertReservedServiceQueue {
public:
    void reset(size_t capacity, const ExpertReservedServiceConfig & config);
    void clear();

    ExpertReservedHandle insert(const ExpertReservedTaskKey & key);
    ExpertReservedSelection select(uint64_t decision_ts_ns, bool shutdown_drain);

    size_t size() const { return live_count_; }
    uint64_t queued_bytes() const { return queued_bytes_; }
    uint64_t credit() const { return credit_; }
    bool pending_debt() const { return pending_debt_; }
    const ExpertReservedServiceConfig & config() const { return config_; }
    const ExpertReservedServiceCounters & counters() const { return counters_; }
    ExpertReservedServiceAudit audit(bool require_empty) const;

private:
    enum class IndexKind { Legacy, Aging };
    static constexpr size_t npos = static_cast<size_t>(-1);

    struct Slot {
        bool live = false;
        uint64_t generation = 0;
        ExpertReservedTaskKey key;
    };

    struct IndexedHeap {
        IndexKind kind = IndexKind::Legacy;
        std::vector<ExpertReservedHandle> heap;
        std::vector<size_t> position;
    };

    bool config_valid(const ExpertReservedServiceConfig & config) const;
    uint64_t saturating_add(uint64_t a, uint64_t b) const;
    Slot * resolve(ExpertReservedHandle handle);
    const Slot * resolve_const(ExpertReservedHandle handle) const;
    bool higher(IndexKind kind, ExpertReservedHandle a, ExpertReservedHandle b) const;
    void heap_swap(IndexedHeap & index, size_t a, size_t b);
    void sift_up(IndexedHeap & index, size_t position);
    void sift_down(IndexedHeap & index, size_t position);
    void index_insert(IndexedHeap & index, ExpertReservedHandle handle);
    void index_erase(IndexedHeap & index, ExpertReservedHandle handle);
    ExpertReservedTaskRef head(const IndexedHeap & index) const;
    bool eligible(const ExpertReservedTaskRef & ref, uint64_t now_ns) const;
    bool hard_urgent(const ExpertReservedTaskRef & ref, uint64_t now_ns) const;
    void fail_invariant(const char * message);

    ExpertReservedServiceConfig config_;
    std::vector<Slot> slots_;
    std::vector<uint32_t> free_slots_;
    std::unordered_map<uint64_t, ExpertReservedHandle> registry_;
    IndexedHeap legacy_;
    IndexedHeap aging_;
    size_t live_count_ = 0;
    uint64_t queued_bytes_ = 0;
    uint64_t credit_ = 0;
    bool pending_debt_ = false;
    ExpertReservedServiceCounters counters_;
};
