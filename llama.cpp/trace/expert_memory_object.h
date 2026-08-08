#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct ExpertMemoryObject {
    int layer = -1;
    int expert = -1;
    std::string tensor;
    uintptr_t addr = 0;
    size_t nbytes = 0;
    uint64_t pending_users = 0;
    uint64_t active_users = 0;
    uint64_t last_demand_step = 0;
    uint64_t last_use_step = 0;
    uint64_t last_touch_seq = 0;
    bool has_demand_step = false;
    bool has_use_step = false;
    bool hint_inflight = false;
    // Logical Falcons working-set membership; this does not describe physical residency.
    bool in_working_set = false;
    // Semantic step of the most recent shadow eviction / probation episode.
    uint64_t last_shadow_eviction_step = 0;
    bool has_shadow_eviction_record = false;
    // Set before the external MADV_COLD syscall so one probation episode has
    // at most one cold-hint attempt.
    bool cold_hint_issued_for_current_eviction = false;
    // Set when this episode's bytes have been counted into
    // cold_eligible_candidate_bytes; prevents scan-only modes from counting
    // the same deferred episode once per layer end. Reset on each eviction.
    bool eligible_counted_for_current_eviction = false;
};

struct ExpertMadVColdCandidate {
    int layer = -1;
    int expert = -1;
    std::string tensor;
    uintptr_t addr = 0;
    size_t nbytes = 0;
};

struct ExpertMemoryObjectCounters {
    uint64_t memory_objects_created = 0;
    uint64_t semantic_demands_registered = 0;
    uint64_t semantic_demands_merged = 0;
    uint64_t demand_activations = 0;
    uint64_t demand_completions = 0;
    uint64_t stale_pending_canceled = 0;
    uint64_t unmatched_first_use = 0;
    uint64_t invariant_violations = 0;
    uint64_t pending = 0;
    uint64_t active = 0;
    uint64_t pending_objects = 0;
    uint64_t active_objects = 0;
    uint64_t peak_pending_objects = 0;
    uint64_t peak_active_objects = 0;
    uint64_t hint_slots_acquired = 0;
    uint64_t inflight_hint_aggregated = 0;
    uint64_t hint_slots_released = 0;
    uint64_t hint_terminal_canceled = 0;
    uint64_t current_hint_inflight_objects = 0;
    uint64_t peak_hint_inflight_objects = 0;
    uint64_t semantic_stale_checked = 0;
    uint64_t semantic_stale_kept_live = 0;
    uint64_t semantic_stale_tasks_canceled = 0;
    uint64_t semantic_stale_bytes_avoided = 0;
    uint64_t working_set_budget_bytes = 0;
    uint64_t working_set_current_bytes = 0;
    uint64_t working_set_peak_bytes = 0;
    uint64_t working_set_objects = 0;
    uint64_t working_set_peak_objects = 0;
    uint64_t working_set_admissions = 0;
    uint64_t working_set_readmissions = 0;
    uint64_t working_set_evictions = 0;
    uint64_t working_set_evicted_bytes = 0;
    uint64_t working_set_protected_skips = 0;
    uint64_t budget_unresolved_due_to_protection = 0;
    uint64_t working_set_lru_scans = 0;
    // Observation-only readmission reuse-distance buckets (gap in semantic steps
    // between last shadow eviction and readmission).
    uint64_t readmission_gap_0 = 0;
    uint64_t readmission_gap_1 = 0;
    uint64_t readmission_gap_2_3 = 0;
    uint64_t readmission_gap_4_7 = 0;
    uint64_t readmission_gap_8_plus = 0;
    uint64_t readmission_gap_no_record = 0;
    uint64_t readmissions_within_1_step = 0;
    uint64_t readmissions_within_3_steps = 0;
    uint64_t probation_entries = 0;
    uint64_t probation_canceled_by_readmission = 0;
    uint64_t madv_cold_candidates = 0;
    uint64_t madv_cold_issued = 0;
    uint64_t madv_cold_failed = 0;
    uint64_t madv_cold_bytes = 0;
    uint64_t post_cold_readmissions = 0;
    uint64_t current_probation_objects = 0;
    uint64_t peak_probation_objects = 0;
    uint64_t cold_skipped_ttl_nonzero = 0;
    uint64_t cold_protected_violation = 0;
    // Observation-only: matured candidates deferred by a re-entry byte budget.
    // Deferred episodes stay un-issued and are reconsidered at a later layer end.
    uint64_t madv_cold_budget_deferred_candidates = 0;
    uint64_t madv_cold_budget_deferred_bytes = 0;
    // Phase 2E-A observation-only: total bytes of candidates that passed all
    // eligibility filters (working-set, LRU, grace, pending/active safety) —
    // counted BEFORE the budget deferral gate. This is the per-step "how much
    // COLD was actually eligible" scale, distinct from the budget-limited
    // madv_cold_bytes that was actually issued. Accumulates across all steps.
    uint64_t cold_eligible_candidate_bytes = 0;
};

class ExpertMemoryObjectTracker {
public:
    explicit ExpertMemoryObjectTracker(uint64_t working_set_budget_bytes = 0);
    bool register_demand(
            uint64_t step,
            int layer,
            int expert,
            const std::string & tensor,
            uintptr_t addr,
            size_t nbytes);
    bool observe_first_use(
            uint64_t step,
            int layer,
            int expert,
            const std::string & tensor);
    bool try_acquire_hint_slot(int layer, int expert, const std::string & tensor);
    bool release_hint_slot(
            int layer,
            int expert,
            const std::string & tensor,
            bool terminal_canceled);
    bool has_live_demand(int layer, int expert, const std::string & tensor) const;
    void record_semantic_stale_check(bool live);
    void record_semantic_stale_cancel(size_t nbytes);
    void end_layer(int layer);
    std::vector<ExpertMadVColdCandidate> end_layer_and_collect_madv_cold_candidates(
            int layer,
            uint64_t step,
            uint64_t grace_steps,
            uint64_t max_collect_bytes = 0);
    void record_madv_cold_result(bool issued, size_t nbytes);
    void record_cold_skipped_ttl_nonzero();
    ExpertMemoryObjectCounters counters() const;

private:
    static std::string object_key(int layer, int expert, const std::string & tensor);
    void subtract_unlocked(uint64_t & value, uint64_t amount);
    void touch_unlocked(ExpertMemoryObject & object);
    void admit_to_working_set_unlocked(
            ExpertMemoryObject & object,
            bool object_was_created,
            uint64_t step);
    void evict_to_working_set_budget_unlocked(uint64_t step);
    void end_layer_unlocked(int layer);

    mutable std::mutex mu_;
    std::unordered_map<std::string, ExpertMemoryObject> objects_;
    ExpertMemoryObjectCounters counters_;
    uint64_t next_touch_seq_ = 0;
};
