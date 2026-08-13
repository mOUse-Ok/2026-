#include "expert_memory_object.h"

#include <algorithm>
#include <limits>
#include <utility>

ExpertMemoryObjectTracker::ExpertMemoryObjectTracker(uint64_t working_set_budget_bytes) {
    counters_.working_set_budget_bytes = working_set_budget_bytes;
}

std::string ExpertMemoryObjectTracker::object_key(
        int layer, int expert, const std::string & tensor) {
    return std::to_string(layer) + ":" + std::to_string(expert) + ":" + tensor;
}

void ExpertMemoryObjectTracker::subtract_unlocked(uint64_t & value, uint64_t amount) {
    if (amount > value) {
        value = 0;
        counters_.invariant_violations++;
        return;
    }
    value -= amount;
}

void ExpertMemoryObjectTracker::touch_unlocked(ExpertMemoryObject & object) {
    if (next_touch_seq_ != std::numeric_limits<uint64_t>::max()) {
        next_touch_seq_++;
    }
    object.last_touch_seq = next_touch_seq_;
}

void ExpertMemoryObjectTracker::admit_to_working_set_unlocked(
        ExpertMemoryObject & object,
        bool object_was_created,
        uint64_t step) {
    if (counters_.working_set_budget_bytes == 0 || object.in_working_set) {
        return;
    }

    if (!object_was_created && object.has_shadow_eviction_record) {
        subtract_unlocked(counters_.current_probation_objects, 1);
        if (object.dontneed_hint_issued_for_current_eviction) {
            counters_.post_dontneed_readmissions++;
        } else if (object.cold_hint_issued_for_current_eviction) {
            counters_.post_cold_readmissions++;
        } else {
            counters_.probation_canceled_by_readmission++;
        }
        object.cold_hint_issued_for_current_eviction = false;
        object.dontneed_hint_issued_for_current_eviction = false;
    }

    object.in_working_set = true;
    counters_.working_set_admissions++;
    if (!object_was_created) {
        counters_.working_set_readmissions++;
        // Observation-only: reuse distance since the last shadow eviction.
        if (!object.has_shadow_eviction_record) {
            counters_.readmission_gap_no_record++;
        } else {
            const uint64_t gap = step - object.last_shadow_eviction_step;
            if (gap == 0) {
                counters_.readmission_gap_0++;
            } else if (gap == 1) {
                counters_.readmission_gap_1++;
            } else if (gap <= 3) {
                counters_.readmission_gap_2_3++;
            } else if (gap <= 7) {
                counters_.readmission_gap_4_7++;
            } else {
                counters_.readmission_gap_8_plus++;
            }
            if (gap <= 1) {
                counters_.readmissions_within_1_step++;
            }
            if (gap <= 3) {
                counters_.readmissions_within_3_steps++;
            }
        }
    }
    if (object.nbytes > std::numeric_limits<uint64_t>::max() -
            counters_.working_set_current_bytes) {
        counters_.working_set_current_bytes = std::numeric_limits<uint64_t>::max();
        counters_.invariant_violations++;
    } else {
        counters_.working_set_current_bytes += (uint64_t) object.nbytes;
    }
    counters_.working_set_objects++;
    counters_.working_set_peak_bytes = std::max(
            counters_.working_set_peak_bytes, counters_.working_set_current_bytes);
    counters_.working_set_peak_objects = std::max(
            counters_.working_set_peak_objects, counters_.working_set_objects);
}

void ExpertMemoryObjectTracker::evict_to_working_set_budget_unlocked(uint64_t step) {
    const uint64_t budget = counters_.working_set_budget_bytes;
    while (counters_.working_set_current_bytes > budget) {
        counters_.working_set_lru_scans++;
        ExpertMemoryObject * victim = nullptr;
        std::string victim_key;
        for (auto & entry : objects_) {
            const std::string & key = entry.first;
            ExpertMemoryObject & object = entry.second;
            if (!object.in_working_set) {
                continue;
            }
            if (object.pending_users > 0 || object.active_users > 0) {
                counters_.working_set_protected_skips++;
                continue;
            }
            if (!victim || object.last_touch_seq < victim->last_touch_seq ||
                    (object.last_touch_seq == victim->last_touch_seq && key < victim_key)) {
                victim = &object;
                victim_key = key;
            }
        }

        if (!victim) {
            counters_.budget_unresolved_due_to_protection++;
            return;
        }

        victim->in_working_set = false;
        victim->last_shadow_eviction_step = step;
        victim->has_shadow_eviction_record = true;
        victim->cold_hint_issued_for_current_eviction = false;
        victim->dontneed_hint_issued_for_current_eviction = false;
        victim->eligible_counted_for_current_eviction = false;
        subtract_unlocked(counters_.working_set_current_bytes, (uint64_t) victim->nbytes);
        subtract_unlocked(counters_.working_set_objects, 1);
        counters_.working_set_evictions++;
        counters_.probation_entries++;
        counters_.current_probation_objects++;
        counters_.peak_probation_objects = std::max(
                counters_.peak_probation_objects, counters_.current_probation_objects);
        if (victim->nbytes > std::numeric_limits<uint64_t>::max() -
                counters_.working_set_evicted_bytes) {
            counters_.working_set_evicted_bytes = std::numeric_limits<uint64_t>::max();
            counters_.invariant_violations++;
        } else {
            counters_.working_set_evicted_bytes += (uint64_t) victim->nbytes;
        }
    }
}

bool ExpertMemoryObjectTracker::register_demand(
        uint64_t step,
        int layer,
        int expert,
        const std::string & tensor,
        uintptr_t addr,
        size_t nbytes) {
    if (layer < 0 || expert < 0 || tensor.empty() || addr == 0 || nbytes == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mu_);
    const std::string key = object_key(layer, expert, tensor);
    auto entry = objects_.find(key);
    bool object_was_created = false;
    if (entry == objects_.end()) {
        ExpertMemoryObject object;
        object.layer = layer;
        object.expert = expert;
        object.tensor = tensor;
        object.addr = addr;
        object.nbytes = nbytes;
        entry = objects_.emplace(key, std::move(object)).first;
        counters_.memory_objects_created++;
        object_was_created = true;
    }

    ExpertMemoryObject & object = entry->second;
    if (object.has_demand_step && object.last_demand_step == step) {
        counters_.semantic_demands_merged++;
        return false;
    }

    object.has_demand_step = true;
    object.last_demand_step = step;
    if (object.pending_users == 0) {
        counters_.pending_objects++;
        counters_.peak_pending_objects =
                std::max(counters_.peak_pending_objects, counters_.pending_objects);
    }
    object.pending_users++;
    counters_.pending++;
    counters_.semantic_demands_registered++;
    touch_unlocked(object);
    admit_to_working_set_unlocked(object, object_was_created, step);
    evict_to_working_set_budget_unlocked(step);
    return true;
}

bool ExpertMemoryObjectTracker::observe_first_use(
        uint64_t step,
        int layer,
        int expert,
        const std::string & tensor) {
    if (layer < 0 || expert < 0 || tensor.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mu_);
    const auto entry = objects_.find(object_key(layer, expert, tensor));
    if (entry == objects_.end()) {
        counters_.unmatched_first_use++;
        return false;
    }

    ExpertMemoryObject & object = entry->second;
    if (object.has_use_step && object.last_use_step == step) {
        return false;
    }
    object.has_use_step = true;
    object.last_use_step = step;
    if (!object.has_demand_step || object.last_demand_step != step || object.pending_users == 0) {
        counters_.unmatched_first_use++;
        return false;
    }

    object.pending_users--;
    subtract_unlocked(counters_.pending, 1);
    if (object.pending_users == 0) {
        subtract_unlocked(counters_.pending_objects, 1);
    }
    if (object.active_users == 0) {
        counters_.active_objects++;
        counters_.peak_active_objects =
                std::max(counters_.peak_active_objects, counters_.active_objects);
    }
    object.active_users++;
    counters_.active++;
    counters_.demand_activations++;
    touch_unlocked(object);
    return true;
}

bool ExpertMemoryObjectTracker::try_acquire_hint_slot(
        int layer,
        int expert,
        const std::string & tensor) {
    if (layer < 0 || expert < 0 || tensor.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mu_);
    const auto entry = objects_.find(object_key(layer, expert, tensor));
    if (entry == objects_.end()) {
        counters_.invariant_violations++;
        return false;
    }

    ExpertMemoryObject & object = entry->second;
    if (object.hint_inflight) {
        counters_.inflight_hint_aggregated++;
        return false;
    }

    object.hint_inflight = true;
    counters_.hint_slots_acquired++;
    counters_.current_hint_inflight_objects++;
    counters_.peak_hint_inflight_objects = std::max(
            counters_.peak_hint_inflight_objects,
            counters_.current_hint_inflight_objects);
    return true;
}

bool ExpertMemoryObjectTracker::release_hint_slot(
        int layer,
        int expert,
        const std::string & tensor,
        bool terminal_canceled) {
    if (layer < 0 || expert < 0 || tensor.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mu_);
    const auto entry = objects_.find(object_key(layer, expert, tensor));
    if (entry == objects_.end() || !entry->second.hint_inflight) {
        counters_.invariant_violations++;
        return false;
    }

    entry->second.hint_inflight = false;
    counters_.hint_slots_released++;
    if (terminal_canceled) {
        counters_.hint_terminal_canceled++;
    }
    subtract_unlocked(counters_.current_hint_inflight_objects, 1);
    return true;
}

bool ExpertMemoryObjectTracker::has_live_demand(
        int layer,
        int expert,
        const std::string & tensor) const {
    if (layer < 0 || expert < 0 || tensor.empty()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mu_);
    const auto entry = objects_.find(object_key(layer, expert, tensor));
    return entry != objects_.end() &&
            (entry->second.pending_users > 0 || entry->second.active_users > 0);
}

void ExpertMemoryObjectTracker::record_semantic_stale_check(bool live) {
    std::lock_guard<std::mutex> lock(mu_);
    counters_.semantic_stale_checked++;
    if (live) {
        counters_.semantic_stale_kept_live++;
    }
}

void ExpertMemoryObjectTracker::record_semantic_stale_cancel(size_t nbytes) {
    std::lock_guard<std::mutex> lock(mu_);
    counters_.semantic_stale_tasks_canceled++;
    counters_.semantic_stale_bytes_avoided += (uint64_t) nbytes;
}

void ExpertMemoryObjectTracker::end_layer_unlocked(int layer) {
    for (auto & entry : objects_) {
        ExpertMemoryObject & object = entry.second;
        if (object.layer != layer) {
            continue;
        }
        if (object.active_users > 0) {
            counters_.demand_completions += object.active_users;
            subtract_unlocked(counters_.active, object.active_users);
            subtract_unlocked(counters_.active_objects, 1);
            object.active_users = 0;
        }
        if (object.pending_users > 0) {
            counters_.stale_pending_canceled += object.pending_users;
            subtract_unlocked(counters_.pending, object.pending_users);
            subtract_unlocked(counters_.pending_objects, 1);
            object.pending_users = 0;
        }
    }
}

void ExpertMemoryObjectTracker::end_layer(int layer) {
    if (layer < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(mu_);
    end_layer_unlocked(layer);
}

std::vector<ExpertMadVColdCandidate>
ExpertMemoryObjectTracker::end_layer_and_collect_madv_cold_candidates(
        int layer,
        uint64_t step,
        uint64_t grace_steps,
        uint64_t max_collect_bytes) {
    std::vector<ExpertMadVColdCandidate> candidates;
    if (layer < 0) {
        return candidates;
    }

    std::lock_guard<std::mutex> lock(mu_);
    end_layer_unlocked(layer);
    uint64_t collected_bytes = 0;
    for (auto & entry : objects_) {
        ExpertMemoryObject & object = entry.second;
        if (object.layer != layer || object.in_working_set ||
                !object.has_shadow_eviction_record ||
                object.cold_hint_issued_for_current_eviction) {
            continue;
        }
        if (object.pending_users > 0 || object.active_users > 0) {
            counters_.cold_protected_violation++;
            continue;
        }
        if (step < object.last_shadow_eviction_step) {
            counters_.invariant_violations++;
            continue;
        }
        if (step - object.last_shadow_eviction_step < grace_steps) {
            continue;
        }

        // Phase 2E-A observation-only: this object passed all eligibility
        // filters (working-set, shadow-eviction record, pending/active safety,
        // grace). Accumulate its bytes BEFORE the budget deferral gate so the
        // calibration profile can learn the true per-step COLD-eligible scale,
        // regardless of whether the budget actually admits it. Count each
        // eviction episode at most once: scan-only (defer-all) modes revisit
        // the same unmarked episode at every later layer end, and recounting
        // would inflate the eligible-bytes scale far beyond the true
        // per-step maturation rate.
        if (!object.eligible_counted_for_current_eviction) {
            object.eligible_counted_for_current_eviction = true;
            if (object.nbytes > std::numeric_limits<uint64_t>::max() -
                    counters_.cold_eligible_candidate_bytes) {
                counters_.cold_eligible_candidate_bytes =
                        std::numeric_limits<uint64_t>::max();
                counters_.invariant_violations++;
            } else {
                counters_.cold_eligible_candidate_bytes += (uint64_t) object.nbytes;
            }
        }

        if (max_collect_bytes > 0 &&
                (object.nbytes > max_collect_bytes ||
                 collected_bytes > max_collect_bytes - object.nbytes)) {
            // Budget-limited: leave the episode un-issued for a later layer end.
            counters_.madv_cold_budget_deferred_candidates++;
            counters_.madv_cold_budget_deferred_bytes += (uint64_t) object.nbytes;
            continue;
        }

        object.cold_hint_issued_for_current_eviction = true;
        collected_bytes += (uint64_t) object.nbytes;
        counters_.madv_cold_candidates++;
        ExpertMadVColdCandidate candidate;
        candidate.layer = object.layer;
        candidate.expert = object.expert;
        candidate.tensor = object.tensor;
        candidate.addr = object.addr;
        candidate.nbytes = object.nbytes;
        candidates.push_back(std::move(candidate));
    }
    return candidates;
}

void ExpertMemoryObjectTracker::record_madv_cold_result(bool issued, size_t nbytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!issued) {
        counters_.madv_cold_failed++;
        return;
    }
    counters_.madv_cold_issued++;
    if (nbytes > std::numeric_limits<uint64_t>::max() - counters_.madv_cold_bytes) {
        counters_.madv_cold_bytes = std::numeric_limits<uint64_t>::max();
        counters_.invariant_violations++;
    } else {
        counters_.madv_cold_bytes += (uint64_t) nbytes;
    }
}

std::vector<ExpertMadVDontNeedCandidate>
ExpertMemoryObjectTracker::end_layer_and_collect_madv_dontneed_candidates(
        int layer,
        uint64_t step,
        uint64_t grace_steps,
        uint64_t max_collect_bytes) {
    std::vector<ExpertMadVDontNeedCandidate> candidates;
    if (layer < 0 || max_collect_bytes == 0) {
        return candidates;
    }

    std::lock_guard<std::mutex> lock(mu_);
    end_layer_unlocked(layer);
    uint64_t collected_bytes = 0;
    for (auto & entry : objects_) {
        ExpertMemoryObject & object = entry.second;
        if (object.layer != layer || object.in_working_set ||
                !object.has_shadow_eviction_record ||
                object.dontneed_hint_issued_for_current_eviction) {
            continue;
        }
        if (object.pending_users > 0 || object.active_users > 0) {
            counters_.madv_dontneed_protected_skipped++;
            continue;
        }
        if (object.hint_inflight) {
            counters_.madv_dontneed_inflight_skipped++;
            continue;
        }
        if (step < object.last_shadow_eviction_step) {
            counters_.invariant_violations++;
            continue;
        }
        if (step - object.last_shadow_eviction_step < grace_steps) {
            continue;
        }
        if (object.nbytes > max_collect_bytes ||
                collected_bytes > max_collect_bytes - object.nbytes) {
            // Leave the episode available to a later Decode step.  The caller
            // supplies the remaining whole-step budget, not a per-layer cap.
            counters_.madv_dontneed_budget_deferred_candidates++;
            counters_.madv_dontneed_budget_deferred_bytes += (uint64_t) object.nbytes;
            continue;
        }

        object.dontneed_hint_issued_for_current_eviction = true;
        collected_bytes += (uint64_t) object.nbytes;
        counters_.madv_dontneed_candidates++;
        ExpertMadVDontNeedCandidate candidate;
        candidate.layer = object.layer;
        candidate.expert = object.expert;
        candidate.tensor = object.tensor;
        candidate.addr = object.addr;
        candidate.nbytes = object.nbytes;
        candidates.push_back(std::move(candidate));
    }
    return candidates;
}

void ExpertMemoryObjectTracker::record_madv_dontneed_result(bool issued, size_t advised_bytes) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!issued) {
        counters_.madv_dontneed_failed++;
        return;
    }
    counters_.madv_dontneed_issued++;
    if (advised_bytes > std::numeric_limits<uint64_t>::max() - counters_.madv_dontneed_bytes) {
        counters_.madv_dontneed_bytes = std::numeric_limits<uint64_t>::max();
        counters_.invariant_violations++;
    } else {
        counters_.madv_dontneed_bytes += (uint64_t) advised_bytes;
    }
}

void ExpertMemoryObjectTracker::record_madv_dontneed_mapping_rejected() {
    std::lock_guard<std::mutex> lock(mu_);
    counters_.madv_dontneed_mapping_rejected++;
}

void ExpertMemoryObjectTracker::record_madv_dontneed_inner_page_skipped() {
    std::lock_guard<std::mutex> lock(mu_);
    counters_.madv_dontneed_inner_page_skipped++;
}

void ExpertMemoryObjectTracker::record_cold_skipped_ttl_nonzero() {
    std::lock_guard<std::mutex> lock(mu_);
    counters_.cold_skipped_ttl_nonzero++;
}

ExpertMemoryObjectCounters ExpertMemoryObjectTracker::counters() const {
    std::lock_guard<std::mutex> lock(mu_);
    return counters_;
}
