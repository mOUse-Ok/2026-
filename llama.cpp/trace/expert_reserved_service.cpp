#include "expert_reserved_service.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

const char * expert_reserved_winner_source_name(ExpertReservedWinnerSource source) {
    switch (source) {
        case ExpertReservedWinnerSource::Legacy:        return "legacy";
        case ExpertReservedWinnerSource::Reserved:      return "reserved";
        case ExpertReservedWinnerSource::HardUrgent:    return "hard_urgent";
        case ExpertReservedWinnerSource::ShutdownDrain: return "shutdown_drain";
    }
    return "legacy";
}

bool ExpertReservedServiceQueue::config_valid(const ExpertReservedServiceConfig & config) const {
    return config.reserved_numerator > 0 &&
            config.reserved_numerator < config.reserved_denominator &&
            config.reserved_numerator <= config.reserved_denominator / 2;
}

uint64_t ExpertReservedServiceQueue::saturating_add(uint64_t a, uint64_t b) const {
    return a > std::numeric_limits<uint64_t>::max() - b ?
            std::numeric_limits<uint64_t>::max() : a + b;
}

void ExpertReservedServiceQueue::reset(
        size_t capacity,
        const ExpertReservedServiceConfig & config) {
    if (capacity == 0 || capacity > UINT32_MAX || !config_valid(config)) {
        throw std::invalid_argument("invalid Reserved-Service queue configuration");
    }
    config_ = config;
    slots_.assign(capacity, Slot{});
    free_slots_.clear();
    free_slots_.reserve(capacity);
    for (size_t i = capacity; i > 0; --i) {
        free_slots_.push_back(static_cast<uint32_t>(i - 1));
    }
    registry_.clear();
    registry_.reserve(capacity);
    legacy_ = {};
    legacy_.kind = IndexKind::Legacy;
    legacy_.heap.reserve(capacity);
    legacy_.position.assign(capacity, npos);
    aging_ = {};
    aging_.kind = IndexKind::Aging;
    aging_.heap.reserve(capacity);
    aging_.position.assign(capacity, npos);
    live_count_ = 0;
    queued_bytes_ = 0;
    credit_ = 0;
    pending_debt_ = false;
    counters_ = {};
}

void ExpertReservedServiceQueue::clear() {
    slots_.clear();
    free_slots_.clear();
    registry_.clear();
    legacy_ = {};
    aging_ = {};
    live_count_ = 0;
    queued_bytes_ = 0;
    credit_ = 0;
    pending_debt_ = false;
}

ExpertReservedServiceQueue::Slot * ExpertReservedServiceQueue::resolve(
        ExpertReservedHandle handle) {
    if (!handle.valid() || handle.slot_id >= slots_.size()) {
        counters_.stale_handle_count++;
        fail_invariant("stale Reserved-Service handle");
    }
    Slot & slot = slots_[handle.slot_id];
    if (!slot.live) {
        counters_.stale_handle_count++;
        fail_invariant("handle references free Reserved-Service slot");
    }
    if (slot.generation != handle.generation) {
        counters_.generation_mismatch_count++;
        fail_invariant("Reserved-Service generation mismatch");
    }
    return &slot;
}

const ExpertReservedServiceQueue::Slot * ExpertReservedServiceQueue::resolve_const(
        ExpertReservedHandle handle) const {
    if (!handle.valid() || handle.slot_id >= slots_.size()) {
        throw std::logic_error("stale Reserved-Service handle");
    }
    const Slot & slot = slots_[handle.slot_id];
    if (!slot.live || slot.generation != handle.generation) {
        throw std::logic_error("Reserved-Service generation mismatch");
    }
    return &slot;
}

bool ExpertReservedServiceQueue::higher(
        IndexKind kind,
        ExpertReservedHandle a,
        ExpertReservedHandle b) const {
    const ExpertReservedTaskKey & ka = resolve_const(a)->key;
    const ExpertReservedTaskKey & kb = resolve_const(b)->key;
    if (kind == IndexKind::Aging) {
        if (ka.enqueued_ts_ns != kb.enqueued_ts_ns) {
            return ka.enqueued_ts_ns < kb.enqueued_ts_ns;
        }
        if (ka.sequence != kb.sequence) {
            return ka.sequence < kb.sequence;
        }
        return ka.task_id < kb.task_id;
    }
    ExpertHintPriorityKey pa;
    pa.step = ka.step;
    pa.layer = ka.layer;
    pa.stage = ka.stage;
    pa.route_score = ka.route_score;
    pa.sequence = ka.sequence;
    pa.deadline_ts_ns = ka.deadline_ts_ns;
    ExpertHintPriorityKey pb;
    pb.step = kb.step;
    pb.layer = kb.layer;
    pb.stage = kb.stage;
    pb.route_score = kb.route_score;
    pb.sequence = kb.sequence;
    pb.deadline_ts_ns = kb.deadline_ts_ns;
    return expert_hint_priority_higher(pa, pb, ExpertAsyncPriorityMode::DeadlineScore);
}

void ExpertReservedServiceQueue::heap_swap(IndexedHeap & index, size_t a, size_t b) {
    std::swap(index.heap[a], index.heap[b]);
    index.position[index.heap[a].slot_id] = a;
    index.position[index.heap[b].slot_id] = b;
}

void ExpertReservedServiceQueue::sift_up(IndexedHeap & index, size_t position) {
    uint64_t & count = index.kind == IndexKind::Legacy ?
            counters_.legacy_heap_sift_count : counters_.aging_heap_sift_count;
    while (position > 0) {
        const size_t parent = (position - 1) / 2;
        count++;
        if (!higher(index.kind, index.heap[position], index.heap[parent])) {
            break;
        }
        heap_swap(index, position, parent);
        position = parent;
    }
}

void ExpertReservedServiceQueue::sift_down(IndexedHeap & index, size_t position) {
    uint64_t & count = index.kind == IndexKind::Legacy ?
            counters_.legacy_heap_sift_count : counters_.aging_heap_sift_count;
    for (;;) {
        const size_t left = position * 2 + 1;
        if (left >= index.heap.size()) {
            return;
        }
        const size_t right = left + 1;
        size_t best = left;
        if (right < index.heap.size()) {
            count++;
            if (higher(index.kind, index.heap[right], index.heap[left])) {
                best = right;
            }
        }
        count++;
        if (!higher(index.kind, index.heap[best], index.heap[position])) {
            return;
        }
        heap_swap(index, position, best);
        position = best;
    }
}

void ExpertReservedServiceQueue::index_insert(
        IndexedHeap & index,
        ExpertReservedHandle handle) {
    if (index.position[handle.slot_id] != npos) {
        fail_invariant("duplicate Reserved-Service index insertion");
    }
    index.position[handle.slot_id] = index.heap.size();
    index.heap.push_back(handle);
    sift_up(index, index.heap.size() - 1);
}

void ExpertReservedServiceQueue::index_erase(
        IndexedHeap & index,
        ExpertReservedHandle handle) {
    if (handle.slot_id >= index.position.size()) {
        counters_.stale_handle_count++;
        fail_invariant("Reserved-Service erase slot out of range");
    }
    const size_t position = index.position[handle.slot_id];
    if (position == npos || position >= index.heap.size() ||
            index.heap[position] != handle) {
        counters_.duplicate_erase_count++;
        fail_invariant("Reserved-Service duplicate or mismatched erase");
    }
    const size_t last = index.heap.size() - 1;
    if (position != last) {
        heap_swap(index, position, last);
    }
    index.heap.pop_back();
    index.position[handle.slot_id] = npos;
    if (position < index.heap.size()) {
        if (position > 0 && higher(
                    index.kind, index.heap[position], index.heap[(position - 1) / 2])) {
            sift_up(index, position);
        } else {
            sift_down(index, position);
        }
    }
}

ExpertReservedTaskRef ExpertReservedServiceQueue::head(const IndexedHeap & index) const {
    ExpertReservedTaskRef result;
    if (index.heap.empty()) {
        return result;
    }
    result.available = true;
    result.handle = index.heap.front();
    result.key = resolve_const(result.handle)->key;
    return result;
}

bool ExpertReservedServiceQueue::eligible(
        const ExpertReservedTaskRef & ref,
        uint64_t now_ns) const {
    return ref.available && ref.key.enqueued_ts_ns != 0 &&
            now_ns >= saturating_add(ref.key.enqueued_ts_ns, config_.eligibility_age_ns);
}

bool ExpertReservedServiceQueue::hard_urgent(
        const ExpertReservedTaskRef & ref,
        uint64_t now_ns) const {
    return ref.available && ref.key.deadline_ts_ns != 0 &&
            ref.key.deadline_ts_ns <= saturating_add(now_ns, config_.hard_urgent_guard_ns);
}

ExpertReservedHandle ExpertReservedServiceQueue::insert(const ExpertReservedTaskKey & key) {
    if (slots_.empty() || free_slots_.empty() || key.task_id == 0 ||
            registry_.find(key.task_id) != registry_.end()) {
        fail_invariant("invalid or full Reserved-Service insertion");
    }
    const uint32_t slot_id = free_slots_.back();
    free_slots_.pop_back();
    Slot & slot = slots_[slot_id];
    if (slot.live || slot.generation == std::numeric_limits<uint64_t>::max()) {
        fail_invariant("Reserved-Service slot generation unavailable");
    }
    slot.live = true;
    slot.generation++;
    slot.key = key;
    const ExpertReservedHandle handle{slot_id, slot.generation};
    try {
        registry_.emplace(key.task_id, handle);
        index_insert(legacy_, handle);
        index_insert(aging_, handle);
    } catch (...) {
        if (aging_.position[slot_id] != npos) {
            index_erase(aging_, handle);
        }
        if (legacy_.position[slot_id] != npos) {
            index_erase(legacy_, handle);
        }
        registry_.erase(key.task_id);
        slot.live = false;
        slot.key = {};
        free_slots_.push_back(slot_id);
        throw;
    }
    live_count_++;
    queued_bytes_ = saturating_add(queued_bytes_, key.nbytes);
    counters_.insert_count++;
    return handle;
}

ExpertReservedSelection ExpertReservedServiceQueue::select(
        uint64_t decision_ts_ns,
        bool shutdown_drain) {
    if (live_count_ == 0) {
        fail_invariant("selection from empty Reserved-Service queue");
    }
    if (credit_ >= config_.reserved_denominator ||
            (pending_debt_ && credit_ >= config_.reserved_numerator)) {
        fail_invariant("invalid queue-global credit/debt state");
    }
    ExpertReservedSelection result;
    result.valid = true;
    result.decision_ts_ns = decision_ts_ns;
    result.size_before = live_count_;
    result.queued_bytes_before = queued_bytes_;
    result.credit_before = credit_;
    result.debt_before = pending_debt_;
    result.legacy_head = head(legacy_);
    result.aging_head = head(aging_);
    result.waiting_eligible = eligible(result.aging_head, decision_ts_ns);
    result.hard_urgent_present = hard_urgent(result.legacy_head, decision_ts_ns);

    ExpertReservedHandle selected = result.legacy_head.handle;
    if (shutdown_drain) {
        result.source = ExpertReservedWinnerSource::ShutdownDrain;
    } else if (!result.waiting_eligible) {
        credit_ = 0;
        pending_debt_ = false;
        result.source = result.hard_urgent_present ?
                ExpertReservedWinnerSource::HardUrgent : ExpertReservedWinnerSource::Legacy;
    } else if (pending_debt_) {
        result.reserved_due = true;
        result.reserved_triggered = true;
        if (result.hard_urgent_present) {
            result.source = ExpertReservedWinnerSource::HardUrgent;
            counters_.hard_urgent_override_count++;
        } else {
            selected = result.aging_head.handle;
            result.source = ExpertReservedWinnerSource::Reserved;
            result.reserved_triggered = true;
            result.reserved_due = true;
            result.debt_repaid = true;
            pending_debt_ = false;
            counters_.debt_repaid_count++;
        }
    } else {
        const uint64_t total = credit_ + config_.reserved_numerator;
        result.credit_accrued = total;
        result.reserved_due = total >= config_.reserved_denominator;
        credit_ = total % config_.reserved_denominator;
        if (result.reserved_due) {
            result.reserved_triggered = true;
            if (result.hard_urgent_present) {
                result.source = ExpertReservedWinnerSource::HardUrgent;
                pending_debt_ = true;
                result.debt_created = true;
                counters_.hard_urgent_override_count++;
                counters_.debt_created_count++;
            } else {
                selected = result.aging_head.handle;
                result.source = ExpertReservedWinnerSource::Reserved;
            }
        } else {
            result.source = result.hard_urgent_present ?
                    ExpertReservedWinnerSource::HardUrgent : ExpertReservedWinnerSource::Legacy;
        }
    }

    result.selected.available = true;
    result.selected.handle = selected;
    result.selected.key = resolve(selected)->key;
    result.winner_changed_vs_legacy = selected != result.legacy_head.handle;
    result.reserved_same_as_legacy =
            result.source == ExpertReservedWinnerSource::Reserved &&
            !result.winner_changed_vs_legacy;

    if (result.reserved_triggered) {
        counters_.reserved_trigger_count++;
    }
    if (result.reserved_due) {
        counters_.reserved_due_count++;
    }
    if (result.source == ExpertReservedWinnerSource::Reserved) {
        counters_.reserved_selected_count++;
    }
    if (result.winner_changed_vs_legacy) {
        counters_.active_winner_changed_count++;
    }
    if (result.reserved_same_as_legacy) {
        counters_.reserved_same_as_legacy_count++;
    }

    const uint64_t selected_bytes = result.selected.key.nbytes;
    index_erase(legacy_, selected);
    index_erase(aging_, selected);
    const size_t registry_erased = registry_.erase(result.selected.key.task_id);
    if (registry_erased != 1) {
        fail_invariant("Reserved-Service registry erase failure");
    }
    Slot & slot = slots_[selected.slot_id];
    slot.live = false;
    slot.key = {};
    free_slots_.push_back(selected.slot_id);
    live_count_--;
    if (selected_bytes > queued_bytes_) {
        fail_invariant("Reserved-Service queued bytes underflow");
    }
    queued_bytes_ -= selected_bytes;
    counters_.erase_count++;
    counters_.selection_count++;

    if (!shutdown_drain && live_count_ != 0 && !eligible(head(aging_), decision_ts_ns)) {
        credit_ = 0;
        pending_debt_ = false;
    }
    if (live_count_ == 0) {
        credit_ = 0;
        pending_debt_ = false;
    }
    result.credit_after = credit_;
    result.debt_after = pending_debt_;
    result.size_after = live_count_;
    result.queued_bytes_after = queued_bytes_;

    return result;
}

ExpertReservedServiceAudit ExpertReservedServiceQueue::audit(bool require_empty) const {
    ExpertReservedServiceAudit result;
    result.store_size = live_count_;
    result.registry_size = registry_.size();
    result.legacy_index_size = legacy_.heap.size();
    result.aging_index_size = aging_.heap.size();
    uint64_t live_bytes = 0;
    size_t scanned_live = 0;
    // This is an explicit diagnostic audit, never a selection fallback.
    for (const Slot & slot : slots_) {
        if (slot.live) {
            scanned_live++;
            live_bytes = saturating_add(live_bytes, slot.key.nbytes);
        }
    }
    result.live_bytes = live_bytes;
    result.queued_bytes = queued_bytes_;
    result.final_queue_empty = live_count_ == 0;
    result.valid = scanned_live == live_count_ &&
            result.registry_size == live_count_ &&
            result.legacy_index_size == live_count_ &&
            result.aging_index_size == live_count_ &&
            result.live_bytes == queued_bytes_ &&
            (!require_empty || result.final_queue_empty);
    return result;
}

void ExpertReservedServiceQueue::fail_invariant(const char * message) {
    counters_.invariant_error_count++;
    throw std::logic_error(message);
}
