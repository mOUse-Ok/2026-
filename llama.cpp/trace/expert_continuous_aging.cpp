#include "expert_continuous_aging.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

bool same_group(
        const ExpertContinuousAgingTaskKey & a,
        const ExpertContinuousAgingTaskKey & b) {
    return a.step == b.step && a.layer == b.layer && a.stage == b.stage;
}

bool group_higher(
        const ExpertContinuousAgingTaskKey & a,
        const ExpertContinuousAgingTaskKey & b) {
    if (a.step != b.step) {
        return a.step < b.step;
    }
    if (a.layer != b.layer) {
        return a.layer < b.layer;
    }
    const int a_stage = expert_continuous_aging_stage_rank(a.stage);
    const int b_stage = expert_continuous_aging_stage_rank(b.stage);
    return a_stage < b_stage;
}

ExpertHintPriorityKey legacy_key(const ExpertContinuousAgingTaskKey & key) {
    ExpertHintPriorityKey result;
    result.step = key.step;
    result.layer = key.layer;
    result.stage = key.stage;
    result.route_score = key.route_score;
    result.sequence = key.sequence;
    result.deadline_ts_ns = key.deadline_ts_ns;
    return result;
}

} // namespace

int expert_continuous_aging_stage_rank(ExpertTensorStage stage) {
    switch (stage) {
        case ExpertTensorStage::Early:   return 0;
        case ExpertTensorStage::Late:    return 1;
        case ExpertTensorStage::Unknown: return 2;
    }
    return 2;
}

long double expert_continuous_aging_static_score(
        const ExpertContinuousAgingTaskKey & key,
        const ExpertContinuousAgingConfig & config) {
    if (!std::isfinite(config.alpha_per_ns) || config.alpha_per_ns <= 0.0 ||
            config.score_epoch_ts_ns == 0 ||
            key.enqueued_ts_ns < config.score_epoch_ts_ns) {
        throw std::invalid_argument("invalid Continuous Aging static-score input");
    }
    const uint64_t offset_ns = key.enqueued_ts_ns - config.score_epoch_ts_ns;
    return static_cast<long double>(key.route_score) -
            static_cast<long double>(config.alpha_per_ns) *
            static_cast<long double>(offset_ns);
}

long double expert_continuous_aging_direct_adjusted_score(
        const ExpertContinuousAgingTaskKey & key,
        uint64_t decision_ts_ns,
        const ExpertContinuousAgingConfig & config) {
    if (!std::isfinite(config.alpha_per_ns) || config.alpha_per_ns <= 0.0 ||
            decision_ts_ns < key.enqueued_ts_ns) {
        throw std::invalid_argument("invalid Continuous Aging direct-score input");
    }
    return static_cast<long double>(key.route_score) +
            static_cast<long double>(config.alpha_per_ns) *
            static_cast<long double>(decision_ts_ns - key.enqueued_ts_ns);
}

bool expert_continuous_aging_higher_static(
        const ExpertContinuousAgingTaskKey & a,
        const ExpertContinuousAgingTaskKey & b,
        const ExpertContinuousAgingConfig & config) {
    if (!same_group(a, b)) {
        return group_higher(a, b);
    }
    const long double a_score = expert_continuous_aging_static_score(a, config);
    const long double b_score = expert_continuous_aging_static_score(b, config);
    if (a_score != b_score) {
        return a_score > b_score;
    }
    if (a.sequence != b.sequence) {
        return a.sequence < b.sequence;
    }
    return a.task_id < b.task_id;
}

bool expert_continuous_aging_higher_direct(
        const ExpertContinuousAgingTaskKey & a,
        const ExpertContinuousAgingTaskKey & b,
        uint64_t decision_ts_ns,
        const ExpertContinuousAgingConfig & config) {
    if (!same_group(a, b)) {
        return group_higher(a, b);
    }
    const long double a_score = expert_continuous_aging_direct_adjusted_score(
            a, decision_ts_ns, config);
    const long double b_score = expert_continuous_aging_direct_adjusted_score(
            b, decision_ts_ns, config);
    if (a_score != b_score) {
        return a_score > b_score;
    }
    if (a.sequence != b.sequence) {
        return a.sequence < b.sequence;
    }
    return a.task_id < b.task_id;
}

void ExpertContinuousAgingQueue::reset(
        size_t capacity,
        const ExpertContinuousAgingConfig & config) {
    if (capacity == 0 || capacity > UINT32_MAX ||
            !std::isfinite(config.alpha_per_ns) || config.alpha_per_ns <= 0.0 ||
            config.score_epoch_ts_ns == 0) {
        throw std::invalid_argument("invalid Continuous Aging queue configuration");
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
    continuous_ = {};
    continuous_.kind = IndexKind::Continuous;
    continuous_.heap.reserve(capacity);
    continuous_.position.assign(capacity, npos);
    live_count_ = 0;
    queued_bytes_ = 0;
    counters_ = {};
}

void ExpertContinuousAgingQueue::clear() {
    slots_.clear();
    free_slots_.clear();
    registry_.clear();
    legacy_ = {};
    continuous_ = {};
    live_count_ = 0;
    queued_bytes_ = 0;
}

ExpertContinuousAgingQueue::Slot * ExpertContinuousAgingQueue::resolve(
        ExpertContinuousAgingHandle handle) {
    if (!handle.valid() || handle.slot_id >= slots_.size()) {
        counters_.stale_handle_count++;
        fail_invariant("stale Continuous Aging handle");
    }
    Slot & slot = slots_[handle.slot_id];
    if (!slot.live) {
        counters_.stale_handle_count++;
        fail_invariant("Continuous Aging handle references a free slot");
    }
    if (slot.generation != handle.generation) {
        counters_.generation_mismatch_count++;
        fail_invariant("Continuous Aging generation mismatch");
    }
    return &slot;
}

const ExpertContinuousAgingQueue::Slot * ExpertContinuousAgingQueue::resolve_const(
        ExpertContinuousAgingHandle handle) const {
    if (!handle.valid() || handle.slot_id >= slots_.size()) {
        throw std::logic_error("stale Continuous Aging handle");
    }
    const Slot & slot = slots_[handle.slot_id];
    if (!slot.live || slot.generation != handle.generation) {
        throw std::logic_error("Continuous Aging generation mismatch");
    }
    return &slot;
}

bool ExpertContinuousAgingQueue::higher(
        IndexKind kind,
        ExpertContinuousAgingHandle a,
        ExpertContinuousAgingHandle b) const {
    const ExpertContinuousAgingTaskKey & ka = resolve_const(a)->key;
    const ExpertContinuousAgingTaskKey & kb = resolve_const(b)->key;
    if (kind == IndexKind::Continuous) {
        return expert_continuous_aging_higher_static(ka, kb, config_);
    }
    return expert_hint_priority_higher(
            legacy_key(ka), legacy_key(kb), ExpertAsyncPriorityMode::DeadlineScore);
}

void ExpertContinuousAgingQueue::heap_swap(
        IndexedHeap & index,
        size_t a,
        size_t b) {
    std::swap(index.heap[a], index.heap[b]);
    index.position[index.heap[a].slot_id] = a;
    index.position[index.heap[b].slot_id] = b;
}

void ExpertContinuousAgingQueue::sift_up(IndexedHeap & index, size_t position) {
    uint64_t & count = index.kind == IndexKind::Legacy ?
            counters_.legacy_heap_sift_count : counters_.continuous_heap_sift_count;
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

void ExpertContinuousAgingQueue::sift_down(IndexedHeap & index, size_t position) {
    uint64_t & count = index.kind == IndexKind::Legacy ?
            counters_.legacy_heap_sift_count : counters_.continuous_heap_sift_count;
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

void ExpertContinuousAgingQueue::index_insert(
        IndexedHeap & index,
        ExpertContinuousAgingHandle handle) {
    if (index.position[handle.slot_id] != npos) {
        fail_invariant("duplicate Continuous Aging index insertion");
    }
    index.position[handle.slot_id] = index.heap.size();
    index.heap.push_back(handle);
    sift_up(index, index.heap.size() - 1);
}

void ExpertContinuousAgingQueue::index_erase(
        IndexedHeap & index,
        ExpertContinuousAgingHandle handle) {
    if (handle.slot_id >= index.position.size()) {
        counters_.stale_handle_count++;
        fail_invariant("Continuous Aging erase slot out of range");
    }
    const size_t position = index.position[handle.slot_id];
    if (position == npos || position >= index.heap.size() ||
            index.heap[position] != handle) {
        counters_.duplicate_erase_count++;
        fail_invariant("Continuous Aging duplicate or mismatched erase");
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

ExpertContinuousAgingTaskRef ExpertContinuousAgingQueue::head(
        const IndexedHeap & index) const {
    ExpertContinuousAgingTaskRef result;
    if (index.heap.empty()) {
        return result;
    }
    result.available = true;
    result.handle = index.heap.front();
    result.key = resolve_const(result.handle)->key;
    return result;
}

ExpertContinuousAgingHandle ExpertContinuousAgingQueue::insert(
        const ExpertContinuousAgingTaskKey & key) {
    if (free_slots_.empty() || key.task_id == 0 || key.enqueued_ts_ns == 0 ||
            key.enqueued_ts_ns < config_.score_epoch_ts_ns ||
            !std::isfinite(key.route_score) || registry_.count(key.task_id) != 0) {
        fail_invariant("invalid Continuous Aging insertion");
    }
    const uint32_t slot_id = free_slots_.back();
    free_slots_.pop_back();
    Slot & slot = slots_[slot_id];
    if (slot.live || slot.generation == std::numeric_limits<uint64_t>::max()) {
        fail_invariant("Continuous Aging slot lifecycle failure");
    }
    slot.live = true;
    slot.generation++;
    if (slot.generation == 0) {
        fail_invariant("Continuous Aging generation wrapped");
    }
    slot.key = key;
    const ExpertContinuousAgingHandle handle{slot_id, slot.generation};
    registry_.emplace(key.task_id, handle);
    try {
        index_insert(legacy_, handle);
        index_insert(continuous_, handle);
    } catch (...) {
        fail_invariant("Continuous Aging index insertion failed");
    }
    live_count_++;
    if (queued_bytes_ > std::numeric_limits<uint64_t>::max() - key.nbytes) {
        fail_invariant("Continuous Aging queued bytes overflow");
    }
    queued_bytes_ += key.nbytes;
    counters_.insert_count++;
    return handle;
}

ExpertContinuousAgingSelection ExpertContinuousAgingQueue::select(
        uint64_t decision_ts_ns) {
    if (live_count_ == 0 || legacy_.heap.empty() || continuous_.heap.empty()) {
        fail_invariant("Continuous Aging selection requires a nonempty queue");
    }
    ExpertContinuousAgingSelection result;
    result.valid = true;
    result.decision_ts_ns = decision_ts_ns;
    result.size_before = live_count_;
    result.queued_bytes_before = queued_bytes_;
    result.legacy_head = head(legacy_);
    result.selected = head(continuous_);
    result.winner_changed_vs_legacy =
            result.selected.key.task_id != result.legacy_head.key.task_id;
    result.legacy_head_hard_urgent =
            result.legacy_head.key.deadline_ts_ns != 0 &&
            decision_ts_ns >= result.legacy_head.key.deadline_ts_ns;
    result.selected_static_score = expert_continuous_aging_static_score(
            result.selected.key, config_);
    result.selected_direct_adjusted_score =
            expert_continuous_aging_direct_adjusted_score(
                    result.selected.key, decision_ts_ns, config_);

    if (result.winner_changed_vs_legacy) {
        counters_.active_winner_changed_count++;
        if (result.legacy_head_hard_urgent) {
            counters_.hard_urgent_bypass_count++;
        }
    } else {
        counters_.winner_same_as_legacy_count++;
    }
    if (result.selected.key.deadline_ts_ns != 0 &&
            decision_ts_ns >= result.selected.key.deadline_ts_ns) {
        counters_.selected_after_deadline_count++;
    }

    const ExpertContinuousAgingHandle handle = result.selected.handle;
    Slot * slot = resolve(handle);
    index_erase(legacy_, handle);
    index_erase(continuous_, handle);
    if (registry_.erase(slot->key.task_id) != 1) {
        fail_invariant("Continuous Aging registry erase mismatch");
    }
    if (queued_bytes_ < slot->key.nbytes || live_count_ == 0) {
        fail_invariant("Continuous Aging conservation underflow");
    }
    queued_bytes_ -= slot->key.nbytes;
    live_count_--;
    slot->live = false;
    slot->key = {};
    free_slots_.push_back(handle.slot_id);
    counters_.erase_count++;
    counters_.selection_count++;
    result.size_after = live_count_;
    result.queued_bytes_after = queued_bytes_;
    return result;
}

ExpertContinuousAgingAudit ExpertContinuousAgingQueue::audit(bool require_empty) const {
    ExpertContinuousAgingAudit result;
    result.store_size = live_count_;
    result.registry_size = registry_.size();
    result.legacy_index_size = legacy_.heap.size();
    result.continuous_index_size = continuous_.heap.size();
    result.queued_bytes = queued_bytes_;
    for (const Slot & slot : slots_) {
        if (slot.live) {
            result.live_bytes += slot.key.nbytes;
        }
    }
    result.final_queue_empty = live_count_ == 0 && queued_bytes_ == 0 &&
            registry_.empty() && legacy_.heap.empty() && continuous_.heap.empty();
    result.valid = result.store_size == result.registry_size &&
            result.store_size == result.legacy_index_size &&
            result.store_size == result.continuous_index_size &&
            result.queued_bytes == result.live_bytes &&
            counters_.stale_handle_count == 0 &&
            counters_.duplicate_erase_count == 0 &&
            counters_.generation_mismatch_count == 0 &&
            counters_.invariant_error_count == 0 &&
            counters_.full_store_scan_count == 0 &&
            (!require_empty || result.final_queue_empty);
    return result;
}

[[noreturn]] void ExpertContinuousAgingQueue::fail_invariant(const char * message) {
    counters_.invariant_error_count++;
    throw std::logic_error(message);
}
