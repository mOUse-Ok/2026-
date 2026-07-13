#include "expert_first_use_matcher.h"

#include <algorithm>
#include <iterator>
#include <limits>

std::string ExpertFirstUseMatcher::semantic_key(
        uint64_t step, int layer, int expert, const std::string & tensor) {
    return std::to_string(step) + ":" + std::to_string(layer) + ":" +
           std::to_string(expert) + ":" + tensor;
}

bool ExpertFirstUseMatcher::ranges_overlap(
        uintptr_t a_addr, size_t a_size, uintptr_t b_addr, size_t b_size) {
    if (a_addr == 0 || b_addr == 0 || a_size == 0 || b_size == 0) {
        return false;
    }
    const uintptr_t max_addr = std::numeric_limits<uintptr_t>::max();
    const uintptr_t a_end = a_size > max_addr - a_addr ? max_addr : a_addr + a_size;
    const uintptr_t b_end = b_size > max_addr - b_addr ? max_addr : b_addr + b_size;
    return a_addr < b_end && b_addr < a_end;
}

void ExpertFirstUseMatcher::observe_duration(
        ExpertDurationAggregate & aggregate, uint64_t duration_ns) {
    aggregate.count++;
    aggregate.total_ns = duration_ns > std::numeric_limits<uint64_t>::max() - aggregate.total_ns ?
            std::numeric_limits<uint64_t>::max() : aggregate.total_ns + duration_ns;
    if (aggregate.count == 1 || duration_ns < aggregate.min_ns) {
        aggregate.min_ns = duration_ns;
    }
    aggregate.max_ns = std::max(aggregate.max_ns, duration_ns);
}

void ExpertFirstUseMatcher::advance_step_unlocked(uint64_t step) {
    if (has_observed_step_ && step <= active_step_) {
        return;
    }

    for (auto entry = pending_.begin(); entry != pending_.end();) {
        std::vector<ExpertIssuedTask> & tasks = entry->second;
        const auto first_current = std::remove_if(
                tasks.begin(), tasks.end(), [&](const ExpertIssuedTask & task) {
                    return task.step < step;
                });
        const uint64_t expired = (uint64_t) std::distance(first_current, tasks.end());
        counters_.matcher_expired_tasks += expired;
        live_tasks_ = expired > live_tasks_ ? 0 : live_tasks_ - expired;
        tasks.erase(first_current, tasks.end());
        if (tasks.empty()) {
            entry = pending_.erase(entry);
        } else {
            ++entry;
        }
    }
    observed_.clear();
    has_observed_step_ = true;
    active_step_ = step;
}

void ExpertFirstUseMatcher::register_issue(ExpertIssuedTask task) {
    std::lock_guard<std::mutex> lock(mu_);
    counters_.eligible_tasks++;
    if (has_observed_step_ && task.step < active_step_) {
        counters_.late_issued_tasks++;
        return;
    }
    const std::string key = semantic_key(task.step, task.layer, task.expert, task.tensor);
    if (observed_.find(key) != observed_.end()) {
        counters_.late_issued_tasks++;
        return;
    }
    pending_[key].push_back(std::move(task));
    live_tasks_++;
    counters_.matcher_peak_live_tasks = std::max(counters_.matcher_peak_live_tasks, live_tasks_);
}

ExpertFirstUseMatch ExpertFirstUseMatcher::observe_first_use(ExpertFirstUseObservation use) {
    ExpertFirstUseMatch result;
    std::lock_guard<std::mutex> lock(mu_);
    if (has_observed_step_ && use.step < active_step_) {
        counters_.ignored_old_uses++;
        return result;
    }
    advance_step_unlocked(use.step);
    const std::string key = semantic_key(use.step, use.layer, use.expert, use.tensor);
    if (!observed_.insert(key).second) {
        counters_.duplicate_first_use_ignored++;
        return result;
    }

    result.considered = true;
    result.use = std::move(use);
    counters_.logical_first_uses++;

    auto pending_it = pending_.find(key);
    if (pending_it == pending_.end()) {
        counters_.unmatched_first_uses++;
        result.unmatched_reason = "no_issued_task";
        return result;
    }

    std::vector<ExpertIssuedTask> & candidates = pending_it->second;
    bool has_same_stage = false;
    bool has_causal_candidate = false;
    bool has_overlapping_candidate = false;
    for (ExpertIssuedTask & task : candidates) {
        if (task.stage != result.use.stage) {
            continue;
        }
        has_same_stage = true;
        if (task.issued_ts_ns > result.use.first_use_ts_ns) {
            continue;
        }
        has_causal_candidate = true;
        if (!ranges_overlap(task.addr, task.nbytes, result.use.addr, result.use.nbytes)) {
            continue;
        }
        has_overlapping_candidate = true;
        result.tasks.push_back(std::move(task));
    }

    live_tasks_ = candidates.size() > live_tasks_ ? 0 : live_tasks_ - candidates.size();
    pending_.erase(pending_it);

    if (result.tasks.empty()) {
        counters_.unmatched_first_uses++;
        if (!has_same_stage) {
            result.unmatched_reason = "stage_mismatch";
        } else if (!has_causal_candidate) {
            result.unmatched_reason = "issue_after_first_use";
        } else if (!has_overlapping_candidate) {
            result.unmatched_reason = "address_mismatch";
        } else {
            result.unmatched_reason = "no_eligible_task";
        }
        return result;
    }

    std::sort(result.tasks.begin(), result.tasks.end(), [](const ExpertIssuedTask & a, const ExpertIssuedTask & b) {
        if (a.issued_ts_ns != b.issued_ts_ns) {
            return a.issued_ts_ns < b.issued_ts_ns;
        }
        return a.task_id < b.task_id;
    });
    counters_.matched_tasks += result.tasks.size();
    if (result.tasks.size() > 1) {
        counters_.ambiguous_matches++;
    }
    for (const ExpertIssuedTask & task : result.tasks) {
        if (task.created_ts_ns != 0 && result.use.first_use_ts_ns >= task.created_ts_ns) {
            observe_duration(
                    counters_.create_to_first_use_ns[expert_tensor_stage_index(task.stage)],
                    result.use.first_use_ts_ns - task.created_ts_ns);
        }
    }
    return result;
}

ExpertFirstUseCounters ExpertFirstUseMatcher::counters() {
    std::lock_guard<std::mutex> lock(mu_);
    ExpertFirstUseCounters result = counters_;
    result.pending_issued_tasks = live_tasks_;
    result.unmatched_tasks = result.eligible_tasks >= result.matched_tasks ?
            result.eligible_tasks - result.matched_tasks : 0;
    return result;
}
