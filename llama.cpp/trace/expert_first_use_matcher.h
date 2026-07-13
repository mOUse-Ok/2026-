#pragma once

#include "expert_tensor_stage.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ExpertDurationAggregate {
    uint64_t count = 0;
    uint64_t total_ns = 0;
    uint64_t min_ns = 0;
    uint64_t max_ns = 0;
};

struct ExpertIssuedTask {
    uint64_t task_id = 0;
    uint64_t issue_id = 0;
    uint64_t step = 0;
    int layer = -1;
    int expert = -1;
    int phase = 0;
    ExpertTensorStage stage = ExpertTensorStage::Unknown;
    std::string tensor;
    uintptr_t addr = 0;
    size_t nbytes = 0;
    uint64_t created_ts_ns = 0;
    uint64_t enqueued_ts_ns = 0;
    uint64_t dequeued_ts_ns = 0;
    uint64_t issued_ts_ns = 0;
};

struct ExpertFirstUseObservation {
    uint64_t step = 0;
    int layer = -1;
    int expert = -1;
    int phase = 0;
    ExpertTensorStage stage = ExpertTensorStage::Unknown;
    std::string tensor;
    uintptr_t addr = 0;
    size_t nbytes = 0;
    uint64_t first_use_ts_ns = 0;
};

struct ExpertFirstUseMatch {
    bool considered = false;
    ExpertFirstUseObservation use;
    std::vector<ExpertIssuedTask> tasks;
    std::string unmatched_reason;

    bool matched() const { return !tasks.empty(); }
    bool ambiguous() const { return tasks.size() > 1; }
};

struct ExpertFirstUseCounters {
    uint64_t eligible_tasks = 0;
    uint64_t logical_first_uses = 0;
    uint64_t matched_tasks = 0;
    uint64_t unmatched_tasks = 0;
    uint64_t unmatched_first_uses = 0;
    uint64_t ambiguous_matches = 0;
    uint64_t duplicate_first_use_ignored = 0;
    uint64_t matcher_peak_live_tasks = 0;
    uint64_t matcher_expired_tasks = 0;
    uint64_t pending_issued_tasks = 0;
    uint64_t late_issued_tasks = 0;
    uint64_t ignored_old_uses = 0;
    std::array<ExpertDurationAggregate, 3> create_to_first_use_ns{};
};

class ExpertFirstUseMatcher {
public:
    void register_issue(ExpertIssuedTask task);
    ExpertFirstUseMatch observe_first_use(ExpertFirstUseObservation use);
    ExpertFirstUseCounters counters();

private:
    static std::string semantic_key(
            uint64_t step, int layer, int expert, const std::string & tensor);
    static bool ranges_overlap(uintptr_t a_addr, size_t a_size, uintptr_t b_addr, size_t b_size);
    static void observe_duration(ExpertDurationAggregate & aggregate, uint64_t duration_ns);
    void advance_step_unlocked(uint64_t step);

    std::mutex mu_;
    bool has_observed_step_ = false;
    uint64_t active_step_ = 0;
    uint64_t live_tasks_ = 0;
    std::unordered_map<std::string, std::vector<ExpertIssuedTask>> pending_;
    std::unordered_set<std::string> observed_;
    ExpertFirstUseCounters counters_;
};

