#include "expert_prefetch_policy.h"

const char * expert_policy_name(ExpertPolicy policy) {
    switch (policy) {
        case ExpertPolicy::Route:      return "route";
        case ExpertPolicy::Lru:        return "lru";
        case ExpertPolicy::Lfu:        return "lfu";
        case ExpertPolicy::WindowLfu:  return "window_lfu";
        case ExpertPolicy::LeastStale: return "least_stale";
    }
    return "route";
}

const char * expert_prefetch_async_priority_mode_name(ExpertAsyncPriorityMode mode) {
    switch (mode) {
        case ExpertAsyncPriorityMode::Score:         return "score";
        case ExpertAsyncPriorityMode::Deadline:      return "deadline";
        case ExpertAsyncPriorityMode::DeadlineScore: return "deadline_score";
        case ExpertAsyncPriorityMode::StageDeadlineScore: return "stage_deadline_score";
    }
    return "score";
}

const char * expert_pressure_level_name(ExpertPressureLevel level) {
    switch (level) {
        case ExpertPressureLevel::Low:      return "low";
        case ExpertPressureLevel::Moderate: return "moderate";
        case ExpertPressureLevel::High:     return "high";
        case ExpertPressureLevel::Critical: return "critical";
    }
    return "low";
}
