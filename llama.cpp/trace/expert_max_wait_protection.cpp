#include "expert_max_wait_protection.h"

#include <limits>

bool expert_max_wait_parse_us(
        const char * text,
        bool allow_zero,
        uint64_t & value_us,
        uint64_t & value_ns) {
    value_us = 0;
    value_ns = 0;
    if (!text || !text[0]) {
        return false;
    }

    uint64_t parsed = 0;
    for (const char * p = text; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        const uint64_t digit = (uint64_t) (*p - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            return false;
        }
        parsed = parsed * 10 + digit;
    }
    if (!allow_zero && parsed == 0) {
        return false;
    }
    if (parsed > std::numeric_limits<uint64_t>::max() / 1000) {
        return false;
    }

    value_us = parsed;
    value_ns = parsed * 1000;
    return true;
}

uint64_t expert_max_wait_saturating_add(uint64_t a, uint64_t b) {
    return b > std::numeric_limits<uint64_t>::max() - a ?
            std::numeric_limits<uint64_t>::max() : a + b;
}

ExpertMaxWaitDecision expert_max_wait_classify(
        const ExpertMaxWaitKey & key,
        uint64_t decision_now_ns,
        const ExpertMaxWaitConfig & config) {
    ExpertMaxWaitDecision result;
    if (key.enqueued_ts_ns == 0) {
        result.reason = ExpertMaxWaitReason::MissingEnqueueFallback;
        return result;
    }
    if (decision_now_ns < key.enqueued_ts_ns) {
        result.reason = ExpertMaxWaitReason::EnqueueTimeRegressionFallback;
        return result;
    }

    result.waiting_available = true;
    result.waiting_ns = decision_now_ns - key.enqueued_ts_ns;
    const uint64_t urgent_limit_ns = expert_max_wait_saturating_add(
            decision_now_ns, config.urgent_guard_ns);
    if (key.priority.deadline_ts_ns != 0 &&
            key.priority.deadline_ts_ns <= urgent_limit_ns) {
        result.task_class = ExpertMaxWaitClass::Urgent;
        result.reason = ExpertMaxWaitReason::DeadlineWithinGuard;
        return result;
    }
    if (result.waiting_ns >= config.threshold_ns) {
        result.task_class = ExpertMaxWaitClass::Protected;
        result.reason = ExpertMaxWaitReason::WaitingAtOrAboveThreshold;
        result.threshold_overshoot_ns = result.waiting_ns - config.threshold_ns;
    }
    return result;
}

bool expert_max_wait_higher(
        const ExpertMaxWaitKey & a,
        const ExpertMaxWaitDecision & a_decision,
        const ExpertMaxWaitKey & b,
        const ExpertMaxWaitDecision & b_decision) {
    if (a_decision.task_class != b_decision.task_class) {
        return (int) a_decision.task_class < (int) b_decision.task_class;
    }
    if (a_decision.task_class == ExpertMaxWaitClass::Protected &&
            a.enqueued_ts_ns != b.enqueued_ts_ns) {
        return a.enqueued_ts_ns < b.enqueued_ts_ns;
    }
    return expert_hint_priority_higher(
            a.priority, b.priority, ExpertAsyncPriorityMode::DeadlineScore);
}

const char * expert_max_wait_class_name(ExpertMaxWaitClass task_class) {
    switch (task_class) {
        case ExpertMaxWaitClass::Urgent:    return "urgent";
        case ExpertMaxWaitClass::Protected: return "protected";
        case ExpertMaxWaitClass::Normal:    return "normal";
    }
    return "normal";
}

const char * expert_max_wait_reason_name(ExpertMaxWaitReason reason) {
    switch (reason) {
        case ExpertMaxWaitReason::DeadlineWithinGuard:
            return "deadline_within_guard";
        case ExpertMaxWaitReason::WaitingAtOrAboveThreshold:
            return "waiting_at_or_above_threshold";
        case ExpertMaxWaitReason::LegacyNormal:
            return "legacy_normal";
        case ExpertMaxWaitReason::MissingEnqueueFallback:
            return "missing_enqueue_fallback";
        case ExpertMaxWaitReason::EnqueueTimeRegressionFallback:
            return "enqueue_time_regression_fallback";
    }
    return "legacy_normal";
}
