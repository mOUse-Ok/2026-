#pragma once

#include "expert_hint_priority.h"

#include <cstdint>

struct ExpertMaxWaitConfig {
    uint64_t threshold_us = 0;
    uint64_t threshold_ns = 0;
    uint64_t urgent_guard_us = 0;
    uint64_t urgent_guard_ns = 0;
};

enum class ExpertMaxWaitClass {
    Urgent = 0,
    Protected = 1,
    Normal = 2,
};

enum class ExpertMaxWaitReason {
    DeadlineWithinGuard,
    WaitingAtOrAboveThreshold,
    LegacyNormal,
    MissingEnqueueFallback,
    EnqueueTimeRegressionFallback,
};

struct ExpertMaxWaitKey {
    ExpertHintPriorityKey priority;
    uint64_t enqueued_ts_ns = 0;
};

struct ExpertMaxWaitDecision {
    ExpertMaxWaitClass task_class = ExpertMaxWaitClass::Normal;
    ExpertMaxWaitReason reason = ExpertMaxWaitReason::LegacyNormal;
    bool waiting_available = false;
    uint64_t waiting_ns = 0;
    uint64_t threshold_overshoot_ns = 0;
};

bool expert_max_wait_parse_us(
        const char * text,
        bool allow_zero,
        uint64_t & value_us,
        uint64_t & value_ns);

uint64_t expert_max_wait_saturating_add(uint64_t a, uint64_t b);

ExpertMaxWaitDecision expert_max_wait_classify(
        const ExpertMaxWaitKey & key,
        uint64_t decision_now_ns,
        const ExpertMaxWaitConfig & config);

bool expert_max_wait_higher(
        const ExpertMaxWaitKey & a,
        const ExpertMaxWaitDecision & a_decision,
        const ExpertMaxWaitKey & b,
        const ExpertMaxWaitDecision & b_decision);

const char * expert_max_wait_class_name(ExpertMaxWaitClass task_class);
const char * expert_max_wait_reason_name(ExpertMaxWaitReason reason);
