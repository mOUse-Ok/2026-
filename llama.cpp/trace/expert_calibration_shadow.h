#pragma once

// Phase 2E-A: Safe Shadow Calibration — observation-only profile that learns
// the current environment's normal scale (prefetch issue ratio, major faults
// per decode step, COLD eligible bytes per step) WITHOUT affecting any control
// behavior. Default OFF; enabled via LLM_MEM_TRACE_OPT_EXPERT_CALIBRATION_SHADOW.
//
// Invariants (spec §15):
//   - Shadow ON/OFF must NOT change Memory Object, Working Set, probation,
//     COLD syscall, Prefetch issued, or Runtime Rescue state machine behavior.
//   - All hooks are strictly additive observation counters.
//   - The profile is per-process only; no persistence, no cross-model reuse.

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "trace_event.h"

enum class CalibrationState {
    Uncalibrated,
    Calibrated,
};

const char * calibration_state_name(CalibrationState state);

struct CalibrationBaseline {
    double median = 0.0;
    double p25 = 0.0;
    double p75 = 0.0;
    bool valid = false;
};

// Context gathered by the wiring layer (tensor_trace.cpp) and passed to
// on_step_end. All cumulative counters are the CURRENT values from the
// ExpertMemoryObjectTracker; the profile computes per-step deltas internally.
struct CalibrationStepContext {
    // True when Runtime Rescue is disabled OR state is Normal/ReEntry.
    bool rescue_state_safe = true;
    // Cumulative major fault count from getrusage (profile computes delta).
    uint64_t current_major_faults = 0;
    // Cumulative cold_eligible_candidate_bytes from the memory object tracker.
    uint64_t cumulative_cold_eligible_bytes = 0;
    // Cumulative lifecycle counters (profile checks no-new-violations this step).
    uint64_t invariant_violations = 0;
    uint64_t cold_protected_violation = 0;
    uint64_t madv_cold_failed = 0;
    // Point-in-time counters (must be 0 for a healthy step).
    uint64_t pending = 0;
    uint64_t active = 0;
    uint64_t current_hint_inflight_objects = 0;
};

// Context for the shutdown summary (memory readers + working set budget).
struct CalibrationSummaryContext {
    uint64_t memory_current = 0;
    uint64_t memory_limit = 0;
    uint64_t rss_bytes = 0;
    uint64_t working_set_budget_bytes = 0;
};

struct CalibrationSample {
    uint64_t step = 0;
    uint64_t prefetch_opportunities = 0;
    uint64_t prefetch_issued = 0;
    uint64_t major_fault_delta = 0;
    uint64_t cold_eligible_bytes = 0;
};

class CalibrationProfile {
public:
    // Per-step accumulators (reset at each decode step end).
    void record_prefetch_opportunity(int phase);
    void record_prefetch_issued(int phase);

    // Called at decode step end. Gathers context, evaluates healthy-sample
    // admission, accumulates samples, computes baselines when >= 16 healthy
    // samples, and emits shadow normalized metrics after calibration is valid.
    void on_step_end(int phase, uint64_t step, uint64_t latency_ns,
                     const CalibrationStepContext & ctx);

    // Writes the EXPERT_CALIBRATION_SUMMARY trace event at shutdown.
    void write_summary(const CalibrationSummaryContext & ctx);

    // Read-only queries for testing.
    CalibrationState state() const;
    size_t healthy_sample_count() const;
    CalibrationBaseline baseline_prefetch_issue_ratio() const;
    CalibrationBaseline baseline_major_faults() const;
    CalibrationBaseline baseline_cold_eligible_bytes() const;
    double median_opportunities() const;
    double median_issued() const;

private:
    mutable std::mutex mu;

    // Per-step accumulators (DECODE only).
    uint64_t step_opportunities = 0;
    uint64_t step_issued = 0;

    // Previous cumulative values (for delta computation).
    uint64_t prev_major_faults = 0;
    bool has_prev_major_faults = false;
    uint64_t prev_cold_eligible_bytes = 0;
    bool has_prev_cold_eligible = false;

    // Previous lifecycle counters (for no-new-violation check).
    uint64_t prev_invariant_violations = 0;
    uint64_t prev_cold_protected_violation = 0;
    uint64_t prev_madv_cold_failed = 0;
    bool has_prev_lifecycle = false;

    // Previous step (for model-reload detection).
    uint64_t prev_step = 0;
    bool has_prev_step = false;

    // Healthy samples and calibration state.
    std::vector<CalibrationSample> samples;
    CalibrationState calibration_state = CalibrationState::Uncalibrated;

    // Baselines (computed when >= 16 healthy samples).
    CalibrationBaseline bl_prefetch_issue_ratio;
    CalibrationBaseline bl_major_faults;
    CalibrationBaseline bl_cold_eligible_bytes;
    double bl_median_opportunities = 0.0;
    double bl_median_issued = 0.0;

    void reset_unlocked();
    void compute_baselines_unlocked();
    static CalibrationBaseline compute_baseline(std::vector<double> values);
    static double percentile_of_sorted(const std::vector<double> & sorted, double p);
    bool lifecycle_clean_unlocked(const CalibrationStepContext & ctx) const;
    void emit_shadow_step_unlocked(uint64_t step, const CalibrationSample & sample) const;
};

CalibrationProfile & expert_calibration_profile();
