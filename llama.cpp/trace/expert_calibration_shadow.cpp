// Phase 2E-A: Safe Shadow Calibration — observation-only implementation.
// See expert_calibration_shadow.h for design and invariants.

#include "expert_calibration_shadow.h"

#include <algorithm>
#include <cmath>
#include <string>

const char * calibration_state_name(CalibrationState state) {
    switch (state) {
        case CalibrationState::Uncalibrated: return "uncalibrated";
        case CalibrationState::Calibrated:   return "calibrated";
    }
    return "uncalibrated";
}

void CalibrationProfile::record_prefetch_opportunity(int phase) {
    if (phase != LLM_MEM_TRACE_PHASE_DECODE) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu);
    ++step_opportunities;
}

void CalibrationProfile::record_prefetch_issued(int phase) {
    if (phase != LLM_MEM_TRACE_PHASE_DECODE) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu);
    ++step_issued;
}

double CalibrationProfile::percentile_of_sorted(const std::vector<double> & sorted, double p) {
    if (sorted.empty()) {
        return 0.0;
    }
    if (sorted.size() == 1) {
        return sorted[0];
    }
    const double rank = (p / 100.0) * (double) (sorted.size() - 1);
    const size_t lo = (size_t) std::floor(rank);
    const size_t hi = (size_t) std::ceil(rank);
    if (lo == hi) {
        return sorted[lo];
    }
    const double frac = rank - (double) lo;
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

CalibrationBaseline CalibrationProfile::compute_baseline(std::vector<double> values) {
    CalibrationBaseline bl;
    if (values.empty()) {
        return bl;
    }
    std::sort(values.begin(), values.end());
    bl.median = percentile_of_sorted(values, 50.0);
    bl.p25 = percentile_of_sorted(values, 25.0);
    bl.p75 = percentile_of_sorted(values, 75.0);
    bl.valid = true;
    return bl;
}

bool CalibrationProfile::lifecycle_clean_unlocked(const CalibrationStepContext & ctx) const {
    if (!has_prev_lifecycle) {
        return false;
    }
    if (ctx.invariant_violations > prev_invariant_violations) {
        return false;
    }
    if (ctx.cold_protected_violation > prev_cold_protected_violation) {
        return false;
    }
    if (ctx.madv_cold_failed > prev_madv_cold_failed) {
        return false;
    }
    if (ctx.pending != 0 || ctx.active != 0 || ctx.current_hint_inflight_objects != 0) {
        return false;
    }
    return true;
}

void CalibrationProfile::reset_unlocked() {
    step_opportunities = 0;
    step_issued = 0;
    prev_major_faults = 0;
    has_prev_major_faults = false;
    prev_cold_eligible_bytes = 0;
    has_prev_cold_eligible = false;
    prev_invariant_violations = 0;
    prev_cold_protected_violation = 0;
    prev_madv_cold_failed = 0;
    has_prev_lifecycle = false;
    prev_step = 0;
    has_prev_step = false;
    samples.clear();
    calibration_state = CalibrationState::Uncalibrated;
    bl_prefetch_issue_ratio = CalibrationBaseline{};
    bl_major_faults = CalibrationBaseline{};
    bl_cold_eligible_bytes = CalibrationBaseline{};
    bl_median_opportunities = 0.0;
    bl_median_issued = 0.0;
}

void CalibrationProfile::compute_baselines_unlocked() {
    std::vector<double> issue_ratios;
    std::vector<double> faults;
    std::vector<double> cold_bytes;
    std::vector<double> opportunities;
    std::vector<double> issued;
    issue_ratios.reserve(samples.size());
    faults.reserve(samples.size());
    cold_bytes.reserve(samples.size());
    opportunities.reserve(samples.size());
    issued.reserve(samples.size());

    for (const CalibrationSample & s : samples) {
        issue_ratios.push_back(s.prefetch_opportunities > 0 ?
                (double) s.prefetch_issued / (double) s.prefetch_opportunities : 0.0);
        faults.push_back((double) s.major_fault_delta);
        cold_bytes.push_back((double) s.cold_eligible_bytes);
        opportunities.push_back((double) s.prefetch_opportunities);
        issued.push_back((double) s.prefetch_issued);
    }

    bl_prefetch_issue_ratio = compute_baseline(std::move(issue_ratios));
    bl_major_faults = compute_baseline(std::move(faults));
    bl_cold_eligible_bytes = compute_baseline(std::move(cold_bytes));

    std::sort(opportunities.begin(), opportunities.end());
    std::sort(issued.begin(), issued.end());
    bl_median_opportunities = percentile_of_sorted(opportunities, 50.0);
    bl_median_issued = percentile_of_sorted(issued, 50.0);
}

void CalibrationProfile::emit_shadow_step_unlocked(
        uint64_t step, const CalibrationSample & sample) const {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    const double normalized_prefetch = bl_median_issued > 0.0 ?
            (double) sample.prefetch_issued / bl_median_issued : 0.0;
    const double normalized_fault = bl_major_faults.median > 0.0 ?
            (double) sample.major_fault_delta / bl_major_faults.median : 0.0;
    const double normalized_cold = bl_cold_eligible_bytes.median > 0.0 ?
            (double) sample.cold_eligible_bytes / bl_cold_eligible_bytes.median : 0.0;

    std::string line;
    line.reserve(320);
    line += "{\"event\":\"EXPERT_CALIBRATION_SHADOW_STEP\",\"ts_ns\":" +
            std::to_string(llm_mem_trace_time_ns());
    line += ",\"step\":" + std::to_string(step);
    line += ",\"prefetch_opportunities\":" +
            std::to_string(sample.prefetch_opportunities);
    line += ",\"prefetch_issued\":" + std::to_string(sample.prefetch_issued);
    line += ",\"major_fault_delta\":" + std::to_string(sample.major_fault_delta);
    line += ",\"cold_eligible_bytes\":" + std::to_string(sample.cold_eligible_bytes);
    line += ",\"normalized_prefetch\":" + std::to_string(normalized_prefetch);
    line += ",\"normalized_fault\":" + std::to_string(normalized_fault);
    line += ",\"normalized_cold_candidates\":" + std::to_string(normalized_cold);
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void CalibrationProfile::on_step_end(
        int phase, uint64_t step, uint64_t latency_ns,
        const CalibrationStepContext & ctx) {
    (void) latency_ns;
    std::lock_guard<std::mutex> lock(mu);

    // Model reload: step went backwards.
    if (has_prev_step && step < prev_step) {
        reset_unlocked();
    }
    prev_step = step;
    has_prev_step = true;

    // Always update cumulative prev values so the next DECODE delta is correct
    // (matches ExpertRuntimeRescueController::on_step_end pattern).
    const uint64_t major_fault_delta = has_prev_major_faults &&
            ctx.current_major_faults >= prev_major_faults ?
            ctx.current_major_faults - prev_major_faults : 0;
    prev_major_faults = ctx.current_major_faults;
    has_prev_major_faults = true;

    const uint64_t cold_eligible_delta = has_prev_cold_eligible &&
            ctx.cumulative_cold_eligible_bytes >= prev_cold_eligible_bytes ?
            ctx.cumulative_cold_eligible_bytes - prev_cold_eligible_bytes : 0;
    prev_cold_eligible_bytes = ctx.cumulative_cold_eligible_bytes;
    has_prev_cold_eligible = true;

    const bool lifecycle_clean = lifecycle_clean_unlocked(ctx);
    prev_invariant_violations = ctx.invariant_violations;
    prev_cold_protected_violation = ctx.cold_protected_violation;
    prev_madv_cold_failed = ctx.madv_cold_failed;
    has_prev_lifecycle = true;

    // Non-DECODE: reset per-step accumulators and return.
    if (phase != LLM_MEM_TRACE_PHASE_DECODE) {
        step_opportunities = 0;
        step_issued = 0;
        return;
    }

    const uint64_t opportunities = step_opportunities;
    const uint64_t issued = step_issued;
    step_opportunities = 0;
    step_issued = 0;

    // Healthy-sample admission (spec §6).
    if (opportunities > 0 && ctx.rescue_state_safe && lifecycle_clean &&
            ctx.pending == 0 && ctx.active == 0 &&
            ctx.current_hint_inflight_objects == 0) {
        const double issue_ratio = (double) issued / (double) opportunities;
        if (issue_ratio >= 0.70) {
            CalibrationSample sample;
            sample.step = step;
            sample.prefetch_opportunities = opportunities;
            sample.prefetch_issued = issued;
            sample.major_fault_delta = major_fault_delta;
            sample.cold_eligible_bytes = cold_eligible_delta;
            samples.push_back(sample);

            if (samples.size() >= 16 &&
                    calibration_state == CalibrationState::Uncalibrated) {
                compute_baselines_unlocked();
                calibration_state = CalibrationState::Calibrated;
            }
        }
    }

    // Emit shadow normalized metrics after calibration is valid.
    if (calibration_state == CalibrationState::Calibrated) {
        CalibrationSample current;
        current.step = step;
        current.prefetch_opportunities = opportunities;
        current.prefetch_issued = issued;
        current.major_fault_delta = major_fault_delta;
        current.cold_eligible_bytes = cold_eligible_delta;
        emit_shadow_step_unlocked(step, current);
    }
}

static void append_baseline_json(
        std::string & line, const char * name, const CalibrationBaseline & bl) {
    line += ",\"" + std::string(name) + "\":{\"median\":" + std::to_string(bl.median);
    line += ",\"p25\":" + std::to_string(bl.p25);
    line += ",\"p75\":" + std::to_string(bl.p75);
    line += ",\"valid\":" + std::string(bl.valid ? "true" : "false");
    line += "}";
}

static double ratio_of(uint64_t value, double baseline) {
    return baseline > 0.0 ? (double) value / baseline : 0.0;
}

void CalibrationProfile::write_summary(const CalibrationSummaryContext & ctx) {
    std::lock_guard<std::mutex> lock(mu);
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }

    // Hardcoded scale-audit constants (Phase 2D/2E thresholds).
    const uint64_t degradation_issued_threshold = 100;
    const uint64_t recovery_issued_threshold = 500;
    const uint64_t degradation_fault_threshold = 2000;
    const uint64_t recovery_fault_threshold = 2500;
    const uint64_t baseline_b_cold_bytes = 271222731;   // ~258.7 MiB, Phase 2D full-rate B
    const uint64_t benefit_threshold_bytes = (1ull << 30); // 1 GiB

    std::string line;
    line.reserve(900);
    line += "{\"event\":\"EXPERT_CALIBRATION_SUMMARY\",\"ts_ns\":" +
            std::to_string(llm_mem_trace_time_ns());
    line += ",\"state\":";
    line += '"';
    line += calibration_state_name(calibration_state);
    line += '"';
    line += ",\"healthy_sample_count\":" + std::to_string(samples.size());

    append_baseline_json(line, "baseline_prefetch_issue_ratio", bl_prefetch_issue_ratio);
    append_baseline_json(line, "baseline_major_faults", bl_major_faults);
    append_baseline_json(line, "baseline_cold_eligible_bytes", bl_cold_eligible_bytes);

    line += ",\"median_opportunities\":" + std::to_string(bl_median_opportunities);
    line += ",\"median_issued\":" + std::to_string(bl_median_issued);

    // Scale audit: express hardcoded thresholds as ratios of learned baselines.
    line += ",\"scale_audit\":{";
    line += "\"degradation_issued_threshold\":" +
            std::to_string(degradation_issued_threshold);
    line += ",\"degradation_issued_ratio_of_median\":" +
            std::to_string(ratio_of(degradation_issued_threshold, bl_median_issued));
    line += ",\"recovery_issued_threshold\":" +
            std::to_string(recovery_issued_threshold);
    line += ",\"recovery_issued_ratio_of_median\":" +
            std::to_string(ratio_of(recovery_issued_threshold, bl_median_issued));
    line += ",\"degradation_fault_threshold\":" +
            std::to_string(degradation_fault_threshold);
    line += ",\"degradation_fault_ratio_of_baseline\":" +
            std::to_string(ratio_of(degradation_fault_threshold, bl_major_faults.median));
    line += ",\"recovery_fault_threshold\":" +
            std::to_string(recovery_fault_threshold);
    line += ",\"recovery_fault_ratio_of_baseline\":" +
            std::to_string(ratio_of(recovery_fault_threshold, bl_major_faults.median));
    line += ",\"baseline_b_cold_bytes\":" +
            std::to_string(baseline_b_cold_bytes);
    line += ",\"baseline_b_ratio_of_cold_eligible\":" +
            std::to_string(ratio_of(baseline_b_cold_bytes, bl_cold_eligible_bytes.median));
    line += ",\"benefit_threshold_bytes\":" +
            std::to_string(benefit_threshold_bytes);
    line += ",\"benefit_as_n_typical_steps\":" +
            std::to_string(ratio_of(benefit_threshold_bytes, bl_cold_eligible_bytes.median));
    line += "}";

    line += ",\"memory_current\":" + std::to_string(ctx.memory_current);
    line += ",\"memory_limit\":" + std::to_string(ctx.memory_limit);
    line += ",\"rss_bytes\":" + std::to_string(ctx.rss_bytes);
    line += ",\"working_set_budget_bytes\":" + std::to_string(ctx.working_set_budget_bytes);
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

CalibrationState CalibrationProfile::state() const {
    std::lock_guard<std::mutex> lock(mu);
    return calibration_state;
}

size_t CalibrationProfile::healthy_sample_count() const {
    std::lock_guard<std::mutex> lock(mu);
    return samples.size();
}

CalibrationBaseline CalibrationProfile::baseline_prefetch_issue_ratio() const {
    std::lock_guard<std::mutex> lock(mu);
    return bl_prefetch_issue_ratio;
}

CalibrationBaseline CalibrationProfile::baseline_major_faults() const {
    std::lock_guard<std::mutex> lock(mu);
    return bl_major_faults;
}

CalibrationBaseline CalibrationProfile::baseline_cold_eligible_bytes() const {
    std::lock_guard<std::mutex> lock(mu);
    return bl_cold_eligible_bytes;
}

double CalibrationProfile::median_opportunities() const {
    std::lock_guard<std::mutex> lock(mu);
    return bl_median_opportunities;
}

double CalibrationProfile::median_issued() const {
    std::lock_guard<std::mutex> lock(mu);
    return bl_median_issued;
}

CalibrationProfile & expert_calibration_profile() {
    static CalibrationProfile profile;
    return profile;
}
