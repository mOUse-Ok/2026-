// Phase 2E-A: unit tests for the observation-only CalibrationProfile.
// Tests sample admission, baseline computation, state transitions, and
// no-control-impact. Compiled WITHOUT LLM_MEM_TRACE so trace writes are no-ops.

#include "expert_calibration_shadow.h"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Trace stubs: the global LLAMA_MEM_TRACE compile definition makes
// expert_calibration_shadow.cpp expect real trace symbols. The test does not
// link the full trace library, so provide no-op stubs.
extern "C" {
uint64_t llm_mem_trace_time_ns(void) { return 0; }
void llm_mem_trace_write(int sink, const char * line, size_t len) {
    (void) sink; (void) line; (void) len;
}
int llm_mem_trace_sink_enabled(int sink) { (void) sink; return 0; }
}

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-expert-calibration-shadow: %s\n", message);
        std::abort();
    }
}

static bool approx_equal(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) < eps;
}

// Drive a healthy DECODE step with the given cumulative counters.
static void drive_healthy_decode_step(
        CalibrationProfile & profile,
        uint64_t step,
        uint64_t opportunities,
        uint64_t issued,
        uint64_t cumulative_major_faults,
        uint64_t cumulative_cold_eligible) {
    for (uint64_t i = 0; i < opportunities; ++i) {
        profile.record_prefetch_opportunity(LLM_MEM_TRACE_PHASE_DECODE);
    }
    for (uint64_t i = 0; i < issued; ++i) {
        profile.record_prefetch_issued(LLM_MEM_TRACE_PHASE_DECODE);
    }
    CalibrationStepContext ctx;
    ctx.rescue_state_safe = true;
    ctx.current_major_faults = cumulative_major_faults;
    ctx.cumulative_cold_eligible_bytes = cumulative_cold_eligible;
    ctx.invariant_violations = 0;
    ctx.cold_protected_violation = 0;
    ctx.madv_cold_failed = 0;
    ctx.pending = 0;
    ctx.active = 0;
    ctx.current_hint_inflight_objects = 0;
    profile.on_step_end(LLM_MEM_TRACE_PHASE_DECODE, step, 1000, ctx);
}

static void test_few_samples_stays_uncalibrated() {
    CalibrationProfile profile;
    // Prime (has_prev_lifecycle false → not admitted).
    drive_healthy_decode_step(profile, 1, 10, 8, 10, 1000);
    require(profile.state() == CalibrationState::Uncalibrated,
            "should be uncalibrated after prime");
    require(profile.healthy_sample_count() == 0, "no samples after prime");

    // 15 healthy steps → 15 samples → still uncalibrated.
    for (uint64_t step = 2; step <= 16; ++step) {
        drive_healthy_decode_step(profile, step, 10, 8, step * 10, step * 1000);
    }
    require(profile.healthy_sample_count() == 15, "should have 15 samples");
    require(profile.state() == CalibrationState::Uncalibrated,
            "15 samples should still be uncalibrated");
}

static void test_sixteen_samples_becomes_calibrated() {
    CalibrationProfile profile;
    drive_healthy_decode_step(profile, 1, 10, 8, 10, 1000); // prime
    for (uint64_t step = 2; step <= 17; ++step) {
        drive_healthy_decode_step(profile, step, 10, 8, step * 10, step * 1000);
    }
    require(profile.healthy_sample_count() == 16, "should have 16 samples");
    require(profile.state() == CalibrationState::Calibrated,
            "16 samples should be calibrated");
}

static void test_unhealthy_samples_excluded() {
    CalibrationProfile profile;
    drive_healthy_decode_step(profile, 1, 10, 8, 10, 1000); // prime
    require(profile.healthy_sample_count() == 0, "prime should not be admitted");

    // Low issue ratio (3/10 = 0.30 < 0.70).
    drive_healthy_decode_step(profile, 2, 10, 3, 20, 2000);
    require(profile.healthy_sample_count() == 0, "low ratio should be excluded");

    // Rescue unsafe.
    {
        for (int i = 0; i < 10; ++i) {
            profile.record_prefetch_opportunity(LLM_MEM_TRACE_PHASE_DECODE);
        }
        for (int i = 0; i < 8; ++i) {
            profile.record_prefetch_issued(LLM_MEM_TRACE_PHASE_DECODE);
        }
        CalibrationStepContext ctx;
        ctx.rescue_state_safe = false;
        ctx.current_major_faults = 30;
        ctx.cumulative_cold_eligible_bytes = 3000;
        ctx.invariant_violations = 0;
        ctx.cold_protected_violation = 0;
        ctx.madv_cold_failed = 0;
        ctx.pending = 0;
        ctx.active = 0;
        ctx.current_hint_inflight_objects = 0;
        profile.on_step_end(LLM_MEM_TRACE_PHASE_DECODE, 3, 1000, ctx);
    }
    require(profile.healthy_sample_count() == 0, "rescue unsafe should be excluded");

    // Lifecycle dirty (new invariant violation).
    {
        for (int i = 0; i < 10; ++i) {
            profile.record_prefetch_opportunity(LLM_MEM_TRACE_PHASE_DECODE);
        }
        for (int i = 0; i < 8; ++i) {
            profile.record_prefetch_issued(LLM_MEM_TRACE_PHASE_DECODE);
        }
        CalibrationStepContext ctx;
        ctx.rescue_state_safe = true;
        ctx.current_major_faults = 40;
        ctx.cumulative_cold_eligible_bytes = 4000;
        ctx.invariant_violations = 1;
        ctx.cold_protected_violation = 0;
        ctx.madv_cold_failed = 0;
        ctx.pending = 0;
        ctx.active = 0;
        ctx.current_hint_inflight_objects = 0;
        profile.on_step_end(LLM_MEM_TRACE_PHASE_DECODE, 4, 1000, ctx);
    }
    require(profile.healthy_sample_count() == 0, "lifecycle dirty should be excluded");

    // Pending != 0 (no new violations, but point-in-time dirty).
    {
        for (int i = 0; i < 10; ++i) {
            profile.record_prefetch_opportunity(LLM_MEM_TRACE_PHASE_DECODE);
        }
        for (int i = 0; i < 8; ++i) {
            profile.record_prefetch_issued(LLM_MEM_TRACE_PHASE_DECODE);
        }
        CalibrationStepContext ctx;
        ctx.rescue_state_safe = true;
        ctx.current_major_faults = 50;
        ctx.cumulative_cold_eligible_bytes = 5000;
        ctx.invariant_violations = 1;
        ctx.cold_protected_violation = 0;
        ctx.madv_cold_failed = 0;
        ctx.pending = 1;
        ctx.active = 0;
        ctx.current_hint_inflight_objects = 0;
        profile.on_step_end(LLM_MEM_TRACE_PHASE_DECODE, 5, 1000, ctx);
    }
    require(profile.healthy_sample_count() == 0, "pending != 0 should be excluded");

    // A healthy step should now be admitted.
    drive_healthy_decode_step(profile, 6, 10, 8, 60, 6000);
    require(profile.healthy_sample_count() == 1, "healthy step should be admitted");
}

static void test_median_computation() {
    CalibrationProfile profile;
    drive_healthy_decode_step(profile, 1, 10, 8, 10, 1000); // prime
    // 16 identical healthy steps.
    for (uint64_t step = 2; step <= 17; ++step) {
        drive_healthy_decode_step(profile, step, 10, 8, step * 10, step * 1000);
    }
    require(profile.state() == CalibrationState::Calibrated, "should be calibrated");

    // All samples: issue_ratio = 8/10 = 0.8, faults = 10, cold = 1000.
    const CalibrationBaseline bl_ratio = profile.baseline_prefetch_issue_ratio();
    require(bl_ratio.valid, "issue ratio baseline should be valid");
    require(approx_equal(bl_ratio.median, 0.8), "median issue ratio should be 0.8");
    require(approx_equal(bl_ratio.p25, 0.8), "p25 issue ratio should be 0.8");
    require(approx_equal(bl_ratio.p75, 0.8), "p75 issue ratio should be 0.8");

    const CalibrationBaseline bl_faults = profile.baseline_major_faults();
    require(bl_faults.valid, "fault baseline should be valid");
    require(approx_equal(bl_faults.median, 10.0), "median faults should be 10");

    const CalibrationBaseline bl_cold = profile.baseline_cold_eligible_bytes();
    require(bl_cold.valid, "cold eligible baseline should be valid");
    require(approx_equal(bl_cold.median, 1000.0), "median cold eligible should be 1000");

    require(approx_equal(profile.median_opportunities(), 10.0),
            "median opportunities should be 10");
    require(approx_equal(profile.median_issued(), 8.0),
            "median issued should be 8");
}

static void test_non_decode_no_samples() {
    CalibrationProfile profile;
    drive_healthy_decode_step(profile, 1, 10, 8, 10, 1000); // prime
    // PREFILL step — should not produce a sample.
    CalibrationStepContext ctx;
    ctx.rescue_state_safe = true;
    ctx.current_major_faults = 20;
    ctx.cumulative_cold_eligible_bytes = 2000;
    ctx.invariant_violations = 0;
    ctx.cold_protected_violation = 0;
    ctx.madv_cold_failed = 0;
    ctx.pending = 0;
    ctx.active = 0;
    ctx.current_hint_inflight_objects = 0;
    profile.on_step_end(LLM_MEM_TRACE_PHASE_PREFILL, 2, 1000, ctx);
    require(profile.healthy_sample_count() == 0, "PREFILL should not produce samples");
}

static void test_model_reload_resets() {
    CalibrationProfile profile;
    drive_healthy_decode_step(profile, 1, 10, 8, 10, 1000); // prime
    for (uint64_t step = 2; step <= 17; ++step) {
        drive_healthy_decode_step(profile, step, 10, 8, step * 10, step * 1000);
    }
    require(profile.state() == CalibrationState::Calibrated, "should be calibrated");
    require(profile.healthy_sample_count() == 16, "should have 16 samples");

    // Step goes backwards → model reload → full reset.
    drive_healthy_decode_step(profile, 1, 10, 8, 10, 1000);
    require(profile.state() == CalibrationState::Uncalibrated,
            "should be uncalibrated after reload");
    require(profile.healthy_sample_count() == 0, "should have 0 samples after reload");
}

static void test_no_control_impact() {
    CalibrationProfile profile1;
    CalibrationProfile profile2;
    // Drive profile1 with data.
    drive_healthy_decode_step(profile1, 1, 10, 8, 10, 1000);
    drive_healthy_decode_step(profile1, 2, 10, 8, 20, 2000);
    // profile2 must be unaffected.
    require(profile2.state() == CalibrationState::Uncalibrated,
            "profile2 should be uncalibrated");
    require(profile2.healthy_sample_count() == 0,
            "profile2 should have 0 samples");
    // Singleton must be unaffected.
    require(expert_calibration_profile().state() == CalibrationState::Uncalibrated,
            "singleton should be uncalibrated");
    require(expert_calibration_profile().healthy_sample_count() == 0,
            "singleton should have 0 samples");
}

int main() {
    test_few_samples_stays_uncalibrated();
    test_sixteen_samples_becomes_calibrated();
    test_unhealthy_samples_excluded();
    test_median_computation();
    test_non_decode_no_samples();
    test_model_reload_resets();
    test_no_control_impact();
    std::printf("test-expert-calibration-shadow: all tests passed\n");
    return 0;
}
