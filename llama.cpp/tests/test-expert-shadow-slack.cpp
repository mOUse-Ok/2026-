#include "expert_shadow_slack.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "test-expert-shadow-slack: " << message << '\n';
        std::exit(1);
    }
}

ExpertShadowConfig config() {
    ExpertShadowConfig result;
    result.enabled = true;
    result.window_capacity = 4;
    result.min_samples = 2;
    result.ewma_alpha = 0.5;
    result.horizon_default_ns = 1000;
    result.horizon_min_ns = 10;
    result.horizon_max_ns = 10'000;
    result.worker_default_ns = 100;
    result.worker_min_ns = 10;
    result.worker_max_ns = 10'000;
    result.pre_issue_default_ns = 50;
    result.pre_issue_min_ns = 10;
    result.pre_issue_max_ns = 10'000;
    result.syscall_service_default_ns = 50;
    result.syscall_service_min_ns = 10;
    result.syscall_service_max_ns = 10'000;
    result.throughput_default_bytes_per_ns = 10.0;
    result.max_pending_tasks = 32;
    result.max_estimator_cells = 128;
    result.step_retention = 2;
    return result;
}

ExpertShadowTaskInput task(
        uint64_t id,
        uint64_t step,
        uint64_t prediction_ts,
        int layer = 3,
        ExpertTensorStage stage = ExpertTensorStage::Early,
        int phase = 2) {
    ExpertShadowTaskInput result;
    result.task_id = id;
    result.step = step;
    result.layer = layer;
    result.expert = 7;
    result.phase = phase;
    result.stage = stage;
    result.tensor = stage == ExpertTensorStage::Late ?
            "blk.3.ffn_down_exps.weight" : "blk.3.ffn_up_exps.weight";
    result.addr = 0x1000;
    result.nbytes = 1000;
    result.prediction_ts_ns = prediction_ts;
    result.enqueued_ts_ns = prediction_ts;
    result.queue_depth_before_enqueue = 4;
    result.queued_bytes_before_enqueue = 1000;
    result.active_workers = 2;
    return result;
}

ExpertShadowFirstUseInput first_use(const ExpertShadowTaskInput & input, uint64_t ts) {
    ExpertShadowFirstUseInput result;
    result.step = input.step;
    result.layer = input.layer;
    result.expert = input.expert;
    result.phase = input.phase;
    result.stage = input.stage;
    result.tensor = input.tensor;
    result.addr = input.addr;
    result.nbytes = input.nbytes;
    result.first_use_ts_ns = ts;
    return result;
}

ExpertShadowIssueInput issue(
        std::vector<uint64_t> ids,
        uint64_t issue_id,
        uint64_t issue_ts,
        uint64_t returned_ts,
        size_t issued_nbytes = 1000) {
    ExpertShadowIssueInput result;
    result.task_ids = std::move(ids);
    result.issue_id = issue_id;
    result.issue_task_count = result.task_ids.size();
    result.issue_ts_ns = issue_ts;
    result.returned_ts_ns = returned_ts;
    result.issued_nbytes = issued_nbytes;
    return result;
}

ExpertShadowTaskObservation finish(
        ExpertShadowSlack & shadow,
        const ExpertShadowTaskInput & input,
        uint64_t dequeue_ts,
        uint64_t issue_ts,
        uint64_t return_ts,
        uint64_t first_use_ts) {
    shadow.observe_dequeue(input.task_id, dequeue_ts);
    require(shadow.observe_issue_group(
            issue({input.task_id}, input.task_id, issue_ts, return_ts)).empty(),
            "task finalized before first-use");
    std::vector<ExpertShadowTaskObservation> finalized =
            shadow.observe_first_use(first_use(input, first_use_ts));
    if (finalized.size() != 1) {
        const ExpertShadowSummary state = shadow.summary();
        std::cerr << "finalize diagnostic: pending=" << state.pending_tasks
                  << " first_uses=" << state.logical_first_uses
                  << " unmatched=" << state.unmatched_first_uses
                  << " expired=" << state.expired_tasks << '\n';
    }
    require(finalized.size() == 1, "task did not finalize at first-use");
    return finalized.front();
}

void test_models_warmup_and_fallback() {
    ExpertShadowSlack shadow(config());
    ExpertShadowTaskInput first = task(1, 1, 1000);
    require(shadow.register_task(first), "first task registration failed");
    ExpertShadowSummary initial = shadow.summary();
    require(initial.predicted_tasks == 1 && initial.pending_tasks == 1,
            "initial prediction accounting mismatch");

    ExpertShadowTaskObservation first_done = finish(shadow, first, 1100, 1150, 1200, 2000);
    require(first_done.predictions.size() == 36, "candidate matrix is not 3x3x2x2");
    require(first_done.predictions[0].predicted_queue_wait_ns == 200,
            "queue-depth model formula mismatch");
    require(first_done.predictions[2].predicted_queue_wait_ns == 100,
            "queued-bytes model formula mismatch");
    require(!first_done.predictions[2].prediction_available &&
            first_done.predictions[2].queue_sample_count == 0,
            "queued-bytes model did not mark missing causal throughput unavailable");
    require(first_done.predictions[0].predicted_worker_occupied_ns == 100,
            "worker occupied fallback mismatch");
    require(first_done.predictions[0].predicted_pre_issue_overhead_ns == 50 &&
            first_done.predictions[0].predicted_hint_syscall_service_ns == 50,
            "split worker component fallbacks mismatch");
    require(first_done.predictions[0].predicted_issue_slack_ns == 750,
            "issue slack formula mismatch");
    require(first_done.predictions[0].predicted_return_slack_ns == 700,
            "return slack formula mismatch");
    require(first_done.actual_pre_issue_overhead_ns == 50 &&
            first_done.actual_hint_syscall_service_ns == 50 &&
            first_done.actual_worker_occupied_ns == 100,
            "actual worker timing decomposition mismatch");
    require(first_done.actual_issue_slack_ns == 850 &&
            first_done.actual_return_slack_ns == 800,
            "actual target slack semantics mismatch");
    require(first_done.predictions[0].estimator_warmup,
            "first estimate should be warmup");

    ExpertShadowTaskInput second = task(2, 2, 3000);
    require(shadow.register_task(second), "second task registration failed");
    ExpertShadowTaskObservation second_done = finish(shadow, second, 3100, 3150, 3200, 5000);
    require(second_done.predictions[2].prediction_available,
            "queued-bytes model stayed unavailable after an earlier issue sample");
    require(second_done.predictions[2].queue_fallback_level ==
                    ExpertShadowFallbackLevel::StaticDefault,
            "immature throughput model did not report its static fallback");
    require(second_done.predictions[0].estimator_sample_count == 1,
            "second prediction leaked its own label");
    require(second_done.predictions[0].fallback_level == ExpertShadowFallbackLevel::StaticDefault,
            "immature cell did not use documented fallback");

    ExpertShadowTaskInput third = task(3, 3, 6000);
    require(shadow.register_task(third), "third task registration failed");
    require(shadow.observe_first_use(first_use(third, 7000)).empty(),
            "third task finalized without issue");
    shadow.observe_dequeue(third.task_id, 6100);
    std::vector<ExpertShadowTaskObservation> third_done = shadow.observe_issue_group(
            issue({third.task_id}, 3, 6150, 6200));
    require(third_done.size() == 1, "first-use-before-issue did not finalize");
    require(third_done[0].predictions[0].estimator_sample_count == 2,
            "mature exact sample count mismatch");
    require(!third_done[0].predictions[0].estimator_warmup,
            "mature exact estimate still marked warmup");
    require(third_done[0].predictions[0].fallback_level == ExpertShadowFallbackLevel::Exact,
            "mature exact estimate unexpectedly fell back");
    require(third_done[0].predictions[2].queue_fallback_level ==
                    ExpertShadowFallbackLevel::Exact,
            "mature throughput model unexpectedly reported fallback");
    require(third_done[0].predictions[0].predicted_first_use_horizon_ns == 1500,
            "EWMA horizon mismatch");
    require(third_done[0].predictions[4].predicted_first_use_horizon_ns == 1500,
            "median horizon mismatch");
    require(third_done[0].predictions[8].predicted_first_use_horizon_ns == 1250,
            "p25 horizon mismatch");

    ExpertShadowTaskInput other_layer = task(4, 4, 8000, 9);
    require(shadow.register_task(other_layer), "other-layer registration failed");
    require(shadow.observe_first_use(first_use(other_layer, 9000)).empty(),
            "other-layer task finalized without issue");
    shadow.observe_dequeue(other_layer.task_id, 8100);
    std::vector<ExpertShadowTaskObservation> other_done = shadow.observe_issue_group(
            issue({other_layer.task_id}, 4, 8150, 8200));
    require(other_done.size() == 1, "other-layer task did not finalize");
    require(other_done[0].predictions[0].estimator_sample_count == 0,
            "layer-specific cell consumed another layer");
    require(other_done[0].predictions[0].fallback_level == ExpertShadowFallbackLevel::Phase,
            "phase fallback was not selected for unseen layer");
}

void test_phase_stage_and_worker_bucket_isolation() {
    ExpertShadowConfig cfg = config();
    cfg.min_samples = 1;
    ExpertShadowSlack shadow(cfg);

    ExpertShadowTaskInput decode_early = task(30, 1, 1000, 3, ExpertTensorStage::Early, 2);
    require(shadow.register_task(decode_early), "decode seed registration failed");
    (void) finish(shadow, decode_early, 1100, 1200, 1500, 2000);

    ExpertShadowTaskInput prefill_early = task(31, 2, 3000, 3, ExpertTensorStage::Early, 1);
    require(shadow.register_task(prefill_early), "prefill isolation registration failed");
    require(prefill_early.phase != decode_early.phase, "phase isolation setup is invalid");
    shadow.observe_dequeue(prefill_early.task_id, 3100);
    require(shadow.observe_issue_group(issue({prefill_early.task_id}, 31, 3200, 3300)).empty(),
            "prefill isolation task finalized before first-use");
    std::vector<ExpertShadowTaskObservation> prefill_done =
            shadow.observe_first_use(first_use(prefill_early, 4000));
    require(prefill_done.size() == 1, "prefill isolation task did not finalize");
    require(prefill_done[0].predictions[0].estimator_sample_count == 0 &&
            prefill_done[0].predictions[0].fallback_level ==
                    ExpertShadowFallbackLevel::StaticDefault,
            "PREFILL prediction consumed DECODE fallback samples");

    ExpertShadowTaskInput decode_late = task(32, 3, 5000, 3, ExpertTensorStage::Late, 2);
    require(shadow.register_task(decode_late), "late isolation registration failed");
    shadow.observe_dequeue(decode_late.task_id, 5100);
    require(shadow.observe_issue_group(issue({decode_late.task_id}, 32, 5200, 5300)).empty(),
            "late isolation task finalized before first-use");
    std::vector<ExpertShadowTaskObservation> late_done =
            shadow.observe_first_use(first_use(decode_late, 6000));
    require(late_done.size() == 1, "late isolation task did not finalize");
    require(late_done[0].predictions[12].estimator_sample_count == 0 &&
            late_done[0].predictions[12].fallback_level == ExpertShadowFallbackLevel::Phase,
            "LATE phase-stage prediction consumed EARLY exact samples");

    ExpertShadowTaskInput decode_unknown = task(
            33, 4, 7000, 3, ExpertTensorStage::Unknown, 2);
    require(shadow.register_task(decode_unknown), "unknown isolation registration failed");
    shadow.observe_dequeue(decode_unknown.task_id, 7100);
    require(shadow.observe_issue_group(issue({decode_unknown.task_id}, 33, 7200, 7300)).empty(),
            "unknown isolation task finalized before first-use");
    std::vector<ExpertShadowTaskObservation> unknown_done =
            shadow.observe_first_use(first_use(decode_unknown, 8000));
    require(unknown_done.size() == 1, "unknown isolation task did not finalize");
    require(unknown_done[0].predictions[12].estimator_sample_count == 0 &&
            unknown_done[0].predictions[12].fallback_level == ExpertShadowFallbackLevel::Phase,
            "UNKNOWN phase-stage prediction consumed EARLY/LATE exact samples");

    ExpertShadowTaskInput large = task(34, 5, 9000);
    large.nbytes = 32 * 1024 * 1024;
    require(shadow.register_task(large), "large worker-bucket registration failed");
    require(large.nbytes > 16 * 1024 * 1024, "worker bucket setup is invalid");
    shadow.observe_dequeue(large.task_id, 9100);
    require(shadow.observe_issue_group(
            issue({large.task_id}, 34, 9200, 9800, large.nbytes)).empty(),
            "large worker-bucket task finalized before first-use");
    std::vector<ExpertShadowTaskObservation> large_done =
            shadow.observe_first_use(first_use(large, 11'000));
    require(large_done.size() == 1, "large worker-bucket task did not finalize");
    require(large_done[0].predictions[0].worker_sample_count == 0 &&
            large_done[0].predictions[0].worker_fallback_level ==
                    ExpertShadowFallbackLevel::Global,
            "unseen large worker bucket did not use global fallback");

    ExpertShadowTaskInput large_again = task(35, 6, 12'000);
    large_again.nbytes = large.nbytes;
    require(shadow.register_task(large_again), "mature large bucket registration failed");
    require(large_again.nbytes == large.nbytes, "large worker bucket changed unexpectedly");
    shadow.observe_dequeue(large_again.task_id, 12'100);
    require(shadow.observe_issue_group(
            issue({large_again.task_id}, 35, 12'200, 12'300, large_again.nbytes)).empty(),
            "mature large task finalized before first-use");
    std::vector<ExpertShadowTaskObservation> mature_large =
            shadow.observe_first_use(first_use(large_again, 13'000));
    require(mature_large.size() == 1, "mature large task did not finalize");
    require(mature_large[0].predictions[0].worker_sample_count == 1 &&
            mature_large[0].predictions[0].worker_fallback_level ==
                    ExpertShadowFallbackLevel::Exact,
            "large worker size bucket did not become independently mature");
}

void test_late_equal_coalesced_and_calibration() {
    ExpertShadowConfig cfg = config();
    cfg.min_samples = 1;
    cfg.horizon_default_ns = 10'000;
    cfg.horizon_max_ns = 20'000;
    ExpertShadowSlack shadow(cfg);

    ExpertShadowTaskInput late = task(10, 10, 10'000);
    require(shadow.register_task(late), "late task registration failed");
    shadow.observe_dequeue(late.task_id, 10'100);
    require(shadow.observe_first_use(first_use(late, 10'500)).empty(),
            "late task finalized before issue");
    std::vector<ExpertShadowTaskObservation> late_done = shadow.observe_issue_group(
            issue({late.task_id}, 10, 10'600, 10'700));
    require(late_done.size() == 1 && late_done[0].issue_ts_ns > late_done[0].first_use_ts_ns,
            "late issue sample was hidden");

    ExpertShadowTaskInput equal = task(11, 11, 20'000, 3, ExpertTensorStage::Late);
    require(shadow.register_task(equal), "equal task registration failed");
    shadow.observe_dequeue(equal.task_id, 20'100);
    require(shadow.observe_issue_group(issue({equal.task_id}, 11, 20'500, 20'600)).empty(),
            "equal task finalized before first-use");
    std::vector<ExpertShadowTaskObservation> equal_done =
            shadow.observe_first_use(first_use(equal, 20'500));
    require(equal_done.size() == 1, "equal task did not finalize");

    ExpertShadowTaskInput a = task(12, 12, 30'000);
    ExpertShadowTaskInput b = task(13, 12, 30'000);
    b.addr += 128;
    require(shadow.register_task(a) && shadow.register_task(b), "coalesced registration failed");
    shadow.observe_dequeue(a.task_id, 30'100);
    shadow.observe_dequeue(b.task_id, 30'100);
    require(shadow.observe_issue_group(issue({a.task_id, b.task_id}, 12, 30'200, 30'300, 2000)).empty(),
            "coalesced tasks finalized before first-use");
    std::vector<ExpertShadowTaskObservation> coalesced =
            shadow.observe_first_use(first_use(a, 31'000));
    require(coalesced.size() == 2, "one-to-many first-use association was lost");
    require(coalesced[0].coalesced && coalesced[1].issue_task_count == 2,
            "coalesced identity fields mismatch");

    ExpertShadowSummary summary = shadow.summary();
    require(summary.worker_duration_observations == 3,
            "coalesced issue updated worker model more than once");
    require(summary.ambiguous_first_uses == 1, "ambiguous first-use was not counted");
    require(summary.candidates[0].issue_overall.false_positive >= 1,
            "late predicted-on-time sample was not a false positive");
    require(summary.candidates[0].issue_overall.true_negative +
            summary.candidates[0].issue_overall.false_positive >= 2,
            "equal issue was not classified as actual late");
}

void test_causal_online_residual_calibration() {
    ExpertShadowConfig cfg = config();
    cfg.min_samples = 1;
    cfg.residual_quantile = 0.25;
    ExpertShadowSlack shadow(cfg);

    ExpertShadowTaskInput seed = task(40, 1, 1000);
    require(shadow.register_task(seed), "residual seed registration failed");
    (void) finish(shadow, seed, 1100, 1150, 1200, 2000);

    ExpertShadowTaskInput residual_seed = task(41, 2, 3000);
    require(shadow.register_task(residual_seed), "residual sample registration failed");
    ExpertShadowTaskObservation residual_seed_done =
            finish(shadow, residual_seed, 3100, 3150, 3200, 6000);
    require(residual_seed_done.predictions[1].residual_sample_count == 0 &&
            residual_seed_done.predictions[1].residual_adjustment_ns == 0,
            "current outcome leaked into its own residual prediction");

    ExpertShadowTaskInput calibrated = task(42, 3, 7000);
    require(shadow.register_task(calibrated), "calibrated registration failed");
    ExpertShadowTaskObservation calibrated_done =
            finish(shadow, calibrated, 7100, 7150, 7200, 11'000);
    require(calibrated_done.predictions[0].raw_predicted_first_use_horizon_ns == 2000,
            "raw EWMA horizon is unexpected");
    require(calibrated_done.predictions[1].calibration_model ==
                    ExpertShadowCalibrationModel::ResidualQuantile &&
            calibrated_done.predictions[1].residual_sample_count == 1 &&
            calibrated_done.predictions[1].residual_adjustment_ns == 2000 &&
            calibrated_done.predictions[1].predicted_first_use_horizon_ns == 4000,
            "causal residual quantile was not applied from older mature history");

    ExpertShadowTaskInput other_phase = task(
            43, 4, 12'000, 3, ExpertTensorStage::Early, 1);
    require(shadow.register_task(other_phase), "residual phase-isolation registration failed");
    ExpertShadowTaskObservation other_phase_done =
            finish(shadow, other_phase, 12'100, 12'150, 12'200, 13'000);
    require(other_phase_done.predictions[1].residual_sample_count == 0 &&
            other_phase_done.predictions[1].residual_fallback_level ==
                    ExpertShadowFallbackLevel::StaticDefault,
            "PREFILL residual prediction consumed DECODE history");
}

void test_capacity_phase_stage_and_overflow() {
    ExpertShadowConfig cfg = config();
    cfg.max_pending_tasks = 1;
    cfg.horizon_default_ns = cfg.horizon_max_ns;
    ExpertShadowSlack shadow(cfg);
    ExpertShadowTaskInput first = task(20, 3, 1000);
    ExpertShadowTaskInput second = task(21, 3, std::numeric_limits<uint64_t>::max() - 5,
                                        3, ExpertTensorStage::Late, 1);
    require(shadow.register_task(first), "capacity first registration failed");
    require(shadow.register_task(second), "capacity second registration failed");
    ExpertShadowSummary summary = shadow.summary();
    require(summary.capacity_expired_tasks == 1 && summary.pending_tasks == 1,
            "bounded pending capacity did not evict exactly one task");

    shadow.observe_dequeue(second.task_id, second.enqueued_ts_ns);
    require(shadow.observe_issue_group(issue(
            {second.task_id}, 21, second.prediction_ts_ns, second.prediction_ts_ns, 1000)).empty(),
            "overflow task finalized before first-use");
    std::vector<ExpertShadowTaskObservation> done = shadow.observe_first_use(
            first_use(second, std::numeric_limits<uint64_t>::max()));
    require(done.size() == 1, "overflow task did not finalize");
    require(done[0].predictions[0].predicted_first_use_ts_ns ==
            std::numeric_limits<uint64_t>::max(),
            "timestamp addition did not saturate");
    require(done[0].predictions[12].estimator_sample_count == 0,
            "phase-stage estimate consumed a different phase/stage");
}

} // namespace

int main() {
    test_models_warmup_and_fallback();
    test_late_equal_coalesced_and_calibration();
    test_capacity_phase_stage_and_overflow();
    test_phase_stage_and_worker_bucket_isolation();
    test_causal_online_residual_calibration();
    std::cout << "expert shadow slack tests passed\n";
    return 0;
}
