#pragma once

#include "expert_tensor_stage.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class ExpertShadowEstimatorKind {
    Ewma,
    Median,
    P25,
};

enum class ExpertShadowGrouping {
    PhaseLayer,
    PhaseStage,
    PhaseLayerStage,
};

enum class ExpertShadowQueueModel {
    QueueDepthWorkerEwma,
    QueuedBytesIssueThroughput,
};

enum class ExpertShadowCalibrationModel {
    Raw,
    ResidualQuantile,
};

enum class ExpertShadowFallbackLevel {
    Exact,
    PhaseStage,
    Phase,
    Global,
    StaticDefault,
};

const char * expert_shadow_estimator_name(ExpertShadowEstimatorKind kind);
const char * expert_shadow_grouping_name(ExpertShadowGrouping grouping);
const char * expert_shadow_queue_model_name(ExpertShadowQueueModel model);
const char * expert_shadow_calibration_model_name(ExpertShadowCalibrationModel model);
const char * expert_shadow_fallback_name(ExpertShadowFallbackLevel level);

struct ExpertShadowConfig {
    bool enabled = false;
    bool config_error = false;
    size_t window_capacity = 64;
    uint64_t min_samples = 8;
    double ewma_alpha = 0.2;
    double residual_quantile = 0.25;
    uint64_t horizon_default_ns = 5'000'000;
    uint64_t horizon_min_ns = 1'000;
    uint64_t horizon_max_ns = 5'000'000'000ull;
    uint64_t worker_default_ns = 50'000;
    uint64_t worker_min_ns = 100;
    uint64_t worker_max_ns = 1'000'000'000ull;
    uint64_t pre_issue_default_ns = 10'000;
    uint64_t pre_issue_min_ns = 100;
    uint64_t pre_issue_max_ns = 1'000'000'000ull;
    uint64_t syscall_service_default_ns = 40'000;
    uint64_t syscall_service_min_ns = 100;
    uint64_t syscall_service_max_ns = 1'000'000'000ull;
    double throughput_default_bytes_per_ns = 1.0;
    size_t max_pending_tasks = 8192;
    size_t max_first_use_keys = 65536;
    size_t max_estimator_cells = 4096;
    size_t max_residual_cells = 4096;
    uint64_t step_retention = 2;
};

struct ExpertShadowPrediction {
    ExpertShadowGrouping grouping = ExpertShadowGrouping::PhaseLayer;
    ExpertShadowEstimatorKind estimator = ExpertShadowEstimatorKind::Ewma;
    ExpertShadowQueueModel queue_model = ExpertShadowQueueModel::QueueDepthWorkerEwma;
    ExpertShadowCalibrationModel calibration_model = ExpertShadowCalibrationModel::Raw;
    ExpertShadowFallbackLevel fallback_level = ExpertShadowFallbackLevel::StaticDefault;
    ExpertShadowFallbackLevel queue_fallback_level = ExpertShadowFallbackLevel::StaticDefault;
    ExpertShadowFallbackLevel worker_fallback_level = ExpertShadowFallbackLevel::StaticDefault;
    ExpertShadowFallbackLevel pre_issue_fallback_level = ExpertShadowFallbackLevel::StaticDefault;
    ExpertShadowFallbackLevel syscall_service_fallback_level = ExpertShadowFallbackLevel::StaticDefault;
    ExpertShadowFallbackLevel residual_fallback_level = ExpertShadowFallbackLevel::StaticDefault;
    uint64_t predicted_first_use_ts_ns = 0;
    uint64_t predicted_first_use_horizon_ns = 0;
    uint64_t raw_predicted_first_use_horizon_ns = 0;
    int64_t residual_adjustment_ns = 0;
    uint64_t predicted_queue_wait_ns = 0;
    uint64_t predicted_pre_issue_overhead_ns = 0;
    uint64_t predicted_hint_syscall_service_ns = 0;
    uint64_t predicted_worker_occupied_ns = 0;
    // Deprecated M4A compatibility alias. New Trace writers must not emit this
    // as an aligned Issue target.
    uint64_t predicted_worker_issue_ns = 0;
    int64_t predicted_issue_slack_ns = 0;
    int64_t predicted_return_slack_ns = 0;
    uint64_t estimator_sample_count = 0;
    uint64_t estimator_effective_sample_count = 0;
    uint64_t residual_sample_count = 0;
    uint64_t residual_effective_sample_count = 0;
    uint64_t queue_sample_count = 0;
    uint64_t worker_sample_count = 0;
    uint64_t pre_issue_sample_count = 0;
    uint64_t syscall_service_sample_count = 0;
    bool estimator_warmup = true;
    bool residual_warmup = true;
    bool queue_warmup = true;
    bool worker_warmup = true;
    bool pre_issue_warmup = true;
    bool syscall_service_warmup = true;
    bool prediction_available = true;
    bool issue_prediction_available = true;
    bool return_prediction_available = true;
    bool clipped_low = false;
    bool clipped_high = false;
};

struct ExpertShadowTaskInput {
    uint64_t task_id = 0;
    uint64_t step = 0;
    int layer = -1;
    int expert = -1;
    int phase = 0;
    ExpertTensorStage stage = ExpertTensorStage::Unknown;
    std::string tensor;
    uintptr_t addr = 0;
    size_t nbytes = 0;
    uint64_t prediction_ts_ns = 0;
    uint64_t enqueued_ts_ns = 0;
    uint64_t queue_depth_before_enqueue = 0;
    uint64_t queued_bytes_before_enqueue = 0;
    uint64_t active_workers = 0;
};

struct ExpertShadowFirstUseInput {
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

struct ExpertShadowIssueInput {
    std::vector<uint64_t> task_ids;
    uint64_t issue_id = 0;
    uint64_t issue_task_count = 0;
    uint64_t issue_ts_ns = 0;
    uint64_t returned_ts_ns = 0;
    size_t issued_nbytes = 0;
};

struct ExpertShadowTaskObservation {
    uint64_t task_id = 0;
    uint64_t issue_id = 0;
    uint64_t issue_task_count = 0;
    uint64_t step = 0;
    int layer = -1;
    int expert = -1;
    int phase = 0;
    ExpertTensorStage stage = ExpertTensorStage::Unknown;
    std::string tensor;
    uintptr_t addr = 0;
    size_t nbytes = 0;
    size_t issued_nbytes = 0;
    uint64_t prediction_ts_ns = 0;
    uint64_t enqueued_ts_ns = 0;
    uint64_t dequeued_ts_ns = 0;
    uint64_t issue_ts_ns = 0;
    uint64_t returned_ts_ns = 0;
    uint64_t first_use_ts_ns = 0;
    uint64_t queue_depth_before_enqueue = 0;
    uint64_t queued_bytes_before_enqueue = 0;
    uint64_t active_workers = 0;
    uint64_t actual_queue_wait_ns = 0;
    uint64_t actual_pre_issue_overhead_ns = 0;
    uint64_t actual_hint_syscall_service_ns = 0;
    uint64_t actual_worker_occupied_ns = 0;
    int64_t actual_issue_slack_ns = 0;
    int64_t actual_return_slack_ns = 0;
    // Deprecated M4A compatibility alias for actual_worker_occupied_ns.
    uint64_t actual_worker_issue_ns = 0;
    bool has_actual_queue_wait = false;
    bool has_actual_pre_issue_overhead = false;
    bool has_actual_hint_syscall_service = false;
    bool has_actual_worker_occupied = false;
    bool has_actual_issue_slack = false;
    bool has_actual_return_slack = false;
    bool has_actual_worker_issue = false;
    bool coalesced = false;
    bool finalized = false;
    bool causality_error = false;
    std::string unavailable_reason;
    std::vector<ExpertShadowPrediction> predictions;
};

struct ExpertShadowCalibrationBucket {
    uint64_t total = 0;
    uint64_t on_time = 0;
};

struct ExpertShadowErrorAggregate {
    uint64_t count = 0;
    uint64_t absolute_error_sum_ns = 0;
    int64_t signed_error_sum_ns = 0;
    uint64_t true_positive = 0;
    uint64_t true_negative = 0;
    uint64_t false_positive = 0;
    uint64_t false_negative = 0;
    uint64_t warmup = 0;
    uint64_t fallback = 0;
    uint64_t clipped = 0;
    uint64_t mature_exact = 0;
    std::array<ExpertShadowCalibrationBucket, 10> calibration{};
    std::array<uint64_t, 7> absolute_error_histogram{};
};

struct ExpertShadowTargetAggregate {
    uint64_t count = 0;
    uint64_t unavailable = 0;
    uint64_t absolute_error_sum_ns = 0;
    int64_t signed_error_sum_ns = 0;
    uint64_t true_positive = 0;
    uint64_t true_negative = 0;
    uint64_t false_positive = 0;
    uint64_t false_negative = 0;
    uint64_t warmup = 0;
    uint64_t fallback = 0;
    uint64_t mature_exact = 0;
    std::array<ExpertShadowCalibrationBucket, 10> calibration{};
};

struct ExpertShadowCandidateSummary {
    ExpertShadowGrouping grouping = ExpertShadowGrouping::PhaseLayer;
    ExpertShadowEstimatorKind estimator = ExpertShadowEstimatorKind::Ewma;
    ExpertShadowQueueModel queue_model = ExpertShadowQueueModel::QueueDepthWorkerEwma;
    ExpertShadowCalibrationModel calibration_model = ExpertShadowCalibrationModel::Raw;
    uint64_t eligible = 0;
    uint64_t unavailable = 0;
    ExpertShadowErrorAggregate overall;
    std::array<ExpertShadowErrorAggregate, 3> by_phase{};
    std::array<ExpertShadowErrorAggregate, 3> by_stage{};
    ExpertShadowTargetAggregate issue_overall;
    ExpertShadowTargetAggregate return_overall;
    std::array<ExpertShadowTargetAggregate, 3> issue_by_phase{};
    std::array<ExpertShadowTargetAggregate, 3> return_by_phase{};
    std::array<ExpertShadowTargetAggregate, 3> issue_by_stage{};
    std::array<ExpertShadowTargetAggregate, 3> return_by_stage{};
};

struct ExpertShadowDurationAggregate {
    uint64_t count = 0;
    uint64_t absolute_error_sum_ns = 0;
    int64_t signed_error_sum_ns = 0;
    uint64_t warmup = 0;
    uint64_t fallback = 0;
};

struct ExpertShadowWorkerBucketSummary {
    uint64_t count = 0;
    uint64_t window_count = 0;
    double ewma_ns = 0.0;
};

struct ExpertShadowSummary {
    ExpertShadowConfig config;
    uint64_t eligible_tasks = 0;
    uint64_t predicted_tasks = 0;
    uint64_t unavailable_tasks = 0;
    uint64_t finalized_tasks = 0;
    uint64_t expired_tasks = 0;
    uint64_t capacity_expired_tasks = 0;
    uint64_t pending_tasks = 0;
    uint64_t peak_live_tasks = 0;
    uint64_t duplicate_task_ids = 0;
    uint64_t logical_first_uses = 0;
    uint64_t unmatched_first_uses = 0;
    uint64_t ambiguous_first_uses = 0;
    uint64_t duplicate_first_uses = 0;
    uint64_t stage_mismatch_tasks = 0;
    uint64_t address_mismatch_tasks = 0;
    uint64_t first_use_key_capacity_skips = 0;
    uint64_t causality_errors = 0;
    uint64_t issue_groups_observed = 0;
    uint64_t worker_duration_observations = 0;
    uint64_t estimator_cells = 0;
    uint64_t estimator_capacity_skips = 0;
    uint64_t residual_cells = 0;
    uint64_t residual_capacity_skips = 0;
    uint64_t expired_without_issue = 0;
    uint64_t expired_without_first_use = 0;
    std::array<uint64_t, 3> finalized_by_phase{};
    std::array<uint64_t, 3> finalized_by_stage{};
    std::vector<ExpertShadowCandidateSummary> candidates;
    std::array<ExpertShadowDurationAggregate, 2> queue_models{};
    ExpertShadowDurationAggregate pre_issue_model;
    ExpertShadowDurationAggregate syscall_service_model;
    ExpertShadowDurationAggregate worker_occupied_model;
    // Deprecated M4A compatibility aggregate for worker_occupied_model.
    ExpertShadowDurationAggregate worker_model;
    std::array<ExpertShadowWorkerBucketSummary, 6> pre_issue_buckets{};
    ExpertShadowWorkerBucketSummary pre_issue_global;
    std::array<ExpertShadowWorkerBucketSummary, 6> syscall_service_buckets{};
    ExpertShadowWorkerBucketSummary syscall_service_global;
    std::array<ExpertShadowWorkerBucketSummary, 6> worker_buckets{};
    ExpertShadowWorkerBucketSummary worker_global;
    uint64_t throughput_sample_count = 0;
    double throughput_ewma_bytes_per_ns = 0.0;
};

class ExpertShadowSlack {
public:
    explicit ExpertShadowSlack(ExpertShadowConfig config = {});

    bool register_task(ExpertShadowTaskInput input);
    void observe_dequeue(uint64_t task_id, uint64_t dequeued_ts_ns);
    std::vector<ExpertShadowTaskObservation> observe_issue_group(ExpertShadowIssueInput input);
    std::vector<ExpertShadowTaskObservation> observe_first_use(ExpertShadowFirstUseInput input);
    void expire_task(uint64_t task_id, const char * reason);
    ExpertShadowSummary summary();
    const ExpertShadowConfig & config() const { return config_; }

private:
    struct SampleCell {
        uint64_t count = 0;
        double ewma = 0.0;
        std::deque<uint64_t> window;
    };

    struct SignedSampleCell {
        uint64_t count = 0;
        std::deque<int64_t> window;
    };

    struct Estimate {
        uint64_t value_ns = 0;
        uint64_t exact_sample_count = 0;
        uint64_t effective_sample_count = 0;
        ExpertShadowFallbackLevel fallback = ExpertShadowFallbackLevel::StaticDefault;
        bool warmup = true;
        bool clipped_low = false;
        bool clipped_high = false;
    };

    struct SignedEstimate {
        int64_t value_ns = 0;
        uint64_t exact_sample_count = 0;
        uint64_t effective_sample_count = 0;
        ExpertShadowFallbackLevel fallback = ExpertShadowFallbackLevel::StaticDefault;
        bool warmup = true;
    };

    struct PendingTask {
        ExpertShadowTaskObservation observation;
        bool issue_seen = false;
        bool first_use_seen = false;
    };

    static std::string semantic_key(
            uint64_t step, int layer, int expert, const std::string & tensor);
    static bool ranges_overlap(uintptr_t a_addr, size_t a_size, uintptr_t b_addr, size_t b_size);
    static size_t phase_index(int phase);
    static size_t stage_index(ExpertTensorStage stage);
    static size_t worker_bucket(size_t nbytes);
    static int calibration_bucket(int64_t slack_ns);
    static uint64_t saturated_add(uint64_t a, uint64_t b);
    static uint64_t saturated_mul_div(uint64_t a, uint64_t b, uint64_t divisor);
    static int64_t saturated_signed_difference(uint64_t a, uint64_t b);
    static int64_t saturated_signed_add(int64_t a, int64_t b);
    static uint64_t saturated_adjust_u64(uint64_t value, int64_t adjustment);

    SampleCell * get_or_create_cell_unlocked(const std::string & key);
    const SampleCell * find_cell_unlocked(const std::string & key) const;
    void observe_cell_unlocked(const std::string & key, uint64_t value_ns);
    uint64_t cell_value_unlocked(const SampleCell & cell, ExpertShadowEstimatorKind kind) const;
    Estimate estimate_horizon_unlocked(
            ExpertShadowGrouping grouping,
            ExpertShadowEstimatorKind kind,
            int phase,
            int layer,
            ExpertTensorStage stage) const;
    Estimate estimate_worker_unlocked(size_t nbytes) const;
    Estimate estimate_duration_unlocked(
            const std::array<SampleCell, 6> & cells,
            const SampleCell & global,
            size_t nbytes,
            uint64_t default_ns,
            uint64_t minimum_ns,
            uint64_t maximum_ns) const;
    SignedEstimate estimate_residual_unlocked(
            ExpertShadowGrouping grouping,
            ExpertShadowEstimatorKind kind,
            int phase,
            int layer,
            ExpertTensorStage stage) const;
    void observe_horizon_unlocked(const PendingTask & task);
    void observe_worker_unlocked(size_t nbytes, uint64_t duration_ns);
    void observe_duration_unlocked(
            std::array<SampleCell, 6> & cells,
            SampleCell & global,
            size_t nbytes,
            uint64_t duration_ns);
    void observe_residuals_unlocked(const PendingTask & task);
    std::vector<ExpertShadowPrediction> make_predictions_unlocked(const ExpertShadowTaskInput & input) const;
    std::vector<ExpertShadowTaskObservation> finalize_ready_unlocked(
            const std::vector<uint64_t> & task_ids);
    void update_summary_unlocked(const ExpertShadowTaskObservation & observation);
    void update_error_aggregate_unlocked(
            ExpertShadowErrorAggregate & aggregate,
            const ExpertShadowPrediction & prediction,
            const ExpertShadowTaskObservation & observation);
    void update_target_aggregate_unlocked(
            ExpertShadowTargetAggregate & aggregate,
            const ExpertShadowPrediction & prediction,
            const ExpertShadowTaskObservation & observation,
            bool return_target);
    void update_duration_aggregate_unlocked(
            ExpertShadowDurationAggregate & aggregate,
            uint64_t predicted_ns,
            uint64_t actual_ns,
            bool warmup,
            bool fallback);
    void advance_step_unlocked(uint64_t step);
    void erase_task_unlocked(uint64_t task_id, bool expired, bool capacity_expired);
    void enforce_capacity_unlocked();

    ExpertShadowConfig config_;
    mutable std::mutex mu_;
    std::unordered_map<std::string, SampleCell> horizon_cells_;
    std::unordered_map<std::string, SignedSampleCell> residual_cells_;
    std::array<SampleCell, 6> pre_issue_cells_{};
    SampleCell pre_issue_global_;
    std::array<SampleCell, 6> syscall_service_cells_{};
    SampleCell syscall_service_global_;
    std::array<SampleCell, 6> worker_cells_{};
    SampleCell worker_global_;
    uint64_t throughput_count_ = 0;
    double throughput_ewma_bytes_per_ns_ = 0.0;
    std::unordered_map<uint64_t, PendingTask> tasks_;
    std::unordered_map<std::string, std::vector<uint64_t>> tasks_by_semantic_key_;
    std::unordered_set<std::string> observed_first_use_keys_;
    std::deque<uint64_t> insertion_order_;
    bool has_active_step_ = false;
    uint64_t active_step_ = 0;
    ExpertShadowSummary summary_;
};
