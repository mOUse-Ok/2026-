#include "expert_shadow_slack.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

std::string key_phase_layer(int phase, int layer) {
    return "pl:" + std::to_string(phase) + ":" + std::to_string(layer);
}

std::string key_phase_stage(int phase, ExpertTensorStage stage) {
    return "ps:" + std::to_string(phase) + ":" +
           std::to_string(expert_tensor_stage_index(stage));
}

std::string key_phase_layer_stage(int phase, int layer, ExpertTensorStage stage) {
    return "pls:" + std::to_string(phase) + ":" + std::to_string(layer) + ":" +
           std::to_string(expert_tensor_stage_index(stage));
}

std::string key_phase(int phase) {
    return "p:" + std::to_string(phase);
}

std::string key_global(int phase) {
    return "g:" + std::to_string(phase);
}

uint64_t clamp_u64(uint64_t value, uint64_t minimum, uint64_t maximum, bool & low, bool & high) {
    if (value < minimum) {
        low = true;
        return minimum;
    }
    if (value > maximum) {
        high = true;
        return maximum;
    }
    return value;
}

uint64_t round_to_u64(long double value) {
    if (!std::isfinite(value) || value <= 0.0L) {
        return 0;
    }
    const long double maximum =
            (long double) std::numeric_limits<uint64_t>::max();
    if (value >= maximum) {
        return std::numeric_limits<uint64_t>::max();
    }
    return (uint64_t) std::floor(value + 0.5L);
}

int64_t saturated_i64_sub(int64_t a, int64_t b) {
    if (b > 0 && a < std::numeric_limits<int64_t>::min() + b) {
        return std::numeric_limits<int64_t>::min();
    }
    if (b < 0 && a > std::numeric_limits<int64_t>::max() + b) {
        return std::numeric_limits<int64_t>::max();
    }
    return a - b;
}

} // namespace

const char * expert_shadow_estimator_name(ExpertShadowEstimatorKind kind) {
    switch (kind) {
        case ExpertShadowEstimatorKind::Ewma:   return "ewma";
        case ExpertShadowEstimatorKind::Median: return "median";
        case ExpertShadowEstimatorKind::P25:    return "p25";
    }
    return "ewma";
}

const char * expert_shadow_grouping_name(ExpertShadowGrouping grouping) {
    switch (grouping) {
        case ExpertShadowGrouping::PhaseLayer:      return "phase_layer";
        case ExpertShadowGrouping::PhaseStage:      return "phase_stage";
        case ExpertShadowGrouping::PhaseLayerStage: return "phase_layer_stage";
    }
    return "phase_layer";
}

const char * expert_shadow_queue_model_name(ExpertShadowQueueModel model) {
    switch (model) {
        case ExpertShadowQueueModel::QueueDepthWorkerEwma:
            return "queue_depth_worker_ewma";
        case ExpertShadowQueueModel::QueuedBytesIssueThroughput:
            return "queued_bytes_issue_throughput";
    }
    return "queue_depth_worker_ewma";
}

const char * expert_shadow_calibration_model_name(ExpertShadowCalibrationModel model) {
    switch (model) {
        case ExpertShadowCalibrationModel::Raw:              return "raw";
        case ExpertShadowCalibrationModel::ResidualQuantile: return "residual_quantile";
    }
    return "raw";
}

const char * expert_shadow_fallback_name(ExpertShadowFallbackLevel level) {
    switch (level) {
        case ExpertShadowFallbackLevel::Exact:         return "exact";
        case ExpertShadowFallbackLevel::PhaseStage:    return "phase_stage";
        case ExpertShadowFallbackLevel::Phase:         return "phase";
        case ExpertShadowFallbackLevel::Global:        return "global";
        case ExpertShadowFallbackLevel::StaticDefault: return "static_default";
    }
    return "static_default";
}

ExpertShadowSlack::ExpertShadowSlack(ExpertShadowConfig config) : config_(config) {
    config_.window_capacity = std::max<size_t>(1, config_.window_capacity);
    config_.min_samples = std::max<uint64_t>(1, config_.min_samples);
    config_.ewma_alpha = std::max(0.000001, std::min(1.0, config_.ewma_alpha));
    config_.residual_quantile = std::max(0.0, std::min(1.0, config_.residual_quantile));
    config_.horizon_min_ns = std::min(config_.horizon_min_ns, config_.horizon_max_ns);
    config_.horizon_default_ns = std::max(
            config_.horizon_min_ns, std::min(config_.horizon_default_ns, config_.horizon_max_ns));
    config_.worker_min_ns = std::min(config_.worker_min_ns, config_.worker_max_ns);
    config_.worker_default_ns = std::max(
            config_.worker_min_ns, std::min(config_.worker_default_ns, config_.worker_max_ns));
    config_.pre_issue_min_ns = std::min(config_.pre_issue_min_ns, config_.pre_issue_max_ns);
    config_.pre_issue_default_ns = std::max(
            config_.pre_issue_min_ns,
            std::min(config_.pre_issue_default_ns, config_.pre_issue_max_ns));
    config_.syscall_service_min_ns = std::min(
            config_.syscall_service_min_ns, config_.syscall_service_max_ns);
    config_.syscall_service_default_ns = std::max(
            config_.syscall_service_min_ns,
            std::min(config_.syscall_service_default_ns, config_.syscall_service_max_ns));
    config_.throughput_default_bytes_per_ns = std::max(0.000001, config_.throughput_default_bytes_per_ns);
    config_.max_pending_tasks = std::max<size_t>(1, config_.max_pending_tasks);
    config_.max_first_use_keys = std::max<size_t>(1, config_.max_first_use_keys);
    config_.max_estimator_cells = std::max<size_t>(8, config_.max_estimator_cells);
    config_.max_residual_cells = std::max<size_t>(8, config_.max_residual_cells);
    summary_.config = config_;

    static const ExpertShadowGrouping groupings[] = {
        ExpertShadowGrouping::PhaseLayer,
        ExpertShadowGrouping::PhaseStage,
        ExpertShadowGrouping::PhaseLayerStage,
    };
    static const ExpertShadowEstimatorKind estimators[] = {
        ExpertShadowEstimatorKind::Ewma,
        ExpertShadowEstimatorKind::Median,
        ExpertShadowEstimatorKind::P25,
    };
    static const ExpertShadowQueueModel queues[] = {
        ExpertShadowQueueModel::QueueDepthWorkerEwma,
        ExpertShadowQueueModel::QueuedBytesIssueThroughput,
    };
    static const ExpertShadowCalibrationModel calibration_models[] = {
        ExpertShadowCalibrationModel::Raw,
        ExpertShadowCalibrationModel::ResidualQuantile,
    };
    summary_.candidates.reserve(36);
    for (ExpertShadowGrouping grouping : groupings) {
        for (ExpertShadowEstimatorKind estimator : estimators) {
            for (ExpertShadowQueueModel queue : queues) {
                for (ExpertShadowCalibrationModel calibration : calibration_models) {
                    ExpertShadowCandidateSummary candidate;
                    candidate.grouping = grouping;
                    candidate.estimator = estimator;
                    candidate.queue_model = queue;
                    candidate.calibration_model = calibration;
                    summary_.candidates.push_back(candidate);
                }
            }
        }
    }
}

std::string ExpertShadowSlack::semantic_key(
        uint64_t step, int layer, int expert, const std::string & tensor) {
    return std::to_string(step) + ":" + std::to_string(layer) + ":" +
           std::to_string(expert) + ":" + tensor;
}

bool ExpertShadowSlack::ranges_overlap(
        uintptr_t a_addr, size_t a_size, uintptr_t b_addr, size_t b_size) {
    if (a_addr == 0 || b_addr == 0 || a_size == 0 || b_size == 0) {
        return false;
    }
    const uintptr_t maximum = std::numeric_limits<uintptr_t>::max();
    const uintptr_t a_end = a_size > maximum - a_addr ? maximum : a_addr + a_size;
    const uintptr_t b_end = b_size > maximum - b_addr ? maximum : b_addr + b_size;
    return a_addr < b_end && b_addr < a_end;
}

size_t ExpertShadowSlack::phase_index(int phase) {
    return phase >= 0 && phase <= 2 ? (size_t) phase : 0;
}

size_t ExpertShadowSlack::stage_index(ExpertTensorStage stage) {
    return expert_tensor_stage_index(stage);
}

size_t ExpertShadowSlack::worker_bucket(size_t nbytes) {
    static const size_t limits[] = {
        64 * 1024,
        256 * 1024,
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
    };
    for (size_t i = 0; i < sizeof(limits) / sizeof(limits[0]); ++i) {
        if (nbytes <= limits[i]) {
            return i;
        }
    }
    return 5;
}

int ExpertShadowSlack::calibration_bucket(int64_t slack_ns) {
    static const int64_t ms = 1'000'000;
    if (slack_ns < -5 * ms) {
        return 0;
    }
    if (slack_ns < -2 * ms) {
        return 1;
    }
    if (slack_ns < -1 * ms) {
        return 2;
    }
    if (slack_ns < -ms / 2) {
        return 3;
    }
    if (slack_ns <= 0) {
        return 4;
    }
    if (slack_ns <= ms / 2) {
        return 5;
    }
    if (slack_ns <= ms) {
        return 6;
    }
    if (slack_ns <= 2 * ms) {
        return 7;
    }
    if (slack_ns <= 5 * ms) {
        return 8;
    }
    return 9;
}

uint64_t ExpertShadowSlack::saturated_add(uint64_t a, uint64_t b) {
    return b > std::numeric_limits<uint64_t>::max() - a ?
            std::numeric_limits<uint64_t>::max() : a + b;
}

uint64_t ExpertShadowSlack::saturated_mul_div(uint64_t a, uint64_t b, uint64_t divisor) {
    if (divisor == 0) {
        return std::numeric_limits<uint64_t>::max();
    }
    const __uint128_t value = (__uint128_t) a * (__uint128_t) b / divisor;
    return value > std::numeric_limits<uint64_t>::max() ?
            std::numeric_limits<uint64_t>::max() : (uint64_t) value;
}

int64_t ExpertShadowSlack::saturated_signed_difference(uint64_t a, uint64_t b) {
    if (a >= b) {
        const uint64_t difference = a - b;
        return difference > (uint64_t) std::numeric_limits<int64_t>::max() ?
                std::numeric_limits<int64_t>::max() : (int64_t) difference;
    }
    const uint64_t difference = b - a;
    const uint64_t negative_limit = (uint64_t) std::numeric_limits<int64_t>::max() + 1ull;
    if (difference >= negative_limit) {
        return std::numeric_limits<int64_t>::min();
    }
    return -(int64_t) difference;
}

int64_t ExpertShadowSlack::saturated_signed_add(int64_t a, int64_t b) {
    if (b > 0 && a > std::numeric_limits<int64_t>::max() - b) {
        return std::numeric_limits<int64_t>::max();
    }
    if (b < 0 && a < std::numeric_limits<int64_t>::min() - b) {
        return std::numeric_limits<int64_t>::min();
    }
    return a + b;
}

uint64_t ExpertShadowSlack::saturated_adjust_u64(uint64_t value, int64_t adjustment) {
    if (adjustment >= 0) {
        return saturated_add(value, (uint64_t) adjustment);
    }
    const uint64_t magnitude = adjustment == std::numeric_limits<int64_t>::min() ?
            (uint64_t) std::numeric_limits<int64_t>::max() + 1ull :
            (uint64_t) -adjustment;
    return magnitude > value ? 0 : value - magnitude;
}

ExpertShadowSlack::SampleCell * ExpertShadowSlack::get_or_create_cell_unlocked(
        const std::string & key) {
    auto found = horizon_cells_.find(key);
    if (found != horizon_cells_.end()) {
        return &found->second;
    }
    if (horizon_cells_.size() >= config_.max_estimator_cells) {
        summary_.estimator_capacity_skips++;
        return nullptr;
    }
    return &horizon_cells_.emplace(key, SampleCell{}).first->second;
}

const ExpertShadowSlack::SampleCell * ExpertShadowSlack::find_cell_unlocked(
        const std::string & key) const {
    const auto found = horizon_cells_.find(key);
    return found == horizon_cells_.end() ? nullptr : &found->second;
}

void ExpertShadowSlack::observe_cell_unlocked(const std::string & key, uint64_t value_ns) {
    SampleCell * cell = get_or_create_cell_unlocked(key);
    if (!cell) {
        return;
    }
    cell->count++;
    const double value = (double) value_ns;
    cell->ewma = cell->count == 1 ? value :
            cell->ewma * (1.0 - config_.ewma_alpha) + value * config_.ewma_alpha;
    cell->window.push_back(value_ns);
    while (cell->window.size() > config_.window_capacity) {
        cell->window.pop_front();
    }
}

uint64_t ExpertShadowSlack::cell_value_unlocked(
        const SampleCell & cell, ExpertShadowEstimatorKind kind) const {
    if (cell.count == 0) {
        return 0;
    }
    if (kind == ExpertShadowEstimatorKind::Ewma) {
        return round_to_u64((long double) cell.ewma);
    }
    std::vector<uint64_t> ordered(cell.window.begin(), cell.window.end());
    std::sort(ordered.begin(), ordered.end());
    if (ordered.empty()) {
        return 0;
    }
    const double q = kind == ExpertShadowEstimatorKind::Median ? 0.5 : 0.25;
    const double position = (double) (ordered.size() - 1) * q;
    const size_t lower = (size_t) position;
    const size_t upper = std::min(lower + 1, ordered.size() - 1);
    const double fraction = position - (double) lower;
    const double value = (double) ordered[lower] * (1.0 - fraction) +
                         (double) ordered[upper] * fraction;
    return round_to_u64((long double) value);
}

ExpertShadowSlack::Estimate ExpertShadowSlack::estimate_horizon_unlocked(
        ExpertShadowGrouping grouping,
        ExpertShadowEstimatorKind kind,
        int phase,
        int layer,
        ExpertTensorStage stage) const {
    Estimate result;
    std::string exact_key;
    if (grouping == ExpertShadowGrouping::PhaseLayer) {
        exact_key = key_phase_layer(phase, layer);
    } else if (grouping == ExpertShadowGrouping::PhaseStage) {
        exact_key = key_phase_stage(phase, stage);
    } else {
        exact_key = key_phase_layer_stage(phase, layer, stage);
    }
    const SampleCell * exact = find_cell_unlocked(exact_key);
    result.exact_sample_count = exact ? exact->count : 0;
    result.warmup = result.exact_sample_count < config_.min_samples;

    const SampleCell * selected = nullptr;
    if (exact && exact->count >= config_.min_samples) {
        selected = exact;
        result.fallback = ExpertShadowFallbackLevel::Exact;
    }
    if (!selected && grouping == ExpertShadowGrouping::PhaseLayerStage) {
        const SampleCell * phase_stage = find_cell_unlocked(key_phase_stage(phase, stage));
        if (phase_stage && phase_stage->count >= config_.min_samples) {
            selected = phase_stage;
            result.fallback = ExpertShadowFallbackLevel::PhaseStage;
        }
    }
    if (!selected) {
        const SampleCell * phase_cell = find_cell_unlocked(key_phase(phase));
        if (phase_cell && phase_cell->count >= config_.min_samples) {
            selected = phase_cell;
            result.fallback = ExpertShadowFallbackLevel::Phase;
        }
    }
    if (!selected) {
        const SampleCell * global = find_cell_unlocked(key_global(phase));
        if (global && global->count >= config_.min_samples) {
            selected = global;
            result.fallback = ExpertShadowFallbackLevel::Global;
        }
    }

    uint64_t value = config_.horizon_default_ns;
    if (selected) {
        value = cell_value_unlocked(*selected, kind);
        result.effective_sample_count = selected->count;
    } else {
        result.fallback = ExpertShadowFallbackLevel::StaticDefault;
    }
    result.value_ns = clamp_u64(
            value, config_.horizon_min_ns, config_.horizon_max_ns,
            result.clipped_low, result.clipped_high);
    return result;
}

ExpertShadowSlack::Estimate ExpertShadowSlack::estimate_worker_unlocked(size_t nbytes) const {
    return estimate_duration_unlocked(
            worker_cells_, worker_global_, nbytes,
            config_.worker_default_ns, config_.worker_min_ns, config_.worker_max_ns);
}

ExpertShadowSlack::Estimate ExpertShadowSlack::estimate_duration_unlocked(
        const std::array<SampleCell, 6> & cells,
        const SampleCell & global,
        size_t nbytes,
        uint64_t default_ns,
        uint64_t minimum_ns,
        uint64_t maximum_ns) const {
    Estimate result;
    const SampleCell & exact = cells[worker_bucket(nbytes)];
    result.exact_sample_count = exact.count;
    result.warmup = exact.count < config_.min_samples;
    const SampleCell * selected = nullptr;
    if (exact.count >= config_.min_samples) {
        selected = &exact;
        result.fallback = ExpertShadowFallbackLevel::Exact;
    } else if (global.count >= config_.min_samples) {
        selected = &global;
        result.fallback = ExpertShadowFallbackLevel::Global;
    }
    uint64_t value = default_ns;
    if (selected) {
        value = cell_value_unlocked(*selected, ExpertShadowEstimatorKind::Ewma);
        result.effective_sample_count = selected->count;
    } else {
        result.fallback = ExpertShadowFallbackLevel::StaticDefault;
    }
    result.value_ns = clamp_u64(
            value, minimum_ns, maximum_ns,
            result.clipped_low, result.clipped_high);
    return result;
}

ExpertShadowSlack::SignedEstimate ExpertShadowSlack::estimate_residual_unlocked(
        ExpertShadowGrouping grouping,
        ExpertShadowEstimatorKind kind,
        int phase,
        int layer,
        ExpertTensorStage stage) const {
    SignedEstimate result;
    const std::string prefix = "r:" + std::string(expert_shadow_grouping_name(grouping)) +
            ":" + expert_shadow_estimator_name(kind) + ":";
    std::string exact_key;
    if (grouping == ExpertShadowGrouping::PhaseLayer) {
        exact_key = key_phase_layer(phase, layer);
    } else if (grouping == ExpertShadowGrouping::PhaseStage) {
        exact_key = key_phase_stage(phase, stage);
    } else {
        exact_key = key_phase_layer_stage(phase, layer, stage);
    }
    auto find = [&](const std::string & key) -> const SignedSampleCell * {
        const auto found = residual_cells_.find(prefix + key);
        return found == residual_cells_.end() ? nullptr : &found->second;
    };
    const SignedSampleCell * exact = find(exact_key);
    result.exact_sample_count = exact ? exact->count : 0;
    result.warmup = result.exact_sample_count < config_.min_samples;
    const SignedSampleCell * selected = nullptr;
    if (exact && exact->count >= config_.min_samples) {
        selected = exact;
        result.fallback = ExpertShadowFallbackLevel::Exact;
    }
    if (!selected && grouping == ExpertShadowGrouping::PhaseLayerStage) {
        const SignedSampleCell * phase_stage = find(key_phase_stage(phase, stage));
        if (phase_stage && phase_stage->count >= config_.min_samples) {
            selected = phase_stage;
            result.fallback = ExpertShadowFallbackLevel::PhaseStage;
        }
    }
    if (!selected) {
        const SignedSampleCell * phase_cell = find(key_phase(phase));
        if (phase_cell && phase_cell->count >= config_.min_samples) {
            selected = phase_cell;
            result.fallback = ExpertShadowFallbackLevel::Phase;
        }
    }
    if (!selected) {
        const SignedSampleCell * global = find(key_global(phase));
        if (global && global->count >= config_.min_samples) {
            selected = global;
            result.fallback = ExpertShadowFallbackLevel::Global;
        }
    }
    if (!selected || selected->window.empty()) {
        result.fallback = ExpertShadowFallbackLevel::StaticDefault;
        return result;
    }
    std::vector<int64_t> ordered(selected->window.begin(), selected->window.end());
    std::sort(ordered.begin(), ordered.end());
    const long double position = (long double) (ordered.size() - 1) * config_.residual_quantile;
    const size_t lower = (size_t) std::floor(position);
    const size_t upper = std::min(lower + 1, ordered.size() - 1);
    const long double fraction = position - (long double) lower;
    const long double value = (long double) ordered[lower] * (1.0L - fraction) +
            (long double) ordered[upper] * fraction;
    if (value >= (long double) std::numeric_limits<int64_t>::max()) {
        result.value_ns = std::numeric_limits<int64_t>::max();
    } else if (value <= (long double) std::numeric_limits<int64_t>::min()) {
        result.value_ns = std::numeric_limits<int64_t>::min();
    } else {
        result.value_ns = (int64_t) std::llround(value);
    }
    result.effective_sample_count = selected->count;
    return result;
}

void ExpertShadowSlack::observe_horizon_unlocked(const PendingTask & task) {
    const ExpertShadowTaskObservation & observation = task.observation;
    if (observation.first_use_ts_ns < observation.prediction_ts_ns) {
        return;
    }
    const uint64_t horizon = observation.first_use_ts_ns - observation.prediction_ts_ns;
    observe_cell_unlocked(key_phase_layer(observation.phase, observation.layer), horizon);
    observe_cell_unlocked(key_phase_stage(observation.phase, observation.stage), horizon);
    observe_cell_unlocked(
            key_phase_layer_stage(observation.phase, observation.layer, observation.stage), horizon);
    observe_cell_unlocked(key_phase(observation.phase), horizon);
    observe_cell_unlocked(key_global(observation.phase), horizon);
}

void ExpertShadowSlack::observe_worker_unlocked(size_t nbytes, uint64_t duration_ns) {
    observe_duration_unlocked(worker_cells_, worker_global_, nbytes, duration_ns);
    summary_.worker_duration_observations++;
}

void ExpertShadowSlack::observe_duration_unlocked(
        std::array<SampleCell, 6> & cells,
        SampleCell & global,
        size_t nbytes,
        uint64_t duration_ns) {
    auto observe = [&](SampleCell & cell) {
        cell.count++;
        const double value = (double) duration_ns;
        cell.ewma = cell.count == 1 ? value :
                cell.ewma * (1.0 - config_.ewma_alpha) + value * config_.ewma_alpha;
        cell.window.push_back(duration_ns);
        while (cell.window.size() > config_.window_capacity) {
            cell.window.pop_front();
        }
    };
    observe(cells[worker_bucket(nbytes)]);
    observe(global);
}

void ExpertShadowSlack::observe_residuals_unlocked(const PendingTask & task) {
    const ExpertShadowTaskObservation & observation = task.observation;
    if (observation.first_use_ts_ns < observation.prediction_ts_ns) {
        return;
    }
    const uint64_t actual_horizon = observation.first_use_ts_ns - observation.prediction_ts_ns;
    for (const ExpertShadowPrediction & prediction : observation.predictions) {
        if (prediction.calibration_model != ExpertShadowCalibrationModel::Raw ||
                prediction.queue_model != ExpertShadowQueueModel::QueueDepthWorkerEwma ||
                prediction.estimator_warmup ||
                prediction.fallback_level != ExpertShadowFallbackLevel::Exact) {
            continue;
        }
        const int64_t residual = saturated_signed_difference(
                actual_horizon, prediction.raw_predicted_first_use_horizon_ns);
        const std::string prefix = "r:" +
                std::string(expert_shadow_grouping_name(prediction.grouping)) + ":" +
                expert_shadow_estimator_name(prediction.estimator) + ":";
        auto observe = [&](const std::string & key) {
            auto found = residual_cells_.find(prefix + key);
            if (found == residual_cells_.end()) {
                if (residual_cells_.size() >= config_.max_residual_cells) {
                    summary_.residual_capacity_skips++;
                    return;
                }
                found = residual_cells_.emplace(prefix + key, SignedSampleCell{}).first;
            }
            SignedSampleCell & cell = found->second;
            cell.count++;
            cell.window.push_back(residual);
            while (cell.window.size() > config_.window_capacity) {
                cell.window.pop_front();
            }
        };
        if (prediction.grouping == ExpertShadowGrouping::PhaseLayer) {
            observe(key_phase_layer(observation.phase, observation.layer));
        } else if (prediction.grouping == ExpertShadowGrouping::PhaseStage) {
            observe(key_phase_stage(observation.phase, observation.stage));
        } else {
            observe(key_phase_layer_stage(
                    observation.phase, observation.layer, observation.stage));
            observe(key_phase_stage(observation.phase, observation.stage));
        }
        observe(key_phase(observation.phase));
        observe(key_global(observation.phase));
    }
}

std::vector<ExpertShadowPrediction> ExpertShadowSlack::make_predictions_unlocked(
        const ExpertShadowTaskInput & input) const {
    static const ExpertShadowGrouping groupings[] = {
        ExpertShadowGrouping::PhaseLayer,
        ExpertShadowGrouping::PhaseStage,
        ExpertShadowGrouping::PhaseLayerStage,
    };
    static const ExpertShadowEstimatorKind estimators[] = {
        ExpertShadowEstimatorKind::Ewma,
        ExpertShadowEstimatorKind::Median,
        ExpertShadowEstimatorKind::P25,
    };
    static const ExpertShadowQueueModel queues[] = {
        ExpertShadowQueueModel::QueueDepthWorkerEwma,
        ExpertShadowQueueModel::QueuedBytesIssueThroughput,
    };
    static const ExpertShadowCalibrationModel calibration_models[] = {
        ExpertShadowCalibrationModel::Raw,
        ExpertShadowCalibrationModel::ResidualQuantile,
    };

    // Queue A needs the service time of tasks already ahead of this task.  That
    // service time is DEQUEUE -> RETURN (worker occupied), while the current
    // task's aligned Issue target subtracts only DEQUEUE -> ISSUE.
    const Estimate occupied = estimate_worker_unlocked(input.nbytes);
    const Estimate pre_issue = estimate_duration_unlocked(
            pre_issue_cells_, pre_issue_global_, input.nbytes,
            config_.pre_issue_default_ns, config_.pre_issue_min_ns,
            config_.pre_issue_max_ns);
    const Estimate syscall_service = estimate_duration_unlocked(
            syscall_service_cells_, syscall_service_global_, input.nbytes,
            config_.syscall_service_default_ns, config_.syscall_service_min_ns,
            config_.syscall_service_max_ns);
    const bool throughput_mature = throughput_count_ >= config_.min_samples &&
            throughput_ewma_bytes_per_ns_ > 0.0;
    const double throughput = throughput_mature ?
            throughput_ewma_bytes_per_ns_ : config_.throughput_default_bytes_per_ns;
    uint64_t queue_b = 0;
    if (throughput > 0.0) {
        const long double value = (long double) input.queued_bytes_before_enqueue / throughput;
        queue_b = round_to_u64(value);
    }
    const uint64_t queue_a = input.active_workers == 0 ? 0 : saturated_mul_div(
            input.queue_depth_before_enqueue, occupied.value_ns, input.active_workers);

    auto subtract_duration = [](int64_t value, uint64_t duration) {
        const int64_t negative = duration > (uint64_t) std::numeric_limits<int64_t>::max() ?
                std::numeric_limits<int64_t>::min() : -(int64_t) duration;
        return saturated_signed_add(value, negative);
    };

    std::vector<ExpertShadowPrediction> predictions;
    predictions.reserve(36);
    for (ExpertShadowGrouping grouping : groupings) {
        for (ExpertShadowEstimatorKind estimator : estimators) {
            const Estimate raw_horizon = estimate_horizon_unlocked(
                    grouping, estimator, input.phase, input.layer, input.stage);
            const SignedEstimate residual = estimate_residual_unlocked(
                    grouping, estimator, input.phase, input.layer, input.stage);
            for (ExpertShadowQueueModel queue : queues) {
                for (ExpertShadowCalibrationModel calibration : calibration_models) {
                    ExpertShadowPrediction prediction;
                    prediction.grouping = grouping;
                    prediction.estimator = estimator;
                    prediction.queue_model = queue;
                    prediction.calibration_model = calibration;
                    prediction.fallback_level = raw_horizon.fallback;
                    prediction.queue_fallback_level =
                            queue == ExpertShadowQueueModel::QueueDepthWorkerEwma ?
                            occupied.fallback :
                            (throughput_mature ? ExpertShadowFallbackLevel::Exact :
                             ExpertShadowFallbackLevel::StaticDefault);
                    prediction.worker_fallback_level = occupied.fallback;
                    prediction.pre_issue_fallback_level = pre_issue.fallback;
                    prediction.syscall_service_fallback_level = syscall_service.fallback;
                    prediction.residual_fallback_level = calibration ==
                            ExpertShadowCalibrationModel::ResidualQuantile ? residual.fallback :
                            ExpertShadowFallbackLevel::Exact;
                    prediction.raw_predicted_first_use_horizon_ns = raw_horizon.value_ns;
                    prediction.residual_adjustment_ns = calibration ==
                            ExpertShadowCalibrationModel::ResidualQuantile ? residual.value_ns : 0;
                    uint64_t calibrated_horizon = saturated_adjust_u64(
                            raw_horizon.value_ns, prediction.residual_adjustment_ns);
                    bool calibration_clipped_low = false;
                    bool calibration_clipped_high = false;
                    calibrated_horizon = clamp_u64(
                            calibrated_horizon, config_.horizon_min_ns,
                            config_.horizon_max_ns, calibration_clipped_low,
                            calibration_clipped_high);
                    prediction.predicted_first_use_horizon_ns = calibrated_horizon;
                    prediction.predicted_first_use_ts_ns = saturated_add(
                            input.prediction_ts_ns, calibrated_horizon);
                    prediction.predicted_queue_wait_ns =
                            queue == ExpertShadowQueueModel::QueueDepthWorkerEwma ? queue_a : queue_b;
                    prediction.predicted_pre_issue_overhead_ns = pre_issue.value_ns;
                    prediction.predicted_hint_syscall_service_ns = syscall_service.value_ns;
                    prediction.predicted_worker_occupied_ns = occupied.value_ns;
                    prediction.predicted_worker_issue_ns = occupied.value_ns;
                    int64_t issue_slack = saturated_signed_difference(
                            calibrated_horizon, prediction.predicted_queue_wait_ns);
                    issue_slack = subtract_duration(issue_slack, pre_issue.value_ns);
                    prediction.predicted_issue_slack_ns = issue_slack;
                    prediction.predicted_return_slack_ns = subtract_duration(
                            issue_slack, syscall_service.value_ns);
                    prediction.estimator_sample_count = raw_horizon.exact_sample_count;
                    prediction.estimator_effective_sample_count =
                            raw_horizon.effective_sample_count;
                    prediction.residual_sample_count = calibration ==
                            ExpertShadowCalibrationModel::ResidualQuantile ?
                            residual.exact_sample_count : 0;
                    prediction.residual_effective_sample_count = calibration ==
                            ExpertShadowCalibrationModel::ResidualQuantile ?
                            residual.effective_sample_count : 0;
                    prediction.queue_sample_count =
                            queue == ExpertShadowQueueModel::QueueDepthWorkerEwma ?
                            occupied.exact_sample_count : throughput_count_;
                    prediction.worker_sample_count = occupied.exact_sample_count;
                    prediction.pre_issue_sample_count = pre_issue.exact_sample_count;
                    prediction.syscall_service_sample_count =
                            syscall_service.exact_sample_count;
                    prediction.estimator_warmup = raw_horizon.warmup;
                    prediction.residual_warmup = calibration ==
                            ExpertShadowCalibrationModel::ResidualQuantile ? residual.warmup : false;
                    prediction.queue_warmup =
                            queue == ExpertShadowQueueModel::QueueDepthWorkerEwma ?
                            occupied.warmup : !throughput_mature;
                    prediction.worker_warmup = occupied.warmup;
                    prediction.pre_issue_warmup = pre_issue.warmup;
                    prediction.syscall_service_warmup = syscall_service.warmup;
                    prediction.prediction_available = input.active_workers > 0 &&
                            (queue == ExpertShadowQueueModel::QueueDepthWorkerEwma ||
                             throughput_count_ > 0);
                    prediction.issue_prediction_available = prediction.prediction_available;
                    prediction.return_prediction_available = prediction.prediction_available;
                    prediction.clipped_low = raw_horizon.clipped_low ||
                            occupied.clipped_low || pre_issue.clipped_low ||
                            syscall_service.clipped_low || calibration_clipped_low;
                    prediction.clipped_high = raw_horizon.clipped_high ||
                            occupied.clipped_high || pre_issue.clipped_high ||
                            syscall_service.clipped_high || calibration_clipped_high ||
                            queue_a == std::numeric_limits<uint64_t>::max() ||
                            queue_b == std::numeric_limits<uint64_t>::max();
                    predictions.push_back(prediction);
                }
            }
        }
    }
    return predictions;
}

bool ExpertShadowSlack::register_task(ExpertShadowTaskInput input) {
    if (!config_.enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    advance_step_unlocked(input.step);
    summary_.eligible_tasks++;
    if (input.task_id == 0 || input.prediction_ts_ns == 0 || input.enqueued_ts_ns == 0 ||
            input.prediction_ts_ns != input.enqueued_ts_ns) {
        summary_.causality_errors++;
        return false;
    }
    if (tasks_.find(input.task_id) != tasks_.end()) {
        summary_.duplicate_task_ids++;
        return false;
    }
    enforce_capacity_unlocked();

    PendingTask pending;
    pending.observation.task_id = input.task_id;
    pending.observation.step = input.step;
    pending.observation.layer = input.layer;
    pending.observation.expert = input.expert;
    pending.observation.phase = input.phase;
    pending.observation.stage = input.stage;
    pending.observation.tensor = input.tensor;
    pending.observation.addr = input.addr;
    pending.observation.nbytes = input.nbytes;
    pending.observation.prediction_ts_ns = input.prediction_ts_ns;
    pending.observation.enqueued_ts_ns = input.enqueued_ts_ns;
    pending.observation.queue_depth_before_enqueue = input.queue_depth_before_enqueue;
    pending.observation.queued_bytes_before_enqueue = input.queued_bytes_before_enqueue;
    pending.observation.active_workers = input.active_workers;
    pending.observation.predictions = make_predictions_unlocked(input);
    if (input.active_workers == 0) {
        pending.observation.unavailable_reason = "no_active_worker";
        summary_.unavailable_tasks++;
    }

    const std::string key = semantic_key(input.step, input.layer, input.expert, input.tensor);
    tasks_by_semantic_key_[key].push_back(input.task_id);
    tasks_.emplace(input.task_id, std::move(pending));
    insertion_order_.push_back(input.task_id);
    summary_.predicted_tasks++;
    summary_.peak_live_tasks = std::max<uint64_t>(summary_.peak_live_tasks, tasks_.size());
    return true;
}

void ExpertShadowSlack::observe_dequeue(uint64_t task_id, uint64_t dequeued_ts_ns) {
    if (!config_.enabled || task_id == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    const auto found = tasks_.find(task_id);
    if (found == tasks_.end()) {
        return;
    }
    ExpertShadowTaskObservation & observation = found->second.observation;
    if (dequeued_ts_ns < observation.enqueued_ts_ns) {
        observation.causality_error = true;
        summary_.causality_errors++;
        return;
    }
    observation.dequeued_ts_ns = dequeued_ts_ns;
    observation.actual_queue_wait_ns = dequeued_ts_ns - observation.enqueued_ts_ns;
    observation.has_actual_queue_wait = true;
}

std::vector<ExpertShadowTaskObservation> ExpertShadowSlack::observe_issue_group(
        ExpertShadowIssueInput input) {
    if (!config_.enabled) {
        return {};
    }
    std::lock_guard<std::mutex> lock(mu_);
    const bool issue_times_valid = input.issue_ts_ns != 0 &&
            input.returned_ts_ns >= input.issue_ts_ns;
    uint64_t earliest_dequeue = 0;
    std::vector<uint64_t> present;
    present.reserve(input.task_ids.size());
    for (uint64_t task_id : input.task_ids) {
        const auto found = tasks_.find(task_id);
        if (found == tasks_.end()) {
            continue;
        }
        PendingTask & task = found->second;
        ExpertShadowTaskObservation & observation = task.observation;
        observation.issue_id = input.issue_id;
        observation.issue_task_count = input.issue_task_count;
        observation.issue_ts_ns = input.issue_ts_ns;
        observation.returned_ts_ns = input.returned_ts_ns;
        observation.issued_nbytes = input.issued_nbytes;
        observation.coalesced = input.issue_task_count > 1;
        task.issue_seen = true;
        if (observation.dequeued_ts_ns != 0 &&
                (earliest_dequeue == 0 || observation.dequeued_ts_ns < earliest_dequeue)) {
            earliest_dequeue = observation.dequeued_ts_ns;
        }
        if (issue_times_valid && observation.dequeued_ts_ns != 0 &&
                input.issue_ts_ns >= observation.dequeued_ts_ns) {
            observation.actual_pre_issue_overhead_ns =
                    input.issue_ts_ns - observation.dequeued_ts_ns;
            observation.actual_hint_syscall_service_ns =
                    input.returned_ts_ns - input.issue_ts_ns;
            observation.actual_worker_occupied_ns =
                    input.returned_ts_ns - observation.dequeued_ts_ns;
            observation.actual_worker_issue_ns = observation.actual_worker_occupied_ns;
            observation.has_actual_pre_issue_overhead = true;
            observation.has_actual_hint_syscall_service = true;
            observation.has_actual_worker_occupied = true;
            observation.has_actual_worker_issue = true;
            observe_duration_unlocked(
                    pre_issue_cells_, pre_issue_global_, observation.nbytes,
                    observation.actual_pre_issue_overhead_ns);
            if (!observation.predictions.empty()) {
                const ExpertShadowPrediction & prediction = observation.predictions.front();
                update_duration_aggregate_unlocked(
                        summary_.pre_issue_model,
                        prediction.predicted_pre_issue_overhead_ns,
                        observation.actual_pre_issue_overhead_ns,
                        prediction.pre_issue_warmup,
                        prediction.pre_issue_fallback_level !=
                                ExpertShadowFallbackLevel::Exact);
            }
        } else if (observation.dequeued_ts_ns != 0) {
            observation.causality_error = true;
            summary_.causality_errors++;
        }
        present.push_back(task_id);
    }
    if (present.empty()) {
        return {};
    }
    summary_.issue_groups_observed++;
    if (issue_times_valid && earliest_dequeue != 0 && input.issue_ts_ns >= earliest_dequeue) {
        const uint64_t syscall_service = input.returned_ts_ns - input.issue_ts_ns;
        const uint64_t occupied = input.returned_ts_ns - earliest_dequeue;
        observe_duration_unlocked(
                syscall_service_cells_, syscall_service_global_, input.issued_nbytes,
                syscall_service);
        observe_worker_unlocked(input.issued_nbytes, occupied);
        if (occupied > 0 && input.issued_nbytes > 0) {
            const double throughput = (double) input.issued_nbytes / (double) occupied;
            throughput_count_++;
            throughput_ewma_bytes_per_ns_ = throughput_count_ == 1 ? throughput :
                    throughput_ewma_bytes_per_ns_ * (1.0 - config_.ewma_alpha) +
                    throughput * config_.ewma_alpha;
        }
        const auto first = tasks_.find(present.front());
        if (first != tasks_.end() && !first->second.observation.predictions.empty()) {
            const ExpertShadowPrediction & prediction =
                    first->second.observation.predictions.front();
            update_duration_aggregate_unlocked(
                    summary_.syscall_service_model,
                    prediction.predicted_hint_syscall_service_ns,
                    syscall_service, prediction.syscall_service_warmup,
                    prediction.syscall_service_fallback_level !=
                            ExpertShadowFallbackLevel::Exact);
            update_duration_aggregate_unlocked(
                    summary_.worker_occupied_model,
                    prediction.predicted_worker_occupied_ns,
                    occupied, prediction.worker_warmup,
                    prediction.worker_fallback_level != ExpertShadowFallbackLevel::Exact);
            update_duration_aggregate_unlocked(
                    summary_.worker_model,
                    prediction.predicted_worker_occupied_ns,
                    occupied, prediction.worker_warmup,
                    prediction.worker_fallback_level != ExpertShadowFallbackLevel::Exact);
        }
    }
    return finalize_ready_unlocked(present);
}

std::vector<ExpertShadowTaskObservation> ExpertShadowSlack::observe_first_use(
        ExpertShadowFirstUseInput input) {
    if (!config_.enabled) {
        return {};
    }
    std::lock_guard<std::mutex> lock(mu_);
    advance_step_unlocked(input.step);
    summary_.logical_first_uses++;
    const std::string key = semantic_key(input.step, input.layer, input.expert, input.tensor);
    const auto observed = observed_first_use_keys_.find(key);
    if (observed != observed_first_use_keys_.end()) {
        summary_.duplicate_first_uses++;
        return {};
    }
    if (observed_first_use_keys_.size() >= config_.max_first_use_keys) {
        summary_.first_use_key_capacity_skips++;
        return {};
    }
    observed_first_use_keys_.insert(key);
    const auto keyed = tasks_by_semantic_key_.find(key);
    if (keyed == tasks_by_semantic_key_.end()) {
        summary_.unmatched_first_uses++;
        return {};
    }

    std::vector<uint64_t> matched;
    for (uint64_t task_id : keyed->second) {
        const auto found = tasks_.find(task_id);
        if (found == tasks_.end()) {
            continue;
        }
        PendingTask & task = found->second;
        ExpertShadowTaskObservation & observation = task.observation;
        if (task.first_use_seen) {
            continue;
        }
        if (observation.stage != input.stage) {
            summary_.stage_mismatch_tasks++;
            continue;
        }
        if (!ranges_overlap(observation.addr, observation.nbytes, input.addr, input.nbytes)) {
            summary_.address_mismatch_tasks++;
            continue;
        }
        if (observation.prediction_ts_ns > input.first_use_ts_ns) {
            observation.causality_error = true;
            summary_.causality_errors++;
            continue;
        }
        observation.first_use_ts_ns = input.first_use_ts_ns;
        task.first_use_seen = true;
        // Prequential ordering is deliberate: the current outcome may update
        // history only after every prediction for this Task has been frozen.
        observe_residuals_unlocked(task);
        observe_horizon_unlocked(task);
        matched.push_back(task_id);
    }
    if (matched.empty()) {
        summary_.unmatched_first_uses++;
        return {};
    }
    if (matched.size() > 1) {
        summary_.ambiguous_first_uses++;
    }
    return finalize_ready_unlocked(matched);
}

void ExpertShadowSlack::expire_task(uint64_t task_id, const char * reason) {
    if (!config_.enabled || task_id == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    const auto found = tasks_.find(task_id);
    if (found == tasks_.end()) {
        return;
    }
    found->second.observation.unavailable_reason = reason ? reason : "expired";
    erase_task_unlocked(task_id, true, false);
}

std::vector<ExpertShadowTaskObservation> ExpertShadowSlack::finalize_ready_unlocked(
        const std::vector<uint64_t> & task_ids) {
    std::vector<ExpertShadowTaskObservation> result;
    for (uint64_t task_id : task_ids) {
        const auto found = tasks_.find(task_id);
        if (found == tasks_.end() || !found->second.issue_seen || !found->second.first_use_seen) {
            continue;
        }
        ExpertShadowTaskObservation & observation = found->second.observation;
        if (observation.issue_ts_ns != 0) {
            observation.actual_issue_slack_ns = saturated_signed_difference(
                    observation.first_use_ts_ns, observation.issue_ts_ns);
            observation.has_actual_issue_slack = true;
        }
        if (observation.returned_ts_ns != 0) {
            observation.actual_return_slack_ns = saturated_signed_difference(
                    observation.first_use_ts_ns, observation.returned_ts_ns);
            observation.has_actual_return_slack = true;
        }
        observation.finalized = true;
        update_summary_unlocked(observation);
        result.push_back(observation);
        erase_task_unlocked(task_id, false, false);
    }
    return result;
}

void ExpertShadowSlack::update_error_aggregate_unlocked(
        ExpertShadowErrorAggregate & aggregate,
        const ExpertShadowPrediction & prediction,
        const ExpertShadowTaskObservation & observation) {
    if (!prediction.prediction_available || observation.first_use_ts_ns == 0 ||
            observation.prediction_ts_ns == 0) {
        return;
    }
    const int64_t error = saturated_signed_difference(
            prediction.predicted_first_use_ts_ns, observation.first_use_ts_ns);
    const uint64_t absolute = error == std::numeric_limits<int64_t>::min() ?
            (uint64_t) std::numeric_limits<int64_t>::max() + 1ull :
            (uint64_t) (error < 0 ? -error : error);
    aggregate.count++;
    aggregate.absolute_error_sum_ns = saturated_add(aggregate.absolute_error_sum_ns, absolute);
    aggregate.signed_error_sum_ns = saturated_signed_add(aggregate.signed_error_sum_ns, error);
    const bool residual_warmup = prediction.calibration_model ==
            ExpertShadowCalibrationModel::ResidualQuantile && prediction.residual_warmup;
    if (prediction.estimator_warmup || residual_warmup) {
        aggregate.warmup++;
    }
    const bool residual_fallback = prediction.calibration_model ==
            ExpertShadowCalibrationModel::ResidualQuantile &&
            prediction.residual_fallback_level != ExpertShadowFallbackLevel::Exact;
    if (prediction.fallback_level != ExpertShadowFallbackLevel::Exact || residual_fallback) {
        aggregate.fallback++;
    }
    if (prediction.clipped_low || prediction.clipped_high) {
        aggregate.clipped++;
    }
    if (!prediction.estimator_warmup && !residual_warmup &&
            prediction.fallback_level == ExpertShadowFallbackLevel::Exact &&
            !residual_fallback) {
        aggregate.mature_exact++;
    }
    static const uint64_t error_limits_ns[] = {
        100'000, 500'000, 1'000'000, 5'000'000, 20'000'000, 100'000'000,
    };
    size_t error_bucket = 0;
    while (error_bucket < sizeof(error_limits_ns) / sizeof(error_limits_ns[0]) &&
            absolute > error_limits_ns[error_bucket]) {
        error_bucket++;
    }
    aggregate.absolute_error_histogram[error_bucket]++;
}

void ExpertShadowSlack::update_target_aggregate_unlocked(
        ExpertShadowTargetAggregate & aggregate,
        const ExpertShadowPrediction & prediction,
        const ExpertShadowTaskObservation & observation,
        bool return_target) {
    const bool available = return_target ? prediction.return_prediction_available :
            prediction.issue_prediction_available;
    const bool has_actual = return_target ? observation.has_actual_return_slack :
            observation.has_actual_issue_slack;
    if (!available || !has_actual) {
        aggregate.unavailable++;
        return;
    }

    const int64_t predicted_slack = return_target ?
            prediction.predicted_return_slack_ns : prediction.predicted_issue_slack_ns;
    const int64_t actual_slack = return_target ?
            observation.actual_return_slack_ns : observation.actual_issue_slack_ns;
    const int64_t error = saturated_i64_sub(predicted_slack, actual_slack);
    const uint64_t absolute = error == std::numeric_limits<int64_t>::min() ?
            (uint64_t) std::numeric_limits<int64_t>::max() + 1ull :
            (uint64_t) (error < 0 ? -error : error);
    aggregate.count++;
    aggregate.absolute_error_sum_ns = saturated_add(aggregate.absolute_error_sum_ns, absolute);
    aggregate.signed_error_sum_ns = saturated_signed_add(aggregate.signed_error_sum_ns, error);

    const bool predicted_on_time = predicted_slack > 0;
    const bool actual_on_time = actual_slack > 0;
    if (predicted_on_time && actual_on_time) {
        aggregate.true_positive++;
    } else if (!predicted_on_time && !actual_on_time) {
        aggregate.true_negative++;
    } else if (predicted_on_time) {
        aggregate.false_positive++;
    } else {
        aggregate.false_negative++;
    }

    const bool residual_warmup = prediction.calibration_model ==
            ExpertShadowCalibrationModel::ResidualQuantile && prediction.residual_warmup;
    const bool warmup = prediction.estimator_warmup || residual_warmup ||
            prediction.queue_warmup || prediction.pre_issue_warmup ||
            (return_target && prediction.syscall_service_warmup);
    const bool residual_fallback = prediction.calibration_model ==
            ExpertShadowCalibrationModel::ResidualQuantile &&
            prediction.residual_fallback_level != ExpertShadowFallbackLevel::Exact;
    const bool fallback = prediction.fallback_level != ExpertShadowFallbackLevel::Exact ||
            residual_fallback ||
            prediction.queue_fallback_level != ExpertShadowFallbackLevel::Exact ||
            prediction.pre_issue_fallback_level != ExpertShadowFallbackLevel::Exact ||
            (return_target && prediction.syscall_service_fallback_level !=
                    ExpertShadowFallbackLevel::Exact);
    aggregate.warmup += warmup;
    aggregate.fallback += fallback;
    aggregate.mature_exact += !warmup && !fallback;

    ExpertShadowCalibrationBucket & calibration =
            aggregate.calibration[(size_t) calibration_bucket(predicted_slack)];
    calibration.total++;
    calibration.on_time += actual_on_time;
}

void ExpertShadowSlack::update_duration_aggregate_unlocked(
        ExpertShadowDurationAggregate & aggregate,
        uint64_t predicted_ns,
        uint64_t actual_ns,
        bool warmup,
        bool fallback) {
    const int64_t error = saturated_signed_difference(predicted_ns, actual_ns);
    const uint64_t absolute = error == std::numeric_limits<int64_t>::min() ?
            (uint64_t) std::numeric_limits<int64_t>::max() + 1ull :
            (uint64_t) (error < 0 ? -error : error);
    aggregate.count++;
    aggregate.absolute_error_sum_ns = saturated_add(aggregate.absolute_error_sum_ns, absolute);
    aggregate.signed_error_sum_ns = saturated_signed_add(aggregate.signed_error_sum_ns, error);
    if (warmup) {
        aggregate.warmup++;
    }
    if (fallback) {
        aggregate.fallback++;
    }
}

void ExpertShadowSlack::update_summary_unlocked(
        const ExpertShadowTaskObservation & observation) {
    summary_.finalized_tasks++;
    summary_.finalized_by_phase[phase_index(observation.phase)]++;
    summary_.finalized_by_stage[stage_index(observation.stage)]++;
    const size_t count = std::min(observation.predictions.size(), summary_.candidates.size());
    for (size_t i = 0; i < count; ++i) {
        const ExpertShadowPrediction & prediction = observation.predictions[i];
        ExpertShadowCandidateSummary & candidate = summary_.candidates[i];
        candidate.eligible++;
        if (!prediction.prediction_available) {
            candidate.unavailable++;
        }
        update_error_aggregate_unlocked(candidate.overall, prediction, observation);
        update_error_aggregate_unlocked(
                candidate.by_phase[phase_index(observation.phase)], prediction, observation);
        update_error_aggregate_unlocked(
                candidate.by_stage[stage_index(observation.stage)], prediction, observation);
        update_target_aggregate_unlocked(
                candidate.issue_overall, prediction, observation, false);
        update_target_aggregate_unlocked(
                candidate.return_overall, prediction, observation, true);
        update_target_aggregate_unlocked(
                candidate.issue_by_phase[phase_index(observation.phase)],
                prediction, observation, false);
        update_target_aggregate_unlocked(
                candidate.return_by_phase[phase_index(observation.phase)],
                prediction, observation, true);
        update_target_aggregate_unlocked(
                candidate.issue_by_stage[stage_index(observation.stage)],
                prediction, observation, false);
        update_target_aggregate_unlocked(
                candidate.return_by_stage[stage_index(observation.stage)],
                prediction, observation, true);
    }
    if (observation.has_actual_queue_wait) {
        bool seen_a = false;
        bool seen_b = false;
        for (const ExpertShadowPrediction & prediction : observation.predictions) {
            if (!prediction.prediction_available) {
                continue;
            }
            const size_t index = prediction.queue_model ==
                    ExpertShadowQueueModel::QueueDepthWorkerEwma ? 0 : 1;
            bool & seen = index == 0 ? seen_a : seen_b;
            if (seen) {
                continue;
            }
            update_duration_aggregate_unlocked(
                    summary_.queue_models[index], prediction.predicted_queue_wait_ns,
                    observation.actual_queue_wait_ns, prediction.queue_warmup,
                    prediction.queue_fallback_level != ExpertShadowFallbackLevel::Exact);
            seen = true;
        }
    }
}

void ExpertShadowSlack::advance_step_unlocked(uint64_t step) {
    if (!has_active_step_ || step > active_step_) {
        has_active_step_ = true;
        active_step_ = step;
        observed_first_use_keys_.clear();
    }
    std::vector<uint64_t> expired;
    for (const auto & entry : tasks_) {
        const uint64_t task_step = entry.second.observation.step;
        if (task_step < step && step - task_step > config_.step_retention) {
            expired.push_back(entry.first);
        }
    }
    for (uint64_t task_id : expired) {
        erase_task_unlocked(task_id, true, false);
    }
}

void ExpertShadowSlack::erase_task_unlocked(
        uint64_t task_id, bool expired, bool capacity_expired) {
    const auto found = tasks_.find(task_id);
    if (found == tasks_.end()) {
        return;
    }
    const ExpertShadowTaskObservation & observation = found->second.observation;
    const bool issue_seen = found->second.issue_seen;
    const bool first_use_seen = found->second.first_use_seen;
    const std::string key = semantic_key(
            observation.step, observation.layer, observation.expert, observation.tensor);
    const auto keyed = tasks_by_semantic_key_.find(key);
    if (keyed != tasks_by_semantic_key_.end()) {
        std::vector<uint64_t> & task_ids = keyed->second;
        task_ids.erase(std::remove(task_ids.begin(), task_ids.end(), task_id), task_ids.end());
        if (task_ids.empty()) {
            tasks_by_semantic_key_.erase(keyed);
        }
    }
    tasks_.erase(found);
    if (expired) {
        summary_.expired_tasks++;
        summary_.expired_without_issue += !issue_seen;
        summary_.expired_without_first_use += !first_use_seen;
    }
    if (capacity_expired) {
        summary_.capacity_expired_tasks++;
    }
    if (insertion_order_.size() > config_.max_pending_tasks &&
            insertion_order_.size() - config_.max_pending_tasks > config_.max_pending_tasks) {
        insertion_order_.clear();
        for (const auto & task : tasks_) {
            insertion_order_.push_back(task.first);
        }
    }
}

void ExpertShadowSlack::enforce_capacity_unlocked() {
    while (tasks_.size() >= config_.max_pending_tasks && !insertion_order_.empty()) {
        const uint64_t task_id = insertion_order_.front();
        insertion_order_.pop_front();
        if (tasks_.find(task_id) != tasks_.end()) {
            erase_task_unlocked(task_id, true, true);
        }
    }
}

ExpertShadowSummary ExpertShadowSlack::summary() {
    std::lock_guard<std::mutex> lock(mu_);
    ExpertShadowSummary result = summary_;
    result.pending_tasks = tasks_.size();
    result.estimator_cells = horizon_cells_.size();
    result.residual_cells = residual_cells_.size();
    for (size_t i = 0; i < pre_issue_cells_.size(); ++i) {
        result.pre_issue_buckets[i].count = pre_issue_cells_[i].count;
        result.pre_issue_buckets[i].window_count = pre_issue_cells_[i].window.size();
        result.pre_issue_buckets[i].ewma_ns = pre_issue_cells_[i].ewma;
        result.syscall_service_buckets[i].count = syscall_service_cells_[i].count;
        result.syscall_service_buckets[i].window_count =
                syscall_service_cells_[i].window.size();
        result.syscall_service_buckets[i].ewma_ns = syscall_service_cells_[i].ewma;
    }
    result.pre_issue_global.count = pre_issue_global_.count;
    result.pre_issue_global.window_count = pre_issue_global_.window.size();
    result.pre_issue_global.ewma_ns = pre_issue_global_.ewma;
    result.syscall_service_global.count = syscall_service_global_.count;
    result.syscall_service_global.window_count = syscall_service_global_.window.size();
    result.syscall_service_global.ewma_ns = syscall_service_global_.ewma;
    for (size_t i = 0; i < worker_cells_.size(); ++i) {
        result.worker_buckets[i].count = worker_cells_[i].count;
        result.worker_buckets[i].window_count = worker_cells_[i].window.size();
        result.worker_buckets[i].ewma_ns = worker_cells_[i].ewma;
    }
    result.worker_global.count = worker_global_.count;
    result.worker_global.window_count = worker_global_.window.size();
    result.worker_global.ewma_ns = worker_global_.ewma;
    result.throughput_sample_count = throughput_count_;
    result.throughput_ewma_bytes_per_ns = throughput_ewma_bytes_per_ns_;
    result.config = config_;
    return result;
}
