#include "expert_queue_overhead_observation.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {

uint64_t saturating_add(uint64_t a, uint64_t b, uint64_t & overflow_count) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        if (overflow_count != std::numeric_limits<uint64_t>::max()) {
            ++overflow_count;
        }
        return std::numeric_limits<uint64_t>::max();
    }
    return a + b;
}

size_t bucket_index(uint64_t value) {
    if (value == 0) {
        return 0;
    }
    size_t index = 1;
    while (value >>= 1) {
        ++index;
    }
    return std::min(index, ExpertQueueBoundedAggregate::kBucketCount - 1);
}

uint64_t bucket_upper(size_t index) {
    if (index == 0) {
        return 0;
    }
    if (index >= 64) {
        return std::numeric_limits<uint64_t>::max();
    }
    return (uint64_t{1} << index) - 1;
}

} // namespace

const char * expert_queue_overhead_mode_name(ExpertQueueOverheadMode mode) {
    switch (mode) {
        case ExpertQueueOverheadMode::Off:     return "off";
        case ExpertQueueOverheadMode::Summary: return "summary";
        case ExpertQueueOverheadMode::Detail:  return "detail";
    }
    return "off";
}

bool expert_queue_overhead_parse_mode(const char * value, ExpertQueueOverheadMode & mode) {
    if (!value || !value[0] || std::strcmp(value, "off") == 0) {
        mode = ExpertQueueOverheadMode::Off;
        return true;
    }
    if (std::strcmp(value, "summary") == 0) {
        mode = ExpertQueueOverheadMode::Summary;
        return true;
    }
    if (std::strcmp(value, "detail") == 0) {
        mode = ExpertQueueOverheadMode::Detail;
        return true;
    }
    return false;
}

ExpertQueueCheckedDuration expert_queue_checked_duration(uint64_t start_ns, uint64_t end_ns) {
    ExpertQueueCheckedDuration result;
    if (end_ns < start_ns) {
        result.regression = true;
        return result;
    }
    result.available = true;
    result.value_ns = end_ns - start_ns;
    return result;
}

void ExpertQueueBoundedAggregate::record(const ExpertQueueCheckedDuration & value) {
    if (!value.available) {
        record_unavailable(value.regression);
        return;
    }
    record_value(value.value_ns);
}

void ExpertQueueBoundedAggregate::record_value(uint64_t value) {
    const uint64_t previous_count = count;
    count = saturating_add(count, 1, overflow_count);
    total = saturating_add(total, value, overflow_count);
    if (previous_count == 0) {
        min = value;
        max = value;
    } else {
        min = std::min(min, value);
        max = std::max(max, value);
    }
    if (value == 0) {
        zero_count = saturating_add(zero_count, 1, overflow_count);
    }
    const size_t index = bucket_index(value);
    buckets[index] = saturating_add(buckets[index], 1, overflow_count);
}

void ExpertQueueBoundedAggregate::record_unavailable(bool regression) {
    unavailable_count = saturating_add(unavailable_count, 1, overflow_count);
    if (regression) {
        regression_count = saturating_add(regression_count, 1, overflow_count);
    }
}

uint64_t ExpertQueueBoundedAggregate::quantile_bucket_upper(double quantile) const {
    if (count == 0) {
        return 0;
    }
    const long double scaled = static_cast<long double>(count) * quantile;
    uint64_t target = static_cast<uint64_t>(scaled);
    if (static_cast<long double>(target) < scaled) {
        ++target;
    }
    target = std::max<uint64_t>(1, target);
    uint64_t cumulative = 0;
    uint64_t ignored_overflow = 0;
    for (size_t i = 0; i < buckets.size(); ++i) {
        cumulative = saturating_add(cumulative, buckets[i], ignored_overflow);
        if (cumulative >= target) {
            return bucket_upper(i);
        }
    }
    return std::numeric_limits<uint64_t>::max();
}

void ExpertQueueOverheadObserver::reset(
        ExpertQueueOverheadMode mode,
        uint64_t workers,
        uint64_t scheduler_batch,
        ExpertQueueOverheadClock clock) {
    std::lock_guard<std::mutex> lock(mu_);
    snapshot_ = {};
    snapshot_.mode = mode;
    snapshot_.workers = workers;
    snapshot_.scheduler_batch = scheduler_batch;
    clock_ = clock;
    if (mode != ExpertQueueOverheadMode::Off && clock_) {
        for (size_t i = 0; i < 64; ++i) {
            const uint64_t start = clock_();
            const uint64_t end = clock_();
            snapshot_.clock_read_count = saturating_add(
                    snapshot_.clock_read_count, 2, snapshot_.global.mutex_hold_ns.overflow_count);
            snapshot_.clock_self_check_ns.record(expert_queue_checked_duration(start, end));
        }
    }
}

ExpertQueueOverheadMode ExpertQueueOverheadObserver::mode() const {
    std::lock_guard<std::mutex> lock(mu_);
    return snapshot_.mode;
}

bool ExpertQueueOverheadObserver::enabled() const {
    return mode() != ExpertQueueOverheadMode::Off;
}

bool ExpertQueueOverheadObserver::detail_enabled() const {
    return mode() == ExpertQueueOverheadMode::Detail;
}

size_t ExpertQueueOverheadObserver::cell_index(ExpertAsyncPriorityMode mode, int phase) {
    size_t mode_index = static_cast<size_t>(mode);
    if (mode_index >= ExpertQueueOverheadSnapshot::kPriorityModeCount) {
        mode_index = 0;
    }
    size_t phase_index = phase >= 0 && phase < static_cast<int>(ExpertQueueOverheadSnapshot::kPhaseCount) ?
            static_cast<size_t>(phase) : 0;
    return mode_index * ExpertQueueOverheadSnapshot::kPhaseCount + phase_index;
}

void ExpertQueueOverheadObserver::record_batch(const ExpertQueueOverheadBatchSample & sample) {
    std::lock_guard<std::mutex> lock(mu_);
    snapshot_.batch_count = saturating_add(
            snapshot_.batch_count, 1, snapshot_.global.mutex_hold_ns.overflow_count);
    snapshot_.condition_wait_count = saturating_add(
            snapshot_.condition_wait_count, sample.condition_wait_count,
            snapshot_.global.mutex_hold_ns.overflow_count);
    snapshot_.condition_reacquire_count = saturating_add(
            snapshot_.condition_reacquire_count, sample.condition_reacquire_count,
            snapshot_.global.mutex_hold_ns.overflow_count);
    snapshot_.repeat_wake_count = saturating_add(
            snapshot_.repeat_wake_count, sample.repeat_wake_count,
            snapshot_.global.mutex_hold_ns.overflow_count);
    snapshot_.clock_read_count = saturating_add(
            snapshot_.clock_read_count, sample.clock_read_count,
            snapshot_.global.mutex_hold_ns.overflow_count);
    const ExpertQueueCheckedDuration acquire = expert_queue_checked_duration(
            sample.lock_wait_start_ts_ns, sample.lock_acquired_ts_ns);
    const ExpertQueueCheckedDuration hold = expert_queue_checked_duration(
            sample.lock_acquired_ts_ns, sample.lock_release_ts_ns);
    snapshot_.global.mutex_acquire_wait_ns.record(acquire);
    snapshot_.global.mutex_hold_ns.record(hold);
    ExpertQueueOverheadCell & cell = snapshot_.cells[cell_index(
            sample.priority_mode, sample.phase)];
    cell.mutex_acquire_wait_ns.record(acquire);
    cell.mutex_hold_ns.record(hold);
    snapshot_.next_batch_id = std::max(snapshot_.next_batch_id, sample.batch_id + 1);
}

void ExpertQueueOverheadObserver::record_idle_wait_exit(
        uint64_t condition_wait_count,
        uint64_t condition_reacquire_count,
        uint64_t repeat_wake_count,
        uint64_t clock_read_count) {
    std::lock_guard<std::mutex> lock(mu_);
    snapshot_.idle_wait_exit_count = saturating_add(
            snapshot_.idle_wait_exit_count, 1,
            snapshot_.global.mutex_hold_ns.overflow_count);
    snapshot_.condition_wait_count = saturating_add(
            snapshot_.condition_wait_count, condition_wait_count,
            snapshot_.global.mutex_hold_ns.overflow_count);
    snapshot_.condition_reacquire_count = saturating_add(
            snapshot_.condition_reacquire_count, condition_reacquire_count,
            snapshot_.global.mutex_hold_ns.overflow_count);
    snapshot_.repeat_wake_count = saturating_add(
            snapshot_.repeat_wake_count, repeat_wake_count,
            snapshot_.global.mutex_hold_ns.overflow_count);
    snapshot_.clock_read_count = saturating_add(
            snapshot_.clock_read_count, clock_read_count,
            snapshot_.global.mutex_hold_ns.overflow_count);
    snapshot_.unsubmitted_lock_clock_read_count = saturating_add(
            snapshot_.unsubmitted_lock_clock_read_count, clock_read_count,
            snapshot_.global.mutex_hold_ns.overflow_count);
}

void ExpertQueueOverheadObserver::record_selection(
        const ExpertQueueOverheadSelectionSample & sample) {
    std::lock_guard<std::mutex> lock(mu_);
    snapshot_.selection_count = saturating_add(
            snapshot_.selection_count, 1, snapshot_.global.queue_scan_ns.overflow_count);
    snapshot_.clock_read_count = saturating_add(
            snapshot_.clock_read_count, sample.clock_read_count,
            snapshot_.global.queue_scan_ns.overflow_count);
    ExpertQueueCheckedDuration scan;
    if (sample.queue_scan_available) {
        scan = expert_queue_checked_duration(sample.scan_start_ts_ns, sample.scan_end_ts_ns);
    }
    snapshot_.global.queue_scan_ns.record(scan);
    snapshot_.global.queue_scan_candidates.record_value(sample.queue_scan_candidates);
    ExpertQueueOverheadCell & cell = snapshot_.cells[cell_index(sample.priority_mode, sample.phase)];
    cell.queue_scan_ns.record(scan);
    cell.queue_scan_candidates.record_value(sample.queue_scan_candidates);
    snapshot_.next_decision_id = std::max(snapshot_.next_decision_id, sample.decision_id + 1);
}

void ExpertQueueOverheadObserver::record_detail_event() {
    std::lock_guard<std::mutex> lock(mu_);
    snapshot_.detail_event_count = saturating_add(
            snapshot_.detail_event_count, 1, snapshot_.global.queue_scan_ns.overflow_count);
    snapshot_.clock_read_count = saturating_add(
            snapshot_.clock_read_count, 1, snapshot_.global.queue_scan_ns.overflow_count);
}

ExpertQueueOverheadSnapshot ExpertQueueOverheadObserver::snapshot() const {
    std::lock_guard<std::mutex> lock(mu_);
    return snapshot_;
}

ExpertQueueObservedLock::ExpertQueueObservedLock(
        std::mutex & mutex,
        ExpertQueueOverheadClock clock) : mutex_(mutex), clock_(clock) {
    lock();
}

ExpertQueueObservedLock::~ExpertQueueObservedLock() {
    if (owns_) {
        mutex_.unlock();
    }
}

uint64_t ExpertQueueObservedLock::now() {
    ++clock_read_count_;
    return clock_ ? clock_() : 0;
}

void ExpertQueueObservedLock::lock() {
    last_wait_start_ts_ns_ = now();
    mutex_.lock();
    last_acquired_ts_ns_ = now();
    owns_ = true;
    ++lock_count_;
}

void ExpertQueueObservedLock::unlock() {
    mutex_.unlock();
    owns_ = false;
}

void ExpertQueueObservedLock::unlock_and_measure() {
    mutex_.unlock();
    owns_ = false;
    last_release_ts_ns_ = now();
}

bool ExpertQueueObservedLock::owns_lock() const {
    return owns_;
}

uint64_t ExpertQueueObservedLock::last_wait_start_ts_ns() const {
    return last_wait_start_ts_ns_;
}

uint64_t ExpertQueueObservedLock::last_acquired_ts_ns() const {
    return last_acquired_ts_ns_;
}

uint64_t ExpertQueueObservedLock::last_release_ts_ns() const {
    return last_release_ts_ns_;
}

uint64_t ExpertQueueObservedLock::lock_count() const {
    return lock_count_;
}

uint64_t ExpertQueueObservedLock::clock_read_count() const {
    return clock_read_count_;
}
