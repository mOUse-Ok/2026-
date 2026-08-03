#pragma once

#include "expert_prefetch_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

enum class ExpertQueueOverheadMode {
    Off,
    Summary,
    Detail,
};

const char * expert_queue_overhead_mode_name(ExpertQueueOverheadMode mode);
bool expert_queue_overhead_parse_mode(const char * value, ExpertQueueOverheadMode & mode);

using ExpertQueueOverheadClock = uint64_t (*)();

struct ExpertQueueCheckedDuration {
    bool available = false;
    bool regression = false;
    uint64_t value_ns = 0;
};

ExpertQueueCheckedDuration expert_queue_checked_duration(uint64_t start_ns, uint64_t end_ns);

struct ExpertQueueBoundedAggregate {
    static constexpr size_t kBucketCount = 65;

    uint64_t count = 0;
    uint64_t total = 0;
    uint64_t min = 0;
    uint64_t max = 0;
    uint64_t zero_count = 0;
    uint64_t unavailable_count = 0;
    uint64_t regression_count = 0;
    uint64_t overflow_count = 0;
    std::array<uint64_t, kBucketCount> buckets{};

    void record(const ExpertQueueCheckedDuration & value);
    void record_value(uint64_t value);
    void record_unavailable(bool regression);
    uint64_t quantile_bucket_upper(double quantile) const;
};

struct ExpertQueueOverheadCell {
    ExpertQueueBoundedAggregate mutex_acquire_wait_ns;
    ExpertQueueBoundedAggregate mutex_hold_ns;
    ExpertQueueBoundedAggregate queue_scan_ns;
    ExpertQueueBoundedAggregate queue_scan_candidates;
};

struct ExpertQueueOverheadBatchSample {
    uint64_t batch_id = 0;
    ExpertAsyncPriorityMode priority_mode = ExpertAsyncPriorityMode::Score;
    int phase = 0;
    uint64_t lock_wait_start_ts_ns = 0;
    uint64_t lock_acquired_ts_ns = 0;
    uint64_t lock_release_ts_ns = 0;
    uint64_t condition_wait_count = 0;
    uint64_t condition_reacquire_count = 0;
    uint64_t repeat_wake_count = 0;
    uint64_t clock_read_count = 0;
};

struct ExpertQueueOverheadSelectionSample {
    uint64_t decision_id = 0;
    uint64_t batch_id = 0;
    uint64_t batch_slot = 0;
    uint64_t worker_id = 0;
    int phase = 0;
    uint64_t step = 0;
    ExpertAsyncPriorityMode priority_mode = ExpertAsyncPriorityMode::Score;
    const char * selection_strategy = "unavailable";
    uint64_t queue_depth_before = 0;
    uint64_t queue_scan_candidates = 0;
    bool queue_scan_available = false;
    uint64_t scan_start_ts_ns = 0;
    uint64_t scan_end_ts_ns = 0;
    uint64_t winner_task_id = 0;
    const char * winner_class = "unavailable";
    uint64_t batch_decision_ts_ns = 0;
    uint64_t clock_read_count = 0;
};

struct ExpertQueueOverheadSnapshot {
    static constexpr size_t kPriorityModeCount = 5;
    static constexpr size_t kPhaseCount = 3;

    ExpertQueueOverheadMode mode = ExpertQueueOverheadMode::Off;
    uint64_t workers = 0;
    uint64_t scheduler_batch = 0;
    uint64_t selection_count = 0;
    uint64_t batch_count = 0;
    uint64_t condition_wait_count = 0;
    uint64_t condition_reacquire_count = 0;
    uint64_t repeat_wake_count = 0;
    uint64_t clock_read_count = 0;
    uint64_t detail_event_count = 0;
    uint64_t idle_wait_exit_count = 0;
    uint64_t unsubmitted_lock_clock_read_count = 0;
    uint64_t next_decision_id = 0;
    uint64_t next_batch_id = 0;
    ExpertQueueBoundedAggregate clock_self_check_ns;
    ExpertQueueOverheadCell global;
    std::array<ExpertQueueOverheadCell, kPriorityModeCount * kPhaseCount> cells{};
};

class ExpertQueueOverheadObserver {
public:
    void reset(
            ExpertQueueOverheadMode mode,
            uint64_t workers,
            uint64_t scheduler_batch,
            ExpertQueueOverheadClock clock);

    ExpertQueueOverheadMode mode() const;
    bool enabled() const;
    bool detail_enabled() const;

    void record_batch(const ExpertQueueOverheadBatchSample & sample);
    void record_idle_wait_exit(
            uint64_t condition_wait_count,
            uint64_t condition_reacquire_count,
            uint64_t repeat_wake_count,
            uint64_t clock_read_count);
    void record_selection(const ExpertQueueOverheadSelectionSample & sample);
    void record_detail_event();
    ExpertQueueOverheadSnapshot snapshot() const;

private:
    static size_t cell_index(ExpertAsyncPriorityMode mode, int phase);

    mutable std::mutex mu_;
    ExpertQueueOverheadSnapshot snapshot_;
    ExpertQueueOverheadClock clock_ = nullptr;
};

class ExpertQueueObservedLock {
public:
    ExpertQueueObservedLock(std::mutex & mutex, ExpertQueueOverheadClock clock);
    ~ExpertQueueObservedLock();

    ExpertQueueObservedLock(const ExpertQueueObservedLock &) = delete;
    ExpertQueueObservedLock & operator=(const ExpertQueueObservedLock &) = delete;

    void lock();
    void unlock();
    void unlock_and_measure();

    bool owns_lock() const;
    uint64_t last_wait_start_ts_ns() const;
    uint64_t last_acquired_ts_ns() const;
    uint64_t last_release_ts_ns() const;
    uint64_t lock_count() const;
    uint64_t clock_read_count() const;

private:
    uint64_t now();

    std::mutex & mutex_;
    ExpertQueueOverheadClock clock_;
    bool owns_ = false;
    uint64_t last_wait_start_ts_ns_ = 0;
    uint64_t last_acquired_ts_ns_ = 0;
    uint64_t last_release_ts_ns_ = 0;
    uint64_t lock_count_ = 0;
    uint64_t clock_read_count_ = 0;
};
