#include "expert_queue_overhead_observation.h"

#include <atomic>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>

namespace {

std::atomic<uint64_t> clock_value{0};
std::atomic<uint64_t> clock_calls{0};

uint64_t counting_clock() {
    clock_calls.fetch_add(1, std::memory_order_relaxed);
    return clock_value.fetch_add(1, std::memory_order_relaxed);
}

std::mutex controlled_clock_mu;
std::condition_variable controlled_clock_cv;
bool controlled_first_read = false;
uint64_t controlled_second_value = 100;
std::atomic<uint64_t> controlled_calls{0};

uint64_t controlled_contention_clock() {
    const uint64_t call = controlled_calls.fetch_add(1, std::memory_order_relaxed);
    if (call == 0) {
        {
            std::lock_guard<std::mutex> lock(controlled_clock_mu);
            controlled_first_read = true;
        }
        controlled_clock_cv.notify_one();
        return 10;
    }
    return controlled_second_value++;
}

std::mutex repeat_clock_mu;
std::condition_variable repeat_clock_cv;
bool repeat_reacquired_once = false;
std::atomic<uint64_t> repeat_clock_calls{0};

uint64_t repeat_wake_clock() {
    const uint64_t call = repeat_clock_calls.fetch_add(1, std::memory_order_relaxed);
    if (call == 3) {
        {
            std::lock_guard<std::mutex> lock(repeat_clock_mu);
            repeat_reacquired_once = true;
        }
        repeat_clock_cv.notify_one();
    }
    return 2000 + call;
}

void test_mode_and_checked_duration() {
    ExpertQueueOverheadMode mode = ExpertQueueOverheadMode::Detail;
    assert(expert_queue_overhead_parse_mode(nullptr, mode));
    assert(mode == ExpertQueueOverheadMode::Off);
    assert(expert_queue_overhead_parse_mode("", mode));
    assert(mode == ExpertQueueOverheadMode::Off);
    assert(expert_queue_overhead_parse_mode("summary", mode));
    assert(mode == ExpertQueueOverheadMode::Summary);
    assert(expert_queue_overhead_parse_mode("detail", mode));
    assert(mode == ExpertQueueOverheadMode::Detail);
    assert(!expert_queue_overhead_parse_mode("SUMMARY", mode));
    assert(!expert_queue_overhead_parse_mode("invalid", mode));

    ExpertQueueCheckedDuration duration = expert_queue_checked_duration(7, 7);
    assert(duration.available);
    assert(!duration.regression);
    assert(duration.value_ns == 0);
    duration = expert_queue_checked_duration(8, 7);
    assert(!duration.available);
    assert(duration.regression);
    assert(duration.value_ns == 0);
}

void test_bounded_aggregate() {
    ExpertQueueBoundedAggregate aggregate;
    aggregate.record_value(0);
    aggregate.record_value(1);
    aggregate.record_value(2);
    aggregate.record_value(3);
    aggregate.record_unavailable(true);
    assert(aggregate.count == 4);
    assert(aggregate.total == 6);
    assert(aggregate.min == 0);
    assert(aggregate.max == 3);
    assert(aggregate.zero_count == 1);
    assert(aggregate.unavailable_count == 1);
    assert(aggregate.regression_count == 1);
    assert(aggregate.quantile_bucket_upper(0.50) == 1);
    assert(aggregate.quantile_bucket_upper(0.95) == 3);
    assert(aggregate.quantile_bucket_upper(0.99) == 3);

    ExpertQueueBoundedAggregate overflow;
    overflow.count = std::numeric_limits<uint64_t>::max();
    overflow.total = std::numeric_limits<uint64_t>::max();
    overflow.buckets[1] = std::numeric_limits<uint64_t>::max();
    overflow.record_value(1);
    assert(overflow.count == std::numeric_limits<uint64_t>::max());
    assert(overflow.total == std::numeric_limits<uint64_t>::max());
    assert(overflow.buckets[1] == std::numeric_limits<uint64_t>::max());
    assert(overflow.overflow_count == 3);
}

void test_clock_self_check_and_observer() {
    ExpertQueueOverheadObserver observer;
    clock_calls.store(0, std::memory_order_relaxed);
    clock_value.store(0, std::memory_order_relaxed);
    observer.reset(ExpertQueueOverheadMode::Off, 2, 1, counting_clock);
    ExpertQueueOverheadSnapshot snapshot = observer.snapshot();
    assert(clock_calls.load(std::memory_order_relaxed) == 0);
    assert(snapshot.clock_read_count == 0);
    assert(snapshot.clock_self_check_ns.count == 0);

    observer.reset(ExpertQueueOverheadMode::Summary, 4, 1, counting_clock);
    snapshot = observer.snapshot();
    assert(clock_calls.load(std::memory_order_relaxed) == 128);
    assert(snapshot.clock_read_count == 128);
    assert(snapshot.clock_self_check_ns.count == 64);
    assert(snapshot.clock_self_check_ns.min == 1);
    assert(snapshot.clock_self_check_ns.max == 1);

    ExpertQueueOverheadBatchSample batch;
    batch.batch_id = 3;
    batch.priority_mode = ExpertAsyncPriorityMode::DeadlineScore;
    batch.phase = 2;
    batch.lock_wait_start_ts_ns = 100;
    batch.lock_acquired_ts_ns = 110;
    batch.lock_release_ts_ns = 150;
    batch.condition_wait_count = 1;
    batch.condition_reacquire_count = 2;
    batch.repeat_wake_count = 1;
    batch.clock_read_count = 7;
    observer.record_batch(batch);

    ExpertQueueOverheadSelectionSample selection;
    selection.decision_id = 8;
    selection.batch_id = 3;
    selection.phase = 2;
    selection.priority_mode = ExpertAsyncPriorityMode::DeadlineScore;
    selection.queue_scan_available = true;
    selection.queue_scan_candidates = 5;
    selection.scan_start_ts_ns = 120;
    selection.scan_end_ts_ns = 130;
    selection.clock_read_count = 2;
    observer.record_selection(selection);
    snapshot = observer.snapshot();
    assert(snapshot.batch_count == 1);
    assert(snapshot.selection_count == 1);
    assert(snapshot.condition_wait_count == 1);
    assert(snapshot.condition_reacquire_count == 2);
    assert(snapshot.repeat_wake_count == 1);
    assert(snapshot.global.mutex_acquire_wait_ns.total == 10);
    assert(snapshot.global.mutex_hold_ns.total == 40);
    assert(snapshot.global.queue_scan_ns.total == 10);
    assert(snapshot.global.queue_scan_candidates.total == 5);
    assert(snapshot.next_batch_id == 4);
    assert(snapshot.next_decision_id == 9);
    const size_t cell_index =
            static_cast<size_t>(ExpertAsyncPriorityMode::DeadlineScore) * 3 + 2;
    assert(snapshot.cells[cell_index].mutex_hold_ns.total == 40);
    assert(snapshot.cells[cell_index].queue_scan_candidates.total == 5);
}

void test_uncontended_lock() {
    std::mutex mutex;
    clock_calls.store(0, std::memory_order_relaxed);
    clock_value.store(10, std::memory_order_relaxed);
    ExpertQueueObservedLock lock(mutex, counting_clock);
    assert(lock.owns_lock());
    assert(lock.last_wait_start_ts_ns() == 10);
    assert(lock.last_acquired_ts_ns() == 11);
    assert(lock.clock_read_count() == 2);
    lock.unlock_and_measure();
    assert(!lock.owns_lock());
    assert(lock.last_release_ts_ns() == 12);
    assert(lock.clock_read_count() == 3);
}

void test_controlled_contention() {
    std::mutex mutex;
    mutex.lock();
    controlled_calls.store(0, std::memory_order_relaxed);
    controlled_second_value = 500;
    {
        std::lock_guard<std::mutex> lock(controlled_clock_mu);
        controlled_first_read = false;
    }
    uint64_t wait_start = 0;
    uint64_t acquired = 0;
    std::thread worker([&] {
        ExpertQueueObservedLock lock(mutex, controlled_contention_clock);
        wait_start = lock.last_wait_start_ts_ns();
        acquired = lock.last_acquired_ts_ns();
        lock.unlock_and_measure();
    });
    {
        std::unique_lock<std::mutex> lock(controlled_clock_mu);
        controlled_clock_cv.wait(lock, [] { return controlled_first_read; });
    }
    mutex.unlock();
    worker.join();
    assert(wait_start == 10);
    assert(acquired == 500);
    assert(expert_queue_checked_duration(wait_start, acquired).value_ns == 490);
}

void test_condition_reacquisition_and_hold_boundary() {
    std::mutex queue_mutex;
    std::condition_variable_any observed_cv;
    std::mutex state_mutex;
    std::condition_variable state_cv;
    bool worker_entered = false;
    bool ready = false;
    uint64_t lock_count = 0;
    uint64_t final_acquired = 0;
    uint64_t final_release = 0;
    uint64_t calls = 0;

    clock_calls.store(0, std::memory_order_relaxed);
    clock_value.store(1000, std::memory_order_relaxed);
    std::thread worker([&] {
        ExpertQueueObservedLock lock(queue_mutex, counting_clock);
        {
            std::lock_guard<std::mutex> state_lock(state_mutex);
            worker_entered = true;
        }
        state_cv.notify_one();
        observed_cv.wait(lock, [&] { return ready; });
        lock_count = lock.lock_count();
        final_acquired = lock.last_acquired_ts_ns();
        lock.unlock_and_measure();
        final_release = lock.last_release_ts_ns();
        calls = lock.clock_read_count();
    });

    {
        std::unique_lock<std::mutex> state_lock(state_mutex);
        state_cv.wait(state_lock, [&] { return worker_entered; });
    }
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex);
        ready = true;
    }
    observed_cv.notify_one();
    worker.join();

    assert(lock_count == 2);
    assert(calls == 5);
    assert(final_release == final_acquired + 1);
    assert(expert_queue_checked_duration(final_acquired, final_release).value_ns == 1);
}

void test_repeat_wake_is_not_a_hold_sample() {
    std::mutex queue_mutex;
    std::condition_variable_any observed_cv;
    std::mutex state_mutex;
    std::condition_variable state_cv;
    bool worker_entered = false;
    bool ready = false;
    uint64_t lock_count = 0;
    uint64_t final_acquired = 0;
    uint64_t final_release = 0;

    repeat_clock_calls.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(repeat_clock_mu);
        repeat_reacquired_once = false;
    }
    std::thread worker([&] {
        ExpertQueueObservedLock lock(queue_mutex, repeat_wake_clock);
        {
            std::lock_guard<std::mutex> state_lock(state_mutex);
            worker_entered = true;
        }
        state_cv.notify_one();
        observed_cv.wait(lock, [&] { return ready; });
        lock_count = lock.lock_count();
        final_acquired = lock.last_acquired_ts_ns();
        lock.unlock_and_measure();
        final_release = lock.last_release_ts_ns();
    });
    {
        std::unique_lock<std::mutex> state_lock(state_mutex);
        state_cv.wait(state_lock, [&] { return worker_entered; });
    }
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex);
        ready = false;
    }
    observed_cv.notify_one();
    {
        std::unique_lock<std::mutex> clock_lock(repeat_clock_mu);
        repeat_clock_cv.wait(clock_lock, [&] { return repeat_reacquired_once; });
    }
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex);
        ready = true;
    }
    observed_cv.notify_one();
    worker.join();

    assert(lock_count == 3);
    assert(repeat_clock_calls.load(std::memory_order_relaxed) == 7);
    assert(final_release == final_acquired + 1);
}

} // namespace

int main() {
    test_mode_and_checked_duration();
    test_bounded_aggregate();
    test_clock_self_check_and_observer();
    test_uncontended_lock();
    test_controlled_contention();
    test_condition_reacquisition_and_hold_boundary();
    test_repeat_wake_is_not_a_hold_sample();
    return 0;
}
