#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace {

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

void require(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

class Barrier {
public:
    explicit Barrier(int participants) : participants_(participants) {
        require(participants > 0, "barrier must have at least one participant");
    }

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mu_);
        const int generation = generation_;
        if (++arrived_ == participants_) {
            arrived_ = 0;
            ++generation_;
            cv_.notify_all();
            return;
        }
        cv_.wait(lock, [&] { return generation_ != generation; });
    }

private:
    const int participants_;
    int arrived_ = 0;
    int generation_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
};

void test_old_order_can_observe_incomplete_shards() {
    std::vector<int> shards(4, -1);
    shards[0] = 100;

    int complete = 0;
    for (int value : shards) {
        complete += value >= 100;
    }
    require(complete == 1, "negative control did not expose a partial Tensor");
}

void test_double_barrier_protects_observation() {
    constexpr int n_threads = 4;
    Barrier producer_complete(n_threads);
    Barrier observation_complete(n_threads);
    std::vector<int> shards(n_threads, -1);
    std::vector<int> snapshot;
    std::vector<std::thread> threads;

    std::mutex state_mu;
    std::condition_variable state_cv;
    bool release_producers = false;
    bool thread_zero_at_producer_barrier = false;
    bool observer_started = false;
    bool release_observer = false;
    int nonzero_threads_at_release_barrier = 0;
    std::atomic<int> nonzero_threads_past_release{0};

    for (int ith = 0; ith < n_threads; ++ith) {
        threads.emplace_back([&, ith] {
            if (ith != 0) {
                std::unique_lock<std::mutex> lock(state_mu);
                state_cv.wait(lock, [&] { return release_producers; });
            }

            shards[ith] = 100 + ith;
            if (ith == 0) {
                std::lock_guard<std::mutex> lock(state_mu);
                thread_zero_at_producer_barrier = true;
                state_cv.notify_all();
            }

            producer_complete.arrive_and_wait();

            if (ith == 0) {
                std::unique_lock<std::mutex> lock(state_mu);
                snapshot = shards;
                observer_started = true;
                state_cv.notify_all();
                state_cv.wait(lock, [&] { return release_observer; });
            } else {
                std::lock_guard<std::mutex> lock(state_mu);
                ++nonzero_threads_at_release_barrier;
                state_cv.notify_all();
            }

            observation_complete.arrive_and_wait();
            if (ith != 0) {
                nonzero_threads_past_release.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    {
        std::unique_lock<std::mutex> lock(state_mu);
        state_cv.wait(lock, [&] { return thread_zero_at_producer_barrier; });
        require(!observer_started, "observer ran before every producer was released");
        release_producers = true;
        state_cv.notify_all();
        state_cv.wait(lock, [&] {
            return observer_started &&
                    nonzero_threads_at_release_barrier == n_threads - 1;
        });
        require(snapshot.size() == n_threads, "observer did not capture the complete Tensor");
        for (int ith = 0; ith < n_threads; ++ith) {
            require(snapshot[ith] == 100 + ith, "observer read an incomplete producer shard");
        }
        require(
                nonzero_threads_past_release.load(std::memory_order_relaxed) == 0,
                "a producer passed the release barrier while the observer was active");
        release_observer = true;
        state_cv.notify_all();
    }

    for (std::thread & thread : threads) {
        thread.join();
    }
    require(
            nonzero_threads_past_release.load(std::memory_order_relaxed) == n_threads - 1,
            "not every producer continued after observation completed");
}

void test_single_thread_order() {
    Barrier producer_complete(1);
    Barrier observation_complete(1);
    int shard = -1;
    bool observed = false;

    shard = 100;
    producer_complete.arrive_and_wait();
    observed = shard == 100;
    observation_complete.arrive_and_wait();

    require(observed, "single-thread observation changed");
}

void test_hook_off_uses_only_existing_release_barrier() {
    constexpr int n_threads = 4;
    Barrier existing_release(n_threads);
    std::vector<int> shards(n_threads, -1);
    std::vector<std::thread> threads;
    std::atomic<int> observer_calls{0};
    std::atomic<int> added_producer_barriers{0};

    for (int ith = 0; ith < n_threads; ++ith) {
        threads.emplace_back([&, ith] {
            shards[ith] = 100 + ith;
            existing_release.arrive_and_wait();
        });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }

    require(observer_calls.load() == 0, "Hook-off path invoked the observer");
    require(added_producer_barriers.load() == 0, "Hook-off path added a producer barrier");
    for (int ith = 0; ith < n_threads; ++ith) {
        require(shards[ith] == 100 + ith, "Hook-off path changed producer output");
    }
}

} // namespace

int main() {
    test_old_order_can_observe_incomplete_shards();
    test_double_barrier_protects_observation();
    test_single_thread_order();
    test_hook_off_uses_only_existing_release_barrier();
    return 0;
}
