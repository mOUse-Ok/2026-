#include "expert_max_wait_protection.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-expert-max-wait-protection: %s\n", message);
        std::abort();
    }
}

static ExpertMaxWaitConfig config(uint64_t threshold_ns = 100, uint64_t guard_ns = 20) {
    ExpertMaxWaitConfig result;
    result.threshold_ns = threshold_ns;
    result.urgent_guard_ns = guard_ns;
    return result;
}

static ExpertMaxWaitKey key(
        uint64_t sequence,
        uint64_t enqueued_ts_ns,
        uint64_t deadline_ts_ns,
        double route_score) {
    ExpertMaxWaitKey result;
    result.enqueued_ts_ns = enqueued_ts_ns;
    result.priority.sequence = sequence;
    result.priority.deadline_ts_ns = deadline_ts_ns;
    result.priority.route_score = route_score;
    return result;
}

static uint64_t select(
        const std::vector<ExpertMaxWaitKey> & keys,
        uint64_t now,
        const ExpertMaxWaitConfig & cfg) {
    size_t best = 0;
    ExpertMaxWaitDecision best_decision = expert_max_wait_classify(keys[0], now, cfg);
    for (size_t i = 1; i < keys.size(); ++i) {
        const ExpertMaxWaitDecision decision = expert_max_wait_classify(keys[i], now, cfg);
        if (expert_max_wait_higher(keys[i], decision, keys[best], best_decision)) {
            best = i;
            best_decision = decision;
        }
    }
    return keys[best].priority.sequence;
}

static void test_threshold_below_matches_deadline_score() {
    const uint64_t now = 1000;
    const ExpertMaxWaitConfig cfg = config(500, 0);
    std::vector<ExpertMaxWaitKey> keys = {
        key(1, 900, 1500, 0.1),
        key(2, 900, 1300, 0.1),
        key(3, 900, 1300, 0.9),
    };
    require(select(keys, now, cfg) == 3, "normal order changed from deadline_score");
}

static void test_class_order_and_boundaries() {
    const uint64_t now = 1000;
    const ExpertMaxWaitConfig cfg = config(100, 20);
    const ExpertMaxWaitKey urgent = key(1, 800, 1020, 0.1);
    const ExpertMaxWaitKey protected_task = key(2, 800, 2000, 0.1);
    const ExpertMaxWaitKey normal = key(3, 950, 1100, 1.0);
    require(expert_max_wait_classify(urgent, now, cfg).task_class == ExpertMaxWaitClass::Urgent,
            "deadline guard boundary was not urgent");
    require(expert_max_wait_classify(protected_task, 900, cfg).task_class == ExpertMaxWaitClass::Protected,
            "waiting threshold boundary was not protected");
    require(select({normal, protected_task, urgent}, now, cfg) == 1,
            "urgent did not beat protected and normal");
    require(select({normal, protected_task}, now, cfg) == 2,
            "protected did not beat normal");
}

static void test_oldest_protected_then_deadline_score() {
    const uint64_t now = 1000;
    const ExpertMaxWaitConfig cfg = config(100, 0);
    const ExpertMaxWaitKey oldest = key(1, 700, 5000, 0.1);
    const ExpertMaxWaitKey newer = key(2, 800, 2000, 1.0);
    require(select({newer, oldest}, now, cfg) == 1, "oldest protected task did not win");

    const ExpertMaxWaitKey lower_score = key(3, 700, 4000, 0.1);
    const ExpertMaxWaitKey higher_score = key(4, 700, 4000, 0.9);
    require(select({lower_score, higher_score}, now, cfg) == 4,
            "protected tie did not use deadline_score");
}

static void test_missing_deadline_and_enqueue_fallbacks() {
    const ExpertMaxWaitConfig cfg = config(100, 0);
    const ExpertMaxWaitDecision missing_deadline = expert_max_wait_classify(
            key(1, 800, 0, 0.1), 1000, cfg);
    require(missing_deadline.task_class == ExpertMaxWaitClass::Protected,
            "missing deadline task could not be protected");

    const ExpertMaxWaitDecision missing_enqueue = expert_max_wait_classify(
            key(2, 0, 900, 0.1), 1000, cfg);
    require(missing_enqueue.task_class == ExpertMaxWaitClass::Normal &&
                    !missing_enqueue.waiting_available &&
                    missing_enqueue.reason == ExpertMaxWaitReason::MissingEnqueueFallback,
            "missing enqueue did not fail closed to normal");

    const ExpertMaxWaitDecision regression = expert_max_wait_classify(
            key(3, 1100, 900, 0.1), 1000, cfg);
    require(regression.task_class == ExpertMaxWaitClass::Normal &&
                    !regression.waiting_available &&
                    regression.reason == ExpertMaxWaitReason::EnqueueTimeRegressionFallback,
            "enqueue time regression underflowed");
}

static void test_saturation_and_strict_config_parse() {
    uint64_t us = 0;
    uint64_t ns = 0;
    require(expert_max_wait_parse_us("1", false, us, ns) && us == 1 && ns == 1000,
            "positive threshold parse failed");
    require(expert_max_wait_parse_us("0", true, us, ns) && us == 0 && ns == 0,
            "zero guard parse failed");
    require(!expert_max_wait_parse_us(nullptr, false, us, ns), "missing value was accepted");
    require(!expert_max_wait_parse_us("", false, us, ns), "empty value was accepted");
    require(!expert_max_wait_parse_us("0", false, us, ns), "zero threshold was accepted");
    require(!expert_max_wait_parse_us("-1", false, us, ns), "negative value was accepted");
    require(!expert_max_wait_parse_us("1.5", false, us, ns), "fractional value was accepted");
    require(!expert_max_wait_parse_us("10us", false, us, ns), "unit suffix was accepted");
    require(!expert_max_wait_parse_us("18446744073709552", false, us, ns),
            "microsecond conversion overflow was accepted");
    require(expert_max_wait_saturating_add(std::numeric_limits<uint64_t>::max() - 1, 10) ==
                    std::numeric_limits<uint64_t>::max(),
            "urgent limit did not saturate");
}

static void test_input_permutation_does_not_change_winner() {
    const ExpertMaxWaitConfig cfg = config(100, 0);
    std::vector<ExpertMaxWaitKey> keys = {
        key(1, 800, 5000, 0.1),
        key(2, 850, 3000, 0.9),
        key(3, 950, 2000, 1.0),
    };
    do {
        require(select(keys, 1000, cfg) == 1, "input permutation changed winner");
    } while (std::next_permutation(
            keys.begin(), keys.end(), [](const ExpertMaxWaitKey & a, const ExpertMaxWaitKey & b) {
                return a.priority.sequence < b.priority.sequence;
            }));
}

int main() {
    test_threshold_below_matches_deadline_score();
    test_class_order_and_boundaries();
    test_oldest_protected_then_deadline_score();
    test_missing_deadline_and_enqueue_fallbacks();
    test_saturation_and_strict_config_parse();
    test_input_permutation_does_not_change_winner();
    return 0;
}
