#include "expert_memory_object.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-expert-memory-object: %s\n", message);
        std::abort();
    }
}

static void test_demand_merge_activation_and_completion() {
    ExpertMemoryObjectTracker tracker;
    const char * tensor = "blk.5.ffn_gate_exps.weight";

    require(tracker.register_demand(10, 5, 7, tensor, 0x1000, 4096),
            "first demand was not registered");
    require(!tracker.register_demand(10, 5, 7, tensor, 0x1000, 4096),
            "same-step demand was not merged");
    require(!tracker.register_demand(10, 5, 7, tensor, 0x1000, 4096),
            "third same-step demand was not merged");

    ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.memory_objects_created == 1, "memory object count is wrong");
    require(counters.semantic_demands_registered == 1, "registered demand count is wrong");
    require(counters.semantic_demands_merged == 2, "merged demand count is wrong");
    require(counters.pending == 1 && counters.pending_objects == 1,
            "registered demand is not pending");

    require(tracker.observe_first_use(10, 5, 7, tensor), "first use did not activate demand");
    require(!tracker.observe_first_use(10, 5, 7, tensor), "duplicate first use activated twice");
    counters = tracker.counters();
    require(counters.demand_activations == 1, "activation count is wrong");
    require(counters.pending == 0 && counters.active == 1, "activation state is wrong");

    tracker.end_layer(5);
    counters = tracker.counters();
    require(counters.demand_completions == 1, "completion count is wrong");
    require(counters.pending == 0 && counters.active == 0, "layer end did not reach IDLE");
    require(counters.memory_objects_created == 1, "layer end deleted the memory object");
    require(counters.invariant_violations == 0, "lifecycle violated a counter invariant");
}

static void test_cross_step_demand_and_stale_cleanup() {
    ExpertMemoryObjectTracker tracker;
    const char * tensor = "blk.5.ffn_up_exps.weight";

    require(tracker.register_demand(10, 5, 7, tensor, 0x2000, 4096),
            "step 10 demand was not registered");
    tracker.end_layer(5);
    require(tracker.register_demand(11, 5, 7, tensor, 0x2000, 4096),
            "cross-step demand was incorrectly merged");
    require(tracker.observe_first_use(11, 5, 7, tensor),
            "step 11 demand did not activate");
    tracker.end_layer(5);

    const ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.memory_objects_created == 1, "cross-step demand recreated the object");
    require(counters.semantic_demands_registered == 2, "cross-step demand count is wrong");
    require(counters.stale_pending_canceled == 1, "unused step 10 demand was not canceled");
    require(counters.demand_completions == 1, "step 11 demand did not complete");
    require(counters.pending == 0 && counters.active == 0, "cross-step lifecycle leaked users");
}

static void test_tensor_identity_and_unmatched_first_use() {
    ExpertMemoryObjectTracker tracker;
    require(tracker.register_demand(
                    3, 2, 1, "blk.2.ffn_gate_exps.weight", 0x3000, 1024),
            "gate demand was not registered");
    require(tracker.register_demand(
                    3, 2, 1, "blk.2.ffn_down_exps.weight", 0x4000, 1024),
            "down demand was not registered");
    require(!tracker.observe_first_use(3, 2, 2, "blk.2.ffn_gate_exps.weight"),
            "unmatched expert activated a demand");
    tracker.end_layer(2);
    require(!tracker.observe_first_use(4, 2, 1, "blk.2.ffn_gate_exps.weight"),
            "first use without a current demand activated");
    require(!tracker.observe_first_use(4, 2, 1, "blk.2.ffn_gate_exps.weight"),
            "duplicate unmatched first use activated");

    const ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.memory_objects_created == 2, "tensor slices were not separate objects");
    require(counters.unmatched_first_use == 2, "logical unmatched first uses were not counted once");
    require(counters.stale_pending_canceled == 2, "pending tensor demands were not canceled");
    require(counters.pending == 0 && counters.active == 0, "stale cleanup leaked users");
}

static void test_inflight_hint_single_flight() {
    ExpertMemoryObjectTracker tracker;
    const char * tensor = "blk.7.ffn_down_exps.weight";
    require(tracker.register_demand(20, 7, 4, tensor, 0x5000, 4096),
            "hint object demand was not registered");
    require(tracker.try_acquire_hint_slot(7, 4, tensor), "first hint slot was not acquired");
    require(!tracker.try_acquire_hint_slot(7, 4, tensor), "duplicate hint slot was not aggregated");

    ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.hint_slots_acquired == 1, "first hint slot count is wrong");
    require(counters.inflight_hint_aggregated == 1, "aggregation count is wrong");
    require(counters.current_hint_inflight_objects == 1, "inflight object count is wrong");

    require(tracker.release_hint_slot(7, 4, tensor, false), "issued hint slot was not released");
    require(tracker.try_acquire_hint_slot(7, 4, tensor), "released slot was not reacquired");
    require(tracker.release_hint_slot(7, 4, tensor, true), "canceled hint slot was not released");

    counters = tracker.counters();
    require(counters.hint_slots_acquired == 2, "reacquired slot count is wrong");
    require(counters.hint_slots_released == 2, "released slot count is wrong");
    require(counters.hint_terminal_canceled == 1, "canceled slot count is wrong");
    require(counters.current_hint_inflight_objects == 0, "hint slot was leaked");
    require(counters.invariant_violations == 0, "hint slot lifecycle violated an invariant");
}

static void test_semantic_demand_liveness() {
    ExpertMemoryObjectTracker tracker;
    const char * tensor = "blk.8.ffn_gate_exps.weight";
    require(tracker.register_demand(30, 8, 2, tensor, 0x6000, 4096),
            "liveness demand was not registered");
    require(tracker.has_live_demand(8, 2, tensor), "pending demand was not live");
    require(tracker.observe_first_use(30, 8, 2, tensor),
            "liveness demand did not activate");
    require(tracker.has_live_demand(8, 2, tensor), "active demand was not live");
    tracker.end_layer(8);
    require(!tracker.has_live_demand(8, 2, tensor),
            "layer-end cleanup left a live demand");

    tracker.record_semantic_stale_check(true);
    tracker.record_semantic_stale_check(false);
    tracker.record_semantic_stale_cancel(4096);
    const ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.semantic_stale_checked == 2, "semantic stale check count is wrong");
    require(counters.semantic_stale_kept_live == 1, "semantic stale live count is wrong");
    require(counters.semantic_stale_tasks_canceled == 1, "semantic stale cancel count is wrong");
    require(counters.semantic_stale_bytes_avoided == 4096, "semantic stale byte count is wrong");
}

static void test_working_set_admission_without_eviction() {
    ExpertMemoryObjectTracker tracker(4096);
    require(tracker.register_demand(1, 1, 1, "blk.1.ffn_gate_exps.weight", 0x7000, 2048),
            "working-set demand was not registered");

    const ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.working_set_budget_bytes == 4096, "working-set budget is wrong");
    require(counters.working_set_admissions == 1, "working-set admission was not counted");
    require(counters.working_set_current_bytes == 2048, "working-set bytes are wrong");
    require(counters.working_set_objects == 1, "working-set object count is wrong");
    require(counters.working_set_evictions == 0, "sufficient budget evicted an object");
}

static void test_working_set_evicts_oldest_idle_object() {
    ExpertMemoryObjectTracker tracker(1024);
    const char * oldest = "blk.2.ffn_gate_exps.weight";
    const char * newest = "blk.3.ffn_gate_exps.weight";
    require(tracker.register_demand(1, 2, 1, oldest, 0x8000, 1024),
            "oldest working-set demand was not registered");
    tracker.end_layer(2);
    require(tracker.register_demand(2, 3, 1, newest, 0x9000, 1024),
            "newest working-set demand was not registered");

    ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.working_set_evictions == 1, "over-budget admission did not evict");
    require(counters.working_set_evicted_bytes == 1024, "evicted byte count is wrong");
    require(counters.working_set_current_bytes == 1024, "working set did not return to budget");

    require(tracker.register_demand(3, 2, 1, oldest, 0x8000, 1024),
            "evicted oldest object did not accept a later demand");
    counters = tracker.counters();
    require(counters.working_set_readmissions == 1,
            "oldest idle object was not the shadow-eviction victim");
}

static void test_working_set_skips_protected_oldest_object() {
    ExpertMemoryObjectTracker tracker(2048);
    const char * protected_oldest = "blk.4.ffn_gate_exps.weight";
    const char * idle_newer = "blk.5.ffn_gate_exps.weight";
    require(tracker.register_demand(1, 4, 1, protected_oldest, 0xa000, 1024),
            "protected oldest demand was not registered");
    require(tracker.register_demand(2, 5, 1, idle_newer, 0xb000, 1024),
            "idle newer demand was not registered");
    tracker.end_layer(5);
    require(tracker.register_demand(3, 6, 1, "blk.6.ffn_gate_exps.weight", 0xc000, 1024),
            "over-budget demand was not registered");

    ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.working_set_evictions == 1, "safe idle object was not evicted");
    require(counters.working_set_protected_skips > 0,
            "pending oldest object was not skipped during victim selection");

    require(tracker.register_demand(4, 5, 1, idle_newer, 0xb000, 1024),
            "idle victim did not accept a later demand");
    counters = tracker.counters();
    require(counters.working_set_readmissions == 1,
            "protected oldest object was selected instead of the idle candidate");
}

static void test_working_set_all_protected_allows_soft_over_budget() {
    ExpertMemoryObjectTracker tracker(1024);
    require(tracker.register_demand(1, 7, 1, "blk.7.ffn_gate_exps.weight", 0xd000, 1024),
            "first protected demand was not registered");
    require(tracker.register_demand(1, 8, 1, "blk.8.ffn_gate_exps.weight", 0xe000, 1024),
            "second protected demand was not registered");

    const ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.working_set_current_bytes == 2048, "protected soft over-budget was lost");
    require(counters.working_set_evictions == 0, "protected object was shadow evicted");
    require(counters.budget_unresolved_due_to_protection == 1,
            "protected over-budget state was not recorded");
    require(counters.working_set_protected_skips >= 2,
            "protected objects were not considered during the LRU scan");
}

static void test_working_set_readmits_shadow_evicted_object() {
    ExpertMemoryObjectTracker tracker(1024);
    const char * first = "blk.9.ffn_gate_exps.weight";
    const char * second = "blk.10.ffn_gate_exps.weight";
    require(tracker.register_demand(1, 9, 1, first, 0xf000, 1024),
            "initial readmission-test demand was not registered");
    tracker.end_layer(9);
    require(tracker.register_demand(2, 10, 1, second, 0x10000, 1024),
            "evicting readmission-test demand was not registered");
    tracker.end_layer(10);
    require(tracker.register_demand(3, 9, 1, first, 0xf000, 1024),
            "shadow-evicted object was not readmitted");

    const ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.memory_objects_created == 2, "shadow eviction recreated an object");
    require(counters.working_set_admissions == 3, "admission count is wrong after readmission");
    require(counters.working_set_readmissions == 1, "readmission was not counted");
    require(counters.working_set_evictions == 2, "readmission did not preserve LRU replacement");
    require(counters.invariant_violations == 0, "working-set bookkeeping violated an invariant");
}

static void test_probation_readmission_cancels_cold() {
    ExpertMemoryObjectTracker tracker(1024);
    const char * victim = "blk.11.ffn_gate_exps.weight";
    require(tracker.register_demand(10, 11, 1, victim, 0x11000, 1024),
            "probation victim demand was not registered");
    tracker.end_layer(11);
    require(tracker.register_demand(10, 12, 1, "blk.12.ffn_gate_exps.weight", 0x12000, 1024),
            "probation eviction demand was not registered");
    require(tracker.register_demand(11, 11, 1, victim, 0x11000, 1024),
            "probation victim did not readmit");

    const std::vector<ExpertMadVColdCandidate> candidates =
            tracker.end_layer_and_collect_madv_cold_candidates(11, 13, 3);
    const ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.probation_entries == 1, "shadow eviction did not enter probation");
    require(counters.probation_canceled_by_readmission == 1,
            "readmission did not cancel probation before COLD");
    require(counters.current_probation_objects == 0, "canceled probation remained live");
    require(candidates.empty(), "readmitted object became a COLD candidate");
}

static void test_probation_grace_and_single_cold_candidate() {
    ExpertMemoryObjectTracker tracker(1024);
    const char * victim = "blk.13.ffn_gate_exps.weight";
    require(tracker.register_demand(10, 13, 1, victim, 0x13000, 1024),
            "grace-test victim demand was not registered");
    tracker.end_layer(13);
    require(tracker.register_demand(10, 14, 1, "blk.14.ffn_gate_exps.weight", 0x14000, 1024),
            "grace-test eviction demand was not registered");

    require(tracker.end_layer_and_collect_madv_cold_candidates(13, 11, 3).empty(),
            "COLD candidate appeared before grace step 3");
    require(tracker.end_layer_and_collect_madv_cold_candidates(13, 12, 3).empty(),
            "COLD candidate appeared before grace expiry");
    const std::vector<ExpertMadVColdCandidate> candidates =
            tracker.end_layer_and_collect_madv_cold_candidates(13, 13, 3);
    require(candidates.size() == 1, "grace-expired victim did not become one COLD candidate");
    require(candidates[0].layer == 13 && candidates[0].nbytes == 1024,
            "COLD candidate identity is wrong");
    tracker.record_madv_cold_result(true, candidates[0].nbytes);
    require(tracker.end_layer_and_collect_madv_cold_candidates(13, 14, 3).empty(),
            "probation episode emitted COLD more than once");

    const ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.madv_cold_candidates == 1, "COLD candidate count is wrong");
    require(counters.madv_cold_issued == 1 && counters.madv_cold_failed == 0,
            "simulated successful COLD result is wrong");
    require(counters.madv_cold_bytes == 1024, "COLD byte count is wrong");
}

static void test_post_cold_readmission_is_allowed() {
    ExpertMemoryObjectTracker tracker(1024);
    const char * victim = "blk.15.ffn_gate_exps.weight";
    require(tracker.register_demand(10, 15, 1, victim, 0x15000, 1024),
            "post-COLD victim demand was not registered");
    tracker.end_layer(15);
    require(tracker.register_demand(10, 16, 1, "blk.16.ffn_gate_exps.weight", 0x16000, 1024),
            "post-COLD eviction demand was not registered");
    const std::vector<ExpertMadVColdCandidate> candidates =
            tracker.end_layer_and_collect_madv_cold_candidates(15, 13, 3);
    require(candidates.size() == 1, "post-COLD victim did not become a candidate");
    tracker.record_madv_cold_result(true, candidates[0].nbytes);
    require(tracker.register_demand(14, 15, 1, victim, 0x15000, 1024),
            "post-COLD demand was incorrectly blocked");

    const ExpertMemoryObjectCounters counters = tracker.counters();
    require(counters.post_cold_readmissions == 1, "post-COLD readmission was not counted");
    require(counters.working_set_readmissions == 1, "post-COLD object was not readmitted");
}

static void test_protected_objects_never_become_cold_candidates() {
    ExpertMemoryObjectTracker tracker(1024);
    require(tracker.register_demand(10, 17, 1, "blk.17.ffn_gate_exps.weight", 0x17000, 1024),
            "active protection demand was not registered");
    require(tracker.observe_first_use(10, 17, 1, "blk.17.ffn_gate_exps.weight"),
            "active protection demand did not activate");
    require(tracker.register_demand(10, 18, 1, "blk.18.ffn_gate_exps.weight", 0x18000, 1024),
            "pending protection demand was not registered");

    const std::vector<ExpertMadVColdCandidate> candidates =
            tracker.end_layer_and_collect_madv_cold_candidates(17, 13, 3);
    const ExpertMemoryObjectCounters counters = tracker.counters();
    require(candidates.empty(), "protected object became a COLD candidate");
    require(counters.working_set_evictions == 0,
            "active/pending objects were shadow evicted before COLD selection");
    require(counters.cold_protected_violation == 0,
            "protected object reached the COLD candidate path");
}

int main() {
    test_demand_merge_activation_and_completion();
    test_cross_step_demand_and_stale_cleanup();
    test_tensor_identity_and_unmatched_first_use();
    test_inflight_hint_single_flight();
    test_semantic_demand_liveness();
    test_working_set_admission_without_eviction();
    test_working_set_evicts_oldest_idle_object();
    test_working_set_skips_protected_oldest_object();
    test_working_set_all_protected_allows_soft_over_budget();
    test_working_set_readmits_shadow_evicted_object();
    test_probation_readmission_cancels_cold();
    test_probation_grace_and_single_cold_candidate();
    test_post_cold_readmission_is_allowed();
    test_protected_objects_never_become_cold_candidates();
    return 0;
}
