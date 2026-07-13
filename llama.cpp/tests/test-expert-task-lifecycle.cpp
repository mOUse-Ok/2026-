#include "expert_task_lifecycle.h"
#include "expert_first_use_matcher.h"
#include "expert_tensor_stage.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-expert-task-lifecycle: %s\n", message);
        std::abort();
    }
}

static void apply(ExpertTaskState & state, ExpertTaskEvent event, ExpertTaskState expected) {
    require(expert_task_apply_event(state, event), "expected transition was rejected");
    require(state == expected, "transition reached the wrong state");
}

static void test_sync_issue() {
    ExpertTaskState state = ExpertTaskState::New;
    apply(state, ExpertTaskEvent::Create, ExpertTaskState::Created);
    apply(state, ExpertTaskEvent::Admit, ExpertTaskState::Admitted);
    apply(state, ExpertTaskEvent::Issue, ExpertTaskState::Issued);
}

static void test_async_issue() {
    ExpertTaskState state = ExpertTaskState::New;
    apply(state, ExpertTaskEvent::Create, ExpertTaskState::Created);
    apply(state, ExpertTaskEvent::Admit, ExpertTaskState::Admitted);
    apply(state, ExpertTaskEvent::Enqueue, ExpertTaskState::Enqueued);
    apply(state, ExpertTaskEvent::Dequeue, ExpertTaskState::Dequeued);
    apply(state, ExpertTaskEvent::Issue, ExpertTaskState::Issued);
}

static void test_reject() {
    ExpertTaskState state = ExpertTaskState::New;
    apply(state, ExpertTaskEvent::Create, ExpertTaskState::Created);
    apply(state, ExpertTaskEvent::Reject, ExpertTaskState::Rejected);
}

static void test_cancel_after_admit() {
    ExpertTaskState state = ExpertTaskState::New;
    apply(state, ExpertTaskEvent::Create, ExpertTaskState::Created);
    apply(state, ExpertTaskEvent::Admit, ExpertTaskState::Admitted);
    apply(state, ExpertTaskEvent::Cancel, ExpertTaskState::Cancelled);
}

static void test_cancel_after_dequeue() {
    ExpertTaskState state = ExpertTaskState::New;
    apply(state, ExpertTaskEvent::Create, ExpertTaskState::Created);
    apply(state, ExpertTaskEvent::Admit, ExpertTaskState::Admitted);
    apply(state, ExpertTaskEvent::Enqueue, ExpertTaskState::Enqueued);
    apply(state, ExpertTaskEvent::Dequeue, ExpertTaskState::Dequeued);
    apply(state, ExpertTaskEvent::Cancel, ExpertTaskState::Cancelled);
}

static void test_invalid_transitions() {
    ExpertTaskState state = ExpertTaskState::New;
    require(!expert_task_apply_event(state, ExpertTaskEvent::Issue), "NEW accepted ISSUE");
    require(state == ExpertTaskState::New, "invalid transition changed NEW");

    apply(state, ExpertTaskEvent::Create, ExpertTaskState::Created);
    require(!expert_task_apply_event(state, ExpertTaskEvent::Enqueue), "CREATED accepted ENQUEUE");
    require(state == ExpertTaskState::Created, "invalid transition changed CREATED");

    apply(state, ExpertTaskEvent::Reject, ExpertTaskState::Rejected);
    require(!expert_task_apply_event(state, ExpertTaskEvent::Admit), "REJECTED accepted ADMIT");
    require(!expert_task_apply_event(state, ExpertTaskEvent::Cancel), "REJECTED accepted CANCEL");
    require(state == ExpertTaskState::Rejected, "terminal state changed");
}

static ExpertIssuedTask issued_task(
        uint64_t task_id, uint64_t issue_id, uint64_t issued_ts_ns, uintptr_t addr) {
    ExpertIssuedTask task;
    task.task_id = task_id;
    task.issue_id = issue_id;
    task.step = 7;
    task.layer = 3;
    task.expert = 11;
    task.stage = ExpertTensorStage::Late;
    task.tensor = "blk.3.ffn_down_exps.weight";
    task.addr = addr;
    task.nbytes = 4096;
    task.created_ts_ns = 50;
    task.enqueued_ts_ns = 70;
    task.dequeued_ts_ns = 90;
    task.issued_ts_ns = issued_ts_ns;
    return task;
}

static ExpertFirstUseObservation first_use(uint64_t ts_ns, uintptr_t addr, uint64_t step = 7) {
    ExpertFirstUseObservation use;
    use.step = step;
    use.layer = 3;
    use.expert = 11;
    use.stage = ExpertTensorStage::Late;
    use.tensor = "blk.3.ffn_down_exps.weight";
    use.addr = addr;
    use.nbytes = 4096;
    use.first_use_ts_ns = ts_ns;
    return use;
}

static void test_multi_token_duplicate_tasks_match_one_to_many() {
    ExpertFirstUseMatcher matcher;
    matcher.register_issue(issued_task(2, 9, 120, 0x1000));
    matcher.register_issue(issued_task(1, 8, 100, 0x1000));

    ExpertFirstUseMatch match = matcher.observe_first_use(first_use(200, 0x1800));
    require(match.considered, "first use was not considered");
    require(match.matched(), "overlapping issued tasks were not matched");
    require(match.ambiguous(), "one-to-many match was not marked ambiguous");
    require(match.tasks.size() == 2, "logical first use did not match both duplicate tasks");
    require(match.tasks[0].task_id == 1 && match.tasks[1].task_id == 2,
            "duplicate tasks were not returned in deterministic issue order");
    require(match.tasks[0].issue_id == 8, "first use lost issue id");

    match = matcher.observe_first_use(first_use(250, 0x1800));
    require(!match.considered, "duplicate logical first use was considered twice");
    const ExpertFirstUseCounters counters = matcher.counters();
    require(counters.eligible_tasks == 2, "eligible task count is wrong");
    require(counters.matched_tasks == 2, "matched task count is wrong");
    require(counters.unmatched_tasks == 0, "matched duplicate task remained unmatched");
    require(counters.ambiguous_matches == 1, "ambiguous match count is wrong");
    require(counters.duplicate_first_use_ignored == 1,
            "duplicate logical first use was not counted");
    require(counters.matcher_peak_live_tasks == 2, "peak live task count is wrong");
    require(counters.pending_issued_tasks == 0, "matched tasks remained pending");
    require(counters.create_to_first_use_ns[1].count == 2,
            "per-stage create-to-first-use count is wrong");
}

static void test_first_use_requires_overlap_and_causality() {
    ExpertFirstUseMatcher future_issue;
    future_issue.register_issue(issued_task(1, 1, 300, 0x1000));
    ExpertFirstUseMatch match = future_issue.observe_first_use(first_use(200, 0x1000));
    require(match.considered && !match.matched(), "future issue matched first use");
    require(match.unmatched_reason == "issue_after_first_use", "future issue reason is wrong");

    ExpertFirstUseMatcher disjoint_range;
    disjoint_range.register_issue(issued_task(1, 1, 100, 0x1000));
    match = disjoint_range.observe_first_use(first_use(200, 0x5000));
    require(match.considered && !match.matched(), "disjoint range matched first use");
    require(match.unmatched_reason == "address_mismatch", "disjoint range reason is wrong");
    require(disjoint_range.counters().unmatched_first_uses == 1,
            "unmatched first use count is wrong");
}

static void test_first_use_requires_exact_semantic_key() {
    ExpertFirstUseMatcher matcher;
    matcher.register_issue(issued_task(1, 1, 100, 0x1000));
    ExpertFirstUseObservation use = first_use(200, 0x1000);
    use.expert = 12;
    const ExpertFirstUseMatch match = matcher.observe_first_use(std::move(use));
    require(match.considered && !match.matched(), "different expert matched first use");
}

static void test_first_use_expires_old_steps_and_rejects_late_issue() {
    ExpertFirstUseMatcher matcher;
    matcher.register_issue(issued_task(1, 1, 100, 0x1000));
    (void) matcher.observe_first_use(first_use(200, 0x1000, 8));
    matcher.register_issue(issued_task(2, 2, 150, 0x1000));
    const ExpertFirstUseCounters counters = matcher.counters();
    require(counters.matcher_expired_tasks == 1, "old pending task did not expire");
    require(counters.late_issued_tasks == 1, "late issue was not counted");
}

static void test_future_issue_does_not_advance_observed_step() {
    ExpertFirstUseMatcher matcher;
    ExpertIssuedTask future = issued_task(2, 2, 100, 0x1000);
    future.step = 8;
    matcher.register_issue(std::move(future));
    matcher.register_issue(issued_task(1, 1, 90, 0x1000));

    const ExpertFirstUseMatch current = matcher.observe_first_use(first_use(150, 0x1000, 7));
    require(current.matched() && current.tasks[0].task_id == 1,
            "future issue advanced the observed step");
    require(matcher.counters().pending_issued_tasks == 1,
            "future issue was expired by the current step");
}

static void test_stage_mismatch_is_not_associated() {
    ExpertFirstUseMatcher matcher;
    ExpertIssuedTask task = issued_task(1, 1, 100, 0x1000);
    task.stage = ExpertTensorStage::Early;
    matcher.register_issue(std::move(task));
    const ExpertFirstUseMatch match = matcher.observe_first_use(first_use(200, 0x1000));
    require(match.considered && !match.matched(), "different stage matched first use");
    require(match.unmatched_reason == "stage_mismatch", "stage mismatch reason is wrong");
}

static void test_expert_tensor_stage_classification() {
    require(classify_expert_tensor_stage("ffn_gate_exps.weight") == ExpertTensorStage::Early,
            "gate stage mismatch");
    require(classify_expert_tensor_stage("blk.9.ffn_up_exps.weight") == ExpertTensorStage::Early,
            "up stage mismatch");
    require(classify_expert_tensor_stage("blk.9.ffn_gate_up_exps.weight") == ExpertTensorStage::Early,
            "gate-up stage mismatch");
    require(classify_expert_tensor_stage("blk.9.ffn_down_exps.weight") == ExpertTensorStage::Late,
            "down stage mismatch");
    require(classify_expert_tensor_stage("blk.9.ffn_norm_exps.weight") == ExpertTensorStage::Unknown,
            "other expert tensor was not UNKNOWN");
    require(classify_expert_tensor_stage("x_ffn_down_exps.weight") == ExpertTensorStage::Unknown,
            "non-component suffix was classified as LATE");
    require(std::strcmp(expert_tensor_stage_name(ExpertTensorStage::Early), "EARLY") == 0,
            "EARLY name mismatch");
}

int main() {
    test_sync_issue();
    test_async_issue();
    test_reject();
    test_cancel_after_admit();
    test_cancel_after_dequeue();
    test_invalid_transitions();
    test_multi_token_duplicate_tasks_match_one_to_many();
    test_first_use_requires_overlap_and_causality();
    test_first_use_requires_exact_semantic_key();
    test_first_use_expires_old_steps_and_rejects_late_issue();
    test_future_issue_does_not_advance_observed_step();
    test_stage_mismatch_is_not_associated();
    test_expert_tensor_stage_classification();
    require(std::strcmp(expert_task_event_name(ExpertTaskEvent::Issue), "ISSUE") == 0,
            "ISSUE name mismatch");
    require(std::strcmp(expert_task_state_name(ExpertTaskState::Cancelled), "CANCELLED") == 0,
            "CANCELLED name mismatch");
    return 0;
}
