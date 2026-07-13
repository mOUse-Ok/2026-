#pragma once

enum class ExpertTaskState {
    New,
    Created,
    Admitted,
    Enqueued,
    Dequeued,
    Rejected,
    Issued,
    Cancelled,
};

enum class ExpertTaskEvent {
    Create,
    Admit,
    Reject,
    Enqueue,
    Dequeue,
    Issue,
    Cancel,
};

bool expert_task_apply_event(ExpertTaskState & state, ExpertTaskEvent event);
const char * expert_task_state_name(ExpertTaskState state);
const char * expert_task_event_name(ExpertTaskEvent event);

