#include "expert_task_lifecycle.h"

bool expert_task_apply_event(ExpertTaskState & state, ExpertTaskEvent event) {
    switch (state) {
        case ExpertTaskState::New:
            if (event == ExpertTaskEvent::Create) {
                state = ExpertTaskState::Created;
                return true;
            }
            return false;
        case ExpertTaskState::Created:
            if (event == ExpertTaskEvent::Admit) {
                state = ExpertTaskState::Admitted;
                return true;
            }
            if (event == ExpertTaskEvent::Reject) {
                state = ExpertTaskState::Rejected;
                return true;
            }
            return false;
        case ExpertTaskState::Admitted:
            if (event == ExpertTaskEvent::Enqueue) {
                state = ExpertTaskState::Enqueued;
                return true;
            }
            if (event == ExpertTaskEvent::Issue) {
                state = ExpertTaskState::Issued;
                return true;
            }
            if (event == ExpertTaskEvent::Cancel) {
                state = ExpertTaskState::Cancelled;
                return true;
            }
            return false;
        case ExpertTaskState::Enqueued:
            if (event == ExpertTaskEvent::Dequeue) {
                state = ExpertTaskState::Dequeued;
                return true;
            }
            return false;
        case ExpertTaskState::Dequeued:
            if (event == ExpertTaskEvent::Issue) {
                state = ExpertTaskState::Issued;
                return true;
            }
            if (event == ExpertTaskEvent::Cancel) {
                state = ExpertTaskState::Cancelled;
                return true;
            }
            return false;
        case ExpertTaskState::Rejected:
        case ExpertTaskState::Issued:
        case ExpertTaskState::Cancelled:
            return false;
    }
    return false;
}

const char * expert_task_state_name(ExpertTaskState state) {
    switch (state) {
        case ExpertTaskState::New:       return "NEW";
        case ExpertTaskState::Created:   return "CREATED";
        case ExpertTaskState::Admitted:  return "ADMITTED";
        case ExpertTaskState::Enqueued:  return "ENQUEUED";
        case ExpertTaskState::Dequeued:  return "DEQUEUED";
        case ExpertTaskState::Rejected:  return "REJECTED";
        case ExpertTaskState::Issued:    return "ISSUED";
        case ExpertTaskState::Cancelled: return "CANCELLED";
    }
    return "UNKNOWN";
}

const char * expert_task_event_name(ExpertTaskEvent event) {
    switch (event) {
        case ExpertTaskEvent::Create:  return "CREATE";
        case ExpertTaskEvent::Admit:   return "ADMIT";
        case ExpertTaskEvent::Reject:  return "REJECT";
        case ExpertTaskEvent::Enqueue: return "ENQUEUE";
        case ExpertTaskEvent::Dequeue: return "DEQUEUE";
        case ExpertTaskEvent::Issue:   return "ISSUE";
        case ExpertTaskEvent::Cancel:  return "CANCEL";
    }
    return "UNKNOWN";
}

