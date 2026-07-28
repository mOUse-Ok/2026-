#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace llm_pressure_shadow {

enum class Mode {
    Off,
    Summary,
    Detail,
};

enum class Status {
    Available,
    NotSampled,
    Unavailable,
    PermissionDenied,
    FieldMissing,
    ParseError,
    IoError,
    Unsupported,
    NoPreviousSample,
    CounterRegression,
    SourceChanged,
    SourceStale,
    NotStarted,
    Stopping,
};

struct PsiValues {
    double some_avg10 = 0.0;
    uint64_t some_total_us = 0;
    double full_avg10 = 0.0;
    uint64_t full_total_us = 0;
};

struct ProcStatValues {
    uint64_t minor_faults = 0;
    uint64_t major_faults = 0;
    uint64_t vsize_bytes = 0;
    uint64_t rss_pages = 0;
};

struct SmapsValues {
    uint64_t rss_bytes = 0;
    uint64_t pss_bytes = 0;
    uint64_t swap_bytes = 0;
    bool swap_available = false;
};

struct QueueSnapshot {
    Status status = Status::Unavailable;
    bool started = false;
    bool stopping = false;
    uint64_t queue_depth = 0;
    uint64_t queued_bytes = 0;
    uint64_t configured_worker_count = 0;
    uint64_t worker_count = 0;
    uint64_t busy_workers = 0;
};

struct CounterDeltaState {
    bool previous_sample_available = false;
    uint64_t previous_value = 0;
    uint64_t previous_read_ts_ns = 0;
    std::string source_identity;
};

struct CounterDeltaResult {
    Status status = Status::Unavailable;
    bool has_value = false;
    uint64_t value = 0;
    uint64_t previous_read_ts_ns = 0;
    uint64_t current_read_ts_ns = 0;
};

using QueueSnapshotProvider = QueueSnapshot (*)();

struct LatestValue {
    Status status = Status::Unavailable;
    bool has_value = false;
    uint64_t value = 0;
};

struct LatestSnapshot {
    uint64_t sequence = 0;
    uint64_t sample_ready_ts_ns = 0;
    int phase = 0;
    uint64_t step = 0;
    LatestValue memory_current_bytes;
    LatestValue memory_high_bytes;
    LatestValue memory_max_bytes;
    LatestValue psi_some_total_us;
    LatestValue psi_full_total_us;
    LatestValue queue_depth;
    LatestValue queued_bytes;
    LatestValue busy_workers;
};

struct CadenceAdvance {
    uint64_t next_scheduled_ts_ns = 0;
    uint64_t missed_intervals = 0;
};

const char * status_name(Status status);
Mode parse_mode(const char * value, bool * valid = nullptr);
bool parse_uint64_scalar(const std::string & text, uint64_t & value, bool & unlimited);
bool parse_key_value(const std::string & text, std::unordered_map<std::string, uint64_t> & values);
bool parse_psi(const std::string & text, PsiValues & values);
bool parse_proc_stat(const std::string & text, ProcStatValues & values);
bool parse_smaps_rollup(const std::string & text, SmapsValues & values);
CounterDeltaResult advance_counter_delta(
        CounterDeltaState & state,
        Status current_status,
        bool current_has_value,
        uint64_t current_value,
        uint64_t current_read_ts_ns,
        const std::string & source_identity);
CadenceAdvance advance_absolute_cadence(
        uint64_t scheduled_ts_ns,
        uint64_t interval_ns,
        uint64_t completion_ts_ns);

Mode mode();
bool enabled();
void set_queue_snapshot_provider(QueueSnapshotProvider provider);
bool latest_snapshot(LatestSnapshot & snapshot);
void start();
void stop();
void observe_step_end(
        int phase,
        uint64_t step,
        uint64_t begin_ts_ns,
        uint64_t end_ts_ns,
        uint64_t latency_ns);
void record_issue(uint64_t step, uint64_t issued_bytes);
void record_hint_call(uint64_t step, uint64_t advised_bytes);

} // namespace llm_pressure_shadow
