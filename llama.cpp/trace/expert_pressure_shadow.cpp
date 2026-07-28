#include "expert_pressure_shadow.h"

#include "trace_event.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <time.h>
#include <unistd.h>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#endif

namespace llm_pressure_shadow {
namespace {

constexpr uint64_t NS_PER_MS = 1000000ull;
constexpr size_t STEP_SLOT_COUNT = 128;
constexpr size_t COUNTER_TRACKER_COUNT = 23;
constexpr size_t INTERVAL_BUCKET_COUNT = 1001;
constexpr uint64_t INTERVAL_BUCKET_NS = 100000ull;
constexpr size_t MAX_SOURCE_BYTES = 1024 * 1024;

enum CounterTrackerIndex : size_t {
    CounterEventsLow,
    CounterEventsHigh,
    CounterEventsMax,
    CounterEventsOom,
    CounterEventsOomKill,
    CounterEventsOomGroupKill,
    CounterRefaultAnon,
    CounterRefaultFile,
    CounterActivateAnon,
    CounterActivateFile,
    CounterRestoreAnon,
    CounterRestoreFile,
    CounterCgroupPgfault,
    CounterCgroupPgmajfault,
    CounterPgscan,
    CounterPgsteal,
    CounterPgrefill,
    CounterPgactivate,
    CounterPgdeactivate,
    CounterPsiSome,
    CounterPsiFull,
    CounterProcessMinorFaults,
    CounterProcessMajorFaults,
};

template <typename T>
struct Observation {
    T value{};
    bool has_value = false;
    bool unlimited = false;
    Status status = Status::Unavailable;
    uint64_t read_ts_ns = 0;
    uint64_t previous_read_ts_ns = 0;
    std::string error;
};

struct ReadResult {
    std::string text;
    Status status = Status::Unavailable;
    uint64_t read_ts_ns = 0;
    uint64_t io_cost_ns = 0;
    int error_number = 0;
    std::string error;
};

struct Aggregate {
    uint64_t observed = 0;
    uint64_t available = 0;
    uint64_t unavailable = 0;
    long double sum = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    std::map<Status, uint64_t> status_counts;

    void add(Status status, bool has_value, double value) {
        observed++;
        status_counts[status]++;
        if (status == Status::Available && has_value && std::isfinite(value)) {
            available++;
            sum += value;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        } else {
            unavailable++;
        }
    }
};

struct FixedDistribution {
    uint64_t count = 0;
    long double sum = 0.0;
    uint64_t minimum = std::numeric_limits<uint64_t>::max();
    uint64_t maximum = 0;
    std::array<uint64_t, INTERVAL_BUCKET_COUNT> buckets{};

    void add(uint64_t value) {
        count++;
        sum += value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        const size_t bucket = std::min<size_t>(
                buckets.size() - 1, (size_t) (value / INTERVAL_BUCKET_NS));
        buckets[bucket]++;
    }

    uint64_t percentile(double quantile) const {
        if (count == 0) {
            return 0;
        }
        const uint64_t target = std::max<uint64_t>(
                1, (uint64_t) std::ceil(quantile * (double) count));
        uint64_t cumulative = 0;
        for (size_t i = 0; i < buckets.size(); ++i) {
            cumulative += buckets[i];
            if (cumulative >= target) {
                return std::min<uint64_t>(
                        maximum, (uint64_t) (i + 1) * INTERVAL_BUCKET_NS);
            }
        }
        return maximum;
    }
};

struct SmapsSnapshot {
    Observation<uint64_t> rss;
    Observation<uint64_t> pss;
    Observation<uint64_t> swap;
};

struct StepSlot {
    std::atomic<uint64_t> step{std::numeric_limits<uint64_t>::max()};
    std::atomic<uint64_t> writers{0};
    std::atomic<uint64_t> update_epoch{0};
    std::atomic<uint64_t> issued_bytes{0};
    std::atomic<uint64_t> hint_calls{0};
    std::atomic<uint64_t> advised_bytes{0};
};

struct StepCountersSnapshot {
    uint64_t step = 0;
    uint64_t issued_bytes = 0;
    uint64_t hint_calls = 0;
    uint64_t advised_bytes = 0;
};

struct ThreadTuning {
    bool nice_applied = false;
    bool sched_idle_applied = false;
    bool affinity_applied = false;
    int affinity_cpu = -1;
};

struct Runtime {
    std::mutex lifecycle_mu;
    std::mutex wait_mu;
    std::condition_variable wait_cv;
    std::thread sampler;
    std::thread pss_sampler;
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};
    std::atomic<QueueSnapshotProvider> queue_provider{nullptr};
    std::mutex latest_mu;
    LatestSnapshot latest;
    bool have_latest = false;
    std::mutex pss_mu;
    SmapsSnapshot latest_smaps;

    std::mutex counters_mu;
    std::array<StepSlot, STEP_SLOT_COUNT> step_slots{};
    std::atomic<uint64_t> total_issued_bytes{0};
    std::atomic<uint64_t> total_hint_calls{0};
    std::atomic<uint64_t> total_advised_bytes{0};
    std::atomic<uint64_t> step_slot_overwrites{0};
    std::atomic<uint64_t> step_slot_late_drops{0};

    std::atomic<int> last_step_phase{LLM_MEM_TRACE_PHASE_UNKNOWN};
    std::atomic<uint64_t> last_step{0};
    std::atomic<uint64_t> last_step_begin_ts_ns{0};
    std::atomic<uint64_t> last_step_end_ts_ns{0};
    std::atomic<uint64_t> last_step_latency_ns{0};

    std::string run_id;
    std::string cgroup_path;
    uint64_t interval_ns = 25 * NS_PER_MS;
    uint64_t pss_interval_ns = 2000 * NS_PER_MS;
    uint64_t started_ts_ns = 0;
    uint64_t stopped_ts_ns = 0;
    uint64_t sample_count = 0;
    uint64_t missed_intervals = 0;
    uint64_t detail_events = 0;
    uint64_t trace_enqueue_cost_ns = 0;
    uint64_t trace_enqueue_cost_max_ns = 0;
    uint64_t total_wall_cost_ns = 0;
    uint64_t total_cpu_cost_ns = 0;
    uint64_t total_io_cost_ns = 0;
    uint64_t total_parse_cost_ns = 0;
    uint64_t total_queue_cost_ns = 0;
    uint64_t total_serialize_cost_ns = 0;
    uint64_t max_wall_cost_ns = 0;
    uint64_t max_cpu_cost_ns = 0;
    uint64_t max_io_cost_ns = 0;
    uint64_t max_parse_cost_ns = 0;
    uint64_t max_queue_cost_ns = 0;
    uint64_t max_serialize_cost_ns = 0;
    uint64_t max_jitter_ns = 0;
    uint64_t previous_sample_ready_ts_ns = 0;
    uint64_t eligible_sample_count = 0;
    uint64_t memory_high_90_crossings = 0;
    uint64_t memory_high_98_crossings = 0;
    uint64_t psi_some_positive_crossings = 0;
    uint64_t psi_full_positive_crossings = 0;
    uint64_t queue_nonempty_crossings = 0;
    FixedDistribution actual_intervals;
    FixedDistribution jitter_distribution;
    std::array<CounterDeltaState, COUNTER_TRACKER_COUNT> counter_trackers{};
    uint64_t pss_sample_count = 0;
    uint64_t pss_missed_intervals = 0;
    uint64_t pss_total_wall_cost_ns = 0;
    uint64_t pss_total_cpu_cost_ns = 0;
    uint64_t pss_total_io_cost_ns = 0;
    uint64_t pss_total_parse_cost_ns = 0;
    uint64_t pss_max_wall_cost_ns = 0;
    ThreadTuning sampler_tuning;
    ThreadTuning pss_tuning;

    std::map<std::string, Aggregate> aggregates;
};

Runtime & runtime() {
    static Runtime value;
    return value;
}

ThreadTuning configure_observer_thread(int cpu_offset) {
    ThreadTuning tuning;
#ifdef __linux__
    const pid_t tid = (pid_t) syscall(SYS_gettid);
    tuning.nice_applied = setpriority(PRIO_PROCESS, tid, 19) == 0;
    struct sched_param parameters{};
    tuning.sched_idle_applied =
            pthread_setschedparam(pthread_self(), SCHED_IDLE, &parameters) == 0;
    const long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu_count > 0 && cpu_count <= CPU_SETSIZE) {
        const int cpu = std::max(0, (int) cpu_count - 1 - cpu_offset);
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        tuning.affinity_cpu = cpu;
        tuning.affinity_applied =
                pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
    }
#else
    (void) cpu_offset;
#endif
    return tuning;
}

uint64_t monotonic_ns() {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

uint64_t thread_cpu_ns() {
#ifdef CLOCK_THREAD_CPUTIME_ID
    struct timespec ts{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
        return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
    }
#endif
    return 0;
}

uint64_t env_u64(const char * name, uint64_t fallback, uint64_t minimum, uint64_t maximum) {
    const char * text = std::getenv(name);
    if (!text || !text[0]) {
        return fallback;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return fallback;
    }
    return std::min<uint64_t>(maximum, std::max<uint64_t>(minimum, parsed));
}

std::string json_escaped(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    out.push_back('"');
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", (unsigned int) ch);
                    out += buffer;
                } else {
                    out.push_back((char) ch);
                }
        }
    }
    out.push_back('"');
    return out;
}

void append_source_json(
        std::string & line,
        const std::string & name,
        const std::string & scope,
        const std::string & path,
        Status status,
        int error_number,
        const std::string & error) {
    line += "\"" + name + "\":{";
    line += "\"scope\":" + json_escaped(scope);
    line += ",\"path\":" + (path.empty() ? std::string("null") : json_escaped(path));
    line += ",\"status\":" + json_escaped(status_name(status));
    line += ",\"errno\":" +
            (error_number == 0 ? std::string("null") : std::to_string(error_number));
    line += ",\"error\":" + (error.empty() ? std::string("null") : json_escaped(error));
    line += "}";
}

Status status_from_errno(int error_number) {
    if (error_number == EACCES || error_number == EPERM || error_number == EROFS) {
        return Status::PermissionDenied;
    }
    if (error_number == ENOENT || error_number == ENOTDIR) {
        return Status::FieldMissing;
    }
    return Status::IoError;
}

ReadResult read_file(const std::string & path) {
    ReadResult result;
    const uint64_t begin = monotonic_ns();
    FILE * fp = std::fopen(path.c_str(), "r");
    if (!fp) {
        const int saved_errno = errno;
        result.read_ts_ns = monotonic_ns();
        result.io_cost_ns = result.read_ts_ns - begin;
        result.status = status_from_errno(saved_errno);
        result.error_number = saved_errno;
        result.error = std::strerror(saved_errno);
        return result;
    }

    std::array<char, 4096> buffer{};
    while (std::fgets(buffer.data(), (int) buffer.size(), fp)) {
        if (result.text.size() + std::strlen(buffer.data()) > MAX_SOURCE_BYTES) {
            result.status = Status::IoError;
            result.error_number = EOVERFLOW;
            result.error = "source exceeds bounded read capacity";
            break;
        }
        result.text += buffer.data();
    }
    if (result.status == Status::IoError) {
        // Preserve the explicit bounded-read failure above.
    } else if (std::ferror(fp)) {
        const int saved_errno = errno != 0 ? errno : EIO;
        result.status = status_from_errno(saved_errno);
        result.error_number = saved_errno;
        result.error = std::strerror(saved_errno);
    } else {
        result.status = Status::Available;
    }
    std::fclose(fp);
    result.read_ts_ns = monotonic_ns();
    result.io_cost_ns = result.read_ts_ns - begin;
    return result;
}

std::vector<std::string> split_words(const std::string & text) {
    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

std::string decode_mount_path(std::string value) {
    const std::array<std::pair<const char *, const char *>, 4> replacements{{
        {"\\040", " "},
        {"\\011", "\t"},
        {"\\012", "\n"},
        {"\\134", "\\"},
    }};
    for (const auto & replacement : replacements) {
        size_t position = 0;
        while ((position = value.find(replacement.first, position)) != std::string::npos) {
            value.replace(position, 4, replacement.second);
            position += std::strlen(replacement.second);
        }
    }
    return value;
}

std::string resolve_cgroup_path() {
#ifdef __linux__
    const ReadResult self_cgroup = read_file("/proc/self/cgroup");
    const ReadResult mountinfo = read_file("/proc/self/mountinfo");
    if (self_cgroup.status != Status::Available || mountinfo.status != Status::Available) {
        return {};
    }

    std::string relative;
    std::istringstream cgroup_lines(self_cgroup.text);
    for (std::string line; std::getline(cgroup_lines, line);) {
        const size_t first = line.find(':');
        const size_t second = first == std::string::npos ? first : line.find(':', first + 1);
        if (first != std::string::npos && second != std::string::npos && line.substr(0, first) == "0") {
            relative = line.substr(second + 1);
            break;
        }
    }
    if (relative.empty()) {
        return {};
    }

    std::istringstream mount_lines(mountinfo.text);
    for (std::string line; std::getline(mount_lines, line);) {
        const size_t separator = line.find(" - ");
        if (separator == std::string::npos) {
            continue;
        }
        const std::vector<std::string> left = split_words(line.substr(0, separator));
        const std::vector<std::string> right = split_words(line.substr(separator + 3));
        if (left.size() < 5 || right.empty() || right[0] != "cgroup2") {
            continue;
        }
        const std::string root = decode_mount_path(left[3]);
        std::string suffix = relative;
        if (root != "/" && suffix.compare(0, root.size(), root) == 0) {
            suffix.erase(0, root.size());
        }
        std::string path = decode_mount_path(left[4]);
        if (!suffix.empty() && suffix[0] != '/') {
            path.push_back('/');
        }
        path += suffix;
        while (path.size() > 1 && path.back() == '/') {
            path.pop_back();
        }
        return path;
    }
#endif
    return {};
}

template <typename T>
Observation<T> failed_observation(const ReadResult & read) {
    Observation<T> field;
    field.status = read.status;
    field.read_ts_ns = read.read_ts_ns;
    field.error = read.error;
    return field;
}

Observation<uint64_t> scalar_observation(
        const ReadResult & read,
        uint64_t & parse_cost_ns,
        bool allow_max) {
    if (read.status != Status::Available) {
        return failed_observation<uint64_t>(read);
    }
    Observation<uint64_t> field;
    field.read_ts_ns = read.read_ts_ns;
    const uint64_t begin = monotonic_ns();
    bool unlimited = false;
    field.has_value = parse_uint64_scalar(read.text, field.value, unlimited);
    parse_cost_ns += monotonic_ns() - begin;
    field.unlimited = unlimited;
    if (field.has_value || (allow_max && unlimited)) {
        field.status = Status::Available;
    } else {
        field.status = Status::ParseError;
        field.error = allow_max ? "expected uint64 or max" : "expected uint64";
    }
    return field;
}

Observation<uint64_t> map_observation(
        const ReadResult & read,
        const std::unordered_map<std::string, uint64_t> & values,
        const std::string & key) {
    if (read.status != Status::Available) {
        return failed_observation<uint64_t>(read);
    }
    Observation<uint64_t> field;
    field.read_ts_ns = read.read_ts_ns;
    const auto found = values.find(key);
    if (found == values.end()) {
        field.status = Status::FieldMissing;
        field.error = "key not present";
    } else {
        field.status = Status::Available;
        field.has_value = true;
        field.value = found->second;
    }
    return field;
}

void append_observation_prefix(
        std::string & line,
        const std::string & name,
        const Runtime & state,
        Status status,
        uint64_t read_ts_ns,
        const std::string & error) {
    line += ",\"" + name + "\":{";
    line += "\"run_id\":" + json_escaped(state.run_id);
    line += ",\"status\":" + json_escaped(status_name(status));
    line += ",\"read_ts_ns\":" + (read_ts_ns == 0 ? std::string("null") : std::to_string(read_ts_ns));
    line += ",\"error\":" + (error.empty() ? std::string("null") : json_escaped(error));
}

void append_observation(
        std::string & line,
        const std::string & name,
        const Runtime & state,
        const Observation<uint64_t> & field) {
    append_observation_prefix(line, name, state, field.status, field.read_ts_ns, field.error);
    line += ",\"previous_read_ts_ns\":" +
            (field.previous_read_ts_ns == 0 ?
             std::string("null") : std::to_string(field.previous_read_ts_ns));
    line += ",\"value\":" + (field.has_value ? std::to_string(field.value) : std::string("null"));
    if (field.unlimited) {
        line += ",\"unlimited\":true";
    }
    line += "}";
}

void append_observation(
        std::string & line,
        const std::string & name,
        const Runtime & state,
        const Observation<double> & field) {
    append_observation_prefix(line, name, state, field.status, field.read_ts_ns, field.error);
    if (field.has_value && std::isfinite(field.value)) {
        line += ",\"value\":" + std::to_string(field.value);
    } else {
        line += ",\"value\":null";
    }
    line += "}";
}

void append_string_observation(
        std::string & line,
        const std::string & name,
        const Runtime & state,
        const std::string & value,
        Status status,
        uint64_t read_ts_ns,
        const std::string & error = {}) {
    append_observation_prefix(line, name, state, status, read_ts_ns, error);
    line += ",\"value\":" + (status == Status::Available ? json_escaped(value) : std::string("null"));
    line += "}";
}

Observation<uint64_t> available_uint64(uint64_t value, uint64_t ts) {
    Observation<uint64_t> field;
    field.value = value;
    field.has_value = true;
    field.status = Status::Available;
    field.read_ts_ns = ts;
    return field;
}

Observation<uint64_t> delta_observation(
        Runtime & state,
        CounterTrackerIndex tracker,
        const Observation<uint64_t> & current,
        const std::string & source_identity) {
    const CounterDeltaResult result = advance_counter_delta(
            state.counter_trackers[(size_t) tracker],
            current.status,
            current.has_value,
            current.value,
            current.read_ts_ns,
            source_identity);
    Observation<uint64_t> delta;
    delta.status = result.status;
    delta.has_value = result.has_value;
    delta.value = result.value;
    delta.read_ts_ns = result.current_read_ts_ns;
    delta.previous_read_ts_ns = result.previous_read_ts_ns;
    switch (result.status) {
        case Status::NoPreviousSample:
            delta.error = "no previous successful adjacent sample";
            break;
        case Status::CounterRegression:
            delta.error = "counter regressed or reset";
            break;
        case Status::SourceChanged:
            delta.error = "counter source identity changed";
            break;
        default:
            if (result.status != Status::Available) {
                delta.error = current.error.empty() ?
                        "current counter unavailable" : current.error;
            }
            break;
    }
    return delta;
}

LatestValue latest_value(const Observation<uint64_t> & observation_value) {
    LatestValue value;
    value.status = observation_value.status;
    value.has_value = observation_value.has_value;
    value.value = observation_value.value;
    return value;
}

Observation<double> available_double(double value, uint64_t ts) {
    Observation<double> field;
    field.value = value;
    field.has_value = true;
    field.status = Status::Available;
    field.read_ts_ns = ts;
    return field;
}

template <typename T>
void add_aggregate(Runtime & state, const std::string & name, const Observation<T> & field) {
    state.aggregates[name].add(field.status, field.has_value, (double) field.value);
}

void sample_once(Runtime & state, uint64_t scheduled_ts_ns, uint64_t missed_before_sample) {
    const uint64_t wall_begin = monotonic_ns();
    const uint64_t cpu_begin = thread_cpu_ns();
    const uint64_t jitter_ns = wall_begin > scheduled_ts_ns ? wall_begin - scheduled_ts_ns : 0;
    uint64_t io_cost_ns = 0;
    uint64_t parse_cost_ns = 0;

    const auto read_cgroup = [&](const char * name) {
        ReadResult read;
        if (state.cgroup_path.empty()) {
            read.status = Status::Unavailable;
            read.read_ts_ns = monotonic_ns();
            read.error = "current cgroup path unavailable";
            return read;
        }
        read = read_file(state.cgroup_path + "/" + name);
        io_cost_ns += read.io_cost_ns;
        return read;
    };

    const ReadResult memory_current_read = read_cgroup("memory.current");
    const ReadResult memory_high_read = read_cgroup("memory.high");
    const ReadResult memory_max_read = read_cgroup("memory.max");
    const ReadResult swap_current_read = read_cgroup("memory.swap.current");
    const ReadResult swap_max_read = read_cgroup("memory.swap.max");
    ReadResult events_read = read_cgroup("memory.events");
    ReadResult stat_read = read_cgroup("memory.stat");
    const ReadResult pressure_read = read_cgroup("memory.pressure");
    ReadResult proc_read = read_file("/proc/self/stat");
    io_cost_ns += proc_read.io_cost_ns;

    const uint64_t scalar_parse_begin_cost = parse_cost_ns;
    Observation<uint64_t> memory_current =
            scalar_observation(memory_current_read, parse_cost_ns, false);
    Observation<uint64_t> memory_high =
            scalar_observation(memory_high_read, parse_cost_ns, true);
    Observation<uint64_t> memory_max =
            scalar_observation(memory_max_read, parse_cost_ns, true);
    Observation<uint64_t> swap_current =
            scalar_observation(swap_current_read, parse_cost_ns, false);
    Observation<uint64_t> swap_max =
            scalar_observation(swap_max_read, parse_cost_ns, true);
    const uint64_t scalar_parse_cost_ns = parse_cost_ns - scalar_parse_begin_cost;

    std::unordered_map<std::string, uint64_t> events;
    const uint64_t events_parse_begin_cost = parse_cost_ns;
    if (events_read.status == Status::Available) {
        const uint64_t begin = monotonic_ns();
        if (!parse_key_value(events_read.text, events)) {
            events.clear();
            events_read.status = Status::ParseError;
            events_read.error = "invalid key/value format";
        }
        parse_cost_ns += monotonic_ns() - begin;
    }
    const uint64_t events_parse_cost_ns = parse_cost_ns - events_parse_begin_cost;
    Observation<uint64_t> events_low = map_observation(events_read, events, "low");
    Observation<uint64_t> events_high = map_observation(events_read, events, "high");
    Observation<uint64_t> events_max = map_observation(events_read, events, "max");
    Observation<uint64_t> events_oom = map_observation(events_read, events, "oom");
    Observation<uint64_t> events_oom_kill = map_observation(events_read, events, "oom_kill");
    Observation<uint64_t> events_oom_group_kill =
            map_observation(events_read, events, "oom_group_kill");
    const std::string events_source = state.cgroup_path + "/memory.events";
    Observation<uint64_t> events_low_delta = delta_observation(
            state, CounterEventsLow, events_low, events_source + ":low");
    Observation<uint64_t> events_high_delta = delta_observation(
            state, CounterEventsHigh, events_high, events_source + ":high");
    Observation<uint64_t> events_max_delta = delta_observation(
            state, CounterEventsMax, events_max, events_source + ":max");
    Observation<uint64_t> events_oom_delta = delta_observation(
            state, CounterEventsOom, events_oom, events_source + ":oom");
    Observation<uint64_t> events_oom_kill_delta = delta_observation(
            state, CounterEventsOomKill, events_oom_kill, events_source + ":oom_kill");
    Observation<uint64_t> events_oom_group_kill_delta = delta_observation(
            state,
            CounterEventsOomGroupKill,
            events_oom_group_kill,
            events_source + ":oom_group_kill");

    std::unordered_map<std::string, uint64_t> stat;
    const uint64_t stat_parse_begin_cost = parse_cost_ns;
    if (stat_read.status == Status::Available) {
        const uint64_t begin = monotonic_ns();
        if (!parse_key_value(stat_read.text, stat)) {
            stat.clear();
            stat_read.status = Status::ParseError;
            stat_read.error = "invalid key/value format";
        }
        parse_cost_ns += monotonic_ns() - begin;
    }
    const uint64_t stat_parse_cost_ns = parse_cost_ns - stat_parse_begin_cost;
    Observation<uint64_t> anon = map_observation(stat_read, stat, "anon");
    Observation<uint64_t> file = map_observation(stat_read, stat, "file");
    Observation<uint64_t> refault_anon = map_observation(stat_read, stat, "workingset_refault_anon");
    Observation<uint64_t> refault_file = map_observation(stat_read, stat, "workingset_refault_file");
    Observation<uint64_t> activate_anon =
            map_observation(stat_read, stat, "workingset_activate_anon");
    Observation<uint64_t> activate_file =
            map_observation(stat_read, stat, "workingset_activate_file");
    Observation<uint64_t> restore_anon =
            map_observation(stat_read, stat, "workingset_restore_anon");
    Observation<uint64_t> restore_file =
            map_observation(stat_read, stat, "workingset_restore_file");
    Observation<uint64_t> cgroup_pgfault = map_observation(stat_read, stat, "pgfault");
    Observation<uint64_t> cgroup_pgmajfault = map_observation(stat_read, stat, "pgmajfault");
    Observation<uint64_t> pgscan = map_observation(stat_read, stat, "pgscan");
    Observation<uint64_t> pgsteal = map_observation(stat_read, stat, "pgsteal");
    Observation<uint64_t> pgrefill = map_observation(stat_read, stat, "pgrefill");
    Observation<uint64_t> pgactivate = map_observation(stat_read, stat, "pgactivate");
    Observation<uint64_t> pgdeactivate = map_observation(stat_read, stat, "pgdeactivate");
    const std::string stat_source = state.cgroup_path + "/memory.stat";
    Observation<uint64_t> refault_anon_delta = delta_observation(
            state,
            CounterRefaultAnon,
            refault_anon,
            stat_source + ":workingset_refault_anon");
    Observation<uint64_t> refault_file_delta = delta_observation(
            state,
            CounterRefaultFile,
            refault_file,
            stat_source + ":workingset_refault_file");
    Observation<uint64_t> activate_anon_delta = delta_observation(
            state,
            CounterActivateAnon,
            activate_anon,
            stat_source + ":workingset_activate_anon");
    Observation<uint64_t> activate_file_delta = delta_observation(
            state,
            CounterActivateFile,
            activate_file,
            stat_source + ":workingset_activate_file");
    Observation<uint64_t> restore_anon_delta = delta_observation(
            state,
            CounterRestoreAnon,
            restore_anon,
            stat_source + ":workingset_restore_anon");
    Observation<uint64_t> restore_file_delta = delta_observation(
            state,
            CounterRestoreFile,
            restore_file,
            stat_source + ":workingset_restore_file");
    Observation<uint64_t> cgroup_pgfault_delta = delta_observation(
            state, CounterCgroupPgfault, cgroup_pgfault, stat_source + ":pgfault");
    Observation<uint64_t> cgroup_pgmajfault_delta = delta_observation(
            state,
            CounterCgroupPgmajfault,
            cgroup_pgmajfault,
            stat_source + ":pgmajfault");
    Observation<uint64_t> pgscan_delta = delta_observation(
            state, CounterPgscan, pgscan, stat_source + ":pgscan");
    Observation<uint64_t> pgsteal_delta = delta_observation(
            state, CounterPgsteal, pgsteal, stat_source + ":pgsteal");
    Observation<uint64_t> pgrefill_delta = delta_observation(
            state, CounterPgrefill, pgrefill, stat_source + ":pgrefill");
    Observation<uint64_t> pgactivate_delta = delta_observation(
            state, CounterPgactivate, pgactivate, stat_source + ":pgactivate");
    Observation<uint64_t> pgdeactivate_delta = delta_observation(
            state, CounterPgdeactivate, pgdeactivate, stat_source + ":pgdeactivate");

    PsiValues psi{};
    bool psi_ok = false;
    const uint64_t psi_parse_begin_cost = parse_cost_ns;
    if (pressure_read.status == Status::Available) {
        const uint64_t begin = monotonic_ns();
        psi_ok = parse_psi(pressure_read.text, psi);
        parse_cost_ns += monotonic_ns() - begin;
    }
    const uint64_t psi_parse_cost_ns = parse_cost_ns - psi_parse_begin_cost;
    Observation<double> psi_some_avg10;
    Observation<double> psi_full_avg10;
    Observation<uint64_t> psi_some_total;
    Observation<uint64_t> psi_full_total;
    if (pressure_read.status != Status::Available) {
        psi_some_avg10 = failed_observation<double>(pressure_read);
        psi_full_avg10 = failed_observation<double>(pressure_read);
        psi_some_total = failed_observation<uint64_t>(pressure_read);
        psi_full_total = failed_observation<uint64_t>(pressure_read);
    } else if (!psi_ok) {
        psi_some_avg10.status = psi_full_avg10.status = Status::ParseError;
        psi_some_total.status = psi_full_total.status = Status::ParseError;
        psi_some_avg10.read_ts_ns = psi_full_avg10.read_ts_ns = pressure_read.read_ts_ns;
        psi_some_total.read_ts_ns = psi_full_total.read_ts_ns = pressure_read.read_ts_ns;
        psi_some_avg10.error = psi_full_avg10.error = "invalid PSI format";
        psi_some_total.error = psi_full_total.error = "invalid PSI format";
    } else {
        psi_some_avg10 = available_double(psi.some_avg10, pressure_read.read_ts_ns);
        psi_full_avg10 = available_double(psi.full_avg10, pressure_read.read_ts_ns);
        psi_some_total = available_uint64(psi.some_total_us, pressure_read.read_ts_ns);
        psi_full_total = available_uint64(psi.full_total_us, pressure_read.read_ts_ns);
    }

    const std::string psi_source = state.cgroup_path + "/memory.pressure";
    Observation<uint64_t> psi_some_delta = delta_observation(
            state, CounterPsiSome, psi_some_total, psi_source + ":some.total_us");
    Observation<uint64_t> psi_full_delta = delta_observation(
            state, CounterPsiFull, psi_full_total, psi_source + ":full.total_us");

    ProcStatValues proc{};
    bool proc_ok = false;
    const uint64_t proc_parse_begin_cost = parse_cost_ns;
    if (proc_read.status == Status::Available) {
        const uint64_t begin = monotonic_ns();
        proc_ok = parse_proc_stat(proc_read.text, proc);
        parse_cost_ns += monotonic_ns() - begin;
    }
    const uint64_t proc_parse_cost_ns = parse_cost_ns - proc_parse_begin_cost;
    Observation<uint64_t> process_rss;
    Observation<uint64_t> process_vms;
    Observation<uint64_t> process_minor_faults;
    Observation<uint64_t> process_major_faults;
    if (proc_read.status != Status::Available) {
        process_rss = failed_observation<uint64_t>(proc_read);
        process_vms = failed_observation<uint64_t>(proc_read);
        process_minor_faults = failed_observation<uint64_t>(proc_read);
        process_major_faults = failed_observation<uint64_t>(proc_read);
    } else if (!proc_ok) {
        process_rss.status = process_vms.status = process_minor_faults.status =
                process_major_faults.status = Status::ParseError;
        process_rss.read_ts_ns = process_vms.read_ts_ns = process_minor_faults.read_ts_ns =
                process_major_faults.read_ts_ns = proc_read.read_ts_ns;
        process_rss.error = process_vms.error = process_minor_faults.error =
                process_major_faults.error = "invalid /proc/self/stat format";
    } else {
        const long page_size = sysconf(_SC_PAGESIZE);
        process_rss = available_uint64(
                proc.rss_pages * (uint64_t) (page_size > 0 ? page_size : 4096),
                proc_read.read_ts_ns);
        process_vms = available_uint64(proc.vsize_bytes, proc_read.read_ts_ns);
        process_minor_faults = available_uint64(proc.minor_faults, proc_read.read_ts_ns);
        process_major_faults = available_uint64(proc.major_faults, proc_read.read_ts_ns);
    }
    Observation<uint64_t> process_minor_faults_delta = delta_observation(
            state,
            CounterProcessMinorFaults,
            process_minor_faults,
            "/proc/self/stat:minor_faults");
    Observation<uint64_t> process_major_faults_delta = delta_observation(
            state,
            CounterProcessMajorFaults,
            process_major_faults,
            "/proc/self/stat:major_faults");

    SmapsSnapshot process_smaps;
    {
        std::lock_guard<std::mutex> lock(state.pss_mu);
        process_smaps = state.latest_smaps;
    }
    const auto mark_stale = [&](Observation<uint64_t> & field) {
        if (field.status == Status::Available && field.read_ts_ns > 0 &&
                wall_begin > field.read_ts_ns &&
                wall_begin - field.read_ts_ns > 2 * state.pss_interval_ns) {
            field.status = Status::SourceStale;
            field.has_value = false;
            field.error = "smaps_rollup sample exceeded freshness limit";
        }
    };
    mark_stale(process_smaps.rss);
    mark_stale(process_smaps.pss);
    mark_stale(process_smaps.swap);
    Observation<uint64_t> process_pss_age;
    if (
        process_smaps.pss.read_ts_ns > 0
        && wall_begin >= process_smaps.pss.read_ts_ns
        && process_smaps.pss.status == Status::Available
    ) {
        process_pss_age = available_uint64(
                wall_begin - process_smaps.pss.read_ts_ns, wall_begin);
    } else {
        process_pss_age.status = process_smaps.pss.status;
        process_pss_age.read_ts_ns = wall_begin;
        process_pss_age.error = process_smaps.pss.error.empty() ?
                "PSS age unavailable" : process_smaps.pss.error;
    }

    const uint64_t queue_begin = monotonic_ns();
    QueueSnapshot queue;
    const bool have_queue_provider =
            state.queue_provider.load(std::memory_order_acquire) != nullptr;
    if (QueueSnapshotProvider provider = state.queue_provider.load(std::memory_order_acquire)) {
        queue = provider();
    }
    const uint64_t queue_ready_ts_ns = monotonic_ns();
    const uint64_t queue_cost_ns = queue_ready_ts_ns - queue_begin;

    const int phase = llm_mem_trace_get_phase();
    const uint64_t step = llm_mem_trace_get_step();
    StepCountersSnapshot step_counters;
    {
        std::lock_guard<std::mutex> lock(state.counters_mu);
        const StepSlot & slot = state.step_slots[step % STEP_SLOT_COUNT];
        if (slot.step.load(std::memory_order_acquire) == step) {
            step_counters.step = step;
            step_counters.issued_bytes =
                    slot.issued_bytes.load(std::memory_order_relaxed);
            step_counters.hint_calls =
                    slot.hint_calls.load(std::memory_order_relaxed);
            step_counters.advised_bytes =
                    slot.advised_bytes.load(std::memory_order_relaxed);
        }
    }

    const uint64_t last_step = state.last_step.load(std::memory_order_acquire);
    const uint64_t last_step_end = state.last_step_end_ts_ns.load(std::memory_order_acquire);
    const uint64_t last_step_latency = state.last_step_latency_ns.load(std::memory_order_acquire);
    const int last_step_phase = state.last_step_phase.load(std::memory_order_acquire);

    const bool queue_values_available =
            queue.status == Status::Available || queue.status == Status::Stopping;
    const auto queue_observation = [&](uint64_t value) {
        if (queue_values_available) {
            return available_uint64(value, queue_ready_ts_ns);
        }
        Observation<uint64_t> unavailable;
        unavailable.status = queue.status;
        unavailable.read_ts_ns = queue_ready_ts_ns;
        unavailable.error = "queue snapshot is not active";
        return unavailable;
    };
    const Observation<uint64_t> queue_depth = queue_observation(queue.queue_depth);
    const Observation<uint64_t> queued_bytes = queue_observation(queue.queued_bytes);
    const Observation<uint64_t> worker_count = queue_observation(queue.worker_count);
    const Observation<uint64_t> busy_workers = queue_observation(queue.busy_workers);
    Observation<uint64_t> configured_worker_count;
    if (have_queue_provider) {
        configured_worker_count =
                available_uint64(queue.configured_worker_count, queue_ready_ts_ns);
    } else {
        configured_worker_count.status = Status::Unavailable;
        configured_worker_count.read_ts_ns = queue_ready_ts_ns;
        configured_worker_count.error = "queue snapshot provider unavailable";
    }
    const Observation<uint64_t> current_step_issued =
            available_uint64(step_counters.issued_bytes, queue_ready_ts_ns);
    const Observation<uint64_t> current_step_hint_calls =
            available_uint64(step_counters.hint_calls, queue_ready_ts_ns);
    const Observation<uint64_t> current_step_advised =
            available_uint64(step_counters.advised_bytes, queue_ready_ts_ns);
    Observation<uint64_t> queue_started;
    Observation<uint64_t> queue_stopping;
    if (have_queue_provider) {
        queue_started = available_uint64(queue.started ? 1 : 0, queue_ready_ts_ns);
        queue_stopping = available_uint64(queue.stopping ? 1 : 0, queue_ready_ts_ns);
    } else {
        queue_started.status = queue_stopping.status = Status::Unavailable;
        queue_started.read_ts_ns = queue_stopping.read_ts_ns = queue_ready_ts_ns;
        queue_started.error = queue_stopping.error =
                "queue snapshot provider unavailable";
    }

    const char * phase_text =
            phase == LLM_MEM_TRACE_PHASE_PREFILL ? "PREFILL" :
            phase == LLM_MEM_TRACE_PHASE_DECODE ? "DECODE" : "UNKNOWN";
    const char * last_phase_text =
            last_step_phase == LLM_MEM_TRACE_PHASE_PREFILL ? "PREFILL" :
            last_step_phase == LLM_MEM_TRACE_PHASE_DECODE ? "DECODE" : "UNKNOWN";

    const uint64_t sample_ready_ts_ns = monotonic_ns();
    const uint64_t cpu_ready = thread_cpu_ns();
    const uint64_t wall_cost_ns = sample_ready_ts_ns - wall_begin;
    const uint64_t cpu_cost_ns = cpu_ready >= cpu_begin ? cpu_ready - cpu_begin : 0;
    const bool have_actual_interval = state.previous_sample_ready_ts_ns != 0 &&
            sample_ready_ts_ns >= state.previous_sample_ready_ts_ns;
    const uint64_t actual_interval_ns = have_actual_interval ?
            sample_ready_ts_ns - state.previous_sample_ready_ts_ns : 0;
    if (have_actual_interval) {
        state.actual_intervals.add(actual_interval_ns);
    }
    state.previous_sample_ready_ts_ns = sample_ready_ts_ns;

    const bool memory_state_available = memory_current.status == Status::Available &&
            memory_current.has_value &&
            ((memory_high.status == Status::Available && memory_high.has_value) ||
             (memory_max.status == Status::Available && memory_max.has_value));
    const bool stall_state_available =
            psi_some_delta.status == Status::Available && psi_some_delta.has_value;
    const bool queue_state_available =
            queue_depth.status == Status::Available && queue_depth.has_value;
    if (memory_state_available && stall_state_available && queue_state_available) {
        state.eligible_sample_count++;
    }
    if (memory_current.has_value && memory_high.has_value && memory_high.value > 0) {
        const long double ratio =
                (long double) memory_current.value / (long double) memory_high.value;
        state.memory_high_90_crossings += ratio >= 0.90L;
        state.memory_high_98_crossings += ratio >= 0.98L;
    }
    state.psi_some_positive_crossings +=
            psi_some_delta.has_value && psi_some_delta.value > 0;
    state.psi_full_positive_crossings +=
            psi_full_delta.has_value && psi_full_delta.value > 0;
    state.queue_nonempty_crossings += queue_depth.has_value && queue_depth.value > 0;

    add_aggregate(state, "memory_current_bytes", memory_current);
    add_aggregate(state, "memory_high_bytes", memory_high);
    add_aggregate(state, "memory_max_bytes", memory_max);
    add_aggregate(state, "swap_current_bytes", swap_current);
    add_aggregate(state, "memory_events_low", events_low);
    add_aggregate(state, "memory_events_high", events_high);
    add_aggregate(state, "memory_events_max", events_max);
    add_aggregate(state, "memory_events_oom", events_oom);
    add_aggregate(state, "memory_events_oom_kill", events_oom_kill);
    add_aggregate(state, "memory_events_oom_group_kill", events_oom_group_kill);
    add_aggregate(state, "memory_events_low_delta", events_low_delta);
    add_aggregate(state, "memory_events_high_delta", events_high_delta);
    add_aggregate(state, "memory_events_max_delta", events_max_delta);
    add_aggregate(state, "memory_events_oom_delta", events_oom_delta);
    add_aggregate(state, "memory_events_oom_kill_delta", events_oom_kill_delta);
    add_aggregate(
            state, "memory_events_oom_group_kill_delta", events_oom_group_kill_delta);
    add_aggregate(state, "anon_bytes", anon);
    add_aggregate(state, "file_bytes", file);
    add_aggregate(state, "workingset_refault_anon", refault_anon);
    add_aggregate(state, "workingset_refault_file", refault_file);
    add_aggregate(state, "workingset_activate_anon", activate_anon);
    add_aggregate(state, "workingset_activate_file", activate_file);
    add_aggregate(state, "workingset_restore_anon", restore_anon);
    add_aggregate(state, "workingset_restore_file", restore_file);
    add_aggregate(state, "cgroup_pgfault", cgroup_pgfault);
    add_aggregate(state, "cgroup_pgmajfault", cgroup_pgmajfault);
    add_aggregate(state, "pgscan", pgscan);
    add_aggregate(state, "pgsteal", pgsteal);
    add_aggregate(state, "pgrefill", pgrefill);
    add_aggregate(state, "pgactivate", pgactivate);
    add_aggregate(state, "pgdeactivate", pgdeactivate);
    add_aggregate(state, "workingset_refault_anon_delta", refault_anon_delta);
    add_aggregate(state, "workingset_refault_file_delta", refault_file_delta);
    add_aggregate(state, "workingset_activate_anon_delta", activate_anon_delta);
    add_aggregate(state, "workingset_activate_file_delta", activate_file_delta);
    add_aggregate(state, "workingset_restore_anon_delta", restore_anon_delta);
    add_aggregate(state, "workingset_restore_file_delta", restore_file_delta);
    add_aggregate(state, "cgroup_pgfault_delta", cgroup_pgfault_delta);
    add_aggregate(state, "cgroup_pgmajfault_delta", cgroup_pgmajfault_delta);
    add_aggregate(state, "pgscan_delta", pgscan_delta);
    add_aggregate(state, "pgsteal_delta", pgsteal_delta);
    add_aggregate(state, "pgrefill_delta", pgrefill_delta);
    add_aggregate(state, "pgactivate_delta", pgactivate_delta);
    add_aggregate(state, "pgdeactivate_delta", pgdeactivate_delta);
    add_aggregate(state, "psi_some_avg10", psi_some_avg10);
    add_aggregate(state, "psi_full_avg10", psi_full_avg10);
    add_aggregate(state, "psi_some_delta_us", psi_some_delta);
    add_aggregate(state, "psi_full_delta_us", psi_full_delta);
    add_aggregate(state, "process_rss_bytes", process_rss);
    add_aggregate(state, "process_vms_bytes", process_vms);
    add_aggregate(state, "process_smaps_rss_bytes", process_smaps.rss);
    add_aggregate(state, "process_pss_bytes", process_smaps.pss);
    add_aggregate(state, "process_smaps_swap_bytes", process_smaps.swap);
    add_aggregate(state, "process_minor_faults", process_minor_faults);
    add_aggregate(state, "process_major_faults", process_major_faults);
    add_aggregate(state, "process_minor_faults_delta", process_minor_faults_delta);
    add_aggregate(state, "process_major_faults_delta", process_major_faults_delta);
    add_aggregate(state, "queue_depth", queue_depth);
    add_aggregate(state, "queued_bytes", queued_bytes);
    add_aggregate(state, "busy_workers", busy_workers);
    add_aggregate(state, "configured_worker_count", configured_worker_count);
    add_aggregate(state, "current_step_issued_bytes", current_step_issued);
    add_aggregate(state, "current_step_hint_calls", current_step_hint_calls);
    add_aggregate(
            state,
            "source_memory_current_io_cost_ns",
            available_uint64(memory_current_read.io_cost_ns, memory_current_read.read_ts_ns));
    add_aggregate(
            state,
            "source_memory_high_io_cost_ns",
            available_uint64(memory_high_read.io_cost_ns, memory_high_read.read_ts_ns));
    add_aggregate(
            state,
            "source_memory_max_io_cost_ns",
            available_uint64(memory_max_read.io_cost_ns, memory_max_read.read_ts_ns));
    add_aggregate(
            state,
            "source_swap_current_io_cost_ns",
            available_uint64(swap_current_read.io_cost_ns, swap_current_read.read_ts_ns));
    add_aggregate(
            state,
            "source_swap_max_io_cost_ns",
            available_uint64(swap_max_read.io_cost_ns, swap_max_read.read_ts_ns));
    add_aggregate(
            state,
            "source_memory_events_io_cost_ns",
            available_uint64(events_read.io_cost_ns, events_read.read_ts_ns));
    add_aggregate(
            state,
            "source_memory_stat_io_cost_ns",
            available_uint64(stat_read.io_cost_ns, stat_read.read_ts_ns));
    add_aggregate(
            state,
            "source_memory_pressure_io_cost_ns",
            available_uint64(pressure_read.io_cost_ns, pressure_read.read_ts_ns));
    add_aggregate(
            state,
            "source_process_stat_io_cost_ns",
            available_uint64(proc_read.io_cost_ns, proc_read.read_ts_ns));
    add_aggregate(
            state,
            "parse_cgroup_scalar_cost_ns",
            available_uint64(scalar_parse_cost_ns, sample_ready_ts_ns));
    add_aggregate(
            state,
            "parse_memory_events_cost_ns",
            available_uint64(events_parse_cost_ns, sample_ready_ts_ns));
    add_aggregate(
            state,
            "parse_memory_stat_cost_ns",
            available_uint64(stat_parse_cost_ns, sample_ready_ts_ns));
    add_aggregate(
            state,
            "parse_memory_pressure_cost_ns",
            available_uint64(psi_parse_cost_ns, sample_ready_ts_ns));
    add_aggregate(
            state,
            "parse_process_stat_cost_ns",
            available_uint64(proc_parse_cost_ns, sample_ready_ts_ns));
    {
        std::lock_guard<std::mutex> lock(state.latest_mu);
        state.latest.sequence = state.sample_count + 1;
        state.latest.sample_ready_ts_ns = sample_ready_ts_ns;
        state.latest.phase = phase;
        state.latest.step = step;
        state.latest.memory_current_bytes = latest_value(memory_current);
        state.latest.memory_high_bytes = latest_value(memory_high);
        state.latest.memory_max_bytes = latest_value(memory_max);
        state.latest.psi_some_total_us = latest_value(psi_some_total);
        state.latest.psi_full_total_us = latest_value(psi_full_total);
        state.latest.queue_depth = latest_value(queue_depth);
        state.latest.queued_bytes = latest_value(queued_bytes);
        state.latest.busy_workers = latest_value(busy_workers);
        state.have_latest = true;
    }

    state.sample_count++;
    state.total_io_cost_ns += io_cost_ns;
    state.total_parse_cost_ns += parse_cost_ns;
    state.total_queue_cost_ns += queue_cost_ns;
    state.max_wall_cost_ns = std::max(state.max_wall_cost_ns, wall_cost_ns);
    state.max_io_cost_ns = std::max(state.max_io_cost_ns, io_cost_ns);
    state.max_parse_cost_ns = std::max(state.max_parse_cost_ns, parse_cost_ns);
    state.max_queue_cost_ns = std::max(state.max_queue_cost_ns, queue_cost_ns);
    state.max_jitter_ns = std::max(state.max_jitter_ns, jitter_ns);
    state.jitter_distribution.add(jitter_ns);

    if (mode() == Mode::Detail) {
        const uint64_t serialize_begin = monotonic_ns();
        std::string line;
        line.reserve(8192);
        line += "{\"event\":\"PRESSURE_SHADOW_SAMPLE\",\"schema_version\":1";
        line += ",\"run_id\":" + json_escaped(state.run_id);
        line += ",\"sample_seq\":" + std::to_string(state.sample_count);
        line += ",\"scheduled_ts_ns\":" + std::to_string(scheduled_ts_ns);
        line += ",\"sample_start_ts_ns\":" + std::to_string(wall_begin);
        line += ",\"sample_begin_ts_ns\":" + std::to_string(wall_begin);
        line += ",\"sample_ready_ts_ns\":" + std::to_string(sample_ready_ts_ns);
        line += ",\"ts_ns\":" + std::to_string(sample_ready_ts_ns);
        line += ",\"sample_index\":" + std::to_string(state.sample_count - 1);
        line += ",\"target_interval_ns\":" + std::to_string(state.interval_ns);
        line += ",\"actual_interval_ns\":" +
                (have_actual_interval ?
                 std::to_string(actual_interval_ns) : std::string("null"));
        line += ",\"deadline_lateness_ns\":" + std::to_string(jitter_ns);
        line += ",\"missed_samples_since_previous\":" +
                std::to_string(missed_before_sample);
        line += ",\"missed_intervals_before_sample\":" + std::to_string(missed_before_sample);
        line += ",\"sample_wall_time_ns\":" + std::to_string(wall_cost_ns);
        line += ",\"sample_thread_cpu_time_ns\":" + std::to_string(cpu_cost_ns);
        line += ",\"trace_context_snapshot_ts_ns\":" +
                std::to_string(queue_ready_ts_ns);
        line += ",\"sources\":{";
        append_source_json(
                line,
                "cgroup_memory",
                "current_process_cgroup",
                state.cgroup_path,
                memory_current_read.status,
                memory_current_read.error_number,
                memory_current_read.error);
        line += ",";
        append_source_json(
                line,
                "cgroup_psi",
                "current_process_cgroup",
                state.cgroup_path.empty() ?
                        std::string() : state.cgroup_path + "/memory.pressure",
                pressure_read.status,
                pressure_read.error_number,
                pressure_read.error);
        line += ",";
        append_source_json(
                line,
                "process_stat",
                "current_process",
                "/proc/self/stat",
                proc_read.status,
                proc_read.error_number,
                proc_read.error);
        line += ",";
        append_source_json(
                line,
                "process_smaps_rollup",
                "current_process",
                "/proc/self/smaps_rollup",
                process_smaps.pss.status,
                0,
                process_smaps.pss.error);
        line += ",";
        append_source_json(
                line,
                "expert_queue",
                "current_process_expert_queue",
                "",
                queue.status,
                0,
                queue_values_available ? std::string() : "queue snapshot is not active");
        line += "}";
        append_observation(line, "sampler_jitter_ns", state, available_uint64(jitter_ns, wall_begin));
        append_observation(line, "sampler_wall_cost_ns", state, available_uint64(wall_cost_ns, sample_ready_ts_ns));
        append_observation(line, "sampler_cpu_cost_ns", state, available_uint64(cpu_cost_ns, sample_ready_ts_ns));
        append_observation(line, "sampler_io_cost_ns", state, available_uint64(io_cost_ns, sample_ready_ts_ns));
        append_observation(line, "sampler_parse_cost_ns", state, available_uint64(parse_cost_ns, sample_ready_ts_ns));
        append_observation(line, "sampler_queue_snapshot_cost_ns", state, available_uint64(queue_cost_ns, queue_ready_ts_ns));
        append_string_observation(line, "phase", state, phase_text, Status::Available, queue_ready_ts_ns);
        append_observation(line, "step", state, available_uint64(step, queue_ready_ts_ns));
        append_observation(line, "memory_current_bytes", state, memory_current);
        append_observation(line, "memory_high_bytes", state, memory_high);
        append_observation(line, "memory_max_bytes", state, memory_max);
        append_observation(line, "swap_current_bytes", state, swap_current);
        append_observation(line, "swap_max_bytes", state, swap_max);
        append_observation(line, "memory_events_low", state, events_low);
        append_observation(line, "memory_events_high", state, events_high);
        append_observation(line, "memory_events_max", state, events_max);
        append_observation(line, "memory_events_oom", state, events_oom);
        append_observation(line, "memory_events_oom_kill", state, events_oom_kill);
        append_observation(
                line, "memory_events_oom_group_kill", state, events_oom_group_kill);
        append_observation(line, "memory_events_low_delta", state, events_low_delta);
        append_observation(line, "memory_events_high_delta", state, events_high_delta);
        append_observation(line, "memory_events_max_delta", state, events_max_delta);
        append_observation(line, "memory_events_oom_delta", state, events_oom_delta);
        append_observation(
                line, "memory_events_oom_kill_delta", state, events_oom_kill_delta);
        append_observation(
                line,
                "memory_events_oom_group_kill_delta",
                state,
                events_oom_group_kill_delta);
        append_observation(line, "anon_bytes", state, anon);
        append_observation(line, "file_bytes", state, file);
        append_observation(line, "workingset_refault_anon", state, refault_anon);
        append_observation(line, "workingset_refault_file", state, refault_file);
        append_observation(
                line, "workingset_activate_anon", state, activate_anon);
        append_observation(
                line, "workingset_activate_file", state, activate_file);
        append_observation(
                line, "workingset_restore_anon", state, restore_anon);
        append_observation(
                line, "workingset_restore_file", state, restore_file);
        append_observation(line, "cgroup_pgfault", state, cgroup_pgfault);
        append_observation(line, "cgroup_pgmajfault", state, cgroup_pgmajfault);
        append_observation(line, "pgscan", state, pgscan);
        append_observation(line, "pgsteal", state, pgsteal);
        append_observation(line, "pgrefill", state, pgrefill);
        append_observation(line, "pgactivate", state, pgactivate);
        append_observation(line, "pgdeactivate", state, pgdeactivate);
        append_observation(
                line, "workingset_refault_anon_delta", state, refault_anon_delta);
        append_observation(
                line, "workingset_refault_file_delta", state, refault_file_delta);
        append_observation(
                line, "workingset_activate_anon_delta", state, activate_anon_delta);
        append_observation(
                line, "workingset_activate_file_delta", state, activate_file_delta);
        append_observation(
                line, "workingset_restore_anon_delta", state, restore_anon_delta);
        append_observation(
                line, "workingset_restore_file_delta", state, restore_file_delta);
        append_observation(line, "cgroup_pgfault_delta", state, cgroup_pgfault_delta);
        append_observation(
                line, "cgroup_pgmajfault_delta", state, cgroup_pgmajfault_delta);
        append_observation(line, "pgscan_delta", state, pgscan_delta);
        append_observation(line, "pgsteal_delta", state, pgsteal_delta);
        append_observation(line, "pgrefill_delta", state, pgrefill_delta);
        append_observation(line, "pgactivate_delta", state, pgactivate_delta);
        append_observation(line, "pgdeactivate_delta", state, pgdeactivate_delta);
        append_observation(line, "psi_some_avg10", state, psi_some_avg10);
        append_observation(line, "psi_full_avg10", state, psi_full_avg10);
        append_observation(line, "psi_some_total_us", state, psi_some_total);
        append_observation(line, "psi_full_total_us", state, psi_full_total);
        append_observation(line, "psi_some_delta_us", state, psi_some_delta);
        append_observation(line, "psi_full_delta_us", state, psi_full_delta);
        append_observation(line, "process_rss_bytes", state, process_rss);
        append_observation(line, "process_vms_bytes", state, process_vms);
        append_observation(line, "process_smaps_rss_bytes", state, process_smaps.rss);
        append_observation(line, "process_pss_bytes", state, process_smaps.pss);
        append_observation(line, "process_smaps_swap_bytes", state, process_smaps.swap);
        append_observation(line, "process_pss_age_ns", state, process_pss_age);
        append_observation(line, "process_minor_faults", state, process_minor_faults);
        append_observation(line, "process_major_faults", state, process_major_faults);
        append_observation(
                line, "process_minor_faults_delta", state, process_minor_faults_delta);
        append_observation(
                line, "process_major_faults_delta", state, process_major_faults_delta);
        append_observation(line, "queue_depth", state, queue_depth);
        append_observation(line, "queued_bytes", state, queued_bytes);
        append_string_observation(
                line,
                "queue_status",
                state,
                status_name(queue.status),
                Status::Available,
                queue_ready_ts_ns);
        append_observation(line, "queue_started", state, queue_started);
        append_observation(line, "queue_stopping", state, queue_stopping);
        append_observation(
                line, "configured_worker_count", state, configured_worker_count);
        append_observation(line, "worker_count", state, worker_count);
        append_observation(line, "busy_workers", state, busy_workers);
        append_observation(line, "current_step_issued_bytes", state, current_step_issued);
        append_observation(line, "current_step_hint_calls", state, current_step_hint_calls);
        append_observation(line, "current_step_advised_bytes", state, current_step_advised);
        append_observation(
                line,
                "total_issued_bytes",
                state,
                available_uint64(state.total_issued_bytes.load(std::memory_order_relaxed), queue_ready_ts_ns));
        append_observation(
                line,
                "total_hint_calls",
                state,
                available_uint64(state.total_hint_calls.load(std::memory_order_relaxed), queue_ready_ts_ns));
        append_string_observation(line, "latest_completed_step_phase", state, last_phase_text,
                                  last_step == 0 ? Status::NotSampled : Status::Available, last_step_end,
                                  last_step == 0 ? "no completed step" : "");
        Observation<uint64_t> latest_step_id;
        Observation<uint64_t> latest_step_latency;
        if (last_step == 0) {
            latest_step_id.status = latest_step_latency.status = Status::NotSampled;
            latest_step_id.read_ts_ns = latest_step_latency.read_ts_ns = sample_ready_ts_ns;
            latest_step_id.error = latest_step_latency.error = "no completed step";
        } else {
            latest_step_id = available_uint64(last_step, last_step_end);
            latest_step_latency = available_uint64(last_step_latency, last_step_end);
        }
        append_observation(line, "latest_completed_step", state, latest_step_id);
        append_observation(line, "latest_completed_step_latency_ns", state, latest_step_latency);
        line += ",\"sampler_trace_enqueue_cost_ns\":{";
        line += "\"run_id\":" + json_escaped(state.run_id);
        line += ",\"status\":\"not_sampled\",\"read_ts_ns\":" + std::to_string(sample_ready_ts_ns);
        line += ",\"error\":\"available in PRESSURE_SHADOW_SUMMARY\",\"value\":null}";
        line += "}";
        const uint64_t serialize_cost_ns = monotonic_ns() - serialize_begin;
        state.total_serialize_cost_ns += serialize_cost_ns;
        state.max_serialize_cost_ns = std::max(state.max_serialize_cost_ns, serialize_cost_ns);
        const uint64_t trace_begin = monotonic_ns();
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
        const uint64_t trace_cost = monotonic_ns() - trace_begin;
        state.trace_enqueue_cost_ns += trace_cost;
        state.trace_enqueue_cost_max_ns = std::max(state.trace_enqueue_cost_max_ns, trace_cost);
        state.detail_events++;
    }
    const uint64_t complete_wall_cost_ns = monotonic_ns() - wall_begin;
    const uint64_t cpu_complete = thread_cpu_ns();
    const uint64_t complete_cpu_cost_ns =
            cpu_complete >= cpu_begin ? cpu_complete - cpu_begin : cpu_cost_ns;
    state.total_wall_cost_ns += complete_wall_cost_ns;
    state.total_cpu_cost_ns += complete_cpu_cost_ns;
    state.max_wall_cost_ns = std::max(state.max_wall_cost_ns, complete_wall_cost_ns);
    state.max_cpu_cost_ns = std::max(state.max_cpu_cost_ns, complete_cpu_cost_ns);
}

void append_aggregate_json(std::string & line, const std::string & name, const Aggregate & aggregate) {
    line += "\"" + name + "\":{";
    line += "\"observed\":" + std::to_string(aggregate.observed);
    line += ",\"available\":" + std::to_string(aggregate.available);
    line += ",\"unavailable\":" + std::to_string(aggregate.unavailable);
    if (aggregate.available > 0) {
        line += ",\"min\":" + std::to_string(aggregate.minimum);
        line += ",\"mean\":" + std::to_string((double) (aggregate.sum / aggregate.available));
        line += ",\"max\":" + std::to_string(aggregate.maximum);
    } else {
        line += ",\"min\":null,\"mean\":null,\"max\":null";
    }
    line += ",\"status_counts\":{";
    bool comma = false;
    for (const auto & entry : aggregate.status_counts) {
        if (comma) {
            line += ",";
        }
        line += json_escaped(status_name(entry.first)) + ":" +
                std::to_string(entry.second);
        comma = true;
    }
    line += "}";
    line += "}";
}

void write_summary(Runtime & state) {
    const uint64_t ts = monotonic_ns();
    std::string line;
    line.reserve(8192);
    line += "{\"event\":\"PRESSURE_SHADOW_SUMMARY\",\"schema_version\":1";
    line += ",\"ts_ns\":" + std::to_string(ts);
    line += ",\"run_id\":" + json_escaped(state.run_id);
    line += ",\"mode\":" + json_escaped(mode() == Mode::Detail ? "detail" : "summary");
    line += ",\"cgroup_path\":" + (state.cgroup_path.empty() ? std::string("null") : json_escaped(state.cgroup_path));
    line += ",\"sample_interval_ms\":" + std::to_string(state.interval_ns / NS_PER_MS);
    line += ",\"pss_interval_ms\":" + std::to_string(state.pss_interval_ns / NS_PER_MS);
    line += ",\"started_ts_ns\":" + std::to_string(state.started_ts_ns);
    line += ",\"stopped_ts_ns\":" + std::to_string(state.stopped_ts_ns);
    line += ",\"sample_count\":" + std::to_string(state.sample_count);
    line += ",\"eligible_sample_count\":" +
            std::to_string(state.eligible_sample_count);
    line += ",\"first_sample_ts_ns\":" +
            (state.sample_count > 0 ?
             std::to_string(state.started_ts_ns) : std::string("null"));
    line += ",\"last_sample_ready_ts_ns\":" +
            (state.previous_sample_ready_ts_ns > 0 ?
             std::to_string(state.previous_sample_ready_ts_ns) : std::string("null"));
    line += ",\"missed_intervals\":" + std::to_string(state.missed_intervals);
    line += ",\"detail_events\":" + std::to_string(state.detail_events);
    line += ",\"shutdown_status\":\"sampler_and_pss_joined\"";
    line += ",\"actual_interval_ns\":{";
    line += "\"count\":" + std::to_string(state.actual_intervals.count);
    if (state.actual_intervals.count > 0) {
        line += ",\"min\":" + std::to_string(state.actual_intervals.minimum);
        line += ",\"mean\":" +
                std::to_string((uint64_t) (
                        state.actual_intervals.sum / state.actual_intervals.count));
        line += ",\"p50\":" +
                std::to_string(state.actual_intervals.percentile(0.50));
        line += ",\"p95\":" +
                std::to_string(state.actual_intervals.percentile(0.95));
        line += ",\"max\":" + std::to_string(state.actual_intervals.maximum);
    } else {
        line += ",\"min\":null,\"mean\":null,\"p50\":null,\"p95\":null,\"max\":null";
    }
    line += ",\"histogram_bucket_width_ns\":" +
            std::to_string(INTERVAL_BUCKET_NS);
    line += ",\"histogram_overflow_floor_ns\":" +
            std::to_string((INTERVAL_BUCKET_COUNT - 1) * INTERVAL_BUCKET_NS);
    line += "}";
    line += ",\"deadline_lateness_ns\":{";
    line += "\"count\":" + std::to_string(state.jitter_distribution.count);
    if (state.jitter_distribution.count > 0) {
        line += ",\"min\":" + std::to_string(state.jitter_distribution.minimum);
        line += ",\"mean\":" +
                std::to_string((uint64_t) (
                        state.jitter_distribution.sum /
                        state.jitter_distribution.count));
        line += ",\"p50\":" +
                std::to_string(state.jitter_distribution.percentile(0.50));
        line += ",\"p95\":" +
                std::to_string(state.jitter_distribution.percentile(0.95));
        line += ",\"max\":" + std::to_string(state.jitter_distribution.maximum);
    } else {
        line += ",\"min\":null,\"mean\":null,\"p50\":null,\"p95\":null,\"max\":null";
    }
    line += "}";
    line += ",\"preregistered_signal_crossings\":{";
    line += "\"memory_current_over_high_0_90\":" +
            std::to_string(state.memory_high_90_crossings);
    line += ",\"memory_current_over_high_0_98\":" +
            std::to_string(state.memory_high_98_crossings);
    line += ",\"psi_some_delta_positive\":" +
            std::to_string(state.psi_some_positive_crossings);
    line += ",\"psi_full_delta_positive\":" +
            std::to_string(state.psi_full_positive_crossings);
    line += ",\"queue_depth_positive\":" +
            std::to_string(state.queue_nonempty_crossings);
    line += "}";
    line += ",\"state_status\":\"unavailable\"";
    line += ",\"state_error\":\"M5A observation only; candidates are computed offline\"";
    line += ",\"runtime_totals\":{";
    line += "\"issued_bytes\":" + std::to_string(state.total_issued_bytes.load(std::memory_order_relaxed));
    line += ",\"hint_calls\":" + std::to_string(state.total_hint_calls.load(std::memory_order_relaxed));
    line += ",\"advised_bytes\":" + std::to_string(state.total_advised_bytes.load(std::memory_order_relaxed));
    line += ",\"step_slot_overwrites\":" +
            std::to_string(state.step_slot_overwrites.load(std::memory_order_relaxed));
    line += ",\"step_slot_late_drops\":" +
            std::to_string(state.step_slot_late_drops.load(std::memory_order_relaxed));
    line += "}";
    const auto append_cost = [&](const char * name, uint64_t total, uint64_t maximum) {
        line += ",\"" + std::string(name) + "\":{";
        line += "\"total_ns\":" + std::to_string(total);
        line += ",\"mean_ns\":" +
                (state.sample_count > 0 ? std::to_string(total / state.sample_count) : std::string("null"));
        line += ",\"max_ns\":" + std::to_string(maximum);
        line += "}";
    };
    append_cost("sampler_wall_cost", state.total_wall_cost_ns, state.max_wall_cost_ns);
    append_cost("sampler_cpu_cost", state.total_cpu_cost_ns, state.max_cpu_cost_ns);
    append_cost("sampler_io_cost", state.total_io_cost_ns, state.max_io_cost_ns);
    append_cost("sampler_parse_cost", state.total_parse_cost_ns, state.max_parse_cost_ns);
    append_cost("queue_snapshot_cost", state.total_queue_cost_ns, state.max_queue_cost_ns);
    append_cost("serialization_cost", state.total_serialize_cost_ns, state.max_serialize_cost_ns);
    append_cost("trace_enqueue_cost", state.trace_enqueue_cost_ns, state.trace_enqueue_cost_max_ns);
    line += ",\"pss_sampler\":{";
    line += "\"sample_count\":" + std::to_string(state.pss_sample_count);
    line += ",\"missed_intervals\":" + std::to_string(state.pss_missed_intervals);
    line += ",\"wall_total_ns\":" + std::to_string(state.pss_total_wall_cost_ns);
    line += ",\"wall_mean_ns\":" +
            (state.pss_sample_count > 0 ?
             std::to_string(state.pss_total_wall_cost_ns / state.pss_sample_count) :
             std::string("null"));
    line += ",\"wall_max_ns\":" + std::to_string(state.pss_max_wall_cost_ns);
    line += ",\"cpu_total_ns\":" + std::to_string(state.pss_total_cpu_cost_ns);
    line += ",\"io_total_ns\":" + std::to_string(state.pss_total_io_cost_ns);
    line += ",\"parse_total_ns\":" + std::to_string(state.pss_total_parse_cost_ns);
    line += "}";
    const auto append_tuning = [&](const char * name, const ThreadTuning & tuning) {
        line += ",\"" + std::string(name) + "\":{";
        line += "\"nice_applied\":" + std::string(tuning.nice_applied ? "true" : "false");
        line += ",\"sched_idle_applied\":" +
                std::string(tuning.sched_idle_applied ? "true" : "false");
        line += ",\"affinity_applied\":" +
                std::string(tuning.affinity_applied ? "true" : "false");
        line += ",\"affinity_cpu\":" +
                (tuning.affinity_cpu >= 0 ?
                 std::to_string(tuning.affinity_cpu) : std::string("null"));
        line += "}";
    };
    append_tuning("sampler_thread_tuning", state.sampler_tuning);
    append_tuning("pss_thread_tuning", state.pss_tuning);
    line += ",\"max_jitter_ns\":" + std::to_string(state.max_jitter_ns);
    line += ",\"signals\":{";
    bool comma = false;
    for (const auto & entry : state.aggregates) {
        if (comma) {
            line += ",";
        }
        append_aggregate_json(line, entry.first, entry.second);
        comma = true;
    }
    line += "}}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void sampler_loop() {
    Runtime & state = runtime();
    state.sampler_tuning = configure_observer_thread(0);
    state.cgroup_path = resolve_cgroup_path();
    state.started_ts_ns = monotonic_ns();
    uint64_t scheduled = state.started_ts_ns;
    uint64_t missed_before_sample = 0;

    while (!state.stop_requested.load(std::memory_order_acquire)) {
        sample_once(state, scheduled, missed_before_sample);
        missed_before_sample = 0;
        const uint64_t now = monotonic_ns();
        const CadenceAdvance advance =
                advance_absolute_cadence(scheduled, state.interval_ns, now);
        scheduled = advance.next_scheduled_ts_ns;
        missed_before_sample = advance.missed_intervals;
        state.missed_intervals += advance.missed_intervals;

        std::unique_lock<std::mutex> lock(state.wait_mu);
        const auto deadline = std::chrono::steady_clock::time_point(
                std::chrono::nanoseconds(scheduled));
        state.wait_cv.wait_until(lock, deadline, [&] {
            return state.stop_requested.load(std::memory_order_acquire);
        });
    }
}

void pss_sampler_loop() {
    Runtime & state = runtime();
    state.pss_tuning = configure_observer_thread(1);
    uint64_t scheduled = monotonic_ns();
    while (!state.stop_requested.load(std::memory_order_acquire)) {
        const uint64_t wall_begin = monotonic_ns();
        const uint64_t cpu_begin = thread_cpu_ns();

        ReadResult smaps_read = read_file("/proc/self/smaps_rollup");
        uint64_t parse_cost_ns = 0;
        SmapsSnapshot process_smaps;
        if (smaps_read.status != Status::Available) {
            process_smaps.rss = failed_observation<uint64_t>(smaps_read);
            process_smaps.pss = failed_observation<uint64_t>(smaps_read);
            process_smaps.swap = failed_observation<uint64_t>(smaps_read);
        } else {
            SmapsValues smaps{};
            const uint64_t parse_begin = monotonic_ns();
            const bool smaps_ok = parse_smaps_rollup(smaps_read.text, smaps);
            parse_cost_ns = monotonic_ns() - parse_begin;
            if (smaps_ok) {
                process_smaps.rss =
                        available_uint64(smaps.rss_bytes, smaps_read.read_ts_ns);
                process_smaps.pss =
                        available_uint64(smaps.pss_bytes, smaps_read.read_ts_ns);
                if (smaps.swap_available) {
                    process_smaps.swap =
                            available_uint64(smaps.swap_bytes, smaps_read.read_ts_ns);
                } else {
                    process_smaps.swap.status = Status::FieldMissing;
                    process_smaps.swap.read_ts_ns = smaps_read.read_ts_ns;
                    process_smaps.swap.error = "Swap field missing";
                }
            } else {
                process_smaps.rss.status = process_smaps.pss.status =
                        process_smaps.swap.status = Status::ParseError;
                process_smaps.rss.read_ts_ns = process_smaps.pss.read_ts_ns =
                        process_smaps.swap.read_ts_ns = smaps_read.read_ts_ns;
                process_smaps.rss.error = process_smaps.pss.error =
                        process_smaps.swap.error =
                        "Rss or Pss field missing or invalid";
            }
        }
        {
            std::lock_guard<std::mutex> lock(state.pss_mu);
            state.latest_smaps = process_smaps;
        }
        const uint64_t wall_cost_ns = monotonic_ns() - wall_begin;
        const uint64_t cpu_end = thread_cpu_ns();
        const uint64_t cpu_cost_ns = cpu_end >= cpu_begin ? cpu_end - cpu_begin : 0;
        const CadenceAdvance advance =
                advance_absolute_cadence(scheduled, state.pss_interval_ns, monotonic_ns());
        scheduled = advance.next_scheduled_ts_ns;
        state.pss_sample_count++;
        state.pss_missed_intervals += advance.missed_intervals;
        state.pss_total_wall_cost_ns += wall_cost_ns;
        state.pss_total_cpu_cost_ns += cpu_cost_ns;
        state.pss_total_io_cost_ns += smaps_read.io_cost_ns;
        state.pss_total_parse_cost_ns += parse_cost_ns;
        state.pss_max_wall_cost_ns = std::max(state.pss_max_wall_cost_ns, wall_cost_ns);

        std::unique_lock<std::mutex> lock(state.wait_mu);
        const auto deadline = std::chrono::steady_clock::time_point(
                std::chrono::nanoseconds(scheduled));
        state.wait_cv.wait_until(lock, deadline, [&] {
            return state.stop_requested.load(std::memory_order_acquire);
        });
    }
}

void update_step_slot(
        Runtime & state,
        uint64_t step,
        uint64_t issued_bytes,
        uint64_t hint_calls,
        uint64_t advised_bytes) {
    StepSlot & slot = state.step_slots[step % STEP_SLOT_COUNT];
    for (;;) {
        uint64_t observed = slot.step.load(std::memory_order_acquire);
        if (observed != step) {
            std::lock_guard<std::mutex> lock(state.counters_mu);
            observed = slot.step.load(std::memory_order_relaxed);
            if (observed != step) {
                if (observed != std::numeric_limits<uint64_t>::max() &&
                        observed != std::numeric_limits<uint64_t>::max() - 1 &&
                        observed > step) {
                    state.step_slot_late_drops.fetch_add(
                            1, std::memory_order_relaxed);
                    return;
                }
                if (observed != std::numeric_limits<uint64_t>::max() &&
                        observed != std::numeric_limits<uint64_t>::max() - 1) {
                    state.step_slot_overwrites.fetch_add(
                            1, std::memory_order_relaxed);
                }
                slot.step.store(
                        std::numeric_limits<uint64_t>::max() - 1,
                        std::memory_order_release);
                while (slot.writers.load(std::memory_order_acquire) != 0) {
                    std::this_thread::yield();
                }
                slot.issued_bytes.store(0, std::memory_order_relaxed);
                slot.hint_calls.store(0, std::memory_order_relaxed);
                slot.advised_bytes.store(0, std::memory_order_relaxed);
                slot.step.store(step, std::memory_order_release);
            }
        }
        slot.writers.fetch_add(1, std::memory_order_acq_rel);
        if (slot.step.load(std::memory_order_acquire) != step) {
            slot.writers.fetch_sub(1, std::memory_order_release);
            continue;
        }
        if (issued_bytes > 0) {
            slot.issued_bytes.fetch_add(issued_bytes, std::memory_order_relaxed);
        }
        if (hint_calls > 0) {
            slot.hint_calls.fetch_add(hint_calls, std::memory_order_relaxed);
        }
        if (advised_bytes > 0) {
            slot.advised_bytes.fetch_add(advised_bytes, std::memory_order_relaxed);
        }
        slot.update_epoch.fetch_add(1, std::memory_order_release);
        slot.writers.fetch_sub(1, std::memory_order_release);
        return;
    }
}

} // namespace

const char * status_name(Status status) {
    switch (status) {
        case Status::Available:        return "available";
        case Status::NotSampled:       return "not_sampled";
        case Status::Unavailable:      return "unavailable";
        case Status::PermissionDenied: return "permission_denied";
        case Status::FieldMissing:     return "field_missing";
        case Status::ParseError:       return "parse_error";
        case Status::IoError:          return "io_error";
        case Status::Unsupported:      return "unsupported";
        case Status::NoPreviousSample: return "no_previous_sample";
        case Status::CounterRegression:return "counter_regression";
        case Status::SourceChanged:    return "source_changed";
        case Status::SourceStale:      return "source_stale";
        case Status::NotStarted:       return "not_started";
        case Status::Stopping:         return "stopping";
    }
    return "unavailable";
}

Mode parse_mode(const char * value, bool * valid) {
    const std::string text = value ? value : "";
    const bool ok = text.empty() || text == "off" || text == "summary" || text == "detail";
    if (valid) {
        *valid = ok;
    }
    if (text == "summary") {
        return Mode::Summary;
    }
    if (text == "detail") {
        return Mode::Detail;
    }
    return Mode::Off;
}

bool parse_uint64_scalar(const std::string & text, uint64_t & value, bool & unlimited) {
    unlimited = false;
    const std::vector<std::string> words = split_words(text);
    if (words.size() != 1) {
        return false;
    }
    if (words[0] == "max") {
        unlimited = true;
        value = 0;
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(words[0].c_str(), &end, 10);
    if (errno != 0 || end == words[0].c_str() || *end != '\0' || words[0][0] == '-') {
        return false;
    }
    value = (uint64_t) parsed;
    return true;
}

bool parse_key_value(const std::string & text, std::unordered_map<std::string, uint64_t> & values) {
    values.clear();
    std::istringstream lines(text);
    std::string line;
    bool any = false;
    while (std::getline(lines, line)) {
        const std::vector<std::string> words = split_words(line);
        if (words.empty()) {
            continue;
        }
        if (words.size() != 2) {
            values.clear();
            return false;
        }
        uint64_t value = 0;
        bool unlimited = false;
        if (!parse_uint64_scalar(words[1], value, unlimited) || unlimited) {
            values.clear();
            return false;
        }
        if (!values.emplace(words[0], value).second) {
            values.clear();
            return false;
        }
        any = true;
    }
    return any;
}

bool parse_psi(const std::string & text, PsiValues & values) {
    bool have_some = false;
    bool have_full = false;
    std::istringstream lines(text);
    for (std::string line; std::getline(lines, line);) {
        const std::vector<std::string> words = split_words(line);
        if (words.empty()) {
            continue;
        }
        if (words[0] != "some" && words[0] != "full") {
            return false;
        }
        double avg10 = 0.0;
        uint64_t total = 0;
        bool found_avg10 = false;
        bool found_total = false;
        for (size_t i = 1; i < words.size(); ++i) {
            const size_t equal = words[i].find('=');
            if (equal == std::string::npos) {
                return false;
            }
            const std::string key = words[i].substr(0, equal);
            const std::string raw = words[i].substr(equal + 1);
            if (key == "avg10") {
                char * end = nullptr;
                errno = 0;
                avg10 = std::strtod(raw.c_str(), &end);
                if (errno != 0 || end == raw.c_str() || *end != '\0' || !std::isfinite(avg10)) {
                    return false;
                }
                found_avg10 = true;
            } else if (key == "total") {
                bool unlimited = false;
                if (!parse_uint64_scalar(raw, total, unlimited) || unlimited) {
                    return false;
                }
                found_total = true;
            }
        }
        if (!found_avg10 || !found_total) {
            return false;
        }
        if (words[0] == "some") {
            values.some_avg10 = avg10;
            values.some_total_us = total;
            have_some = true;
        } else {
            values.full_avg10 = avg10;
            values.full_total_us = total;
            have_full = true;
        }
    }
    return have_some && have_full;
}

bool parse_proc_stat(const std::string & text, ProcStatValues & values) {
    const size_t close = text.rfind(')');
    if (close == std::string::npos || close + 2 >= text.size()) {
        return false;
    }
    const std::vector<std::string> fields = split_words(text.substr(close + 2));
    // fields[0] is field 3 (state), so indexes 7, 9, 20, 21 are fields 10, 12, 23, 24.
    if (fields.size() <= 21) {
        return false;
    }
    const auto parse_at = [&](size_t index, uint64_t & out, bool signed_value = false) {
        const std::string & raw = fields[index];
        if (signed_value && !raw.empty() && raw[0] == '-') {
            return false;
        }
        bool unlimited = false;
        return parse_uint64_scalar(raw, out, unlimited) && !unlimited;
    };
    return parse_at(7, values.minor_faults) &&
           parse_at(9, values.major_faults) &&
           parse_at(20, values.vsize_bytes) &&
           parse_at(21, values.rss_pages, true);
}

bool parse_smaps_rollup(const std::string & text, SmapsValues & values) {
    values = {};
    bool have_rss = false;
    bool have_pss = false;
    std::istringstream lines(text);
    for (std::string line; std::getline(lines, line);) {
        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, colon);
        const std::vector<std::string> words = split_words(line.substr(colon + 1));
        if ((key != "Rss" && key != "Pss" && key != "Swap") || words.empty()) {
            continue;
        }
        uint64_t kib = 0;
        bool unlimited = false;
        if (!parse_uint64_scalar(words[0], kib, unlimited) || unlimited ||
                kib > std::numeric_limits<uint64_t>::max() / 1024ull) {
            return false;
        }
        if (key == "Rss") {
            values.rss_bytes = kib * 1024ull;
            have_rss = true;
        } else if (key == "Pss") {
            values.pss_bytes = kib * 1024ull;
            have_pss = true;
        } else {
            values.swap_bytes = kib * 1024ull;
            values.swap_available = true;
        }
    }
    return have_rss && have_pss;
}

CounterDeltaResult advance_counter_delta(
        CounterDeltaState & state,
        Status current_status,
        bool current_has_value,
        uint64_t current_value,
        uint64_t current_read_ts_ns,
        const std::string & source_identity) {
    CounterDeltaResult result;
    result.status = current_status;
    result.current_read_ts_ns = current_read_ts_ns;
    if (current_status != Status::Available || !current_has_value) {
        state.previous_sample_available = false;
        return result;
    }

    if (!state.source_identity.empty() && state.source_identity != source_identity) {
        result.status = Status::SourceChanged;
    } else if (!state.previous_sample_available) {
        result.status = Status::NoPreviousSample;
    } else if (current_value < state.previous_value) {
        result.status = Status::CounterRegression;
        result.previous_read_ts_ns = state.previous_read_ts_ns;
    } else {
        result.status = Status::Available;
        result.has_value = true;
        result.value = current_value - state.previous_value;
        result.previous_read_ts_ns = state.previous_read_ts_ns;
    }

    state.previous_sample_available = true;
    state.previous_value = current_value;
    state.previous_read_ts_ns = current_read_ts_ns;
    state.source_identity = source_identity;
    return result;
}

CadenceAdvance advance_absolute_cadence(
        uint64_t scheduled_ts_ns,
        uint64_t interval_ns,
        uint64_t completion_ts_ns) {
    CadenceAdvance result;
    if (interval_ns == 0 ||
            scheduled_ts_ns > std::numeric_limits<uint64_t>::max() - interval_ns) {
        return result;
    }
    result.next_scheduled_ts_ns = scheduled_ts_ns + interval_ns;
    if (completion_ts_ns > result.next_scheduled_ts_ns) {
        result.missed_intervals =
                (completion_ts_ns - result.next_scheduled_ts_ns) / interval_ns + 1;
        if (result.missed_intervals >
                (std::numeric_limits<uint64_t>::max() - result.next_scheduled_ts_ns) /
                interval_ns) {
            result.next_scheduled_ts_ns = 0;
            result.missed_intervals = 0;
            return result;
        }
        result.next_scheduled_ts_ns += result.missed_intervals * interval_ns;
    }
    return result;
}

Mode mode() {
    static const Mode configured = parse_mode(std::getenv("LLM_MEM_TRACE_PRESSURE_SHADOW_MODE"));
    return configured;
}

bool enabled() {
    return mode() != Mode::Off;
}

void set_queue_snapshot_provider(QueueSnapshotProvider provider) {
    runtime().queue_provider.store(provider, std::memory_order_release);
}

bool latest_snapshot(LatestSnapshot & snapshot) {
    Runtime & state = runtime();
    if (!enabled()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(state.latest_mu);
    if (!state.have_latest) {
        return false;
    }
    snapshot = state.latest;
    return true;
}

void start() {
    if (!enabled()) {
        return;
    }
    Runtime & state = runtime();
    std::lock_guard<std::mutex> lock(state.lifecycle_mu);
    if (state.running.load(std::memory_order_acquire)) {
        return;
    }
    const char * configured_run_id = std::getenv("LLM_MEM_TRACE_RUN_ID");
    state.run_id = configured_run_id && configured_run_id[0] ? configured_run_id : "missing_run_id";
    state.interval_ns = env_u64(
            "LLM_MEM_TRACE_PRESSURE_SHADOW_SAMPLE_MS", 25, 10, 50) * NS_PER_MS;
    state.pss_interval_ns = env_u64(
            "LLM_MEM_TRACE_PRESSURE_SHADOW_PSS_SAMPLE_MS", 2000, 250, 60000) * NS_PER_MS;
    {
        std::lock_guard<std::mutex> pss_lock(state.pss_mu);
        state.latest_smaps = {};
        for (Observation<uint64_t> * field : {
                &state.latest_smaps.rss,
                &state.latest_smaps.pss,
                &state.latest_smaps.swap}) {
            field->status = Status::NotSampled;
            field->error = "smaps_rollup sampler has not published";
        }
    }
    state.stop_requested.store(false, std::memory_order_release);
    state.running.store(true, std::memory_order_release);
    state.sampler = std::thread(sampler_loop);
    state.pss_sampler = std::thread(pss_sampler_loop);
}

void stop() {
    Runtime & state = runtime();
    std::lock_guard<std::mutex> lock(state.lifecycle_mu);
    if (!state.running.load(std::memory_order_acquire)) {
        return;
    }
    state.stop_requested.store(true, std::memory_order_release);
    state.wait_cv.notify_all();
    if (state.sampler.joinable()) {
        state.sampler.join();
    }
    if (state.pss_sampler.joinable()) {
        state.pss_sampler.join();
    }
    state.stopped_ts_ns = monotonic_ns();
    write_summary(state);
    state.running.store(false, std::memory_order_release);
}

void observe_step_end(
        int phase,
        uint64_t step,
        uint64_t begin_ts_ns,
        uint64_t end_ts_ns,
        uint64_t latency_ns) {
    if (!enabled()) {
        return;
    }
    Runtime & state = runtime();
    state.last_step_phase.store(phase, std::memory_order_release);
    state.last_step.store(step, std::memory_order_release);
    state.last_step_begin_ts_ns.store(begin_ts_ns, std::memory_order_release);
    state.last_step_latency_ns.store(latency_ns, std::memory_order_release);
    state.last_step_end_ts_ns.store(end_ts_ns, std::memory_order_release);
}

void record_issue(uint64_t step, uint64_t issued_bytes) {
    if (!enabled()) {
        return;
    }
    Runtime & state = runtime();
    state.total_issued_bytes.fetch_add(issued_bytes, std::memory_order_relaxed);
    update_step_slot(state, step, issued_bytes, 0, 0);
}

void record_hint_call(uint64_t step, uint64_t advised_bytes) {
    if (!enabled()) {
        return;
    }
    Runtime & state = runtime();
    state.total_hint_calls.fetch_add(1, std::memory_order_relaxed);
    state.total_advised_bytes.fetch_add(advised_bytes, std::memory_order_relaxed);
    update_step_slot(state, step, 0, 1, advised_bytes);
}

} // namespace llm_pressure_shadow
