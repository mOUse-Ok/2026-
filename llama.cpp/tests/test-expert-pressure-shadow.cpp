#include "expert_pressure_shadow.h"
#include "trace_event.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { return __LINE__; } } while (false)

static std::mutex captured_mu;
static std::vector<std::string> captured_lines;

extern "C" int llm_mem_trace_get_phase(void) {
    return 0;
}

extern "C" uint64_t llm_mem_trace_get_step(void) {
    return 0;
}

extern "C" void llm_mem_trace_write(int, const char * line, size_t size) {
    std::lock_guard<std::mutex> lock(captured_mu);
    captured_lines.emplace_back(line, size);
}

static std::string make_proc_stat() {
    std::vector<std::string> fields(22, "0");
    fields[0] = "R";
    fields[7] = "11";
    fields[9] = "13";
    fields[20] = "4096";
    fields[21] = "7";
    std::string text = "42 (worker ) tricky name) ";
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) {
            text += " ";
        }
        text += fields[i];
    }
    return text;
}

static llm_pressure_shadow::QueueSnapshot test_queue_snapshot() {
    llm_pressure_shadow::QueueSnapshot snapshot;
    snapshot.status = llm_pressure_shadow::Status::Available;
    snapshot.started = true;
    snapshot.configured_worker_count = 2;
    snapshot.queue_depth = 3;
    snapshot.queued_bytes = 4096;
    snapshot.worker_count = 2;
    snapshot.busy_workers = 1;
    return snapshot;
}

static llm_pressure_shadow::QueueSnapshot test_queue_not_started_snapshot() {
    llm_pressure_shadow::QueueSnapshot snapshot;
    snapshot.status = llm_pressure_shadow::Status::NotStarted;
    snapshot.configured_worker_count = 2;
    return snapshot;
}

int main(int argc, char ** argv) {
    using namespace llm_pressure_shadow;

    bool valid = false;
    CHECK(parse_mode(nullptr, &valid) == Mode::Off && valid);
    CHECK(parse_mode("off", &valid) == Mode::Off && valid);
    CHECK(parse_mode("summary", &valid) == Mode::Summary && valid);
    CHECK(parse_mode("detail", &valid) == Mode::Detail && valid);
    CHECK(parse_mode("active", &valid) == Mode::Off && !valid);

    uint64_t scalar = 0;
    bool unlimited = false;
    CHECK(parse_uint64_scalar(" 18446744073709551615\n", scalar, unlimited));
    CHECK(scalar == UINT64_MAX && !unlimited);
    CHECK(!parse_uint64_scalar("max\n", scalar, unlimited) && unlimited);
    CHECK(!parse_uint64_scalar("-1", scalar, unlimited));
    CHECK(!parse_uint64_scalar("18446744073709551616", scalar, unlimited));
    CHECK(!parse_uint64_scalar("12 garbage", scalar, unlimited));

    std::unordered_map<std::string, uint64_t> values;
    CHECK(parse_key_value("anon 17\nfile 23\npgmajfault 2\n", values));
    CHECK(values.at("anon") == 17);
    CHECK(values.at("file") == 23);
    CHECK(values.at("pgmajfault") == 2);
    CHECK(!parse_key_value("anon\n", values));
    CHECK(!parse_key_value("anon -1\n", values));
    CHECK(!parse_key_value("anon 1\nanon 2\n", values));
    CHECK(!parse_key_value("", values));

    PsiValues psi;
    CHECK(parse_psi(
            "some avg10=1.25 avg60=0.50 avg300=0.10 total=1234\n"
            "full avg10=0.75 avg60=0.20 avg300=0.05 total=456\n",
            psi));
    CHECK(psi.some_avg10 == 1.25);
    CHECK(psi.some_total_us == 1234);
    CHECK(psi.full_avg10 == 0.75);
    CHECK(psi.full_total_us == 456);
    CHECK(!parse_psi("some avg10=1.0 total=1\n", psi));
    CHECK(!parse_psi(
            "some avg10=nan total=1\n"
            "full avg10=0.0 total=0\n",
            psi));

    ProcStatValues proc;
    CHECK(parse_proc_stat(make_proc_stat(), proc));
    CHECK(proc.minor_faults == 11);
    CHECK(proc.major_faults == 13);
    CHECK(proc.vsize_bytes == 4096);
    CHECK(proc.rss_pages == 7);
    CHECK(!parse_proc_stat("42 missing-close R 0 0", proc));

    SmapsValues smaps;
    CHECK(parse_smaps_rollup(
            "00400000-00452000 r--p 00000000 00:00 0\n"
            "Rss:                1024 kB\n"
            "Pss:                 768 kB\n"
            "Swap:                 64 kB\n"
            "Anonymous:            12 kB\n",
            smaps));
    CHECK(smaps.rss_bytes == 1024ull * 1024ull);
    CHECK(smaps.pss_bytes == 768ull * 1024ull);
    CHECK(smaps.swap_available);
    CHECK(smaps.swap_bytes == 64ull * 1024ull);
    CHECK(!parse_smaps_rollup("Rss: 1 kB\n", smaps));

    CHECK(std::string(status_name(Status::PermissionDenied)) == "permission_denied");
    CHECK(std::string(status_name(Status::FieldMissing)) == "field_missing");
    CHECK(std::string(status_name(Status::CounterRegression)) == "counter_regression");

    CounterDeltaState counter;
    CounterDeltaResult delta = advance_counter_delta(
            counter, Status::Available, true, 10, 100, "scope-a");
    CHECK(delta.status == Status::NoPreviousSample && !delta.has_value);
    delta = advance_counter_delta(
            counter, Status::Available, true, 15, 200, "scope-a");
    CHECK(delta.status == Status::Available && delta.has_value);
    CHECK(delta.value == 5 && delta.previous_read_ts_ns == 100);
    delta = advance_counter_delta(
            counter, Status::PermissionDenied, false, 0, 300, "scope-a");
    CHECK(delta.status == Status::PermissionDenied && !delta.has_value);
    delta = advance_counter_delta(
            counter, Status::Available, true, 20, 400, "scope-a");
    CHECK(delta.status == Status::NoPreviousSample && !delta.has_value);
    delta = advance_counter_delta(
            counter, Status::Available, true, 19, 500, "scope-a");
    CHECK(delta.status == Status::CounterRegression && !delta.has_value);
    delta = advance_counter_delta(
            counter, Status::Available, true, 25, 600, "scope-b");
    CHECK(delta.status == Status::SourceChanged && !delta.has_value);

    for (uint64_t interval : {10ull, 25ull, 50ull}) {
        const CadenceAdvance on_time =
                advance_absolute_cadence(100, interval, 100 + interval);
        CHECK(on_time.next_scheduled_ts_ns == 100 + interval);
        CHECK(on_time.missed_intervals == 0);
        const CadenceAdvance late =
                advance_absolute_cadence(100, interval, 100 + interval + 1);
        CHECK(late.next_scheduled_ts_ns == 100 + 2 * interval);
        CHECK(late.missed_intervals == 1);
        const CadenceAdvance very_late =
                advance_absolute_cadence(100, interval, 100 + 3 * interval + 1);
        CHECK(very_late.next_scheduled_ts_ns == 100 + 4 * interval);
        CHECK(very_late.missed_intervals == 3);
    }

    if (argc > 1 && std::string(argv[1]) == "--runtime-off") {
        unsetenv("LLM_MEM_TRACE_PRESSURE_SHADOW_MODE");
        start();
        stop();
        CHECK(!enabled());
        CHECK(captured_lines.empty());
    }

    if (argc > 1 && std::string(argv[1]) == "--runtime-detail") {
        setenv("LLM_MEM_TRACE_PRESSURE_SHADOW_MODE", "detail", 1);
        setenv("LLM_MEM_TRACE_PRESSURE_SHADOW_SAMPLE_MS", "10", 1);
        setenv("LLM_MEM_TRACE_PRESSURE_SHADOW_PSS_SAMPLE_MS", "25", 1);
        setenv("LLM_MEM_TRACE_RUN_ID", "unit_runtime", 1);
        set_queue_snapshot_provider(test_queue_snapshot);
        start();
        record_issue(0, 8192);
        record_hint_call(0, 4096);
        record_hint_call(0, 4096);
        observe_step_end(2, 7, 100, 300, 200);
        std::this_thread::sleep_for(std::chrono::milliseconds(45));
        stop();
        LatestSnapshot latest;
        CHECK(latest_snapshot(latest));
        CHECK(latest.sequence > 0);
        CHECK(latest.sample_ready_ts_ns > 0);
        CHECK(latest.queue_depth.status == Status::Available);
        CHECK(latest.queue_depth.has_value);
        CHECK(latest.queue_depth.value == 3);
        CHECK(latest.busy_workers.value <= 2);
        bool have_sample = false;
        bool have_summary = false;
        {
            std::lock_guard<std::mutex> lock(captured_mu);
            for (const std::string & line : captured_lines) {
                have_sample = have_sample ||
                        line.find("\"event\":\"PRESSURE_SHADOW_SAMPLE\"") != std::string::npos;
                have_summary = have_summary ||
                        line.find("\"event\":\"PRESSURE_SHADOW_SUMMARY\"") != std::string::npos;
                CHECK(line.find("\"run_id\":\"unit_runtime\"") != std::string::npos);
            }
        }
        CHECK(have_sample);
        CHECK(have_summary);
        const size_t lines_after_stop = captured_lines.size();
        stop();
        CHECK(captured_lines.size() == lines_after_stop);
    }

    if (argc > 1 && std::string(argv[1]) == "--runtime-not-started") {
        setenv("LLM_MEM_TRACE_PRESSURE_SHADOW_MODE", "detail", 1);
        setenv("LLM_MEM_TRACE_PRESSURE_SHADOW_SAMPLE_MS", "10", 1);
        setenv("LLM_MEM_TRACE_PRESSURE_SHADOW_PSS_SAMPLE_MS", "250", 1);
        setenv("LLM_MEM_TRACE_RUN_ID", "unit_not_started", 1);
        set_queue_snapshot_provider(test_queue_not_started_snapshot);
        start();
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        stop();
        LatestSnapshot latest;
        CHECK(latest_snapshot(latest));
        CHECK(latest.queue_depth.status == Status::NotStarted);
        CHECK(!latest.queue_depth.has_value);
        bool have_not_started = false;
        {
            std::lock_guard<std::mutex> lock(captured_mu);
            for (const std::string & line : captured_lines) {
                have_not_started = have_not_started ||
                        line.find("\"value\":\"not_started\"") != std::string::npos;
            }
        }
        CHECK(have_not_started);
    }

    return 0;
}
