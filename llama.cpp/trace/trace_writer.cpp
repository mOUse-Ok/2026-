#include "trace_event.h"

#include "llama-batch.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <time.h>

namespace {

struct TraceWriter {
    std::string path;
    FILE * fp = nullptr;
    std::mutex mu;
    std::condition_variable cv;
    std::condition_variable cv_space;
    std::vector<std::string> queue;
    std::thread thread;
    std::atomic<bool> running{false};
    std::atomic<uint64_t> enqueued{0};
    std::atomic<uint64_t> written{0};
    std::atomic<uint64_t> dropped{0};
    size_t max_queue = 8192;
    bool allow_drop = false;

    void start(const std::string & file_path, size_t queue_limit, bool drop_when_full) {
        path = file_path;
        max_queue = queue_limit > 0 ? queue_limit : 8192;
        allow_drop = drop_when_full;
        fp = std::fopen(path.c_str(), "w");
        if (!fp) {
            return;
        }
        running.store(true, std::memory_order_release);
        thread = std::thread([this] { this->flush_loop(); });
    }

    void stop() {
        running.store(false, std::memory_order_release);
        cv.notify_all();
        cv_space.notify_all();
        if (thread.joinable()) {
            thread.join();
        }
        flush();
        if (fp) {
            std::fclose(fp);
            fp = nullptr;
        }
    }

    void enqueue(std::string && line) {
        if (!running.load(std::memory_order_acquire)) {
            return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (allow_drop) {
            if (queue.size() >= max_queue) {
                dropped.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        } else {
            cv_space.wait(lock, [this] {
                return queue.size() < max_queue || !running.load(std::memory_order_acquire);
            });
            if (!running.load(std::memory_order_acquire)) {
                return;
            }
        }
        queue.emplace_back(std::move(line));
        enqueued.fetch_add(1, std::memory_order_relaxed);
        lock.unlock();
        cv.notify_one();
    }

    void flush_loop() {
        std::unique_lock<std::mutex> lock(mu);
        while (running.load(std::memory_order_acquire)) {
            cv.wait_for(lock, std::chrono::milliseconds(50));
            if (!queue.empty()) {
                std::vector<std::string> local;
                local.swap(queue);
                cv_space.notify_all();
                lock.unlock();
                write_lines(local);
                lock.lock();
            }
        }
    }

    void flush() {
        std::vector<std::string> local;
        {
            std::lock_guard<std::mutex> lock(mu);
            if (queue.empty()) {
                return;
            }
            local.swap(queue);
        }
        cv_space.notify_all();
        write_lines(local);
    }

    void write_lines(const std::vector<std::string> & lines) {
        if (!fp) {
            return;
        }
        for (const auto & line : lines) {
            std::fwrite(line.data(), 1, line.size(), fp);
            std::fwrite("\n", 1, 1, fp);
        }
        written.fetch_add(lines.size(), std::memory_order_relaxed);
        std::fflush(fp);
    }
};

struct TraceState {
    bool enabled = false;
    bool sink_enabled[4] = { false, false, false, false };
    std::string dir;
    size_t queue_limit = 65536;
    bool allow_drop = false;

    TraceWriter tensor;
    TraceWriter kv;
    TraceWriter expert;
    TraceWriter memory;

    std::atomic<uint64_t> step{0};
};

struct TraceContext {
    std::atomic<const llama_ubatch *> ubatch{nullptr};
    std::atomic<int> phase{LLM_MEM_TRACE_PHASE_UNKNOWN};
    std::atomic<uint64_t> step_id{0};
    std::atomic<uint64_t> step_begin_ts{0};
    std::mutex token_mu;
    std::vector<uint64_t> token_begin_ts;
};

TraceState & state() {
    static TraceState s;
    return s;
}

TraceContext & context() {
    static TraceContext ctx;
    return ctx;
}

bool env_truthy(const char * value) {
    if (!value) {
        return false;
    }
    if (value[0] == '0' && value[1] == '\0') {
        return false;
    }
    return true;
}

bool env_bool_or_default(const char * key, bool def_value) {
    const char * val = std::getenv(key);
    if (!val) {
        return def_value;
    }
    return env_truthy(val);
}

std::string env_str(const char * key, const char * def_value) {
    const char * val = std::getenv(key);
    return val ? std::string(val) : std::string(def_value);
}

size_t env_size_or_default(const char * key, size_t def_value) {
    const char * val = std::getenv(key);
    if (!val || !val[0]) {
        return def_value;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(val, &end, 10);
    return end != val && parsed > 0 ? (size_t) parsed : def_value;
}

const char * phase_name(int phase) {
    switch (phase) {
        case LLM_MEM_TRACE_PHASE_PREFILL: return "PREFILL";
        case LLM_MEM_TRACE_PHASE_DECODE:  return "DECODE";
        default: return "UNKNOWN";
    }
}

void write_summary() {
    auto & s = state();
    std::string path = s.dir + "/summary.json";
    FILE * fp = std::fopen(path.c_str(), "w");
    if (!fp) {
        return;
    }

    std::string line = "{\"schema_version\":2";
    line += ",\"allow_drop\":" + std::string(s.allow_drop ? "true" : "false");
    line += ",\"queue_limit\":" + std::to_string(s.queue_limit);
    const auto append_counts = [&line](const char * name, const TraceWriter & writer, bool enabled, bool comma) {
        line += "\"" + std::string(name) + "\":{";
        line += "\"enabled\":" + std::string(enabled ? "true" : "false") + ",";
        line += "\"enqueued\":" + std::to_string(writer.enqueued.load()) + ",";
        line += "\"written\":" + std::to_string(writer.written.load()) + ",";
        line += "\"dropped\":" + std::to_string(writer.dropped.load()) + "}";
        if (comma) {
            line += ",";
        }
    };
    line += ",\"sinks\":{";
    append_counts("tensor", s.tensor, s.sink_enabled[LLM_MEM_TRACE_SINK_TENSOR], true);
    append_counts("kv", s.kv, s.sink_enabled[LLM_MEM_TRACE_SINK_KV], true);
    append_counts("expert", s.expert, s.sink_enabled[LLM_MEM_TRACE_SINK_EXPERT], true);
    append_counts("memory", s.memory, s.sink_enabled[LLM_MEM_TRACE_SINK_MEMORY], false);
    line += "}}";

    std::fwrite(line.data(), 1, line.size(), fp);
    std::fwrite("\n", 1, 1, fp);
    std::fclose(fp);
}

} // namespace

extern "C" uint64_t llm_mem_trace_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

extern "C" uint64_t llm_mem_trace_next_step(void) {
    return state().step.fetch_add(1, std::memory_order_relaxed) + 1;
}

extern "C" void llm_mem_trace_init(const char * dir) {
    static std::once_flag init_flag;
    std::call_once(init_flag, [dir] {
        auto & s = state();
        if (!env_truthy(std::getenv("LLM_MEM_TRACE"))) {
            return;
        }

        s.enabled = true;
        s.dir = dir && dir[0] ? std::string(dir) : env_str("LLM_MEM_TRACE_DIR", "trace");

        std::error_code ec;
        std::filesystem::create_directories(s.dir, ec);

        s.sink_enabled[LLM_MEM_TRACE_SINK_TENSOR] = env_bool_or_default("LLM_MEM_TRACE_TENSOR", true);
        s.sink_enabled[LLM_MEM_TRACE_SINK_KV]     = env_bool_or_default("LLM_MEM_TRACE_KV", true);
        s.sink_enabled[LLM_MEM_TRACE_SINK_EXPERT] = env_bool_or_default("LLM_MEM_TRACE_EXPERT", true);
        s.sink_enabled[LLM_MEM_TRACE_SINK_MEMORY] = env_bool_or_default("LLM_MEM_TRACE_MEMORY", true);
        s.queue_limit = env_size_or_default("LLM_MEM_TRACE_QUEUE_LIMIT", 65536);
        s.allow_drop = env_bool_or_default("LLM_MEM_TRACE_ALLOW_DROP", false);

        if (s.sink_enabled[LLM_MEM_TRACE_SINK_TENSOR]) {
            s.tensor.start(s.dir + "/tensor_trace.jsonl", s.queue_limit, s.allow_drop);
        }
        if (s.sink_enabled[LLM_MEM_TRACE_SINK_KV]) {
            s.kv.start(s.dir + "/kv_trace.jsonl", s.queue_limit, s.allow_drop);
        }
        if (s.sink_enabled[LLM_MEM_TRACE_SINK_EXPERT]) {
            s.expert.start(s.dir + "/expert_trace.jsonl", s.queue_limit, s.allow_drop);
        }
        if (s.sink_enabled[LLM_MEM_TRACE_SINK_MEMORY]) {
            s.memory.start(s.dir + "/memory_trace.jsonl", s.queue_limit, s.allow_drop);
            const uint64_t ts = llm_mem_trace_time_ns();
            const std::string line = "{\"event\":\"TRACE_START\",\"ts_ns\":" + std::to_string(ts) + "}";
            llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
            llm_mem_trace_memory_sample("trace_init");
        }

        std::atexit(llm_mem_trace_shutdown);
    });
}

extern "C" void llm_mem_trace_shutdown(void) {
    auto & s = state();
    if (!s.enabled) {
        return;
    }

    if (s.sink_enabled[LLM_MEM_TRACE_SINK_MEMORY]) {
        llm_mem_trace_memory_sample("trace_shutdown");
        const uint64_t ts = llm_mem_trace_time_ns();
        const std::string line = "{\"event\":\"TRACE_END\",\"ts_ns\":" + std::to_string(ts) + "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }

    s.tensor.stop();
    s.kv.stop();
    s.expert.stop();
    s.memory.stop();

    write_summary();
    s.enabled = false;
}

extern "C" int llm_mem_trace_enabled(void) {
    return state().enabled ? 1 : 0;
}

extern "C" int llm_mem_trace_sink_enabled(int sink) {
    if (sink < 0 || sink >= 4) {
        return 0;
    }
    return state().enabled && state().sink_enabled[sink];
}

extern "C" void llm_mem_trace_set_ubatch(const struct llama_ubatch * ubatch, int phase, uint64_t step_id) {
    auto & ctx = context();
    ctx.phase.store(phase, std::memory_order_release);
    ctx.step_id.store(step_id, std::memory_order_release);
    ctx.step_begin_ts.store(0, std::memory_order_release);
    ctx.ubatch.store(ubatch, std::memory_order_release);

    if (ubatch) {
        std::lock_guard<std::mutex> lock(ctx.token_mu);
        ctx.token_begin_ts.assign(ubatch->n_tokens, 0);
    }
}

extern "C" void llm_mem_trace_clear_ubatch(void) {
    auto & ctx = context();
    ctx.ubatch.store(nullptr, std::memory_order_release);
    ctx.phase.store(LLM_MEM_TRACE_PHASE_UNKNOWN, std::memory_order_release);
    ctx.step_id.store(0, std::memory_order_release);
    ctx.step_begin_ts.store(0, std::memory_order_release);
    std::lock_guard<std::mutex> lock(ctx.token_mu);
    ctx.token_begin_ts.clear();
}

extern "C" const struct llama_ubatch * llm_mem_trace_get_ubatch(void) {
    return context().ubatch.load(std::memory_order_acquire);
}

extern "C" int llm_mem_trace_get_phase(void) {
    return context().phase.load(std::memory_order_acquire);
}

extern "C" uint64_t llm_mem_trace_get_step(void) {
    return context().step_id.load(std::memory_order_acquire);
}

extern "C" void llm_mem_trace_step_begin(void) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    auto & ctx = context();
    const llama_ubatch * ubatch = llm_mem_trace_get_ubatch();
    if (!ubatch) {
        return;
    }
    const uint64_t ts = llm_mem_trace_time_ns();
    ctx.step_begin_ts.store(ts, std::memory_order_release);

    std::string line;
    line.reserve(192);
    line += "{\"event\":\"STEP_BEGIN\",\"ts_ns\":" + std::to_string(ts);
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    line += ",\"n_tokens\":" + std::to_string(ubatch->n_tokens) + "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

extern "C" void llm_mem_trace_step_end(void) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    auto & ctx = context();
    const llama_ubatch * ubatch = llm_mem_trace_get_ubatch();
    const uint64_t start_ts = ctx.step_begin_ts.exchange(0, std::memory_order_acq_rel);
    if (!ubatch || start_ts == 0) {
        return;
    }
    const uint64_t ts = llm_mem_trace_time_ns();

    std::string line;
    line.reserve(224);
    line += "{\"event\":\"STEP_END\",\"ts_ns\":" + std::to_string(ts);
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    line += ",\"n_tokens\":" + std::to_string(ubatch->n_tokens);
    line += ",\"latency_ns\":" + std::to_string(ts - start_ts) + "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    llm_mem_trace_memory_sample("step_end");
}

extern "C" void llm_mem_trace_token_begin(int token_idx) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    const llama_ubatch * ubatch = llm_mem_trace_get_ubatch();
    if (!ubatch || !ubatch->token || token_idx < 0 || (uint32_t) token_idx >= ubatch->n_tokens) {
        return;
    }

    const uint64_t ts = llm_mem_trace_time_ns();
    {
        auto & ctx = context();
        std::lock_guard<std::mutex> lock(ctx.token_mu);
        if ((size_t) token_idx < ctx.token_begin_ts.size()) {
            ctx.token_begin_ts[token_idx] = ts;
        }
    }

    std::string line;
    line.reserve(256);
    line += "{\"event\":\"TOKEN_BEGIN\",\"ts_ns\":" + std::to_string(ts);
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    line += ",\"token_idx\":" + std::to_string(token_idx);
    line += ",\"token\":" + std::to_string(ubatch->token[token_idx]);
    if (ubatch->pos) {
        line += ",\"pos\":" + std::to_string(ubatch->pos[token_idx]);
    }
    if (ubatch->seq_id && ubatch->n_seq_id) {
        if (ubatch->n_seq_id[token_idx] > 0) {
            line += ",\"seq_id\":" + std::to_string(ubatch->seq_id[token_idx][0]);
        }
    }
    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

extern "C" void llm_mem_trace_token_end(int token_idx) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    const llama_ubatch * ubatch = llm_mem_trace_get_ubatch();
    if (!ubatch || !ubatch->token || token_idx < 0 || (uint32_t) token_idx >= ubatch->n_tokens) {
        return;
    }

    const uint64_t ts = llm_mem_trace_time_ns();
    uint64_t start_ts = 0;
    {
        auto & ctx = context();
        std::lock_guard<std::mutex> lock(ctx.token_mu);
        if ((size_t) token_idx < ctx.token_begin_ts.size()) {
            start_ts = ctx.token_begin_ts[token_idx];
        }
    }

    std::string line;
    line.reserve(256);
    line += "{\"event\":\"TOKEN_END\",\"ts_ns\":" + std::to_string(ts);
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    line += ",\"token_idx\":" + std::to_string(token_idx);
    line += ",\"token\":" + std::to_string(ubatch->token[token_idx]);
    if (ubatch->pos) {
        line += ",\"pos\":" + std::to_string(ubatch->pos[token_idx]);
    }
    if (ubatch->seq_id && ubatch->n_seq_id) {
        if (ubatch->n_seq_id[token_idx] > 0) {
            line += ",\"seq_id\":" + std::to_string(ubatch->seq_id[token_idx][0]);
        }
    }
    if (start_ts != 0) {
        line += ",\"latency_ns\":" + std::to_string(ts - start_ts);
        line += ",\"latency_scope\":\"ubatch_legacy\"";
    }
    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

extern "C" void llm_mem_trace_write(int sink, const char * line, size_t len) {
    auto & s = state();
    if (!s.enabled || !line || len == 0) {
        return;
    }
    if (sink < 0 || sink >= 4 || !s.sink_enabled[sink]) {
        return;
    }

    std::string copy(line, len);
    switch (sink) {
        case LLM_MEM_TRACE_SINK_TENSOR:
            s.tensor.enqueue(std::move(copy));
            break;
        case LLM_MEM_TRACE_SINK_KV:
            s.kv.enqueue(std::move(copy));
            break;
        case LLM_MEM_TRACE_SINK_EXPERT:
            s.expert.enqueue(std::move(copy));
            break;
        case LLM_MEM_TRACE_SINK_MEMORY:
            s.memory.enqueue(std::move(copy));
            break;
        default:
            break;
    }
}
