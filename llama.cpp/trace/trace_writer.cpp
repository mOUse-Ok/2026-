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
    std::vector<std::string> queue;
    std::thread thread;
    std::atomic<bool> running{false};
    std::atomic<uint64_t> dropped{0};
    size_t max_queue = 8192;

    void start(const std::string & file_path) {
        path = file_path;
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
        if (!mu.try_lock()) {
            dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (queue.size() >= max_queue) {
            dropped.fetch_add(1, std::memory_order_relaxed);
            mu.unlock();
            return;
        }
        queue.emplace_back(std::move(line));
        mu.unlock();
        cv.notify_one();
    }

    void flush_loop() {
        std::unique_lock<std::mutex> lock(mu);
        while (running.load(std::memory_order_acquire)) {
            cv.wait_for(lock, std::chrono::milliseconds(50));
            if (!queue.empty()) {
                std::vector<std::string> local;
                local.swap(queue);
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
        std::fflush(fp);
    }
};

struct TraceState {
    bool enabled = false;
    bool sink_enabled[4] = { false, false, false, false };
    std::string dir;

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

const char * phase_name(int phase) {
    switch (phase) {
        case LLM_MEM_TRACE_PHASE_PREFILL: return "PREFILL";
        case LLM_MEM_TRACE_PHASE_DECODE:  return "DECODE";
        default: return "UNKNOWN";
    }
}

void write_summary() {
    auto & s = state();
    if (!s.enabled) {
        return;
    }

    std::string path = s.dir + "/summary.json";
    FILE * fp = std::fopen(path.c_str(), "w");
    if (!fp) {
        return;
    }

    std::string line = "{";
    line += "\"dropped\":{";
    line += "\"tensor\":" + std::to_string(s.tensor.dropped.load()) + ",";
    line += "\"kv\":" + std::to_string(s.kv.dropped.load()) + ",";
    line += "\"expert\":" + std::to_string(s.expert.dropped.load()) + ",";
    line += "\"memory\":" + std::to_string(s.memory.dropped.load());
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

        if (s.sink_enabled[LLM_MEM_TRACE_SINK_TENSOR]) {
            s.tensor.start(s.dir + "/tensor_trace.jsonl");
        }
        if (s.sink_enabled[LLM_MEM_TRACE_SINK_KV]) {
            s.kv.start(s.dir + "/kv_trace.jsonl");
        }
        if (s.sink_enabled[LLM_MEM_TRACE_SINK_EXPERT]) {
            s.expert.start(s.dir + "/expert_trace.jsonl");
        }
        if (s.sink_enabled[LLM_MEM_TRACE_SINK_MEMORY]) {
            s.memory.start(s.dir + "/memory_trace.jsonl");
        }

        std::atexit(llm_mem_trace_shutdown);
    });
}

extern "C" void llm_mem_trace_shutdown(void) {
    auto & s = state();
    if (!s.enabled) {
        return;
    }

    s.enabled = false;

    s.tensor.stop();
    s.kv.stop();
    s.expert.stop();
    s.memory.stop();

    write_summary();
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
    }
    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    llm_mem_trace_memory_sample("token_end");
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
