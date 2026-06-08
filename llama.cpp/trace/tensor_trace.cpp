#include "trace_event.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>

namespace {

void json_escape_append(std::string & out, const char * value) {
    out.push_back('"');
    if (value) {
        for (const char * p = value; *p; ++p) {
            if (*p == '"' || *p == '\\') {
                out.push_back('\\');
            }
            out.push_back(*p);
        }
    }
    out.push_back('"');
}

const char * phase_name(int phase) {
    switch (phase) {
        case LLM_MEM_TRACE_PHASE_PREFILL: return "PREFILL";
        case LLM_MEM_TRACE_PHASE_DECODE:  return "DECODE";
        default: return "UNKNOWN";
    }
}

int parse_layer_from_name(const char * name) {
    if (!name) {
        return -1;
    }
    const char * blk = std::strstr(name, "blk.");
    if (blk) {
        blk += 4;
        int layer = 0;
        bool found = false;
        while (*blk >= '0' && *blk <= '9') {
            found = true;
            layer = layer * 10 + (*blk - '0');
            ++blk;
        }
        return found ? layer : -1;
    }

    const char * cache = std::strstr(name, "cache_");
    if (cache) {
        const char * lpos = std::strstr(cache, "_l");
        if (lpos) {
            lpos += 2;
            int layer = 0;
            bool found = false;
            while (*lpos >= '0' && *lpos <= '9') {
                found = true;
                layer = layer * 10 + (*lpos - '0');
                ++lpos;
            }
            return found ? layer : -1;
        }
    }

    const char * dash = std::strrchr(name, '-');
    if (dash && dash[1]) {
        int layer = 0;
        bool found = false;
        const char * p = dash + 1;
        while (*p >= '0' && *p <= '9') {
            found = true;
            layer = layer * 10 + (*p - '0');
            ++p;
        }
        return found ? layer : -1;
    }

    return -1;
}

const char * tensor_backend_name(const ggml_tensor * t) {
    if (!t) {
        return "unknown";
    }
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buf) {
        return "unknown";
    }
    return ggml_backend_buffer_name(buf);
}

uintptr_t tensor_addr(const ggml_tensor * t) {
    if (!t) {
        return 0;
    }
    if (t->data) {
        return reinterpret_cast<uintptr_t>(t->data);
    }
    if (t->view_src && t->view_src->data) {
        return reinterpret_cast<uintptr_t>(t->view_src->data);
    }
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (buf) {
        void * base = ggml_backend_buffer_get_base(buf);
        return reinterpret_cast<uintptr_t>(base);
    }
    return 0;
}

bool is_param_tensor(const ggml_tensor * t) {
    if (!t) {
        return false;
    }
    if (t->flags & GGML_TENSOR_FLAG_PARAM) {
        return true;
    }
    if (t->op != GGML_OP_NONE) {
        return false;
    }
    const char * name = ggml_get_name(t);
    if (!name) {
        return false;
    }
    return std::strstr(name, "weight") || std::strstr(name, "bias") || std::strstr(name, "tok_embd");
}

struct FirstTouch {
    std::mutex mu;
    std::unordered_set<const ggml_tensor *> seen;

    bool mark(const ggml_tensor * t) {
        if (!mu.try_lock()) {
            return false;
        }
        const bool inserted = seen.insert(t).second;
        mu.unlock();
        return inserted;
    }
};

FirstTouch & first_touch() {
    static FirstTouch ft;
    return ft;
}

void log_tensor_event(const ggml_tensor * t, const char * access_kind) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_TENSOR)) {
        return;
    }

    const uint64_t ts = llm_mem_trace_time_ns();
    const char * name = ggml_get_name(t);
    const char * op_name = ggml_op_name(t->op);
    const size_t nbytes = ggml_nbytes(t);
    const int layer = parse_layer_from_name(name);
    const uintptr_t addr = tensor_addr(t);
    const char * backend = tensor_backend_name(t);

    bool first = false;
    if (access_kind && std::strcmp(access_kind, "begin") == 0) {
        first = first_touch().mark(t);
    }

    char addr_buf[32];
    std::snprintf(addr_buf, sizeof(addr_buf), "0x%llx", (unsigned long long) addr);

    std::string line;
    line.reserve(256);
    line += "{\"event\":\"TENSOR_ACCESS\",\"ts_ns\":" + std::to_string(ts);
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    line += ",\"access\":";
    json_escape_append(line, access_kind ? access_kind : "unknown");
    line += ",\"tensor\":";
    json_escape_append(line, name ? name : "");
    if (op_name) {
        line += ",\"op\":";
        json_escape_append(line, op_name);
    }
    if (layer >= 0) {
        line += ",\"layer\":" + std::to_string(layer);
    }
    line += ",\"size\":" + std::to_string(nbytes);
    line += ",\"addr\":";
    json_escape_append(line, addr_buf);
    line += ",\"backend\":";
    json_escape_append(line, backend);
    if (first) {
        line += ",\"first_touch\":true";
    }
    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_TENSOR, line.c_str(), line.size());
}

void log_param_access(const ggml_tensor * t, const char * parent_name) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_TENSOR)) {
        return;
    }
    if (!is_param_tensor(t)) {
        return;
    }

    const uint64_t ts = llm_mem_trace_time_ns();
    const char * name = ggml_get_name(t);
    const size_t nbytes = ggml_nbytes(t);
    const int layer = parse_layer_from_name(name);
    const uintptr_t addr = tensor_addr(t);
    const char * backend = tensor_backend_name(t);

    bool first = first_touch().mark(t);

    char addr_buf[32];
    std::snprintf(addr_buf, sizeof(addr_buf), "0x%llx", (unsigned long long) addr);

    std::string line;
    line.reserve(256);
    line += "{\"event\":\"TENSOR_ACCESS\",\"ts_ns\":" + std::to_string(ts);
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    line += ",\"access\":\"param\"";
    line += ",\"tensor\":";
    json_escape_append(line, name ? name : "");
    if (parent_name) {
        line += ",\"param_of\":";
        json_escape_append(line, parent_name);
    }
    if (layer >= 0) {
        line += ",\"layer\":" + std::to_string(layer);
    }
    line += ",\"size\":" + std::to_string(nbytes);
    line += ",\"addr\":";
    json_escape_append(line, addr_buf);
    line += ",\"backend\":";
    json_escape_append(line, backend);
    if (first) {
        line += ",\"first_touch\":true";
    }
    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_TENSOR, line.c_str(), line.size());
}

struct LayerTracker {
    std::mutex mu;
    uint64_t step_id = 0;
    std::unordered_set<int> begun;
    std::unordered_set<int> ended;

    void reset_if_needed(uint64_t step) {
        if (step == step_id) {
            return;
        }
        step_id = step;
        begun.clear();
        ended.clear();
    }

    void on_begin(int layer) {
        if (layer < 0) {
            return;
        }
        const uint64_t step = llm_mem_trace_get_step();
        std::lock_guard<std::mutex> lock(mu);
        reset_if_needed(step);
        if (!begun.insert(layer).second) {
            return;
        }

        const uint64_t ts = llm_mem_trace_time_ns();
        std::string line;
        line.reserve(128);
        line += "{\"event\":\"LAYER_BEGIN\",\"ts_ns\":" + std::to_string(ts);
        line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
        line += ",\"step\":" + std::to_string(step);
        line += ",\"layer\":" + std::to_string(layer);
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }

    void on_end(int layer) {
        if (layer < 0) {
            return;
        }
        const uint64_t step = llm_mem_trace_get_step();
        std::lock_guard<std::mutex> lock(mu);
        reset_if_needed(step);
        if (!ended.insert(layer).second) {
            return;
        }

        const uint64_t ts = llm_mem_trace_time_ns();
        std::string line;
        line.reserve(128);
        line += "{\"event\":\"LAYER_END\",\"ts_ns\":" + std::to_string(ts);
        line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
        line += ",\"step\":" + std::to_string(step);
        line += ",\"layer\":" + std::to_string(layer);
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }
};

LayerTracker & layer_tracker() {
    static LayerTracker tracker;
    return tracker;
}

bool is_layer_end_tensor(const char * name) {
    if (!name) {
        return false;
    }
    return std::strstr(name, "ffn_out") || std::strstr(name, "ffn_moe_out");
}

} // namespace

extern "C" void llm_mem_trace_tensor_begin(const ggml_tensor * t) {
    if (!llm_mem_trace_enabled() || !t) {
        return;
    }

    log_tensor_event(t, "begin");

    const char * name = ggml_get_name(t);
    const int layer = parse_layer_from_name(name);
    layer_tracker().on_begin(layer);

    if (t->src[0]) {
        log_param_access(t->src[0], name);
    }
    if (t->src[1]) {
        log_param_access(t->src[1], name);
    }
}

extern "C" void llm_mem_trace_tensor_end(const ggml_tensor * t) {
    if (!llm_mem_trace_enabled() || !t) {
        return;
    }

    log_tensor_event(t, "end");

    const char * name = ggml_get_name(t);
    const int layer = parse_layer_from_name(name);
    if (is_layer_end_tensor(name)) {
        layer_tracker().on_end(layer);
    }
}

extern "C" void llm_mem_trace_tensor_loaded(const ggml_tensor * t, const char * stage) {
    llm_mem_trace_init(nullptr);
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_TENSOR) || !t) {
        return;
    }

    const uint64_t ts = llm_mem_trace_time_ns();
    const char * name = ggml_get_name(t);
    const size_t nbytes = ggml_nbytes(t);
    const int layer = parse_layer_from_name(name);
    const uintptr_t addr = tensor_addr(t);
    const char * backend = tensor_backend_name(t);

    char addr_buf[32];
    std::snprintf(addr_buf, sizeof(addr_buf), "0x%llx", (unsigned long long) addr);

    std::string line;
    line.reserve(256);
    line += "{\"event\":\"TENSOR_LOAD\",\"ts_ns\":" + std::to_string(ts);
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    line += ",\"tensor\":";
    json_escape_append(line, name ? name : "");
    if (stage) {
        line += ",\"stage\":";
        json_escape_append(line, stage);
    }
    if (layer >= 0) {
        line += ",\"layer\":" + std::to_string(layer);
    }
    line += ",\"size\":" + std::to_string(nbytes);
    line += ",\"addr\":";
    json_escape_append(line, addr_buf);
    line += ",\"backend\":";
    json_escape_append(line, backend);
    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_TENSOR, line.c_str(), line.size());
}
