#include "trace_event.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

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

bool env_truthy(const char * value) {
    if (!value) {
        return false;
    }
    return !(value[0] == '0' && value[1] == '\0');
}

bool residency_enabled() {
    static const bool enabled = env_truthy(std::getenv("LLM_MEM_TRACE_RESIDENCY"));
    return enabled;
}

size_t env_size_or_default(const char * key, size_t def_value) {
    const char * val = std::getenv(key);
    if (!val || !val[0]) {
        return def_value;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(val, &end, 10);
    return end && *end == '\0' && parsed > 0 ? (size_t) parsed : def_value;
}

struct ResidencyInfo {
    bool available = false;
    bool exact = false;
    int error = 0;
    uint64_t page_size = 0;
    uint64_t page_count = 0;
    uint64_t sampled_pages = 0;
    uint64_t resident_pages = 0;
};

#ifdef __linux__
ResidencyInfo query_residency(uintptr_t addr, size_t nbytes) {
    ResidencyInfo info;
    if (!residency_enabled() || addr == 0 || nbytes == 0) {
        return info;
    }

    const long sys_page_size = sysconf(_SC_PAGESIZE);
    if (sys_page_size <= 0) {
        return info;
    }

    const uintptr_t page_size = (uintptr_t) sys_page_size;
    const uintptr_t start = addr & ~(page_size - 1);
    const uintptr_t last = addr + nbytes - 1;
    if (last < addr) {
        return info;
    }
    const uintptr_t end = (last & ~(page_size - 1)) + page_size;
    const uint64_t page_count = (uint64_t) ((end - start) / page_size);
    if (page_count == 0) {
        return info;
    }

    info.available = true;
    info.page_size = (uint64_t) page_size;
    info.page_count = page_count;

    const size_t max_pages = env_size_or_default("LLM_MEM_TRACE_RESIDENCY_MAX_PAGES", 4096);
    if (page_count <= max_pages) {
        std::vector<unsigned char> vec((size_t) page_count);
        if (mincore(reinterpret_cast<void *>(start), (size_t) (end - start), vec.data()) != 0) {
            info.error = errno;
            return info;
        }
        uint64_t resident = 0;
        for (unsigned char v : vec) {
            resident += (v & 1u) ? 1u : 0u;
        }
        info.exact = true;
        info.sampled_pages = page_count;
        info.resident_pages = resident;
        return info;
    }

    uint64_t resident = 0;
    uint64_t sampled = 0;
    const uint64_t samples = max_pages > 0 ? (uint64_t) max_pages : 1;
    for (uint64_t i = 0; i < samples; ++i) {
        const uint64_t idx = samples == 1 ? 0 : (i * (page_count - 1)) / (samples - 1);
        unsigned char vec = 0;
        if (mincore(reinterpret_cast<void *>(start + idx * page_size), (size_t) page_size, &vec) != 0) {
            info.error = errno;
            return info;
        }
        resident += (vec & 1u) ? 1u : 0u;
        ++sampled;
    }

    info.exact = false;
    info.sampled_pages = sampled;
    info.resident_pages = sampled ? (resident * page_count + sampled / 2) / sampled : 0;
    return info;
}
#else
ResidencyInfo query_residency(uintptr_t addr, size_t nbytes) {
    (void) addr;
    (void) nbytes;
    return {};
}
#endif

void append_residency(std::string & line, uintptr_t addr, size_t nbytes) {
    const ResidencyInfo info = query_residency(addr, nbytes);
    if (!info.available) {
        return;
    }
    line += ",\"page_size\":" + std::to_string(info.page_size);
    line += ",\"page_count\":" + std::to_string(info.page_count);
    line += ",\"resident_sample_pages\":" + std::to_string(info.sampled_pages);
    line += ",\"resident_exact\":" + std::string(info.exact ? "true" : "false");
    if (info.error != 0) {
        line += ",\"resident_error\":" + std::to_string(info.error);
        return;
    }
    line += ",\"resident_pages\":" + std::to_string(info.resident_pages);
    line += ",\"resident_bytes\":" + std::to_string(info.resident_pages * info.page_size);
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
        append_residency(line, addr, nbytes);
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
        append_residency(line, addr, nbytes);
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
    append_residency(line, addr, nbytes);
    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_TENSOR, line.c_str(), line.size());
}
