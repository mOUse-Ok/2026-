#include "trace_event.h"
#include "expert_prefetch_types.h"
#include "expert_tensor_registry.h"
#include "expert_prefetch_policy.h"
#include "expert_hint_priority.h"
#include "expert_tensor_stage.h"
#include "expert_task_lifecycle.h"
#include "expert_first_use_matcher.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
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

bool trace_profile_is_benchmark() {
    static const bool benchmark = [] {
        const char * profile = std::getenv("TRACE_PROFILE");
        return profile && std::strcmp(profile, "benchmark") == 0;
    }();
    return benchmark;
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

bool os_hints_enabled() {
    static const bool enabled = env_truthy(std::getenv("LLM_MEM_TRACE_OS_HINTS"));
    return enabled && llm_mem_trace_enabled();
}

bool os_hint_opt_enabled(const char * key) {
    return os_hints_enabled() && env_truthy(std::getenv(key));
}

uint64_t env_u64_or_default(const char * key, uint64_t def_value) {
    const char * val = std::getenv(key);
    if (!val || !val[0]) {
        return def_value;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(val, &end, 10);
    return end && *end == '\0' ? (uint64_t) parsed : def_value;
}

double env_double_or_default(const char * key, double def_value) {
    const char * val = std::getenv(key);
    if (!val || !val[0]) {
        return def_value;
    }
    char * end = nullptr;
    const double parsed = std::strtod(val, &end);
    return end && *end == '\0' && std::isfinite(parsed) ? parsed : def_value;
}

bool env_bool_or_default(const char * key, bool def_value) {
    const char * value = std::getenv(key);
    return value && value[0] ? env_truthy(value) : def_value;
}

bool contains_substring_token(const char * name, const char * start, size_t len) {
    if (!name || !start || len == 0) {
        return false;
    }
    if (len == 1 && start[0] == '*') {
        return true;
    }
    const std::string token(start, len);
    return std::strstr(name, token.c_str()) != nullptr;
}

bool os_hint_target_matches(const char * name) {
    const char * filter = std::getenv("LLM_MEM_TRACE_OPT_TARGETS");
    if (!filter || !filter[0]) {
        filter = "token_embd.weight,output.weight,ffn_down_exps.weight";
    }

    const char * p = filter;
    while (*p) {
        while (*p == ',' || std::isspace((unsigned char) *p)) {
            ++p;
        }
        const char * begin = p;
        while (*p && *p != ',') {
            ++p;
        }
        const char * end = p;
        while (end > begin && std::isspace((unsigned char) *(end - 1))) {
            --end;
        }
        if (contains_substring_token(name, begin, (size_t) (end - begin))) {
            return true;
        }
    }
    return false;
}

bool os_hint_size_allowed(size_t nbytes) {
    const uint64_t max_bytes = env_u64_or_default("LLM_MEM_TRACE_OPT_MAX_BYTES", 512ull * 1024ull * 1024ull);
    return max_bytes == 0 || nbytes <= max_bytes;
}

bool page_aligned_range(uintptr_t addr, size_t nbytes, uintptr_t & start, size_t & len) {
    if (addr == 0 || nbytes == 0) {
        return false;
    }
#ifdef __linux__
    const long sys_page_size = sysconf(_SC_PAGESIZE);
    if (sys_page_size <= 0) {
        return false;
    }
    const uintptr_t page_size = (uintptr_t) sys_page_size;
    start = addr & ~(page_size - 1);
    const uintptr_t last = addr + nbytes - 1;
    if (last < addr) {
        return false;
    }
    const uintptr_t end = (last & ~(page_size - 1)) + page_size;
    len = (size_t) (end - start);
    return len > 0;
#else
    (void) addr;
    (void) nbytes;
    (void) start;
    (void) len;
    return false;
#endif
}

struct OsHintMeta {
    const char * policy = nullptr;
    const char * decision = nullptr;
    uint64_t cache_bytes = 0;
    uint64_t cache_capacity_bytes = 0;
    bool cache_hit = false;
    bool has_cache_hit = false;
    bool has_trace_context = false;
    int phase = LLM_MEM_TRACE_PHASE_UNKNOWN;
    uint64_t step = 0;
    bool has_control = false;
    double route_score = 0.0;
    double route_confidence = 0.0;
    uint64_t enqueue_ts_ns = 0;
    uint64_t deadline_ts_ns = 0;
    uint64_t slack_ns = 0;
    uint64_t predicted_service_ns = 0;
    uint64_t predicted_benefit_ns = 0;
    uint64_t predicted_cost_ns = 0;
    double value_ratio = 0.0;
    const char * pressure_level = nullptr;
    uint64_t memory_current_bytes = 0;
    uint64_t memory_limit_bytes = 0;
    uint64_t prefetch_budget_bytes = 0;
    uint64_t workingset_refault = 0;
    uint64_t refault_delta = 0;
    double psi_some_avg10 = 0.0;
    double psi_full_avg10 = 0.0;
    bool predicted = false;
    int prediction_source_layer = -1;
    int token_idx = -1;
    uint64_t issue_id = 0;
    uint64_t issue_task_count = 0;
};

void write_os_hint_event(
        const char * action,
        const char * trigger,
        const char * tensor_name,
        int layer,
        int expert,
        uintptr_t addr,
        size_t nbytes,
        size_t advised_bytes,
        int result,
        int error_code,
        uint64_t file_offset = 0,
        const OsHintMeta * meta = nullptr) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    char addr_buf[32];
    std::snprintf(addr_buf, sizeof(addr_buf), "0x%llx", (unsigned long long) addr);

    std::string line;
    line.reserve(256);
    const int phase = meta && meta->has_trace_context ? meta->phase : llm_mem_trace_get_phase();
    const uint64_t step = meta && meta->has_trace_context ? meta->step : llm_mem_trace_get_step();
    line += "{\"event\":\"OS_HINT\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
    line += ",\"phase\":\"" + std::string(phase_name(phase)) + "\"";
    line += ",\"step\":" + std::to_string(step);
    line += ",\"action\":";
    json_escape_append(line, action ? action : "");
    line += ",\"trigger\":";
    json_escape_append(line, trigger ? trigger : "");
    line += ",\"tensor\":";
    json_escape_append(line, tensor_name ? tensor_name : "");
    if (layer >= 0) {
        line += ",\"layer\":" + std::to_string(layer);
    }
    if (expert >= 0) {
        line += ",\"expert\":" + std::to_string(expert);
    }
    line += ",\"addr\":";
    json_escape_append(line, addr_buf);
    line += ",\"size\":" + std::to_string(nbytes);
    line += ",\"advised_bytes\":" + std::to_string(advised_bytes);
    if (meta && meta->policy && meta->policy[0]) {
        line += ",\"policy\":";
        json_escape_append(line, meta->policy);
    }
    if (meta && meta->decision && meta->decision[0]) {
        line += ",\"decision\":";
        json_escape_append(line, meta->decision);
    }
    if (meta) {
        line += ",\"cache_bytes\":" + std::to_string(meta->cache_bytes);
        line += ",\"cache_capacity_bytes\":" + std::to_string(meta->cache_capacity_bytes);
        if (meta->has_cache_hit) {
            line += ",\"cache_hit\":" + std::string(meta->cache_hit ? "true" : "false");
        }
        if (meta->has_control) {
            line += ",\"route_score\":" + std::to_string(meta->route_score);
            line += ",\"route_confidence\":" + std::to_string(meta->route_confidence);
            line += ",\"enqueue_ts_ns\":" + std::to_string(meta->enqueue_ts_ns);
            line += ",\"deadline_ts_ns\":" + std::to_string(meta->deadline_ts_ns);
            line += ",\"slack_ns\":" + std::to_string(meta->slack_ns);
            line += ",\"predicted_service_ns\":" + std::to_string(meta->predicted_service_ns);
            line += ",\"predicted_benefit_ns\":" + std::to_string(meta->predicted_benefit_ns);
            line += ",\"predicted_cost_ns\":" + std::to_string(meta->predicted_cost_ns);
            line += ",\"value_ratio\":" + std::to_string(meta->value_ratio);
            line += ",\"pressure_level\":";
            json_escape_append(line, meta->pressure_level ? meta->pressure_level : "unknown");
            line += ",\"memory_current_bytes\":" + std::to_string(meta->memory_current_bytes);
            line += ",\"memory_limit_bytes\":" + std::to_string(meta->memory_limit_bytes);
            line += ",\"prefetch_budget_bytes\":" + std::to_string(meta->prefetch_budget_bytes);
            line += ",\"workingset_refault\":" + std::to_string(meta->workingset_refault);
            line += ",\"refault_delta\":" + std::to_string(meta->refault_delta);
            line += ",\"psi_some_avg10\":" + std::to_string(meta->psi_some_avg10);
            line += ",\"psi_full_avg10\":" + std::to_string(meta->psi_full_avg10);
            line += ",\"predicted\":" + std::string(meta->predicted ? "true" : "false");
            if (meta->prediction_source_layer >= 0) {
                line += ",\"prediction_source_layer\":" + std::to_string(meta->prediction_source_layer);
            }
            if (meta->token_idx >= 0) {
                line += ",\"token_idx\":" + std::to_string(meta->token_idx);
            }
        }
        if (meta->issue_id != 0) {
            line += ",\"issue_id\":" + std::to_string(meta->issue_id);
            line += ",\"issue_task_count\":" + std::to_string(meta->issue_task_count);
        }
    }
    if (file_offset != 0) {
        line += ",\"file_offset\":" + std::to_string(file_offset);
    }
    line += ",\"result\":" + std::to_string(result);
    line += ",\"errno\":" + std::to_string(error_code);
    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void apply_madvise_hint(
        const char * action,
        int advice,
        const char * trigger,
        const char * tensor_name,
        int layer,
        int expert,
        uintptr_t addr,
        size_t nbytes,
        const OsHintMeta * meta = nullptr) {
#ifdef __linux__
    uintptr_t start = 0;
    size_t len = 0;
    if (!page_aligned_range(addr, nbytes, start, len)) {
        return;
    }
    errno = 0;
    const int rc = madvise(reinterpret_cast<void *>(start), len, advice);
    const int err = rc == 0 ? 0 : errno;
    write_os_hint_event(action, trigger, tensor_name, layer, expert, addr, nbytes, len, rc, err, 0, meta);
#else
    (void) action; (void) advice; (void) trigger; (void) tensor_name; (void) layer; (void) expert; (void) addr; (void) nbytes; (void) meta;
#endif
}

#ifdef __linux__
struct FileMapping {
    uintptr_t start = 0;
    uintptr_t end = 0;
    uint64_t offset = 0;
    std::string path;
};

bool find_file_mapping(uintptr_t addr, FileMapping & out) {
    FILE * fp = std::fopen("/proc/self/maps", "r");
    if (!fp) {
        return false;
    }

    char line[4096];
    while (std::fgets(line, sizeof(line), fp)) {
        unsigned long long start = 0;
        unsigned long long end = 0;
        unsigned long long offset = 0;
        char perms[8] = {};
        int path_pos = 0;
        const int scanned = std::sscanf(line, "%llx-%llx %7s %llx %*s %*s %n", &start, &end, perms, &offset, &path_pos);
        if (scanned < 4 || addr < (uintptr_t) start || addr >= (uintptr_t) end) {
            continue;
        }
        std::fclose(fp);
        if (std::strchr(perms, 'r') == nullptr || path_pos <= 0 || line[path_pos] == '\0') {
            return false;
        }
        char * path = line + path_pos;
        size_t path_len = std::strlen(path);
        while (path_len > 0 && (path[path_len - 1] == '\n' || path[path_len - 1] == '\r')) {
            path[--path_len] = '\0';
        }
        if (path_len == 0 || path[0] == '[') {
            return false;
        }
        out.start = (uintptr_t) start;
        out.end = (uintptr_t) end;
        out.offset = (uint64_t) offset;
        out.path = path;
        return true;
    }

    std::fclose(fp);
    return false;
}
#endif

void apply_posix_fadvise_hint(
        const char * action,
        const char * trigger,
        const char * tensor_name,
        int layer,
        int expert,
        uintptr_t addr,
        size_t nbytes,
        const OsHintMeta * meta = nullptr) {
#ifdef __linux__
    FileMapping mapping;
    if (!find_file_mapping(addr, mapping)) {
        write_os_hint_event(action, trigger, tensor_name, layer, expert, addr, nbytes, 0, -1, ENOENT, 0, meta);
        return;
    }
    if (addr >= mapping.end) {
        return;
    }
    const uint64_t file_offset = mapping.offset + (uint64_t) (addr - mapping.start);
    const size_t max_len = (size_t) (mapping.end - addr);
    const size_t advise_len = std::min(nbytes, max_len);
    const int fd = open(mapping.path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        write_os_hint_event(action, trigger, tensor_name, layer, expert, addr, nbytes, advise_len, -1, errno, file_offset, meta);
        return;
    }
    const int rc = posix_fadvise(fd, (off_t) file_offset, (off_t) advise_len, POSIX_FADV_WILLNEED);
    close(fd);
    write_os_hint_event(action, trigger, tensor_name, layer, expert, addr, nbytes, advise_len, rc == 0 ? 0 : -1, rc, file_offset, meta);
#else
    (void) action; (void) trigger; (void) tensor_name; (void) layer; (void) expert; (void) addr; (void) nbytes; (void) meta;
#endif
}

void apply_load_os_hints(
        const char * trigger,
        const char * tensor_name,
        int layer,
        uintptr_t addr,
        size_t nbytes,
        bool mapped_tensor) {
    if (!os_hints_enabled() || !mapped_tensor || !os_hint_target_matches(tensor_name) || !os_hint_size_allowed(nbytes)) {
        return;
    }

#ifdef __linux__
    if (os_hint_opt_enabled("LLM_MEM_TRACE_OPT_MADVISE_SEQUENTIAL")) {
        apply_madvise_hint("madvise_sequential", MADV_SEQUENTIAL, trigger, tensor_name, layer, -1, addr, nbytes);
    }
#ifdef MADV_HUGEPAGE
    if (os_hint_opt_enabled("LLM_MEM_TRACE_OPT_THP")) {
        apply_madvise_hint("madvise_hugepage", MADV_HUGEPAGE, trigger, tensor_name, layer, -1, addr, nbytes);
    }
#endif
    if (os_hint_opt_enabled("LLM_MEM_TRACE_OPT_MADVISE_WILLNEED")) {
        apply_madvise_hint("madvise_willneed", MADV_WILLNEED, trigger, tensor_name, layer, -1, addr, nbytes);
    }
#endif
    if (os_hint_opt_enabled("LLM_MEM_TRACE_OPT_POSIX_FADVISE")) {
        apply_posix_fadvise_hint("posix_fadvise_willneed", trigger, tensor_name, layer, -1, addr, nbytes);
    }
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

bool is_expert_weight_tensor_name(const char * name) {
    return name &&
           std::strstr(name, "blk.") &&
           std::strstr(name, "_exps.weight") &&
           (std::strstr(name, "ffn_gate_exps.weight") ||
            std::strstr(name, "ffn_up_exps.weight") ||
            std::strstr(name, "ffn_down_exps.weight") ||
            std::strstr(name, "ffn_gate_up_exps.weight"));
}

/*
struct ExpertTensorInfo {
    std::string name;
    int layer = -1;
    uintptr_t addr = 0;
    size_t nbytes = 0;
    int64_t n_expert = 0;
    size_t expert_stride = 0;
};

struct ExpertTensorRegistry {
    std::mutex mu;
    std::vector<ExpertTensorInfo> tensors;
    std::unordered_set<std::string> hinted;
    std::unordered_map<std::string, uint64_t> recent_hints;
    uint64_t route_hint_ttl_steps_config = 0;
    uint64_t route_hint_candidates = 0;
    uint64_t route_hint_issued = 0;
    uint64_t route_hint_skipped = 0;
    uint64_t route_hint_duplicate_skipped = 0;
    uint64_t route_hint_ttl_skipped = 0;

    void add(const ggml_tensor * t, const char * name, int layer, uintptr_t addr, size_t nbytes) {
        if (!t || layer < 0 || addr == 0 || nbytes == 0 || !is_expert_weight_tensor_name(name)) {
            return;
        }
        const int64_t n_expert = t->ne[2];
        const size_t expert_stride = (size_t) t->nb[2];
        if (n_expert <= 0 || expert_stride == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(mu);
        for (const ExpertTensorInfo & info : tensors) {
            if (info.addr == addr && info.nbytes == nbytes) {
                return;
            }
        }
        tensors.push_back({name ? name : "", layer, addr, nbytes, n_expert, expert_stride});
    }

    std::vector<ExpertTensorInfo> for_layer(int layer) {
        std::vector<ExpertTensorInfo> out;
        std::lock_guard<std::mutex> lock(mu);
        for (const ExpertTensorInfo & info : tensors) {
            if (info.layer == layer) {
                out.push_back(info);
            }
        }
        return out;
    }

    bool was_hinted(uint64_t step, int layer, int expert, uintptr_t addr, uint64_t ttl_steps) {
        const std::string slice_key = std::to_string(layer) + ":" + std::to_string(expert) + ":" +
                                      std::to_string((uint64_t) addr);
        std::lock_guard<std::mutex> lock(mu);
        if (ttl_steps > 0) {
            auto it = recent_hints.find(slice_key);
            return it != recent_hints.end() && step >= it->second && step - it->second <= ttl_steps;
        }
        return hinted.find(std::to_string(step) + ":" + slice_key) != hinted.end();
    }

    bool mark_hinted(uint64_t step, int layer, int expert, uintptr_t addr, uint64_t ttl_steps) {
        const std::string slice_key = std::to_string(layer) + ":" + std::to_string(expert) + ":" +
                                      std::to_string((uint64_t) addr);
        std::lock_guard<std::mutex> lock(mu);
        route_hint_ttl_steps_config = std::max(route_hint_ttl_steps_config, ttl_steps);
        route_hint_candidates++;
        if (ttl_steps > 0) {
            auto it = recent_hints.find(slice_key);
            if (it != recent_hints.end() && step >= it->second && step - it->second <= ttl_steps) {
                route_hint_skipped++;
                if (step == it->second) {
                    route_hint_duplicate_skipped++;
                } else {
                    route_hint_ttl_skipped++;
                }
                return false;
            }
            recent_hints[slice_key] = step;
            route_hint_issued++;
            return true;
        }
        std::string key = std::to_string(step) + ":" + slice_key;
        const bool inserted = hinted.insert(std::move(key)).second;
        if (inserted) {
            route_hint_issued++;
        } else {
            route_hint_skipped++;
            route_hint_duplicate_skipped++;
        }
        return inserted;
    }

    void write_route_hint_summary() {
        if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
            return;
        }
        uint64_t ttl_steps = 0;
        uint64_t candidates = 0;
        uint64_t issued = 0;
        uint64_t skipped = 0;
        uint64_t duplicate_skipped = 0;
        uint64_t ttl_skipped = 0;
        {
            std::lock_guard<std::mutex> lock(mu);
            candidates = route_hint_candidates;
            if (candidates == 0) {
                return;
            }
            ttl_steps = route_hint_ttl_steps_config;
            issued = route_hint_issued;
            skipped = route_hint_skipped;
            duplicate_skipped = route_hint_duplicate_skipped;
            ttl_skipped = route_hint_ttl_skipped;
        }

        std::string line;
        line.reserve(256);
        line += "{\"event\":\"EXPERT_ROUTE_HINT_SUMMARY\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
        line += ",\"ttl_steps\":" + std::to_string(ttl_steps);
        line += ",\"candidates\":" + std::to_string(candidates);
        line += ",\"issued\":" + std::to_string(issued);
        line += ",\"skipped\":" + std::to_string(skipped);
        line += ",\"duplicate_skipped\":" + std::to_string(duplicate_skipped);
        line += ",\"ttl_skipped\":" + std::to_string(ttl_skipped);
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }
};

ExpertTensorRegistry & expert_tensor_registry() {
    static ExpertTensorRegistry registry;
    return registry;
}

bool expert_slice_range(const ExpertTensorInfo & info, int expert, uintptr_t & addr, size_t & nbytes) {
    if (expert < 0 || expert >= info.n_expert || info.addr == 0 || info.expert_stride == 0) {
        return false;
    }
    const size_t offset = (size_t) expert * info.expert_stride;
    if (offset >= info.nbytes) {
        return false;
    }
    addr = info.addr + offset;
    nbytes = std::min(info.expert_stride, info.nbytes - offset);
    return nbytes > 0;
}
*/

/* moved to expert_prefetch_types.h
enum class ExpertPolicy {
    Route,
    Lru,
    Lfu,
    WindowLfu,
    LeastStale,
}; */

ExpertPolicy expert_policy() {
    static const ExpertPolicy policy = [] {
        const char * value = std::getenv("LLM_MEM_TRACE_OPT_EXPERT_POLICY");
        if (!value || !value[0] || std::strcmp(value, "route") == 0) {
            return ExpertPolicy::Route;
        }
        if (std::strcmp(value, "lru") == 0) {
            return ExpertPolicy::Lru;
        }
        if (std::strcmp(value, "lfu") == 0) {
            return ExpertPolicy::Lfu;
        }
        if (std::strcmp(value, "window_lfu") == 0) {
            return ExpertPolicy::WindowLfu;
        }
        if (std::strcmp(value, "least_stale") == 0) {
            return ExpertPolicy::LeastStale;
        }
        return ExpertPolicy::Route;
    }();
    return policy;
}

/* moved to expert_prefetch_policy.cpp
const char * expert_policy_name(ExpertPolicy policy) {
    switch (policy) {
        case ExpertPolicy::Route:      return "route";
        case ExpertPolicy::Lru:        return "lru";
        case ExpertPolicy::Lfu:        return "lfu";
        case ExpertPolicy::WindowLfu:  return "window_lfu";
        case ExpertPolicy::LeastStale: return "least_stale";
    }
    return "route";
} */

/* moved to expert_prefetch_types.h
enum class ExpertEvictAdvice {
    None,
    Cold,
    DontNeed,
    PageOut,
}; */

ExpertEvictAdvice expert_evict_advice() {
    static const ExpertEvictAdvice advice = [] {
        const char * value = std::getenv("LLM_MEM_TRACE_OPT_EXPERT_EVICT");
        if (!value || !value[0] || std::strcmp(value, "cold") == 0) {
            return ExpertEvictAdvice::Cold;
        }
        if (std::strcmp(value, "none") == 0) {
            return ExpertEvictAdvice::None;
        }
        if (std::strcmp(value, "dontneed") == 0) {
            return ExpertEvictAdvice::DontNeed;
        }
        if (std::strcmp(value, "pageout") == 0) {
            return ExpertEvictAdvice::PageOut;
        }
        return ExpertEvictAdvice::Cold;
    }();
    return advice;
}

uint64_t expert_cache_capacity_bytes() {
    const uint64_t mib = 1024ull * 1024ull;
    const uint64_t mb = env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_CACHE_MB", 512);
    if (mb > std::numeric_limits<uint64_t>::max() / mib) {
        return std::numeric_limits<uint64_t>::max();
    }
    return mb * mib;
}

uint64_t expert_ttl_steps() {
    return env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_TTL_STEPS", 4);
}

uint64_t expert_route_hint_ttl_steps() {
    return env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS", 0);
}

uint64_t env_u64_or_inherit(const char * key, uint64_t def_value) {
    const char * value = std::getenv(key);
    if (!value || !value[0]) {
        return def_value;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return end && *end == '\0' ? (uint64_t) parsed : def_value;
}

uint64_t expert_route_hint_ttl_steps_for_phase(int phase) {
    const uint64_t global_ttl = expert_route_hint_ttl_steps();
    if (phase == LLM_MEM_TRACE_PHASE_PREFILL) {
        return env_u64_or_inherit("LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_PREFILL_STEPS", global_ttl);
    }
    if (phase == LLM_MEM_TRACE_PHASE_DECODE) {
        return env_u64_or_inherit("LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_DECODE_STEPS", global_ttl);
    }
    return global_ttl;
}

int expert_prefetch_topk() {
    const uint64_t value = env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK", 0);
    return value > (uint64_t) std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : (int) value;
}

int env_topk_or_default(const char * key, int def_value) {
    const char * value = std::getenv(key);
    if (!value || !value[0]) {
        return def_value;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (!end || *end != '\0') {
        return def_value;
    }
    return parsed > (unsigned long long) std::numeric_limits<int>::max() ?
            std::numeric_limits<int>::max() : (int) parsed;
}

int expert_prefetch_topk_for_phase(int phase) {
    const int global_topk = expert_prefetch_topk();
    if (phase == LLM_MEM_TRACE_PHASE_PREFILL) {
        return env_topk_or_default("LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_PREFILL_TOPK", global_topk);
    }
    if (phase == LLM_MEM_TRACE_PHASE_DECODE) {
        return env_topk_or_default("LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_DECODE_TOPK", global_topk);
    }
    return global_topk;
}

bool expert_prefetch_coalesce_enabled() {
    static const bool enabled = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_COALESCE");
    return enabled;
}

bool expert_prefetch_async_enabled() {
    static const bool enabled = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_ASYNC");
    return enabled;
}

size_t expert_prefetch_async_queue_capacity() {
    static const size_t value = env_size_or_default("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE", 65536);
    return value;
}

size_t expert_prefetch_async_workers() {
    static const size_t value = env_size_or_default("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS", 1);
    return std::max<size_t>(1, value);
}

bool expert_prefetch_async_priority_enabled() {
    static const bool enabled = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY");
    return enabled;
}

bool expert_prefetch_async_priority_heap_enabled() {
    static const bool enabled = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP");
    return enabled;
}

size_t expert_prefetch_async_batch_size() {
    static const size_t value = env_size_or_default("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH", 1);
    return std::max<size_t>(1, std::min<size_t>(value, 256));
}

uint64_t expert_prefetch_async_batch_wait_us() {
    static const uint64_t value = env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_WAIT_US", 100);
    return std::min<uint64_t>(value, 10000);
}

bool expert_prefetch_async_batch_coalesce_enabled() {
    static const bool enabled = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_COALESCE");
    return enabled;
}

bool expert_prefetch_async_fallback_enabled() {
    static const bool enabled = env_bool_or_default("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK", true);
    return enabled;
}

/* moved to expert_prefetch_types.h
enum class ExpertAsyncPriorityMode {
    Score,
    Deadline,
    DeadlineScore,
    StageDeadlineScore,
}; */

ExpertAsyncPriorityMode expert_prefetch_async_priority_mode() {
    static const ExpertAsyncPriorityMode mode = [] {
        const char * value = std::getenv("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE");
        if (!value || !value[0] || std::strcmp(value, "score") == 0) {
            return ExpertAsyncPriorityMode::Score;
        }
        if (std::strcmp(value, "deadline") == 0) {
            return ExpertAsyncPriorityMode::Deadline;
        }
        if (std::strcmp(value, "deadline_score") == 0) {
            return ExpertAsyncPriorityMode::DeadlineScore;
        }
        if (std::strcmp(value, "stage_deadline_score") == 0) {
            return ExpertAsyncPriorityMode::StageDeadlineScore;
        }
        return ExpertAsyncPriorityMode::Score;
    }();
    return mode;
}

/* moved to expert_prefetch_policy.cpp
const char * expert_prefetch_async_priority_mode_name(ExpertAsyncPriorityMode mode) {
    switch (mode) {
        case ExpertAsyncPriorityMode::Score:         return "score";
        case ExpertAsyncPriorityMode::Deadline:      return "deadline";
        case ExpertAsyncPriorityMode::DeadlineScore: return "deadline_score";
        case ExpertAsyncPriorityMode::StageDeadlineScore: return "stage_deadline_score";
    }
    return "score";
} */

uint64_t expert_prefetch_coalesce_max_gap_bytes() {
    static const uint64_t value = env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_COALESCE_MAX_GAP_BYTES", 0);
    return value;
}

bool expert_feedback_enabled() {
    static const bool enabled = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK");
    return enabled;
}

bool expert_slack_enabled() {
    static const bool enabled = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_SLACK");
    return enabled;
}

bool expert_deadline_observation_enabled() {
    static const bool enabled = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE");
    return enabled;
}

bool expert_value_gate_enabled() {
    static const bool enabled = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE");
    return enabled;
}

/* moved to expert_prefetch_types.h
enum class ExpertPressureLevel {
    Low = 0,
    Moderate = 1,
    High = 2,
    Critical = 3,
}; */

/* moved to expert_prefetch_policy.cpp
const char * expert_pressure_level_name(ExpertPressureLevel level) {
    switch (level) {
        case ExpertPressureLevel::Low:      return "low";
        case ExpertPressureLevel::Moderate: return "moderate";
        case ExpertPressureLevel::High:     return "high";
        case ExpertPressureLevel::Critical: return "critical";
    }
    return "low";
} */

struct ExpertPressureSnapshot {
    ExpertPressureLevel level = ExpertPressureLevel::Low;
    uint64_t sampled_ts_ns = 0;
    uint64_t memory_current_bytes = 0;
    uint64_t memory_limit_bytes = 0;
    uint64_t swap_current_bytes = 0;
    uint64_t prefetch_budget_bytes = 0;
    uint64_t workingset_refault = 0;
    uint64_t refault_delta = 0;
    double memory_ratio_pct = 0.0;
    double psi_some_avg10 = 0.0;
    double psi_full_avg10 = 0.0;
    bool available = false;
};

#ifdef __linux__
bool read_small_file(const std::string & path, std::string & out) {
    FILE * fp = std::fopen(path.c_str(), "r");
    if (!fp) {
        return false;
    }
    char buffer[4096];
    out.clear();
    while (std::fgets(buffer, sizeof(buffer), fp)) {
        out += buffer;
        if (out.size() >= 16384) {
            break;
        }
    }
    std::fclose(fp);
    return true;
}

bool parse_u64_file_value(const std::string & path, uint64_t & value) {
    std::string text;
    if (!read_small_file(path, text) || text.empty() || text.compare(0, 3, "max") == 0) {
        return false;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (end == text.c_str()) {
        return false;
    }
    value = (uint64_t) parsed;
    return true;
}

std::string current_cgroup_v2_dir() {
    std::string text;
    if (!read_small_file("/proc/self/cgroup", text)) {
        return {};
    }
    const std::string marker = "0::";
    const size_t pos = text.find(marker);
    if (pos == std::string::npos) {
        return {};
    }
    size_t end = text.find('\n', pos);
    std::string relative = text.substr(pos + marker.size(), end - pos - marker.size());
    while (!relative.empty() && (relative.back() == '\r' || relative.back() == '\n')) {
        relative.pop_back();
    }
    if (relative.empty() || relative == "/") {
        return "/sys/fs/cgroup";
    }
    return "/sys/fs/cgroup/" + (relative.front() == '/' ? relative.substr(1) : relative);
}

double parse_psi_avg10(const std::string & text, const char * category) {
    const std::string marker = std::string(category) + " ";
    const size_t line = text.find(marker);
    if (line == std::string::npos) {
        return 0.0;
    }
    const size_t key = text.find("avg10=", line + marker.size());
    if (key == std::string::npos) {
        return 0.0;
    }
    const char * begin = text.c_str() + key + 6;
    char * end = nullptr;
    const double value = std::strtod(begin, &end);
    return end != begin && std::isfinite(value) ? value : 0.0;
}

uint64_t parse_memory_stat_refault(const std::string & text) {
    uint64_t total = 0;
    size_t line_start = 0;
    while (line_start < text.size()) {
        const size_t line_end = text.find('\n', line_start);
        const size_t length = (line_end == std::string::npos ? text.size() : line_end) - line_start;
        const std::string line = text.substr(line_start, length);
        if (line.compare(0, 24, "workingset_refault_anon ") == 0 ||
                line.compare(0, 24, "workingset_refault_file ") == 0) {
            const size_t separator = line.find(' ');
            if (separator != std::string::npos) {
                total += (uint64_t) std::strtoull(line.c_str() + separator + 1, nullptr, 10);
            }
        }
        if (line_end == std::string::npos) {
            break;
        }
        line_start = line_end + 1;
    }
    return total;
}
#endif

struct ExpertPressureController {
    std::mutex mu;
    ExpertPressureSnapshot last;
    std::string cgroup_dir;

    ExpertPressureSnapshot snapshot(bool force = false) {
        const uint64_t base_budget = expert_cache_capacity_bytes();
        if (!expert_feedback_enabled()) {
            ExpertPressureSnapshot out;
            out.prefetch_budget_bytes = base_budget;
            return out;
        }

        const uint64_t now = llm_mem_trace_time_ns();
        const uint64_t interval_ns = env_u64_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_PRESSURE_SAMPLE_MS", 50) * 1000000ull;
        std::lock_guard<std::mutex> lock(mu);
        if (!force && last.sampled_ts_ns != 0 && now >= last.sampled_ts_ns &&
                now - last.sampled_ts_ns < interval_ns) {
            return last;
        }

        ExpertPressureSnapshot next;
        next.sampled_ts_ns = now;
        next.prefetch_budget_bytes = base_budget;
#ifdef __linux__
        if (cgroup_dir.empty()) {
            cgroup_dir = current_cgroup_v2_dir();
        }
        if (!cgroup_dir.empty()) {
            next.available = parse_u64_file_value(cgroup_dir + "/memory.current", next.memory_current_bytes);
            uint64_t high = 0;
            uint64_t maximum = 0;
            const bool have_high = parse_u64_file_value(cgroup_dir + "/memory.high", high);
            const bool have_max = parse_u64_file_value(cgroup_dir + "/memory.max", maximum);
            next.memory_limit_bytes = have_high ? high : (have_max ? maximum : 0);
            (void) parse_u64_file_value(cgroup_dir + "/memory.swap.current", next.swap_current_bytes);
            std::string pressure;
            if (read_small_file(cgroup_dir + "/memory.pressure", pressure)) {
                next.psi_some_avg10 = parse_psi_avg10(pressure, "some");
                next.psi_full_avg10 = parse_psi_avg10(pressure, "full");
                next.available = true;
            }
            std::string memory_stat;
            if (read_small_file(cgroup_dir + "/memory.stat", memory_stat)) {
                next.workingset_refault = parse_memory_stat_refault(memory_stat);
                if (last.sampled_ts_ns != 0 && next.workingset_refault >= last.workingset_refault) {
                    next.refault_delta = next.workingset_refault - last.workingset_refault;
                }
            }
        }
#endif
        if (next.memory_limit_bytes > 0) {
            next.memory_ratio_pct = 100.0 * (double) next.memory_current_bytes /
                                    (double) next.memory_limit_bytes;
        }

        const double moderate_pct = env_double_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_PRESSURE_MODERATE_PCT", 75.0);
        const double high_pct = env_double_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_PRESSURE_HIGH_PCT", 88.0);
        const double critical_pct = env_double_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_PRESSURE_CRITICAL_PCT", 96.0);
        const double psi_moderate = env_double_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_PSI_SOME_MODERATE", 0.5);
        const double psi_high = env_double_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_PSI_SOME_HIGH", 2.0);
        const double psi_critical = env_double_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_PSI_FULL_CRITICAL", 1.0);

        if (next.memory_ratio_pct >= critical_pct || next.psi_full_avg10 >= psi_critical) {
            next.level = ExpertPressureLevel::Critical;
        } else if (next.memory_ratio_pct >= high_pct || next.psi_some_avg10 >= psi_high) {
            next.level = ExpertPressureLevel::High;
        } else if (next.memory_ratio_pct >= moderate_pct || next.psi_some_avg10 >= psi_moderate) {
            next.level = ExpertPressureLevel::Moderate;
        }
        const uint64_t refault_moderate = env_u64_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_REFAULT_MODERATE", 64);
        const uint64_t refault_high = env_u64_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_REFAULT_HIGH", 1024);
        const uint64_t refault_critical = env_u64_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_REFAULT_CRITICAL", 8192);
        if (next.refault_delta >= refault_critical) {
            next.level = ExpertPressureLevel::Critical;
        } else if (next.refault_delta >= refault_high && next.level < ExpertPressureLevel::High) {
            next.level = ExpertPressureLevel::High;
        } else if (next.refault_delta >= refault_moderate && next.level < ExpertPressureLevel::Moderate) {
            next.level = ExpertPressureLevel::Moderate;
        }

        uint64_t budget_pct = 100;
        switch (next.level) {
            case ExpertPressureLevel::Low:
                budget_pct = 100;
                break;
            case ExpertPressureLevel::Moderate:
                budget_pct = env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_BUDGET_MODERATE_PCT", 75);
                break;
            case ExpertPressureLevel::High:
                budget_pct = env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_BUDGET_HIGH_PCT", 50);
                break;
            case ExpertPressureLevel::Critical:
                budget_pct = env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_BUDGET_CRITICAL_PCT", 20);
                break;
        }
        budget_pct = std::min<uint64_t>(budget_pct, 100);
        next.prefetch_budget_bytes = base_budget / 100 * budget_pct;
        last = next;
        write_event_unlocked(next);
        return last;
    }

    void write_event_unlocked(const ExpertPressureSnapshot & value) const {
        if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
            return;
        }
        std::string line;
        line.reserve(320);
        line += "{\"event\":\"EXPERT_PRESSURE\",\"ts_ns\":" + std::to_string(value.sampled_ts_ns);
        line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
        line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
        line += ",\"available\":" + std::string(value.available ? "true" : "false");
        line += ",\"level\":\"" + std::string(expert_pressure_level_name(value.level)) + "\"";
        line += ",\"memory_current_bytes\":" + std::to_string(value.memory_current_bytes);
        line += ",\"memory_limit_bytes\":" + std::to_string(value.memory_limit_bytes);
        line += ",\"memory_ratio_pct\":" + std::to_string(value.memory_ratio_pct);
        line += ",\"swap_current_bytes\":" + std::to_string(value.swap_current_bytes);
        line += ",\"psi_some_avg10\":" + std::to_string(value.psi_some_avg10);
        line += ",\"psi_full_avg10\":" + std::to_string(value.psi_full_avg10);
        line += ",\"prefetch_budget_bytes\":" + std::to_string(value.prefetch_budget_bytes);
        line += ",\"workingset_refault\":" + std::to_string(value.workingset_refault);
        line += ",\"refault_delta\":" + std::to_string(value.refault_delta);
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }
};

ExpertPressureController & expert_pressure_controller() {
    static ExpertPressureController controller;
    return controller;
}

struct ExpertTimingModel {
    std::mutex mu;
    uint64_t active_step = 0;
    int active_layer = -1;
    int active_phase = LLM_MEM_TRACE_PHASE_UNKNOWN;
    uint64_t active_begin_ns = 0;
    double prefill_layer_ewma_ns = 0.0;
    double decode_layer_ewma_ns = 0.0;
    double syscall_ewma_ns = 0.0;

    double default_layer_ns() const {
        return env_double_or_default("LLM_MEM_TRACE_OPT_EXPERT_SLACK_DEFAULT_LAYER_US", 5000.0) * 1000.0;
    }

    void on_layer_begin(uint64_t step, int layer, int phase, uint64_t ts) {
        std::lock_guard<std::mutex> lock(mu);
        active_step = step;
        active_layer = layer;
        active_phase = phase;
        active_begin_ns = ts;
    }

    void on_layer_end(uint64_t step, int layer, int phase, uint64_t ts) {
        std::lock_guard<std::mutex> lock(mu);
        if (active_step != step || active_layer != layer || active_begin_ns == 0 || ts <= active_begin_ns) {
            return;
        }
        const double duration = (double) (ts - active_begin_ns);
        double & ewma = phase == LLM_MEM_TRACE_PHASE_DECODE ? decode_layer_ewma_ns : prefill_layer_ewma_ns;
        ewma = ewma == 0.0 ? duration : ewma * 0.8 + duration * 0.2;
        active_begin_ns = 0;
    }

    uint64_t estimate_slack_ns(uint64_t step, int target_layer, int phase, uint64_t now) {
        std::lock_guard<std::mutex> lock(mu);
        const double average = phase == LLM_MEM_TRACE_PHASE_DECODE ? decode_layer_ewma_ns : prefill_layer_ewma_ns;
        const double layer_ns = average > 0.0 ? average : default_layer_ns();
        double remaining = layer_ns;
        if (active_step == step && active_begin_ns > 0 && now >= active_begin_ns) {
            remaining = std::max(0.0, layer_ns - (double) (now - active_begin_ns));
            if (target_layer > active_layer) {
                // The target must be ready when its layer begins, not when that
                // layer finishes. Only complete layers in between add slack.
                remaining += (double) std::max(0, target_layer - active_layer - 1) * layer_ns;
            }
        }
        const double margin_pct = std::max(1.0, std::min(100.0,
                env_double_or_default("LLM_MEM_TRACE_OPT_EXPERT_SLACK_MARGIN_PCT", 80.0)));
        const double minimum = env_double_or_default("LLM_MEM_TRACE_OPT_EXPERT_SLACK_MIN_US", 250.0) * 1000.0;
        return (uint64_t) std::max(minimum, remaining * margin_pct / 100.0);
    }

    uint64_t predicted_transfer_ns(size_t nbytes) const {
        const double mbps = std::max(1.0,
                env_double_or_default("LLM_MEM_TRACE_OPT_EXPERT_PAGEIN_MBPS", 512.0));
        return (uint64_t) ((double) nbytes * 1e9 / (mbps * 1024.0 * 1024.0));
    }

    uint64_t predicted_syscall_ns() {
        std::lock_guard<std::mutex> lock(mu);
        if (syscall_ewma_ns > 0.0) {
            return (uint64_t) syscall_ewma_ns;
        }
        return (uint64_t) (env_double_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SYSCALL_DEFAULT_US", 50.0) * 1000.0);
    }

    void observe_syscall(uint64_t duration_ns) {
        std::lock_guard<std::mutex> lock(mu);
        const double value = (double) duration_ns;
        syscall_ewma_ns = syscall_ewma_ns == 0.0 ? value : syscall_ewma_ns * 0.8 + value * 0.2;
    }
};

ExpertTimingModel & expert_timing_model() {
    static ExpertTimingModel model;
    return model;
}

std::string expert_slice_key(const ExpertTensorInfo & info, int expert) {
    return std::to_string(info.layer) + ":" + std::to_string(expert) + ":" +
           std::to_string((uint64_t) info.addr) + ":" + info.name;
}

void write_expert_cache_event(
        const char * action,
        const char * trigger,
        const char * policy,
        const char * decision,
        bool cache_hit,
        const char * tensor_name,
        int layer,
        int expert,
        uintptr_t addr,
        size_t nbytes,
        uint64_t cache_bytes,
        uint64_t cache_capacity_bytes) {
    OsHintMeta meta;
    meta.policy = policy;
    meta.decision = decision;
    meta.cache_bytes = cache_bytes;
    meta.cache_capacity_bytes = cache_capacity_bytes;
    meta.cache_hit = cache_hit;
    meta.has_cache_hit = true;
    write_os_hint_event(action, trigger, tensor_name, layer, expert, addr, nbytes, 0, 0, 0, 0, &meta);
}

void apply_expert_prefetch_hint(
        const ExpertTensorInfo & info,
        int expert,
        uintptr_t addr,
        size_t nbytes,
        const char * reason,
        const char * policy,
        uint64_t cache_bytes,
        uint64_t cache_capacity_bytes) {
    OsHintMeta meta;
    meta.policy = policy;
    meta.decision = "prefetch";
    meta.cache_bytes = cache_bytes;
    meta.cache_capacity_bytes = cache_capacity_bytes;
    meta.cache_hit = false;
    meta.has_cache_hit = true;
#ifdef __linux__
    apply_madvise_hint("expert_madvise_willneed", MADV_WILLNEED,
                       reason ? reason : "expert_prefetch",
                       info.name.c_str(), info.layer, expert, addr, nbytes, &meta);
#else
    write_os_hint_event("expert_madvise_willneed", reason ? reason : "expert_prefetch",
                        info.name.c_str(), info.layer, expert, addr, nbytes, 0, -1, ENOSYS, 0, &meta);
#endif
    if (os_hint_opt_enabled("LLM_MEM_TRACE_OPT_POSIX_FADVISE")) {
        apply_posix_fadvise_hint("expert_posix_fadvise_willneed",
                                 reason ? reason : "expert_prefetch",
                                 info.name.c_str(), info.layer, expert, addr, nbytes, &meta);
    }
}

struct ExpertTaskLifecycleRecord {
    uint64_t task_id = 0;
    uint64_t issue_id = 0;
    uint64_t issue_task_count = 0;
    ExpertTaskState state = ExpertTaskState::New;
    uint64_t created_ts_ns = 0;
    uint64_t enqueued_ts_ns = 0;
    uint64_t dequeued_ts_ns = 0;
    uint64_t issued_ts_ns = 0;
    uint64_t returned_ts_ns = 0;
    uint64_t deadline_ts_ns = 0;
    uint64_t sequence = 0;
    uint64_t step = 0;
    int layer = -1;
    int expert = -1;
    int phase = LLM_MEM_TRACE_PHASE_UNKNOWN;
    ExpertTensorStage stage = ExpertTensorStage::Unknown;
    std::string tensor_name;
    uintptr_t addr = 0;
    size_t nbytes = 0;
    double score = 0.0;
};

struct ExpertHintTask {
    std::string action;
    std::string fadvise_action;
    std::string trigger;
    std::string tensor_name;
    std::string policy;
    int layer = -1;
    int expert = -1;
    uintptr_t addr = 0;
    size_t nbytes = 0;
    uint64_t cache_bytes = 0;
    uint64_t cache_capacity_bytes = 0;
    int phase = LLM_MEM_TRACE_PHASE_UNKNOWN;
    ExpertTensorStage stage = ExpertTensorStage::Unknown;
    uint64_t step = 0;
    uint64_t sequence = 0;
    double route_score = 0.0;
    double route_confidence = 0.0;
    uint64_t enqueue_ts_ns = 0;
    uint64_t deadline_ts_ns = 0;
    uint64_t predicted_service_ns = 0;
    uint64_t predicted_benefit_ns = 0;
    uint64_t predicted_cost_ns = 0;
    double value_ratio = 0.0;
    ExpertPressureLevel pressure_level = ExpertPressureLevel::Low;
    uint64_t memory_current_bytes = 0;
    uint64_t memory_limit_bytes = 0;
    uint64_t prefetch_budget_bytes = 0;
    uint64_t workingset_refault = 0;
    uint64_t refault_delta = 0;
    double psi_some_avg10 = 0.0;
    double psi_full_avg10 = 0.0;
    bool predicted = false;
    int prediction_source_layer = -1;
    int token_idx = -1;
    bool use_fadvise = false;
    ExpertTaskLifecycleRecord lifecycle;
    std::vector<ExpertTaskLifecycleRecord> coalesced_lifecycles;
    uint64_t coalesced_task_count = 1;
    uint64_t issue_id = 0;
};

enum class ExpertTaskTraceMode {
    Off,
    Summary,
    Detail,
};

ExpertTaskTraceMode expert_task_trace_mode() {
    static const ExpertTaskTraceMode mode = [] {
        const char * configured = std::getenv("LLM_MEM_TRACE_EXPERT_TASK_MODE");
        if (configured && configured[0]) {
            if (std::strcmp(configured, "off") == 0) {
                return ExpertTaskTraceMode::Off;
            }
            if (std::strcmp(configured, "summary") == 0) {
                return ExpertTaskTraceMode::Summary;
            }
            if (std::strcmp(configured, "detail") == 0) {
                return ExpertTaskTraceMode::Detail;
            }
        }
        const char * legacy = std::getenv("LLM_MEM_TRACE_EXPERT_TASK_EVENTS");
        if (legacy && legacy[0]) {
            return env_truthy(legacy) ? ExpertTaskTraceMode::Detail : ExpertTaskTraceMode::Summary;
        }
        const char * profile = std::getenv("TRACE_PROFILE");
        return profile && std::strcmp(profile, "benchmark") == 0 ?
                ExpertTaskTraceMode::Summary : ExpertTaskTraceMode::Detail;
    }();
    return mode;
}

const char * expert_task_trace_mode_name() {
    switch (expert_task_trace_mode()) {
        case ExpertTaskTraceMode::Off:     return "off";
        case ExpertTaskTraceMode::Summary: return "summary";
        case ExpertTaskTraceMode::Detail:  return "detail";
    }
    return "off";
}

struct ExpertTaskLifecycleStats {
    std::atomic<uint64_t> created{0};
    std::atomic<uint64_t> admitted{0};
    std::atomic<uint64_t> rejected{0};
    std::atomic<uint64_t> enqueued{0};
    std::atomic<uint64_t> dequeued{0};
    std::atomic<uint64_t> issued{0};
    std::atomic<uint64_t> cancelled{0};
    std::atomic<uint64_t> invalid_transitions{0};
    std::atomic<uint64_t> rejected_pressure{0};
    std::atomic<uint64_t> rejected_value{0};
    std::atomic<uint64_t> cancelled_pressure{0};
    std::atomic<uint64_t> cancelled_value{0};
    std::atomic<uint64_t> cancelled_expired{0};
    std::atomic<uint64_t> cancelled_queue_full{0};
    std::atomic<uint64_t> issue_groups{0};
    std::atomic<uint64_t> coalesced_issue_groups{0};
    std::atomic<uint64_t> same_stage_issue_groups{0};
    std::atomic<uint64_t> cross_stage_issue_groups{0};
    std::atomic<uint64_t> early_task_count{0};
    std::atomic<uint64_t> late_task_count{0};
    std::atomic<uint64_t> unknown_task_count{0};
    std::array<std::atomic<uint64_t>, 3> enqueued_by_stage{};
    std::array<std::atomic<uint64_t>, 3> issued_by_stage{};
    std::array<std::atomic<uint64_t>, 3> late_count_by_stage{};

    struct DurationAggregate {
        std::atomic<uint64_t> count{0};
        std::atomic<uint64_t> total_ns{0};
        std::atomic<uint64_t> min_ns{std::numeric_limits<uint64_t>::max()};
        std::atomic<uint64_t> max_ns{0};
    };
    std::array<DurationAggregate, 3> queue_wait_ns;
};

ExpertTaskLifecycleStats & expert_task_lifecycle_stats() {
    static ExpertTaskLifecycleStats stats;
    return stats;
}

bool expert_task_detail_events_enabled() {
    return expert_task_trace_mode() == ExpertTaskTraceMode::Detail;
}

void observe_atomic_duration(
        ExpertTaskLifecycleStats::DurationAggregate & aggregate,
        uint64_t duration_ns) {
    aggregate.count.fetch_add(1, std::memory_order_relaxed);
    aggregate.total_ns.fetch_add(duration_ns, std::memory_order_relaxed);
    uint64_t minimum = aggregate.min_ns.load(std::memory_order_relaxed);
    while (duration_ns < minimum &&
            !aggregate.min_ns.compare_exchange_weak(
                    minimum, duration_ns, std::memory_order_relaxed)) {
    }
    uint64_t maximum = aggregate.max_ns.load(std::memory_order_relaxed);
    while (duration_ns > maximum &&
            !aggregate.max_ns.compare_exchange_weak(
                    maximum, duration_ns, std::memory_order_relaxed)) {
    }
}

void record_expert_task_event_count(
        ExpertTaskEvent event,
        uint64_t count,
        const char * reason,
        ExpertTensorStage stage = ExpertTensorStage::Unknown) {
    ExpertTaskLifecycleStats & stats = expert_task_lifecycle_stats();
    switch (event) {
        case ExpertTaskEvent::Create:
            stats.created.fetch_add(count, std::memory_order_relaxed);
            switch (stage) {
                case ExpertTensorStage::Early:
                    stats.early_task_count.fetch_add(count, std::memory_order_relaxed);
                    break;
                case ExpertTensorStage::Late:
                    stats.late_task_count.fetch_add(count, std::memory_order_relaxed);
                    break;
                case ExpertTensorStage::Unknown:
                    stats.unknown_task_count.fetch_add(count, std::memory_order_relaxed);
                    break;
            }
            break;
        case ExpertTaskEvent::Admit:   stats.admitted.fetch_add(count, std::memory_order_relaxed); break;
        case ExpertTaskEvent::Reject:  stats.rejected.fetch_add(count, std::memory_order_relaxed); break;
        case ExpertTaskEvent::Enqueue:
            stats.enqueued.fetch_add(count, std::memory_order_relaxed);
            stats.enqueued_by_stage[expert_tensor_stage_index(stage)].fetch_add(
                    count, std::memory_order_relaxed);
            break;
        case ExpertTaskEvent::Dequeue: stats.dequeued.fetch_add(count, std::memory_order_relaxed); break;
        case ExpertTaskEvent::Issue:
            stats.issued.fetch_add(count, std::memory_order_relaxed);
            stats.issued_by_stage[expert_tensor_stage_index(stage)].fetch_add(
                    count, std::memory_order_relaxed);
            break;
        case ExpertTaskEvent::Cancel:  stats.cancelled.fetch_add(count, std::memory_order_relaxed); break;
    }
    if (!reason || count == 0) {
        return;
    }
    if (event == ExpertTaskEvent::Reject) {
        if (std::strcmp(reason, "pressure_budget") == 0) {
            stats.rejected_pressure.fetch_add(count, std::memory_order_relaxed);
        } else if (std::strcmp(reason, "benefit_below_cost") == 0) {
            stats.rejected_value.fetch_add(count, std::memory_order_relaxed);
        }
    } else if (event == ExpertTaskEvent::Cancel) {
        if (std::strcmp(reason, "pressure_changed") == 0) {
            stats.cancelled_pressure.fetch_add(count, std::memory_order_relaxed);
        } else if (std::strcmp(reason, "value_changed") == 0) {
            stats.cancelled_value.fetch_add(count, std::memory_order_relaxed);
        } else if (std::strcmp(reason, "deadline_missed") == 0) {
            stats.cancelled_expired.fetch_add(count, std::memory_order_relaxed);
        } else if (std::strcmp(reason, "queue_full") == 0) {
            stats.cancelled_queue_full.fetch_add(count, std::memory_order_relaxed);
        }
    }
}

void write_expert_task_event(
        const ExpertTaskLifecycleRecord & task,
        ExpertTaskEvent event,
        uint64_t event_ts_ns,
        const char * reason) {
    if (!expert_task_detail_events_enabled() ||
            !llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    char addr_buf[32];
    std::snprintf(addr_buf, sizeof(addr_buf), "0x%llx", (unsigned long long) task.addr);
    std::string line;
    line.reserve(320);
    line += "{\"event\":\"EXPERT_TASK\",\"ts_ns\":" + std::to_string(event_ts_ns);
    line += ",\"lifecycle_event\":\"" + std::string(expert_task_event_name(event)) + "\"";
    line += ",\"state\":\"" + std::string(expert_task_state_name(task.state)) + "\"";
    line += ",\"task_id\":" + std::to_string(task.task_id);
    if (task.issue_id != 0) {
        line += ",\"issue_id\":" + std::to_string(task.issue_id);
        line += ",\"issue_task_count\":" + std::to_string(task.issue_task_count);
    }
    line += ",\"step\":" + std::to_string(task.step);
    line += ",\"layer\":" + std::to_string(task.layer);
    line += ",\"expert\":" + std::to_string(task.expert);
    line += ",\"phase\":\"" + std::string(phase_name(task.phase)) + "\"";
    line += ",\"stage\":\"" + std::string(expert_tensor_stage_name(task.stage)) + "\"";
    line += ",\"tensor\":";
    json_escape_append(line, task.tensor_name.c_str());
    line += ",\"addr\":";
    json_escape_append(line, addr_buf);
    line += ",\"nbytes\":" + std::to_string(task.nbytes);
    line += ",\"score\":" + std::to_string(task.score);
    line += ",\"sequence\":" + std::to_string(task.sequence);
    line += ",\"deadline_ts_ns\":" + std::to_string(task.deadline_ts_ns);
    line += ",\"created_ts_ns\":" + std::to_string(task.created_ts_ns);
    line += ",\"enqueued_ts_ns\":" + std::to_string(task.enqueued_ts_ns);
    line += ",\"dequeued_ts_ns\":" + std::to_string(task.dequeued_ts_ns);
    line += ",\"issued_ts_ns\":" + std::to_string(task.issued_ts_ns);
    if (task.enqueued_ts_ns != 0 && task.dequeued_ts_ns >= task.enqueued_ts_ns) {
        line += ",\"queue_wait_ns\":" +
                std::to_string(task.dequeued_ts_ns - task.enqueued_ts_ns);
    } else {
        line += ",\"queue_wait_ns\":null";
    }
    if (task.returned_ts_ns != 0) {
        line += ",\"returned_ts_ns\":" + std::to_string(task.returned_ts_ns);
    }
    if (reason && reason[0]) {
        line += ",\"reason\":";
        json_escape_append(line, reason);
    }
    if (event == ExpertTaskEvent::Issue) {
        line += ",\"hint_status\":\"returned\"";
    }
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

bool transition_expert_task(
        ExpertTaskLifecycleRecord & task,
        ExpertTaskEvent event,
        const char * reason = nullptr,
        uint64_t issued_ts_ns = 0,
        uint64_t returned_ts_ns = 0) {
    if (expert_task_trace_mode() == ExpertTaskTraceMode::Off) {
        return true;
    }
    const bool needs_timestamp = expert_task_trace_mode() != ExpertTaskTraceMode::Off;
    const uint64_t now = returned_ts_ns != 0 ? returned_ts_ns :
            (needs_timestamp ? llm_mem_trace_time_ns() : 0);
    if (!expert_task_apply_event(task.state, event)) {
        expert_task_lifecycle_stats().invalid_transitions.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    switch (event) {
        case ExpertTaskEvent::Create:  task.created_ts_ns = now; break;
        case ExpertTaskEvent::Enqueue: task.enqueued_ts_ns = now; break;
        case ExpertTaskEvent::Dequeue:
            task.dequeued_ts_ns = now;
            if (task.enqueued_ts_ns != 0 && now >= task.enqueued_ts_ns) {
                observe_atomic_duration(
                        expert_task_lifecycle_stats().queue_wait_ns[
                                expert_tensor_stage_index(task.stage)],
                        now - task.enqueued_ts_ns);
            }
            break;
        case ExpertTaskEvent::Issue:
            task.issued_ts_ns = issued_ts_ns != 0 ? issued_ts_ns : now;
            task.returned_ts_ns = returned_ts_ns != 0 ? returned_ts_ns : now;
            if (task.deadline_ts_ns != 0 && task.issued_ts_ns >= task.deadline_ts_ns) {
                expert_task_lifecycle_stats().late_count_by_stage[
                        expert_tensor_stage_index(task.stage)].fetch_add(
                                1, std::memory_order_relaxed);
            }
            break;
        case ExpertTaskEvent::Admit:
        case ExpertTaskEvent::Reject:
        case ExpertTaskEvent::Cancel:
            break;
    }
    record_expert_task_event_count(event, 1, reason, task.stage);
    write_expert_task_event(task, event, now, reason);
    return true;
}

void append_atomic_stage_duration_map(
        std::string & line,
        const char * field,
        const std::array<ExpertTaskLifecycleStats::DurationAggregate, 3> & aggregates) {
    static const ExpertTensorStage stages[] = {
        ExpertTensorStage::Early,
        ExpertTensorStage::Late,
        ExpertTensorStage::Unknown,
    };
    line += ",\"" + std::string(field) + "\":{";
    for (size_t i = 0; i < 3; ++i) {
        if (i != 0) {
            line += ",";
        }
        const ExpertTaskLifecycleStats::DurationAggregate & aggregate = aggregates[i];
        const uint64_t count = aggregate.count.load(std::memory_order_relaxed);
        const uint64_t minimum = aggregate.min_ns.load(std::memory_order_relaxed);
        line += "\"" + std::string(expert_tensor_stage_name(stages[i])) + "\":{";
        line += "\"count\":" + std::to_string(count);
        line += ",\"total_ns\":" +
                std::to_string(aggregate.total_ns.load(std::memory_order_relaxed));
        line += ",\"min_ns\":" +
                std::to_string(count == 0 || minimum == std::numeric_limits<uint64_t>::max() ? 0 : minimum);
        line += ",\"max_ns\":" +
                std::to_string(aggregate.max_ns.load(std::memory_order_relaxed));
        line += "}";
    }
    line += "}";
}

void append_atomic_stage_count_map(
        std::string & line,
        const char * field,
        const std::array<std::atomic<uint64_t>, 3> & counts) {
    static const ExpertTensorStage stages[] = {
        ExpertTensorStage::Early,
        ExpertTensorStage::Late,
        ExpertTensorStage::Unknown,
    };
    line += ",\"" + std::string(field) + "\":{";
    for (size_t i = 0; i < 3; ++i) {
        if (i != 0) {
            line += ",";
        }
        line += "\"" + std::string(expert_tensor_stage_name(stages[i])) + "\":" +
                std::to_string(counts[i].load(std::memory_order_relaxed));
    }
    line += "}";
}

void write_expert_task_summary() {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    const ExpertTaskLifecycleStats & stats = expert_task_lifecycle_stats();
    const uint64_t created = stats.created.load(std::memory_order_relaxed);
    const uint64_t rejected = stats.rejected.load(std::memory_order_relaxed);
    const uint64_t issued = stats.issued.load(std::memory_order_relaxed);
    const uint64_t cancelled = stats.cancelled.load(std::memory_order_relaxed);
    const uint64_t terminal = rejected + issued + cancelled;
    std::string line;
    line.reserve(320);
    line += "{\"event\":\"EXPERT_TASK_SUMMARY\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
    line += ",\"trace_mode\":\"" + std::string(expert_task_trace_mode_name()) + "\"";
    line += ",\"detail_events_enabled\":" + std::string(expert_task_detail_events_enabled() ? "true" : "false");
    line += ",\"created\":" + std::to_string(created);
    line += ",\"admitted\":" + std::to_string(stats.admitted.load(std::memory_order_relaxed));
    line += ",\"rejected\":" + std::to_string(rejected);
    line += ",\"enqueued\":" + std::to_string(stats.enqueued.load(std::memory_order_relaxed));
    line += ",\"dequeued\":" + std::to_string(stats.dequeued.load(std::memory_order_relaxed));
    line += ",\"issued\":" + std::to_string(issued);
    line += ",\"cancelled\":" + std::to_string(cancelled);
    line += ",\"terminal\":" + std::to_string(terminal);
    line += ",\"in_flight\":" + std::to_string(created >= terminal ? created - terminal : 0);
    line += ",\"invalid_transitions\":" + std::to_string(stats.invalid_transitions.load(std::memory_order_relaxed));
    line += ",\"rejected_pressure\":" + std::to_string(stats.rejected_pressure.load(std::memory_order_relaxed));
    line += ",\"rejected_value\":" + std::to_string(stats.rejected_value.load(std::memory_order_relaxed));
    line += ",\"cancelled_pressure\":" + std::to_string(stats.cancelled_pressure.load(std::memory_order_relaxed));
    line += ",\"cancelled_value\":" + std::to_string(stats.cancelled_value.load(std::memory_order_relaxed));
    line += ",\"cancelled_expired\":" + std::to_string(stats.cancelled_expired.load(std::memory_order_relaxed));
    line += ",\"cancelled_queue_full\":" + std::to_string(stats.cancelled_queue_full.load(std::memory_order_relaxed));
    line += ",\"issue_groups\":" + std::to_string(stats.issue_groups.load(std::memory_order_relaxed));
    line += ",\"coalesced_issue_groups\":" + std::to_string(stats.coalesced_issue_groups.load(std::memory_order_relaxed));
    line += ",\"same_stage_issue_groups\":" +
            std::to_string(stats.same_stage_issue_groups.load(std::memory_order_relaxed));
    line += ",\"cross_stage_issue_groups\":" +
            std::to_string(stats.cross_stage_issue_groups.load(std::memory_order_relaxed));
    line += ",\"early_task_count\":" +
            std::to_string(stats.early_task_count.load(std::memory_order_relaxed));
    line += ",\"late_task_count\":" +
            std::to_string(stats.late_task_count.load(std::memory_order_relaxed));
    line += ",\"unknown_task_count\":" +
            std::to_string(stats.unknown_task_count.load(std::memory_order_relaxed));
    append_atomic_stage_count_map(line, "enqueued_by_stage", stats.enqueued_by_stage);
    append_atomic_stage_count_map(line, "issued_by_stage", stats.issued_by_stage);
    append_atomic_stage_count_map(line, "late_count_by_stage", stats.late_count_by_stage);
    append_atomic_stage_duration_map(line, "queue_wait_ns_by_stage", stats.queue_wait_ns);
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void ensure_expert_task_summary_registered() {
    // Construct the counters before registering the writer so the atexit
    // callback observes a live stats object.
    (void) expert_task_lifecycle_stats();
    static const bool registered = [] {
        std::atexit(write_expert_task_summary);
        return true;
    }();
    (void) registered;
}

uint64_t next_expert_task_id() {
    static std::atomic<uint64_t> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
}

uint64_t next_expert_issue_id() {
    static std::atomic<uint64_t> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
}

ExpertFirstUseMatcher & expert_first_use_matcher() {
    static ExpertFirstUseMatcher matcher;
    return matcher;
}

void append_stage_duration_map(
        std::string & line,
        const char * field,
        const std::array<ExpertDurationAggregate, 3> & aggregates) {
    static const ExpertTensorStage stages[] = {
        ExpertTensorStage::Early,
        ExpertTensorStage::Late,
        ExpertTensorStage::Unknown,
    };
    line += ",\"" + std::string(field) + "\":{";
    for (size_t i = 0; i < 3; ++i) {
        if (i != 0) {
            line += ",";
        }
        const ExpertDurationAggregate & aggregate = aggregates[i];
        line += "\"" + std::string(expert_tensor_stage_name(stages[i])) + "\":{";
        line += "\"count\":" + std::to_string(aggregate.count);
        line += ",\"total_ns\":" + std::to_string(aggregate.total_ns);
        line += ",\"min_ns\":" + std::to_string(aggregate.min_ns);
        line += ",\"max_ns\":" + std::to_string(aggregate.max_ns);
        line += "}";
    }
    line += "}";
}

void write_expert_first_use_summary() {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY) ||
            expert_task_trace_mode() == ExpertTaskTraceMode::Off) {
        return;
    }
    const ExpertFirstUseCounters counters = expert_first_use_matcher().counters();
    std::string line;
    line.reserve(320);
    line += "{\"event\":\"EXPERT_FIRST_USE_SUMMARY\",\"ts_ns\":" +
            std::to_string(llm_mem_trace_time_ns());
    line += ",\"semantics\":\"logical_first_use\",\"physical_load_observed\":false";
    line += ",\"eligible_tasks\":" + std::to_string(counters.eligible_tasks);
    line += ",\"logical_first_uses\":" + std::to_string(counters.logical_first_uses);
    line += ",\"matched_tasks\":" + std::to_string(counters.matched_tasks);
    line += ",\"unmatched_tasks\":" + std::to_string(counters.unmatched_tasks);
    line += ",\"unmatched_first_uses\":" + std::to_string(counters.unmatched_first_uses);
    line += ",\"ambiguous_matches\":" + std::to_string(counters.ambiguous_matches);
    line += ",\"duplicate_first_use_ignored\":" +
            std::to_string(counters.duplicate_first_use_ignored);
    line += ",\"matcher_peak_live_tasks\":" +
            std::to_string(counters.matcher_peak_live_tasks);
    line += ",\"matcher_expired_tasks\":" +
            std::to_string(counters.matcher_expired_tasks);
    line += ",\"late_issued_tasks\":" + std::to_string(counters.late_issued_tasks);
    line += ",\"pending_issued_tasks\":" + std::to_string(counters.pending_issued_tasks);
    line += ",\"ignored_old_uses\":" + std::to_string(counters.ignored_old_uses);
    append_stage_duration_map(
            line, "create_to_first_use_ns_by_stage", counters.create_to_first_use_ns);
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void ensure_expert_first_use_summary_registered() {
    (void) expert_first_use_matcher();
    static const bool registered = [] {
        std::atexit(write_expert_first_use_summary);
        return true;
    }();
    (void) registered;
}

void register_expert_task_for_first_use(const ExpertTaskLifecycleRecord & task) {
    if (expert_task_trace_mode() == ExpertTaskTraceMode::Off) {
        return;
    }
    ExpertIssuedTask issued;
    issued.task_id = task.task_id;
    issued.issue_id = task.issue_id;
    issued.step = task.step;
    issued.layer = task.layer;
    issued.expert = task.expert;
    issued.phase = task.phase;
    issued.stage = task.stage;
    issued.tensor = task.tensor_name;
    issued.addr = task.addr;
    issued.nbytes = task.nbytes;
    issued.created_ts_ns = task.created_ts_ns;
    issued.enqueued_ts_ns = task.enqueued_ts_ns;
    issued.dequeued_ts_ns = task.dequeued_ts_ns;
    issued.issued_ts_ns = task.issued_ts_ns;
    expert_first_use_matcher().register_issue(std::move(issued));
}

void write_expert_first_use_event(const ExpertFirstUseMatch & match) {
    if (!match.considered || !expert_task_detail_events_enabled() ||
            !llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    char addr_buf[32];
    std::snprintf(addr_buf, sizeof(addr_buf), "0x%llx", (unsigned long long) match.use.addr);
    const auto write_one = [&](const ExpertIssuedTask * task, size_t match_index) {
        std::string line;
        line.reserve(480);
        line += "{\"event\":\"EXPERT_FIRST_USE\",\"ts_ns\":" +
                std::to_string(match.use.first_use_ts_ns);
        line += ",\"semantics\":\"logical_first_use\",\"physical_load_observed\":false";
        line += ",\"matched\":" + std::string(task ? "true" : "false");
        line += ",\"match_count\":" + std::to_string(match.tasks.size());
        line += ",\"match_index\":" + std::to_string(match_index);
        line += ",\"ambiguous_match\":" + std::string(match.ambiguous() ? "true" : "false");
        line += ",\"step\":" + std::to_string(match.use.step);
        line += ",\"layer\":" + std::to_string(match.use.layer);
        line += ",\"expert\":" + std::to_string(match.use.expert);
        line += ",\"phase\":\"" + std::string(phase_name(match.use.phase)) + "\"";
        line += ",\"stage\":\"" +
                std::string(expert_tensor_stage_name(match.use.stage)) + "\"";
        line += ",\"tensor\":";
        json_escape_append(line, match.use.tensor.c_str());
        line += ",\"addr\":";
        json_escape_append(line, addr_buf);
        line += ",\"nbytes\":" + std::to_string(match.use.nbytes);
        line += ",\"first_use_ts_ns\":" + std::to_string(match.use.first_use_ts_ns);
        if (task) {
            line += ",\"task_id\":" + std::to_string(task->task_id);
            line += ",\"issue_id\":" + std::to_string(task->issue_id);
            line += ",\"issued_ts_ns\":" + std::to_string(task->issued_ts_ns);
            line += ",\"create_to_first_use_ns\":" +
                    std::to_string(match.use.first_use_ts_ns - task->created_ts_ns);
            line += ",\"issue_to_first_use_ns\":" +
                    std::to_string(match.use.first_use_ts_ns - task->issued_ts_ns);
            const uint64_t queue_wait_ns = task->enqueued_ts_ns != 0 &&
                    task->dequeued_ts_ns >= task->enqueued_ts_ns ?
                    task->dequeued_ts_ns - task->enqueued_ts_ns : 0;
            line += ",\"queue_wait_ns\":" + std::to_string(queue_wait_ns);
        } else {
            line += ",\"create_to_first_use_ns\":null";
            line += ",\"issue_to_first_use_ns\":null";
            line += ",\"queue_wait_ns\":null";
            line += ",\"unmatched_reason\":";
            json_escape_append(line, match.unmatched_reason.c_str());
        }
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    };
    if (match.tasks.empty()) {
        write_one(nullptr, 0);
        return;
    }
    for (size_t i = 0; i < match.tasks.size(); ++i) {
        write_one(&match.tasks[i], i);
    }
}

void apply_pressure_snapshot(ExpertHintTask & task, const ExpertPressureSnapshot & pressure) {
    task.pressure_level = pressure.level;
    task.memory_current_bytes = pressure.memory_current_bytes;
    task.memory_limit_bytes = pressure.memory_limit_bytes;
    task.prefetch_budget_bytes = pressure.prefetch_budget_bytes;
    task.workingset_refault = pressure.workingset_refault;
    task.refault_delta = pressure.refault_delta;
    task.psi_some_avg10 = pressure.psi_some_avg10;
    task.psi_full_avg10 = pressure.psi_full_avg10;
}

double expert_pressure_cost_factor(ExpertPressureLevel level) {
    switch (level) {
        case ExpertPressureLevel::Low:      return 0.05;
        case ExpertPressureLevel::Moderate: return 0.20;
        case ExpertPressureLevel::High:     return 0.50;
        case ExpertPressureLevel::Critical: return 1.00;
    }
    return 0.05;
}

void refresh_expert_task_estimate(ExpertHintTask & task) {
    const uint64_t transfer_ns = expert_timing_model().predicted_transfer_ns(task.nbytes);
    const uint64_t syscall_ns = expert_timing_model().predicted_syscall_ns();
    task.predicted_service_ns = transfer_ns + syscall_ns;
    task.predicted_benefit_ns = (uint64_t) ((double) transfer_ns * task.route_confidence);
    task.predicted_cost_ns = syscall_ns +
            (uint64_t) ((double) transfer_ns * expert_pressure_cost_factor(task.pressure_level));
    task.value_ratio = task.predicted_cost_ns > 0 ?
            (double) task.predicted_benefit_ns / (double) task.predicted_cost_ns : 0.0;
}

bool expert_task_exceeds_pressure_budget(const ExpertHintTask & task, uint64_t queued_bytes) {
    if (!expert_feedback_enabled()) {
        return false;
    }
    const double critical_min_confidence = env_double_or_default(
            "LLM_MEM_TRACE_OPT_EXPERT_PRESSURE_CRITICAL_MIN_CONFIDENCE", 0.25);
    return task.prefetch_budget_bytes == 0 || task.nbytes > task.prefetch_budget_bytes ||
            queued_bytes > task.prefetch_budget_bytes -
                    std::min<uint64_t>(task.nbytes, task.prefetch_budget_bytes) ||
            (task.pressure_level == ExpertPressureLevel::Critical &&
             task.route_confidence < critical_min_confidence);
}

bool expert_task_below_value_threshold(const ExpertHintTask & task) {
    if (!expert_value_gate_enabled()) {
        return false;
    }
    const double min_confidence = env_double_or_default(
            "LLM_MEM_TRACE_OPT_EXPERT_VALUE_MIN_CONFIDENCE", 0.01);
    const double min_ratio = env_double_or_default(
            "LLM_MEM_TRACE_OPT_EXPERT_VALUE_MIN_RATIO", 1.0);
    return task.route_confidence < min_confidence || task.value_ratio < min_ratio;
}

void fill_expert_task_meta(const ExpertHintTask & task, OsHintMeta & meta, const char * decision) {
    const uint64_t now = llm_mem_trace_time_ns();
    meta.policy = task.policy.c_str();
    meta.decision = decision;
    meta.cache_bytes = task.cache_bytes;
    meta.cache_capacity_bytes = task.cache_capacity_bytes;
    meta.cache_hit = false;
    meta.has_cache_hit = true;
    meta.has_trace_context = true;
    meta.phase = task.phase;
    meta.step = task.step;
    meta.has_control = expert_feedback_enabled() || expert_slack_enabled() || expert_value_gate_enabled() || task.predicted;
    meta.route_score = task.route_score;
    meta.route_confidence = task.route_confidence;
    meta.enqueue_ts_ns = task.enqueue_ts_ns;
    meta.deadline_ts_ns = task.deadline_ts_ns;
    meta.slack_ns = task.deadline_ts_ns > now ? task.deadline_ts_ns - now : 0;
    meta.predicted_service_ns = task.predicted_service_ns;
    meta.predicted_benefit_ns = task.predicted_benefit_ns;
    meta.predicted_cost_ns = task.predicted_cost_ns;
    meta.value_ratio = task.value_ratio;
    meta.pressure_level = expert_pressure_level_name(task.pressure_level);
    meta.memory_current_bytes = task.memory_current_bytes;
    meta.memory_limit_bytes = task.memory_limit_bytes;
    meta.prefetch_budget_bytes = task.prefetch_budget_bytes;
    meta.workingset_refault = task.workingset_refault;
    meta.refault_delta = task.refault_delta;
    meta.psi_some_avg10 = task.psi_some_avg10;
    meta.psi_full_avg10 = task.psi_full_avg10;
    meta.predicted = task.predicted;
    meta.prediction_source_layer = task.prediction_source_layer;
    meta.token_idx = task.token_idx;
    meta.issue_id = task.issue_id;
    meta.issue_task_count = task.coalesced_task_count;
}

void write_expert_task_skip(const ExpertHintTask & task, const char * action, const char * trigger) {
    // The task summary already aggregates these new reject/cancel reasons.
    // Preserve all pre-existing OS_HINT records outside this task path.
    if (trace_profile_is_benchmark()) {
        return;
    }
    OsHintMeta meta;
    fill_expert_task_meta(task, meta, "skip");
    write_os_hint_event(action, trigger ? trigger : task.trigger.c_str(), task.tensor_name.c_str(),
                        task.layer, task.expert, task.addr, task.nbytes, 0, 0, 0, 0, &meta);
}

void record_expert_issue_group_stage(const ExpertHintTask & task) {
    bool stages[3] = {false, false, false};
    if (task.coalesced_lifecycles.empty()) {
        stages[expert_tensor_stage_index(task.lifecycle.stage)] = true;
    } else {
        for (const ExpertTaskLifecycleRecord & lifecycle : task.coalesced_lifecycles) {
            stages[expert_tensor_stage_index(lifecycle.stage)] = true;
        }
    }
    const size_t distinct = (size_t) stages[0] + (size_t) stages[1] + (size_t) stages[2];
    ExpertTaskLifecycleStats & stats = expert_task_lifecycle_stats();
    if (distinct > 1) {
        stats.cross_stage_issue_groups.fetch_add(1, std::memory_order_relaxed);
    } else {
        stats.same_stage_issue_groups.fetch_add(1, std::memory_order_relaxed);
    }
}

uint64_t issue_expert_hint_task(ExpertHintTask & task) {
    if (expert_task_detail_events_enabled()) {
        task.issue_id = next_expert_issue_id();
    }
    if (expert_task_trace_mode() != ExpertTaskTraceMode::Off) {
        ExpertTaskLifecycleStats & stats = expert_task_lifecycle_stats();
        stats.issue_groups.fetch_add(1, std::memory_order_relaxed);
        if (task.coalesced_task_count > 1) {
            stats.coalesced_issue_groups.fetch_add(1, std::memory_order_relaxed);
        }
        record_expert_issue_group_stage(task);
    }
    const uint64_t begin = llm_mem_trace_time_ns();
    if (expert_task_trace_mode() != ExpertTaskTraceMode::Off) {
        const auto prepare = [&](ExpertTaskLifecycleRecord & lifecycle) {
            lifecycle.issue_id = task.issue_id;
            lifecycle.issue_task_count = task.coalesced_task_count;
            lifecycle.issued_ts_ns = begin;
            register_expert_task_for_first_use(lifecycle);
        };
        if (task.coalesced_lifecycles.empty()) {
            prepare(task.lifecycle);
        } else {
            for (ExpertTaskLifecycleRecord & lifecycle : task.coalesced_lifecycles) {
                prepare(lifecycle);
            }
        }
    }
    OsHintMeta meta;
    fill_expert_task_meta(task, meta, "prefetch");
#ifdef __linux__
    apply_madvise_hint(task.action.c_str(), MADV_WILLNEED, task.trigger.c_str(),
                       task.tensor_name.c_str(), task.layer, task.expert, task.addr, task.nbytes, &meta);
#else
    write_os_hint_event(task.action.c_str(), task.trigger.c_str(), task.tensor_name.c_str(),
                        task.layer, task.expert, task.addr, task.nbytes, 0, -1, ENOSYS, 0, &meta);
#endif
    if (task.use_fadvise) {
        apply_posix_fadvise_hint(task.fadvise_action.c_str(), task.trigger.c_str(),
                                 task.tensor_name.c_str(), task.layer, task.expert, task.addr, task.nbytes, &meta);
    }
    const uint64_t end = llm_mem_trace_time_ns();
    const uint64_t duration = end >= begin ? end - begin : 0;
    expert_timing_model().observe_syscall(duration);
    if (task.coalesced_lifecycles.empty()) {
        transition_expert_task(task.lifecycle, ExpertTaskEvent::Issue, nullptr, begin, end);
        if (task.coalesced_task_count > 1) {
            record_expert_task_event_count(
                    ExpertTaskEvent::Issue, task.coalesced_task_count - 1, nullptr);
        }
    } else {
        for (ExpertTaskLifecycleRecord & lifecycle : task.coalesced_lifecycles) {
            transition_expert_task(lifecycle, ExpertTaskEvent::Issue, nullptr, begin, end);
        }
    }
    return duration;
}

uintptr_t expert_task_range_end(const ExpertHintTask & task) {
    const uintptr_t max_addr = std::numeric_limits<uintptr_t>::max();
    return task.nbytes > max_addr - task.addr ? max_addr : task.addr + task.nbytes;
}

std::vector<ExpertHintTask> coalesce_expert_hint_batch(std::vector<ExpertHintTask> batch) {
    if (!expert_prefetch_async_batch_coalesce_enabled() || batch.size() < 2) {
        return batch;
    }
    std::sort(batch.begin(), batch.end(), [](const ExpertHintTask & a, const ExpertHintTask & b) {
        if (a.tensor_name != b.tensor_name) {
            return a.tensor_name < b.tensor_name;
        }
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.addr < b.addr;
    });

    const uint64_t max_gap = expert_prefetch_coalesce_max_gap_bytes();
    std::vector<ExpertHintTask> merged;
    merged.reserve(batch.size());
    for (ExpertHintTask & task : batch) {
        if (merged.empty()) {
            merged.emplace_back(std::move(task));
            continue;
        }
        ExpertHintTask & current = merged.back();
        const uintptr_t current_end = expert_task_range_end(current);
        const uintptr_t gap_limit = max_gap > std::numeric_limits<uintptr_t>::max() - current_end ?
                std::numeric_limits<uintptr_t>::max() : current_end + (uintptr_t) max_gap;
        const bool compatible = current.tensor_name == task.tensor_name &&
                current.layer == task.layer && current.action == task.action &&
                current.fadvise_action == task.fadvise_action && current.trigger == task.trigger &&
                current.policy == task.policy && current.phase == task.phase &&
                current.step == task.step && current.use_fadvise == task.use_fadvise &&
                task.addr <= gap_limit;
        if (!compatible) {
            merged.emplace_back(std::move(task));
            continue;
        }

        current.coalesced_task_count += task.coalesced_task_count;
        if (expert_task_trace_mode() != ExpertTaskTraceMode::Off) {
            if (current.coalesced_lifecycles.empty()) {
                current.coalesced_lifecycles.push_back(current.lifecycle);
            }
            if (task.coalesced_lifecycles.empty()) {
                current.coalesced_lifecycles.push_back(std::move(task.lifecycle));
            } else {
                for (ExpertTaskLifecycleRecord & lifecycle : task.coalesced_lifecycles) {
                    current.coalesced_lifecycles.push_back(std::move(lifecycle));
                }
            }
        }
        const uintptr_t end = std::max(current_end, expert_task_range_end(task));
        current.nbytes = end > current.addr ? (size_t) (end - current.addr) : current.nbytes;
        if (current.expert != task.expert) {
            current.expert = -1;
        }
        current.route_score = std::max(current.route_score, task.route_score);
        current.route_confidence = std::max(current.route_confidence, task.route_confidence);
        current.enqueue_ts_ns = std::min(current.enqueue_ts_ns, task.enqueue_ts_ns);
        if (current.deadline_ts_ns == 0 || (task.deadline_ts_ns != 0 && task.deadline_ts_ns < current.deadline_ts_ns)) {
            current.deadline_ts_ns = task.deadline_ts_ns;
        }
        current.predicted_benefit_ns += task.predicted_benefit_ns;
        current.predicted_cost_ns = std::max(current.predicted_cost_ns, task.predicted_cost_ns);
        current.value_ratio = current.predicted_cost_ns > 0 ?
                (double) current.predicted_benefit_ns / (double) current.predicted_cost_ns : 0.0;
        current.predicted = current.predicted || task.predicted;
        if (current.action.find("_batch") == std::string::npos) {
            current.action += "_batch";
            current.fadvise_action += "_batch";
        }
    }
    return merged;
}

struct ExpertHintQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<ExpertHintTask> tasks;
    std::vector<ExpertHintTask> priority_heap;
    std::vector<ExpertHintTask> legacy_priority_heap;
    std::vector<std::thread> workers;
    bool started = false;
    bool stopping = false;
    size_t capacity = 0;
    size_t worker_count = 0;
    bool priority_enabled = false;
    bool priority_heap_enabled = false;
    ExpertAsyncPriorityMode priority_mode = ExpertAsyncPriorityMode::Score;
    uint64_t next_sequence = 0;
    uint64_t enqueued_tasks = 0;
    uint64_t issued_tasks = 0;
    uint64_t issued_candidates = 0;
    uint64_t priority_pops = 0;
    uint64_t priority_heap_pops = 0;
    uint64_t fallback_tasks = 0;
    uint64_t queue_full_fallbacks = 0;
    uint64_t start_fail_fallbacks = 0;
    uint64_t max_queue_depth = 0;
    uint64_t queued_bytes = 0;
    uint64_t max_queued_bytes = 0;
    uint64_t cancelled_expired = 0;
    uint64_t cancelled_pressure = 0;
    uint64_t cancelled_value = 0;
    uint64_t cancelled_queue_full = 0;
    uint64_t worker_batches = 0;
    uint64_t batched_candidates = 0;
    uint64_t coalesced_syscalls_saved = 0;

    ~ExpertHintQueue() {
        shutdown();
    }

    bool enqueue(ExpertHintTask && task) {
        if (!ensure_started()) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mu);
            if (queue_depth_unlocked() >= capacity) {
                queue_full_fallbacks++;
                return false;
            }
            const uint64_t task_bytes = (uint64_t) task.nbytes;
            task.sequence = next_sequence++;
            task.lifecycle.sequence = task.sequence;
            transition_expert_task(task.lifecycle, ExpertTaskEvent::Enqueue);
            if (priority_enabled && priority_heap_enabled) {
                std::vector<ExpertHintTask> & heap =
                        priority_mode == ExpertAsyncPriorityMode::StageDeadlineScore &&
                                expert_hint_priority_uses_legacy_partition(task.stage) ?
                        legacy_priority_heap : priority_heap;
                heap.emplace_back(std::move(task));
                auto cmp = [this](const ExpertHintTask & a, const ExpertHintTask & b) {
                    return is_higher_priority(b, a);
                };
                std::push_heap(heap.begin(), heap.end(), cmp);
            } else {
                tasks.emplace_back(std::move(task));
            }
            enqueued_tasks++;
            queued_bytes += task_bytes;
            max_queued_bytes = std::max(max_queued_bytes, queued_bytes);
            max_queue_depth = std::max<uint64_t>(max_queue_depth, (uint64_t) queue_depth_unlocked());
        }
        cv.notify_one();
        return true;
    }

    bool ensure_started() {
        std::lock_guard<std::mutex> lock(mu);
        if (started) {
            return true;
        }
        capacity = std::max<size_t>(1, expert_prefetch_async_queue_capacity());
        priority_enabled = expert_prefetch_async_priority_enabled();
        priority_heap_enabled = priority_enabled && expert_prefetch_async_priority_heap_enabled();
        priority_mode = expert_prefetch_async_priority_mode();
        stopping = false;
        const size_t n_workers = std::min<size_t>(expert_prefetch_async_workers(), 16);
        worker_count = n_workers;
        try {
            workers.reserve(n_workers);
            for (size_t i = 0; i < n_workers; ++i) {
                workers.emplace_back([this] { run(); });
            }
        } catch (...) {
            start_fail_fallbacks++;
            stopping = true;
            cv.notify_all();
            for (std::thread & worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            workers.clear();
            stopping = false;
            worker_count = 0;
            priority_enabled = false;
            priority_heap_enabled = false;
            priority_mode = ExpertAsyncPriorityMode::Score;
            started = false;
            return false;
        }
        started = true;
        return true;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mu);
            if (!started) {
                return;
            }
            stopping = true;
        }
        cv.notify_all();
        for (std::thread & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers.clear();
        write_summary();
        {
            std::lock_guard<std::mutex> lock(mu);
            started = false;
            stopping = false;
            tasks.clear();
            priority_heap.clear();
            legacy_priority_heap.clear();
            queued_bytes = 0;
            worker_count = 0;
            priority_enabled = false;
            priority_heap_enabled = false;
            priority_mode = ExpertAsyncPriorityMode::Score;
        }
    }

    void run() {
        for (;;) {
            std::vector<ExpertHintTask> batch;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] { return stopping || !queue_empty_unlocked(); });
                if (queue_empty_unlocked()) {
                    if (stopping) {
                        break;
                    }
                    continue;
                }
                const size_t batch_limit = expert_prefetch_async_batch_size();
                const uint64_t wait_us = expert_prefetch_async_batch_wait_us();
                if (!stopping && wait_us > 0 && batch_limit > 1 && queue_depth_unlocked() < batch_limit) {
                    cv.wait_for(lock, std::chrono::microseconds(wait_us), [&] {
                        return stopping || queue_depth_unlocked() >= batch_limit;
                    });
                }
                const size_t count = std::min(batch_limit, queue_depth_unlocked());
                batch.reserve(count);
                for (size_t i = 0; i < count; ++i) {
                    batch.emplace_back(pop_one_unlocked());
                }
                worker_batches++;
                batched_candidates += count;
            }

            for (ExpertHintTask & task : batch) {
                transition_expert_task(task.lifecycle, ExpertTaskEvent::Dequeue);
            }

            std::vector<ExpertHintTask> ready;
            ready.reserve(batch.size());
            for (ExpertHintTask & task : batch) {
                const uint64_t now = llm_mem_trace_time_ns();
                if (expert_feedback_enabled()) {
                    apply_pressure_snapshot(task, expert_pressure_controller().snapshot());
                }
                refresh_expert_task_estimate(task);
                if (expert_task_exceeds_pressure_budget(task, 0)) {
                    transition_expert_task(
                            task.lifecycle, ExpertTaskEvent::Cancel, "pressure_changed");
                    write_expert_task_skip(task, "expert_prefetch_cancel_pressure", "pressure_changed");
                    std::lock_guard<std::mutex> lock(mu);
                    cancelled_pressure++;
                    continue;
                }
                if (expert_task_below_value_threshold(task)) {
                    transition_expert_task(
                            task.lifecycle, ExpertTaskEvent::Cancel, "value_changed");
                    write_expert_task_skip(task, "expert_prefetch_cancel_value", "value_changed");
                    std::lock_guard<std::mutex> lock(mu);
                    cancelled_value++;
                    continue;
                }
                if (expert_slack_enabled() && task.deadline_ts_ns != 0 &&
                        now + task.predicted_service_ns >= task.deadline_ts_ns) {
                    transition_expert_task(
                            task.lifecycle, ExpertTaskEvent::Cancel, "deadline_missed");
                    write_expert_task_skip(task, "expert_prefetch_cancel_expired", "deadline_missed");
                    std::lock_guard<std::mutex> lock(mu);
                    cancelled_expired++;
                    continue;
                }
                ready.emplace_back(std::move(task));
            }

            const size_t ready_candidates = ready.size();
            std::vector<ExpertHintTask> issued = coalesce_expert_hint_batch(std::move(ready));
            for (ExpertHintTask & task : issued) {
                task.predicted_service_ns = expert_timing_model().predicted_transfer_ns(task.nbytes) +
                                            expert_timing_model().predicted_syscall_ns();
                issue_expert_hint_task(task);
            }
            {
                std::lock_guard<std::mutex> lock(mu);
                issued_candidates += ready_candidates;
                issued_tasks += issued.size();
                if (ready_candidates > issued.size()) {
                    coalesced_syscalls_saved += ready_candidates - issued.size();
                }
            }
        }
    }

    ExpertHintTask pop_one_unlocked() {
        ExpertHintTask task;
        if (priority_enabled && priority_heap_enabled) {
            auto cmp = [this](const ExpertHintTask & a, const ExpertHintTask & b) {
                return is_higher_priority(b, a);
            };
            std::vector<ExpertHintTask> * heap = &priority_heap;
            if (priority_mode == ExpertAsyncPriorityMode::StageDeadlineScore) {
                if (priority_heap.empty()) {
                    heap = &legacy_priority_heap;
                } else if (!legacy_priority_heap.empty() &&
                        is_higher_priority(legacy_priority_heap.front(), priority_heap.front())) {
                    heap = &legacy_priority_heap;
                }
            }
            std::pop_heap(heap->begin(), heap->end(), cmp);
            task = std::move(heap->back());
            heap->pop_back();
            priority_pops++;
            priority_heap_pops++;
        } else if (priority_enabled) {
            auto best_known = tasks.end();
            auto best_legacy = tasks.end();
            for (auto it = tasks.begin(); it != tasks.end(); ++it) {
                auto & best = priority_mode == ExpertAsyncPriorityMode::StageDeadlineScore &&
                                expert_hint_priority_uses_legacy_partition(it->stage) ?
                        best_legacy : best_known;
                if (best == tasks.end() || is_higher_priority(*it, *best)) {
                    best = it;
                }
            }
            auto best = best_known;
            if (best == tasks.end() || (best_legacy != tasks.end() &&
                    is_higher_priority(*best_legacy, *best))) {
                best = best_legacy;
            }
            task = std::move(*best);
            tasks.erase(best);
            priority_pops++;
        } else {
            task = std::move(tasks.front());
            tasks.pop_front();
        }
        queued_bytes -= std::min<uint64_t>(queued_bytes, (uint64_t) task.nbytes);
        return task;
    }

    void record_fallback() {
        std::lock_guard<std::mutex> lock(mu);
        fallback_tasks++;
    }

    void record_cancelled_pressure() {
        std::lock_guard<std::mutex> lock(mu);
        cancelled_pressure++;
    }

    void record_cancelled_value() {
        std::lock_guard<std::mutex> lock(mu);
        cancelled_value++;
    }

    void record_cancelled_queue_full() {
        std::lock_guard<std::mutex> lock(mu);
        cancelled_queue_full++;
    }

    uint64_t queued_bytes_snapshot() {
        std::lock_guard<std::mutex> lock(mu);
        return queued_bytes;
    }

    void write_summary() {
        if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
            return;
        }
        uint64_t enqueued = 0;
        uint64_t issued = 0;
        uint64_t issued_input = 0;
        uint64_t priority = 0;
        uint64_t heap_pops = 0;
        uint64_t fallback = 0;
        uint64_t queue_full = 0;
        uint64_t start_fail = 0;
        uint64_t high_water = 0;
        uint64_t queued_bytes_high_water = 0;
        uint64_t expired = 0;
        uint64_t pressure = 0;
        uint64_t value = 0;
        uint64_t queue_cancel = 0;
        uint64_t batches = 0;
        uint64_t batch_candidates = 0;
        uint64_t coalesced_saved = 0;
        size_t cap = 0;
        size_t workers_started = 0;
        bool priority_on = false;
        bool heap_on = false;
        ExpertAsyncPriorityMode mode = ExpertAsyncPriorityMode::Score;
        {
            std::lock_guard<std::mutex> lock(mu);
            enqueued = enqueued_tasks;
            issued = issued_tasks;
            issued_input = issued_candidates;
            priority = priority_pops;
            heap_pops = priority_heap_pops;
            fallback = fallback_tasks;
            queue_full = queue_full_fallbacks;
            start_fail = start_fail_fallbacks;
            high_water = max_queue_depth;
            queued_bytes_high_water = max_queued_bytes;
            expired = cancelled_expired;
            pressure = cancelled_pressure;
            value = cancelled_value;
            queue_cancel = cancelled_queue_full;
            batches = worker_batches;
            batch_candidates = batched_candidates;
            coalesced_saved = coalesced_syscalls_saved;
            cap = capacity;
            workers_started = worker_count;
            priority_on = priority_enabled;
            heap_on = priority_heap_enabled;
            mode = priority_mode;
        }

        std::string line;
        line.reserve(256);
        line += "{\"event\":\"EXPERT_ASYNC_SUMMARY\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
        line += ",\"enqueued\":" + std::to_string(enqueued);
        line += ",\"issued\":" + std::to_string(issued);
        line += ",\"issued_candidates\":" + std::to_string(issued_input);
        line += ",\"priority_enabled\":" + std::string(priority_on ? "true" : "false");
        line += ",\"priority_heap_enabled\":" + std::string(heap_on ? "true" : "false");
        line += ",\"priority_mode\":";
        json_escape_append(line, expert_prefetch_async_priority_mode_name(mode));
        line += ",\"priority_pops\":" + std::to_string(priority);
        line += ",\"priority_heap_pops\":" + std::to_string(heap_pops);
        line += ",\"fallback\":" + std::to_string(fallback);
        line += ",\"queue_full_fallbacks\":" + std::to_string(queue_full);
        line += ",\"start_fail_fallbacks\":" + std::to_string(start_fail);
        line += ",\"max_queue_depth\":" + std::to_string(high_water);
        line += ",\"max_queued_bytes\":" + std::to_string(queued_bytes_high_water);
        line += ",\"queue_capacity\":" + std::to_string(cap);
        line += ",\"workers\":" + std::to_string(workers_started);
        line += ",\"cancelled_expired\":" + std::to_string(expired);
        line += ",\"cancelled_pressure\":" + std::to_string(pressure);
        line += ",\"cancelled_value\":" + std::to_string(value);
        line += ",\"cancelled_queue_full\":" + std::to_string(queue_cancel);
        line += ",\"worker_batches\":" + std::to_string(batches);
        line += ",\"batched_candidates\":" + std::to_string(batch_candidates);
        line += ",\"coalesced_syscalls_saved\":" + std::to_string(coalesced_saved);
        line += ",\"batch_size\":" + std::to_string(expert_prefetch_async_batch_size());
        line += ",\"batch_wait_us\":" + std::to_string(expert_prefetch_async_batch_wait_us());
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }

    size_t queue_depth_unlocked() const {
        return priority_enabled && priority_heap_enabled ?
                priority_heap.size() + legacy_priority_heap.size() : tasks.size();
    }

    bool queue_empty_unlocked() const {
        return queue_depth_unlocked() == 0;
    }

    bool is_higher_priority(const ExpertHintTask & a, const ExpertHintTask & b) const {
        const auto key = [](const ExpertHintTask & task) {
            ExpertHintPriorityKey result;
            result.step = task.step;
            result.layer = task.layer;
            result.stage = task.stage;
            result.route_score = task.route_score;
            result.sequence = task.sequence;
            result.deadline_ts_ns = task.deadline_ts_ns;
            return result;
        };
        return expert_hint_priority_higher(key(a), key(b), priority_mode);
    }
};

ExpertHintQueue & expert_hint_queue() {
    static ExpertHintQueue queue;
    return queue;
}

void shutdown_expert_hint_queue() {
    expert_hint_queue().shutdown();
}

void write_expert_route_hint_summary() {
    expert_tensor_registry().write_route_hint_summary();
}

enum class ExpertTaskGateResult {
    Accept,
    Pressure,
    Value,
};

ExpertTaskGateResult prepare_expert_hint_task(ExpertHintTask & task) {
    task.enqueue_ts_ns = llm_mem_trace_time_ns();
    if (task.route_confidence <= 0.0) {
        task.route_confidence = 1.0;
    }
    const ExpertPressureSnapshot pressure = expert_pressure_controller().snapshot();
    apply_pressure_snapshot(task, pressure);

    const bool stage_deadline_observation = expert_prefetch_async_enabled() &&
            expert_prefetch_async_priority_enabled() &&
            expert_prefetch_async_priority_mode() == ExpertAsyncPriorityMode::StageDeadlineScore;
    if (expert_slack_enabled() || expert_deadline_observation_enabled() ||
            stage_deadline_observation) {
        const uint64_t slack = expert_timing_model().estimate_slack_ns(
                task.step, task.layer, task.phase, task.enqueue_ts_ns);
        task.deadline_ts_ns = task.enqueue_ts_ns + slack;
    }
    task.lifecycle.deadline_ts_ns = task.deadline_ts_ns;

    refresh_expert_task_estimate(task);

    if (expert_feedback_enabled()) {
        const uint64_t queued = expert_prefetch_async_enabled() ?
                expert_hint_queue().queued_bytes_snapshot() : 0;
        if (expert_task_exceeds_pressure_budget(task, queued)) {
            return ExpertTaskGateResult::Pressure;
        }
    }

    if (expert_task_below_value_threshold(task)) {
        return ExpertTaskGateResult::Value;
    }
    return ExpertTaskGateResult::Accept;
}

bool submit_expert_hint_task(ExpertHintTask && task) {
    const ExpertTaskGateResult gate = prepare_expert_hint_task(task);
    if (gate == ExpertTaskGateResult::Pressure) {
        transition_expert_task(task.lifecycle, ExpertTaskEvent::Reject, "pressure_budget");
        write_expert_task_skip(task, "expert_prefetch_skip_pressure", "pressure_budget");
        if (expert_prefetch_async_enabled()) {
            expert_hint_queue().record_cancelled_pressure();
        }
        return false;
    }
    if (gate == ExpertTaskGateResult::Value) {
        transition_expert_task(task.lifecycle, ExpertTaskEvent::Reject, "benefit_below_cost");
        write_expert_task_skip(task, "expert_prefetch_skip_value", "benefit_below_cost");
        if (expert_prefetch_async_enabled()) {
            expert_hint_queue().record_cancelled_value();
        }
        return false;
    }
    transition_expert_task(task.lifecycle, ExpertTaskEvent::Admit);
    if (expert_prefetch_async_enabled()) {
        static const bool registered = [] {
            std::atexit(shutdown_expert_hint_queue);
            return true;
        }();
        (void) registered;
        if (expert_hint_queue().enqueue(std::move(task))) {
            return true;
        }
        if (!expert_prefetch_async_fallback_enabled() || expert_slack_enabled()) {
            expert_hint_queue().record_cancelled_queue_full();
            transition_expert_task(task.lifecycle, ExpertTaskEvent::Cancel, "queue_full");
            write_expert_task_skip(task, "expert_prefetch_cancel_queue_full", "queue_full");
            return false;
        }
        expert_hint_queue().record_fallback();
        task.action += "_fallback";
        task.fadvise_action += "_fallback";
    }
    issue_expert_hint_task(task);
    return true;
}

ExpertHintTask make_expert_hint_task(
        const char * action,
        const char * fadvise_action,
        const char * reason,
        const char * policy,
        const char * tensor_name,
        int layer,
        int expert,
        uintptr_t addr,
        size_t nbytes,
        uint64_t cache_bytes,
        uint64_t cache_capacity_bytes,
        double route_score = 0.0,
        double route_confidence = 0.0,
        bool predicted = false,
        int prediction_source_layer = -1,
        int token_idx = -1) {
    ExpertHintTask task;
    task.action = action ? action : "expert_madvise_willneed";
    task.fadvise_action = fadvise_action ? fadvise_action : "expert_posix_fadvise_willneed";
    task.trigger = reason ? reason : "expert_prefetch";
    task.tensor_name = tensor_name ? tensor_name : "";
    task.policy = policy ? policy : "";
    task.layer = layer;
    task.expert = expert;
    task.addr = addr;
    task.nbytes = nbytes;
    task.cache_bytes = cache_bytes;
    task.cache_capacity_bytes = cache_capacity_bytes;
    task.route_score = route_score == route_score ? route_score : 0.0;
    task.route_confidence = route_confidence == route_confidence ? route_confidence : 0.0;
    task.predicted = predicted;
    task.prediction_source_layer = prediction_source_layer;
    task.token_idx = token_idx;
    task.phase = llm_mem_trace_get_phase();
    task.stage = classify_expert_tensor_stage(task.tensor_name.c_str());
    task.step = llm_mem_trace_get_step();
    task.use_fadvise = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_POSIX_FADVISE");
    if (expert_task_trace_mode() != ExpertTaskTraceMode::Off) {
        task.lifecycle.step = task.step;
        task.lifecycle.layer = task.layer;
        task.lifecycle.expert = task.expert;
        task.lifecycle.phase = task.phase;
        task.lifecycle.stage = task.stage;
        task.lifecycle.tensor_name = task.tensor_name;
        task.lifecycle.addr = task.addr;
        task.lifecycle.nbytes = task.nbytes;
        task.lifecycle.score = task.route_score;
        ensure_expert_task_summary_registered();
        ensure_expert_first_use_summary_registered();
    }
    if (expert_task_detail_events_enabled()) {
        task.lifecycle.task_id = next_expert_task_id();
    }
    transition_expert_task(task.lifecycle, ExpertTaskEvent::Create);
    return task;
}

struct PendingExpertPrefetch {
    const ExpertTensorInfo * info = nullptr;
    uintptr_t addr = 0;
    size_t nbytes = 0;
    int expert = -1;
    double score = 0.0;
    double confidence = 0.0;
};

uintptr_t saturated_range_end(uintptr_t addr, size_t nbytes) {
    const uintptr_t max_addr = std::numeric_limits<uintptr_t>::max();
    if (nbytes > max_addr - addr) {
        return max_addr;
    }
    return addr + nbytes;
}

void apply_route_coalesced_prefetch_hints(
        std::vector<PendingExpertPrefetch> & pending,
        const char * reason,
        const char * policy) {
    if (pending.empty()) {
        return;
    }

    std::sort(pending.begin(), pending.end(), [](const PendingExpertPrefetch & a, const PendingExpertPrefetch & b) {
        const uintptr_t a_tensor = a.info ? a.info->addr : 0;
        const uintptr_t b_tensor = b.info ? b.info->addr : 0;
        if (a_tensor != b_tensor) {
            return a_tensor < b_tensor;
        }
        if (a.addr != b.addr) {
            return a.addr < b.addr;
        }
        return a.expert < b.expert;
    });

    struct MergedRange {
        const ExpertTensorInfo * info = nullptr;
        uintptr_t start = 0;
        uintptr_t end = 0;
        int expert = -1;
        int count = 0;
        double score = 0.0;
        double confidence = 0.0;
    };

    auto flush = [&](const MergedRange & range) {
        if (!range.info || range.start == 0 || range.end <= range.start) {
            return;
        }
        const size_t nbytes = (size_t) (range.end - range.start);
        const int expert = range.count == 1 ? range.expert : -1;
        submit_expert_hint_task(make_expert_hint_task(
                "expert_madvise_willneed_coalesced",
                "expert_posix_fadvise_willneed_coalesced",
                reason,
                policy,
                range.info->name.c_str(),
                range.info->layer,
                expert,
                range.start,
                nbytes,
                0,
                0,
                range.score,
                range.confidence));
    };

    MergedRange current;
    const uint64_t max_gap_bytes = expert_prefetch_coalesce_max_gap_bytes();
    for (const PendingExpertPrefetch & entry : pending) {
        if (!entry.info) {
            continue;
        }
        uintptr_t start = 0;
        size_t len = 0;
        if (!page_aligned_range(entry.addr, entry.nbytes, start, len)) {
            continue;
        }
        const uintptr_t end = saturated_range_end(start, len);
        if (end <= start) {
            continue;
        }

        const uintptr_t merge_limit = saturated_range_end(current.end, (size_t) std::min<uint64_t>(max_gap_bytes, (uint64_t) std::numeric_limits<size_t>::max()));
        if (current.info == entry.info && start <= merge_limit) {
            current.end = std::max(current.end, end);
            current.count++;
            current.score = std::max(current.score, entry.score);
            current.confidence = std::max(current.confidence, entry.confidence);
            if (current.expert != entry.expert) {
                current.expert = -1;
            }
            continue;
        }

        flush(current);
        current.info = entry.info;
        current.start = start;
        current.end = end;
        current.expert = entry.expert;
        current.count = 1;
        current.score = entry.score;
        current.confidence = entry.confidence;
    }
    flush(current);
}

bool expert_cross_layer_predict_enabled() {
    static const bool enabled = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT");
    return enabled;
}

size_t expert_cross_layer_predict_topk() {
    static const size_t value = env_size_or_default("LLM_MEM_TRACE_OPT_EXPERT_PREDICT_TOPK", 2);
    return std::max<size_t>(1, std::min<size_t>(value, 32));
}

struct ExpertPrediction {
    int expert = -1;
    double confidence = 0.0;
};

struct ExpertRouteObservation {
    int layer = -1;
    std::vector<int> experts;
    std::vector<double> weights;
};

struct ExpertTransitionBucket {
    uint64_t samples = 0;
    std::unordered_map<int, uint64_t> destination_hits;
};

struct PendingExpertPrediction {
    int target_layer = -1;
    std::vector<int> experts;
};

struct ExpertCrossLayerPredictor {
    std::mutex mu;
    uint64_t active_step = 0;
    std::unordered_map<int, ExpertRouteObservation> token_routes;
    std::unordered_map<int, PendingExpertPrediction> token_predictions;
    std::unordered_map<uint64_t, ExpertTransitionBucket> transitions;
    uint64_t observed_routes = 0;
    uint64_t learned_transitions = 0;
    uint64_t prediction_sets = 0;
    uint64_t prediction_candidates = 0;
    uint64_t evaluated_sets = 0;
    uint64_t evaluated_candidates = 0;
    uint64_t prediction_hits = 0;
    uint64_t prediction_set_hits = 0;
    uint64_t actual_experts_evaluated = 0;
    uint64_t unevaluated_sets = 0;
    uint64_t capacity_skips = 0;
    uint64_t destination_replacements = 0;

    static uint64_t transition_key(int layer, int expert) {
        return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert;
    }

    std::vector<ExpertPrediction> observe_and_predict(
            uint64_t step,
            int token_idx,
            int layer,
            const int * experts,
            const float * scores,
            int n_experts,
            bool allow_prediction) {
        std::vector<ExpertPrediction> result;
        if (!expert_cross_layer_predict_enabled() || token_idx < 0 || layer < 0 ||
                !experts || n_experts <= 0) {
            return result;
        }

        static const bool registered = [] {
            std::atexit(write_summary_at_exit);
            return true;
        }();
        (void) registered;

        ExpertRouteObservation current;
        current.layer = layer;
        double positive_sum = 0.0;
        if (scores) {
            for (int i = 0; i < n_experts; ++i) {
                if (experts[i] >= 0 && std::isfinite(scores[i]) && scores[i] > 0.0f) {
                    positive_sum += (double) scores[i];
                }
            }
        }
        std::unordered_set<int> seen;
        for (int i = 0; i < n_experts; ++i) {
            const int expert = experts[i];
            if (expert < 0 || !seen.insert(expert).second) {
                continue;
            }
            const double weight = positive_sum > 0.0 && scores && scores[i] > 0.0f ?
                    (double) scores[i] / positive_sum : 1.0 / (double) std::max(1, n_experts);
            current.experts.push_back(expert);
            current.weights.push_back(weight);
        }
        if (current.experts.empty()) {
            return result;
        }

        const uint64_t min_samples = env_u64_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_PREDICT_MIN_SAMPLES", 8);
        const size_t max_buckets = env_size_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_PREDICT_MAX_BUCKETS", 16384);
        const size_t max_destinations = env_size_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_PREDICT_MAX_DESTINATIONS", 64);
        const double min_confidence = env_double_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_PREDICT_MIN_CONFIDENCE", 0.10);

        std::lock_guard<std::mutex> lock(mu);
        if (active_step != step) {
            unevaluated_sets += token_predictions.size();
            token_predictions.clear();
            token_routes.clear();
            active_step = step;
        }
        observed_routes++;

        auto pending_it = token_predictions.find(token_idx);
        if (pending_it != token_predictions.end()) {
            if (pending_it->second.target_layer == layer) {
                std::unordered_set<int> actual(current.experts.begin(), current.experts.end());
                bool set_hit = false;
                evaluated_sets++;
                evaluated_candidates += pending_it->second.experts.size();
                actual_experts_evaluated += actual.size();
                for (int predicted : pending_it->second.experts) {
                    if (actual.find(predicted) != actual.end()) {
                        prediction_hits++;
                        set_hit = true;
                    }
                }
                if (set_hit) {
                    prediction_set_hits++;
                }
            } else if (pending_it->second.target_layer > layer) {
                pending_it = token_predictions.end();
            } else {
                unevaluated_sets++;
            }
            if (pending_it != token_predictions.end()) {
                token_predictions.erase(pending_it);
            }
        }

        auto previous_it = token_routes.find(token_idx);
        if (previous_it != token_routes.end() && previous_it->second.layer + 1 == layer) {
            for (int source_expert : previous_it->second.experts) {
                const uint64_t key = transition_key(previous_it->second.layer, source_expert);
                auto bucket_it = transitions.find(key);
                if (bucket_it == transitions.end()) {
                    if (transitions.size() >= max_buckets) {
                        capacity_skips++;
                        continue;
                    }
                    bucket_it = transitions.emplace(key, ExpertTransitionBucket{}).first;
                }
                ExpertTransitionBucket & bucket = bucket_it->second;
                bucket.samples++;
                for (int destination : current.experts) {
                    auto destination_it = bucket.destination_hits.find(destination);
                    if (destination_it == bucket.destination_hits.end()) {
                        if (bucket.destination_hits.size() >= max_destinations) {
                            auto minimum = std::min_element(
                                    bucket.destination_hits.begin(), bucket.destination_hits.end(),
                                    [](const auto & a, const auto & b) { return a.second < b.second; });
                            const uint64_t replacement_count = minimum != bucket.destination_hits.end() ?
                                    minimum->second + 1 : 1;
                            if (minimum != bucket.destination_hits.end()) {
                                bucket.destination_hits.erase(minimum);
                            }
                            bucket.destination_hits.emplace(destination, replacement_count);
                            destination_replacements++;
                            continue;
                        }
                        destination_it = bucket.destination_hits.emplace(destination, 0).first;
                    }
                    destination_it->second++;
                }
                learned_transitions++;
            }
        }
        token_routes[token_idx] = current;

        if (!allow_prediction) {
            return result;
        }

        std::unordered_map<int, double> confidence_by_expert;
        for (size_t i = 0; i < current.experts.size(); ++i) {
            const auto bucket_it = transitions.find(transition_key(layer, current.experts[i]));
            if (bucket_it == transitions.end() || bucket_it->second.samples < min_samples) {
                continue;
            }
            const ExpertTransitionBucket & bucket = bucket_it->second;
            const double source_weight = i < current.weights.size() ? current.weights[i] : 0.0;
            for (const auto & item : bucket.destination_hits) {
                const double probability = std::min(1.0, (double) item.second / (double) bucket.samples);
                confidence_by_expert[item.first] += source_weight * probability;
            }
        }

        result.reserve(confidence_by_expert.size());
        for (const auto & item : confidence_by_expert) {
            if (item.second >= min_confidence) {
                result.push_back({item.first, std::min(1.0, item.second)});
            }
        }
        std::sort(result.begin(), result.end(), [](const ExpertPrediction & a, const ExpertPrediction & b) {
            if (a.confidence != b.confidence) {
                return a.confidence > b.confidence;
            }
            return a.expert < b.expert;
        });
        if (result.size() > expert_cross_layer_predict_topk()) {
            result.resize(expert_cross_layer_predict_topk());
        }
        if (!result.empty()) {
            PendingExpertPrediction pending;
            pending.target_layer = layer + 1;
            for (const ExpertPrediction & prediction : result) {
                pending.experts.push_back(prediction.expert);
            }
            token_predictions[token_idx] = std::move(pending);
            prediction_sets++;
            prediction_candidates += result.size();
        }
        return result;
    }

    void write_summary() {
        if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu);
        const double precision = evaluated_candidates > 0 ?
                100.0 * (double) prediction_hits / (double) evaluated_candidates : 0.0;
        const double recall = actual_experts_evaluated > 0 ?
                100.0 * (double) prediction_hits / (double) actual_experts_evaluated : 0.0;
        const double set_hit_rate = evaluated_sets > 0 ?
                100.0 * (double) prediction_set_hits / (double) evaluated_sets : 0.0;
        std::string line;
        line.reserve(384);
        line += "{\"event\":\"EXPERT_PREDICT_SUMMARY\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
        line += ",\"observed_routes\":" + std::to_string(observed_routes);
        line += ",\"learned_transitions\":" + std::to_string(learned_transitions);
        line += ",\"transition_buckets\":" + std::to_string(transitions.size());
        line += ",\"prediction_sets\":" + std::to_string(prediction_sets);
        line += ",\"prediction_candidates\":" + std::to_string(prediction_candidates);
        line += ",\"evaluated_sets\":" + std::to_string(evaluated_sets);
        line += ",\"evaluated_candidates\":" + std::to_string(evaluated_candidates);
        line += ",\"prediction_hits\":" + std::to_string(prediction_hits);
        line += ",\"prediction_set_hits\":" + std::to_string(prediction_set_hits);
        line += ",\"actual_experts_evaluated\":" + std::to_string(actual_experts_evaluated);
        line += ",\"precision_pct\":" + std::to_string(precision);
        line += ",\"recall_pct\":" + std::to_string(recall);
        line += ",\"set_hit_rate_pct\":" + std::to_string(set_hit_rate);
        line += ",\"unevaluated_sets\":" + std::to_string(unevaluated_sets + token_predictions.size());
        line += ",\"capacity_skips\":" + std::to_string(capacity_skips);
        line += ",\"destination_replacements\":" + std::to_string(destination_replacements);
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }

    static void write_summary_at_exit();
};

ExpertCrossLayerPredictor & expert_cross_layer_predictor() {
    static ExpertCrossLayerPredictor predictor;
    return predictor;
}

void ExpertCrossLayerPredictor::write_summary_at_exit() {
    expert_cross_layer_predictor().write_summary();
}

size_t submit_cross_layer_predictions(
        uint64_t step,
        int source_layer,
        int token_idx,
        const std::vector<ExpertPrediction> & predictions,
        int phase) {
    if (predictions.empty()) {
        return 0;
    }
    const int target_layer = source_layer + 1;
    const std::vector<ExpertTensorInfo> tensors = expert_tensor_registry().for_layer(target_layer);
    if (tensors.empty()) {
        return 0;
    }
    const uint64_t ttl = expert_route_hint_ttl_steps_for_phase(phase);
    size_t accepted = 0;
    for (const ExpertPrediction & prediction : predictions) {
        for (const ExpertTensorInfo & info : tensors) {
            uintptr_t slice_addr = 0;
            size_t slice_bytes = 0;
            if (!expert_slice_range(info, prediction.expert, slice_addr, slice_bytes) ||
                    !os_hint_size_allowed(slice_bytes) ||
                    expert_tensor_registry().was_hinted(step, target_layer, prediction.expert, info.addr, ttl)) {
                continue;
            }
            ExpertHintTask task = make_expert_hint_task(
                    "expert_madvise_willneed_predicted",
                    "expert_posix_fadvise_willneed_predicted",
                    "cross_layer_predict",
                    "cross_layer_value",
                    info.name.c_str(),
                    target_layer,
                    prediction.expert,
                    slice_addr,
                    slice_bytes,
                    0,
                    0,
                    prediction.confidence,
                    prediction.confidence,
                    true,
                    source_layer,
                    token_idx);
            if (submit_expert_hint_task(std::move(task))) {
                (void) expert_tensor_registry().mark_hinted(
                        step, target_layer, prediction.expert, info.addr, ttl);
                accepted++;
            }
        }
    }
    return accepted;
}

struct ExpertCacheItem {
    std::string key;
    std::string tensor_name;
    int layer = -1;
    int expert = -1;
    uintptr_t addr = 0;
    size_t nbytes = 0;
    uint64_t first_step = 0;
    uint64_t last_step = 0;
    uint64_t hit_count = 0;
    uint64_t recent_hits = 0;
    uint64_t recent_epoch = 0;
    uint64_t avg_gap = 0;
    double score = 0.0;
    bool advised = false;
    bool resident = false;
};

void apply_expert_evict_hint(
        const ExpertCacheItem & item,
        const char * reason,
        const char * policy,
        uint64_t cache_bytes,
        uint64_t cache_capacity_bytes) {
    OsHintMeta meta;
    meta.policy = policy;
    meta.decision = "evict";
    meta.cache_bytes = cache_bytes;
    meta.cache_capacity_bytes = cache_capacity_bytes;
    meta.cache_hit = false;
    meta.has_cache_hit = true;

    switch (expert_evict_advice()) {
        case ExpertEvictAdvice::None:
            write_os_hint_event("expert_cache_evict", reason ? reason : "expert_cache",
                                item.tensor_name.c_str(), item.layer, item.expert, item.addr, item.nbytes,
                                0, 0, 0, 0, &meta);
            return;
        case ExpertEvictAdvice::Cold:
#ifdef MADV_COLD
            apply_madvise_hint("expert_madvise_cold", MADV_COLD, reason ? reason : "expert_cache",
                               item.tensor_name.c_str(), item.layer, item.expert, item.addr, item.nbytes, &meta);
#else
            write_os_hint_event("expert_madvise_cold", reason ? reason : "expert_cache",
                                item.tensor_name.c_str(), item.layer, item.expert, item.addr, item.nbytes,
                                0, -1, ENOSYS, 0, &meta);
#endif
            return;
        case ExpertEvictAdvice::DontNeed:
#ifdef __linux__
            apply_madvise_hint("expert_madvise_dontneed", MADV_DONTNEED, reason ? reason : "expert_cache",
                               item.tensor_name.c_str(), item.layer, item.expert, item.addr, item.nbytes, &meta);
#else
            write_os_hint_event("expert_madvise_dontneed", reason ? reason : "expert_cache",
                                item.tensor_name.c_str(), item.layer, item.expert, item.addr, item.nbytes,
                                0, -1, ENOSYS, 0, &meta);
#endif
            return;
        case ExpertEvictAdvice::PageOut:
#ifdef MADV_PAGEOUT
            apply_madvise_hint("expert_madvise_pageout", MADV_PAGEOUT, reason ? reason : "expert_cache",
                               item.tensor_name.c_str(), item.layer, item.expert, item.addr, item.nbytes, &meta);
#else
            write_os_hint_event("expert_madvise_pageout", reason ? reason : "expert_cache",
                                item.tensor_name.c_str(), item.layer, item.expert, item.addr, item.nbytes,
                                0, -1, ENOSYS, 0, &meta);
#endif
            return;
    }
}

struct ExpertSliceCache {
    std::mutex mu;
    std::unordered_map<std::string, ExpertCacheItem> items;
    uint64_t bytes = 0;

    void touch(
            const ExpertTensorInfo & info,
            int expert,
            double score,
            uintptr_t addr,
            size_t nbytes,
            uint64_t step,
            const char * reason) {
        const ExpertPolicy policy = expert_policy();
        const char * policy_name = expert_policy_name(policy);
        const ExpertPressureSnapshot pressure = expert_pressure_controller().snapshot();
        const uint64_t capacity = expert_feedback_enabled() ?
                pressure.prefetch_budget_bytes : expert_cache_capacity_bytes();
        if (capacity == 0 || nbytes > capacity) {
            write_expert_cache_event("expert_cache_skip", reason ? reason : "expert_cache",
                                     policy_name, "skip", false, info.name.c_str(), info.layer, expert,
                                     addr, nbytes, bytes, capacity);
            return;
        }

        const std::string key = expert_slice_key(info, expert);
        std::lock_guard<std::mutex> lock(mu);

        auto existing = items.find(key);
        if (existing != items.end()) {
            update_item(existing->second, step, score);
            write_expert_cache_event("expert_cache_hit", reason ? reason : "expert_cache",
                                     policy_name, "hit", true, info.name.c_str(), info.layer, expert,
                                     addr, nbytes, bytes, capacity);
            return;
        }

        ExpertCacheItem item;
        item.key = key;
        item.tensor_name = info.name;
        item.layer = info.layer;
        item.expert = expert;
        item.addr = addr;
        item.nbytes = nbytes;
        item.first_step = step;
        item.last_step = step;
        item.hit_count = 1;
        item.recent_hits = 1;
        item.recent_epoch = step;
        item.score = score;
        item.advised = true;
        item.resident = true;
        items.emplace(key, item);
        bytes += (uint64_t) nbytes;

        apply_expert_prefetch_hint(info, expert, addr, nbytes, reason, policy_name, bytes, capacity);
        evict_stale(step, key, reason, policy_name, capacity);
        evict_until_within_budget(step, key, reason, policy_name, capacity);
    }

    void update_item(ExpertCacheItem & item, uint64_t step, double score) {
        const uint64_t ttl = std::max<uint64_t>(1, expert_ttl_steps());
        if (item.last_step != 0 && step > item.last_step) {
            const uint64_t gap = step - item.last_step;
            item.avg_gap = item.avg_gap == 0 ? gap : (item.avg_gap * 3 + gap + 2) / 4;
        }
        if (item.recent_epoch == 0) {
            item.recent_epoch = step;
        } else if (step > item.recent_epoch + ttl) {
            const uint64_t windows = std::min<uint64_t>((step - item.recent_epoch) / ttl, 8);
            item.recent_hits >>= windows;
            item.recent_epoch = step;
        }
        item.last_step = step;
        item.hit_count++;
        item.recent_hits++;
        item.score = score;
        item.advised = true;
        item.resident = true;
    }

    void evict_stale(uint64_t step, const std::string & protected_key, const char * reason, const char * policy, uint64_t capacity) {
        const uint64_t ttl = expert_ttl_steps();
        if (ttl == 0) {
            return;
        }

        std::vector<std::string> stale;
        stale.reserve(items.size());
        for (const auto & kv : items) {
            const ExpertCacheItem & item = kv.second;
            if (item.key != protected_key && step > item.last_step && step - item.last_step > ttl) {
                stale.push_back(item.key);
            }
        }

        for (const std::string & key : stale) {
            auto it = items.find(key);
            if (it == items.end()) {
                continue;
            }
            ExpertCacheItem item = it->second;
            bytes -= std::min<uint64_t>(bytes, item.nbytes);
            items.erase(it);
            apply_expert_evict_hint(item, reason, policy, bytes, capacity);
        }
    }

    void evict_until_within_budget(uint64_t step, const std::string & protected_key, const char * reason, const char * policy, uint64_t capacity) {
        while (bytes > capacity && !items.empty()) {
            auto victim = choose_victim(step, protected_key);
            if (victim == items.end()) {
                break;
            }
            ExpertCacheItem item = victim->second;
            bytes -= std::min<uint64_t>(bytes, item.nbytes);
            items.erase(victim);
            apply_expert_evict_hint(item, reason, policy, bytes, capacity);
        }
    }

    std::unordered_map<std::string, ExpertCacheItem>::iterator choose_victim(uint64_t step, const std::string & protected_key) {
        auto best = items.end();
        for (auto it = items.begin(); it != items.end(); ++it) {
            if (it->first == protected_key && items.size() > 1) {
                continue;
            }
            if (best == items.end() || is_better_victim(it->second, best->second, step)) {
                best = it;
            }
        }
        return best;
    }

    bool is_better_victim(const ExpertCacheItem & cur, const ExpertCacheItem & best, uint64_t step) const {
        const ExpertPolicy policy = expert_policy();
        switch (policy) {
            case ExpertPolicy::Lru:
                return cur.last_step < best.last_step ||
                       (cur.last_step == best.last_step && cur.hit_count < best.hit_count);
            case ExpertPolicy::Lfu:
                return cur.hit_count < best.hit_count ||
                       (cur.hit_count == best.hit_count && cur.last_step < best.last_step);
            case ExpertPolicy::WindowLfu:
                return cur.recent_hits < best.recent_hits ||
                       (cur.recent_hits == best.recent_hits && cur.last_step < best.last_step);
            case ExpertPolicy::LeastStale:
                return least_stale_victim_score(cur, step) > least_stale_victim_score(best, step) ||
                       (least_stale_victim_score(cur, step) == least_stale_victim_score(best, step) &&
                        cur.hit_count < best.hit_count);
            case ExpertPolicy::Route:
                return cur.last_step < best.last_step;
        }
        return false;
    }

    int64_t least_stale_victim_score(const ExpertCacheItem & item, uint64_t step) const {
        const uint64_t gap = item.avg_gap > 0 ? item.avg_gap : std::max<uint64_t>(1, expert_ttl_steps());
        const uint64_t predicted_next = item.last_step + gap;
        if (predicted_next <= step) {
            return (int64_t) (1000000000ull + step - predicted_next);
        }
        return (int64_t) (predicted_next - step);
    }
};

ExpertSliceCache & expert_slice_cache() {
    static ExpertSliceCache cache;
    return cache;
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

    const char * name = ggml_get_name(t);
    const size_t nbytes = ggml_nbytes(t);
    const int layer = parse_layer_from_name(name);
    const uintptr_t addr = tensor_addr(t);
    const char * backend = tensor_backend_name(t);
    const uint64_t ts = llm_mem_trace_time_ns();

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
        expert_timing_model().on_layer_begin(step, layer, llm_mem_trace_get_phase(), ts);
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
        expert_timing_model().on_layer_end(step, layer, llm_mem_trace_get_phase(), ts);
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

bool host_readable_tensor(const ggml_tensor * t) {
    if (!t || !t->data) {
        return false;
    }
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    return buffer && ggml_backend_buffer_is_host(buffer);
}

int read_expert_id(const ggml_tensor * ids, int64_t index0, int64_t index1) {
    const char * ptr = static_cast<const char *>(ids->data) +
            (size_t) index0 * ids->nb[0] + (size_t) index1 * ids->nb[1];
    if (ids->type == GGML_TYPE_I32) {
        int32_t value = -1;
        std::memcpy(&value, ptr, sizeof(value));
        return value;
    }
    if (ids->type == GGML_TYPE_I64) {
        int64_t value = -1;
        std::memcpy(&value, ptr, sizeof(value));
        return (int) value;
    }
    return -1;
}

void observe_expert_logical_first_use(const ggml_tensor * operation) {
    if (expert_task_trace_mode() == ExpertTaskTraceMode::Off || !operation ||
            operation->op != GGML_OP_MUL_MAT_ID) {
        return;
    }
    ensure_expert_task_summary_registered();
    ensure_expert_first_use_summary_registered();
    const ggml_tensor * weights = operation->src[0];
    const ggml_tensor * ids = operation->src[2];
    const char * tensor_name = weights ? ggml_get_name(weights) : nullptr;
    if (!weights || !ids || !is_expert_weight_tensor_name(tensor_name) ||
            !host_readable_tensor(ids) ||
            (ids->type != GGML_TYPE_I32 && ids->type != GGML_TYPE_I64)) {
        return;
    }

    ExpertTensorInfo info;
    info.name = tensor_name ? tensor_name : "";
    info.layer = parse_layer_from_name(tensor_name);
    info.addr = tensor_addr(weights);
    info.nbytes = ggml_nbytes(weights);
    info.n_expert = weights->ne[2];
    info.expert_stride = (size_t) weights->nb[2];
    if (info.layer < 0 || info.addr == 0 || info.nbytes == 0 ||
            info.n_expert <= 0 || info.expert_stride == 0) {
        return;
    }

    std::unordered_set<int> experts;
    for (int64_t token = 0; token < ids->ne[1]; ++token) {
        for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
            const int expert = read_expert_id(ids, rank, token);
            if (expert >= 0 && expert < info.n_expert) {
                experts.insert(expert);
            }
        }
    }

    const uint64_t step = llm_mem_trace_get_step();
    const uint64_t first_use_ts_ns = llm_mem_trace_time_ns();
    for (int expert : experts) {
        uintptr_t slice_addr = 0;
        size_t slice_bytes = 0;
        if (!expert_slice_range(info, expert, slice_addr, slice_bytes)) {
            continue;
        }
        ExpertFirstUseObservation use;
        use.step = step;
        use.layer = info.layer;
        use.expert = expert;
        use.phase = llm_mem_trace_get_phase();
        use.stage = classify_expert_tensor_stage(info.name.c_str());
        use.tensor = info.name;
        use.addr = slice_addr;
        use.nbytes = slice_bytes;
        use.first_use_ts_ns = first_use_ts_ns;
        write_expert_first_use_event(expert_first_use_matcher().observe_first_use(std::move(use)));
    }
}

} // namespace

extern "C" void llm_mem_trace_tensor_begin(const ggml_tensor * t) {
    if (!llm_mem_trace_enabled() || !t) {
        return;
    }

    observe_expert_logical_first_use(t);
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
    if (!t) {
        return;
    }

    const char * name = ggml_get_name(t);
    const size_t nbytes = ggml_nbytes(t);
    const int layer = parse_layer_from_name(name);
    const uintptr_t addr = tensor_addr(t);
    const char * backend = tensor_backend_name(t);
    const bool mapped_tensor = (stage && std::strcmp(stage, "mmap") == 0) ||
                               (backend && std::strstr(backend, "Mapped") != nullptr);

    expert_tensor_registry().add(t, name, layer, addr, nbytes);
    apply_load_os_hints("tensor_load", name, layer, addr, nbytes, mapped_tensor);

    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_TENSOR)) {
        return;
    }

    const uint64_t ts = llm_mem_trace_time_ns();
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

extern "C" void llm_mem_trace_prefetch_expert_layer(int layer, int token_idx, const int * experts, const float * scores, int n_experts, const char * reason) {
    if (!os_hints_enabled() || !os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_PREFETCH") ||
            layer < 0 || !experts || n_experts <= 0) {
        return;
    }

    const std::vector<ExpertTensorInfo> tensors = expert_tensor_registry().for_layer(layer);
    if (tensors.empty()) {
        return;
    }

    const uint64_t step = llm_mem_trace_get_step();
    const int phase = llm_mem_trace_get_phase();
    const uint64_t route_hint_ttl = expert_route_hint_ttl_steps_for_phase(phase);
    const int topk = expert_prefetch_topk_for_phase(phase);
    const int limit = topk > 0 ? std::min(n_experts, topk) : n_experts;
    const ExpertPolicy policy = expert_policy();
    const char * policy_name = expert_policy_name(policy);
    const bool coalesce_route = policy == ExpertPolicy::Route && expert_prefetch_coalesce_enabled();
    if (policy == ExpertPolicy::Route) {
        static const bool registered = [] {
            std::atexit(write_expert_route_hint_summary);
            return true;
        }();
        (void) registered;
    }
    std::vector<PendingExpertPrefetch> pending_coalesced;
    if (coalesce_route) {
        pending_coalesced.reserve((size_t) limit * tensors.size());
    }

    for (int i = 0; i < limit; ++i) {
        const int expert = experts[i];
        if (expert < 0) {
            continue;
        }
        const double score = scores ? (double) scores[i] : 0.0;
        // Routed experts are certain to execute; router weights rank their contribution,
        // but are not probabilities that the selected expert will be used.
        const double confidence = 1.0;
        for (const ExpertTensorInfo & info : tensors) {
            uintptr_t slice_addr = 0;
            size_t slice_bytes = 0;
            if (!expert_slice_range(info, expert, slice_addr, slice_bytes)) {
                continue;
            }
            if (!os_hint_size_allowed(slice_bytes)) {
                continue;
            }

            if (policy != ExpertPolicy::Route) {
                expert_slice_cache().touch(info, expert, score, slice_addr, slice_bytes, step, reason);
                continue;
            }

            if (!expert_tensor_registry().mark_hinted(step, layer, expert, info.addr, route_hint_ttl)) {
                continue;
            }

            if (coalesce_route) {
                pending_coalesced.push_back({&info, slice_addr, slice_bytes, expert, score, confidence});
                continue;
            }

            submit_expert_hint_task(make_expert_hint_task(
                    "expert_madvise_willneed",
                    "expert_posix_fadvise_willneed",
                    reason,
                    policy_name,
                    info.name.c_str(),
                    layer,
                    expert,
                    slice_addr,
                    slice_bytes,
                    0,
                    0,
                    score,
                    confidence));
        }
    }

    if (coalesce_route) {
        apply_route_coalesced_prefetch_hints(pending_coalesced, reason, policy_name);
    }

    if (policy == ExpertPolicy::Route && expert_cross_layer_predict_enabled()) {
        const bool has_next_expert_layer = !expert_tensor_registry().for_layer(layer + 1).empty();
        const std::vector<ExpertPrediction> predictions =
                expert_cross_layer_predictor().observe_and_predict(
                        step, token_idx, layer, experts, scores, n_experts, has_next_expert_layer);
        (void) submit_cross_layer_predictions(step, layer, token_idx, predictions, phase);
    }
}
