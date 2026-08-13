#include "trace_event.h"
#include "expert_memory_object.h"
#include "expert_calibration_shadow.h"
#include "expert_prefetch_types.h"
#include "expert_tensor_registry.h"
#include "expert_prefetch_policy.h"
#include "expert_hint_priority.h"
#include "expert_tensor_stage.h"
#include "expert_task_lifecycle.h"
#include "expert_first_use_matcher.h"
#include "residency_attribution.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
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

bool expert_inflight_hint_aggregation_requested() {
    static const bool requested = env_truthy(
            std::getenv("LLM_MEM_TRACE_OPT_EXPERT_INFLIGHT_HINT_AGGREGATION"));
    return requested;
}

bool expert_semantic_stale_cancel_requested() {
    static const bool requested = env_truthy(
            std::getenv("LLM_MEM_TRACE_OPT_EXPERT_SEMANTIC_STALE_CANCEL"));
    return requested;
}

bool expert_madv_cold_reclaim_requested() {
    static const bool requested = env_truthy(
            std::getenv("LLM_MEM_TRACE_OPT_EXPERT_MADV_COLD_RECLAIM"));
    return requested;
}

uint64_t expert_madv_cold_reclaim_grace_steps() {
    static const uint64_t grace_steps = [] {
        const char * value = std::getenv("LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_GRACE_STEPS");
        if (!value || !value[0]) {
            return uint64_t{3};
        }
        char * end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        return end && *end == '\0' && parsed > 0 ? (uint64_t) parsed : uint64_t{3};
    }();
    return grace_steps;
}

uint64_t expert_working_set_budget_bytes() {
    static const uint64_t budget = [] {
        const char * value = std::getenv("LLM_MEM_TRACE_OPT_EXPERT_WORKING_SET_MB");
        if (!value || !value[0]) {
            return uint64_t{0};
        }
        char * end = nullptr;
        const unsigned long long mb = std::strtoull(value, &end, 10);
        if (!end || *end != '\0' || mb == 0) {
            return uint64_t{0};
        }
        constexpr uint64_t mib = 1024ull * 1024ull;
        return mb > std::numeric_limits<uint64_t>::max() / mib ?
                std::numeric_limits<uint64_t>::max() : (uint64_t) mb * mib;
    }();
    return budget;
}

bool expert_working_set_requested() {
    return expert_working_set_budget_bytes() > 0;
}

bool expert_madv_cold_reclaim_enabled() {
    return expert_madv_cold_reclaim_requested() && expert_working_set_requested() &&
            llm_mem_trace_enabled();
}

bool expert_memory_objects_enabled() {
    static const bool lifecycle_enabled = env_truthy(
            std::getenv("LLM_MEM_TRACE_OPT_EXPERT_MEMORY_OBJECTS"));
    return (lifecycle_enabled || expert_inflight_hint_aggregation_requested() ||
            expert_semantic_stale_cancel_requested() || expert_working_set_requested() ||
            expert_madv_cold_reclaim_enabled()) &&
            llm_mem_trace_enabled();
}

bool expert_inflight_hint_aggregation_enabled() {
    return expert_inflight_hint_aggregation_requested() && llm_mem_trace_enabled();
}

bool expert_semantic_stale_cancel_enabled() {
    return expert_semantic_stale_cancel_requested() && llm_mem_trace_enabled();
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

bool expert_prefetch_control_enabled() {
    return os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_PREFETCH");
}

// Phase 2E-A: observation-only shadow calibration. Default OFF; enabled via
// LLM_MEM_TRACE_OS_HINTS + LLM_MEM_TRACE_OPT_EXPERT_CALIBRATION_SHADOW.
// Strictly additive — never affects any control behavior (spec §15).
bool expert_calibration_shadow_enabled() {
    return os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_CALIBRATION_SHADOW") &&
            llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY);
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

uint64_t env_us_to_ns_or_default(const char * key, uint64_t def_us) {
    const uint64_t value_us = env_u64_or_default(key, def_us);
    return value_us > std::numeric_limits<uint64_t>::max() / 1000ull ?
            std::numeric_limits<uint64_t>::max() : value_us * 1000ull;
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
    const char * decision = nullptr;
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
    if (meta && meta->decision && meta->decision[0]) {
        line += ",\"decision\":";
        json_escape_append(line, meta->decision);
    }
    if (meta) {
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

int apply_madvise_hint(
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
        return -1;
    }
    errno = 0;
    const int rc = madvise(reinterpret_cast<void *>(start), len, advice);
    const int err = rc == 0 ? 0 : errno;
    write_os_hint_event(action, trigger, tensor_name, layer, expert, addr, nbytes, len, rc, err, 0, meta);
    return rc;
#else
    (void) action; (void) advice; (void) trigger; (void) tensor_name; (void) layer; (void) expert; (void) addr; (void) nbytes; (void) meta;
    return -1;
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

const char * residency_object_class(const char * name) {
    if (!name) {
        return "Other";
    }
    if (std::strstr(name, "_exps.weight")) {
        return "Routed Expert";
    }
    if (std::strstr(name, "_shexp.")) {
        return "Shared Expert";
    }
    if (std::strstr(name, "ffn_gate_inp.weight")) {
        return "Router/Gate";
    }
    if (std::strcmp(name, "token_embd.weight") == 0) {
        return "Embedding";
    }
    if (std::strcmp(name, "output.weight") == 0) {
        return "Output";
    }
    if (std::strstr(name, ".attn_")) {
        return "Attention";
    }
    if (std::strstr(name, ".ssm_") || std::strstr(name, ".ssm_a") ||
            std::strstr(name, ".ssm_b") || std::strstr(name, ".ssm_conv1d")) {
        return "SSM";
    }
    if (std::strstr(name, "norm")) {
        return "Norm";
    }
    return "Other";
}

const char * residency_tensor_subclass(const char * name) {
    if (!name) {
        return "";
    }
    if (std::strstr(name, "ffn_down_exps.weight")) {
        return "Down";
    }
    if (std::strstr(name, "ffn_gate_exps.weight") ||
            std::strstr(name, "ffn_up_exps.weight") ||
            std::strstr(name, "ffn_gate_up_exps.weight")) {
        return "Gate/Up";
    }
    return "";
}

bool host_readable_tensor(const ggml_tensor * t);
int read_expert_id(const ggml_tensor * ids, int64_t index0, int64_t index1);

void observe_param_residency_demand(
        const ggml_tensor * t,
        const ggml_tensor * operation) {
    if (!llm_mem_trace_residency_attribution_enabled() || !t) {
        return;
    }
    const char * name = ggml_get_name(t);
    // Routed Expert tensors are accounted by their actual selected Expert
    // slices in observe_expert_logical_first_use(). Never charge the full
    // container tensor here.
    if (!is_param_tensor(t) || is_expert_weight_tensor_name(name)) {
        return;
    }
    if (std::strcmp(name, "token_embd.weight") == 0 && operation &&
            operation->op == GGML_OP_GET_ROWS && operation->src[1] &&
            host_readable_tensor(operation->src[1]) &&
            (operation->src[1]->type == GGML_TYPE_I32 ||
             operation->src[1]->type == GGML_TYPE_I64)) {
        const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
        std::unordered_set<int> rows;
        for (int64_t i1 = 0; i1 < operation->src[1]->ne[1]; ++i1) {
            for (int64_t i0 = 0; i0 < operation->src[1]->ne[0]; ++i0) {
                const int row = read_expert_id(operation->src[1], i0, i1);
                if (row >= 0 && row < t->ne[1]) {
                    rows.insert(row);
                }
            }
        }
        for (int row : rows) {
            llm_mem_trace_residency_attribution_observe(
                    "Embedding", name, "Token Row",
                    parse_layer_from_name(name), -1,
                    tensor_addr(t) + (uintptr_t) row * t->nb[1], row_bytes);
        }
        return;
    }
    llm_mem_trace_residency_attribution_observe(
            residency_object_class(name), name, residency_tensor_subclass(name),
            parse_layer_from_name(name), -1, tensor_addr(t), ggml_nbytes(t));
}

uint64_t expert_prefetch_budget_bytes() {
    const uint64_t mib = 1024ull * 1024ull;
    const uint64_t mb = env_u64_or_default(
            "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_BUDGET_MB", 512);
    if (mb > std::numeric_limits<uint64_t>::max() / mib) {
        return std::numeric_limits<uint64_t>::max();
    }
    return mb * mib;
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

const char * expert_prefetch_selection_policy() {
    static const char * policy = [] {
        const char * value = std::getenv(
                "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_SELECTION");
        return value && std::strcmp(value, "random") == 0 ? "random" : "router";
    }();
    return policy;
}

uint64_t expert_prefetch_random_seed() {
    static const uint64_t seed = [] {
        const char * value = std::getenv(
                "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_RANDOM_SEED");
        if (!value || !value[0]) {
            value = std::getenv("PREFETCH_RANDOM_SEED");
        }
        if (!value || !value[0]) {
            return uint64_t{1234};
        }
        char * end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        return end && *end == '\0' ? (uint64_t) parsed : uint64_t{1234};
    }();
    return seed;
}

uint64_t expert_prefetch_splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

std::vector<int> expert_prefetch_targets(
        int phase, uint64_t step, int layer, int token_idx,
        const std::vector<ExpertTensorInfo> & tensors,
        const int * router_experts, int router_count, int limit) {
    std::vector<int> targets;
    if (std::strcmp(expert_prefetch_selection_policy(), "random") != 0) {
        for (int i = 0; i < limit && i < router_count; ++i) {
            targets.push_back(router_experts[i]);
        }
        return targets;
    }

    int total_experts = 0;
    for (const ExpertTensorInfo & info : tensors) {
        total_experts = std::max(total_experts, (int) info.n_expert);
    }
    total_experts = std::max(total_experts, 256);
    limit = std::min(limit, total_experts);
    targets.resize((size_t) total_experts);
    std::iota(targets.begin(), targets.end(), 0);
    uint64_t state = expert_prefetch_random_seed();
    state ^= (uint64_t) (phase + 1) * 0x632be59bd9b4e019ull;
    state ^= (step + 1) * 0x8cb92baa5f1d6f47ull;
    state ^= (uint64_t) (layer + 1) * 0x4f1bbcdc676f3a21ull;
    state ^= (uint64_t) (token_idx + 1) * 0x94d049bb133111ebull;
    for (int i = 0; i < limit; ++i) {
        state = expert_prefetch_splitmix64(state);
        const int remaining = total_experts - i;
        const int selected = i + (int) (state % (uint64_t) remaining);
        std::swap(targets[(size_t) i], targets[(size_t) selected]);
    }
    targets.resize((size_t) limit);
    return targets;
}

const char * expert_prefetch_tensor_type(const char * name) {
    if (name && std::strstr(name, "ffn_down_exps.weight")) {
        return "Down";
    }
    if (name && (std::strstr(name, "ffn_gate_exps.weight") ||
            std::strstr(name, "ffn_up_exps.weight") ||
            std::strstr(name, "ffn_gate_up_exps.weight"))) {
        return "Gate/Up";
    }
    return "Other";
}

struct ExpertPrefetchSelectionStats {
    std::atomic<uint64_t> selection_events{0};
    std::atomic<uint64_t> selected_experts{0};
    std::atomic<uint64_t> target_events{0};
    std::atomic<uint64_t> requested_bytes{0};
    std::atomic<uint64_t> gate_up_bytes{0};
    std::atomic<uint64_t> down_bytes{0};
    std::atomic<uint64_t> eligible_targets{0};
    std::atomic<uint64_t> dedup_skipped{0};
    std::atomic<uint64_t> task_created{0};
};

ExpertPrefetchSelectionStats & expert_prefetch_selection_stats() {
    static ExpertPrefetchSelectionStats stats;
    return stats;
}

void write_expert_prefetch_selection_summary() {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    const ExpertPrefetchSelectionStats & stats = expert_prefetch_selection_stats();
    std::string line;
    line.reserve(640);
    line += "{\"event\":\"EXPERT_PREFETCH_SELECTION_SUMMARY\"";
    line += ",\"selection_policy\":";
    json_escape_append(line, expert_prefetch_selection_policy());
    line += ",\"random_seed\":" + std::to_string(expert_prefetch_random_seed());
    line += ",\"selection_events\":" + std::to_string(
            stats.selection_events.load(std::memory_order_relaxed));
    line += ",\"selected_experts\":" + std::to_string(
            stats.selected_experts.load(std::memory_order_relaxed));
    line += ",\"target_events\":" + std::to_string(
            stats.target_events.load(std::memory_order_relaxed));
    line += ",\"requested_prefetch_bytes\":" + std::to_string(
            stats.requested_bytes.load(std::memory_order_relaxed));
    line += ",\"gate_up_bytes\":" + std::to_string(
            stats.gate_up_bytes.load(std::memory_order_relaxed));
    line += ",\"down_bytes\":" + std::to_string(
            stats.down_bytes.load(std::memory_order_relaxed));
    line += ",\"eligible_targets\":" + std::to_string(
            stats.eligible_targets.load(std::memory_order_relaxed));
    line += ",\"dedup_skipped\":" + std::to_string(
            stats.dedup_skipped.load(std::memory_order_relaxed));
    line += ",\"task_created\":" + std::to_string(
            stats.task_created.load(std::memory_order_relaxed));
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void ensure_expert_prefetch_selection_summary_registered() {
    (void) expert_prefetch_selection_stats();
    static const bool registered = [] {
        std::atexit(write_expert_prefetch_selection_summary);
        return true;
    }();
    (void) registered;
}

void write_expert_prefetch_selection_event(
        int phase, uint64_t step, int layer, int token_idx,
        const std::vector<int> & router_experts,
        const std::vector<int> & targets) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_EXPERT)) {
        return;
    }
    std::string line;
    line.reserve(256 + targets.size() * 8);
    line += "{\"event\":\"EXPERT_PREFETCH_SELECTION\",\"phase\":";
    json_escape_append(line, phase_name(phase));
    line += ",\"step\":" + std::to_string(step);
    line += ",\"layer\":" + std::to_string(layer);
    line += ",\"token_index\":" + std::to_string(token_idx);
    line += ",\"selection_policy\":";
    json_escape_append(line, expert_prefetch_selection_policy());
    line += ",\"selected_experts\":[";
    for (size_t i = 0; i < targets.size(); ++i) {
        if (i) line += ",";
        line += std::to_string(targets[i]);
    }
    line += "],\"router_experts\":[";
    for (size_t i = 0; i < router_experts.size(); ++i) {
        if (i) line += ",";
        line += std::to_string(router_experts[i]);
    }
    line += "]}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_EXPERT, line.c_str(), line.size());
}

void write_expert_prefetch_target_event(
        int phase, uint64_t step, int layer, int token_idx, int expert,
        const ExpertTensorInfo & info, size_t slice_bytes, bool actual_selected,
        bool eligible, bool dedup_skipped) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_EXPERT)) {
        return;
    }
    std::string line;
    line.reserve(360);
    line += "{\"event\":\"EXPERT_PREFETCH_TARGET\",\"phase\":";
    json_escape_append(line, phase_name(phase));
    line += ",\"step\":" + std::to_string(step);
    line += ",\"layer\":" + std::to_string(layer);
    line += ",\"token_index\":" + std::to_string(token_idx);
    line += ",\"expert_id\":" + std::to_string(expert);
    line += ",\"tensor\":";
    json_escape_append(line, info.name.c_str());
    line += ",\"tensor_type\":";
    json_escape_append(line, expert_prefetch_tensor_type(info.name.c_str()));
    line += ",\"bytes\":" + std::to_string(slice_bytes);
    line += ",\"selection_policy\":";
    json_escape_append(line, expert_prefetch_selection_policy());
    line += ",\"actual_router_selected\":" +
            std::string(actual_selected ? "true" : "false");
    line += ",\"eligible\":" + std::string(eligible ? "true" : "false");
    line += ",\"dedup_skipped\":" + std::string(dedup_skipped ? "true" : "false");
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_EXPERT, line.c_str(), line.size());
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

bool expert_prefetch_async_fallback_enabled() {
    static const bool enabled = env_bool_or_default("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK", true);
    return enabled;
}

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
        return ExpertAsyncPriorityMode::Score;
    }();
    return mode;
}

bool expert_feedback_enabled() {
    static const bool enabled = os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK");
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

enum class ExpertRuntimeRescueMode {
    ReclaimBackoff,
    GateRecovery,
};

const char * expert_runtime_rescue_mode_name(ExpertRuntimeRescueMode mode) {
    switch (mode) {
        case ExpertRuntimeRescueMode::ReclaimBackoff: return "reclaim_backoff";
        case ExpertRuntimeRescueMode::GateRecovery:   return "gate_recovery";
    }
    return "reclaim_backoff";
}

bool expert_runtime_rescue_requested() {
    static const bool requested = env_truthy(
            std::getenv("LLM_MEM_TRACE_OPT_EXPERT_RUNTIME_RESCUE"));
    return requested;
}

// Formal Phase 2E flag: feedback-driven COLD rate recovery ladder.
// Default OFF; when OFF the rescue controller behaves exactly as before.
bool expert_cold_rate_recovery_requested() {
    static const bool requested = env_truthy(
            std::getenv("LLM_MEM_TRACE_OPT_EXPERT_COLD_RATE_RECOVERY"));
    return requested;
}

ExpertRuntimeRescueMode expert_runtime_rescue_mode() {
    static const ExpertRuntimeRescueMode mode = [] {
        if (expert_cold_rate_recovery_requested()) {
            // The rate-recovery ladder always uses the gate-recovery rescue flow.
            return ExpertRuntimeRescueMode::GateRecovery;
        }
        const char * value = std::getenv("LLM_MEM_TRACE_OPT_EXPERT_RUNTIME_RESCUE_MODE");
        return value && std::strcmp(value, "gate_recovery") == 0 ?
                ExpertRuntimeRescueMode::GateRecovery :
                ExpertRuntimeRescueMode::ReclaimBackoff;
    }();
    return mode;
}

bool expert_runtime_rescue_enabled() {
    return (expert_runtime_rescue_requested() || expert_cold_rate_recovery_requested()) &&
            llm_mem_trace_enabled() &&
            llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY);
}

enum class ExpertRuntimeRescueState {
    Normal,
    ColdSuspended,
    GateRecovery,
    ReEntry,
    Probe25,
    Probe50,
    Probe100,
    Disabled,
};

const char * expert_runtime_rescue_state_name(ExpertRuntimeRescueState state) {
    switch (state) {
        case ExpertRuntimeRescueState::Normal:        return "normal";
        case ExpertRuntimeRescueState::ColdSuspended: return "cold_suspended";
        case ExpertRuntimeRescueState::GateRecovery:  return "gate_recovery";
        case ExpertRuntimeRescueState::ReEntry:       return "reentry";
        case ExpertRuntimeRescueState::Probe25:       return "probe_25";
        case ExpertRuntimeRescueState::Probe50:       return "probe_50";
        case ExpertRuntimeRescueState::Probe100:      return "probe_100";
        case ExpertRuntimeRescueState::Disabled:      return "disabled";
    }
    return "normal";
}

// Phase 2E ladder budgets (bytes per decode step), from the Phase 2D measured
// full-rate baseline B = 271222731 bytes/step.
uint64_t expert_probe_state_budget_bytes(ExpertRuntimeRescueState state) {
    switch (state) {
        case ExpertRuntimeRescueState::Probe25:  return 67805682;
        case ExpertRuntimeRescueState::Probe50:  return 135611365;
        case ExpertRuntimeRescueState::Probe100: return 271222731;
        default:                                 return 0;
    }
}

// Minimal self-contained readers for the benefit check (defined before the
// rescue controller; the heavier cgroup helpers live later in this file).
uint64_t rescue_read_cgroup_file(const char * name) {
    char cgroup_line[512];
    FILE * f = std::fopen("/proc/self/cgroup", "r");
    if (!f) {
        return 0;
    }
    std::string relative;
    if (std::fgets(cgroup_line, sizeof(cgroup_line), f)) {
        const char * marker = std::strstr(cgroup_line, "0::");
        if (marker) {
            relative = marker + 3;
            while (!relative.empty() &&
                    (relative.back() == '\n' || relative.back() == '\r')) {
                relative.pop_back();
            }
        }
    }
    std::fclose(f);
    if (relative.empty() || relative == "/") {
        return 0;
    }
    std::string path = "/sys/fs/cgroup/" +
            (relative.front() == '/' ? relative.substr(1) : relative) + "/" + name;
    FILE * vf = std::fopen(path.c_str(), "r");
    if (!vf) {
        return 0;
    }
    unsigned long long value = 0;
    const int rc = std::fscanf(vf, "%llu", &value);
    std::fclose(vf);
    return rc == 1 ? (uint64_t) value : 0;
}

uint64_t rescue_read_memory_current() {
    return rescue_read_cgroup_file("memory.current");
}

uint64_t rescue_read_memory_limit() {
    const uint64_t high = rescue_read_cgroup_file("memory.high");
    if (high > 0 && high != std::numeric_limits<uint64_t>::max()) {
        return high;
    }
    const uint64_t maximum = rescue_read_cgroup_file("memory.max");
    return maximum == std::numeric_limits<uint64_t>::max() ? 0 : maximum;
}

uint64_t rescue_read_rss_bytes() {
    FILE * f = std::fopen("/proc/self/statm", "r");
    if (!f) {
        return 0;
    }
    unsigned long long total_pages = 0, resident_pages = 0;
    const int rc = std::fscanf(f, "%llu %llu", &total_pages, &resident_pages);
    std::fclose(f);
    return rc == 2 ? (uint64_t) resident_pages * 4096ull : 0;
}

// Experiment-only (Phase 2D): single COLD re-entry attempt with a per-decode-step
// byte budget. rate percent 0 disables re-entry (suspend-only, existing behavior).
uint64_t expert_reentry_rate_percent() {
    static const uint64_t rate = env_u64_or_default(
            "LLM_MEM_TRACE_OPT_EXPERT_REENTRY_RATE_PERCENT", 0);
    return rate;
}

uint64_t expert_reentry_budget_base_bytes() {
    static const uint64_t base = env_u64_or_default(
            "LLM_MEM_TRACE_OPT_EXPERT_REENTRY_BUDGET_BYTES_PER_STEP", 0);
    return base;
}

uint64_t expert_reentry_step_budget_bytes() {
    const uint64_t base = expert_reentry_budget_base_bytes();
    const uint64_t rate = expert_reentry_rate_percent();
    if (base == 0 || rate == 0 || rate > 100) {
        return 0;
    }
    return base * rate / 100;
}

struct ExpertRuntimeRescueWindow {
    uint64_t steps = 0;
    uint64_t issued = 0;
    uint64_t major_faults = 0;
    uint64_t latency_ns = 0;

    void observe(uint64_t step_issued, uint64_t step_major_faults, uint64_t step_latency_ns) {
        steps++;
        issued += step_issued;
        major_faults += step_major_faults;
        latency_ns += step_latency_ns;
    }
};

struct ExpertRuntimeRescueCounters {
    uint64_t decode_steps_observed = 0;
    uint64_t early_issued_sum = 0;
    uint64_t early_major_fault_sum = 0;
    uint64_t runtime_rescue_triggered = 0;
    uint64_t runtime_rescue_trigger_step = 0;
    uint64_t runtime_rescue_trigger_decode_step = 0;
    uint64_t runtime_rescue_cold_suspended = 0;
    uint64_t runtime_rescue_gate_bypass_steps = 0;
    uint64_t runtime_rescue_prefetch_bypassed_tasks = 0;
    // Phase 2D re-entry experiment counters
    uint64_t reentry_attempted = 0;
    uint64_t reentry_start_decode_step = 0;
    uint64_t reentry_failed = 0;
    uint64_t reentry_failure_decode_step = 0;
    uint64_t reentry_step_budget_used_bytes = 0;
    // Phase 2E rate-recovery ladder counters
    uint64_t recovery_completed_step = 0;
    uint64_t probe_25_start_step = 0;
    uint64_t probe_50_start_step = 0;
    uint64_t probe_100_start_step = 0;
    uint64_t re_degradation_detected = 0;
    uint64_t re_degradation_step = 0;
    uint64_t cold_disabled_for_run = 0;
    int cold_disabled_reason = 0; // 0 none, 1 RE_DEGRADATION, 2 LOW_BENEFIT
    uint64_t probe_cold_issued_bytes = 0;
    uint64_t benefit_check_performed = 0;
    int64_t benefit_memory_delta = 0;
    int64_t benefit_rss_delta = 0;
    int64_t benefit_fault_delta = 0;
    ExpertRuntimeRescueWindow early_window;
    ExpertRuntimeRescueWindow post_trigger_5;
    ExpertRuntimeRescueWindow post_trigger_10;
    ExpertRuntimeRescueWindow post_trigger_rest;
};

uint64_t current_major_fault_count() {
#ifdef __linux__
    struct rusage usage = {};
    return getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_majflt > 0 ?
            (uint64_t) usage.ru_majflt : 0;
#else
    return 0;
#endif
}

struct ExpertRuntimeRescueController {
    mutable std::mutex mu;
    ExpertRuntimeRescueState state = ExpertRuntimeRescueState::Normal;
    ExpertRuntimeRescueCounters counters;
    std::unordered_map<uint64_t, uint64_t> issued_by_step;
    uint64_t gate_recovery_steps_remaining = 0;
    uint64_t previous_major_faults = 0;
    bool has_previous_major_faults = false;
    std::deque<uint64_t> recovery_issued;
    std::deque<uint64_t> recovery_faults;
    std::deque<uint64_t> redeg_issued;
    std::deque<uint64_t> redeg_faults;
    // Phase 2E ladder state
    uint64_t ladder_steps_in_probe = 0;
    uint64_t ladder_baseline_mem_current = 0;
    uint64_t ladder_baseline_mem_limit = 0;
    uint64_t ladder_baseline_rss = 0;
    double ladder_baseline_fault_avg3 = 0.0;
    std::deque<uint64_t> probe_issued;
    std::deque<uint64_t> probe_faults;

    static double avg_last_n(const std::deque<uint64_t> & values, size_t n) {
        if (values.size() < n) {
            return 0.0;
        }
        double sum = 0.0;
        for (size_t i = values.size() - n; i < values.size(); ++i) {
            sum += (double) values[i];
        }
        return sum / (double) n;
    }

    void record_prefetch_issued(int phase, uint64_t step) {
        if (phase != LLM_MEM_TRACE_PHASE_DECODE) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu);
        issued_by_step[step]++;
    }

    bool value_gate_bypass_active(int phase) const {
        if (phase != LLM_MEM_TRACE_PHASE_DECODE) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mu);
        return state == ExpertRuntimeRescueState::GateRecovery;
    }

    void record_value_gate_bypass() {
        std::lock_guard<std::mutex> lock(mu);
        if (state == ExpertRuntimeRescueState::GateRecovery) {
            counters.runtime_rescue_prefetch_bypassed_tasks++;
        }
    }

    bool cold_suspended() const {
        std::lock_guard<std::mutex> lock(mu);
        return state != ExpertRuntimeRescueState::Normal;
    }

    // Phase 2E-A observation-only: returns true when the current rescue state
    // is considered safe for calibration healthy-sample admission. Only Normal
    // and ReEntry are allowed; GateRecovery, ColdSuspended, Probe* and Disabled
    // are excluded (spec §6). This is a const query with zero control impact.
    bool state_allows_calibration() const {
        std::lock_guard<std::mutex> lock(mu);
        return state == ExpertRuntimeRescueState::Normal ||
               state == ExpertRuntimeRescueState::ReEntry;
    }

    // 0 = normal (unlimited), 1 = suspended (skip real MADV_COLD), 2 = probe/re-entry (byte budget).
    int cold_issue_mode() const {
        std::lock_guard<std::mutex> lock(mu);
        switch (state) {
            case ExpertRuntimeRescueState::Normal:   return 0;
            case ExpertRuntimeRescueState::ReEntry:
            case ExpertRuntimeRescueState::Probe25:
            case ExpertRuntimeRescueState::Probe50:
            case ExpertRuntimeRescueState::Probe100: return 2;
            default:                                 return 1;
        }
    }

    uint64_t reentry_budget_remaining() const {
        std::lock_guard<std::mutex> lock(mu);
        const uint64_t budget = expert_cold_rate_recovery_requested() ?
                expert_probe_state_budget_bytes(state) :
                expert_reentry_step_budget_bytes();
        return budget > counters.reentry_step_budget_used_bytes ?
                budget - counters.reentry_step_budget_used_bytes : 0;
    }

    void reentry_budget_commit(uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mu);
        counters.reentry_step_budget_used_bytes += bytes;
        if (state == ExpertRuntimeRescueState::Probe25 ||
                state == ExpertRuntimeRescueState::Probe50 ||
                state == ExpertRuntimeRescueState::Probe100) {
            counters.probe_cold_issued_bytes += bytes;
        }
    }

    void on_step_end(int phase, uint64_t step, uint64_t latency_ns) {
        const uint64_t current_major_faults = current_major_fault_count();
        std::lock_guard<std::mutex> lock(mu);
        const uint64_t major_fault_delta = has_previous_major_faults &&
                current_major_faults >= previous_major_faults ?
                current_major_faults - previous_major_faults : 0;
        previous_major_faults = current_major_faults;
        has_previous_major_faults = true;
        if (phase != LLM_MEM_TRACE_PHASE_DECODE) {
            return;
        }

        const uint64_t issued = issued_by_step[step];
        counters.decode_steps_observed++;
        const uint64_t decode_step = counters.decode_steps_observed;
        const ExpertRuntimeRescueState state_for_step = state;

        if (counters.runtime_rescue_triggered == 0 && decode_step <= 3) {
            counters.early_window.observe(issued, major_fault_delta, latency_ns);
            counters.early_issued_sum += issued;
            counters.early_major_fault_sum += major_fault_delta;
            if (decode_step == 3 && counters.early_issued_sum < 300 &&
                    counters.early_major_fault_sum > 6000) {
                counters.runtime_rescue_triggered = 1;
                counters.runtime_rescue_trigger_step = step;
                counters.runtime_rescue_trigger_decode_step = decode_step;
                counters.runtime_rescue_cold_suspended = 1;
                if (expert_runtime_rescue_mode() == ExpertRuntimeRescueMode::GateRecovery) {
                    state = ExpertRuntimeRescueState::GateRecovery;
                    gate_recovery_steps_remaining = 5;
                } else {
                    state = ExpertRuntimeRescueState::ColdSuspended;
                }
            }
        } else if (counters.runtime_rescue_triggered != 0 &&
                decode_step > counters.runtime_rescue_trigger_decode_step) {
            const uint64_t relative_step = decode_step - counters.runtime_rescue_trigger_decode_step;
            if (relative_step <= 5) {
                counters.post_trigger_5.observe(issued, major_fault_delta, latency_ns);
            }
            if (relative_step <= 10) {
                counters.post_trigger_10.observe(issued, major_fault_delta, latency_ns);
            }
            if (relative_step > 10) {
                counters.post_trigger_rest.observe(issued, major_fault_delta, latency_ns);
            }
            if (state_for_step == ExpertRuntimeRescueState::GateRecovery &&
                    gate_recovery_steps_remaining > 0) {
                counters.runtime_rescue_gate_bypass_steps++;
                gate_recovery_steps_remaining--;
                if (gate_recovery_steps_remaining == 0) {
                    state = ExpertRuntimeRescueState::ColdSuspended;
                }
            }
            // Phase 2E: formal COLD rate-recovery ladder. Takes precedence over
            // the Phase 2D single-shot re-entry experiment when enabled.
            if (expert_cold_rate_recovery_requested()) {
                if (state == ExpertRuntimeRescueState::ColdSuspended) {
                    recovery_issued.push_back(issued);
                    recovery_faults.push_back(major_fault_delta);
                    if (recovery_issued.size() > 5) {
                        recovery_issued.pop_front();
                        recovery_faults.pop_front();
                    }
                    if (recovery_issued.size() == 5 &&
                            avg_last_n(recovery_issued, 5) >= 500.0 &&
                            avg_last_n(recovery_faults, 5) <= 2500.0) {
                        state = ExpertRuntimeRescueState::Probe25;
                        counters.recovery_completed_step = decode_step;
                        counters.probe_25_start_step = decode_step;
                        ladder_steps_in_probe = 0;
                        ladder_baseline_mem_current = rescue_read_memory_current();
                        ladder_baseline_mem_limit = rescue_read_memory_limit();
                        ladder_baseline_rss = rescue_read_rss_bytes();
                        ladder_baseline_fault_avg3 = avg_last_n(recovery_faults, 3);
                        probe_issued.clear();
                        probe_faults.clear();
                    }
                } else if (state == ExpertRuntimeRescueState::Probe25 ||
                        state == ExpertRuntimeRescueState::Probe50 ||
                        state == ExpertRuntimeRescueState::Probe100) {
                    ladder_steps_in_probe++;
                    probe_issued.push_back(issued);
                    probe_faults.push_back(major_fault_delta);
                    if (probe_issued.size() > 5) {
                        probe_issued.pop_front();
                        probe_faults.pop_front();
                    }
                    const double avg3_issued = avg_last_n(probe_issued, 3);
                    const double avg3_faults = avg_last_n(probe_faults, 3);
                    // Benefit check: once, after >=10 probe steps and >=1 GiB of COLD.
                    if (counters.benefit_check_performed == 0 &&
                            ladder_steps_in_probe >= 10 &&
                            counters.probe_cold_issued_bytes >= (1ull << 30)) {
                        counters.benefit_check_performed = 1;
                        const uint64_t now_mem = rescue_read_memory_current();
                        const uint64_t now_rss = rescue_read_rss_bytes();
                        counters.benefit_memory_delta = (int64_t) ladder_baseline_mem_current -
                                (int64_t) now_mem;
                        counters.benefit_rss_delta = (int64_t) ladder_baseline_rss -
                                (int64_t) now_rss;
                        counters.benefit_fault_delta = (int64_t) llround(
                                ladder_baseline_fault_avg3 - avg3_faults);
                        const uint64_t mem_ref = ladder_baseline_mem_limit > 0 ?
                                ladder_baseline_mem_limit : ladder_baseline_mem_current;
                        const bool mem_not_improved = mem_ref == 0 ||
                                counters.benefit_memory_delta < (int64_t) (mem_ref / 100);
                        const bool rss_not_improved = ladder_baseline_rss == 0 ||
                                counters.benefit_rss_delta < (int64_t) (ladder_baseline_rss / 100);
                        const bool fault_not_improved = ladder_baseline_fault_avg3 <= 0.0 ||
                                (ladder_baseline_fault_avg3 - avg3_faults) <
                                        0.10 * ladder_baseline_fault_avg3;
                        if (mem_not_improved && rss_not_improved && fault_not_improved) {
                            state = ExpertRuntimeRescueState::Disabled;
                            counters.cold_disabled_for_run = 1;
                            counters.cold_disabled_reason = 2;
                        }
                    }
                    // Re-degradation: rolling 3-step collapse.
                    if (state != ExpertRuntimeRescueState::Disabled &&
                            probe_issued.size() >= 3 &&
                            avg3_issued < 100.0 && avg3_faults > 2000.0) {
                        state = ExpertRuntimeRescueState::Disabled;
                        counters.re_degradation_detected = 1;
                        counters.re_degradation_step = decode_step;
                        counters.cold_disabled_for_run = 1;
                        counters.cold_disabled_reason = 1;
                    }
                    // Promotion: 5 consecutive stable steps.
                    if (state == ExpertRuntimeRescueState::Probe25 ||
                            state == ExpertRuntimeRescueState::Probe50) {
                        if (probe_issued.size() == 5 &&
                                avg_last_n(probe_issued, 5) >= 500.0 &&
                                avg_last_n(probe_faults, 5) <= 2500.0) {
                            if (state == ExpertRuntimeRescueState::Probe25) {
                                state = ExpertRuntimeRescueState::Probe50;
                                counters.probe_50_start_step = decode_step;
                            } else {
                                state = ExpertRuntimeRescueState::Probe100;
                                counters.probe_100_start_step = decode_step;
                            }
                            probe_issued.clear();
                            probe_faults.clear();
                        }
                    }
                }
            } else if (state == ExpertRuntimeRescueState::ColdSuspended &&
                    counters.reentry_attempted == 0 &&
                    expert_reentry_step_budget_bytes() > 0) {
                recovery_issued.push_back(issued);
                recovery_faults.push_back(major_fault_delta);
                if (recovery_issued.size() > 5) {
                    recovery_issued.pop_front();
                    recovery_faults.pop_front();
                }
                if (recovery_issued.size() == 5) {
                    const uint64_t sum_issued = std::accumulate(
                            recovery_issued.begin(), recovery_issued.end(), uint64_t{0});
                    const uint64_t sum_faults = std::accumulate(
                            recovery_faults.begin(), recovery_faults.end(), uint64_t{0});
                    if (sum_issued >= 5 * 500 && sum_faults <= 5 * 2500) {
                        state = ExpertRuntimeRescueState::ReEntry;
                        counters.reentry_attempted = 1;
                        counters.reentry_start_decode_step = decode_step;
                    }
                }
            }
            if (state == ExpertRuntimeRescueState::ReEntry) {
                redeg_issued.push_back(issued);
                redeg_faults.push_back(major_fault_delta);
                if (redeg_issued.size() > 3) {
                    redeg_issued.pop_front();
                    redeg_faults.pop_front();
                }
                if (redeg_issued.size() == 3) {
                    const uint64_t sum_issued = std::accumulate(
                            redeg_issued.begin(), redeg_issued.end(), uint64_t{0});
                    const uint64_t sum_faults = std::accumulate(
                            redeg_faults.begin(), redeg_faults.end(), uint64_t{0});
                    if (sum_issued < 3 * 100 && sum_faults > 3 * 2000) {
                        state = ExpertRuntimeRescueState::ColdSuspended;
                        counters.reentry_failed = 1;
                        counters.reentry_failure_decode_step = decode_step;
                    }
                }
            }
        }
        counters.reentry_step_budget_used_bytes = 0;

        if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
            return;
        }
        std::string line;
        line.reserve(288);
        line += "{\"event\":\"EXPERT_RUNTIME_RESCUE_STEP\",\"ts_ns\":" +
                std::to_string(llm_mem_trace_time_ns());
        line += ",\"step\":" + std::to_string(step);
        line += ",\"decode_step\":" + std::to_string(decode_step);
        line += ",\"issued\":" + std::to_string(issued);
        line += ",\"major_fault_delta\":" + std::to_string(major_fault_delta);
        line += ",\"latency_ns\":" + std::to_string(latency_ns);
        line += ",\"state\":";
        json_escape_append(line, expert_runtime_rescue_state_name(state_for_step));
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }

    ExpertRuntimeRescueCounters snapshot() const {
        std::lock_guard<std::mutex> lock(mu);
        return counters;
    }
};

ExpertRuntimeRescueController & expert_runtime_rescue_controller() {
    static ExpertRuntimeRescueController controller;
    return controller;
}

bool expert_runtime_rescue_cold_suspended() {
    return expert_runtime_rescue_enabled() && expert_runtime_rescue_controller().cold_suspended();
}

int expert_runtime_rescue_cold_issue_mode() {
    if (!expert_runtime_rescue_enabled()) {
        return 0;
    }
    return expert_runtime_rescue_controller().cold_issue_mode();
}

bool expert_runtime_rescue_value_gate_bypass_active(int phase) {
    return expert_runtime_rescue_enabled() &&
            expert_runtime_rescue_controller().value_gate_bypass_active(phase);
}

// ------------------------------------------------------------------
// Phase 2E-B: Calibrated Adaptive Control.
// Default OFF (LLM_MEM_TRACE_OPT_EXPERT_CALIBRATED_CONTROL). When ON:
// BOOTSTRAP (no real MADV_COLD, shadow calibration) -> frozen baseline ->
// PROBE_25 -> PROBE_50; relative-scale DEGRADING detection -> RECOVERY
// (gate bypass) -> PROBE_25 once; any second degradation or LOW_BENEFIT
// -> DISABLED for the rest of the inference.
// ------------------------------------------------------------------

bool expert_calibrated_control_requested() {
    static const bool requested = env_truthy(
            std::getenv("LLM_MEM_TRACE_OPT_EXPERT_CALIBRATED_CONTROL"));
    return requested;
}

bool expert_calibrated_control_enabled() {
    return expert_calibrated_control_requested() && llm_mem_trace_enabled() &&
            llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY);
}

enum class CalibratedControlState {
    Bootstrap,
    Probe25,
    Probe50,
    Recovery,
    Disabled,
};

const char * calibrated_control_state_name(CalibratedControlState state) {
    switch (state) {
        case CalibratedControlState::Bootstrap: return "bootstrap";
        case CalibratedControlState::Probe25:   return "probe_25";
        case CalibratedControlState::Probe50:   return "probe_50";
        case CalibratedControlState::Recovery:  return "recovery";
        case CalibratedControlState::Disabled:  return "disabled";
    }
    return "bootstrap";
}

struct ExpertCalibratedControlCounters {
    uint64_t calibration_valid = 0;
    uint64_t calibration_valid_step = 0;
    double frozen_baseline_issue_ratio = 0.0;
    double frozen_baseline_faults = 0.0;
    double frozen_baseline_cold_eligible_bytes = 0.0;
    uint64_t probe25_budget_bytes = 0;
    uint64_t probe50_budget_bytes = 0;
    uint64_t disaster_bypass_count = 0;
    uint64_t recovery_triggered = 0;
    uint64_t recovery_trigger_step = 0;
    uint64_t recovery_completed = 0;
    uint64_t recovery_completed_step = 0;
    uint64_t re_degradation = 0;
    uint64_t re_degradation_step = 0;
    uint64_t cold_disabled_for_run = 0;
    int disabled_reason = 0; // 0 none, 1 RE_DEGRADATION, 2 LOW_BENEFIT
    double degradation_prefetch_health = 0.0;
    double degradation_fault_amp = 0.0;
    uint64_t probe_cold_issued_bytes = 0;
    uint64_t benefit_required_bytes = 0;
    uint64_t benefit_check_performed = 0;
    int64_t benefit_memory_delta = 0;
    int64_t benefit_rss_delta = 0;
    int64_t benefit_fault_delta = 0;
};

struct ExpertCalibratedController {
    mutable std::mutex mu;
    CalibratedControlState state = CalibratedControlState::Bootstrap;
    ExpertCalibratedControlCounters counters;

    // Per-step accumulators (DECODE only).
    uint64_t step_issued = 0;
    uint64_t step_opportunities = 0;

    // Major fault delta tracking.
    uint64_t previous_major_faults = 0;
    bool has_previous_major_faults = false;

    // Model reload detection.
    uint64_t prev_step = 0;
    bool has_prev_step = false;

    // Frozen baseline (taken once when calibration first becomes valid).
    bool frozen = false;

    // Rolling windows of valid (opportunities>0) decode steps.
    std::deque<double> health_window;   // prefetch_health per valid step
    std::deque<double> amp_window;      // fault_amp per valid step

    uint64_t bypass_steps_remaining = 0;
    bool recovery_used = false;
    uint64_t steps_in_probe = 0;
    uint64_t probe_step_budget_used = 0;

    // Benefit baselines recorded at PROBE_25 entry.
    uint64_t benefit_base_mem_current = 0;
    uint64_t benefit_base_mem_limit = 0;
    uint64_t benefit_base_rss = 0;
    double benefit_base_fault_avg3 = 0.0;

    void record_prefetch_issued(int phase) {
        if (phase != LLM_MEM_TRACE_PHASE_DECODE) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu);
        step_issued++;
    }

    void record_prefetch_opportunity(int phase) {
        if (phase != LLM_MEM_TRACE_PHASE_DECODE) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu);
        step_opportunities++;
    }

    bool state_allows_calibration() const {
        std::lock_guard<std::mutex> lock(mu);
        return state == CalibratedControlState::Bootstrap &&
                bypass_steps_remaining == 0;
    }

    bool value_gate_bypass_active(int phase) const {
        if (phase != LLM_MEM_TRACE_PHASE_DECODE) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mu);
        return bypass_steps_remaining > 0;
    }

    // 2 = budgeted real COLD, 3 = scan eligible only (no syscall, no marking).
    int cold_issue_mode() const {
        std::lock_guard<std::mutex> lock(mu);
        switch (state) {
            case CalibratedControlState::Probe25:
            case CalibratedControlState::Probe50: return 2;
            default:                              return 3;
        }
    }

    uint64_t state_budget_bytes_unlocked() const {
        const double base = counters.frozen_baseline_cold_eligible_bytes;
        if (!frozen || base <= 0.0) {
            return 0;
        }
        const double fraction = state == CalibratedControlState::Probe25 ? 0.25 :
                state == CalibratedControlState::Probe50 ? 0.50 : 0.0;
        const uint64_t raw = (uint64_t) (base * fraction);
        return raw & ~uint64_t{4095}; // page-align down; may be 0 -> no COLD this step
    }

    uint64_t cold_budget_remaining() {
        std::lock_guard<std::mutex> lock(mu);
        const uint64_t budget = state_budget_bytes_unlocked();
        return budget > probe_step_budget_used ? budget - probe_step_budget_used : 0;
    }

    void cold_budget_commit(uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mu);
        probe_step_budget_used += bytes;
        counters.probe_cold_issued_bytes += bytes;
    }

    void freeze_baseline_unlocked(uint64_t step) {
        const CalibrationBaseline ratio = expert_calibration_profile().baseline_prefetch_issue_ratio();
        const CalibrationBaseline faults = expert_calibration_profile().baseline_major_faults();
        const CalibrationBaseline cold = expert_calibration_profile().baseline_cold_eligible_bytes();
        if (!ratio.valid || !faults.valid || !cold.valid ||
                !(ratio.median > 0.0) || !(faults.median > 0.0) || !(cold.median > 0.0) ||
                !std::isfinite(ratio.median) || !std::isfinite(faults.median) ||
                !std::isfinite(cold.median)) {
            return; // conservative: stay in BOOTSTRAP, never enable real COLD
        }
        frozen = true;
        counters.calibration_valid = 1;
        counters.calibration_valid_step = step;
        counters.frozen_baseline_issue_ratio = ratio.median;
        counters.frozen_baseline_faults = faults.median;
        counters.frozen_baseline_cold_eligible_bytes = cold.median;
        counters.probe25_budget_bytes = ((uint64_t) (cold.median * 0.25)) & ~uint64_t{4095};
        counters.probe50_budget_bytes = ((uint64_t) (cold.median * 0.50)) & ~uint64_t{4095};
        counters.benefit_required_bytes = (uint64_t) (cold.median * 10.0);
    }

    void enter_probe25_unlocked(uint64_t decode_step) {
        state = CalibratedControlState::Probe25;
        steps_in_probe = 0;
        probe_step_budget_used = 0;
        health_window.clear();
        amp_window.clear();
        benefit_base_mem_current = rescue_read_memory_current();
        benefit_base_mem_limit = rescue_read_memory_limit();
        benefit_base_rss = rescue_read_rss_bytes();
        // fault baseline for the benefit check: rolling last-3 will be captured
        // from the recovery/healthy window at entry; store 0 until 3 valid steps.
        benefit_base_fault_avg3 = -1.0;
        (void) decode_step;
    }

    void reset_unlocked() {
        state = CalibratedControlState::Bootstrap;
        frozen = false;
        counters = ExpertCalibratedControlCounters{};
        step_issued = 0;
        step_opportunities = 0;
        health_window.clear();
        amp_window.clear();
        bypass_steps_remaining = 0;
        recovery_used = false;
        steps_in_probe = 0;
        probe_step_budget_used = 0;
    }

    void on_step_end(int phase, uint64_t step, uint64_t latency_ns) {
        (void) latency_ns;
        const uint64_t current_major_faults = current_major_fault_count();
        std::lock_guard<std::mutex> lock(mu);
        const uint64_t major_fault_delta = has_previous_major_faults &&
                current_major_faults >= previous_major_faults ?
                current_major_faults - previous_major_faults : 0;
        previous_major_faults = current_major_faults;
        has_previous_major_faults = true;

        if (has_prev_step && step < prev_step) {
            reset_unlocked();
        }
        prev_step = step;
        has_prev_step = true;

        if (phase != LLM_MEM_TRACE_PHASE_DECODE) {
            step_issued = 0;
            step_opportunities = 0;
            return;
        }

        const uint64_t issued = step_issued;
        const uint64_t opportunities = step_opportunities;
        step_issued = 0;
        step_opportunities = 0;
        probe_step_budget_used = 0;

        // Freeze the baseline the first time calibration becomes valid.
        if (!frozen &&
                expert_calibration_profile().state() == CalibrationState::Calibrated) {
            freeze_baseline_unlocked(step);
            if (frozen && state == CalibratedControlState::Bootstrap) {
                enter_probe25_unlocked(0);
            }
        }

        const double health = (frozen && opportunities > 0) ?
                ((double) issued / (double) opportunities) /
                        counters.frozen_baseline_issue_ratio : 0.0;
        const double amp = frozen ?
                (double) major_fault_delta /
                        std::max(counters.frozen_baseline_faults, 1.0) : 0.0;

        if (bypass_steps_remaining > 0) {
            bypass_steps_remaining--;
        }

        switch (state) {
            case CalibratedControlState::Bootstrap: {
                // Scale-independent disaster guard: prefetch fully collapsed.
                if (opportunities > 0) {
                    health_window.push_back(issued == 0 ? 0.0 : 1.0);
                    if (health_window.size() > 3) {
                        health_window.pop_front();
                    }
                    if (health_window.size() == 3 &&
                            std::accumulate(health_window.begin(), health_window.end(), 0.0) == 0.0) {
                        bypass_steps_remaining = 5;
                        counters.disaster_bypass_count++;
                        health_window.clear();
                    }
                }
                break;
            }
            case CalibratedControlState::Probe25:
            case CalibratedControlState::Probe50: {
                if (opportunities == 0) {
                    break;
                }
                steps_in_probe++;
                health_window.push_back(health);
                amp_window.push_back(amp);
                if (health_window.size() > 5) {
                    health_window.pop_front();
                    amp_window.pop_front();
                }
                const double avg3_health = health_window.size() >= 3 ?
                        std::accumulate(health_window.end() - 3, health_window.end(), 0.0) / 3.0 : 1.0;
                const double avg3_amp = amp_window.size() >= 3 ?
                        std::accumulate(amp_window.end() - 3, amp_window.end(), 0.0) / 3.0 : 0.0;
                if (benefit_base_fault_avg3 < 0.0 && amp_window.size() >= 3) {
                    // capture entry fault scale from the first 3 probe steps
                    benefit_base_fault_avg3 =
                            std::accumulate(amp_window.end() - 3, amp_window.end(), 0.0) / 3.0 *
                            counters.frozen_baseline_faults;
                }
                // Benefit check (once): >=10 probe steps AND >=10*B cold bytes.
                if (counters.benefit_check_performed == 0 && steps_in_probe >= 10 &&
                        counters.probe_cold_issued_bytes >= counters.benefit_required_bytes) {
                    counters.benefit_check_performed = 1;
                    const uint64_t now_mem = rescue_read_memory_current();
                    const uint64_t now_rss = rescue_read_rss_bytes();
                    counters.benefit_memory_delta = (int64_t) benefit_base_mem_current - (int64_t) now_mem;
                    counters.benefit_rss_delta = (int64_t) benefit_base_rss - (int64_t) now_rss;
                    const double now_fault_rate = amp_window.size() >= 3 ?
                            std::accumulate(amp_window.end() - 3, amp_window.end(), 0.0) / 3.0 *
                            counters.frozen_baseline_faults : (double) major_fault_delta;
                    counters.benefit_fault_delta = (int64_t) llround(
                            benefit_base_fault_avg3 - now_fault_rate);
                    const uint64_t mem_ref = benefit_base_mem_limit > 0 ?
                            benefit_base_mem_limit : benefit_base_mem_current;
                    const bool mem_not_improved = mem_ref == 0 ||
                            counters.benefit_memory_delta < (int64_t) (mem_ref / 100);
                    const bool rss_not_improved = benefit_base_rss == 0 ||
                            counters.benefit_rss_delta < (int64_t) (benefit_base_rss / 100);
                    const bool fault_not_improved = benefit_base_fault_avg3 <= 0.0 ||
                            (benefit_base_fault_avg3 - now_fault_rate) <
                                    0.10 * benefit_base_fault_avg3;
                    if (mem_not_improved && rss_not_improved && fault_not_improved) {
                        state = CalibratedControlState::Disabled;
                        counters.cold_disabled_for_run = 1;
                        counters.disabled_reason = 2;
                        break;
                    }
                }
                // DEGRADING: rolling 3 valid steps.
                if (health_window.size() >= 3 && avg3_health < 0.10 && avg3_amp > 8.0) {
                    counters.degradation_prefetch_health = avg3_health;
                    counters.degradation_fault_amp = avg3_amp;
                    if (!recovery_used) {
                        recovery_used = true;
                        counters.recovery_triggered = 1;
                        counters.recovery_trigger_step = step;
                        state = CalibratedControlState::Recovery;
                        bypass_steps_remaining = 5;
                    } else {
                        state = CalibratedControlState::Disabled;
                        counters.re_degradation = 1;
                        counters.re_degradation_step = step;
                        counters.cold_disabled_for_run = 1;
                        counters.disabled_reason = 1;
                    }
                    health_window.clear();
                    amp_window.clear();
                    break;
                }
                // Promotion: 5 consecutive stable steps.
                if (state == CalibratedControlState::Probe25 && health_window.size() == 5) {
                    const double avg5_health = std::accumulate(
                            health_window.begin(), health_window.end(), 0.0) / 5.0;
                    const double avg5_amp = std::accumulate(
                            amp_window.begin(), amp_window.end(), 0.0) / 5.0;
                    if (avg5_health >= 0.50 && avg5_amp <= 10.0) {
                        state = CalibratedControlState::Probe50;
                        health_window.clear();
                        amp_window.clear();
                    }
                }
                break;
            }
            case CalibratedControlState::Recovery: {
                if (opportunities == 0) {
                    break;
                }
                health_window.push_back(health);
                amp_window.push_back(amp);
                if (health_window.size() > 5) {
                    health_window.pop_front();
                    amp_window.pop_front();
                }
                if (health_window.size() == 5) {
                    const double avg5_health = std::accumulate(
                            health_window.begin(), health_window.end(), 0.0) / 5.0;
                    const double avg5_amp = std::accumulate(
                            amp_window.begin(), amp_window.end(), 0.0) / 5.0;
                    if (avg5_health >= 0.50 && avg5_amp <= 10.0) {
                        counters.recovery_completed = 1;
                        counters.recovery_completed_step = step;
                        enter_probe25_unlocked(step);
                    }
                }
                break;
            }
            case CalibratedControlState::Disabled:
                break;
        }

        if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
            return;
        }
        std::string line;
        line.reserve(256);
        line += "{\"event\":\"EXPERT_CALIBRATED_STEP\",\"ts_ns\":" +
                std::to_string(llm_mem_trace_time_ns());
        line += ",\"step\":" + std::to_string(step);
        line += ",\"issued\":" + std::to_string(issued);
        line += ",\"opportunities\":" + std::to_string(opportunities);
        line += ",\"major_fault_delta\":" + std::to_string(major_fault_delta);
        line += ",\"prefetch_health\":" + std::to_string(health);
        line += ",\"fault_amp\":" + std::to_string(amp);
        line += ",\"state\":";
        json_escape_append(line, calibrated_control_state_name(state));
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }

    ExpertCalibratedControlCounters snapshot() const {
        std::lock_guard<std::mutex> lock(mu);
        return counters;
    }

    CalibratedControlState current_state() const {
        std::lock_guard<std::mutex> lock(mu);
        return state;
    }
};

ExpertCalibratedController & expert_calibrated_controller() {
    static ExpertCalibratedController controller;
    return controller;
}

void write_expert_calibrated_control_summary() {
    if (!expert_calibrated_control_enabled()) {
        return;
    }
    const ExpertCalibratedControlCounters counters = expert_calibrated_controller().snapshot();
    std::string line;
    line.reserve(900);
    line += "{\"event\":\"EXPERT_CALIBRATED_CONTROL_SUMMARY\",\"ts_ns\":" +
            std::to_string(llm_mem_trace_time_ns());
    line += ",\"calibrated_control_enabled\":true";
    line += ",\"controller_state\":";
    json_escape_append(line, calibrated_control_state_name(
            expert_calibrated_controller().current_state()));
    line += ",\"calibration_valid\":" + std::to_string(counters.calibration_valid);
    line += ",\"calibration_valid_step\":" + std::to_string(counters.calibration_valid_step);
    line += ",\"frozen_baseline_issue_ratio\":" +
            std::to_string(counters.frozen_baseline_issue_ratio);
    line += ",\"frozen_baseline_faults\":" +
            std::to_string(counters.frozen_baseline_faults);
    line += ",\"frozen_baseline_cold_eligible_bytes\":" +
            std::to_string(counters.frozen_baseline_cold_eligible_bytes);
    line += ",\"probe25_budget_bytes\":" + std::to_string(counters.probe25_budget_bytes);
    line += ",\"probe50_budget_bytes\":" + std::to_string(counters.probe50_budget_bytes);
    line += ",\"disaster_bypass_count\":" + std::to_string(counters.disaster_bypass_count);
    line += ",\"recovery_triggered\":" + std::to_string(counters.recovery_triggered);
    line += ",\"recovery_trigger_step\":" + std::to_string(counters.recovery_trigger_step);
    line += ",\"recovery_completed\":" + std::to_string(counters.recovery_completed);
    line += ",\"recovery_completed_step\":" + std::to_string(counters.recovery_completed_step);
    line += ",\"re_degradation\":" + std::to_string(counters.re_degradation);
    line += ",\"re_degradation_step\":" + std::to_string(counters.re_degradation_step);
    line += ",\"cold_disabled_for_run\":" + std::to_string(counters.cold_disabled_for_run);
    line += ",\"disabled_reason\":";
    json_escape_append(line, counters.disabled_reason == 1 ? "RE_DEGRADATION" :
            counters.disabled_reason == 2 ? "LOW_BENEFIT" : "none");
    line += ",\"degradation_prefetch_health\":" +
            std::to_string(counters.degradation_prefetch_health);
    line += ",\"degradation_fault_amp\":" +
            std::to_string(counters.degradation_fault_amp);
    line += ",\"probe_cold_issued_bytes\":" +
            std::to_string(counters.probe_cold_issued_bytes);
    line += ",\"benefit_required_bytes\":" + std::to_string(counters.benefit_required_bytes);
    line += ",\"benefit_check_performed\":" + std::to_string(counters.benefit_check_performed);
    line += ",\"benefit_memory_delta\":" + std::to_string(counters.benefit_memory_delta);
    line += ",\"benefit_rss_delta\":" + std::to_string(counters.benefit_rss_delta);
    line += ",\"benefit_fault_delta\":" + std::to_string(counters.benefit_fault_delta);
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void ensure_expert_calibrated_control_summary_registered() {
    (void) expert_calibrated_controller();
    static const bool registered = [] {
        std::atexit(write_expert_calibrated_control_summary);
        return true;
    }();
    (void) registered;
}

// Combined value-gate bypass: runtime rescue (2C/2E) or calibrated control (2E-B).
bool expert_value_gate_bypass_active_any(int phase) {
    if (expert_calibrated_control_enabled() &&
            expert_calibrated_controller().value_gate_bypass_active(phase)) {
        return true;
    }
    return expert_runtime_rescue_value_gate_bypass_active(phase);
}

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
        const uint64_t base_budget = expert_prefetch_budget_bytes();
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
    int layer = -1;
    int expert = -1;
    uintptr_t addr = 0;
    size_t nbytes = 0;
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
    bool use_fadvise = false;
    bool memory_object_hint_slot_acquired = false;
    ExpertTaskLifecycleRecord lifecycle;
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
    std::atomic<uint64_t> cancelled_queue_full{0};
    std::atomic<uint64_t> issue_groups{0};
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
    line += ",\"cancelled_queue_full\":" + std::to_string(stats.cancelled_queue_full.load(std::memory_order_relaxed));
    line += ",\"issue_groups\":" + std::to_string(stats.issue_groups.load(std::memory_order_relaxed));
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

ExpertMemoryObjectTracker & expert_memory_object_tracker() {
    static ExpertMemoryObjectTracker tracker(expert_working_set_budget_bytes());
    return tracker;
}

void write_expert_memory_object_summary() {
    if (!expert_memory_objects_enabled() ||
            !llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    const ExpertMemoryObjectCounters counters = expert_memory_object_tracker().counters();
    std::string line;
    line.reserve(400);
    line += "{\"event\":\"EXPERT_MEMORY_OBJECT_SUMMARY\",\"ts_ns\":" +
            std::to_string(llm_mem_trace_time_ns());
    line += ",\"enabled\":true";
    line += ",\"inflight_hint_aggregation_enabled\":" + std::string(
            expert_inflight_hint_aggregation_enabled() ? "true" : "false");
    line += ",\"semantic_stale_cancel_enabled\":" + std::string(
            expert_semantic_stale_cancel_enabled() ? "true" : "false");
    line += ",\"working_set_enabled\":" + std::string(
            expert_working_set_requested() ? "true" : "false");
    line += ",\"madv_cold_reclaim_enabled\":" + std::string(
            expert_madv_cold_reclaim_enabled() ? "true" : "false");
    line += ",\"madv_cold_grace_steps\":" +
            std::to_string(expert_madv_cold_reclaim_grace_steps());
    line += ",\"memory_objects_created\":" +
            std::to_string(counters.memory_objects_created);
    line += ",\"semantic_demands_registered\":" +
            std::to_string(counters.semantic_demands_registered);
    line += ",\"semantic_demands_merged\":" +
            std::to_string(counters.semantic_demands_merged);
    line += ",\"demand_activations\":" +
            std::to_string(counters.demand_activations);
    line += ",\"demand_completions\":" +
            std::to_string(counters.demand_completions);
    line += ",\"stale_pending_canceled\":" +
            std::to_string(counters.stale_pending_canceled);
    line += ",\"unmatched_first_use\":" +
            std::to_string(counters.unmatched_first_use);
    line += ",\"invariant_violations\":" +
            std::to_string(counters.invariant_violations);
    line += ",\"pending\":" + std::to_string(counters.pending);
    line += ",\"active\":" + std::to_string(counters.active);
    line += ",\"pending_objects\":" + std::to_string(counters.pending_objects);
    line += ",\"active_objects\":" + std::to_string(counters.active_objects);
    line += ",\"peak_pending_objects\":" +
            std::to_string(counters.peak_pending_objects);
    line += ",\"peak_active_objects\":" +
            std::to_string(counters.peak_active_objects);
    line += ",\"hint_slots_acquired\":" +
            std::to_string(counters.hint_slots_acquired);
    line += ",\"inflight_hint_aggregated\":" +
            std::to_string(counters.inflight_hint_aggregated);
    line += ",\"hint_slots_released\":" +
            std::to_string(counters.hint_slots_released);
    line += ",\"hint_terminal_canceled\":" +
            std::to_string(counters.hint_terminal_canceled);
    line += ",\"current_hint_inflight_objects\":" +
            std::to_string(counters.current_hint_inflight_objects);
    line += ",\"peak_hint_inflight_objects\":" +
            std::to_string(counters.peak_hint_inflight_objects);
    line += ",\"semantic_stale_checked\":" +
            std::to_string(counters.semantic_stale_checked);
    line += ",\"semantic_stale_kept_live\":" +
            std::to_string(counters.semantic_stale_kept_live);
    line += ",\"semantic_stale_tasks_canceled\":" +
            std::to_string(counters.semantic_stale_tasks_canceled);
    line += ",\"semantic_stale_bytes_avoided\":" +
            std::to_string(counters.semantic_stale_bytes_avoided);
    line += ",\"working_set_budget_bytes\":" +
            std::to_string(counters.working_set_budget_bytes);
    line += ",\"working_set_current_bytes\":" +
            std::to_string(counters.working_set_current_bytes);
    line += ",\"working_set_peak_bytes\":" +
            std::to_string(counters.working_set_peak_bytes);
    line += ",\"working_set_objects\":" +
            std::to_string(counters.working_set_objects);
    line += ",\"working_set_peak_objects\":" +
            std::to_string(counters.working_set_peak_objects);
    line += ",\"working_set_admissions\":" +
            std::to_string(counters.working_set_admissions);
    line += ",\"working_set_readmissions\":" +
            std::to_string(counters.working_set_readmissions);
    line += ",\"working_set_evictions\":" +
            std::to_string(counters.working_set_evictions);
    line += ",\"working_set_evicted_bytes\":" +
            std::to_string(counters.working_set_evicted_bytes);
    line += ",\"working_set_protected_skips\":" +
            std::to_string(counters.working_set_protected_skips);
    line += ",\"budget_unresolved_due_to_protection\":" +
            std::to_string(counters.budget_unresolved_due_to_protection);
    line += ",\"working_set_lru_scans\":" +
            std::to_string(counters.working_set_lru_scans);
    line += ",\"readmission_gap_0\":" +
            std::to_string(counters.readmission_gap_0);
    line += ",\"readmission_gap_1\":" +
            std::to_string(counters.readmission_gap_1);
    line += ",\"readmission_gap_2_3\":" +
            std::to_string(counters.readmission_gap_2_3);
    line += ",\"readmission_gap_4_7\":" +
            std::to_string(counters.readmission_gap_4_7);
    line += ",\"readmission_gap_8_plus\":" +
            std::to_string(counters.readmission_gap_8_plus);
    line += ",\"readmission_gap_no_record\":" +
            std::to_string(counters.readmission_gap_no_record);
    line += ",\"readmissions_within_1_step\":" +
            std::to_string(counters.readmissions_within_1_step);
    line += ",\"readmissions_within_3_steps\":" +
            std::to_string(counters.readmissions_within_3_steps);
    line += ",\"probation_entries\":" + std::to_string(counters.probation_entries);
    line += ",\"probation_canceled_by_readmission\":" +
            std::to_string(counters.probation_canceled_by_readmission);
    line += ",\"madv_cold_candidates\":" + std::to_string(counters.madv_cold_candidates);
    line += ",\"madv_cold_issued\":" + std::to_string(counters.madv_cold_issued);
    line += ",\"madv_cold_failed\":" + std::to_string(counters.madv_cold_failed);
    line += ",\"madv_cold_bytes\":" + std::to_string(counters.madv_cold_bytes);
    line += ",\"post_cold_readmissions\":" +
            std::to_string(counters.post_cold_readmissions);
    line += ",\"current_probation_objects\":" +
            std::to_string(counters.current_probation_objects);
    line += ",\"peak_probation_objects\":" +
            std::to_string(counters.peak_probation_objects);
    line += ",\"cold_skipped_ttl_nonzero\":" +
            std::to_string(counters.cold_skipped_ttl_nonzero);
    line += ",\"cold_protected_violation\":" +
            std::to_string(counters.cold_protected_violation);
    line += ",\"madv_cold_budget_deferred_candidates\":" +
            std::to_string(counters.madv_cold_budget_deferred_candidates);
    line += ",\"madv_cold_budget_deferred_bytes\":" +
            std::to_string(counters.madv_cold_budget_deferred_bytes);
    line += ",\"cold_eligible_candidate_bytes\":" +
            std::to_string(counters.cold_eligible_candidate_bytes);
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void append_runtime_rescue_window(
        std::string & line,
        const char * name,
        const ExpertRuntimeRescueWindow & window) {
    line += ",\"" + std::string(name) + "\":{\"steps\":" +
            std::to_string(window.steps);
    line += ",\"issued\":" + std::to_string(window.issued);
    line += ",\"major_faults\":" + std::to_string(window.major_faults);
    line += ",\"latency_ns\":" + std::to_string(window.latency_ns);
    line += "}";
}

void write_expert_runtime_rescue_summary() {
    if (!expert_runtime_rescue_enabled() ||
            !llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    const ExpertRuntimeRescueCounters counters = expert_runtime_rescue_controller().snapshot();
    std::string line;
    line.reserve(800);
    line += "{\"event\":\"EXPERT_RUNTIME_RESCUE_SUMMARY\",\"ts_ns\":" +
            std::to_string(llm_mem_trace_time_ns());
    line += ",\"enabled\":true";
    line += ",\"mode\":";
    json_escape_append(line, expert_runtime_rescue_mode_name(expert_runtime_rescue_mode()));
    line += ",\"decode_steps_observed\":" +
            std::to_string(counters.decode_steps_observed);
    line += ",\"runtime_rescue_triggered\":" +
            std::to_string(counters.runtime_rescue_triggered);
    line += ",\"runtime_rescue_trigger_step\":" +
            std::to_string(counters.runtime_rescue_trigger_step);
    line += ",\"runtime_rescue_trigger_decode_step\":" +
            std::to_string(counters.runtime_rescue_trigger_decode_step);
    line += ",\"runtime_rescue_cold_suspended\":" +
            std::to_string(counters.runtime_rescue_cold_suspended);
    line += ",\"runtime_rescue_gate_bypass_steps\":" +
            std::to_string(counters.runtime_rescue_gate_bypass_steps);
    line += ",\"runtime_rescue_prefetch_bypassed_tasks\":" +
            std::to_string(counters.runtime_rescue_prefetch_bypassed_tasks);
    line += ",\"reentry_rate_percent\":" +
            std::to_string(expert_reentry_rate_percent());
    line += ",\"reentry_step_budget_bytes\":" +
            std::to_string(expert_reentry_step_budget_bytes());
    line += ",\"reentry_attempted\":" +
            std::to_string(counters.reentry_attempted);
    line += ",\"reentry_start_decode_step\":" +
            std::to_string(counters.reentry_start_decode_step);
    line += ",\"reentry_failed\":" +
            std::to_string(counters.reentry_failed);
    line += ",\"reentry_failure_decode_step\":" +
            std::to_string(counters.reentry_failure_decode_step);
    line += ",\"cold_rate_recovery_enabled\":" + std::string(
            expert_cold_rate_recovery_requested() ? "true" : "false");
    line += ",\"recovery_completed_step\":" +
            std::to_string(counters.recovery_completed_step);
    line += ",\"probe_25_start_step\":" +
            std::to_string(counters.probe_25_start_step);
    line += ",\"probe_50_start_step\":" +
            std::to_string(counters.probe_50_start_step);
    line += ",\"probe_100_start_step\":" +
            std::to_string(counters.probe_100_start_step);
    line += ",\"re_degradation_detected\":" +
            std::to_string(counters.re_degradation_detected);
    line += ",\"re_degradation_step\":" +
            std::to_string(counters.re_degradation_step);
    line += ",\"cold_disabled_for_run\":" +
            std::to_string(counters.cold_disabled_for_run);
    line += ",\"cold_disabled_reason\":";
    json_escape_append(line, counters.cold_disabled_reason == 1 ? "RE_DEGRADATION" :
            counters.cold_disabled_reason == 2 ? "LOW_BENEFIT" : "none");
    line += ",\"probe_cold_issued_bytes\":" +
            std::to_string(counters.probe_cold_issued_bytes);
    line += ",\"benefit_check_performed\":" +
            std::to_string(counters.benefit_check_performed);
    line += ",\"benefit_memory_delta\":" +
            std::to_string(counters.benefit_memory_delta);
    line += ",\"benefit_rss_delta\":" +
            std::to_string(counters.benefit_rss_delta);
    line += ",\"benefit_fault_delta\":" +
            std::to_string(counters.benefit_fault_delta);
    append_runtime_rescue_window(line, "early_3_steps", counters.early_window);
    append_runtime_rescue_window(line, "post_trigger_first_5", counters.post_trigger_5);
    append_runtime_rescue_window(line, "post_trigger_first_10", counters.post_trigger_10);
    append_runtime_rescue_window(line, "post_trigger_rest", counters.post_trigger_rest);
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void ensure_expert_runtime_rescue_summary_registered() {
    (void) expert_runtime_rescue_controller();
    static const bool registered = [] {
        std::atexit(write_expert_runtime_rescue_summary);
        return true;
    }();
    (void) registered;
}

void ensure_expert_memory_object_summary_registered() {
    (void) expert_memory_object_tracker();
    static const bool registered = [] {
        std::atexit(write_expert_memory_object_summary);
        return true;
    }();
    (void) registered;
}

void write_expert_calibration_shadow_summary() {
    if ((!expert_calibration_shadow_enabled() && !expert_calibrated_control_enabled()) ||
            !llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    CalibrationSummaryContext ctx;
    ctx.memory_current = rescue_read_memory_current();
    ctx.memory_limit = rescue_read_memory_limit();
    ctx.rss_bytes = rescue_read_rss_bytes();
    ctx.working_set_budget_bytes = expert_working_set_budget_bytes();
    expert_calibration_profile().write_summary(ctx);
}

void ensure_expert_calibration_shadow_summary_registered() {
    (void) expert_calibration_profile();
    static const bool registered = [] {
        std::atexit(write_expert_calibration_shadow_summary);
        return true;
    }();
    (void) registered;
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
    meta.decision = decision;
    meta.has_trace_context = true;
    meta.phase = task.phase;
    meta.step = task.step;
    meta.has_control = expert_feedback_enabled() || expert_value_gate_enabled();
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
    meta.issue_id = task.issue_id;
    meta.issue_task_count = task.issue_id != 0 ? 1 : 0;
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

void release_expert_task_hint_slot(
        const ExpertHintTask & task,
        bool terminal_canceled) {
    if (!task.memory_object_hint_slot_acquired) {
        return;
    }
    expert_memory_object_tracker().release_hint_slot(
            task.layer, task.expert, task.tensor_name, terminal_canceled);
}

uint64_t issue_expert_hint_task(ExpertHintTask & task) {
    if (expert_task_detail_events_enabled()) {
        task.issue_id = next_expert_issue_id();
    }
    if (expert_task_trace_mode() != ExpertTaskTraceMode::Off) {
        expert_task_lifecycle_stats().issue_groups.fetch_add(1, std::memory_order_relaxed);
    }
    const uint64_t begin = llm_mem_trace_time_ns();
    if (expert_task_trace_mode() != ExpertTaskTraceMode::Off) {
        task.lifecycle.issue_id = task.issue_id;
        task.lifecycle.issue_task_count = task.issue_id != 0 ? 1 : 0;
        task.lifecycle.issued_ts_ns = begin;
        register_expert_task_for_first_use(task.lifecycle);
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
    if (expert_runtime_rescue_enabled()) {
        expert_runtime_rescue_controller().record_prefetch_issued(task.phase, task.step);
    }
    // Phase 2E-A: count issued prefetch (observation-only, gated on calibration only).
    // Phase 2E-B: calibrated control also feeds the profile.
    if (expert_calibration_shadow_enabled() || expert_calibrated_control_enabled()) {
        expert_calibration_profile().record_prefetch_issued(task.phase);
    }
    if (expert_calibrated_control_enabled()) {
        expert_calibrated_controller().record_prefetch_issued(task.phase);
    }
    transition_expert_task(task.lifecycle, ExpertTaskEvent::Issue, nullptr, begin, end);
    release_expert_task_hint_slot(task, false);
    return duration;
}

struct ExpertHintQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<ExpertHintTask> tasks;
    std::vector<ExpertHintTask> priority_heap;
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
    uint64_t priority_pops = 0;
    uint64_t priority_heap_pops = 0;
    uint64_t fallback_tasks = 0;
    uint64_t queue_full_fallbacks = 0;
    uint64_t start_fail_fallbacks = 0;
    uint64_t max_queue_depth = 0;
    uint64_t queued_bytes = 0;
    uint64_t max_queued_bytes = 0;
    uint64_t cancelled_pressure = 0;
    uint64_t cancelled_value = 0;
    uint64_t cancelled_queue_full = 0;

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
                priority_heap.emplace_back(std::move(task));
                auto cmp = [this](const ExpertHintTask & a, const ExpertHintTask & b) {
                    return is_higher_priority(b, a);
                };
                std::push_heap(priority_heap.begin(), priority_heap.end(), cmp);
            } else {
                tasks.emplace_back(std::move(task));
            }
            enqueued_tasks++;
            queued_bytes += task_bytes;
            max_queued_bytes = std::max(max_queued_bytes, queued_bytes);
            max_queue_depth = std::max<uint64_t>(
                    max_queue_depth, (uint64_t) queue_depth_unlocked());
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
            queued_bytes = 0;
            worker_count = 0;
            priority_enabled = false;
            priority_heap_enabled = false;
            priority_mode = ExpertAsyncPriorityMode::Score;
        }
    }

    void run() {
        for (;;) {
            ExpertHintTask task;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] { return stopping || !queue_empty_unlocked(); });
                if (queue_empty_unlocked()) {
                    if (stopping) {
                        break;
                    }
                    continue;
                }
                task = pop_one_unlocked();
            }

            transition_expert_task(task.lifecycle, ExpertTaskEvent::Dequeue);
            if (expert_feedback_enabled()) {
                apply_pressure_snapshot(task, expert_pressure_controller().snapshot());
            }
            refresh_expert_task_estimate(task);
            if (expert_task_exceeds_pressure_budget(task, 0)) {
                transition_expert_task(
                        task.lifecycle, ExpertTaskEvent::Cancel, "pressure_changed");
                write_expert_task_skip(
                        task, "expert_prefetch_cancel_pressure", "pressure_changed");
                release_expert_task_hint_slot(task, true);
                std::lock_guard<std::mutex> lock(mu);
                cancelled_pressure++;
                continue;
            }
            if (expert_task_below_value_threshold(task)) {
                // R2 deliberately overrides only this value re-check.  Pressure,
                // queue, semantic-stale, lifecycle and hint-slot safeguards stay
                // on their normal paths.
                if (expert_value_gate_bypass_active_any(task.phase)) {
                    expert_runtime_rescue_controller().record_value_gate_bypass();
                } else {
                    transition_expert_task(
                            task.lifecycle, ExpertTaskEvent::Cancel, "value_changed");
                    write_expert_task_skip(
                            task, "expert_prefetch_cancel_value", "value_changed");
                    release_expert_task_hint_slot(task, true);
                    std::lock_guard<std::mutex> lock(mu);
                    cancelled_value++;
                    continue;
                }
            }

            if (expert_semantic_stale_cancel_enabled() &&
                    expert_route_hint_ttl_steps_for_phase(task.phase) == 0) {
                const bool live = expert_memory_object_tracker().has_live_demand(
                        task.layer, task.expert, task.tensor_name);
                expert_memory_object_tracker().record_semantic_stale_check(live);
                if (!live) {
                    transition_expert_task(
                            task.lifecycle, ExpertTaskEvent::Cancel, "semantic_stale");
                    write_expert_task_skip(
                            task, "expert_prefetch_cancel_semantic_stale", "semantic_stale");
                    expert_memory_object_tracker().record_semantic_stale_cancel(task.nbytes);
                    release_expert_task_hint_slot(task, true);
                    continue;
                }
            }

            task.predicted_service_ns =
                    expert_timing_model().predicted_transfer_ns(task.nbytes) +
                    expert_timing_model().predicted_syscall_ns();
            issue_expert_hint_task(task);
            {
                std::lock_guard<std::mutex> lock(mu);
                issued_tasks++;
            }
        }
    }

    ExpertHintTask pop_one_unlocked() {
        ExpertHintTask task;
        if (priority_enabled && priority_heap_enabled) {
            auto cmp = [this](const ExpertHintTask & a, const ExpertHintTask & b) {
                return is_higher_priority(b, a);
            };
            std::pop_heap(priority_heap.begin(), priority_heap.end(), cmp);
            task = std::move(priority_heap.back());
            priority_heap.pop_back();
            priority_pops++;
            priority_heap_pops++;
        } else if (priority_enabled) {
            auto best = tasks.begin();
            for (auto it = tasks.begin(); it != tasks.end(); ++it) {
                if (is_higher_priority(*it, *best)) {
                    best = it;
                }
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
        uint64_t priority = 0;
        uint64_t heap_pops = 0;
        uint64_t fallback = 0;
        uint64_t queue_full = 0;
        uint64_t start_fail = 0;
        uint64_t high_water = 0;
        uint64_t queued_bytes_high_water = 0;
        uint64_t pressure = 0;
        uint64_t value = 0;
        uint64_t queue_cancel = 0;
        uint64_t final_queue_depth = 0;
        size_t cap = 0;
        size_t workers_started = 0;
        bool priority_on = false;
        bool heap_on = false;
        ExpertAsyncPriorityMode mode = ExpertAsyncPriorityMode::Score;
        {
            std::lock_guard<std::mutex> lock(mu);
            enqueued = enqueued_tasks;
            issued = issued_tasks;
            priority = priority_pops;
            heap_pops = priority_heap_pops;
            fallback = fallback_tasks;
            queue_full = queue_full_fallbacks;
            start_fail = start_fail_fallbacks;
            high_water = max_queue_depth;
            queued_bytes_high_water = max_queued_bytes;
            pressure = cancelled_pressure;
            value = cancelled_value;
            queue_cancel = cancelled_queue_full;
            final_queue_depth = queue_depth_unlocked();
            cap = capacity;
            workers_started = worker_count;
            priority_on = priority_enabled;
            heap_on = priority_heap_enabled;
            mode = priority_mode;
        }

        std::string line;
        line.reserve(512);
        line += "{\"event\":\"EXPERT_ASYNC_SUMMARY\",\"ts_ns\":" +
                std::to_string(llm_mem_trace_time_ns());
        line += ",\"enqueued\":" + std::to_string(enqueued);
        line += ",\"issued\":" + std::to_string(issued);
        line += ",\"issued_candidates\":" + std::to_string(issued);
        line += ",\"priority_enabled\":" +
                std::string(priority_on ? "true" : "false");
        line += ",\"priority_heap_enabled\":" +
                std::string(heap_on ? "true" : "false");
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
        line += ",\"cancelled_pressure\":" + std::to_string(pressure);
        line += ",\"cancelled_value\":" + std::to_string(value);
        line += ",\"cancelled_queue_full\":" + std::to_string(queue_cancel);
        line += ",\"final_queue_depth\":" + std::to_string(final_queue_depth);
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }

    size_t queue_depth_unlocked() const {
        return priority_enabled && priority_heap_enabled ? priority_heap.size() : tasks.size();
    }

    bool queue_empty_unlocked() const {
        return queue_depth_unlocked() == 0;
    }

    bool is_higher_priority(const ExpertHintTask & a, const ExpertHintTask & b) const {
        const auto key = [](const ExpertHintTask & task) {
            ExpertHintPriorityKey result;
            result.step = task.step;
            result.layer = task.layer;
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

    if (expert_deadline_observation_enabled()) {
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
        if (expert_value_gate_bypass_active_any(task.phase)) {
            expert_runtime_rescue_controller().record_value_gate_bypass();
        } else {
            return ExpertTaskGateResult::Value;
        }
    }
    return ExpertTaskGateResult::Accept;
}

bool submit_expert_hint_task(ExpertHintTask && task) {
    // Phase 2E-A: count every DECODE prefetch opportunity BEFORE gates.
    // Phase 2E-B: calibrated control also feeds the profile.
    if (expert_calibration_shadow_enabled() || expert_calibrated_control_enabled()) {
        expert_calibration_profile().record_prefetch_opportunity(task.phase);
    }
    if (expert_calibrated_control_enabled()) {
        expert_calibrated_controller().record_prefetch_opportunity(task.phase);
    }
    const ExpertTaskGateResult gate = prepare_expert_hint_task(task);
    if (gate == ExpertTaskGateResult::Pressure) {
        transition_expert_task(task.lifecycle, ExpertTaskEvent::Reject, "pressure_budget");
        write_expert_task_skip(task, "expert_prefetch_skip_pressure", "pressure_budget");
        if (expert_prefetch_async_enabled()) {
            expert_hint_queue().record_cancelled_pressure();
        }
        release_expert_task_hint_slot(task, true);
        return false;
    }
    if (gate == ExpertTaskGateResult::Value) {
        transition_expert_task(task.lifecycle, ExpertTaskEvent::Reject, "benefit_below_cost");
        write_expert_task_skip(task, "expert_prefetch_skip_value", "benefit_below_cost");
        if (expert_prefetch_async_enabled()) {
            expert_hint_queue().record_cancelled_value();
        }
        release_expert_task_hint_slot(task, true);
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
        if (!expert_prefetch_async_fallback_enabled()) {
            expert_hint_queue().record_cancelled_queue_full();
            transition_expert_task(task.lifecycle, ExpertTaskEvent::Cancel, "queue_full");
            write_expert_task_skip(task, "expert_prefetch_cancel_queue_full", "queue_full");
            release_expert_task_hint_slot(task, true);
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
        const char * tensor_name,
        int layer,
        int expert,
        uintptr_t addr,
        size_t nbytes,
        double route_score = 0.0,
        double route_confidence = 0.0) {
    ExpertHintTask task;
    task.action = action ? action : "expert_madvise_willneed";
    task.fadvise_action = fadvise_action ? fadvise_action : "expert_posix_fadvise_willneed";
    task.trigger = reason ? reason : "expert_prefetch";
    task.tensor_name = tensor_name ? tensor_name : "";
    task.layer = layer;
    task.expert = expert;
    task.addr = addr;
    task.nbytes = nbytes;
    task.route_score = route_score == route_score ? route_score : 0.0;
    task.route_confidence = route_confidence == route_confidence ? route_confidence : 0.0;
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

void log_param_access(
        const ggml_tensor * t,
        const ggml_tensor * parent,
        const char * parent_name) {
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

    observe_param_residency_demand(t, parent);
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
        if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
            return;
        }
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

        if (expert_memory_objects_enabled()) {
            const int cold_mode = expert_runtime_rescue_cold_issue_mode();
            const bool calibrated = expert_calibrated_control_enabled() &&
                    expert_madv_cold_reclaim_enabled();
            const int cal_mode = calibrated ?
                    expert_calibrated_controller().cold_issue_mode() : -1;
            if (calibrated && cal_mode == 3) {
                // Phase 2E-B scan-only: keep eligible-bytes observation, defer all,
                // never mark episodes, never call madvise.
                if (expert_route_hint_ttl_steps_for_phase(llm_mem_trace_get_phase()) != 0) {
                    expert_memory_object_tracker().end_layer(layer);
                    expert_memory_object_tracker().record_cold_skipped_ttl_nonzero();
                } else {
                    (void) expert_memory_object_tracker().end_layer_and_collect_madv_cold_candidates(
                            layer, step, expert_madv_cold_reclaim_grace_steps(), 1);
                }
            } else if (calibrated && cal_mode == 2) {
                const std::vector<ExpertMadVColdCandidate> candidates =
                        expert_memory_object_tracker().end_layer_and_collect_madv_cold_candidates(
                                layer, step, expert_madv_cold_reclaim_grace_steps(),
                                expert_calibrated_controller().cold_budget_remaining());
                if (!candidates.empty()) {
                    uint64_t committed = 0;
                    for (const ExpertMadVColdCandidate & candidate : candidates) {
                        committed += (uint64_t) candidate.nbytes;
                    }
                    expert_calibrated_controller().cold_budget_commit(committed);
                }
                for (const ExpertMadVColdCandidate & candidate : candidates) {
#ifdef __linux__
#ifdef MADV_COLD
                    const int rc = apply_madvise_hint(
                            "expert_madvise_cold",
                            MADV_COLD,
                            "expert_probation_reclaim",
                            candidate.tensor.c_str(),
                            candidate.layer,
                            candidate.expert,
                            candidate.addr,
                            candidate.nbytes);
                    expert_memory_object_tracker().record_madv_cold_result(
                            rc == 0, candidate.nbytes);
#else
                    expert_memory_object_tracker().record_madv_cold_result(false, candidate.nbytes);
#endif
#else
                    expert_memory_object_tracker().record_madv_cold_result(false, candidate.nbytes);
#endif
                }
            } else if (expert_madv_cold_reclaim_enabled() && cold_mode != 1) {
                if (expert_route_hint_ttl_steps_for_phase(llm_mem_trace_get_phase()) != 0) {
                    expert_memory_object_tracker().end_layer(layer);
                    expert_memory_object_tracker().record_cold_skipped_ttl_nonzero();
                } else {
                    const uint64_t max_collect_bytes = cold_mode == 2 ?
                            expert_runtime_rescue_controller().reentry_budget_remaining() : 0;
                    const std::vector<ExpertMadVColdCandidate> candidates =
                            expert_memory_object_tracker().end_layer_and_collect_madv_cold_candidates(
                                    layer, step, expert_madv_cold_reclaim_grace_steps(),
                                    max_collect_bytes);
                    if (cold_mode == 2 && !candidates.empty()) {
                        uint64_t committed = 0;
                        for (const ExpertMadVColdCandidate & candidate : candidates) {
                            committed += (uint64_t) candidate.nbytes;
                        }
                        expert_runtime_rescue_controller().reentry_budget_commit(committed);
                    }
                    for (const ExpertMadVColdCandidate & candidate : candidates) {
#ifdef __linux__
#ifdef MADV_COLD
                        const int rc = apply_madvise_hint(
                                "expert_madvise_cold",
                                MADV_COLD,
                                "expert_probation_reclaim",
                                candidate.tensor.c_str(),
                                candidate.layer,
                                candidate.expert,
                                candidate.addr,
                                candidate.nbytes);
                        expert_memory_object_tracker().record_madv_cold_result(
                                rc == 0, candidate.nbytes);
#else
                        expert_memory_object_tracker().record_madv_cold_result(false, candidate.nbytes);
#endif
#else
                        expert_memory_object_tracker().record_madv_cold_result(false, candidate.nbytes);
#endif
                    }
                }
            } else {
                expert_memory_object_tracker().end_layer(layer);
            }
        }

        const uint64_t ts = llm_mem_trace_time_ns();
        expert_timing_model().on_layer_end(step, layer, llm_mem_trace_get_phase(), ts);
        if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
            return;
        }
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
    const bool observe_tasks = expert_task_trace_mode() != ExpertTaskTraceMode::Off;
    const bool observe_memory_objects = expert_memory_objects_enabled();
    const bool observe_residency = llm_mem_trace_residency_attribution_enabled();
    if ((!observe_tasks && !observe_memory_objects && !observe_residency) || !operation ||
            operation->op != GGML_OP_MUL_MAT_ID) {
        return;
    }
    if (observe_tasks) {
        ensure_expert_task_summary_registered();
        ensure_expert_first_use_summary_registered();
    }
    if (observe_memory_objects) {
        ensure_expert_memory_object_summary_registered();
    }
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
        if (observe_residency) {
            llm_mem_trace_residency_attribution_observe(
                    "Routed Expert", info.name.c_str(),
                    residency_tensor_subclass(info.name.c_str()), info.layer, expert,
                    slice_addr, slice_bytes);
        }
        if (observe_memory_objects) {
            expert_memory_object_tracker().observe_first_use(
                    step, info.layer, expert, info.name);
        }
        if (!observe_tasks) {
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
        write_expert_first_use_event(
                expert_first_use_matcher().observe_first_use(std::move(use)));
    }
}

} // namespace

extern "C" void llm_mem_trace_runtime_rescue_step_end(uint64_t latency_ns) {
    // Phase 2E-A: shadow calibration step-end (observation-only, NOT gated on rescue).
    // Phase 2E-B: calibrated control also requires the profile to run.
    if (expert_calibration_shadow_enabled() || expert_calibrated_control_enabled()) {
        ensure_expert_calibration_shadow_summary_registered();
        const ExpertMemoryObjectCounters counters =
                expert_memory_object_tracker().counters();
        CalibrationStepContext ctx;
        if (expert_calibrated_control_enabled()) {
            ctx.rescue_state_safe =
                    expert_calibrated_controller().state_allows_calibration();
        } else {
            ctx.rescue_state_safe = !expert_runtime_rescue_enabled() ||
                    expert_runtime_rescue_controller().state_allows_calibration();
        }
        ctx.current_major_faults = current_major_fault_count();
        ctx.cumulative_cold_eligible_bytes = counters.cold_eligible_candidate_bytes;
        ctx.invariant_violations = counters.invariant_violations;
        ctx.cold_protected_violation = counters.cold_protected_violation;
        ctx.madv_cold_failed = counters.madv_cold_failed;
        ctx.pending = counters.pending;
        ctx.active = counters.active;
        ctx.current_hint_inflight_objects = counters.current_hint_inflight_objects;
        expert_calibration_profile().on_step_end(
                llm_mem_trace_get_phase(), llm_mem_trace_get_step(), latency_ns, ctx);
    }

    if (expert_calibrated_control_enabled()) {
        ensure_expert_calibrated_control_summary_registered();
        expert_calibrated_controller().on_step_end(
                llm_mem_trace_get_phase(), llm_mem_trace_get_step(), latency_ns);
    }

    if (!expert_runtime_rescue_enabled()) {
        return;
    }
    ensure_expert_runtime_rescue_summary_registered();
    expert_runtime_rescue_controller().on_step_end(
            llm_mem_trace_get_phase(), llm_mem_trace_get_step(), latency_ns);
}

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
        log_param_access(t->src[0], t, name);
    }
    if (t->src[1]) {
        log_param_access(t->src[1], t, name);
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

extern "C" int llm_mem_trace_moe_control_requires_router(void) {
    if (!llm_mem_trace_enabled()) {
        return 0;
    }
    return expert_prefetch_control_enabled() || expert_memory_objects_enabled();
}

extern "C" void llm_mem_trace_prefetch_expert_layer(int layer, int token_idx, const int * experts, const float * scores, int n_experts, const char * reason) {
    const bool prefetch_enabled = expert_prefetch_control_enabled();
    const bool observe_memory_objects = expert_memory_objects_enabled();
    if ((!prefetch_enabled && !observe_memory_objects) ||
            layer < 0 || !experts || n_experts <= 0) {
        return;
    }

    const std::vector<ExpertTensorInfo> tensors = expert_tensor_registry().for_layer(layer);
    if (tensors.empty()) {
        return;
    }

    const uint64_t step = llm_mem_trace_get_step();
    const int phase = llm_mem_trace_get_phase();
    if (observe_memory_objects) {
        ensure_expert_memory_object_summary_registered();
    }

    // Memory-object demand observation always follows the real Router output;
    // the 5A random policy is allowed to change only prefetch targets.
    if (observe_memory_objects) {
        for (int i = 0; i < n_experts; ++i) {
            const int expert = experts[i];
            if (expert < 0) {
                continue;
            }
            for (const ExpertTensorInfo & info : tensors) {
                uintptr_t slice_addr = 0;
                size_t slice_bytes = 0;
                if (expert_slice_range(info, expert, slice_addr, slice_bytes)) {
                    expert_memory_object_tracker().register_demand(
                            step, layer, expert, info.name, slice_addr, slice_bytes);
                }
            }
        }
    }

    // Memory Object demand tracking consumes the raw Router result above.  It
    // does not need prefetch target selection, score ordering, task lifecycle,
    // or any EXPERT-sink event construction.
    if (!prefetch_enabled) {
        return;
    }

    const uint64_t route_hint_ttl = expert_route_hint_ttl_steps_for_phase(phase);
    const int topk = expert_prefetch_topk_for_phase(phase);
    const int limit = topk > 0 ? std::min(n_experts, topk) : n_experts;
    const std::vector<int> router_targets(experts, experts + n_experts);
    const std::vector<int> targets = expert_prefetch_targets(
            phase, step, layer, token_idx, tensors, experts, n_experts, limit);
    ensure_expert_prefetch_selection_summary_registered();
    expert_prefetch_selection_stats().selection_events.fetch_add(
            1, std::memory_order_relaxed);
    expert_prefetch_selection_stats().selected_experts.fetch_add(
            targets.size(), std::memory_order_relaxed);
    write_expert_prefetch_selection_event(
            phase, step, layer, token_idx, router_targets, targets);
    static const bool registered = [] {
        std::atexit(write_expert_route_hint_summary);
        return true;
    }();
    (void) registered;

    std::unordered_set<int> router_expert_set(
            router_targets.begin(), router_targets.end());
    for (int expert : targets) {
        if (expert < 0) {
            continue;
        }
        const int router_index = [&] {
            for (int i = 0; i < n_experts; ++i) {
                if (experts[i] == expert) return i;
            }
            return -1;
        }();
        const double score = std::strcmp(
                expert_prefetch_selection_policy(), "router") == 0 &&
                scores && router_index >= 0 ? (double) scores[router_index] : 0.0;
        // Routed experts are certain to execute; router weights rank their contribution,
        // but are not probabilities that the selected expert will be used.
        const double confidence = 1.0;
        for (const ExpertTensorInfo & info : tensors) {
            uintptr_t slice_addr = 0;
            size_t slice_bytes = 0;
            if (!expert_slice_range(info, expert, slice_addr, slice_bytes)) {
                continue;
            }
            if (!prefetch_enabled) {
                continue;
            }
            const bool eligible = os_hint_size_allowed(slice_bytes);
            const char * tensor_type = expert_prefetch_tensor_type(info.name.c_str());
            expert_prefetch_selection_stats().target_events.fetch_add(
                    1, std::memory_order_relaxed);
            expert_prefetch_selection_stats().requested_bytes.fetch_add(
                    slice_bytes, std::memory_order_relaxed);
            if (std::strcmp(tensor_type, "Gate/Up") == 0) {
                expert_prefetch_selection_stats().gate_up_bytes.fetch_add(
                        slice_bytes, std::memory_order_relaxed);
            } else if (std::strcmp(tensor_type, "Down") == 0) {
                expert_prefetch_selection_stats().down_bytes.fetch_add(
                        slice_bytes, std::memory_order_relaxed);
            }
            if (eligible) {
                expert_prefetch_selection_stats().eligible_targets.fetch_add(
                        1, std::memory_order_relaxed);
            }

            if (!eligible) {
                write_expert_prefetch_target_event(
                        phase, step, layer, token_idx, expert, info, slice_bytes,
                        router_expert_set.count(expert) != 0, false, false);
                continue;
            }

            if (!expert_tensor_registry().mark_hinted(step, layer, expert, info.addr, route_hint_ttl)) {
                expert_prefetch_selection_stats().dedup_skipped.fetch_add(
                        1, std::memory_order_relaxed);
                write_expert_prefetch_target_event(
                        phase, step, layer, token_idx, expert, info, slice_bytes,
                        router_expert_set.count(expert) != 0, true, true);
                continue;
            }

            write_expert_prefetch_target_event(
                    phase, step, layer, token_idx, expert, info, slice_bytes,
                    router_expert_set.count(expert) != 0, true, false);

            const bool hint_slot_acquired = expert_inflight_hint_aggregation_enabled() &&
                    expert_memory_object_tracker().try_acquire_hint_slot(
                            layer, expert, info.name);
            if (expert_inflight_hint_aggregation_enabled() && !hint_slot_acquired) {
                continue;
            }

            ExpertHintTask task = make_expert_hint_task(
                    "expert_madvise_willneed",
                    "expert_posix_fadvise_willneed",
                    reason,
                    info.name.c_str(),
                    layer,
                    expert,
                    slice_addr,
                    slice_bytes,
                    score,
                    confidence);
            expert_prefetch_selection_stats().task_created.fetch_add(
                    1, std::memory_order_relaxed);
            task.memory_object_hint_slot_acquired = hint_slot_acquired;
            submit_expert_hint_task(std::move(task));
        }
    }
}
