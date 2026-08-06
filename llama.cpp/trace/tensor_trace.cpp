#include "trace_event.h"
#include "expert_prefetch_types.h"
#include "expert_tensor_registry.h"
#include "expert_prefetch_policy.h"
#include "expert_hint_priority.h"
#include "expert_max_wait_protection.h"
#include "expert_reserved_service.h"
#include "expert_queue_overhead_observation.h"
#include "expert_tensor_stage.h"
#include "expert_task_lifecycle.h"
#include "expert_first_use_matcher.h"
#include "expert_shadow_slack.h"
#include "expert_pressure_shadow.h"

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
#include <memory>
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

bool router_score_diagnostic_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLM_MEM_TRACE_ROUTER_SCORE_DIAGNOSTIC");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

uint64_t f64_bits(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void append_f64_bits(std::string & line, double value) {
    char buffer[24];
    std::snprintf(
            buffer,
            sizeof(buffer),
            "0x%016llx",
            (unsigned long long) f64_bits(value));
    json_escape_append(line, buffer);
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

uint64_t env_us_to_ns_or_default(const char * key, uint64_t def_us) {
    const uint64_t value_us = env_u64_or_default(key, def_us);
    return value_us > std::numeric_limits<uint64_t>::max() / 1000ull ?
            std::numeric_limits<uint64_t>::max() : value_us * 1000ull;
}

const ExpertShadowConfig & expert_shadow_config() {
    static const ExpertShadowConfig config = [] {
        ExpertShadowConfig result;
        const char * mode = std::getenv("LLM_MEM_TRACE_OPT_EXPERT_SLACK_MODE");
        if (mode && mode[0]) {
            if (std::strcmp(mode, "shadow") == 0) {
                result.enabled = true;
            } else if (std::strcmp(mode, "off") != 0) {
                result.config_error = true;
            }
        }
        result.window_capacity = std::min<size_t>(
                env_size_or_default("LLM_MEM_TRACE_OPT_EXPERT_SHADOW_WINDOW", 64), 1024);
        result.min_samples = std::min<uint64_t>(
                env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_SHADOW_MIN_SAMPLES", 8), 1'000'000);
        result.ewma_alpha = env_double_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_EWMA_ALPHA", 0.2);
        result.residual_quantile = env_double_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_RESIDUAL_QUANTILE", 0.25);
        result.horizon_default_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_HORIZON_DEFAULT_US", 5000);
        result.horizon_min_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_HORIZON_MIN_US", 1);
        result.horizon_max_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_HORIZON_MAX_US", 5'000'000);
        result.worker_default_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_WORKER_OCCUPIED_DEFAULT_US",
                env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_SHADOW_WORKER_DEFAULT_US", 50));
        result.worker_min_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_WORKER_OCCUPIED_MIN_US",
                env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_SHADOW_WORKER_MIN_US", 1));
        result.worker_max_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_WORKER_OCCUPIED_MAX_US",
                env_u64_or_default(
                        "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_WORKER_MAX_US", 1'000'000));
        result.pre_issue_default_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_PRE_ISSUE_DEFAULT_US", 10);
        result.pre_issue_min_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_PRE_ISSUE_MIN_US", 1);
        result.pre_issue_max_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_PRE_ISSUE_MAX_US", 1'000'000);
        result.syscall_service_default_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_SYSCALL_SERVICE_DEFAULT_US", 40);
        result.syscall_service_min_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_SYSCALL_SERVICE_MIN_US", 1);
        result.syscall_service_max_ns = env_us_to_ns_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_SYSCALL_SERVICE_MAX_US", 1'000'000);
        const double throughput_mib_s = std::max(0.001, env_double_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_THROUGHPUT_DEFAULT_MIB_S", 512.0));
        result.throughput_default_bytes_per_ns =
                throughput_mib_s * 1024.0 * 1024.0 / 1e9;
        result.max_pending_tasks = std::min<size_t>(
                env_size_or_default("LLM_MEM_TRACE_OPT_EXPERT_SHADOW_MAX_PENDING", 8192), 65536);
        result.max_first_use_keys = std::min<size_t>(env_size_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_MAX_FIRST_USE_KEYS", 65536), 262144);
        result.max_estimator_cells = std::min<size_t>(
                env_size_or_default("LLM_MEM_TRACE_OPT_EXPERT_SHADOW_MAX_CELLS", 4096), 16384);
        result.max_residual_cells = std::min<size_t>(env_size_or_default(
                "LLM_MEM_TRACE_OPT_EXPERT_SHADOW_MAX_RESIDUAL_CELLS", 4096), 16384);
        result.step_retention = std::min<uint64_t>(
                env_u64_or_default("LLM_MEM_TRACE_OPT_EXPERT_SHADOW_STEP_RETENTION", 2), 64);
        return result;
    }();
    return config;
}

bool expert_shadow_enabled() {
    return expert_shadow_config().enabled;
}

bool expert_shadow_summary_requested() {
    const ExpertShadowConfig & config = expert_shadow_config();
    return config.enabled || config.config_error;
}

ExpertShadowSlack & expert_shadow_slack() {
    static ExpertShadowSlack shadow(expert_shadow_config());
    return shadow;
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
    llm_pressure_shadow::record_hint_call(
            meta && meta->has_trace_context ? meta->step : llm_mem_trace_get_step(),
            len);
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
    llm_pressure_shadow::record_hint_call(
            meta && meta->has_trace_context ? meta->step : llm_mem_trace_get_step(),
            advise_len);
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

bool expert_reserved_service_active_enabled() {
    static const bool enabled =
            os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_RESERVED_SERVICE_ACTIVE");
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
        if (std::strcmp(value, "max_wait_protection") == 0) {
            return ExpertAsyncPriorityMode::MaxWaitProtection;
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
    if (router_score_diagnostic_enabled()) {
        line += ",\"score_f64_bits\":";
        append_f64_bits(line, task.score);
    }
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

void append_shadow_error_aggregate(
        std::string & line, const ExpertShadowErrorAggregate & aggregate) {
    line += "{\"count\":" + std::to_string(aggregate.count);
    line += ",\"absolute_error_sum_ns\":" +
            std::to_string(aggregate.absolute_error_sum_ns);
    line += ",\"signed_error_sum_ns\":" + std::to_string(aggregate.signed_error_sum_ns);
    line += ",\"true_positive\":" + std::to_string(aggregate.true_positive);
    line += ",\"true_negative\":" + std::to_string(aggregate.true_negative);
    line += ",\"false_positive\":" + std::to_string(aggregate.false_positive);
    line += ",\"false_negative\":" + std::to_string(aggregate.false_negative);
    line += ",\"warmup\":" + std::to_string(aggregate.warmup);
    line += ",\"fallback\":" + std::to_string(aggregate.fallback);
    line += ",\"clipped\":" + std::to_string(aggregate.clipped);
    line += ",\"mature_exact\":" + std::to_string(aggregate.mature_exact);
    line += ",\"absolute_error_histogram\":[";
    for (size_t i = 0; i < aggregate.absolute_error_histogram.size(); ++i) {
        if (i != 0) {
            line += ",";
        }
        line += std::to_string(aggregate.absolute_error_histogram[i]);
    }
    line += "]";
    line += ",\"calibration\":[";
    for (size_t i = 0; i < aggregate.calibration.size(); ++i) {
        if (i != 0) {
            line += ",";
        }
        line += "{\"total\":" + std::to_string(aggregate.calibration[i].total);
        line += ",\"on_time\":" + std::to_string(aggregate.calibration[i].on_time) + "}";
    }
    line += "]}";
}

void append_shadow_duration_aggregate(
        std::string & line, const ExpertShadowDurationAggregate & aggregate) {
    line += "{\"count\":" + std::to_string(aggregate.count);
    line += ",\"absolute_error_sum_ns\":" +
            std::to_string(aggregate.absolute_error_sum_ns);
    line += ",\"signed_error_sum_ns\":" + std::to_string(aggregate.signed_error_sum_ns);
    line += ",\"warmup\":" + std::to_string(aggregate.warmup);
    line += ",\"fallback\":" + std::to_string(aggregate.fallback) + "}";
}

void append_shadow_target_aggregate(
        std::string & line, const ExpertShadowTargetAggregate & aggregate) {
    line += "{\"count\":" + std::to_string(aggregate.count);
    line += ",\"unavailable\":" + std::to_string(aggregate.unavailable);
    line += ",\"absolute_error_sum_ns\":" +
            std::to_string(aggregate.absolute_error_sum_ns);
    line += ",\"signed_error_sum_ns\":" +
            std::to_string(aggregate.signed_error_sum_ns);
    line += ",\"true_positive\":" + std::to_string(aggregate.true_positive);
    line += ",\"true_negative\":" + std::to_string(aggregate.true_negative);
    line += ",\"false_positive\":" + std::to_string(aggregate.false_positive);
    line += ",\"false_negative\":" + std::to_string(aggregate.false_negative);
    line += ",\"warmup\":" + std::to_string(aggregate.warmup);
    line += ",\"fallback\":" + std::to_string(aggregate.fallback);
    line += ",\"mature_exact\":" + std::to_string(aggregate.mature_exact);
    line += ",\"calibration\":[";
    for (size_t i = 0; i < aggregate.calibration.size(); ++i) {
        if (i != 0) {
            line += ",";
        }
        line += "{\"total\":" + std::to_string(aggregate.calibration[i].total);
        line += ",\"on_time\":" + std::to_string(aggregate.calibration[i].on_time) + "}";
    }
    line += "]}";
}

void write_expert_shadow_observations(
        const std::vector<ExpertShadowTaskObservation> & observations) {
    if (!expert_shadow_enabled() || !expert_task_detail_events_enabled() ||
            !llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    for (const ExpertShadowTaskObservation & observation : observations) {
        char addr_buf[32];
        std::snprintf(
                addr_buf, sizeof(addr_buf), "0x%llx",
                (unsigned long long) observation.addr);
        std::string line;
        line.reserve(32768);
        line += "{\"event\":\"EXPERT_SHADOW_SLACK\",\"ts_ns\":" +
                std::to_string(std::max(observation.returned_ts_ns, observation.first_use_ts_ns));
        line += ",\"schema_version\":2";
        line += ",\"semantics\":\"logical_first_use\",\"physical_load_observed\":false";
        line += ",\"issue_target\":\"issue_ts < logical_first_use_ts\"";
        line += ",\"return_target\":\"final_enabled_hint_return_ts < logical_first_use_ts\"";
        line += ",\"task_id\":" + std::to_string(observation.task_id);
        line += ",\"issue_id\":" + std::to_string(observation.issue_id);
        line += ",\"issue_task_count\":" + std::to_string(observation.issue_task_count);
        line += ",\"step\":" + std::to_string(observation.step);
        line += ",\"layer\":" + std::to_string(observation.layer);
        line += ",\"expert\":" + std::to_string(observation.expert);
        line += ",\"phase\":\"" + std::string(phase_name(observation.phase)) + "\"";
        line += ",\"stage\":\"" +
                std::string(expert_tensor_stage_name(observation.stage)) + "\"";
        line += ",\"tensor\":";
        json_escape_append(line, observation.tensor.c_str());
        line += ",\"addr\":";
        json_escape_append(line, addr_buf);
        line += ",\"nbytes\":" + std::to_string(observation.nbytes);
        line += ",\"issued_nbytes\":" + std::to_string(observation.issued_nbytes);
        line += ",\"prediction_ts_ns\":" + std::to_string(observation.prediction_ts_ns);
        line += ",\"enqueued_ts_ns\":" + std::to_string(observation.enqueued_ts_ns);
        line += ",\"dequeued_ts_ns\":" + std::to_string(observation.dequeued_ts_ns);
        line += ",\"issue_ts_ns\":" + std::to_string(observation.issue_ts_ns);
        line += ",\"issued_ts_ns\":" + std::to_string(observation.issue_ts_ns);
        line += ",\"returned_ts_ns\":" + std::to_string(observation.returned_ts_ns);
        line += ",\"first_use_ts_ns\":" + std::to_string(observation.first_use_ts_ns);
        line += ",\"queue_depth_before_enqueue\":" +
                std::to_string(observation.queue_depth_before_enqueue);
        line += ",\"queued_bytes_before_enqueue\":" +
                std::to_string(observation.queued_bytes_before_enqueue);
        line += ",\"active_workers\":" + std::to_string(observation.active_workers);
        if (observation.has_actual_queue_wait) {
            line += ",\"actual_queue_wait_ns\":" +
                    std::to_string(observation.actual_queue_wait_ns);
        } else {
            line += ",\"actual_queue_wait_ns\":null";
        }
        if (observation.first_use_ts_ns >= observation.prediction_ts_ns) {
            line += ",\"actual_first_use_horizon_ns\":" +
                    std::to_string(observation.first_use_ts_ns - observation.prediction_ts_ns);
        } else {
            line += ",\"actual_first_use_horizon_ns\":null";
        }
        if (observation.has_actual_pre_issue_overhead) {
            line += ",\"actual_pre_issue_overhead_ns\":" +
                    std::to_string(observation.actual_pre_issue_overhead_ns);
        } else {
            line += ",\"actual_pre_issue_overhead_ns\":null";
        }
        if (observation.has_actual_hint_syscall_service) {
            line += ",\"actual_hint_syscall_service_ns\":" +
                    std::to_string(observation.actual_hint_syscall_service_ns);
        } else {
            line += ",\"actual_hint_syscall_service_ns\":null";
        }
        if (observation.has_actual_worker_occupied) {
            line += ",\"actual_worker_occupied_ns\":" +
                    std::to_string(observation.actual_worker_occupied_ns);
        } else {
            line += ",\"actual_worker_occupied_ns\":null";
        }
        if (observation.has_actual_issue_slack) {
            line += ",\"actual_issue_slack_ns\":" +
                    std::to_string(observation.actual_issue_slack_ns);
            line += ",\"issue_on_time\":" + std::string(
                    observation.actual_issue_slack_ns > 0 ? "true" : "false");
        } else {
            line += ",\"actual_issue_slack_ns\":null,\"issue_on_time\":null";
        }
        if (observation.has_actual_return_slack) {
            line += ",\"actual_return_slack_ns\":" +
                    std::to_string(observation.actual_return_slack_ns);
            line += ",\"return_on_time\":" + std::string(
                    observation.actual_return_slack_ns > 0 ? "true" : "false");
        } else {
            line += ",\"actual_return_slack_ns\":null,\"return_on_time\":null";
        }
        line += ",\"coalesced\":" + std::string(observation.coalesced ? "true" : "false");
        line += ",\"finalized\":" + std::string(observation.finalized ? "true" : "false");
        line += ",\"causality_error\":" +
                std::string(observation.causality_error ? "true" : "false");
        if (!observation.unavailable_reason.empty()) {
            line += ",\"unavailable_reason\":";
            json_escape_append(line, observation.unavailable_reason.c_str());
        } else {
            line += ",\"unavailable_reason\":null";
        }
        line += ",\"predictions\":[";
        for (size_t i = 0; i < observation.predictions.size(); ++i) {
            if (i != 0) {
                line += ",";
            }
            const ExpertShadowPrediction & prediction = observation.predictions[i];
            line += "{\"predicted_first_use_ts_ns\":" +
                    std::to_string(prediction.predicted_first_use_ts_ns);
            line += ",\"predicted_first_use_horizon_ns\":" +
                    std::to_string(prediction.predicted_first_use_horizon_ns);
            line += ",\"raw_predicted_first_use_horizon_ns\":" +
                    std::to_string(prediction.raw_predicted_first_use_horizon_ns);
            line += ",\"residual_adjustment_ns\":" +
                    std::to_string(prediction.residual_adjustment_ns);
            line += ",\"predicted_queue_wait_ns\":" +
                    std::to_string(prediction.predicted_queue_wait_ns);
            line += ",\"predicted_pre_issue_overhead_ns\":" +
                    std::to_string(prediction.predicted_pre_issue_overhead_ns);
            line += ",\"predicted_hint_syscall_service_ns\":" +
                    std::to_string(prediction.predicted_hint_syscall_service_ns);
            line += ",\"predicted_worker_occupied_ns\":" +
                    std::to_string(prediction.predicted_worker_occupied_ns);
            line += ",\"predicted_issue_slack_ns\":" +
                    std::to_string(prediction.predicted_issue_slack_ns);
            line += ",\"predicted_return_slack_ns\":" +
                    std::to_string(prediction.predicted_return_slack_ns);
            line += ",\"estimator_sample_count\":" +
                    std::to_string(prediction.estimator_sample_count);
            line += ",\"estimator_effective_sample_count\":" +
                    std::to_string(prediction.estimator_effective_sample_count);
            line += ",\"residual_sample_count\":" +
                    std::to_string(prediction.residual_sample_count);
            line += ",\"residual_effective_sample_count\":" +
                    std::to_string(prediction.residual_effective_sample_count);
            line += ",\"queue_sample_count\":" +
                    std::to_string(prediction.queue_sample_count);
            line += ",\"worker_sample_count\":" +
                    std::to_string(prediction.worker_sample_count);
            line += ",\"pre_issue_sample_count\":" +
                    std::to_string(prediction.pre_issue_sample_count);
            line += ",\"syscall_service_sample_count\":" +
                    std::to_string(prediction.syscall_service_sample_count);
            line += ",\"estimator_warmup\":" +
                    std::string(prediction.estimator_warmup ? "true" : "false");
            line += ",\"queue_warmup\":" +
                    std::string(prediction.queue_warmup ? "true" : "false");
            line += ",\"worker_warmup\":" +
                    std::string(prediction.worker_warmup ? "true" : "false");
            line += ",\"pre_issue_warmup\":" +
                    std::string(prediction.pre_issue_warmup ? "true" : "false");
            line += ",\"syscall_service_warmup\":" +
                    std::string(prediction.syscall_service_warmup ? "true" : "false");
            line += ",\"residual_warmup\":" +
                    std::string(prediction.residual_warmup ? "true" : "false");
            line += ",\"deadline_model\":\"" +
                    std::string(expert_shadow_grouping_name(prediction.grouping)) + "_" +
                    expert_shadow_estimator_name(prediction.estimator) + "\"";
            line += ",\"queue_model\":\"" +
                    std::string(expert_shadow_queue_model_name(prediction.queue_model)) + "\"";
            line += ",\"calibration_model\":\"" + std::string(
                    expert_shadow_calibration_model_name(prediction.calibration_model)) + "\"";
            line += ",\"fallback_level\":\"" +
                    std::string(expert_shadow_fallback_name(prediction.fallback_level)) + "\"";
            line += ",\"queue_fallback_level\":\"" +
                    std::string(expert_shadow_fallback_name(prediction.queue_fallback_level)) + "\"";
            line += ",\"worker_fallback_level\":\"" +
                    std::string(expert_shadow_fallback_name(prediction.worker_fallback_level)) + "\"";
            line += ",\"pre_issue_fallback_level\":\"" + std::string(
                    expert_shadow_fallback_name(prediction.pre_issue_fallback_level)) + "\"";
            line += ",\"syscall_service_fallback_level\":\"" + std::string(
                    expert_shadow_fallback_name(
                            prediction.syscall_service_fallback_level)) + "\"";
            line += ",\"residual_fallback_level\":\"" + std::string(
                    expert_shadow_fallback_name(prediction.residual_fallback_level)) + "\"";
            line += ",\"stage\":\"" +
                    std::string(expert_tensor_stage_name(observation.stage)) + "\"";
            line += ",\"phase\":\"" + std::string(phase_name(observation.phase)) + "\"";
            line += ",\"layer\":" + std::to_string(observation.layer);
            line += ",\"prediction_available\":" +
                    std::string(prediction.prediction_available ? "true" : "false");
            line += ",\"issue_prediction_available\":" +
                    std::string(prediction.issue_prediction_available ? "true" : "false");
            line += ",\"return_prediction_available\":" +
                    std::string(prediction.return_prediction_available ? "true" : "false");
            if (!prediction.prediction_available) {
                line += ",\"prediction_unavailable_reason\":\"";
                line += observation.active_workers == 0 ? "no_active_worker" :
                        (prediction.queue_model ==
                                 ExpertShadowQueueModel::QueuedBytesIssueThroughput &&
                         prediction.queue_sample_count == 0 ?
                                 "no_throughput_sample" : "unavailable");
                line += "\"";
            } else {
                line += ",\"prediction_unavailable_reason\":null";
            }
            line += ",\"clipped_low\":" +
                    std::string(prediction.clipped_low ? "true" : "false");
            line += ",\"clipped_high\":" +
                    std::string(prediction.clipped_high ? "true" : "false") + "}";
        }
        line += "]}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }
}

void write_expert_shadow_summary() {
    if (!expert_shadow_summary_requested() ||
            !llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    const ExpertShadowSummary summary = expert_shadow_slack().summary();
    const ExpertShadowConfig & config = summary.config;
    std::string line;
    line.reserve(65536);
    line += "{\"event\":\"EXPERT_SHADOW_SLACK_SUMMARY\",\"ts_ns\":" +
            std::to_string(llm_mem_trace_time_ns());
    line += ",\"schema_version\":2";
    line += ",\"mode\":\"" + std::string(config.enabled ? "shadow" : "off") + "\"";
    line += ",\"config_error\":" + std::string(config.config_error ? "true" : "false");
    line += ",\"semantics\":\"logical_first_use\",\"physical_load_observed\":false";
    line += ",\"targets\":{\"issue\":{\"prediction\":\"first_use_horizon - queue_wait - pre_issue_overhead\",\"actual_label\":\"issue_ts < logical_first_use_ts\"},\"return\":{\"prediction\":\"first_use_horizon - queue_wait - pre_issue_overhead - hint_syscall_service\",\"actual_label\":\"final_enabled_hint_return_ts < logical_first_use_ts\"}}";
    line += ",\"window_capacity\":" + std::to_string(config.window_capacity);
    line += ",\"min_samples\":" + std::to_string(config.min_samples);
    line += ",\"ewma_alpha\":" + std::to_string(config.ewma_alpha);
    line += ",\"residual_quantile\":" + std::to_string(config.residual_quantile);
    line += ",\"horizon_default_ns\":" + std::to_string(config.horizon_default_ns);
    line += ",\"horizon_min_ns\":" + std::to_string(config.horizon_min_ns);
    line += ",\"horizon_max_ns\":" + std::to_string(config.horizon_max_ns);
    line += ",\"worker_occupied_default_ns\":" + std::to_string(config.worker_default_ns);
    line += ",\"worker_occupied_min_ns\":" + std::to_string(config.worker_min_ns);
    line += ",\"worker_occupied_max_ns\":" + std::to_string(config.worker_max_ns);
    line += ",\"pre_issue_default_ns\":" + std::to_string(config.pre_issue_default_ns);
    line += ",\"pre_issue_min_ns\":" + std::to_string(config.pre_issue_min_ns);
    line += ",\"pre_issue_max_ns\":" + std::to_string(config.pre_issue_max_ns);
    line += ",\"syscall_service_default_ns\":" +
            std::to_string(config.syscall_service_default_ns);
    line += ",\"syscall_service_min_ns\":" +
            std::to_string(config.syscall_service_min_ns);
    line += ",\"syscall_service_max_ns\":" +
            std::to_string(config.syscall_service_max_ns);
    line += ",\"throughput_default_bytes_per_ns\":" +
            std::to_string(config.throughput_default_bytes_per_ns);
    line += ",\"max_pending_tasks\":" + std::to_string(config.max_pending_tasks);
    line += ",\"max_first_use_keys\":" + std::to_string(config.max_first_use_keys);
    line += ",\"max_estimator_cells\":" + std::to_string(config.max_estimator_cells);
    line += ",\"max_residual_cells\":" + std::to_string(config.max_residual_cells);
    line += ",\"step_retention\":" + std::to_string(config.step_retention);
    line += ",\"absolute_error_histogram_bounds_ns\":[100000,500000,1000000,5000000,20000000,100000000]";
    line += ",\"calibration_bucket_labels\":[\"< -5 ms\",\"[-5 ms, -2 ms)\",\"[-2 ms, -1 ms)\",\"[-1 ms, -0.5 ms)\",\"[-0.5 ms, 0]\",\"(0, 0.5 ms]\",\"(0.5 ms, 1 ms]\",\"(1 ms, 2 ms]\",\"(2 ms, 5 ms]\",\"> 5 ms\"]";
    line += ",\"eligible_tasks\":" + std::to_string(summary.eligible_tasks);
    line += ",\"predicted_tasks\":" + std::to_string(summary.predicted_tasks);
    line += ",\"unavailable_tasks\":" + std::to_string(summary.unavailable_tasks);
    line += ",\"finalized_tasks\":" + std::to_string(summary.finalized_tasks);
    line += ",\"expired_tasks\":" + std::to_string(summary.expired_tasks);
    line += ",\"capacity_expired_tasks\":" +
            std::to_string(summary.capacity_expired_tasks);
    line += ",\"pending_tasks\":" + std::to_string(summary.pending_tasks);
    line += ",\"peak_live_tasks\":" + std::to_string(summary.peak_live_tasks);
    line += ",\"duplicate_task_ids\":" + std::to_string(summary.duplicate_task_ids);
    line += ",\"logical_first_uses\":" + std::to_string(summary.logical_first_uses);
    line += ",\"unmatched_first_uses\":" +
            std::to_string(summary.unmatched_first_uses);
    line += ",\"ambiguous_first_uses\":" +
            std::to_string(summary.ambiguous_first_uses);
    line += ",\"duplicate_first_uses\":" +
            std::to_string(summary.duplicate_first_uses);
    line += ",\"stage_mismatch_tasks\":" +
            std::to_string(summary.stage_mismatch_tasks);
    line += ",\"address_mismatch_tasks\":" +
            std::to_string(summary.address_mismatch_tasks);
    line += ",\"first_use_key_capacity_skips\":" +
            std::to_string(summary.first_use_key_capacity_skips);
    line += ",\"causality_errors\":" + std::to_string(summary.causality_errors);
    line += ",\"issue_groups_observed\":" +
            std::to_string(summary.issue_groups_observed);
    line += ",\"worker_duration_observations\":" +
            std::to_string(summary.worker_duration_observations);
    line += ",\"estimator_cells\":" + std::to_string(summary.estimator_cells);
    line += ",\"estimator_capacity_skips\":" +
            std::to_string(summary.estimator_capacity_skips);
    line += ",\"residual_cells\":" + std::to_string(summary.residual_cells);
    line += ",\"residual_capacity_skips\":" +
            std::to_string(summary.residual_capacity_skips);
    line += ",\"expired_without_issue\":" +
            std::to_string(summary.expired_without_issue);
    line += ",\"expired_without_first_use\":" +
            std::to_string(summary.expired_without_first_use);
    static const char * phase_names[] = {"UNKNOWN", "PREFILL", "DECODE"};
    static const ExpertTensorStage stages[] = {
        ExpertTensorStage::Early,
        ExpertTensorStage::Late,
        ExpertTensorStage::Unknown,
    };
    line += ",\"finalized_by_phase\":{";
    for (size_t i = 0; i < 3; ++i) {
        if (i != 0) {
            line += ",";
        }
        line += "\"" + std::string(phase_names[i]) + "\":" +
                std::to_string(summary.finalized_by_phase[i]);
    }
    line += "},\"finalized_by_stage\":{";
    for (size_t i = 0; i < 3; ++i) {
        if (i != 0) {
            line += ",";
        }
        line += "\"" + std::string(expert_tensor_stage_name(stages[i])) + "\":" +
                std::to_string(summary.finalized_by_stage[i]);
    }
    line += "},\"queue_models\":[";
    static const ExpertShadowQueueModel queue_models[] = {
        ExpertShadowQueueModel::QueueDepthWorkerEwma,
        ExpertShadowQueueModel::QueuedBytesIssueThroughput,
    };
    for (size_t i = 0; i < 2; ++i) {
        if (i != 0) {
            line += ",";
        }
        line += "{\"queue_model\":\"" +
                std::string(expert_shadow_queue_model_name(queue_models[i])) + "\",\"error\":";
        append_shadow_duration_aggregate(line, summary.queue_models[i]);
        line += "}";
    }
    line += "],\"pre_issue_model\":";
    append_shadow_duration_aggregate(line, summary.pre_issue_model);
    line += ",\"hint_syscall_service_model\":";
    append_shadow_duration_aggregate(line, summary.syscall_service_model);
    line += ",\"worker_occupied_model\":";
    append_shadow_duration_aggregate(line, summary.worker_occupied_model);
    static const char * worker_bucket_names[] = {
        "le_64k", "le_256k", "le_1m", "le_4m", "le_16m", "gt_16m",
    };
    line += ",\"pre_issue_duration_buckets\":[";
    for (size_t i = 0; i < summary.pre_issue_buckets.size(); ++i) {
        if (i != 0) {
            line += ",";
        }
        line += "{\"bucket\":\"" + std::string(worker_bucket_names[i]) + "\"";
        line += ",\"count\":" + std::to_string(summary.pre_issue_buckets[i].count);
        line += ",\"window_count\":" +
                std::to_string(summary.pre_issue_buckets[i].window_count);
        line += ",\"ewma_ns\":" + std::to_string(summary.pre_issue_buckets[i].ewma_ns) + "}";
    }
    line += "],\"pre_issue_duration_global\":{\"count\":" +
            std::to_string(summary.pre_issue_global.count);
    line += ",\"window_count\":" + std::to_string(summary.pre_issue_global.window_count);
    line += ",\"ewma_ns\":" + std::to_string(summary.pre_issue_global.ewma_ns) + "}";
    line += ",\"hint_syscall_service_duration_buckets\":[";
    for (size_t i = 0; i < summary.syscall_service_buckets.size(); ++i) {
        if (i != 0) {
            line += ",";
        }
        line += "{\"bucket\":\"" + std::string(worker_bucket_names[i]) + "\"";
        line += ",\"count\":" +
                std::to_string(summary.syscall_service_buckets[i].count);
        line += ",\"window_count\":" +
                std::to_string(summary.syscall_service_buckets[i].window_count);
        line += ",\"ewma_ns\":" +
                std::to_string(summary.syscall_service_buckets[i].ewma_ns) + "}";
    }
    line += "],\"hint_syscall_service_duration_global\":{\"count\":" +
            std::to_string(summary.syscall_service_global.count);
    line += ",\"window_count\":" +
            std::to_string(summary.syscall_service_global.window_count);
    line += ",\"ewma_ns\":" + std::to_string(summary.syscall_service_global.ewma_ns) + "}";
    line += ",\"worker_occupied_duration_buckets\":[";
    for (size_t i = 0; i < summary.worker_buckets.size(); ++i) {
        if (i != 0) {
            line += ",";
        }
        line += "{\"bucket\":\"" + std::string(worker_bucket_names[i]) + "\"";
        line += ",\"count\":" + std::to_string(summary.worker_buckets[i].count);
        line += ",\"window_count\":" + std::to_string(summary.worker_buckets[i].window_count);
        line += ",\"ewma_ns\":" + std::to_string(summary.worker_buckets[i].ewma_ns) + "}";
    }
    line += "],\"worker_occupied_duration_global\":{\"count\":" +
            std::to_string(summary.worker_global.count);
    line += ",\"window_count\":" + std::to_string(summary.worker_global.window_count);
    line += ",\"ewma_ns\":" + std::to_string(summary.worker_global.ewma_ns) + "}";
    line += ",\"throughput_sample_count\":" +
            std::to_string(summary.throughput_sample_count);
    line += ",\"throughput_ewma_bytes_per_ns\":" +
            std::to_string(summary.throughput_ewma_bytes_per_ns);
    line += ",\"candidates\":[";
    for (size_t i = 0; i < summary.candidates.size(); ++i) {
        if (i != 0) {
            line += ",";
        }
        const ExpertShadowCandidateSummary & candidate = summary.candidates[i];
        line += "{\"deadline_model\":\"" +
                std::string(expert_shadow_grouping_name(candidate.grouping)) + "_" +
                expert_shadow_estimator_name(candidate.estimator) + "\"";
        line += ",\"queue_model\":\"" +
                std::string(expert_shadow_queue_model_name(candidate.queue_model)) + "\"";
        line += ",\"calibration_model\":\"" + std::string(
                expert_shadow_calibration_model_name(candidate.calibration_model)) + "\"";
        line += ",\"eligible\":" + std::to_string(candidate.eligible);
        line += ",\"unavailable\":" + std::to_string(candidate.unavailable);
        line += ",\"first_use_error\":";
        append_shadow_error_aggregate(line, candidate.overall);
        line += ",\"first_use_by_phase\":{";
        for (size_t j = 0; j < 3; ++j) {
            if (j != 0) {
                line += ",";
            }
            line += "\"" + std::string(phase_names[j]) + "\":";
            append_shadow_error_aggregate(line, candidate.by_phase[j]);
        }
        line += "},\"first_use_by_stage\":{";
        for (size_t j = 0; j < 3; ++j) {
            if (j != 0) {
                line += ",";
            }
            line += "\"" + std::string(expert_tensor_stage_name(stages[j])) + "\":";
            append_shadow_error_aggregate(line, candidate.by_stage[j]);
        }
        line += "},\"issue_target\":{\"overall\":";
        append_shadow_target_aggregate(line, candidate.issue_overall);
        line += ",\"by_phase\":{";
        for (size_t j = 0; j < 3; ++j) {
            if (j != 0) {
                line += ",";
            }
            line += "\"" + std::string(phase_names[j]) + "\":";
            append_shadow_target_aggregate(line, candidate.issue_by_phase[j]);
        }
        line += "},\"by_stage\":{";
        for (size_t j = 0; j < 3; ++j) {
            if (j != 0) {
                line += ",";
            }
            line += "\"" + std::string(expert_tensor_stage_name(stages[j])) + "\":";
            append_shadow_target_aggregate(line, candidate.issue_by_stage[j]);
        }
        line += "}},\"return_target\":{\"overall\":";
        append_shadow_target_aggregate(line, candidate.return_overall);
        line += ",\"by_phase\":{";
        for (size_t j = 0; j < 3; ++j) {
            if (j != 0) {
                line += ",";
            }
            line += "\"" + std::string(phase_names[j]) + "\":";
            append_shadow_target_aggregate(line, candidate.return_by_phase[j]);
        }
        line += "},\"by_stage\":{";
        for (size_t j = 0; j < 3; ++j) {
            if (j != 0) {
                line += ",";
            }
            line += "\"" + std::string(expert_tensor_stage_name(stages[j])) + "\":";
            append_shadow_target_aggregate(line, candidate.return_by_stage[j]);
        }
        line += "}}}";
    }
    line += "]}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void ensure_expert_shadow_summary_registered() {
    (void) expert_shadow_slack();
    static const bool registered = [] {
        std::atexit(write_expert_shadow_summary);
        return true;
    }();
    (void) registered;
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
    llm_pressure_shadow::record_issue(task.step, task.nbytes);
    if (expert_task_detail_events_enabled() || expert_shadow_enabled()) {
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
    if (expert_shadow_enabled()) {
        ExpertShadowIssueInput input;
        input.issue_id = task.issue_id;
        input.issue_task_count = task.coalesced_task_count;
        input.issue_ts_ns = begin;
        input.returned_ts_ns = end;
        input.issued_nbytes = task.nbytes;
        if (task.coalesced_lifecycles.empty()) {
            input.task_ids.push_back(task.lifecycle.task_id);
        } else {
            input.task_ids.reserve(task.coalesced_lifecycles.size());
            for (const ExpertTaskLifecycleRecord & lifecycle : task.coalesced_lifecycles) {
                input.task_ids.push_back(lifecycle.task_id);
            }
        }
        write_expert_shadow_observations(
                expert_shadow_slack().observe_issue_group(std::move(input)));
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
        if (expert_task_trace_mode() != ExpertTaskTraceMode::Off || expert_shadow_enabled()) {
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

[[noreturn]] void expert_max_wait_config_fatal(const char * message) {
    std::fprintf(stderr, "ERROR: invalid max_wait_protection configuration: %s\n", message);
    std::abort();
}

ExpertMaxWaitConfig load_expert_max_wait_config(
        bool priority_enabled,
        bool priority_heap_enabled) {
    if (!expert_prefetch_async_enabled()) {
        expert_max_wait_config_fatal("LLM_MEM_TRACE_OPT_EXPERT_ASYNC must be 1");
    }
    if (!priority_enabled) {
        expert_max_wait_config_fatal("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY must be 1");
    }
    if (priority_heap_enabled) {
        expert_max_wait_config_fatal("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP must be 0");
    }
    if (expert_prefetch_async_batch_size() != 1) {
        expert_max_wait_config_fatal("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH must be 1");
    }
    if (expert_prefetch_async_batch_wait_us() != 0) {
        expert_max_wait_config_fatal("LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_WAIT_US must be 0");
    }
    if (!expert_deadline_observation_enabled()) {
        expert_max_wait_config_fatal("LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE must be 1");
    }

    ExpertMaxWaitConfig config;
    if (!expert_max_wait_parse_us(
                std::getenv("LLM_MEM_TRACE_OPT_EXPERT_MAX_WAIT_THRESHOLD_US"),
                false,
                config.threshold_us,
                config.threshold_ns)) {
        expert_max_wait_config_fatal(
                "LLM_MEM_TRACE_OPT_EXPERT_MAX_WAIT_THRESHOLD_US must be an explicit positive integer");
    }
    if (!expert_max_wait_parse_us(
                std::getenv("LLM_MEM_TRACE_OPT_EXPERT_URGENT_GUARD_US"),
                true,
                config.urgent_guard_us,
                config.urgent_guard_ns)) {
        expert_max_wait_config_fatal(
                "LLM_MEM_TRACE_OPT_EXPERT_URGENT_GUARD_US must be an explicit non-negative integer");
    }
    return config;
}

[[noreturn]] void expert_queue_overhead_config_fatal(const char * message) {
    std::fprintf(stderr, "ERROR: invalid queue overhead observation configuration: %s\n", message);
    std::abort();
}

ExpertQueueOverheadMode load_expert_queue_overhead_mode() {
    ExpertQueueOverheadMode mode = ExpertQueueOverheadMode::Off;
    if (!expert_queue_overhead_parse_mode(
                std::getenv("LLM_MEM_TRACE_QUEUE_OVERHEAD_MODE"), mode)) {
        expert_queue_overhead_config_fatal(
                "LLM_MEM_TRACE_QUEUE_OVERHEAD_MODE must be off, summary, or detail");
    }
    if (mode != ExpertQueueOverheadMode::Off &&
            !llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        expert_queue_overhead_config_fatal(
                "summary/detail requires the Memory Trace sink");
    }
    return mode;
}

ExpertMaxWaitKey expert_max_wait_key(const ExpertHintTask & task) {
    ExpertMaxWaitKey result;
    result.priority.step = task.step;
    result.priority.layer = task.layer;
    result.priority.stage = task.stage;
    result.priority.route_score = task.route_score;
    result.priority.sequence = task.sequence;
    result.priority.deadline_ts_ns = task.deadline_ts_ns;
    result.enqueued_ts_ns = task.lifecycle.enqueued_ts_ns;
    return result;
}

struct ExpertMaxWaitSelectionMeta {
    bool valid = false;
    uint64_t decision_ts_ns = 0;
    uint64_t enqueued_ts_ns = 0;
    ExpertMaxWaitDecision decision;
    uint64_t protected_candidate_count = 0;
    bool normal_competitor_present = false;
};

struct ExpertPrioritySelectionMeta {
    bool valid = false;
    uint64_t decision_ts_ns = 0;
    uint64_t candidate_count = 0;
    ExpertAsyncPriorityMode mode = ExpertAsyncPriorityMode::Score;
};

struct ExpertQueueScanMeta {
    bool available = false;
    const char * strategy = "unavailable";
    uint64_t candidates = 0;
    uint64_t start_ts_ns = 0;
    uint64_t end_ts_ns = 0;
    uint64_t clock_read_count = 0;
};

struct ExpertQueueOverheadDetailMeta {
    ExpertQueueOverheadBatchSample batch;
    ExpertQueueOverheadSelectionSample selection;
};

void append_checked_duration_json(
        std::string & line,
        const ExpertQueueCheckedDuration & duration) {
    if (duration.available) {
        line += std::to_string(duration.value_ns);
    } else {
        line += "null";
    }
}

void write_expert_queue_overhead_selection(
        const ExpertQueueOverheadDetailMeta & meta) {
    const ExpertQueueOverheadBatchSample & batch = meta.batch;
    const ExpertQueueOverheadSelectionSample & selection = meta.selection;
    const ExpertQueueCheckedDuration acquire = expert_queue_checked_duration(
            batch.lock_wait_start_ts_ns, batch.lock_acquired_ts_ns);
    const ExpertQueueCheckedDuration hold = expert_queue_checked_duration(
            batch.lock_acquired_ts_ns, batch.lock_release_ts_ns);
    ExpertQueueCheckedDuration scan;
    if (selection.queue_scan_available) {
        scan = expert_queue_checked_duration(
                selection.scan_start_ts_ns, selection.scan_end_ts_ns);
    }

    const char * configured_run_id = std::getenv("LLM_MEM_TRACE_RUN_ID");
    std::string line;
    line.reserve(1024);
    line += "{\"event\":\"EXPERT_QUEUE_OVERHEAD_SELECTION\",\"ts_ns\":" +
            std::to_string(llm_mem_trace_time_ns());
    line += ",\"schema_version\":\"m6b2.1-queue-overhead-v1\"";
    line += ",\"run_id\":";
    json_escape_append(line, configured_run_id && configured_run_id[0] ?
            configured_run_id : "missing_run_id");
    line += ",\"semantics\":\"direct_queue_selection_measurement\"";
    line += ",\"decision_id\":" + std::to_string(selection.decision_id);
    line += ",\"batch_id\":" + std::to_string(selection.batch_id);
    line += ",\"batch_slot\":" + std::to_string(selection.batch_slot);
    line += ",\"worker_id\":" + std::to_string(selection.worker_id);
    line += ",\"phase\":";
    json_escape_append(line, phase_name(selection.phase));
    line += ",\"step\":" + std::to_string(selection.step);
    line += ",\"priority_mode\":";
    json_escape_append(line, expert_prefetch_async_priority_mode_name(selection.priority_mode));
    line += ",\"selection_strategy\":";
    json_escape_append(line, selection.selection_strategy);
    line += ",\"queue_depth_before\":" + std::to_string(selection.queue_depth_before);
    line += ",\"queue_scan_candidates\":" +
            std::to_string(selection.queue_scan_candidates);
    line += ",\"lock_wait_start_ts_ns\":" +
            std::to_string(batch.lock_wait_start_ts_ns);
    line += ",\"lock_acquired_ts_ns\":" +
            std::to_string(batch.lock_acquired_ts_ns);
    line += ",\"lock_release_ts_ns\":" +
            std::to_string(batch.lock_release_ts_ns);
    line += ",\"scan_start_ts_ns\":" + std::to_string(selection.scan_start_ts_ns);
    line += ",\"scan_end_ts_ns\":" + std::to_string(selection.scan_end_ts_ns);
    line += ",\"mutex_acquire_wait_ns\":";
    append_checked_duration_json(line, acquire);
    line += ",\"mutex_hold_ns\":";
    append_checked_duration_json(line, hold);
    line += ",\"queue_scan_ns\":";
    append_checked_duration_json(line, scan);
    line += ",\"winner_task_id\":" + std::to_string(selection.winner_task_id);
    line += ",\"winner_class\":";
    json_escape_append(line, selection.winner_class);
    line += ",\"batch_decision_ts_ns\":" +
            std::to_string(selection.batch_decision_ts_ns);
    line += ",\"clock_read_count\":" + std::to_string(
            batch.clock_read_count + selection.clock_read_count + 1);
    line += ",\"condition_reacquire_count\":" +
            std::to_string(batch.condition_reacquire_count);
    line += ",\"error_flags\":[";
    bool needs_comma = false;
    auto append_error = [&](const char * error) {
        if (needs_comma) {
            line += ',';
        }
        json_escape_append(line, error);
        needs_comma = true;
    };
    if (acquire.regression) append_error("mutex_acquire_clock_regression");
    if (hold.regression) append_error("mutex_hold_clock_regression");
    if (!selection.queue_scan_available) append_error("queue_scan_unavailable");
    if (scan.regression) append_error("queue_scan_clock_regression");
    line += "]";
    line += ",\"physical_load_observed\":false}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void append_queue_overhead_aggregate(
        std::string & line,
        const ExpertQueueBoundedAggregate & aggregate) {
    line += "{\"count\":" + std::to_string(aggregate.count);
    line += ",\"total\":";
    if (aggregate.count != 0) {
        line += std::to_string(aggregate.total);
    } else {
        line += "null";
    }
    line += ",\"mean\":";
    if (aggregate.count != 0) {
        line += std::to_string(
                static_cast<double>(aggregate.total) /
                static_cast<double>(aggregate.count));
    } else {
        line += "null";
    }
    line += ",\"min\":";
    line += aggregate.count != 0 ? std::to_string(aggregate.min) : "null";
    line += ",\"max\":";
    line += aggregate.count != 0 ? std::to_string(aggregate.max) : "null";
    line += ",\"p50_bucket_upper_bound\":";
    line += aggregate.count != 0 ?
            std::to_string(aggregate.quantile_bucket_upper(0.50)) : "null";
    line += ",\"p95_bucket_upper_bound\":";
    line += aggregate.count != 0 ?
            std::to_string(aggregate.quantile_bucket_upper(0.95)) : "null";
    line += ",\"p99_bucket_upper_bound\":";
    line += aggregate.count != 0 ?
            std::to_string(aggregate.quantile_bucket_upper(0.99)) : "null";
    line += ",\"zero_count\":" + std::to_string(aggregate.zero_count);
    line += ",\"unavailable_count\":" + std::to_string(aggregate.unavailable_count);
    line += ",\"clock_regression_count\":" +
            std::to_string(aggregate.regression_count);
    line += ",\"overflow_count\":" + std::to_string(aggregate.overflow_count);
    if (aggregate.count == 0) {
        line += ",\"empty_reason\":";
        json_escape_append(line, aggregate.unavailable_count != 0 ?
                "unavailable" : "no_samples");
    }
    line += ",\"quantile_semantics\":\"bucket_upper_bound\"";
    line += ",\"histogram_schema\":\"zero_then_log2_upper_bound\"";
    line += ",\"histogram\":[";
    for (size_t i = 0; i < aggregate.buckets.size(); ++i) {
        if (i != 0) {
            line += ',';
        }
        line += std::to_string(aggregate.buckets[i]);
    }
    line += "]}";
}

uint64_t queue_overhead_cell_overflow_count(const ExpertQueueOverheadCell & cell) {
    return cell.mutex_acquire_wait_ns.overflow_count +
            cell.mutex_hold_ns.overflow_count +
            cell.queue_scan_ns.overflow_count +
            cell.queue_scan_candidates.overflow_count;
}

uint64_t queue_overhead_cell_regression_count(const ExpertQueueOverheadCell & cell) {
    return cell.mutex_acquire_wait_ns.regression_count +
            cell.mutex_hold_ns.regression_count +
            cell.queue_scan_ns.regression_count;
}

uint64_t queue_overhead_cell_zero_count(const ExpertQueueOverheadCell & cell) {
    return cell.mutex_acquire_wait_ns.zero_count +
            cell.mutex_hold_ns.zero_count +
            cell.queue_scan_ns.zero_count;
}

bool queue_overhead_cell_has_samples(const ExpertQueueOverheadCell & cell) {
    return cell.mutex_acquire_wait_ns.count != 0 ||
            cell.mutex_acquire_wait_ns.unavailable_count != 0 ||
            cell.mutex_hold_ns.count != 0 ||
            cell.mutex_hold_ns.unavailable_count != 0 ||
            cell.queue_scan_ns.count != 0 ||
            cell.queue_scan_ns.unavailable_count != 0 ||
            cell.queue_scan_candidates.count != 0 ||
            cell.queue_scan_candidates.unavailable_count != 0;
}

void append_queue_overhead_cell(
        std::string & line,
        const ExpertQueueOverheadCell & cell) {
    line += "{\"mutex_acquire_wait_ns\":";
    append_queue_overhead_aggregate(line, cell.mutex_acquire_wait_ns);
    line += ",\"mutex_hold_ns\":";
    append_queue_overhead_aggregate(line, cell.mutex_hold_ns);
    line += ",\"queue_scan_ns\":";
    append_queue_overhead_aggregate(line, cell.queue_scan_ns);
    line += ",\"queue_scan_candidates\":";
    append_queue_overhead_aggregate(line, cell.queue_scan_candidates);
    line += "}";
}

void write_expert_priority_selection(
        const ExpertHintTask & task,
        const ExpertPrioritySelectionMeta & meta) {
    if (!meta.valid || !router_score_diagnostic_enabled() ||
            !expert_task_detail_events_enabled() ||
            !llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }

    std::string line;
    line.reserve(384);
    line += "{\"event\":\"EXPERT_PRIORITY_SELECTION\",\"ts_ns\":" +
            std::to_string(llm_mem_trace_time_ns());
    line += ",\"semantics\":\"queue_selection_diagnostic\"";
    line += ",\"physical_load_observed\":false";
    line += ",\"mode\":";
    json_escape_append(line, expert_prefetch_async_priority_mode_name(meta.mode));
    line += ",\"task_id\":" + std::to_string(task.lifecycle.task_id);
    line += ",\"decision_ts_ns\":" + std::to_string(meta.decision_ts_ns);
    line += ",\"candidate_count\":" + std::to_string(meta.candidate_count);
    line += ",\"deadline_ts_ns\":" + std::to_string(task.deadline_ts_ns);
    line += ",\"route_score\":" + std::to_string(task.route_score);
    line += ",\"route_score_f64_bits\":";
    append_f64_bits(line, task.route_score);
    line += ",\"sequence\":" + std::to_string(task.sequence);
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

void write_expert_max_wait_selection(
        const ExpertHintTask & task,
        const ExpertMaxWaitSelectionMeta & meta,
        const ExpertMaxWaitConfig & config) {
    if (!meta.valid || !expert_task_detail_events_enabled() ||
            !llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }

    std::string line;
    line.reserve(512);
    line += "{\"event\":\"EXPERT_MAX_WAIT_SELECTION\",\"ts_ns\":" +
            std::to_string(llm_mem_trace_time_ns());
    line += ",\"semantics\":\"queue_selection\"";
    line += ",\"physical_load_observed\":false";
    line += ",\"task_id\":" + std::to_string(task.lifecycle.task_id);
    line += ",\"decision_ts_ns\":" + std::to_string(meta.decision_ts_ns);
    line += ",\"enqueued_ts_ns\":" + std::to_string(meta.enqueued_ts_ns);
    line += ",\"waiting_ns\":";
    if (meta.decision.waiting_available) {
        line += std::to_string(meta.decision.waiting_ns);
    } else {
        line += "null";
    }
    line += ",\"threshold_us\":" + std::to_string(config.threshold_us);
    line += ",\"threshold_ns\":" + std::to_string(config.threshold_ns);
    line += ",\"urgent_guard_us\":" + std::to_string(config.urgent_guard_us);
    line += ",\"urgent_guard_ns\":" + std::to_string(config.urgent_guard_ns);
    line += ",\"class\":\"" +
            std::string(expert_max_wait_class_name(meta.decision.task_class)) + "\"";
    line += ",\"reason\":\"" +
            std::string(expert_max_wait_reason_name(meta.decision.reason)) + "\"";
    line += ",\"deadline_ts_ns\":" + std::to_string(task.deadline_ts_ns);
    line += ",\"route_score\":" + std::to_string(task.route_score);
    line += ",\"sequence\":" + std::to_string(task.sequence);
    line += ",\"protected_candidate_count\":" +
            std::to_string(meta.protected_candidate_count);
    line += ",\"normal_competitor_present\":" +
            std::string(meta.normal_competitor_present ? "true" : "false");
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

struct ExpertReservedRuntimeSelectionMeta {
    ExpertReservedSelection policy;
    uint64_t decision_id = 0;
    uint64_t batch_id = 0;
    uint64_t batch_slot = 0;
    uint64_t worker_id = 0;
    uint64_t index_op_start_ts_ns = 0;
    uint64_t index_op_end_ts_ns = 0;
};

[[noreturn]] void expert_reserved_service_fatal(const char * reason) {
    std::fprintf(stderr, "fatal: Reserved-Service Active invariant: %s\n",
            reason ? reason : "unknown");
    std::abort();
}

uint64_t expert_route_score_bits(double score) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(score), "F64 size mismatch");
    std::memcpy(&bits, &score, sizeof(bits));
    return bits;
}

std::string expert_u64_hex(uint64_t value) {
    static const char digits[] = "0123456789abcdef";
    std::string result(18, '0');
    result[0] = '0';
    result[1] = 'x';
    for (size_t i = 0; i < 16; ++i) {
        result[17 - i] = digits[value & 0xf];
        value >>= 4;
    }
    return result;
}

void append_expert_reserved_task_ref(
        std::string & line,
        const ExpertReservedTaskRef & ref) {
    if (!ref.available) {
        line += "null";
        return;
    }
    line += "{\"task_id\":" + std::to_string(ref.key.task_id);
    line += ",\"slot_id\":" + std::to_string(ref.handle.slot_id);
    line += ",\"generation\":" + std::to_string(ref.handle.generation);
    line += ",\"enqueued_ts_ns\":" + std::to_string(ref.key.enqueued_ts_ns);
    line += ",\"deadline_ts_ns\":" + std::to_string(ref.key.deadline_ts_ns);
    line += ",\"step\":" + std::to_string(ref.key.step);
    line += ",\"layer\":" + std::to_string(ref.key.layer);
    line += ",\"stage\":";
    json_escape_append(line, expert_tensor_stage_name(ref.key.stage));
    line += ",\"route_score_f64_bits\":";
    json_escape_append(line, expert_u64_hex(
            expert_route_score_bits(ref.key.route_score)).c_str());
    line += ",\"sequence\":" + std::to_string(ref.key.sequence);
    line += "}";
}

void write_expert_reserved_service_selection(
        const ExpertHintTask & task,
        const ExpertReservedRuntimeSelectionMeta & meta) {
    if (!meta.policy.valid ||
            !llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }
    const ExpertReservedSelection & policy = meta.policy;
    std::string line;
    line.reserve(2048);
    line += "{\"event\":\"EXPERT_RESERVED_SERVICE_SELECTION\",\"ts_ns\":" +
            std::to_string(llm_mem_trace_time_ns());
    line += ",\"schema_version\":\"m6c-active-v1\"";
    line += ",\"feature_flag\":\"LLM_MEM_TRACE_OPT_EXPERT_RESERVED_SERVICE_ACTIVE\"";
    line += ",\"decision_id\":" + std::to_string(meta.decision_id);
    line += ",\"batch_id\":" + std::to_string(meta.batch_id);
    line += ",\"batch_slot\":" + std::to_string(meta.batch_slot);
    line += ",\"worker_id\":" + std::to_string(meta.worker_id);
    line += ",\"decision_ts_ns\":" + std::to_string(policy.decision_ts_ns);
    line += ",\"selected_task_id\":" + std::to_string(task.lifecycle.task_id);
    line += ",\"winner_source\":";
    json_escape_append(line, expert_reserved_winner_source_name(policy.source));
    line += ",\"legacy_head\":";
    append_expert_reserved_task_ref(line, policy.legacy_head);
    line += ",\"aging_head\":";
    append_expert_reserved_task_ref(line, policy.aging_head);
    line += ",\"selected\":";
    append_expert_reserved_task_ref(line, policy.selected);
    line += ",\"waiting_eligible\":" +
            std::string(policy.waiting_eligible ? "true" : "false");
    line += ",\"hard_urgent_present\":" +
            std::string(policy.hard_urgent_present ? "true" : "false");
    line += ",\"reserved_triggered\":" +
            std::string(policy.reserved_triggered ? "true" : "false");
    line += ",\"reserved_due\":" +
            std::string(policy.reserved_due ? "true" : "false");
    line += ",\"active_winner_changed_vs_legacy\":" +
            std::string(policy.winner_changed_vs_legacy ? "true" : "false");
    line += ",\"reserved_same_as_legacy_head\":" +
            std::string(policy.reserved_same_as_legacy ? "true" : "false");
    line += ",\"credit_before\":" + std::to_string(policy.credit_before);
    line += ",\"credit_accrued\":" + std::to_string(policy.credit_accrued);
    line += ",\"credit_after\":" + std::to_string(policy.credit_after);
    line += ",\"debt_before\":" + std::string(policy.debt_before ? "true" : "false");
    line += ",\"debt_after\":" + std::string(policy.debt_after ? "true" : "false");
    line += ",\"debt_created\":" + std::string(policy.debt_created ? "true" : "false");
    line += ",\"debt_repaid\":" + std::string(policy.debt_repaid ? "true" : "false");
    line += ",\"store_size_before\":" + std::to_string(policy.size_before);
    line += ",\"store_size_after\":" + std::to_string(policy.size_after);
    line += ",\"queued_bytes_before\":" + std::to_string(policy.queued_bytes_before);
    line += ",\"queued_bytes_after\":" + std::to_string(policy.queued_bytes_after);
    line += ",\"index_op_start_ts_ns\":" +
            std::to_string(meta.index_op_start_ts_ns);
    line += ",\"index_op_end_ts_ns\":" +
            std::to_string(meta.index_op_end_ts_ns);
    line += ",\"index_op_duration_ns\":" + std::to_string(
            meta.index_op_end_ts_ns >= meta.index_op_start_ts_ns ?
            meta.index_op_end_ts_ns - meta.index_op_start_ts_ns : 0);
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

struct ExpertHintQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::unique_ptr<std::condition_variable_any> observed_cv;
    std::deque<ExpertHintTask> tasks;
    std::vector<ExpertHintTask> priority_heap;
    std::vector<ExpertHintTask> legacy_priority_heap;
    std::vector<std::unique_ptr<ExpertHintTask>> reserved_task_store;
    ExpertReservedServiceQueue reserved_service_queue;
    std::vector<std::thread> workers;
    bool started = false;
    bool stopping = false;
    size_t capacity = 0;
    size_t worker_count = 0;
    bool priority_enabled = false;
    bool priority_heap_enabled = false;
    bool reserved_service_active = false;
    ExpertAsyncPriorityMode priority_mode = ExpertAsyncPriorityMode::Score;
    ExpertMaxWaitConfig max_wait_config;
    ExpertQueueOverheadMode queue_overhead_mode = ExpertQueueOverheadMode::Off;
    std::unique_ptr<ExpertQueueOverheadObserver> queue_overhead_observer;
    uint64_t queue_overhead_next_batch_id = 0;
    uint64_t queue_overhead_next_decision_id = 0;
    uint64_t reserved_next_batch_id = 0;
    uint64_t reserved_next_decision_id = 0;
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
    uint64_t max_wait_eligible_count = 0;
    uint64_t max_wait_eligible_decisions = 0;
    uint64_t max_wait_selected_protected = 0;
    uint64_t max_wait_selected_urgent = 0;
    uint64_t max_wait_selected_normal = 0;
    uint64_t max_wait_still_waiting = 0;
    uint64_t max_wait_still_waiting_max = 0;
    uint64_t max_wait_protected_wait_count = 0;
    uint64_t max_wait_protected_wait_total_ns = 0;
    uint64_t max_wait_protected_wait_max_ns = 0;
    uint64_t max_wait_overshoot_count = 0;
    uint64_t max_wait_overshoot_total_ns = 0;
    uint64_t max_wait_overshoot_max_ns = 0;
    uint64_t max_wait_protected_over_normal = 0;
    uint64_t max_wait_missing_deadline = 0;
    uint64_t max_wait_missing_enqueue_timestamp = 0;
    uint64_t max_wait_enqueue_time_regression = 0;
    uint64_t max_wait_selection_count = 0;
    uint64_t enqueue_queue_op_count = 0;
    uint64_t enqueue_queue_op_total_ns = 0;
    uint64_t enqueue_queue_op_max_ns = 0;
    uint64_t reserved_enqueue_index_op_count = 0;
    uint64_t reserved_enqueue_index_op_total_ns = 0;
    uint64_t reserved_enqueue_index_op_max_ns = 0;
    uint64_t reserved_dequeue_index_op_count = 0;
    uint64_t reserved_dequeue_index_op_total_ns = 0;
    uint64_t reserved_dequeue_index_op_max_ns = 0;
    uint64_t reserved_hard_urgent_safety_violation = 0;
    uint64_t reserved_cross_store_invariant_error = 0;
    std::atomic<uint64_t> busy_workers{0};

    ~ExpertHintQueue() {
        shutdown();
    }

    bool enqueue(ExpertHintTask && task) {
        if (!ensure_started()) {
            return false;
        }
        ExpertQueueOverheadMode notify_mode = ExpertQueueOverheadMode::Off;
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
            if (expert_shadow_enabled()) {
                ExpertShadowTaskInput input;
                input.task_id = task.lifecycle.task_id;
                input.step = task.step;
                input.layer = task.layer;
                input.expert = task.expert;
                input.phase = task.phase;
                input.stage = task.stage;
                input.tensor = task.tensor_name;
                input.addr = task.addr;
                input.nbytes = task.nbytes;
                input.prediction_ts_ns = task.lifecycle.enqueued_ts_ns != 0 ?
                        task.lifecycle.enqueued_ts_ns : llm_mem_trace_time_ns();
                input.enqueued_ts_ns = input.prediction_ts_ns;
                input.queue_depth_before_enqueue = queue_depth_unlocked();
                input.queued_bytes_before_enqueue = queued_bytes;
                input.active_workers = worker_count;
                (void) expert_shadow_slack().register_task(std::move(input));
            }
            const uint64_t enqueue_op_start = llm_mem_trace_time_ns();
            if (reserved_service_active) {
                std::unique_ptr<ExpertHintTask> entity(
                        new ExpertHintTask(std::move(task)));
                ExpertReservedTaskKey key;
                key.task_id = entity->lifecycle.task_id;
                key.step = entity->step;
                key.layer = entity->layer;
                key.stage = entity->stage;
                key.route_score = entity->route_score;
                key.sequence = entity->sequence;
                key.deadline_ts_ns = entity->deadline_ts_ns;
                key.enqueued_ts_ns = entity->lifecycle.enqueued_ts_ns;
                key.nbytes = task_bytes;
                const uint64_t op_start = llm_mem_trace_time_ns();
                const ExpertReservedHandle handle = reserved_service_queue.insert(key);
                const uint64_t op_end = llm_mem_trace_time_ns();
                if (handle.slot_id >= reserved_task_store.size() ||
                        reserved_task_store[handle.slot_id]) {
                    reserved_cross_store_invariant_error++;
                    expert_reserved_service_fatal("Task store slot is not uniquely available");
                }
                reserved_task_store[handle.slot_id] = std::move(entity);
                const uint64_t duration = op_end >= op_start ? op_end - op_start : 0;
                reserved_enqueue_index_op_count++;
                reserved_enqueue_index_op_total_ns = expert_max_wait_saturating_add(
                        reserved_enqueue_index_op_total_ns, duration);
                reserved_enqueue_index_op_max_ns = std::max(
                        reserved_enqueue_index_op_max_ns, duration);
            } else if (priority_enabled && priority_heap_enabled) {
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
            const uint64_t enqueue_op_end = llm_mem_trace_time_ns();
            const uint64_t enqueue_op_duration =
                    enqueue_op_end >= enqueue_op_start ?
                    enqueue_op_end - enqueue_op_start : 0;
            enqueue_queue_op_count++;
            enqueue_queue_op_total_ns = expert_max_wait_saturating_add(
                    enqueue_queue_op_total_ns, enqueue_op_duration);
            enqueue_queue_op_max_ns = std::max(
                    enqueue_queue_op_max_ns, enqueue_op_duration);
            enqueued_tasks++;
            if (reserved_service_active) {
                queued_bytes = reserved_service_queue.queued_bytes();
            } else {
                queued_bytes += task_bytes;
            }
            max_queued_bytes = std::max(max_queued_bytes, queued_bytes);
            max_queue_depth = std::max<uint64_t>(max_queue_depth, (uint64_t) queue_depth_unlocked());
            notify_mode = queue_overhead_mode;
        }
        if (notify_mode == ExpertQueueOverheadMode::Off) {
            cv.notify_one();
        } else {
            observed_cv->notify_one();
        }
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
        reserved_service_active = expert_reserved_service_active_enabled();
        if (reserved_service_active &&
                (!priority_enabled || priority_mode != ExpertAsyncPriorityMode::DeadlineScore)) {
            expert_reserved_service_fatal(
                    "Active requires priority_enabled=1 and priority_mode=deadline_score");
        }
        queue_overhead_mode = load_expert_queue_overhead_mode();
        if (priority_mode == ExpertAsyncPriorityMode::MaxWaitProtection) {
            max_wait_config = load_expert_max_wait_config(
                    priority_enabled, priority_heap_enabled);
        }
        stopping = false;
        const size_t n_workers = std::min<size_t>(expert_prefetch_async_workers(), 16);
        worker_count = n_workers;
        queue_overhead_next_batch_id = 0;
        queue_overhead_next_decision_id = 0;
        reserved_next_batch_id = 0;
        reserved_next_decision_id = 0;
        try {
            if (reserved_service_active) {
                const ExpertReservedServiceConfig frozen_config;
                reserved_service_queue.reset(capacity, frozen_config);
                reserved_task_store.clear();
                reserved_task_store.resize(capacity);
            }
            if (queue_overhead_mode != ExpertQueueOverheadMode::Off) {
                observed_cv.reset(new std::condition_variable_any());
                queue_overhead_observer.reset(new ExpertQueueOverheadObserver());
                queue_overhead_observer->reset(
                        queue_overhead_mode,
                        n_workers,
                        expert_prefetch_async_batch_size(),
                        llm_mem_trace_time_ns);
            }
            workers.reserve(n_workers);
            for (size_t i = 0; i < n_workers; ++i) {
                workers.emplace_back([this, i] { run(i); });
            }
        } catch (...) {
            start_fail_fallbacks++;
            stopping = true;
            if (queue_overhead_mode == ExpertQueueOverheadMode::Off) {
                cv.notify_all();
            } else if (observed_cv) {
                observed_cv->notify_all();
            }
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
            reserved_service_active = false;
            priority_mode = ExpertAsyncPriorityMode::Score;
            max_wait_config = {};
            queue_overhead_mode = ExpertQueueOverheadMode::Off;
            queue_overhead_observer.reset();
            observed_cv.reset();
            reserved_task_store.clear();
            reserved_service_queue.clear();
            started = false;
            return false;
        }
        started = true;
        return true;
    }

    void shutdown() {
        ExpertQueueOverheadMode notify_mode = ExpertQueueOverheadMode::Off;
        {
            std::lock_guard<std::mutex> lock(mu);
            if (!started) {
                return;
            }
            stopping = true;
            notify_mode = queue_overhead_mode;
        }
        if (notify_mode == ExpertQueueOverheadMode::Off) {
            cv.notify_all();
        } else {
            observed_cv->notify_all();
        }
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
            reserved_task_store.clear();
            reserved_service_queue.clear();
            queued_bytes = 0;
            worker_count = 0;
            busy_workers.store(0, std::memory_order_relaxed);
            priority_enabled = false;
            priority_heap_enabled = false;
            reserved_service_active = false;
            priority_mode = ExpertAsyncPriorityMode::Score;
            max_wait_config = {};
            queue_overhead_mode = ExpertQueueOverheadMode::Off;
            queue_overhead_next_batch_id = 0;
            queue_overhead_next_decision_id = 0;
            reserved_next_batch_id = 0;
            reserved_next_decision_id = 0;
            queue_overhead_observer.reset();
            observed_cv.reset();
        }
    }

    void run(size_t worker_id) {
        for (;;) {
            std::vector<ExpertHintTask> batch;
            std::vector<ExpertPrioritySelectionMeta> priority_selections;
            std::vector<ExpertReservedRuntimeSelectionMeta> reserved_selections;
            ExpertMaxWaitSelectionMeta max_wait_selection;
            const bool pressure_shadow_on = llm_pressure_shadow::enabled();
            if (queue_overhead_mode == ExpertQueueOverheadMode::Off) {
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
                priority_selections.reserve(count);
                reserved_selections.reserve(count);
                if (reserved_service_active) {
                    const uint64_t batch_id = reserved_next_batch_id++;
                    const uint64_t decision_ts_ns = llm_mem_trace_time_ns();
                    for (size_t i = 0; i < count; ++i) {
                        ExpertReservedRuntimeSelectionMeta selection;
                        selection.decision_id = reserved_next_decision_id++;
                        selection.batch_id = batch_id;
                        selection.batch_slot = i;
                        selection.worker_id = worker_id;
                        batch.emplace_back(pop_one_reserved_unlocked(
                                decision_ts_ns, selection));
                        priority_selections.emplace_back();
                        reserved_selections.emplace_back(selection);
                    }
                } else if (priority_mode == ExpertAsyncPriorityMode::MaxWaitProtection) {
                    batch.emplace_back(pop_one_max_wait_unlocked<false>(
                            max_wait_selection, nullptr));
                    priority_selections.emplace_back();
                } else {
                    for (size_t i = 0; i < count; ++i) {
                        ExpertPrioritySelectionMeta selection;
                        batch.emplace_back(pop_one_unlocked<false>(&selection, nullptr));
                        priority_selections.emplace_back(selection);
                    }
                }
                worker_batches++;
                batched_candidates += count;
                if (pressure_shadow_on) {
                    busy_workers.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                ExpertQueueOverheadBatchSample batch_sample;
                std::vector<ExpertQueueOverheadSelectionSample> selection_samples;
                selection_samples.reserve(expert_prefetch_async_batch_size());
                ExpertQueueObservedLock lock(mu, llm_mem_trace_time_ns);
                uint64_t reacquire_count = 0;
                uint64_t condition_wait_count = 0;

                const uint64_t first_wait_lock_count = lock.lock_count();
                observed_cv->wait(lock, [&] { return stopping || !queue_empty_unlocked(); });
                const uint64_t first_wait_reacquires =
                        lock.lock_count() - first_wait_lock_count;
                if (first_wait_reacquires != 0) {
                    ++condition_wait_count;
                    reacquire_count += first_wait_reacquires;
                }
                if (queue_empty_unlocked()) {
                    const bool should_stop = stopping;
                    const uint64_t repeat_wake_count =
                            first_wait_reacquires > 0 ?
                            first_wait_reacquires - 1 : 0;
                    lock.unlock();
                    queue_overhead_observer->record_idle_wait_exit(
                            first_wait_reacquires != 0 ? 1 : 0,
                            first_wait_reacquires,
                            repeat_wake_count,
                            lock.clock_read_count());
                    if (should_stop) {
                        break;
                    }
                    continue;
                }

                const size_t batch_limit = expert_prefetch_async_batch_size();
                const uint64_t wait_us = expert_prefetch_async_batch_wait_us();
                if (!stopping && wait_us > 0 && batch_limit > 1 &&
                        queue_depth_unlocked() < batch_limit) {
                    const uint64_t batch_wait_lock_count = lock.lock_count();
                    observed_cv->wait_for(lock, std::chrono::microseconds(wait_us), [&] {
                        return stopping || queue_depth_unlocked() >= batch_limit;
                    });
                    const uint64_t batch_wait_reacquires =
                            lock.lock_count() - batch_wait_lock_count;
                    if (batch_wait_reacquires != 0) {
                        ++condition_wait_count;
                        reacquire_count += batch_wait_reacquires;
                    }
                }

                const size_t count = std::min(batch_limit, queue_depth_unlocked());
                const uint64_t batch_id = queue_overhead_next_batch_id++;
                uint64_t batch_decision_ts_ns = 0;
                bool legacy_decision_clock = false;
                if (!reserved_service_active &&
                        priority_mode != ExpertAsyncPriorityMode::MaxWaitProtection) {
                    batch_decision_ts_ns = llm_mem_trace_time_ns();
                    legacy_decision_clock = true;
                }
                batch.reserve(count);
                priority_selections.reserve(count);
                reserved_selections.reserve(count);
                if (reserved_service_active) {
                    batch_decision_ts_ns = llm_mem_trace_time_ns();
                    const uint64_t reserved_batch_id = reserved_next_batch_id++;
                    for (size_t i = 0; i < count; ++i) {
                        ExpertReservedRuntimeSelectionMeta selection;
                        selection.decision_id = reserved_next_decision_id++;
                        selection.batch_id = reserved_batch_id;
                        selection.batch_slot = i;
                        selection.worker_id = worker_id;
                        const uint64_t queue_depth_before = queue_depth_unlocked();
                        batch.emplace_back(pop_one_reserved_unlocked(
                                batch_decision_ts_ns, selection));
                        priority_selections.emplace_back();
                        reserved_selections.emplace_back(selection);

                        ExpertQueueOverheadSelectionSample sample;
                        sample.decision_id = queue_overhead_next_decision_id++;
                        sample.batch_id = batch_id;
                        sample.batch_slot = i;
                        sample.worker_id = worker_id;
                        sample.phase = batch.back().phase;
                        sample.step = batch.back().step;
                        sample.priority_mode = priority_mode;
                        sample.selection_strategy = "indexed_dual_heap";
                        sample.queue_depth_before = queue_depth_before;
                        sample.queue_scan_candidates = 0;
                        sample.queue_scan_available = true;
                        sample.scan_start_ts_ns = selection.index_op_start_ts_ns;
                        sample.scan_end_ts_ns = selection.index_op_end_ts_ns;
                        sample.winner_task_id = batch.back().lifecycle.task_id;
                        sample.winner_class = expert_reserved_winner_source_name(
                                selection.policy.source);
                        sample.batch_decision_ts_ns = batch_decision_ts_ns;
                        sample.clock_read_count = 2;
                        selection_samples.emplace_back(sample);
                    }
                } else if (priority_mode == ExpertAsyncPriorityMode::MaxWaitProtection) {
                    ExpertQueueScanMeta scan;
                    const uint64_t queue_depth_before = queue_depth_unlocked();
                    batch.emplace_back(pop_one_max_wait_unlocked<true>(
                            max_wait_selection, &scan));
                    priority_selections.emplace_back();
                    batch_decision_ts_ns = max_wait_selection.decision_ts_ns;

                    ExpertQueueOverheadSelectionSample sample;
                    sample.decision_id = queue_overhead_next_decision_id++;
                    sample.batch_id = batch_id;
                    sample.batch_slot = 0;
                    sample.worker_id = worker_id;
                    sample.phase = batch.back().phase;
                    sample.step = batch.back().step;
                    sample.priority_mode = priority_mode;
                    sample.selection_strategy = scan.strategy;
                    sample.queue_depth_before = queue_depth_before;
                    sample.queue_scan_candidates = scan.candidates;
                    sample.queue_scan_available = scan.available;
                    sample.scan_start_ts_ns = scan.start_ts_ns;
                    sample.scan_end_ts_ns = scan.end_ts_ns;
                    sample.winner_task_id = batch.back().lifecycle.task_id;
                    sample.winner_class =
                            expert_max_wait_class_name(max_wait_selection.decision.task_class);
                    sample.batch_decision_ts_ns = batch_decision_ts_ns;
                    sample.clock_read_count = scan.clock_read_count;
                    selection_samples.emplace_back(sample);
                } else {
                    for (size_t i = 0; i < count; ++i) {
                        ExpertPrioritySelectionMeta selection;
                        ExpertQueueScanMeta scan;
                        const uint64_t queue_depth_before = queue_depth_unlocked();
                        batch.emplace_back(pop_one_unlocked<true>(&selection, &scan));
                        priority_selections.emplace_back(selection);

                        ExpertQueueOverheadSelectionSample sample;
                        sample.decision_id = queue_overhead_next_decision_id++;
                        sample.batch_id = batch_id;
                        sample.batch_slot = i;
                        sample.worker_id = worker_id;
                        sample.phase = batch.back().phase;
                        sample.step = batch.back().step;
                        sample.priority_mode = priority_mode;
                        sample.selection_strategy = scan.strategy;
                        sample.queue_depth_before = queue_depth_before;
                        sample.queue_scan_candidates = scan.candidates;
                        sample.queue_scan_available = scan.available;
                        sample.scan_start_ts_ns = scan.start_ts_ns;
                        sample.scan_end_ts_ns = scan.end_ts_ns;
                        sample.winner_task_id = batch.back().lifecycle.task_id;
                        sample.winner_class = "legacy";
                        sample.batch_decision_ts_ns = batch_decision_ts_ns;
                        sample.clock_read_count = scan.clock_read_count;
                        selection_samples.emplace_back(sample);
                    }
                }
                worker_batches++;
                batched_candidates += count;
                if (pressure_shadow_on) {
                    busy_workers.fetch_add(1, std::memory_order_relaxed);
                }

                lock.unlock_and_measure();
                batch_sample.batch_id = batch_id;
                batch_sample.priority_mode = priority_mode;
                batch_sample.phase = batch.empty() ?
                        LLM_MEM_TRACE_PHASE_UNKNOWN : batch.front().phase;
                batch_sample.lock_wait_start_ts_ns = lock.last_wait_start_ts_ns();
                batch_sample.lock_acquired_ts_ns = lock.last_acquired_ts_ns();
                batch_sample.condition_wait_count = condition_wait_count;
                batch_sample.condition_reacquire_count = reacquire_count;
                batch_sample.repeat_wake_count =
                        reacquire_count > 0 ? reacquire_count - 1 : 0;
                batch_sample.lock_release_ts_ns = lock.last_release_ts_ns();
                batch_sample.clock_read_count = lock.clock_read_count() +
                        (legacy_decision_clock ? 1 : 0);

                queue_overhead_observer->record_batch(batch_sample);
                for (const ExpertQueueOverheadSelectionSample & sample : selection_samples) {
                    queue_overhead_observer->record_selection(sample);
                    if (queue_overhead_mode == ExpertQueueOverheadMode::Detail) {
                        ExpertQueueOverheadDetailMeta detail;
                        detail.batch = batch_sample;
                        detail.selection = sample;
                        queue_overhead_observer->record_detail_event();
                        write_expert_queue_overhead_selection(detail);
                    }
                }
            }

            for (size_t i = 0; i < batch.size(); ++i) {
                ExpertHintTask & task = batch[i];
                transition_expert_task(task.lifecycle, ExpertTaskEvent::Dequeue);
                write_expert_priority_selection(task, priority_selections[i]);
                if (i < reserved_selections.size()) {
                    write_expert_reserved_service_selection(
                            task, reserved_selections[i]);
                }
                if (i == 0 && max_wait_selection.valid) {
                    write_expert_max_wait_selection(task, max_wait_selection, max_wait_config);
                }
                if (expert_shadow_enabled()) {
                    const uint64_t dequeued_ts_ns = task.lifecycle.dequeued_ts_ns != 0 ?
                            task.lifecycle.dequeued_ts_ns : llm_mem_trace_time_ns();
                    expert_shadow_slack().observe_dequeue(
                            task.lifecycle.task_id, dequeued_ts_ns);
                }
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
                    if (expert_shadow_enabled()) {
                        expert_shadow_slack().expire_task(
                                task.lifecycle.task_id, "pressure_changed");
                    }
                    write_expert_task_skip(task, "expert_prefetch_cancel_pressure", "pressure_changed");
                    std::lock_guard<std::mutex> lock(mu);
                    cancelled_pressure++;
                    continue;
                }
                if (expert_task_below_value_threshold(task)) {
                    transition_expert_task(
                            task.lifecycle, ExpertTaskEvent::Cancel, "value_changed");
                    if (expert_shadow_enabled()) {
                        expert_shadow_slack().expire_task(
                                task.lifecycle.task_id, "value_changed");
                    }
                    write_expert_task_skip(task, "expert_prefetch_cancel_value", "value_changed");
                    std::lock_guard<std::mutex> lock(mu);
                    cancelled_value++;
                    continue;
                }
                if (expert_slack_enabled() && task.deadline_ts_ns != 0 &&
                        now + task.predicted_service_ns >= task.deadline_ts_ns) {
                    transition_expert_task(
                            task.lifecycle, ExpertTaskEvent::Cancel, "deadline_missed");
                    if (expert_shadow_enabled()) {
                        expert_shadow_slack().expire_task(
                                task.lifecycle.task_id, "deadline_missed");
                    }
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
            if (pressure_shadow_on) {
                busy_workers.fetch_sub(1, std::memory_order_relaxed);
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

    ExpertHintTask pop_one_reserved_unlocked(
            uint64_t decision_ts_ns,
            ExpertReservedRuntimeSelectionMeta & meta) {
        if (!reserved_service_active || !priority_enabled ||
                priority_mode != ExpertAsyncPriorityMode::DeadlineScore ||
                reserved_service_queue.size() == 0) {
            reserved_cross_store_invariant_error++;
            expert_reserved_service_fatal("invalid internal Active queue state during selection");
        }
        meta.index_op_start_ts_ns = llm_mem_trace_time_ns();
        meta.policy = reserved_service_queue.select(decision_ts_ns, stopping);
        meta.index_op_end_ts_ns = llm_mem_trace_time_ns();
        const ExpertReservedHandle handle = meta.policy.selected.handle;
        if (handle.slot_id >= reserved_task_store.size() ||
                !reserved_task_store[handle.slot_id]) {
            reserved_cross_store_invariant_error++;
            expert_reserved_service_fatal("selected Active Task entity is missing");
        }
        std::unique_ptr<ExpertHintTask> entity =
                std::move(reserved_task_store[handle.slot_id]);
        if (entity->lifecycle.task_id != meta.policy.selected.key.task_id ||
                entity->sequence != meta.policy.selected.key.sequence ||
                (uint64_t) entity->nbytes != meta.policy.selected.key.nbytes) {
            reserved_cross_store_invariant_error++;
            expert_reserved_service_fatal("selected Active Task identity mismatch");
        }
        if (meta.policy.hard_urgent_present &&
                meta.policy.source == ExpertReservedWinnerSource::Reserved) {
            reserved_hard_urgent_safety_violation++;
            expert_reserved_service_fatal("reserved winner bypassed hard-urgent Legacy head");
        }
        const uint64_t duration =
                meta.index_op_end_ts_ns >= meta.index_op_start_ts_ns ?
                meta.index_op_end_ts_ns - meta.index_op_start_ts_ns : 0;
        reserved_dequeue_index_op_count++;
        reserved_dequeue_index_op_total_ns = expert_max_wait_saturating_add(
                reserved_dequeue_index_op_total_ns, duration);
        reserved_dequeue_index_op_max_ns = std::max(
                reserved_dequeue_index_op_max_ns, duration);
        queued_bytes = reserved_service_queue.queued_bytes();
        priority_pops++;
        return std::move(*entity);
    }

    template<bool Observe>
    ExpertHintTask pop_one_max_wait_unlocked(
            ExpertMaxWaitSelectionMeta & meta,
            ExpertQueueScanMeta * scan) {
        if (!priority_enabled || priority_heap_enabled ||
                priority_mode != ExpertAsyncPriorityMode::MaxWaitProtection ||
                tasks.empty()) {
            expert_max_wait_config_fatal("invalid internal queue state during selection");
        }

        const uint64_t decision_now_ns = llm_mem_trace_time_ns();
        auto best = tasks.end();
        ExpertMaxWaitKey best_key;
        ExpertMaxWaitDecision best_decision;
        uint64_t protected_candidates = 0;
        bool normal_present = false;

        if constexpr (Observe) {
            scan->available = true;
            scan->strategy = "linear_scan";
            scan->start_ts_ns = llm_mem_trace_time_ns();
            scan->clock_read_count++;
        }
        for (auto it = tasks.begin(); it != tasks.end(); ++it) {
            if constexpr (Observe) {
                scan->candidates++;
            }
            const ExpertMaxWaitKey key = expert_max_wait_key(*it);
            const ExpertMaxWaitDecision decision = expert_max_wait_classify(
                    key, decision_now_ns, max_wait_config);
            if (decision.task_class == ExpertMaxWaitClass::Protected) {
                protected_candidates = expert_max_wait_saturating_add(
                        protected_candidates, 1);
            } else if (decision.task_class == ExpertMaxWaitClass::Normal) {
                normal_present = true;
            }
            if (best == tasks.end() || expert_max_wait_higher(
                        key, decision, best_key, best_decision)) {
                best = it;
                best_key = key;
                best_decision = decision;
            }
        }
        if constexpr (Observe) {
            scan->end_ts_ns = llm_mem_trace_time_ns();
            scan->clock_read_count++;
        }

        meta.valid = true;
        meta.decision_ts_ns = decision_now_ns;
        meta.enqueued_ts_ns = best_key.enqueued_ts_ns;
        meta.decision = best_decision;
        meta.protected_candidate_count = protected_candidates;
        meta.normal_competitor_present = normal_present;

        max_wait_eligible_count = expert_max_wait_saturating_add(
                max_wait_eligible_count, protected_candidates);
        if (protected_candidates != 0) {
            max_wait_eligible_decisions = expert_max_wait_saturating_add(
                    max_wait_eligible_decisions, 1);
        }
        max_wait_selection_count = expert_max_wait_saturating_add(
                max_wait_selection_count, 1);

        const bool selected_protected =
                best_decision.task_class == ExpertMaxWaitClass::Protected;
        if (best_decision.task_class == ExpertMaxWaitClass::Urgent) {
            max_wait_selected_urgent = expert_max_wait_saturating_add(
                    max_wait_selected_urgent, 1);
        } else if (selected_protected) {
            max_wait_selected_protected = expert_max_wait_saturating_add(
                    max_wait_selected_protected, 1);
            max_wait_protected_wait_count = expert_max_wait_saturating_add(
                    max_wait_protected_wait_count, 1);
            max_wait_protected_wait_total_ns = expert_max_wait_saturating_add(
                    max_wait_protected_wait_total_ns, best_decision.waiting_ns);
            max_wait_protected_wait_max_ns = std::max(
                    max_wait_protected_wait_max_ns, best_decision.waiting_ns);
            max_wait_overshoot_count = expert_max_wait_saturating_add(
                    max_wait_overshoot_count, 1);
            max_wait_overshoot_total_ns = expert_max_wait_saturating_add(
                    max_wait_overshoot_total_ns, best_decision.threshold_overshoot_ns);
            max_wait_overshoot_max_ns = std::max(
                    max_wait_overshoot_max_ns, best_decision.threshold_overshoot_ns);
            if (normal_present) {
                max_wait_protected_over_normal = expert_max_wait_saturating_add(
                        max_wait_protected_over_normal, 1);
            }
        } else {
            max_wait_selected_normal = expert_max_wait_saturating_add(
                    max_wait_selected_normal, 1);
        }

        max_wait_still_waiting = protected_candidates - (selected_protected ? 1 : 0);
        max_wait_still_waiting_max = std::max(
                max_wait_still_waiting_max, max_wait_still_waiting);
        if (best_key.priority.deadline_ts_ns == 0) {
            max_wait_missing_deadline = expert_max_wait_saturating_add(
                    max_wait_missing_deadline, 1);
        }
        if (best_decision.reason == ExpertMaxWaitReason::MissingEnqueueFallback) {
            max_wait_missing_enqueue_timestamp = expert_max_wait_saturating_add(
                    max_wait_missing_enqueue_timestamp, 1);
        } else if (best_decision.reason ==
                ExpertMaxWaitReason::EnqueueTimeRegressionFallback) {
            max_wait_enqueue_time_regression = expert_max_wait_saturating_add(
                    max_wait_enqueue_time_regression, 1);
        }

        ExpertHintTask task = std::move(*best);
        tasks.erase(best);
        priority_pops++;
        queued_bytes -= std::min<uint64_t>(queued_bytes, (uint64_t) task.nbytes);
        return task;
    }

    template<bool Observe>
    ExpertHintTask pop_one_unlocked(
            ExpertPrioritySelectionMeta * selection,
            ExpertQueueScanMeta * scan) {
        if (selection && router_score_diagnostic_enabled() && priority_enabled) {
            selection->valid = true;
            selection->decision_ts_ns = llm_mem_trace_time_ns();
            selection->candidate_count = queue_depth_unlocked();
            selection->mode = priority_mode;
        }
        ExpertHintTask task;
        if (priority_enabled && priority_heap_enabled) {
            if constexpr (Observe) {
                scan->strategy = "heap";
            }
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
            if constexpr (Observe) {
                scan->available = true;
                scan->strategy = "linear_scan";
                scan->start_ts_ns = llm_mem_trace_time_ns();
                scan->clock_read_count++;
            }
            auto best_known = tasks.end();
            auto best_legacy = tasks.end();
            for (auto it = tasks.begin(); it != tasks.end(); ++it) {
                if constexpr (Observe) {
                    scan->candidates++;
                }
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
            if constexpr (Observe) {
                scan->end_ts_ns = llm_mem_trace_time_ns();
                scan->clock_read_count++;
            }
            task = std::move(*best);
            tasks.erase(best);
            priority_pops++;
        } else {
            if constexpr (Observe) {
                scan->strategy = "fifo";
            }
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
        uint64_t enqueue_ops = 0;
        uint64_t enqueue_op_total_ns = 0;
        uint64_t enqueue_op_max_ns = 0;
        size_t cap = 0;
        size_t workers_started = 0;
        bool priority_on = false;
        bool heap_on = false;
        bool reserved_on = false;
        uint64_t final_queue_depth = 0;
        uint64_t final_queued_bytes = 0;
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
            enqueue_ops = enqueue_queue_op_count;
            enqueue_op_total_ns = enqueue_queue_op_total_ns;
            enqueue_op_max_ns = enqueue_queue_op_max_ns;
            cap = capacity;
            workers_started = worker_count;
            priority_on = priority_enabled;
            heap_on = priority_heap_enabled;
            reserved_on = reserved_service_active;
            final_queue_depth = queue_depth_unlocked();
            final_queued_bytes = queued_bytes;
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
        line += ",\"reserved_service_active\":" +
                std::string(reserved_on ? "true" : "false");
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
        line += ",\"enqueue_queue_op_count\":" + std::to_string(enqueue_ops);
        line += ",\"enqueue_queue_op_total_ns\":" + std::to_string(enqueue_op_total_ns);
        line += ",\"enqueue_queue_op_mean_ns\":" + std::to_string(
                enqueue_ops ? enqueue_op_total_ns / enqueue_ops : 0);
        line += ",\"enqueue_queue_op_max_ns\":" + std::to_string(enqueue_op_max_ns);
        line += ",\"batch_size\":" + std::to_string(expert_prefetch_async_batch_size());
        line += ",\"batch_wait_us\":" + std::to_string(expert_prefetch_async_batch_wait_us());
        line += ",\"final_queue_depth\":" + std::to_string(final_queue_depth);
        line += ",\"final_queued_bytes\":" + std::to_string(final_queued_bytes);
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
        if (mode == ExpertAsyncPriorityMode::MaxWaitProtection) {
            write_max_wait_summary();
        }
        if (reserved_on) {
            write_reserved_service_summary();
        }
        if (queue_overhead_mode != ExpertQueueOverheadMode::Off) {
            write_queue_overhead_summary(priority_on, heap_on, mode);
        }
    }

    void write_reserved_service_summary() {
        ExpertReservedServiceConfig config;
        ExpertReservedServiceCounters counters;
        ExpertReservedServiceAudit audit;
        uint64_t task_entity_count = 0;
        uint64_t enqueue_count = 0;
        uint64_t enqueue_total_ns = 0;
        uint64_t enqueue_max_ns = 0;
        uint64_t dequeue_count = 0;
        uint64_t dequeue_total_ns = 0;
        uint64_t dequeue_max_ns = 0;
        uint64_t hard_urgent_violations = 0;
        uint64_t cross_store_errors = 0;
        uint64_t runtime_queued_bytes = 0;
        uint64_t final_credit = 0;
        bool final_debt = false;
        {
            std::lock_guard<std::mutex> lock(mu);
            if (!reserved_service_active) {
                return;
            }
            config = reserved_service_queue.config();
            counters = reserved_service_queue.counters();
            audit = reserved_service_queue.audit(true);
            for (const auto & entity : reserved_task_store) {
                if (entity) {
                    task_entity_count++;
                }
            }
            enqueue_count = reserved_enqueue_index_op_count;
            enqueue_total_ns = reserved_enqueue_index_op_total_ns;
            enqueue_max_ns = reserved_enqueue_index_op_max_ns;
            dequeue_count = reserved_dequeue_index_op_count;
            dequeue_total_ns = reserved_dequeue_index_op_total_ns;
            dequeue_max_ns = reserved_dequeue_index_op_max_ns;
            hard_urgent_violations = reserved_hard_urgent_safety_violation;
            cross_store_errors = reserved_cross_store_invariant_error;
            runtime_queued_bytes = queued_bytes;
            final_credit = reserved_service_queue.credit();
            final_debt = reserved_service_queue.pending_debt();
        }
        const bool conserved = audit.valid && task_entity_count == audit.store_size &&
                runtime_queued_bytes == audit.queued_bytes;
        std::string line;
        line.reserve(2048);
        line += "{\"event\":\"EXPERT_RESERVED_SERVICE_SUMMARY\",\"ts_ns\":" +
                std::to_string(llm_mem_trace_time_ns());
        line += ",\"schema_version\":\"m6c-active-v1\"";
        line += ",\"active\":true";
        line += ",\"base_priority_mode\":\"deadline_score\"";
        line += ",\"reserved_numerator\":" + std::to_string(config.reserved_numerator);
        line += ",\"reserved_denominator\":" + std::to_string(config.reserved_denominator);
        line += ",\"eligibility_age_ns\":" + std::to_string(config.eligibility_age_ns);
        line += ",\"hard_urgent_guard_ns\":" + std::to_string(config.hard_urgent_guard_ns);
        line += ",\"eligibility_rule\":\"AGE_GATED_ALL\"";
        line += ",\"debt_policy\":\"single_pending_latch\"";
        line += ",\"reset_policy\":\"reset_when_no_eligible\"";
        line += ",\"reserved_winner\":\"oldest_eligible\"";
        line += ",\"store_kind\":\"bounded_unique_task_store\"";
        line += ",\"index_kind\":\"dual_indexed_binary_heap\"";
        line += ",\"reserved_trigger_count\":" +
                std::to_string(counters.reserved_trigger_count);
        line += ",\"reserved_due_count\":" +
                std::to_string(counters.reserved_due_count);
        line += ",\"reserved_selected_count\":" +
                std::to_string(counters.reserved_selected_count);
        line += ",\"active_winner_changed_count\":" +
                std::to_string(counters.active_winner_changed_count);
        line += ",\"reserved_same_as_legacy_head_count\":" +
                std::to_string(counters.reserved_same_as_legacy_count);
        line += ",\"hard_urgent_override_count\":" +
                std::to_string(counters.hard_urgent_override_count);
        line += ",\"hard_urgent_safety_violation\":" +
                std::to_string(hard_urgent_violations);
        line += ",\"debt_created_count\":" +
                std::to_string(counters.debt_created_count);
        line += ",\"debt_repaid_count\":" +
                std::to_string(counters.debt_repaid_count);
        line += ",\"insert_count\":" + std::to_string(counters.insert_count);
        line += ",\"erase_count\":" + std::to_string(counters.erase_count);
        line += ",\"selection_count\":" + std::to_string(counters.selection_count);
        line += ",\"legacy_heap_sift_count\":" +
                std::to_string(counters.legacy_heap_sift_count);
        line += ",\"aging_heap_sift_count\":" +
                std::to_string(counters.aging_heap_sift_count);
        line += ",\"stale_handle_count\":" +
                std::to_string(counters.stale_handle_count);
        line += ",\"duplicate_erase_count\":" +
                std::to_string(counters.duplicate_erase_count);
        line += ",\"generation_mismatch_count\":" +
                std::to_string(counters.generation_mismatch_count);
        line += ",\"full_store_scan_count\":" +
                std::to_string(counters.full_store_scan_count);
        line += ",\"invariant_error_count\":" + std::to_string(
                counters.invariant_error_count + cross_store_errors);
        line += ",\"store_size\":" + std::to_string(audit.store_size);
        line += ",\"task_entity_count\":" + std::to_string(task_entity_count);
        line += ",\"registry_size\":" + std::to_string(audit.registry_size);
        line += ",\"legacy_index_size\":" + std::to_string(audit.legacy_index_size);
        line += ",\"aging_index_size\":" + std::to_string(audit.aging_index_size);
        line += ",\"queued_bytes\":" + std::to_string(audit.queued_bytes);
        line += ",\"runtime_queued_bytes\":" + std::to_string(runtime_queued_bytes);
        line += ",\"store_index_registry_bytes_conserved\":" +
                std::string(conserved ? "true" : "false");
        line += ",\"final_queue_empty\":" +
                std::string(audit.final_queue_empty ? "true" : "false");
        line += ",\"final_credit\":" + std::to_string(final_credit);
        line += ",\"final_pending_debt\":" +
                std::string(final_debt ? "true" : "false");
        line += ",\"enqueue_index_op_count\":" + std::to_string(enqueue_count);
        line += ",\"enqueue_index_op_total_ns\":" + std::to_string(enqueue_total_ns);
        line += ",\"enqueue_index_op_mean_ns\":" + std::to_string(
                enqueue_count ? enqueue_total_ns / enqueue_count : 0);
        line += ",\"enqueue_index_op_max_ns\":" + std::to_string(enqueue_max_ns);
        line += ",\"dequeue_index_op_count\":" + std::to_string(dequeue_count);
        line += ",\"dequeue_index_op_total_ns\":" + std::to_string(dequeue_total_ns);
        line += ",\"dequeue_index_op_mean_ns\":" + std::to_string(
                dequeue_count ? dequeue_total_ns / dequeue_count : 0);
        line += ",\"dequeue_index_op_max_ns\":" + std::to_string(dequeue_max_ns);
        line += ",\"physical_system_reexecuted\":true";
        line += ",\"performance_claim\":false}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }

    void write_queue_overhead_summary(
            bool priority_on,
            bool heap_on,
            ExpertAsyncPriorityMode scheduler_mode) {
        const ExpertQueueOverheadSnapshot snapshot =
                queue_overhead_observer->snapshot();
        if (snapshot.mode == ExpertQueueOverheadMode::Off) {
            return;
        }

        const uint64_t clock_regression_count =
                snapshot.clock_self_check_ns.regression_count +
                queue_overhead_cell_regression_count(snapshot.global);
        const uint64_t clock_equality_count =
                snapshot.clock_self_check_ns.zero_count +
                queue_overhead_cell_zero_count(snapshot.global);
        const uint64_t overflow_count =
                snapshot.clock_self_check_ns.overflow_count +
                queue_overhead_cell_overflow_count(snapshot.global);
        const uint64_t event_ts_ns = llm_mem_trace_time_ns();

        std::string line;
        line.reserve(32768);
        line += "{\"event\":\"EXPERT_QUEUE_OVERHEAD_SUMMARY\",\"ts_ns\":" +
                std::to_string(event_ts_ns);
        line += ",\"schema_version\":\"m6b2.1-queue-overhead-v1\"";
        line += ",\"mode\":";
        json_escape_append(line, expert_queue_overhead_mode_name(snapshot.mode));
        line += ",\"semantics\":\"direct_queue_selection_measurement\"";
        line += ",\"clock\":{\"name\":\"llm_mem_trace_time_ns\"";
        line += ",\"semantics\":\"project_monotonic_clock\"";
        line += ",\"self_check\":";
        append_queue_overhead_aggregate(line, snapshot.clock_self_check_ns);
        line += "}";
        line += ",\"workers\":" + std::to_string(snapshot.workers);
        line += ",\"scheduler_batch\":" + std::to_string(snapshot.scheduler_batch);
        line += ",\"priority_enabled\":" +
                std::string(priority_on ? "true" : "false");
        line += ",\"priority_mode\":";
        json_escape_append(line, expert_prefetch_async_priority_mode_name(scheduler_mode));
        line += ",\"priority_heap_enabled\":" +
                std::string(heap_on ? "true" : "false");
        line += ",\"selection_count\":" + std::to_string(snapshot.selection_count);
        line += ",\"batch_count\":" + std::to_string(snapshot.batch_count);
        line += ",\"condition_wait_count\":" +
                std::to_string(snapshot.condition_wait_count);
        line += ",\"condition_reacquire_count\":" +
                std::to_string(snapshot.condition_reacquire_count);
        line += ",\"spurious_or_repeat_wake_count\":" +
                std::to_string(snapshot.repeat_wake_count);
        line += ",\"clock_read_count\":" +
                std::to_string(snapshot.clock_read_count + 1);
        line += ",\"clock_regression_count\":" +
                std::to_string(clock_regression_count);
        line += ",\"clock_equality_count\":" +
                std::to_string(clock_equality_count);
        line += ",\"overflow_count\":" + std::to_string(overflow_count);
        line += ",\"detail_event_count\":" +
                std::to_string(snapshot.detail_event_count);
        line += ",\"idle_wait_exit_count\":" +
                std::to_string(snapshot.idle_wait_exit_count);
        line += ",\"unsubmitted_lock_clock_read_count\":" +
                std::to_string(snapshot.unsubmitted_lock_clock_read_count);
        line += ",\"next_decision_id\":" +
                std::to_string(snapshot.next_decision_id);
        line += ",\"next_batch_id\":" + std::to_string(snapshot.next_batch_id);
        line += ",\"global\":";
        append_queue_overhead_cell(line, snapshot.global);
        line += ",\"by_priority_mode_phase\":[";
        bool needs_comma = false;
        for (size_t mode_index = 0;
                mode_index < ExpertQueueOverheadSnapshot::kPriorityModeCount;
                ++mode_index) {
            for (size_t phase_index = 0;
                    phase_index < ExpertQueueOverheadSnapshot::kPhaseCount;
                    ++phase_index) {
                const size_t index =
                        mode_index * ExpertQueueOverheadSnapshot::kPhaseCount +
                        phase_index;
                const ExpertQueueOverheadCell & cell = snapshot.cells[index];
                if (!queue_overhead_cell_has_samples(cell)) {
                    continue;
                }
                if (needs_comma) {
                    line += ',';
                }
                needs_comma = true;
                line += "{\"priority_mode\":";
                json_escape_append(line, expert_prefetch_async_priority_mode_name(
                        static_cast<ExpertAsyncPriorityMode>(mode_index)));
                line += ",\"phase\":";
                json_escape_append(line, phase_name(static_cast<int>(phase_index)));
                line += ",\"aggregates\":";
                append_queue_overhead_cell(line, cell);
                line += "}";
            }
        }
        line += "]";
        line += ",\"physical_load_observed\":false}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }

    void write_max_wait_summary() {
        ExpertMaxWaitConfig config;
        uint64_t eligible = 0;
        uint64_t eligible_decisions = 0;
        uint64_t selected_protected = 0;
        uint64_t selected_urgent = 0;
        uint64_t selected_normal = 0;
        uint64_t still_waiting = 0;
        uint64_t still_waiting_max = 0;
        uint64_t protected_wait_count = 0;
        uint64_t protected_wait_total_ns = 0;
        uint64_t protected_wait_max_ns = 0;
        uint64_t overshoot_count = 0;
        uint64_t overshoot_total_ns = 0;
        uint64_t overshoot_max_ns = 0;
        uint64_t protected_over_normal = 0;
        uint64_t missing_deadline = 0;
        uint64_t missing_enqueue = 0;
        uint64_t enqueue_regression = 0;
        uint64_t selections = 0;
        {
            std::lock_guard<std::mutex> lock(mu);
            if (priority_mode != ExpertAsyncPriorityMode::MaxWaitProtection) {
                return;
            }
            config = max_wait_config;
            eligible = max_wait_eligible_count;
            eligible_decisions = max_wait_eligible_decisions;
            selected_protected = max_wait_selected_protected;
            selected_urgent = max_wait_selected_urgent;
            selected_normal = max_wait_selected_normal;
            still_waiting = max_wait_still_waiting;
            still_waiting_max = max_wait_still_waiting_max;
            protected_wait_count = max_wait_protected_wait_count;
            protected_wait_total_ns = max_wait_protected_wait_total_ns;
            protected_wait_max_ns = max_wait_protected_wait_max_ns;
            overshoot_count = max_wait_overshoot_count;
            overshoot_total_ns = max_wait_overshoot_total_ns;
            overshoot_max_ns = max_wait_overshoot_max_ns;
            protected_over_normal = max_wait_protected_over_normal;
            missing_deadline = max_wait_missing_deadline;
            missing_enqueue = max_wait_missing_enqueue_timestamp;
            enqueue_regression = max_wait_enqueue_time_regression;
            selections = max_wait_selection_count;
        }

        std::string line;
        line.reserve(768);
        line += "{\"event\":\"EXPERT_MAX_WAIT_SUMMARY\",\"ts_ns\":" +
                std::to_string(llm_mem_trace_time_ns());
        line += ",\"mode\":\"max_wait_protection\"";
        line += ",\"semantics\":\"queue_selection\"";
        line += ",\"physical_load_observed\":false";
        line += ",\"threshold_us\":" + std::to_string(config.threshold_us);
        line += ",\"threshold_ns\":" + std::to_string(config.threshold_ns);
        line += ",\"urgent_guard_us\":" + std::to_string(config.urgent_guard_us);
        line += ",\"urgent_guard_ns\":" + std::to_string(config.urgent_guard_ns);
        line += ",\"protection_eligible_count\":" + std::to_string(eligible);
        line += ",\"protection_eligible_semantics\":\"candidate_observations\"";
        line += ",\"protection_eligible_decisions\":" +
                std::to_string(eligible_decisions);
        line += ",\"protection_selected_count\":" + std::to_string(selected_protected);
        line += ",\"protection_still_waiting_count\":" + std::to_string(still_waiting);
        line += ",\"protection_still_waiting_max\":" + std::to_string(still_waiting_max);
        line += ",\"urgent_selected_count\":" + std::to_string(selected_urgent);
        line += ",\"normal_selected_count\":" + std::to_string(selected_normal);
        line += ",\"protected_wait_count\":" + std::to_string(protected_wait_count);
        line += ",\"protected_wait_total_ns\":" + std::to_string(protected_wait_total_ns);
        line += ",\"protected_wait_mean_ns\":" + std::to_string(
                protected_wait_count ? protected_wait_total_ns / protected_wait_count : 0);
        line += ",\"protected_wait_max_ns\":" + std::to_string(protected_wait_max_ns);
        line += ",\"threshold_overshoot_count\":" + std::to_string(overshoot_count);
        line += ",\"threshold_overshoot_total_ns\":" + std::to_string(overshoot_total_ns);
        line += ",\"threshold_overshoot_mean_ns\":" + std::to_string(
                overshoot_count ? overshoot_total_ns / overshoot_count : 0);
        line += ",\"threshold_overshoot_max_ns\":" + std::to_string(overshoot_max_ns);
        line += ",\"protected_over_normal_count\":" +
                std::to_string(protected_over_normal);
        line += ",\"missing_deadline_count\":" + std::to_string(missing_deadline);
        line += ",\"missing_enqueue_timestamp_count\":" +
                std::to_string(missing_enqueue);
        line += ",\"enqueue_time_regression_count\":" +
                std::to_string(enqueue_regression);
        line += ",\"selection_count\":" + std::to_string(selections);
        line += "}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }

    size_t queue_depth_unlocked() const {
        if (reserved_service_active) {
            return reserved_service_queue.size();
        }
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

llm_pressure_shadow::QueueSnapshot pressure_shadow_queue_snapshot() {
    ExpertHintQueue & queue = expert_hint_queue();
    llm_pressure_shadow::QueueSnapshot snapshot;
    snapshot.configured_worker_count = expert_prefetch_async_workers();
    std::lock_guard<std::mutex> lock(queue.mu);
    snapshot.started = queue.started;
    snapshot.stopping = queue.stopping;
    snapshot.status = !queue.started ?
            llm_pressure_shadow::Status::NotStarted :
            (queue.stopping ?
             llm_pressure_shadow::Status::Stopping :
             llm_pressure_shadow::Status::Available);
    if (queue.started) {
        snapshot.queue_depth = queue.queue_depth_unlocked();
        snapshot.queued_bytes = queue.queued_bytes;
        snapshot.worker_count = queue.worker_count;
        snapshot.busy_workers = std::min<uint64_t>(
                queue.busy_workers.load(std::memory_order_relaxed), queue.worker_count);
    }
    return snapshot;
}

struct PressureShadowQueueProviderRegistration {
    PressureShadowQueueProviderRegistration() {
        llm_pressure_shadow::set_queue_snapshot_provider(pressure_shadow_queue_snapshot);
    }
};

PressureShadowQueueProviderRegistration pressure_shadow_queue_provider_registration;

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
    if (expert_task_trace_mode() != ExpertTaskTraceMode::Off || expert_shadow_enabled()) {
        task.lifecycle.step = task.step;
        task.lifecycle.layer = task.layer;
        task.lifecycle.expert = task.expert;
        task.lifecycle.phase = task.phase;
        task.lifecycle.stage = task.stage;
        task.lifecycle.tensor_name = task.tensor_name;
        task.lifecycle.addr = task.addr;
        task.lifecycle.nbytes = task.nbytes;
        task.lifecycle.score = task.route_score;
        if (expert_task_trace_mode() != ExpertTaskTraceMode::Off) {
            ensure_expert_task_summary_registered();
            ensure_expert_first_use_summary_registered();
        }
    }
    if (expert_shadow_summary_requested()) {
        ensure_expert_shadow_summary_registered();
    }
    if (expert_task_detail_events_enabled() || expert_shadow_enabled()) {
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
    if ((!expert_shadow_enabled() && expert_task_trace_mode() == ExpertTaskTraceMode::Off) || !operation ||
            operation->op != GGML_OP_MUL_MAT_ID) {
        return;
    }
    if (expert_task_trace_mode() != ExpertTaskTraceMode::Off) {
        ensure_expert_task_summary_registered();
        ensure_expert_first_use_summary_registered();
    }
    if (expert_shadow_summary_requested()) {
        ensure_expert_shadow_summary_registered();
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
        if (expert_shadow_enabled()) {
            ExpertShadowFirstUseInput shadow_use;
            shadow_use.step = use.step;
            shadow_use.layer = use.layer;
            shadow_use.expert = use.expert;
            shadow_use.phase = use.phase;
            shadow_use.stage = use.stage;
            shadow_use.tensor = use.tensor;
            shadow_use.addr = use.addr;
            shadow_use.nbytes = use.nbytes;
            shadow_use.first_use_ts_ns = use.first_use_ts_ns;
            write_expert_shadow_observations(
                    expert_shadow_slack().observe_first_use(std::move(shadow_use)));
        }
        if (expert_task_trace_mode() != ExpertTaskTraceMode::Off) {
            write_expert_first_use_event(
                    expert_first_use_matcher().observe_first_use(std::move(use)));
        }
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
