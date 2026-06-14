#include "trace_event.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
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
        uint64_t file_offset = 0) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }

    char addr_buf[32];
    std::snprintf(addr_buf, sizeof(addr_buf), "0x%llx", (unsigned long long) addr);

    std::string line;
    line.reserve(256);
    line += "{\"event\":\"OS_HINT\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
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
        size_t nbytes) {
#ifdef __linux__
    uintptr_t start = 0;
    size_t len = 0;
    if (!page_aligned_range(addr, nbytes, start, len)) {
        return;
    }
    errno = 0;
    const int rc = madvise(reinterpret_cast<void *>(start), len, advice);
    const int err = rc == 0 ? 0 : errno;
    write_os_hint_event(action, trigger, tensor_name, layer, expert, addr, nbytes, len, rc, err);
#else
    (void) action; (void) advice; (void) trigger; (void) tensor_name; (void) layer; (void) expert; (void) addr; (void) nbytes;
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
        size_t nbytes) {
#ifdef __linux__
    FileMapping mapping;
    if (!find_file_mapping(addr, mapping)) {
        write_os_hint_event(action, trigger, tensor_name, layer, expert, addr, nbytes, 0, -1, ENOENT);
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
        write_os_hint_event(action, trigger, tensor_name, layer, expert, addr, nbytes, advise_len, -1, errno, file_offset);
        return;
    }
    const int rc = posix_fadvise(fd, (off_t) file_offset, (off_t) advise_len, POSIX_FADV_WILLNEED);
    close(fd);
    write_os_hint_event(action, trigger, tensor_name, layer, expert, addr, nbytes, advise_len, rc == 0 ? 0 : -1, rc, file_offset);
#else
    (void) action; (void) trigger; (void) tensor_name; (void) layer; (void) expert; (void) addr; (void) nbytes;
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

    bool mark_hinted(uint64_t step, int layer, int expert, uintptr_t addr) {
        std::string key = std::to_string(step) + ":" + std::to_string(layer) + ":" +
                          std::to_string(expert) + ":" + std::to_string((uint64_t) addr);
        std::lock_guard<std::mutex> lock(mu);
        return hinted.insert(std::move(key)).second;
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

extern "C" void llm_mem_trace_prefetch_expert_layer(int layer, const int * experts, int n_experts, const char * reason) {
    if (!os_hints_enabled() || !os_hint_opt_enabled("LLM_MEM_TRACE_OPT_EXPERT_PREFETCH") ||
            layer < 0 || !experts || n_experts <= 0) {
        return;
    }

    const std::vector<ExpertTensorInfo> tensors = expert_tensor_registry().for_layer(layer);
    if (tensors.empty()) {
        return;
    }

    const uint64_t step = llm_mem_trace_get_step();
    for (int i = 0; i < n_experts; ++i) {
        const int expert = experts[i];
        if (expert < 0) {
            continue;
        }
        for (const ExpertTensorInfo & info : tensors) {
            uintptr_t slice_addr = 0;
            size_t slice_bytes = 0;
            if (!expert_slice_range(info, expert, slice_addr, slice_bytes)) {
                continue;
            }
            if (!os_hint_size_allowed(slice_bytes)) {
                continue;
            }
            if (!expert_tensor_registry().mark_hinted(step, layer, expert, info.addr)) {
                continue;
            }
#ifdef __linux__
            apply_madvise_hint("expert_madvise_willneed", MADV_WILLNEED,
                               reason ? reason : "expert_prefetch",
                               info.name.c_str(), layer, expert, slice_addr, slice_bytes);
#endif
            if (os_hint_opt_enabled("LLM_MEM_TRACE_OPT_POSIX_FADVISE")) {
                apply_posix_fadvise_hint("expert_posix_fadvise_willneed",
                                         reason ? reason : "expert_prefetch",
                                         info.name.c_str(), layer, expert, slice_addr, slice_bytes);
            }
        }
    }
}
