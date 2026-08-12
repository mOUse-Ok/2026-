#include "trace_event.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

namespace {

const char * phase_name(int phase) {
    switch (phase) {
        case LLM_MEM_TRACE_PHASE_PREFILL: return "PREFILL";
        case LLM_MEM_TRACE_PHASE_DECODE:  return "DECODE";
        default: return "UNKNOWN";
    }
}

struct StatData {
    uint64_t minflt = 0;
    uint64_t majflt = 0;
    uint64_t vsize = 0;
    uint64_t rss_pages = 0;
};

#ifdef __linux__

struct SmapsRollupData {
    bool ok = false;
    uint64_t rss_kb = 0;
    uint64_t pss_kb = 0;
    uint64_t shared_clean_kb = 0;
    uint64_t shared_dirty_kb = 0;
    uint64_t private_clean_kb = 0;
    uint64_t private_dirty_kb = 0;
    uint64_t referenced_kb = 0;
    uint64_t anonymous_kb = 0;
    uint64_t swap_kb = 0;
};

std::string current_cgroup_v2_dir() {
    FILE * fp = std::fopen("/proc/self/cgroup", "r");
    if (!fp) {
        return {};
    }
    char line[4096];
    std::string relative;
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "0::", 3) == 0) {
            relative = line + 3;
            while (!relative.empty() &&
                    (relative.back() == '\n' || relative.back() == '\r')) {
                relative.pop_back();
            }
            break;
        }
    }
    std::fclose(fp);
    if (relative.empty() || relative == "/") {
        return "/sys/fs/cgroup";
    }
    return "/sys/fs/cgroup/" + (relative.front() == '/' ? relative.substr(1) : relative);
}

bool read_cgroup_workingset_refault(uint64_t & value) {
    const std::string cgroup_dir = current_cgroup_v2_dir();
    if (cgroup_dir.empty()) {
        return false;
    }
    const std::string path = cgroup_dir + "/memory.stat";
    FILE * fp = std::fopen(path.c_str(), "r");
    if (!fp) {
        return false;
    }
    char line[256];
    uint64_t total = 0;
    bool found = false;
    while (std::fgets(line, sizeof(line), fp)) {
        const bool is_anon = std::strncmp(line, "workingset_refault_anon ", 24) == 0;
        const bool is_file = std::strncmp(line, "workingset_refault_file ", 24) == 0;
        if (!is_anon && !is_file) {
            continue;
        }
        char * end = nullptr;
        const unsigned long long parsed = std::strtoull(line + 24, &end, 10);
        if (end != line + 24) {
            total += (uint64_t) parsed;
            found = true;
        }
    }
    std::fclose(fp);
    if (found) {
        value = total;
    }
    return found;
}

bool env_truthy(const char * value) {
    if (!value) {
        return false;
    }
    return !(value[0] == '0' && value[1] == '\0');
}

bool smaps_rollup_enabled() {
    static const bool enabled = env_truthy(std::getenv("LLM_MEM_TRACE_SMAPS"));
    return enabled;
}

bool read_proc_stat(StatData & out) {
    FILE * fp = std::fopen("/proc/self/stat", "r");
    if (!fp) {
        return false;
    }

    char buf[4096];
    if (!std::fgets(buf, sizeof(buf), fp)) {
        std::fclose(fp);
        return false;
    }
    std::fclose(fp);

    char * end = std::strrchr(buf, ')');
    if (!end) {
        return false;
    }

    char * p = end + 2; // skip ") "
    int field = 3;
    uint64_t value = 0;

    while (*p) {
        while (*p == ' ') {
            ++p;
        }
        if (!*p) {
            break;
        }

        char * next = p;
        value = std::strtoull(p, &next, 10);

        // Non-numeric field (e.g. state char 'S' at field 3) —
        // skip to the next space-delimited token and continue counting
        if (next == p) {
            while (*p && *p != ' ') { ++p; }
            ++field;
            continue;
        }

        if (field == 10) {
            out.minflt = value;
        } else if (field == 12) {
            out.majflt = value;
        } else if (field == 23) {
            out.vsize = value;
        } else if (field == 24) {
            out.rss_pages = value;
            break;  // got everything we need
        }

        p = next;
        ++field;
    }

    return true;
}

bool parse_kb_line(const char * line, const char * key, uint64_t & out) {
    const size_t key_len = std::strlen(key);
    if (std::strncmp(line, key, key_len) != 0) {
        return false;
    }
    const char * p = line + key_len;
    while (*p == ' ' || *p == '\t') {
        ++p;
    }
    char * end = nullptr;
    const uint64_t val = std::strtoull(p, &end, 10);
    if (end == p) {
        return false;
    }
    out = val;
    return true;
}

SmapsRollupData read_smaps_rollup() {
    SmapsRollupData data;
    if (!smaps_rollup_enabled()) {
        return data;
    }

    FILE * fp = std::fopen("/proc/self/smaps_rollup", "r");
    if (!fp) {
        return data;
    }

    char line[512];
    while (std::fgets(line, sizeof(line), fp)) {
        parse_kb_line(line, "Rss:", data.rss_kb) ||
        parse_kb_line(line, "Pss:", data.pss_kb) ||
        parse_kb_line(line, "Shared_Clean:", data.shared_clean_kb) ||
        parse_kb_line(line, "Shared_Dirty:", data.shared_dirty_kb) ||
        parse_kb_line(line, "Private_Clean:", data.private_clean_kb) ||
        parse_kb_line(line, "Private_Dirty:", data.private_dirty_kb) ||
        parse_kb_line(line, "Referenced:", data.referenced_kb) ||
        parse_kb_line(line, "Anonymous:", data.anonymous_kb) ||
        parse_kb_line(line, "Swap:", data.swap_kb);
    }

    std::fclose(fp);
    data.ok = true;
    return data;
}

uint64_t count_maps() {
    FILE * fp = std::fopen("/proc/self/maps", "r");
    if (!fp) {
        return 0;
    }
    char line[512];
    uint64_t count = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        ++count;
    }
    std::fclose(fp);
    return count;
}

#endif  // __linux__

} // namespace

extern "C" void llm_mem_trace_memory_sample(const char * reason) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
        return;
    }

#ifdef __linux__
    static StatData prev = {};
    static bool have_prev = false;

    StatData cur = {};
    if (!read_proc_stat(cur)) {
        return;
    }

    const uint64_t page_size = (uint64_t) sysconf(_SC_PAGESIZE);
    const uint64_t rss_bytes = cur.rss_pages * page_size;
    const uint64_t vms_bytes = cur.vsize;

    const uint64_t minflt_delta = have_prev && cur.minflt >= prev.minflt ? cur.minflt - prev.minflt : 0;
    const uint64_t majflt_delta = have_prev && cur.majflt >= prev.majflt ? cur.majflt - prev.majflt : 0;

    prev = cur;
    have_prev = true;

    const uint64_t mmap_count = count_maps();
    const SmapsRollupData smaps = read_smaps_rollup();
    static uint64_t previous_cgroup_refault = 0;
    static bool have_previous_cgroup_refault = false;
    uint64_t cgroup_refault = 0;
    const bool have_cgroup_refault = read_cgroup_workingset_refault(cgroup_refault);
    const uint64_t cgroup_refault_delta = have_cgroup_refault &&
            have_previous_cgroup_refault && cgroup_refault >= previous_cgroup_refault ?
            cgroup_refault - previous_cgroup_refault : 0;
    if (have_cgroup_refault) {
        previous_cgroup_refault = cgroup_refault;
        have_previous_cgroup_refault = true;
    }

    std::string line;
    line.reserve(256);
    line += "{\"event\":\"MEMORY_STAT\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    if (reason) {
        line += ",\"reason\":\"" + std::string(reason) + "\"";
    }
    line += ",\"rss_bytes\":" + std::to_string(rss_bytes);
    line += ",\"vms_bytes\":" + std::to_string(vms_bytes);
    line += ",\"minor_faults\":" + std::to_string(cur.minflt);
    line += ",\"major_faults\":" + std::to_string(cur.majflt);
    line += ",\"minor_faults_delta\":" + std::to_string(minflt_delta);
    line += ",\"major_faults_delta\":" + std::to_string(majflt_delta);
    line += ",\"mmap_count\":" + std::to_string(mmap_count);
    if (have_cgroup_refault) {
        line += ",\"cgroup_workingset_refault\":" + std::to_string(cgroup_refault);
        line += ",\"cgroup_workingset_refault_delta\":" +
                std::to_string(cgroup_refault_delta);
        line += ",\"cgroup_workingset_refault_source\":\"cgroup_v2_memory.stat\"";
    }
    if (smaps.ok) {
        line += ",\"smaps_rss_bytes\":" + std::to_string(smaps.rss_kb * 1024ull);
        line += ",\"pss_bytes\":" + std::to_string(smaps.pss_kb * 1024ull);
        line += ",\"shared_clean_bytes\":" + std::to_string(smaps.shared_clean_kb * 1024ull);
        line += ",\"shared_dirty_bytes\":" + std::to_string(smaps.shared_dirty_kb * 1024ull);
        line += ",\"private_clean_bytes\":" + std::to_string(smaps.private_clean_kb * 1024ull);
        line += ",\"private_dirty_bytes\":" + std::to_string(smaps.private_dirty_kb * 1024ull);
        line += ",\"referenced_bytes\":" + std::to_string(smaps.referenced_kb * 1024ull);
        line += ",\"anonymous_bytes\":" + std::to_string(smaps.anonymous_kb * 1024ull);
        line += ",\"swap_bytes\":" + std::to_string(smaps.swap_kb * 1024ull);
    }
    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
#else
    // Non-Linux: emit minimal event — /proc/self/stat and /proc/self/maps
    // are not available; RSS / VMS / page-fault stats will be absent.
    std::string line;
    line.reserve(128);
    line += "{\"event\":\"MEMORY_STAT\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    if (reason) {
        line += ",\"reason\":\"" + std::string(reason) + "\"";
    }
    line += ",\"platform\":\"unsupported\"";
    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
#endif
}
