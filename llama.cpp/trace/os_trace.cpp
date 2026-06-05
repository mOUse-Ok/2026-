#include "trace_event.h"

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

    StatData cur = {};
    if (!read_proc_stat(cur)) {
        return;
    }

    const uint64_t page_size = (uint64_t) sysconf(_SC_PAGESIZE);
    const uint64_t rss_bytes = cur.rss_pages * page_size;
    const uint64_t vms_bytes = cur.vsize;

    const uint64_t minflt_delta = cur.minflt >= prev.minflt ? cur.minflt - prev.minflt : 0;
    const uint64_t majflt_delta = cur.majflt >= prev.majflt ? cur.majflt - prev.majflt : 0;

    prev = cur;

    const uint64_t mmap_count = count_maps();

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
