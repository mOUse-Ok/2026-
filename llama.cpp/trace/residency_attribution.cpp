#include "residency_attribution.h"

#include "trace_event.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr const char * kEvent = "RESIDENCY_DEMAND";
constexpr const char * kSummaryEvent = "RESIDENCY_ATTRIBUTION_SUMMARY";

bool env_truthy(const char * value) {
    return value && !(value[0] == '0' && value[1] == '\0');
}

bool enabled() {
    static const bool requested = env_truthy(
            std::getenv("LLM_MEM_TRACE_RESIDENCY_ATTRIBUTION"));
    return requested && llm_mem_trace_enabled() &&
            llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_TENSOR);
}

const char * phase_name(int phase) {
    switch (phase) {
        case LLM_MEM_TRACE_PHASE_PREFILL: return "PREFILL";
        case LLM_MEM_TRACE_PHASE_DECODE: return "DECODE";
        default: return "UNKNOWN";
    }
}

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

size_t max_pages() {
    static const size_t value = [] {
        const char * text = std::getenv("LLM_MEM_TRACE_RESIDENCY_MAX_PAGES");
        if (!text || !text[0]) {
            return size_t{4096};
        }
        char * end = nullptr;
        const unsigned long long parsed = std::strtoull(text, &end, 10);
        return end && *end == '\0' && parsed > 0 ? (size_t) parsed : size_t{4096};
    }();
    return value;
}

struct FileMapping {
    uintptr_t start = 0;
    uintptr_t end = 0;
    uint64_t offset = 0;
    uint64_t inode = 0;
    std::string path;
};

struct MappingCache {
    std::mutex mu;
    bool loaded = false;
    std::vector<FileMapping> mappings;

    bool find(uintptr_t address, FileMapping & result) {
#ifdef __linux__
        std::lock_guard<std::mutex> lock(mu);
        if (!loaded) {
            loaded = true;
            FILE * fp = std::fopen("/proc/self/maps", "r");
            if (fp) {
                char line[4096];
                while (std::fgets(line, sizeof(line), fp)) {
                    unsigned long long start = 0;
                    unsigned long long end = 0;
                    unsigned long long offset = 0;
                    unsigned long long inode = 0;
                    char perms[8] = {};
                    int path_pos = 0;
                    const int scanned = std::sscanf(
                            line, "%llx-%llx %7s %llx %*s %llu %n",
                            &start, &end, perms, &offset, &inode, &path_pos);
                    if (scanned < 5 || path_pos <= 0 ||
                            std::strchr(perms, 'r') == nullptr ||
                            line[path_pos] == '\0') {
                        continue;
                    }
                    char * path = line + path_pos;
                    size_t path_len = std::strlen(path);
                    while (path_len > 0 &&
                            (path[path_len - 1] == '\n' || path[path_len - 1] == '\r')) {
                        path[--path_len] = '\0';
                    }
                    if (path_len == 0 || path[0] == '[') {
                        continue;
                    }
                    FileMapping mapping;
                    mapping.start = (uintptr_t) start;
                    mapping.end = (uintptr_t) end;
                    mapping.offset = (uint64_t) offset;
                    mapping.inode = (uint64_t) inode;
                    mapping.path = path;
                    if (mapping.inode == 0) {
                        struct stat st = {};
                        if (stat(mapping.path.c_str(), &st) == 0) {
                            mapping.inode = (uint64_t) st.st_ino;
                        }
                    }
                    mappings.push_back(std::move(mapping));
                }
                std::fclose(fp);
            }
        }
        for (const FileMapping & mapping : mappings) {
            if (address >= mapping.start && address < mapping.end) {
                result = mapping;
                return true;
            }
        }
#else
        (void) address;
        (void) result;
#endif
        return false;
    }
};

MappingCache & mapping_cache() {
    static MappingCache cache;
    return cache;
}

struct ResidencySample {
    bool available = false;
    bool exact = false;
    int error = 0;
    uint64_t page_size = 0;
    uint64_t page_count = 0;
    uint64_t sampled_pages = 0;
    uint64_t resident_pages = 0;
    uint64_t resident_bytes = 0;
    uint64_t nonresident_bytes = 0;
    uintptr_t first_page = 0;
    std::vector<unsigned char> page_resident;
};

#ifdef __linux__
ResidencySample query_residency(uintptr_t address, size_t nbytes) {
    ResidencySample result;
    if (address == 0 || nbytes == 0) {
        return result;
    }
    const long system_page_size = sysconf(_SC_PAGESIZE);
    if (system_page_size <= 0) {
        return result;
    }
    const uintptr_t page_size = (uintptr_t) system_page_size;
    const uintptr_t last = address + nbytes - 1;
    if (last < address) {
        return result;
    }
    const uintptr_t first_page = address & ~(page_size - 1);
    const uintptr_t end = (last & ~(page_size - 1)) + page_size;
    const uint64_t page_count = (uint64_t) ((end - first_page) / page_size);
    if (page_count == 0) {
        return result;
    }

    result.available = true;
    result.page_size = (uint64_t) page_size;
    result.page_count = page_count;
    result.first_page = first_page;

    auto add_exact_bytes = [&](const std::vector<unsigned char> & states) {
        uint64_t resident_bytes = 0;
        for (uint64_t index = 0; index < page_count; ++index) {
            const uintptr_t page_begin = first_page + index * page_size;
            const uintptr_t begin = std::max(address, page_begin);
            const uintptr_t finish = std::min(end, page_begin + page_size);
            const uint64_t overlap = finish > begin ? (uint64_t) (finish - begin) : 0;
            if (states[(size_t) index] & 1u) {
                resident_bytes += overlap;
            }
        }
        result.resident_bytes = resident_bytes;
        result.nonresident_bytes = (uint64_t) nbytes > resident_bytes ?
                (uint64_t) nbytes - resident_bytes : 0;
    };

    if (page_count <= max_pages()) {
        result.page_resident.resize((size_t) page_count);
        if (mincore(reinterpret_cast<void *>(first_page), (size_t) (end - first_page),
                    result.page_resident.data()) != 0) {
            result.error = errno;
            result.page_resident.clear();
            return result;
        }
        result.exact = true;
        result.sampled_pages = page_count;
        for (unsigned char state : result.page_resident) {
            result.resident_pages += (state & 1u) ? 1u : 0u;
        }
        add_exact_bytes(result.page_resident);
        return result;
    }

    const uint64_t samples = (uint64_t) std::max<size_t>(1, max_pages());
    uint64_t resident = 0;
    for (uint64_t index = 0; index < samples; ++index) {
        const uint64_t page_index = samples == 1 ? 0 :
                (index * (page_count - 1)) / (samples - 1);
        unsigned char state = 0;
        if (mincore(reinterpret_cast<void *>(first_page + page_index * page_size),
                    (size_t) page_size, &state) != 0) {
            result.error = errno;
            return result;
        }
        resident += (state & 1u) ? 1u : 0u;
    }
    result.exact = false;
    result.sampled_pages = samples;
    result.resident_pages = (resident * page_count + samples / 2) / samples;
    result.resident_bytes = ((uint64_t) nbytes * result.resident_pages + page_count / 2) /
            page_count;
    result.nonresident_bytes = (uint64_t) nbytes > result.resident_bytes ?
            (uint64_t) nbytes - result.resident_bytes : 0;
    return result;
}
#else
ResidencySample query_residency(uintptr_t address, size_t nbytes) {
    (void) address;
    (void) nbytes;
    return {};
}
#endif

struct PageKey {
    uint64_t inode = 0;
    uint64_t offset = 0;

    bool operator==(const PageKey & other) const {
        return inode == other.inode && offset == other.offset;
    }
};

struct PageKeyHash {
    size_t operator()(const PageKey & key) const {
        const uint64_t mixed = key.inode ^ (key.offset + 0x9e3779b97f4a7c15ull +
                (key.inode << 6) + (key.inode >> 2));
        return (size_t) (mixed ^ (mixed >> 32));
    }
};

struct DemandKey {
    int phase = 0;
    uint64_t step = 0;
    uintptr_t address = 0;

    bool operator==(const DemandKey & other) const {
        return phase == other.phase && step == other.step &&
                address == other.address;
    }
};

struct DemandKeyHash {
    size_t operator()(const DemandKey & key) const {
        const uint64_t mixed = (uint64_t) key.address ^
                (key.step + 0x9e3779b97f4a7c15ull + (key.step << 6) +
                 (key.step >> 2)) ^
                ((uint64_t) (uint32_t) key.phase << 32);
        return (size_t) (mixed ^ (mixed >> 32));
    }
};

struct AttributionStats {
    uint64_t demand_events = 0;
    uint64_t demanded_bytes = 0;
    uint64_t resident_before_use_bytes = 0;
    uint64_t nonresident_before_use_bytes = 0;
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    uint64_t nonresident_pages = 0;
    uint64_t unique_demand_pages = 0;
    uint64_t unique_missing_pages = 0;
    uint64_t exact_events = 0;
    uint64_t sampled_events = 0;
    uint64_t mapping_missing_events = 0;
};

struct AttributionTracker {
    std::mutex mu;
    std::unordered_set<DemandKey, DemandKeyHash> observed_demands;
    std::unordered_map<std::string, AttributionStats> stats[4];
    std::unordered_set<PageKey, PageKeyHash> unique_demand_pages[4];
    std::unordered_map<PageKey, std::string, PageKeyHash> unique_missing_owner[4];
    uint64_t observations = 0;
    uint64_t residency_errors = 0;

    bool claim_demand(int phase, uint64_t step, uintptr_t address) {
        std::lock_guard<std::mutex> lock(mu);
        return observed_demands.insert({phase, step, address}).second;
    }

    static int phase_index(int phase) {
        return phase == LLM_MEM_TRACE_PHASE_PREFILL ? 0 :
                phase == LLM_MEM_TRACE_PHASE_DECODE ? 1 : 2;
    }

    void account(
            int phase,
            const char * object_class,
            const ResidencySample & sample,
            const FileMapping * mapping,
            size_t nbytes) {
        const std::string klass = object_class && object_class[0] ? object_class : "Other";
        std::lock_guard<std::mutex> lock(mu);
        ++observations;
        for (int bucket : {phase_index(phase), 3}) {
            AttributionStats & current = stats[bucket][klass];
            ++current.demand_events;
            current.demanded_bytes += (uint64_t) nbytes;
            current.resident_before_use_bytes += sample.resident_bytes;
            current.nonresident_before_use_bytes += sample.nonresident_bytes;
            current.total_pages += sample.page_count;
            current.resident_pages += sample.resident_pages;
            current.nonresident_pages += sample.page_count > sample.resident_pages ?
                    sample.page_count - sample.resident_pages : 0;
            if (sample.exact) {
                ++current.exact_events;
            } else {
                ++current.sampled_events;
            }
            if (!mapping) {
                ++current.mapping_missing_events;
            }
            if (!mapping || !sample.exact) {
                continue;
            }
            for (uint64_t index = 0; index < sample.page_count; ++index) {
                const uint64_t page_offset = mapping->offset +
                        (uint64_t) (sample.first_page - mapping->start) +
                        index * sample.page_size;
                const PageKey key{mapping->inode, page_offset};
                if (unique_demand_pages[bucket].insert(key).second) {
                    ++current.unique_demand_pages;
                }
                const bool is_resident =
                        (sample.page_resident[(size_t) index] & 1u) != 0;
                if (!is_resident &&
                        unique_missing_owner[bucket].emplace(key, klass).second) {
                    ++current.unique_missing_pages;
                }
            }
        }
    }

    static void append_stats(std::string & line, const AttributionStats & value) {
        line += "{\"demand_events\":" + std::to_string(value.demand_events);
        line += ",\"demanded_bytes\":" + std::to_string(value.demanded_bytes);
        line += ",\"resident_before_use_bytes\":" +
                std::to_string(value.resident_before_use_bytes);
        line += ",\"nonresident_before_use_bytes\":" +
                std::to_string(value.nonresident_before_use_bytes);
        line += ",\"total_pages\":" + std::to_string(value.total_pages);
        line += ",\"resident_pages\":" + std::to_string(value.resident_pages);
        line += ",\"nonresident_pages\":" + std::to_string(value.nonresident_pages);
        line += ",\"unique_demand_pages\":" +
                std::to_string(value.unique_demand_pages);
        line += ",\"unique_missing_pages\":" +
                std::to_string(value.unique_missing_pages);
        line += ",\"exact_events\":" + std::to_string(value.exact_events);
        line += ",\"sampled_events\":" + std::to_string(value.sampled_events);
        line += ",\"mapping_missing_events\":" +
                std::to_string(value.mapping_missing_events);
        line += "}";
    }

    void write_summary() {
        if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu);
        std::string line;
        line.reserve(8192);
        line += "{\"event\":\"";
        line += kSummaryEvent;
        line += "\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
        line += ",\"semantics\":\"mincore_before_logical_demand\"";
        line += ",\"unique_page_semantics\":\"first_observed_missing_page\"";
        bool complete = true;
        for (int bucket = 0; bucket < 3; ++bucket) {
            for (const auto & entry : stats[bucket]) {
                complete = complete && entry.second.sampled_events == 0 &&
                        entry.second.mapping_missing_events == 0;
            }
        }
        line += ",\"unique_pages_complete\":";
        line += complete ? "true" : "false";
        line += ",\"observations\":" + std::to_string(observations);
        line += ",\"residency_errors\":" + std::to_string(residency_errors);
        line += ",\"by_phase\":{";
        bool first_phase = true;
        const char * names[] = {"PREFILL", "DECODE", "UNKNOWN", "ALL"};
        for (int bucket = 0; bucket < 4; ++bucket) {
            if (!first_phase) {
                line += ",";
            }
            first_phase = false;
            json_escape_append(line, names[bucket]);
            line += ":{";
            bool first_class = true;
            for (const auto & entry : stats[bucket]) {
                if (!first_class) {
                    line += ",";
                }
                first_class = false;
                json_escape_append(line, entry.first.c_str());
                line += ":";
                append_stats(line, entry.second);
            }
            line += "}";
        }
        line += "}}";
        llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
    }
};

AttributionTracker & tracker() {
    static AttributionTracker result;
    return result;
}

void write_summary() {
    tracker().write_summary();
}

void ensure_summary_registered() {
    // Construct the tracker before registering the atexit callback.  Function
    // static destruction is registered after this callback, so this keeps the
    // accumulated maps alive until write_summary() has emitted them.
    (void) tracker();
    static const bool registered = [] {
        std::atexit(write_summary);
        return true;
    }();
    (void) registered;
}

} // namespace

bool llm_mem_trace_residency_attribution_enabled() {
    return enabled();
}

void llm_mem_trace_residency_attribution_observe(
        const char * object_class,
        const char * tensor_name,
        const char * tensor_subclass,
        int layer,
        int expert_id,
        uintptr_t virtual_address,
        size_t tensor_bytes) {
    if (!enabled() || virtual_address == 0 || tensor_bytes == 0) {
        return;
    }
    ensure_summary_registered();

    FileMapping mapping;
    const int phase = llm_mem_trace_get_phase();
    if (!tracker().claim_demand(phase, llm_mem_trace_get_step(), virtual_address)) {
        return;
    }
    const bool mapped = mapping_cache().find(virtual_address, mapping);
    const ResidencySample sample = query_residency(virtual_address, tensor_bytes);

    std::string line;
    line.reserve(640);
    line += "{\"event\":\"";
    line += kEvent;
    line += "\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
    line += ",\"phase\":";
    json_escape_append(line, phase_name(phase));
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    line += ",\"layer\":" + std::to_string(layer);
    line += ",\"expert_id\":" + std::to_string(expert_id);
    line += ",\"object_class\":";
    json_escape_append(line, object_class && object_class[0] ? object_class : "Other");
    line += ",\"tensor\":";
    json_escape_append(line, tensor_name ? tensor_name : "");
    line += ",\"tensor_subclass\":";
    json_escape_append(line, tensor_subclass ? tensor_subclass : "");
    char address_buffer[32];
    std::snprintf(address_buffer, sizeof(address_buffer), "0x%" PRIxPTR, virtual_address);
    line += ",\"virtual_address\":";
    json_escape_append(line, address_buffer);
    line += ",\"tensor_bytes\":" + std::to_string(tensor_bytes);
    line += ",\"file_backed\":" + std::string(mapped ? "true" : "false");
    if (mapped) {
        line += ",\"file_inode\":" + std::to_string(mapping.inode);
        line += ",\"file_offset\":" + std::to_string(mapping.offset +
                (uint64_t) (virtual_address - mapping.start));
    }
    line += ",\"residency_available\":" +
            std::string(sample.available ? "true" : "false");
    if (sample.error != 0) {
        line += ",\"residency_error\":" + std::to_string(sample.error);
    }
    if (sample.available) {
        line += ",\"page_size\":" + std::to_string(sample.page_size);
        line += ",\"total_pages\":" + std::to_string(sample.page_count);
        line += ",\"resident_pages\":" + std::to_string(sample.resident_pages);
        line += ",\"nonresident_pages\":" + std::to_string(
                sample.page_count > sample.resident_pages ?
                sample.page_count - sample.resident_pages : 0);
        line += ",\"resident_sample_pages\":" + std::to_string(sample.sampled_pages);
        line += ",\"resident_bytes\":" + std::to_string(sample.resident_bytes);
        line += ",\"nonresident_bytes\":" + std::to_string(sample.nonresident_bytes);
        line += ",\"resident_ratio\":" + std::to_string(
                tensor_bytes > 0 ? (double) sample.resident_bytes / tensor_bytes : 0.0);
        line += ",\"resident_exact\":" +
                std::string(sample.exact ? "true" : "false");
    }
    line += "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_TENSOR, line.c_str(), line.size());

    if (!sample.available || sample.error != 0) {
        std::lock_guard<std::mutex> lock(tracker().mu);
        ++tracker().residency_errors;
        return;
    }
    tracker().account(phase, object_class, sample, mapped ? &mapping : nullptr, tensor_bytes);
}
