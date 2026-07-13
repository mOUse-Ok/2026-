#include "expert_tensor_registry.h"

#include "ggml.h"
#include "trace_event.h"

#include <algorithm>
#include <cstring>

namespace {

bool is_expert_weight_tensor_name(const char * name) {
    return name &&
           std::strstr(name, "blk.") &&
           std::strstr(name, "_exps.weight") &&
           (std::strstr(name, "ffn_gate_exps.weight") ||
            std::strstr(name, "ffn_up_exps.weight") ||
            std::strstr(name, "ffn_down_exps.weight") ||
            std::strstr(name, "ffn_gate_up_exps.weight"));
}

}

void ExpertTensorRegistry::add(const ggml_tensor * t, const char * name, int layer, uintptr_t addr, size_t nbytes) {
    if (!t || layer < 0 || addr == 0 || nbytes == 0 || !is_expert_weight_tensor_name(name)) return;
    const int64_t n_expert = t->ne[2];
    const size_t expert_stride = (size_t) t->nb[2];
    if (n_expert <= 0 || expert_stride == 0) return;
    std::lock_guard<std::mutex> lock(mu);
    for (const ExpertTensorInfo & info : tensors) if (info.addr == addr && info.nbytes == nbytes) return;
    tensors.push_back({name ? name : "", layer, addr, nbytes, n_expert, expert_stride});
}

std::vector<ExpertTensorInfo> ExpertTensorRegistry::for_layer(int layer) {
    std::vector<ExpertTensorInfo> out;
    std::lock_guard<std::mutex> lock(mu);
    for (const ExpertTensorInfo & info : tensors) if (info.layer == layer) out.push_back(info);
    return out;
}

bool ExpertTensorRegistry::was_hinted(uint64_t step, int layer, int expert, uintptr_t addr, uint64_t ttl_steps) {
    const std::string slice_key = std::to_string(layer) + ":" + std::to_string(expert) + ":" + std::to_string((uint64_t) addr);
    std::lock_guard<std::mutex> lock(mu);
    if (ttl_steps > 0) {
        auto it = recent_hints.find(slice_key);
        return it != recent_hints.end() && step >= it->second && step - it->second <= ttl_steps;
    }
    return hinted.find(std::to_string(step) + ":" + slice_key) != hinted.end();
}

bool ExpertTensorRegistry::mark_hinted(uint64_t step, int layer, int expert, uintptr_t addr, uint64_t ttl_steps) {
    const std::string slice_key = std::to_string(layer) + ":" + std::to_string(expert) + ":" + std::to_string((uint64_t) addr);
    std::lock_guard<std::mutex> lock(mu);
    route_hint_ttl_steps_config = std::max(route_hint_ttl_steps_config, ttl_steps);
    route_hint_candidates++;
    if (ttl_steps > 0) {
        auto it = recent_hints.find(slice_key);
        if (it != recent_hints.end() && step >= it->second && step - it->second <= ttl_steps) {
            route_hint_skipped++;
            if (step == it->second) route_hint_duplicate_skipped++; else route_hint_ttl_skipped++;
            return false;
        }
        recent_hints[slice_key] = step;
        route_hint_issued++;
        return true;
    }
    const bool inserted = hinted.insert(std::to_string(step) + ":" + slice_key).second;
    if (inserted) route_hint_issued++; else { route_hint_skipped++; route_hint_duplicate_skipped++; }
    return inserted;
}

void ExpertTensorRegistry::write_route_hint_summary() {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_MEMORY)) return;
    uint64_t ttl_steps, candidates, issued, skipped, duplicate_skipped, ttl_skipped;
    {
        std::lock_guard<std::mutex> lock(mu);
        candidates = route_hint_candidates;
        if (candidates == 0) return;
        ttl_steps = route_hint_ttl_steps_config; issued = route_hint_issued; skipped = route_hint_skipped;
        duplicate_skipped = route_hint_duplicate_skipped; ttl_skipped = route_hint_ttl_skipped;
    }
    std::string line;
    line.reserve(256);
    line += "{\"event\":\"EXPERT_ROUTE_HINT_SUMMARY\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
    line += ",\"ttl_steps\":" + std::to_string(ttl_steps);
    line += ",\"candidates\":" + std::to_string(candidates);
    line += ",\"issued\":" + std::to_string(issued);
    line += ",\"skipped\":" + std::to_string(skipped);
    line += ",\"duplicate_skipped\":" + std::to_string(duplicate_skipped);
    line += ",\"ttl_skipped\":" + std::to_string(ttl_skipped) + "}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, line.c_str(), line.size());
}

ExpertTensorRegistry & expert_tensor_registry() { static ExpertTensorRegistry registry; return registry; }

bool expert_slice_range(const ExpertTensorInfo & info, int expert, uintptr_t & addr, size_t & nbytes) {
    if (expert < 0 || expert >= info.n_expert || info.addr == 0 || info.expert_stride == 0) return false;
    const size_t offset = (size_t) expert * info.expert_stride;
    if (offset >= info.nbytes) return false;
    addr = info.addr + offset;
    nbytes = std::min(info.expert_stride, info.nbytes - offset);
    return nbytes > 0;
}
