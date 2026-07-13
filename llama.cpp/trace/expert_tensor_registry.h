#pragma once

#include "expert_prefetch_types.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ggml_tensor;

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

    void add(const ggml_tensor * t, const char * name, int layer, uintptr_t addr, size_t nbytes);
    std::vector<ExpertTensorInfo> for_layer(int layer);
    bool was_hinted(uint64_t step, int layer, int expert, uintptr_t addr, uint64_t ttl_steps);
    bool mark_hinted(uint64_t step, int layer, int expert, uintptr_t addr, uint64_t ttl_steps);
    void write_route_hint_summary();
};

ExpertTensorRegistry & expert_tensor_registry();
bool expert_slice_range(const ExpertTensorInfo & info, int expert, uintptr_t & addr, size_t & nbytes);
