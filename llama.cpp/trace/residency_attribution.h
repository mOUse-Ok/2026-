#pragma once

#include <cstddef>
#include <cstdint>

// Observation-only object residency attribution. The caller supplies the
// runtime semantic identity and the exact byte range about to be consumed.
// Routed Expert callers must pass an Expert Slice, never the container tensor.
bool llm_mem_trace_residency_attribution_enabled();

void llm_mem_trace_residency_attribution_observe(
        const char * object_class,
        const char * tensor_name,
        const char * tensor_subclass,
        int layer,
        int expert_id,
        uintptr_t virtual_address,
        size_t tensor_bytes);

