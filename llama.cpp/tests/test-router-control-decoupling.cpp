#include "trace_event.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace {

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "test-router-control-decoupling: %s\n", message);
    std::exit(1);
}

void require(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

} // namespace

int main() {
    char directory[] = "/tmp/llama-router-control-XXXXXX";
    require(mkdtemp(directory) != nullptr, "failed to create temporary trace directory");

    // All event sinks are off.  Memory Object control must nevertheless keep
    // Router observation synchronized, otherwise the state manager receives no
    // routed Expert IDs in a performance-oriented trace configuration.
    setenv("LLM_MEM_TRACE", "1", 1);
    setenv("LLM_MEM_TRACE_DIR", directory, 1);
    setenv("LLM_MEM_TRACE_TENSOR", "0", 1);
    setenv("LLM_MEM_TRACE_KV", "0", 1);
    setenv("LLM_MEM_TRACE_EXPERT", "0", 1);
    setenv("LLM_MEM_TRACE_MEMORY", "0", 1);
    setenv("LLM_MEM_TRACE_OPT_EXPERT_MEMORY_OBJECTS", "1", 1);

    llm_mem_trace_init(directory);
    require(!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_EXPERT),
            "EXPERT sink unexpectedly enabled");
    require(llm_mem_trace_moe_control_requires_router(),
            "Memory Object control did not request Router semantics");

    ggml_tensor router = {};
    router.op = GGML_OP_GET_ROWS;
    std::snprintf(router.name, sizeof(router.name), "ffn_moe_weights-7");
    require(llm_mem_trace_moe_weights_requires_sync(&router),
            "Router observation was disabled with EXPERT sink off");

    llm_mem_trace_shutdown();
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    return 0;
}
