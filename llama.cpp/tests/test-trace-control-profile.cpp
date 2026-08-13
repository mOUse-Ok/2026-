#include "trace_event.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "test-trace-control-profile: %s\n", message);
    std::exit(1);
}

void require(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

} // namespace

int main() {
    char directory[] = "/tmp/llama-trace-control-XXXXXX";
    require(mkdtemp(directory) != nullptr, "failed to create temporary trace directory");

    setenv("LLM_MEM_TRACE", "1", 1);
    setenv("LLM_MEM_TRACE_DIR", directory, 1);
    setenv("LLM_MEM_TRACE_TENSOR", "0", 1);
    setenv("LLM_MEM_TRACE_KV", "0", 1);
    setenv("LLM_MEM_TRACE_EXPERT", "0", 1);
    setenv("LLM_MEM_TRACE_MEMORY", "1", 1);
    setenv("LLM_MEM_TRACE_CONTROL_ONLY", "1", 1);

    llm_mem_trace_init(directory);
    require(llm_mem_trace_control_only(), "control-only state was not enabled");

    const std::string raw = "{\"event\":\"STEP_END\",\"latency_ns\":1}";
    const std::string aggregate = "{\"event\":\"TEST_CONTROL_SUMMARY\",\"count\":1}";
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, raw.c_str(), raw.size());
    llm_mem_trace_write(LLM_MEM_TRACE_SINK_MEMORY, aggregate.c_str(), aggregate.size());
    llm_mem_trace_shutdown();

    std::ifstream memory(std::string(directory) + "/memory_trace.jsonl");
    require(memory.good(), "control MEMORY sink was not created");
    std::vector<std::string> lines;
    for (std::string line; std::getline(memory, line); ) {
        lines.push_back(line);
    }
    require(lines.size() == 2, "control trace wrote a non-summary record");
    require(lines[0].find("TEST_CONTROL_SUMMARY") != std::string::npos,
            "control summary record was not retained");
    require(lines[1].find("CONTROL_TRACE_SUMMARY") != std::string::npos,
            "process-end control summary was not retained");

    std::ifstream summary(std::string(directory) + "/summary.json");
    std::string summary_text((std::istreambuf_iterator<char>(summary)), std::istreambuf_iterator<char>());
    require(summary_text.find("\"control_only\":true") != std::string::npos,
            "summary.json did not record control-only mode");

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    return 0;
}
