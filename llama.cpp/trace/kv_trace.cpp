#include "trace_event.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-batch.h"

#include <algorithm>
#include <cstring>
#include <string>

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
    return -1;
}

uint32_t guess_ctx_len(const ggml_tensor * dst) {
    if (!dst) {
        return 0;
    }
    uint32_t best = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (dst->ne[i] > best) {
            best = (uint32_t) dst->ne[i];
        }
    }
    return best;
}

uint32_t guess_tokens(const ggml_tensor * src) {
    if (!src) {
        return 0;
    }
    uint32_t best = 0;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src->ne[i] > best) {
            best = (uint32_t) src->ne[i];
        }
    }
    return best;
}

const char * backend_name(const ggml_tensor * t) {
    ggml_backend_buffer_t buf = t && (t->view_src ? t->view_src->buffer : t->buffer) ? (t->view_src ? t->view_src->buffer : t->buffer) : nullptr;
    return buf ? ggml_backend_buffer_name(buf) : "unknown";
}

uintptr_t tensor_addr(const ggml_tensor * t) {
    if (!t) {
        return 0;
    }
    if (t->data) {
        return reinterpret_cast<uintptr_t>(t->data);
    }
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (buf) {
        void * base = ggml_backend_buffer_get_base(buf);
        return reinterpret_cast<uintptr_t>(base);
    }
    return 0;
}

} // namespace

extern "C" void llm_mem_trace_kv_set_rows(const ggml_tensor * t) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_KV) || !t) {
        return;
    }
    if (t->op != GGML_OP_SET_ROWS) {
        return;
    }

    const ggml_tensor * src = t->src[0];
    const ggml_tensor * dst = t->src[2];
    if (!dst) {
        return;
    }

    const char * name = ggml_get_name(dst);
    if (!name) {
        return;
    }

    const bool is_k = std::strstr(name, "cache_k_l") != nullptr;
    const bool is_v = std::strstr(name, "cache_v_l") != nullptr;
    if (!is_k && !is_v) {
        return;
    }

    const int layer = parse_layer_from_name(name);
    const uint32_t n_tokens = guess_tokens(src);
    const uint32_t ctx_len = guess_ctx_len(dst);
    const size_t kv_bytes = src ? ggml_nbytes(src) : 0;
    const uintptr_t addr = tensor_addr(dst);

    char addr_buf[32];
    std::snprintf(addr_buf, sizeof(addr_buf), "0x%llx", (unsigned long long) addr);

    const llama_ubatch * ubatch = llm_mem_trace_get_ubatch();

    std::string line;
    line.reserve(256);
    line += "{\"event\":\"KV_APPEND\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    if (layer >= 0) {
        line += ",\"layer\":" + std::to_string(layer);
    }
    line += ",\"kind\":\"" + std::string(is_k ? "K" : "V") + "\"";
    line += ",\"n_tokens\":" + std::to_string(n_tokens);
    line += ",\"ctx_len\":" + std::to_string(ctx_len);
    line += ",\"kv_bytes\":" + std::to_string(kv_bytes);
    line += ",\"kv_addr\":";
    json_escape_append(line, addr_buf);
    line += ",\"backend\":";
    json_escape_append(line, backend_name(dst));

    if (ubatch && ubatch->token && ubatch->n_tokens > 0) {
        line += ",\"token_ids\":[";
        const uint32_t count = std::min<uint32_t>(ubatch->n_tokens, n_tokens ? n_tokens : ubatch->n_tokens);
        for (uint32_t i = 0; i < count; ++i) {
            if (i) {
                line += ",";
            }
            line += std::to_string(ubatch->token[i]);
        }
        line += "]";
    }

    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_KV, line.c_str(), line.size());
}

extern "C" void llm_mem_trace_kv_reuse(uint32_t n_tokens, uint32_t reused) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_KV)) {
        return;
    }

    std::string line;
    line.reserve(128);
    line += "{\"event\":\"KV_REUSE\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
    line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
    line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
    line += ",\"n_tokens\":" + std::to_string(n_tokens);
    line += ",\"reused\":" + std::to_string(reused);
    line += "}";

    llm_mem_trace_write(LLM_MEM_TRACE_SINK_KV, line.c_str(), line.size());
}
