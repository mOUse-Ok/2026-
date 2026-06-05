#include "trace_event.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-batch.h"

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
    const char * dash = std::strrchr(name, '-');
    if (!dash || !dash[1]) {
        return -1;
    }
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

bool is_weights_tensor(const char * name) {
    if (!name) {
        return false;
    }
    if (!std::strstr(name, "ffn_moe_weights")) {
        return false;
    }
    if (std::strstr(name, "weights_sum")) {
        return false;
    }
    return true;
}

bool is_host_tensor(const ggml_tensor * t) {
    ggml_backend_buffer_t buf = t ? (t->view_src ? t->view_src->buffer : t->buffer) : nullptr;
    return buf && ggml_backend_buffer_is_host(buf) && t->data;
}

float read_f32(const ggml_tensor * t, const char * base, size_t offset) {
    switch (t->type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(base + offset);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(base + offset));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(base + offset));
        default:
            return 0.0f;
    }
}

int read_idx(const ggml_tensor * ids, const char * base, size_t offset) {
    if (ids->type == GGML_TYPE_I32) {
        return *reinterpret_cast<const int32_t *>(base + offset);
    }
    if (ids->type == GGML_TYPE_I64) {
        return (int) *reinterpret_cast<const int64_t *>(base + offset);
    }
    return -1;
}

} // namespace

extern "C" void llm_mem_trace_moe_weights(const ggml_tensor * t) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_EXPERT) || !t) {
        return;
    }

    const char * name = ggml_get_name(t);
    if (!is_weights_tensor(name)) {
        return;
    }

    if (!is_host_tensor(t)) {
        return;
    }

    const ggml_tensor * ids = t->src[1];
    if (!ids || !is_host_tensor(ids) || (ids->type != GGML_TYPE_I32 && ids->type != GGML_TYPE_I64)) {
        return;
    }

    const int layer = parse_layer_from_name(name);
    const int64_t n_expert_used = t->ne[1] > 0 ? t->ne[1] : 0;
    const int64_t n_tokens = t->ne[2] > 0 ? t->ne[2] : t->ne[1];

    if (n_expert_used <= 0 || n_tokens <= 0) {
        return;
    }

    const llama_ubatch * ubatch = llm_mem_trace_get_ubatch();
    const uint32_t ubatch_tokens = ubatch ? ubatch->n_tokens : 0;

    const char * weights_base = reinterpret_cast<const char *>(t->data);
    const char * ids_base = reinterpret_cast<const char *>(ids->data);

    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        std::string line;
        line.reserve(256 + (size_t) n_expert_used * 8);
        line += "{\"event\":\"EXPERT_ROUTE\",\"ts_ns\":" + std::to_string(llm_mem_trace_time_ns());
        line += ",\"phase\":\"" + std::string(phase_name(llm_mem_trace_get_phase())) + "\"";
        line += ",\"step\":" + std::to_string(llm_mem_trace_get_step());
        if (layer >= 0) {
            line += ",\"layer\":" + std::to_string(layer);
        }
        if (ubatch && ubatch->token && tok < ubatch_tokens) {
            line += ",\"token\":" + std::to_string(ubatch->token[tok]);
        }
        line += ",\"experts\":[";

        for (int64_t e = 0; e < n_expert_used; ++e) {
            if (e) {
                line += ",";
            }
            const size_t idx_offset = (size_t) e * ids->nb[0] + (size_t) tok * ids->nb[1];
            const int expert = read_idx(ids, ids_base, idx_offset);
            line += std::to_string(expert);
        }
        line += "]";

        line += ",\"scores\":[";
        for (int64_t e = 0; e < n_expert_used; ++e) {
            if (e) {
                line += ",";
            }
            const size_t w_offset = (size_t) e * t->nb[1] + (size_t) tok * t->nb[2];
            const float score = read_f32(t, weights_base, w_offset);
            line += std::to_string(score);
        }
        line += "]";

        line += "}";

        llm_mem_trace_write(LLM_MEM_TRACE_SINK_EXPERT, line.c_str(), line.size());
    }
}
