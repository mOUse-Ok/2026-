#include "trace_event.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-batch.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
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

int env_int_or_default(const char * key, int def_value) {
    const char * val = std::getenv(key);
    if (!val || !val[0]) {
        return def_value;
    }
    char * end = nullptr;
    const long parsed = std::strtol(val, &end, 10);
    return end && *end == '\0' && parsed > 0 ? (int) parsed : def_value;
}

bool router_score_diagnostic_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLM_MEM_TRACE_ROUTER_SCORE_DIAGNOSTIC");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

uint32_t raw_score_bits(const ggml_tensor * t, const char * base, size_t offset) {
    uint32_t bits = 0;
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(&bits, base + offset, sizeof(bits));
    } else if (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16) {
        uint16_t bits16 = 0;
        std::memcpy(&bits16, base + offset, sizeof(bits16));
        bits = bits16;
    }
    return bits;
}

uint32_t f32_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void append_hex_bits(std::string & line, uint32_t bits, int width) {
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), width == 4 ? "0x%04x" : "0x%08x", bits);
    json_escape_append(line, buffer);
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

extern "C" int llm_mem_trace_moe_weights_requires_sync(const ggml_tensor * t) {
    if (!llm_mem_trace_sink_enabled(LLM_MEM_TRACE_SINK_EXPERT) || !t) {
        return 0;
    }

    // This predicate runs on every graph compute thread before the producer
    // starts. It must inspect metadata only so every participant makes the
    // same barrier decision without reading an incomplete Tensor payload.
    return t->op == GGML_OP_GET_ROWS && is_weights_tensor(ggml_get_name(t));
}

extern "C" void llm_mem_trace_moe_weights(const ggml_tensor * t) {
    if (!llm_mem_trace_moe_weights_requires_sync(t)) {
        return;
    }

    const char * name = ggml_get_name(t);
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
    const int max_topk = env_int_or_default("LLM_MEM_TRACE_EXPERT_TOPK_MAX", 16);
    if (n_expert_used > max_topk) {
        return;
    }

    const int max_expert_id = env_int_or_default("LLM_MEM_TRACE_MAX_EXPERT_ID", 255);

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
        int experts[64];
        float scores[64];
        uint32_t source_bits[64];
        uint32_t converted_bits[64];
        const bool score_diagnostic = router_score_diagnostic_enabled();
        bool valid = n_expert_used <= (int64_t) (sizeof(experts) / sizeof(experts[0]));

        for (int64_t e = 0; valid && e < n_expert_used; ++e) {
            const size_t idx_offset = (size_t) e * ids->nb[0] + (size_t) tok * ids->nb[1];
            const int expert = read_idx(ids, ids_base, idx_offset);
            if (expert < 0 || expert > max_expert_id) {
                valid = false;
                break;
            }
            const size_t w_offset = (size_t) e * t->nb[1] + (size_t) tok * t->nb[2];
            experts[e] = expert;
            scores[e] = read_f32(t, weights_base, w_offset);
            if (score_diagnostic) {
                source_bits[e] = raw_score_bits(t, weights_base, w_offset);
                converted_bits[e] = f32_bits(scores[e]);
            }
        }

        if (!valid) {
            continue;
        }

        llm_mem_trace_prefetch_expert_layer(layer, (int) tok, experts, scores, (int) n_expert_used, "moe_route");

        line += ",\"top_k\":" + std::to_string(n_expert_used);
        line += ",\"experts\":[";

        for (int64_t e = 0; e < n_expert_used; ++e) {
            if (e) {
                line += ",";
            }
            line += std::to_string(experts[e]);
        }
        line += "]";

        line += ",\"scores\":[";
        for (int64_t e = 0; e < n_expert_used; ++e) {
            if (e) {
                line += ",";
            }
            line += std::to_string(scores[e]);
        }
        line += "]";

        if (score_diagnostic) {
            line += ",\"observation_sync\":\"producer_barrier_hook_release_barrier\"";
            line += ",\"observation_sync_protocol\":\"m6b1.2-v1\"";
            line += ",\"score_source_dtype\":";
            json_escape_append(line, ggml_type_name(t->type));
            line += ",\"score_shape\":[" + std::to_string(t->ne[0]) + "," +
                    std::to_string(t->ne[1]) + "," + std::to_string(t->ne[2]) + "]";
            line += ",\"score_strides_bytes\":[" + std::to_string(t->nb[0]) + "," +
                    std::to_string(t->nb[1]) + "," + std::to_string(t->nb[2]) + "]";
            line += ",\"score_raw_bits\":[";
            const int source_width = t->type == GGML_TYPE_F32 ? 8 : 4;
            for (int64_t e = 0; e < n_expert_used; ++e) {
                if (e) {
                    line += ",";
                }
                append_hex_bits(line, source_bits[e], source_width);
            }
            line += "]";
            line += ",\"score_f32_bits\":[";
            for (int64_t e = 0; e < n_expert_used; ++e) {
                if (e) {
                    line += ",";
                }
                append_hex_bits(line, converted_bits[e], 8);
            }
            line += "]";
        }

        line += "}";

        llm_mem_trace_write(LLM_MEM_TRACE_SINK_EXPERT, line.c_str(), line.size());
    }
}
