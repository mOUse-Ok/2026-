#include "expert_tensor_stage.h"

#include <cstring>

namespace {

bool has_tensor_suffix(const char * tensor_name, const char * suffix) {
    if (!tensor_name || !suffix) {
        return false;
    }
    const size_t name_len = std::strlen(tensor_name);
    const size_t suffix_len = std::strlen(suffix);
    if (name_len < suffix_len ||
            std::memcmp(tensor_name + name_len - suffix_len, suffix, suffix_len) != 0) {
        return false;
    }
    return name_len == suffix_len || tensor_name[name_len - suffix_len - 1] == '.';
}

} // namespace

ExpertTensorStage classify_expert_tensor_stage(const char * tensor_name) {
    if (has_tensor_suffix(tensor_name, "ffn_gate_exps.weight") ||
            has_tensor_suffix(tensor_name, "ffn_up_exps.weight") ||
            has_tensor_suffix(tensor_name, "ffn_gate_up_exps.weight")) {
        return ExpertTensorStage::Early;
    }
    if (has_tensor_suffix(tensor_name, "ffn_down_exps.weight")) {
        return ExpertTensorStage::Late;
    }
    return ExpertTensorStage::Unknown;
}

const char * expert_tensor_stage_name(ExpertTensorStage stage) {
    switch (stage) {
        case ExpertTensorStage::Early:   return "EARLY";
        case ExpertTensorStage::Late:    return "LATE";
        case ExpertTensorStage::Unknown: return "UNKNOWN";
    }
    return "UNKNOWN";
}

size_t expert_tensor_stage_index(ExpertTensorStage stage) {
    switch (stage) {
        case ExpertTensorStage::Early:   return 0;
        case ExpertTensorStage::Late:    return 1;
        case ExpertTensorStage::Unknown: return 2;
    }
    return 2;
}

