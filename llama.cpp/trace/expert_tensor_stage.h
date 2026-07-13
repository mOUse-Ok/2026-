#pragma once

#include <cstddef>

enum class ExpertTensorStage {
    Early,
    Late,
    Unknown,
};

ExpertTensorStage classify_expert_tensor_stage(const char * tensor_name);
const char * expert_tensor_stage_name(ExpertTensorStage stage);
size_t expert_tensor_stage_index(ExpertTensorStage stage);

