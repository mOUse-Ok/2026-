#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ExpertTensorInfo {
    std::string name;
    int layer = -1;
    uintptr_t addr = 0;
    size_t nbytes = 0;
    int64_t n_expert = 0;
    size_t expert_stride = 0;
};

enum class ExpertAsyncPriorityMode {
    Score,
    Deadline,
    DeadlineScore,
};

enum class ExpertPressureLevel {
    Low = 0,
    Moderate = 1,
    High = 2,
    Critical = 3,
};
