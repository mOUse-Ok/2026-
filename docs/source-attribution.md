# 外部来源说明

## 总说明

本项目包含本队新增实现，也包含第三方开源项目和参考论文/文档思想。所有非本队来源在本文档中说明。

## 第三方开源代码

### llama.cpp

- 路径：`llama.cpp/`
- 来源：llama.cpp 开源项目
- 原始协议：MIT License，见 `llama.cpp/LICENSE`
- 使用方式：作为 LLM 本地推理基础工程，本项目在其基础上增加 trace、分析和 OS hint 实验代码。

### ggml 与 vendor 组件

- 路径：`llama.cpp/ggml/`、`llama.cpp/vendor/`
- 来源：随 `llama.cpp` 分发的第三方依赖或子模块代码
- 协议：保留各自原始许可证，见对应目录下的 `LICENSE` 或 README。

## 本队新增或主要修改代码

以下文件为本项目主要新增或修改内容：

- `llama.cpp/trace/tensor_trace.cpp`
- `llama.cpp/trace/expert_trace.cpp`
- `llama.cpp/trace/trace_event.h`
- `llama.cpp/trace/analyze_trace.py`
- `llama.cpp/trace/simulate_expert_cache.py`
- `llama.cpp/trace/simulate_kv_cache_policy.py`
- `llama.cpp/trace/compare_trace_runs.py`
- `llama.cpp/trace/summarize_repeat_runs.py`
- `llama.cpp/trace/run_trace_pipeline.sh`
- `llama.cpp/trace/run_finalist_repeat_matrix.sh`
- `llama.cpp/trace/run_cgroup_memory_matrix.sh`

## 参考方向

本项目实现过程中参考了以下公开研究方向和系统机制：

- MoE-Infinity：request-level expert tracing/cache/prefetch 思路。
- SpecMD：Least-Stale 替换策略思想。
- ST-MoE：MoE expert 跨 token/跨层可预测性分析方向。
- PagedAttention/vAttention：KV cache 虚拟内存式管理方向。
- Linux MGLRU/DAMON/madvise：页面回收、冷热页和用户态 hint 机制。
- Linux cgroup v2：受限内存实验和 memory controller 指标采集机制。
- StreamingLLM：attention sink 与 recent window 的长上下文 KV 保留方向。
- H2O：heavy-hitter KV cache token 保留方向；当前项目仅在文档和模拟脚本中标记其需要 attention score 埋点，未实现该算法。
- KIVI：KV cache 量化方向；当前项目仅做 trace-driven 内存预算估算，未复制其量化实现。

主要公开来源：

- PagedAttention / vLLM：<https://arxiv.org/abs/2309.06180>，<https://github.com/vllm-project/vllm>
- vAttention：<https://arxiv.org/abs/2405.04437>
- FlashAttention：<https://arxiv.org/abs/2205.14135>，<https://github.com/Dao-AILab/flash-attention>
- SmoothQuant：<https://arxiv.org/abs/2211.10438>，<https://github.com/mit-han-lab/smoothquant>
- AWQ：<https://arxiv.org/abs/2306.00978>，<https://github.com/mit-han-lab/llm-awq>
- StreamingLLM：<https://arxiv.org/abs/2309.17453>，<https://github.com/mit-han-lab/streaming-llm>
- H2O：<https://arxiv.org/abs/2306.14048>，<https://github.com/FMInference/H2O>
- KIVI：<https://arxiv.org/abs/2402.02750>，<https://github.com/jy-yuan/KIVI>
- DuoAttention：<https://arxiv.org/abs/2410.10819>，<https://github.com/mit-han-lab/duo-attention>
- MoE-Infinity：<https://arxiv.org/abs/2401.14361>，<https://github.com/TorchMoE/MoE-Infinity>
- SpecMD：<https://arxiv.org/abs/2602.03921>
- ST-MoE：<https://arxiv.org/abs/2606.15453>
- Tutel：<https://arxiv.org/abs/2206.03382>，<https://github.com/microsoft/tutel>
- Linux cgroup v2：<https://docs.kernel.org/admin-guide/cgroup-v2.html>

说明：上述内容作为算法和系统设计参考，本项目没有直接复制论文代码。

## 模型和测试数据

默认脚本引用的模型路径：

```text
models/Qwen3.5-35B-A3B-Q3_K_M.gguf
```

- 模型来源：魔搭社区 ModelScope，使用模型为 Qwen3.5-35B-A3B-Q3_K_M。该模型为 Qwen/Qwen3.5-35B-A3B 的 GGUF 量化版本，量化仓库为 unsloth/Qwen3.5-35B-A3B-GGUF，具体使用文件为 Qwen3.5-35B-A3B-Q3_K_M.gguf。
- 模型许可证：Apache License 2.0（Apache-2.0）。
- 下载方式或准备方式：从魔搭社区 ModelScope 下载预量化 GGUF 模型文件 Qwen3.5-35B-A3B-Q3_K_M.gguf 至本地环境。未对模型进行重新训练或微调，仅作为本地推理模型使用，通过 llama.cpp / GGUF 兼容推理框架加载。
- 是否允许比赛提交或评测使用：从模型许可证角度，Apache-2.0 许可证允许使用、复制和分发，因此允许在比赛提交或评测中使用；同时在提交文档中保留模型来源、许可证和量化版本说明，并遵守比赛官方对外部模型使用的具体规定。

测试 prompt 由本项目脚本内置，用于触发 CPU、内存、GPU、OS、LLM 推理等多主题上下文，便于观察 MoE expert 和 KV cache 行为。

## 图片和实验结果

分析图表由本项目脚本从本地 trace 输出生成，位于 `llama.cpp/trace_output/` 下。该目录不提交仓库。

## 开源协议关系

- 本队新增代码和文档：Apache License 2.0。
- 上游 `llama.cpp`：MIT License。
- 第三方 vendor：保留原始许可证。

提交前应确认所有许可证文本和来源说明完整保留。
