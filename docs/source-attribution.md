# 外部来源说明

## 总说明

本项目包含本队新增实现，也包含第三方开源项目和参考论文/文档思想。所有非本队来源在本文档中说明。提交前队伍应补充实际使用的模型、数据、图片、脚本来源。

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
- `llama.cpp/trace/compare_trace_runs.py`
- `llama.cpp/trace/summarize_repeat_runs.py`
- `llama.cpp/trace/run_trace_pipeline.sh`
- `llama.cpp/trace/run_finalist_repeat_matrix.sh`

## 参考方向

本项目实现过程中参考了以下公开研究方向和系统机制：

- MoE-Infinity：request-level expert tracing/cache/prefetch 思路。
- SpecMD：Least-Stale 替换策略思想。
- ST-MoE：MoE expert 跨 token/跨层可预测性分析方向。
- PagedAttention/vAttention：KV cache 虚拟内存式管理方向。
- Linux MGLRU/DAMON/madvise：页面回收、冷热页和用户态 hint 机制。

说明：上述内容作为算法和系统设计参考，本项目没有直接复制论文代码。

## 模型和测试数据

默认脚本引用的模型路径：

```text
models/Qwen3.5-35B-A3B-Q3_K_M.gguf
```

模型文件不提交仓库。提交前应由队伍补充：

- 模型来源：待填写
- 模型许可证：待填写
- 下载方式或准备方式：待填写
- 是否允许比赛提交或评测使用：待填写

测试 prompt 由本项目脚本内置，用于触发 CPU、内存、GPU、OS、LLM 推理等多主题上下文，便于观察 MoE expert 和 KV cache 行为。

## 图片和实验结果

分析图表由本项目脚本从本地 trace 输出生成，位于 `llama.cpp/trace_output/` 下。该目录不提交仓库。

## 开源协议关系

- 本队新增代码和文档：Apache License 2.0。
- 上游 `llama.cpp`：MIT License。
- 第三方 vendor：保留原始许可证。

提交前应确认所有许可证文本和来源说明完整保留。
