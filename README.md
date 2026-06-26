# LLM 推理内存管理优化系统

本仓库是“2026 年全国大学生计算机系统能力大赛操作系统设计赛，全国赛，OS 功能挑战赛道”初赛阶段作品工程。项目基于 `llama.cpp` 扩展 LLM 推理过程中的访存追踪、专家激活分析、KV cache 分析、OS hint 实验和用户态 expert prefetch 调度策略，用于探索受限物理内存设备上的 LLM 推理优化。

## 项目信息

- 赛题编号：proj59
- 赛题中文名称：内存受限环境的大语言模型推理优化文
- 赛题英文名称：Runtime Optimization of LLM Inference for the Memory Constraint System
- 队伍名称：法尔孔
- 队员信息：李子恒（队长）、邓晓川、蔡梓涵
- 项目名称：LLM 推理内存管理优化系统
- 开源协议：本队新增代码和文档采用 Apache-2.0；第三方 `llama.cpp` 代码保留其原始 MIT License。

## 工程入口

主要工程入口位于：

- `llama.cpp/`：上游 `llama.cpp` 源码及本项目修改。
- `llama.cpp/trace/`：本项目新增或扩展的访存追踪、分析、模拟和复现实验脚本。
- `docs/`：初赛提交所需设计、开发、测试、对比、复现和来源说明文档。

关键文件：

- `llama.cpp/trace/tensor_trace.cpp`：OS hint、expert slice cache、异步 expert prefetch、priority queue、route TTL 等核心实现。
- `llama.cpp/trace/analyze_trace.py`：trace 分析、指标聚合和报告生成。
- `llama.cpp/trace/simulate_expert_cache.py`：trace-driven expert cache 替换策略模拟。
- `llama.cpp/trace/compare_trace_runs.py`：多次运行指标对比和 Pareto 分析。
- `llama.cpp/trace/summarize_repeat_runs.py`：重复实验均值、标准差和变异系数聚合。
- `llama.cpp/trace/run_trace_pipeline.sh`：单次 trace pipeline 运行入口。
- `llama.cpp/trace/run_finalist_repeat_matrix.sh`：最终候选策略重复实验入口。

## 构建

推荐在 Linux 环境中使用 CMake 构建，并开启 `LLAMA_MEM_TRACE`：

```bash
cmake -S llama.cpp -B llama.cpp/build -DLLAMA_MEM_TRACE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-cli -j"$(nproc)"
```

如果已有 `llama.cpp/build`，可直接增量构建：

```bash
cmake --build llama.cpp/build --target llama-cli
```

## 运行

准备 GGUF 模型文件后，默认路径为：

```text
models/Qwen3.5-35B-A3B-Q3_K_M.gguf
```

也可以通过 `MODEL_FILE` 指定其他模型：

```bash
MODEL_FILE=/path/to/model.gguf \
RUN_NAME=baseline \
NUM_TOKENS_PREDICT=80 \
bash llama.cpp/trace/run_trace_pipeline.sh
```

运行当前主推荐策略：

```bash
MODEL_FILE=/path/to/model.gguf \
RUN_NAME=deadline_score \
NUM_TOKENS_PREDICT=80 \
LLM_MEM_TRACE_OS_HINTS=1 \
LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1 \
LLM_MEM_TRACE_OPT_EXPERT_POLICY=route \
LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0 \
LLM_MEM_TRACE_OPT_EXPERT_ASYNC=1 \
LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE=131072 \
LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=4 \
LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=1 \
LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=deadline_score \
bash llama.cpp/trace/run_trace_pipeline.sh
```

输出目录默认位于 `llama.cpp/trace_output/<RUN_NAME>/`。该目录数据量较大，已在 `.gitignore` 中排除。

## 测试与复现

静态检查：

```bash
python3 -m py_compile \
  llama.cpp/trace/analyze_trace.py \
  llama.cpp/trace/compare_trace_runs.py \
  llama.cpp/trace/simulate_expert_cache.py \
  llama.cpp/trace/summarize_repeat_runs.py

bash -n llama.cpp/trace/run_trace_pipeline.sh
bash -n llama.cpp/trace/run_finalist_repeat_matrix.sh
```

最终矩阵 dry-run：

```bash
REPEAT_COUNT=1 RUN_PREFIX=dryrun_check bash llama.cpp/trace/run_finalist_repeat_matrix.sh
```

执行最终 N 次重复实验：

```bash
RUN_REPEAT_MATRIX_EXECUTE=1 \
RUN_PREFIX=contest_finalist \
REPEAT_COUNT=3 \
MODEL_FILE=/path/to/model.gguf \
bash llama.cpp/trace/run_finalist_repeat_matrix.sh
```

更多复现细节见 [docs/reproduce.md](docs/reproduce.md)。

## 文档索引

- [设计文档](docs/design.md)
- [开发过程记录](docs/development-log.md)
- [测试报告](docs/test-report.md)
- [与类似项目对比](docs/comparison.md)
- [外部来源说明](docs/source-attribution.md)
- [AI 使用说明模板](docs/AI_USAGE.md)
- [复现说明](docs/reproduce.md)
- [初赛提交检查清单](docs/submission-checklist.md)

## 提交前检查清单

- 根目录存在 README、LICENSE 和 docs 文档。
- `llama.cpp/` 包含完整源码，不依赖局部补丁复原。
- 构建、运行、测试、复现命令在 README 和 docs 中列出。
- `.gitignore` 覆盖构建产物、trace 输出、模型文件、缓存目录和本地配置。
- 不提交模型权重、trace 输出、构建目录、缓存、IDE 私有配置、Token、密码或个人隐私数据。
- 所有非本队来源在 `docs/source-attribution.md` 中说明。
- AI 使用说明由队伍根据实际使用情况填写并人工审核。
