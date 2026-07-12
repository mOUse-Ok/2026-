# LLM 推理内存管理优化系统

本仓库是“2026 年全国大学生计算机系统能力大赛操作系统设计赛，全国赛，OS 功能挑战赛道”参赛作品工程。项目基于 `llama.cpp` 扩展 LLM 推理过程中的访存追踪、专家激活分析、KV cache 分析、OS hint 实验和用户态 expert prefetch 调度策略，用于探索受限物理内存设备上的 LLM 推理优化。

## 项目信息

- 赛题编号：proj59
- 赛题中文名称：内存受限环境的大语言模型推理优化
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

- `llama.cpp/trace/tensor_trace.cpp`：OS hint、expert slice cache、异步队列、语义与压力双反馈控制、slack 取消及跨层预测核心实现。
- `llama.cpp/trace/analyze_trace.py`：trace 分析、指标聚合和报告生成。
- `llama.cpp/trace/trace_metrics.py`：STEP 延迟、trace-window faults 和尾延迟等无第三方依赖核心指标。
- `llama.cpp/trace/simulate_expert_cache.py`：trace-driven expert cache 替换策略模拟。
- `llama.cpp/trace/simulate_kv_cache_policy.py`：trace-driven KV cache 页面、窗口、预算和量化策略模拟。
- `llama.cpp/trace/compare_trace_runs.py`：多次运行指标对比和 Pareto 分析。
- `llama.cpp/trace/summarize_repeat_runs.py`：重复实验均值、标准差和变异系数聚合。
- `llama.cpp/trace/run_trace_pipeline.sh`：单次 trace pipeline 运行入口。
- `llama.cpp/trace/run_finalist_repeat_matrix.sh`：最终候选策略重复实验入口。
- `llama.cpp/trace/run_cgroup_memory_matrix.sh`：Linux cgroup v2 受限物理内存实验矩阵入口，默认 dry-run。
- `llama.cpp/trace/tests/`：指标语义、正式运行一致性和缺失指标回归测试。

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

运行当前主线“双反馈 + slack”控制器：

```bash
MODEL_FILE=/path/to/model.gguf \
RUN_NAME=feedback_slack \
NUM_TOKENS_PREDICT=80 \
LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=feedback_slack \
bash llama.cpp/trace/run_trace_pipeline.sh
```

尝试带成本门控的在线相邻层预测：

```bash
MODEL_FILE=/path/to/model.gguf \
RUN_NAME=feedback_slack_predict \
NUM_TOKENS_PREDICT=80 \
LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=feedback_slack_predict \
bash llama.cpp/trace/run_trace_pipeline.sh
```

控制器读取 cgroup v2 的 `memory.current/high/max`、`memory.pressure`、`memory.swap.current` 和 `memory.stat` refault，动态缩放 expert 预取预算；队列按在线 layer EWMA 估计 deadline，在出队时取消已过期、压力过高或收益不足的任务。预测器只学习当前请求中相邻层 expert 转移，默认 top-2，并把预测命中率和实际预取决策分开统计。所有功能默认关闭，是否优于旧 `deadline_score` 以受控复测为准。

输出目录默认位于 `llama.cpp/trace_output/<RUN_NAME>/`。该目录数据量较大，已在 `.gitignore` 中排除。

单次运行支持两种主要 profile：`evidence` 生成完整行为证据，`benchmark` 关闭高流量 tensor/KV 和驻留采样，用于策略性能对比。每个有效运行同时保存 manifest、缓存准备记录、GNU time 全进程指标、trace 完整性摘要和确定性输出 hash。

## 测试与复现

静态检查：

```bash
python3 -m pip install -r llama.cpp/trace/requirements-analysis.txt

python3 -m py_compile \
  llama.cpp/trace/analyze_trace.py \
  llama.cpp/trace/trace_metrics.py \
  llama.cpp/trace/compare_trace_runs.py \
  llama.cpp/trace/simulate_expert_cache.py \
  llama.cpp/trace/simulate_kv_cache_policy.py \
  llama.cpp/trace/summarize_repeat_runs.py \
  llama.cpp/trace/prepare_model_cache.py \
  llama.cpp/trace/write_run_manifest.py \
  llama.cpp/trace/validate_trace_summary.py

python3 -m unittest discover -s llama.cpp/trace/tests -p 'test_*.py' -v

bash -n llama.cpp/trace/run_trace_pipeline.sh
bash -n llama.cpp/trace/run_finalist_repeat_matrix.sh
bash -n llama.cpp/trace/run_cgroup_memory_matrix.sh
```

最终矩阵 dry-run：

```bash
REPEAT_COUNT=1 RUN_PREFIX=dryrun_check bash llama.cpp/trace/run_finalist_repeat_matrix.sh
```

cgroup v2 受限内存矩阵 dry-run：

```bash
MEMORY_LIMITS_MB=4096,5120 \
RUN_GROUPS=baseline,deadline_score,feedback_slack,feedback_slack_predict \
REPEAT_COUNT=1 \
bash llama.cpp/trace/run_cgroup_memory_matrix.sh
```

如需真实运行，先确认当前用户拥有可写的 cgroup v2 delegated parent，然后设置：

```bash
RUN_MEMORY_PRESSURE_EXECUTE=1 \
CGROUP_PARENT=/sys/fs/cgroup/<delegated-parent> \
MEMORY_LIMITS_MB=4096,5120 \
RUN_GROUPS=baseline,deadline_score,feedback_slack,feedback_slack_predict \
bash llama.cpp/trace/run_cgroup_memory_matrix.sh
```

执行最终 N 次重复实验：

```bash
RUN_REPEAT_MATRIX_EXECUTE=1 \
RUN_PREFIX=contest_finalist \
REPEAT_COUNT=8 \
TRACE_PROFILE=benchmark \
CACHE_MODE=cold \
ORDER_MODE=latin \
ORDER_SEED=0 \
MEMORY_MAX=8G \
MEMORY_SWAP_MAX=1G \
MODEL_FILE=/path/to/model.gguf \
bash llama.cpp/trace/run_finalist_repeat_matrix.sh
```

正式矩阵拒绝脏仓库、trace 丢失、非零退出、产物不完整、模型或参数不一致及输出 hash 不一致的运行。当前文档中的旧 N=3 数据仅保留为研发过程记录，不作为最终策略优劣结论。

更多复现细节见 [docs/reproduce.md](docs/reproduce.md)。

## 文档索引

- [设计文档](docs/design.md)
- [开发过程记录](docs/development-log.md)
- [测试报告](docs/test-report.md)
- [与类似项目对比](docs/comparison.md)
- [外部来源说明](docs/source-attribution.md)
- [AI 使用说明](docs/AI_USAGE.md)
- [复现说明](docs/reproduce.md)

## 提交前检查清单

- 根目录存在 README、LICENSE 和 docs 文档。
- `llama.cpp/` 包含完整源码，不依赖局部补丁复原。
- 构建、运行、测试、复现命令在 README 和 docs 中列出。
- `.gitignore` 覆盖构建产物、trace 输出、模型文件、缓存目录和本地配置。
- 不提交模型权重、trace 输出、构建目录、缓存、IDE 私有配置、Token、密码或个人隐私数据。
- 所有非本队来源在 `docs/source-attribution.md` 中说明。
- AI 使用说明由队伍根据实际使用情况填写并人工审核。
