# 设计文档

## 项目基本信息

- 赛题编号：proj59
- 赛题中文名称：内存受限环境的大语言模型推理优化
- 赛题英文名称：Runtime Optimization of LLM Inference for the Memory Constraint System
- 队伍名称：法尔孔
- 队员信息：李子恒、邓晓川、蔡梓涵
- 项目名称：LLM 推理内存管理优化系统
- 初赛目标：基于完整可构建的 `llama.cpp` 工程，实现 LLM 推理访存行为分析、OS hint 实验、专家激活预取和页面替换策略评估，并提供可复现实验脚本与文档。

## 背景与需求分析

大语言模型推理需要同时加载大量模型参数、维护 KV cache，并在 MoE 模型中按 token 激活少量专家。受限设备上物理内存不足会导致频繁 major page faults、swap 增长和 decode 延迟抖动。赛题要求从操作系统角度分析数据访问、传输、存储和预取行为，探索使用虚拟内存与按需加载策略降低物理内存压力，并通过预取覆盖 I/O 延迟。

初赛阶段需要达成：

- 能从干净环境构建工程。
- 能运行 LLM 推理 trace pipeline。
- 能采集 tensor、KV、expert、memory trace。
- 能比较 baseline、expert prefetch、异步 priority prefetch 等策略。
- 能输出开发文档、测试报告、来源说明和复现说明。

## 总体设计

系统基于 `llama.cpp` 扩展，主要模块如下：

| 模块 | 路径 | 职责 |
| --- | --- | --- |
| 推理引擎 | `llama.cpp/` | 保留上游 llama.cpp 的模型加载、推理、CMake 构建体系 |
| Trace 事件写入 | `llama.cpp/trace/trace_writer.cpp` | 将 JSONL trace 写入输出目录 |
| Tensor/OS hint 跟踪 | `llama.cpp/trace/tensor_trace.cpp` | 记录 tensor load/first touch，执行可选 OS hint 和 expert prefetch |
| Expert trace | `llama.cpp/trace/expert_trace.cpp` | 记录 MoE expert routing 信息 |
| KV trace | `llama.cpp/trace/kv_trace.cpp` | 记录 KV cache append/reuse 行为 |
| 分析脚本 | `llama.cpp/trace/analyze_trace.py` | 聚合指标并生成图表/报告 |
| 替换策略模拟 | `llama.cpp/trace/simulate_expert_cache.py` | 离线比较 route/LRU/LFU/window-LFU/least-stale |
| 多运行对比 | `llama.cpp/trace/compare_trace_runs.py` | 比较多次运行的 latency、faults、RSS、swap、hint events |
| 重复实验聚合 | `llama.cpp/trace/summarize_repeat_runs.py` | 计算 N 次重复实验的均值、标准差、CV |
| 自动运行入口 | `llama.cpp/trace/run_trace_pipeline.sh` | 运行推理、收集 trace、执行分析 |
| 最终矩阵入口 | `llama.cpp/trace/run_finalist_repeat_matrix.sh` | dry-run 或执行 finalist repeat matrix |

关键数据流：

1. `llama-cli` 执行模型推理。
2. 插桩代码记录 tensor、KV、expert、memory 事件。
3. 可选 OS hint 策略在 tensor/expert 路径上触发 `madvise`/`posix_fadvise`。
4. `run_trace_pipeline.sh` 将 JSONL、stderr、stdout、summary 输出到 `trace_output/<RUN_NAME>/`。
5. `analyze_trace.py` 生成 metrics、图表和 HTML/Markdown 分析。
6. `compare_trace_runs.py` 和 `summarize_repeat_runs.py` 进行跨运行对比。

## 详细实现

### 1. 默认关闭的 trace 与 OS hint

所有优化策略均通过环境变量显式开启。默认情况下，`LLM_MEM_TRACE_OS_HINTS=0`，不会改变 baseline 推理行为。

关键环境变量：

- `LLM_MEM_TRACE=1`：开启 trace。
- `LLM_MEM_TRACE_DIR`：设置 trace 输出目录。
- `LLM_MEM_TRACE_OS_HINTS=1`：开启 OS hint 实验。
- `LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1`：开启 expert-aware prefetch。

### 2. Expert slice cache 与替换策略

expert slice 以 `(layer, expert, tensor)` 作为逻辑缓存项，记录 size、last_step、hit_count、score、resident/advised 状态。离线模拟支持：

- `route`
- `lru`
- `lfu`
- `window_lfu`
- `least_stale`

实验结论显示，在当前 trace 和 <=1 GiB 预算下，朴素 LRU/LFU 类策略 eviction 过多，不适合作为真实推理主路线。

### 3. Route-based expert prefetch

route 策略依据模型实际 routed experts 对 expert slice 进行预取。它可以显著降低 major faults，但同步 hint call 数量较高。

优化尝试包括：

- top-k 限制：降低 hint 数量，但会破坏覆盖率。
- coalescing：合并相邻 slice，减少 syscall，但地址连续性有限。
- route hint TTL：减少短窗口重复 hint。
- async queue：将 hint 从 decode 关键路径移出。
- priority scheduling：优先处理更接近使用点的 expert slice。

### 4. 异步 expert prefetch 队列

异步队列由 `LLM_MEM_TRACE_OPT_EXPERT_ASYNC` 开启。核心机制：

- producer 在推理路径提交 `ExpertHintTask`。
- worker 线程异步执行 `madvise(MADV_WILLNEED)` 和可选 `posix_fadvise`。
- 队列满或 worker 启动失败时同步 fallback。
- 进程退出时 flush pending task 并写出 `EXPERT_ASYNC_SUMMARY`。

priority mode：

- `score`：按 router score 排序。
- `deadline`：按 step/layer 接近程度排序。
- `deadline_score`：先按 step/layer，再按 score，当前综合效果最好。

### 5. KV cache 分析

分析脚本统计：

- K/V 各自占用。
- PREFILL/DECODE 阶段占用。
- 追加 token 估算。
- 每 1k tokens KV 增长。
- 2k/4k/8k 上下文投影。
- layer imbalance。

KV 分析用于说明长上下文内存增长趋势，当前主要优化收益仍来自 expert prefetch。

## 运行与复现

完整命令见 [reproduce.md](reproduce.md)。核心步骤：

```bash
cmake -S llama.cpp -B llama.cpp/build -DLLAMA_MEM_TRACE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-cli -j"$(nproc)"

MODEL_FILE=/path/to/model.gguf RUN_NAME=baseline bash llama.cpp/trace/run_trace_pipeline.sh

RUN_REPEAT_MATRIX_EXECUTE=1 \
RUN_PREFIX=contest_finalist \
REPEAT_COUNT=3 \
MODEL_FILE=/path/to/model.gguf \
bash llama.cpp/trace/run_finalist_repeat_matrix.sh
```

## 测试结果

当前 N=3 重复实验摘要：

| 方案 | Decode 均值 | Major faults 均值 | RSS 均值 | Swap 均值 | Hint calls 均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 376,069 us | 757,052 | 6.228 GiB | 217.86 MiB | 0 |
| expert_prefetch | 214,724 us | 49,613 | 6.486 GiB | 157.44 MiB | 99,907 |
| deadline_score | 194,971 us | 71,060 | 6.433 GiB | 123.67 MiB | 98,893 |
| decode_ttl1 | 216,819 us | 69,406 | 6.501 GiB | 174.25 MiB | 82,711 |

结论：`deadline_score` 是当前综合 Pareto 最优；同步 `expert_prefetch` 是 major faults 最优；`decode_ttl1` 是较低 hint call 的备选。

## 创新性分析

- 将 MoE expert routing 与 OS page hint 结合，围绕实际 routed experts 进行预取。
- 使用 trace-driven offline simulation 先筛选替换策略，减少真实推理试错成本。
- 将 syscall 从 decode 关键路径迁移到用户态异步队列。
- 引入 `deadline_score` 调度，将运行时使用顺序和 router score 结合。
- 同时评估 latency、major faults、RSS、swap、hint events，避免单指标优化。

## 与类似项目对比

详见 [comparison.md](comparison.md)。

## 开发过程

详见 [development-log.md](development-log.md)。

## 外部来源说明

详见 [source-attribution.md](source-attribution.md)。

## AI 使用说明

详见 [AI_USAGE.md](AI_USAGE.md)。该文件为队伍自行填写模板，提交前需人工确认。

## 开源协议说明

本队新增代码和文档采用 Apache License 2.0。仓库中第三方 `llama.cpp` 及其 vendor 组件保留原始许可证，详见 `llama.cpp/LICENSE` 和 `llama.cpp/vendor/*/LICENSE`。
