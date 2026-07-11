# 设计文档

## 项目基本信息

- 赛题编号：proj59
- 赛题中文名称：内存受限环境的大语言模型推理优化
- 赛题英文名称：Runtime Optimization of LLM Inference for the Memory Constraint System
- 队伍名称：法尔孔
- 队员信息：李子恒、邓晓川、蔡梓涵
- 项目名称：LLM 推理内存管理优化系统
- 项目目标：在不修改 Linux 内核、不改变模型计算结果的前提下，观测 LLM 推理访存行为，并利用模型语义、虚拟内存接口和异步预取降低受限内存环境中的 I/O 等待与延迟抖动。

## 背景与需求分析

LLM 推理同时需要模型权重、运行时张量和持续增长的 KV cache。MoE 模型虽然每个 token 只计算少量专家，但大量专家权重仍需保存在虚拟地址空间中；当物理内存不足时，专家页的首次访问和反复换入会产生 major faults、swap 和 decode 延迟抖动。

赛题要求从操作系统视角分析参数加载、KV cache、专家激活、数据换入换出和预取行为。本项目选择用户态路线：保留完整 `llama.cpp` 工程，通过 trace 获得内核不可见的 layer、expert、phase 和 route score，再使用 Linux `madvise`、`posix_fadvise` 与 cgroup v2 开展可关闭、可复现的实验。

## 总体设计

| 模块 | 主要路径 | 职责 |
| --- | --- | --- |
| 推理引擎 | `llama.cpp/` | 模型加载、计算图和推理执行 |
| Trace writer | `llama.cpp/trace/trace_writer.cpp` | 异步写入 JSONL，并记录 sink 完整性 |
| Tensor/OS hint | `llama.cpp/trace/tensor_trace.cpp` | 记录张量访问，维护 expert slice 信息，提交页面提示任务 |
| Expert/KV trace | `llama.cpp/trace/expert_trace.cpp`、`kv_trace.cpp` | 记录路由和 KV append/reuse |
| 分析与模拟 | `llama.cpp/trace/*.py` | 指标聚合、替换策略模拟、KV 预算估算和 Pareto 分析 |
| 运行控制 | `run_trace_pipeline.sh`、矩阵脚本 | 缓存准备、cgroup、重复顺序、证据产物和结果校验 |

关键数据流如下：

1. `llama-cli` 在每个 ubatch 前设置 phase、step 和 token 上下文。
2. `STEP_BEGIN/STEP_END` 包围一次 `process_ubatch()`，形成权威内部延迟记录。
3. tensor、expert、KV 和 memory 事件进入独立 trace sink。
4. 可选策略根据 routed experts 提交 `madvise(MADV_WILLNEED)` 等任务。
5. pipeline 同时保存 trace、GNU time 全进程指标、运行 manifest 和输出 hash。
6. 分析脚本生成单次指标；聚合脚本先验证运行一致性，再计算均值、标准差和 CV。

## 详细实现

### Trace 完整性与测量口径

Trace writer 为每个 sink 记录 `enqueued`、`written` 和 `dropped`。正式运行默认不允许静默丢事件，队列满时生产者等待；只有显式设置 `LLM_MEM_TRACE_ALLOW_DROP=1` 才允许丢弃。

提供两种主要 profile：

- `evidence`：开启 tensor、KV、expert、memory、页驻留和 smaps，用于行为解释。
- `benchmark`：关闭高流量 tensor/KV 与高开销驻留采样，保留 expert 和 memory 证据，用于策略性能对比。

`TOKEN_END.latency_ns` 是旧 trace 的 ubatch 延迟重复记录，只作为兼容回退。新 trace 的 prefill/decode mean、p50、p95、p99 和吞吐量均从 `STEP_END` 计算。全进程 wall time、峰值 RSS 和 faults 由 GNU time 提供，trace RSS/PSS 用于阶段时间线。

### Expert-aware 页面提示

expert slice 使用 `(layer, expert, tensor)` 标识。route 策略根据模型实际路由结果计算分片地址，只对即将使用的专家页发出提示。所有 OS hint 和异步功能均为 opt-in，`LLM_MEM_TRACE_OS_HINTS=0` 时不改变默认推理行为。

支持的实验项包括：

- route/LRU/LFU/window-LFU/least-stale 策略；
- prefill/decode 分阶段 top-k；
- route hint TTL；
- 相邻地址 coalescing；
- 同步或多 worker 异步队列；
- score、deadline、deadline_score 优先级；
- `MADV_COLD`、显式 `MADV_DONTNEED` 或 `MADV_PAGEOUT` 回收提示。

### 异步 priority queue

推理线程只构造 `ExpertHintTask` 并入队，worker 执行实际系统调用。队列满或线程启动失败时记录 fallback。`deadline_score` 先比较 step/layer 紧迫度，再比较 router score；该机制是当前项目最接近自主设计的部分，但仍属于静态启发式，不等同于经过证明的真实 deadline 调度算法。

### 离线策略模拟

`simulate_expert_cache.py` 使用已有 trace 比较不同 expert cache 预算和替换策略，避免每个候选都重跑大模型。`simulate_kv_cache_policy.py` 估算完整预留、按块提交、KV 量化、滑动窗口和预算策略的内存上界。

其中 `paged_blocks` 是保持完整上下文的工程候选；滑动窗口、sink-recent 和 budget-LRU 会丢弃上下文，只能作为压力测试。当前 trace 缺少 attention score，因此不能据此声称已经实现 H2O/heavy-hitter。

### 受控实验协议

单次运行必须生成：

- `run_manifest.json`
- `cache_preparation.json`
- `process_metrics.json`
- `summary.json`
- `output.sha256`
- `analysis/metrics.json`

正式矩阵使用文件级冷缓存、固定 seed、CPU expert 路径和四方案位置轮换。聚合前校验 commit、模型、二进制、prompt、参数、cgroup、trace 完整性、进程退出码与输出 hash。缺少任一证据的运行不进入正式结果。

## 当前创新性判断

项目目前具有三项系统组合创新：

1. 将真实 MoE routing 语义映射到 Linux 页级提示，而不是通用顺序预取。
2. 将 decode 路径上的同步 hint 迁移到带紧迫度和 route score 的异步队列。
3. 使用同一套 trace 闭环联合评估延迟、faults、RSS、swap、hint 和正确性。

这些内容可以作为工程与系统机制创新，但尚不足以宣称已经形成决赛级新算法。当前策略缺少真实 slack 估计、系统压力反馈和动态内存预算，KV 方向也仍以离线估算为主。

后续主线是语义与压力双反馈控制器：联合 route score、层访问期限、历史复用、cgroup `memory.current/high`、PSI、refault、swap 和队列积压，动态选择 `prefetch/keep/cold/pageout/skip`。在此基础上加入可取消、可批量合并的 slack 驱动预取，使创新落在闭环控制和在线成本决策，而不是简单复现已有预取方法。

## 测试结论状态

初赛 N=3 数据来自探索性运行，缓存状态、统计窗口和固定运行顺序不足以支持最终优劣结论，因此仅作为研发过程记录。完成新的 Linux 受控矩阵并人工复核前，不预设 `deadline_score` 或其他策略必然优于 baseline。

完整构建、运行和验证命令见 [reproduce.md](reproduce.md)，测试状态见 [test-report.md](test-report.md)。

## 开源协议说明

本队新增代码和文档采用 Apache License 2.0。第三方 `llama.cpp` 及 vendor 组件保留原始许可证，来源见 [source-attribution.md](source-attribution.md)。
