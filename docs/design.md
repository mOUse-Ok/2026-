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

推理线程只构造 `ExpertHintTask` 并入队，worker 执行实际系统调用。控制器使用 layer 执行时间 EWMA 估计 expert 的使用期限，并将任务按真实 `deadline_ts_ns` 和 route score 排序。worker 出队时重新计算预计服务时间；已经错过期限的任务直接取消，不回退到 decode 同步系统调用。

队列支持短等待窗口内最多 8 项的 micro-batch，并尝试合并同一 tensor、layer、step 中相邻的地址区间。trace 记录 batch 数、输入候选、最终系统调用、合并节省量、队列字节高水位以及 deadline/pressure/value/queue-full 四类取消原因。

### 语义与压力双反馈控制器

设置 `LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=feedback_slack` 后，控制器联合两类信号：

- 模型语义：实际 routed expert、router score、当前 layer、phase 和在线 layer 时间 EWMA；真实 routed expert 的使用置信度为 1，router score 只参与优先级。
- 系统反馈：cgroup v2 的 `memory.current/high/max`、`memory.swap.current`、PSI some/full 和 `memory.stat` workingset refault 增量，以及异步队列积压字节。

压力被分成 low、moderate、high、critical 四级，并把基础 expert 预算动态缩放为 100%、75%、50%、20%。每个任务计算预计 page-in/系统调用成本、可隐藏收益和 value ratio；提交前和出队时各检查一次，因压力或成本变化而失去价值的任务可被取消。当前 page-in 时间使用可配置带宽模型，系统调用时间由运行时 EWMA 更新，属于可验证的一阶在线模型，不宣称是精确 I/O 完成时间。

### 成本门控的相邻层 expert 预测

`feedback_slack_predict` 在上述控制器上增加轻量在线预测。预测器按 token 观察相邻 MoE 层的 routed expert 集合，维护有界的 `(source_layer, source_expert) -> destination_expert` 重频统计；达到最小样本数后，为下一层产生 top-2 候选。目标集合超过容量时采用最小计数替换，避免表无限增长或永久保留最早出现的 expert。

预测本身不直接触发 I/O。候选仍需通过 confidence、pressure、slack 和 value ratio 门控，并且只有任务实际入队或执行后才写入 route 去重状态。trace 单独报告 prediction precision、recall、set hit rate、候选数和实际预取/跳过数，防止把“预测正确”误写成“系统优化有效”。

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

项目目前具有以下系统组合与机制创新：

1. 将真实 MoE routing 语义映射到 Linux 页级提示，而不是通用顺序预取。
2. 将静态 deadline 启发式升级为受 cgroup 水位、PSI、refault 和队列积压反馈的动态预算控制器。
3. 在线估计 layer 时间与 hint 服务成本，执行真实 deadline 排序、出队重判和可取消 micro-batch。
4. 使用有界在线转移预测提出下一层候选，但由收益、I/O、内存压力共同决定是否预取。
5. 使用同一套 trace 闭环联合评估延迟、faults、RSS、swap、hint、预测质量和正确性。

前 3 项组合已经形成可运行的自主控制机制，第 4 项是探索性扩展。当前仍不宣称它已经优于旧策略：page-in 估计是一阶模型，预测只覆盖相邻层，KV 方向仍以离线估算为主，而且尚未完成正式 N=8 cgroup 复测。

后续重点是根据正式数据校准压力阈值、page-in 带宽、batch wait 和 value ratio，验证控制器能否同时降低 major faults、RSS/swap 与 hint 数，并将 decode p95 保持在旧 `deadline_score` 的 5% 范围内。预测扩展只有在 precision、实际预取覆盖和端到端指标同时改善时才进入主方案。

## 测试结论状态

初赛 N=3 数据来自探索性运行，缓存状态、统计窗口和固定运行顺序不足以支持最终优劣结论，因此仅作为研发过程记录。完成新的 Linux 受控矩阵并人工复核前，不预设 `deadline_score` 或其他策略必然优于 baseline。

完整构建、运行和验证命令见 [reproduce.md](reproduce.md)，测试状态见 [test-report.md](test-report.md)。

## 开源协议说明

本队新增代码和文档采用 Apache License 2.0。第三方 `llama.cpp` 及 vendor 组件保留原始许可证，来源见 [source-attribution.md](source-attribution.md)。
