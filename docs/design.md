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
- score、deadline、deadline_score、stage_deadline_score 优先级；
- `MADV_COLD`、显式 `MADV_DONTNEED` 或 `MADV_PAGEOUT` 回收提示。

### Expert Prefetch 任务生命周期

每个 detail 任务使用进程内单调递增的 `task_id`；跨运行分析必须使用 `(run_id, task_id)`，不把地址或业务字段当作全局 ID。状态机只描述既有调度路径：

```text
NEW -> CREATE -> CREATED
CREATED -> ADMIT -> ADMITTED -> ISSUE -> ISSUED
CREATED -> REJECT -> REJECTED
ADMITTED -> ENQUEUE -> ENQUEUED -> DEQUEUE -> DEQUEUED
DEQUEUED -> ISSUE -> ISSUED
ADMITTED/DEQUEUED -> CANCEL -> CANCELLED
```

`EXPERT_TASK` 记录 `step/layer/expert/tensor/addr/nbytes/score`、唯一分类器产生的 `stage`，以及 created/enqueued/dequeued/issued/returned 时间戳。分类规则固定为 gate/up/gate-up=`EARLY`、down=`LATE`、其余 Expert Tensor=`UNKNOWN`。Task CREATE 保存该结果，Issue matcher 传递该结果，logical first-use 也调用同一个 C++ 分类函数；离线分析只读取 trace 中的 `stage`，不按 tensor 名另行推断。多个原始任务合并成一次系统调用时共享一个 `issue_id`，并用 `issue_task_count` 显式表达一对多关系。`EXPERT_TASK_SUMMARY` 聚合状态、阶段、取消原因、per-stage enqueue/issue/deadline-late count、queue wait（含每阶段 `max_ns`）及 same/cross-stage issue group；`off|summary|detail` 三种模式分别用于无任务统计、低开销聚合和完整 evidence。这里的 late 指非零估计 deadline 存在且 ISSUE 时间不早于该 deadline，仍不表示物理加载完成。

`EXPERT_FIRST_USE` 的含义严格限定为模型计算图中的逻辑首次使用。匹配键为 `(step, layer, expert, tensor)`，同时要求 stage 相同、地址区间重叠且 `first_use_ts_ns >= issued_ts_ns`。多 Token ubatch 可能为同一键创建重复 Task；一次 logical first-use 会关联全部满足条件的重复 Task，并为每个 Task 输出独立的 `create_to_first_use_ns`、`issue_to_first_use_ns` 和 `queue_wait_ns`。同一次观测关联多个 Task 计为一次 `ambiguous_matches`，后续重复 first-use 计入 `duplicate_first_use_ignored`。`madvise` 返回只表示 hint 调用已经返回，不代表物理页已换入，也不代表物理加载完成。

离线 Stage 时序分析按 `(run_id, step, layer, expert)` 聚合 trace 提供的 first-use stage。每个 stage 有多条 logical first-use 时使用最早时间戳，并保留多观测诊断；输出总体、PREFILL、DECODE 和逐 Layer 的 EARLY/LATE 配对、先后计数、比率与有符号 `delta_ns` 分位数。该分析只报告观测顺序，不把 EARLY 当作必然早于 LATE。

Stage 分类本身不参与 Task Admission、coalescing、Slack、Pressure Control、worker 数量或预取范围；只有显式选择 `stage_deadline_score` 时，它才参与队列优先级，其他 priority mode 行为保持不变。

### M2.5 离线 Stage Scheduling Opportunity Analysis

M2.5 阶段本身只增加离线分析；detail Task 额外落盘现有队列的 `sequence` 与现有任务的 `deadline_ts_ns`，使模拟不必用文件顺序或零 deadline 代替真实字段。后续运行时实现以独立、默认关闭的 `stage_deadline_score` mode 落地，不改变原有三种 mode。

`no_issued_task` 原因采用显式证据白名单：只在同一 `(run_id, step, layer, expert, tensor)` 上看到明确 TTL duplicate、cache hit、policy reject/skip、queue/worker failure 或终止时仍 pending 的 Task 时才分类；其余全部保留 `other`。输出同时按 phase、stage 和 phase-stage 交叉聚合，不根据缺失事件猜测原因。

Stage inversion 的计数单位是 LATE Task。对每个 LATE Task 取首个 DEQUEUE/ISSUE 事件；若同一 run、step、layer 存在已 ENQUEUE 且尚未 DEQUEUE/ISSUE 的 EARLY Task，则该 LATE Task 计为一次 inversion。`blocked_early_tasks` 是 LATE/EARLY 阻塞关系数，另输出去重后的 `unique_blocked_early_tasks`；阻塞时长为该 LATE 事件到对应 EARLY 首次 DEQUEUE/ISSUE 的差值。相同纳秒时间戳按 trace 事件顺序判定。

离线模拟只纳入具有真实 ENQUEUE、ISSUE 和 RETURN 的 Task，并固定使用观测到的 ISSUE→RETURN service duration。A 策略复现现有 `deadline_score`：非零 deadline 早、score 高、sequence 小；B 策略 `stage_deadline_score` 对已知阶段按 step、layer、EARLY 优先于 LATE、score、sequence 排序，因此下一层 EARLY 不会越过当前层 LATE。UNKNOWN 进入独立 legacy heap，彼此排序及与已知阶段堆头的仲裁均使用 `deadline_score`，不把 UNKNOWN 降为最低级或跳过。模拟分别使用 1/2/4 workers；on-time 严格表示 ISSUE 时间早于 logical first-use。模拟只报告潜在 ISSUE 时间变化和 LATE 变晚风险，不表示物理页驻留完成，也不支持性能提升或 major fault 下降结论。

### 异步 priority queue

推理线程只构造 `ExpertHintTask` 并入队，worker 执行实际系统调用。控制器使用 layer 执行时间 EWMA 估计 expert 的使用期限，并将任务按真实 `deadline_ts_ns` 和 route score 排序。worker 出队时重新计算预计服务时间；已经错过期限的任务直接取消，不回退到 decode 同步系统调用。

队列支持短等待窗口内最多 8 项的 micro-batch，并尝试合并同一 tensor、layer、step 中相邻的地址区间。trace 记录 batch 数、输入候选、最终系统调用、合并节省量、队列字节高水位以及 deadline/pressure/value/queue-full 四类取消原因。

`stage_deadline_score` 使用两个 heap：EARLY/LATE 的已知阶段 heap 执行固定五级顺序，UNKNOWN legacy heap 沿用 `deadline_score`，两个堆头也用 legacy 规则仲裁。这样避免混合 comparator 的非传递关系。该 mode 即使关闭 Slack Admission，也会生成 deadline 作为 UNKNOWN 仲裁和 late telemetry；独立 A/B 另为两组设置 `LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE=1`，使 legacy 组拥有同口径 deadline。只有 `LLM_MEM_TRACE_OPT_EXPERT_SLACK=1` 才允许 deadline-missed 取消。固定规则下，同一 step/layer 持续到达的 EARLY 理论上可延迟 LATE；更晚 step/layer 不会越过更早 LATE。本版不加入会改变固定顺序的 aging，风险通过 `late_count_by_stage` 和 `queue_wait_ns_by_stage.<stage>.max_ns` 观测。

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

`stage_scheduling_analysis.py` 从 detail trace 生成 `stage_scheduling_opportunity.json`，包括 no-issued 原因、总体/phase/Layer inversion，以及 `deadline_score` 与 `stage_deadline_score` 的 1/2/4-worker 对照。它不调用或修改 C++ queue。

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
