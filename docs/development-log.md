# 开发过程记录

> **Historical / Archived — Not Current Mainline**：本文是开发过程的历史记录，反映各阶段当时的设计与结论。当前冻结主线（模型映射准入、阶段感知预读、MoE 语义内存对象）以根 README 第 3 节为准；本文提到的 `feedback_slack`、`stage_deadline_score`、连续 aging 等均已退出主线，仅作历史理解。

## 说明

本文档记录初赛阶段以及后续收敛阶段的主要开发过程、关键问题、解决方法和阶段性结论。更细粒度的本地实验流水记录保存在 `llama.cpp/trace_output/contest_runs/progress_log.md`，该路径用于本地实验记录，不提交仓库。

## 阶段 1：建立 trace 与 baseline

### 目标

建立可运行的 LLM 推理访存分析链路，观察 tensor load、KV cache、expert routing、RSS、swap 和 page faults。

### 处理

- 在 `llama.cpp/trace/` 中扩展 trace sink。
- 使用 `run_trace_pipeline.sh` 自动运行推理、收集 JSONL trace、生成分析结果。
- 使用 `analyze_trace.py` 聚合关键指标。

### 结果

baseline 能稳定产出 tensor、KV、expert、memory trace，为后续 OS hint 实验提供对照。

## 阶段 2：安全 OS hint 原型

### 问题

LLM 推理首次访问权重和 MoE expert slice 时存在 major faults 集中爆发，decode latency 受到影响。

### 解决方法

实现默认关闭的 OS hint 实验路径：

- `madvise(MADV_WILLNEED)`
- `madvise(MADV_SEQUENTIAL)`
- `posix_fadvise(POSIX_FADV_WILLNEED)`
- expert-aware prefetch

所有策略通过环境变量开启，避免改变默认推理行为。

### 结论

早期样本曾显示 expert-aware prefetch 可能降低 major faults 和 decode latency，但同时提高 RSS 并产生大量 hint calls。该观察受实验条件影响，后续没有直接作为当前性能结论。

## 阶段 3：Expert cache 替换策略模拟

### 问题

需要判断 LRU/LFU/window-LFU/least-stale 是否能在有限 cache budget 下替代 route prefetch。

### 解决方法

实现 trace-driven 离线模拟，比较：

- `route`
- `lru`
- `lfu`
- `window_lfu`
- `least_stale`

并测试 128/256/512/768/1024 MiB 预算。

### 遇到的问题

heap 化 `least_stale` eviction 时初版排序方向错误，可能淘汰更早复用的 item。修正后改为优先淘汰预计更晚复用的 item。

### 结论

在当前 trace 下，朴素 LRU/LFU 类策略在 <=1 GiB 预算下 miss 和 eviction 过高，不适合作为真实运行主候选。

## 阶段 4：Route top-k 和 coalescing

### 问题

完整 route prefetch 保留 coverage，但 hint calls 太高。需要减少 syscall 数量。

### 解决方法

- 测试 `LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=1/2/4/6`。
- 实现 route slice coalescing。

### 结论

top-k 会显著破坏 prefetch coverage，major faults 回升明显。coalescing 受限于 expert slice 地址连续性，收益有限。因此不能通过简单截断解决问题。

## 阶段 5：异步 expert prefetch

### 问题

同步 hint call 会干扰 decode 关键路径。

### 解决方法

实现用户态异步 hint queue：

- `LLM_MEM_TRACE_OPT_EXPERT_ASYNC`
- `LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE`
- `LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS`

增加 `EXPERT_ASYNC_SUMMARY`，记录 enqueue、issued、fallback、queue depth 等指标。

### 结论

异步化能降低 syscall 对 decode 路径的直接影响，但 FIFO 不够，需要优先级调度。

## 阶段 6：Deadline-aware priority

### 问题

异步队列如果不区分任务紧迫度，可能先处理距离使用点较远的 expert slice。

### 解决方法

实现 priority mode：

- `score`
- `deadline`
- `deadline_score`

其中 `deadline_score` 先按 step/layer 接近程度排序，再按 route score 排序。

### 阶段观察

早期 N=3 重复实验中，`deadline_score` 在当时的采集口径下表现较好，因此被选入正式复测候选。该结论不再作为最终获胜结论。

## 阶段 7：Route TTL 与重复实验

### 问题

需要减少重复 hint，并避免单次运行噪声影响结论。

### 解决方法

- 实现 route hint TTL。
- 将大量 skip 明细改成 `EXPERT_ROUTE_HINT_SUMMARY`。
- 新增 `summarize_repeat_runs.py` 聚合 N 次运行。
- 新增 `run_finalist_repeat_matrix.sh` 固化最终矩阵。

### 阶段观察

`decode_ttl1` 在早期数据中减少了 hint calls，但延迟和 RSS 未显示稳定优势。N=3 数据保留为研发记录，后续性能结论改用更严格的受控重复实验。

## 阶段 8：可信基准修订

### 问题

复核旧实验后发现，单次运行是否可比较缺少强制证据：推理阶段计时边界、文件缓存状态、trace 丢失、全进程 faults、输入输出一致性和运行顺序都可能影响结论。

### 解决方法

- 以一次 `process_ubatch()` 为权威范围增加 `STEP_BEGIN/STEP_END`，旧 `TOKEN_END` 仅保留兼容。
- 为每个 trace sink 增加 `enqueued/written/dropped` 计数；正式证据要求零丢失。
- 增加 `evidence` 与 `benchmark` profile，区分完整观测和低开销性能测试。
- 使用文件级 `POSIX_FADV_DONTNEED` 准备冷缓存，失败时拒绝运行。
- 使用 GNU time 采集全进程 wall time、峰值 RSS、major/minor faults 和文件 I/O。
- 每次运行生成 Manifest、缓存准备结果、进程指标、trace summary、输出哈希和分析指标。
- 正式矩阵采用四方案位置轮换，默认 N=8；聚合器拒绝脏仓库、缺失产物、条件不一致和输出不一致的样本。
- 修正 Pareto 缺失值处理，删除脚本自动生成的固定“最佳策略”结论。

### 阶段结论

旧 N=3 数据可用于候选筛选，但证据强度不足以支撑最终排名。下一次正式结论必须来自 clean commit、可验证冷缓存、零丢失 trace、固定 cgroup 条件和 N=8 位置轮换矩阵。

## 阶段 9（历史探索）：双反馈、slack 取消与成本门控预测

### 问题

旧 `deadline_score` 只按 step/layer 和 router score 静态排序，不知道系统是否正在 refault，也不能判断任务出队时是否已经错过使用期限。简单跨层预测还可能增加误取页面和 RSS。

### 解决方法

- 读取 cgroup v2 memory current/high/max、swap、PSI 和 workingset refault 增量，将压力分为四级并动态缩放 expert 预算。
- 使用每层执行时间和 hint 系统调用时间 EWMA 估计 slack；priority heap 按真实 deadline 排序。
- worker 出队时重新检查 deadline、压力和 value ratio，支持取消，不回退到推理线程同步调用。
- 增加有 100 us 等待上限的 micro-batch 和相邻区间合并。
- 按 token 在线学习相邻层 expert 转移，生成有最小样本和置信度要求的 top-2 候选。
- 预测候选继续接受成本和压力门控，单独报告 precision/recall 与实际 predicted prefetch 数。

### 功能观察

`feedback_slack_predict` 短 smoke 中预测 precision 为 60.13%，set hit rate 为 80.13%。当时 WSL 根 cgroup 呈高 PSI/refault，控制器共执行 636 次 hint，其中 72 次来自跨层预测，并按 slack 取消 24 项。该结果说明控制链生效，但不构成性能改善证据。

## 阶段 10：控制器收敛与实验路径清理（2026-08-03～08-07）

### 处理

- 对专家预取任务、Router 读取同步和最大等待保护进行了最后一轮检查。
- 回退连续 aging、reserved service、复杂 pressure/shadow 和 Stage priority 等支线。
- 删除不再进入当前主线的实验脚本和测试，并将本地 `experiments/` 归档目录从版本控制中排除。

### 阶段结论

当前可执行 pipeline 收敛为 `off` 与 `expert_prefetch` 两个 controller；旧的
`feedback_slack`、`feedback_slack_predict`、`stage_deadline_score` 等名称只保留在历史报告中。
这次清理降低了文档和代码把实验性想法误写成稳定能力的风险。

## 阶段 11：Memory Object 与 Calibration Shadow（2026-08-08）

### 处理

- 增加以 `(layer, expert, tensor)` 为核心的 Memory Object 状态追踪。
- 增加 Working Set、eviction/probation、slot 和 stale demand 等语义记录。
- 增加默认关闭的 Calibration Shadow，记录 prefetch、fault 和 COLD candidate 的环境尺度。
- 为 Memory Object 和 Calibration Shadow 增加定向单元测试。

### 阶段结论

这部分首先验证状态闭环和观测字段，不直接宣称性能收益；后续如果要形成性能结论，仍需单独的受控模型运行。

## 阶段 12：README 与证据整理（2026-08-10）

根据当前 HEAD 的代码和已有实验，对 README、图表、证据边界和技术考古材料进行整理。主要调整是把“可运行机制”“历史结果”“负结果”和“尚未证明的性能收益”分开，避免用早期 N=3 数字代表当前系统。

## 阶段 13：对象级 Residency Attribution（2026-08-12）

### 处理

- 在语义 tensor demand 前增加 `RESIDENCY_DEMAND` 事件。
- 按 Routed Expert、Shared Expert、Attention、Embedding、Norm 等对象类别统计 resident/nonresident bytes 和页级信息。
- 增加 `analyze_residency_attribution.py`、`summarize_experiment_4b.py` 和 `run_experiment_4b.sh`。
- 增加 `TRACE_PROFILE=attribution`，支持 Unlimited 与 `MemoryMax=7G` 的对象级观察实验。

### 阶段结论

Residency Attribution 用于解释“哪些语义对象在使用前更可能不驻留”，不是 Major Fault 的因果归因，也不是性能提升证明。当前仍以 N=1/小规模观察和脚本链路为主，正式结论需要后续人工整理。

## 当前状态

- 当前主线：`off`、`expert_prefetch`、trace 完整性校验、异步 hint 和 deadline-score。
- 当前实验性扩展：Memory Object、Working Set、Calibration Shadow、`MADV_COLD` 和 Residency Attribution，默认关闭或单独运行。
- 历史归档：通用 expert cache、Stage priority、slack/pressure 主动控制、跨层预测、continuous aging 和 reserved service。
- 当前更可信的项目定位是语义内存观测与专家 hint 实验平台，尚未证明跨环境稳定的端到端加速。

## 后续计划

- 在当前 HEAD 上完成最小 `off`/`expert_prefetch` 受控矩阵，确认输出、trace 和指标口径一致。
- 单独复核 Memory Object、Calibration Shadow 和 Residency Attribution 的开关影响，避免把观察成本混入性能结果。
- 根据实验数据补充不同内存上限、输入长度和模型条件；没有足够证据时保持结论为阶段性观察。
