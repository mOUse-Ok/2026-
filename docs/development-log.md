# 开发过程记录

> **Historical / Archived — Not Current Mainline**：本文是开发过程的历史记录，按大阶段反映各时期的设计与结论。当前主线（模型映射准入、阶段感知预读、MoE 语义内存对象）以根 README 第 3 节为准；本文提到的 `feedback_slack`、`stage_deadline_score`、连续 aging 等均已退出主线，仅作历史理解。

## 说明

本文档按大阶段记录初赛与后续收敛过程中的主要开发工作、关键问题、解决方法和阶段性结论。更细粒度的本地实验流水记录保存在 `llama.cpp/trace_output/contest_runs/progress_log.md`，该路径用于本地实验记录，不提交仓库。

## 大阶段一：观测链路建立与 OS hint 原型探索

### 目标

建立可运行的 LLM 推理访存分析链路，观察 tensor load、KV cache、expert routing、RSS、swap 和 page faults，并试验各类 OS 内存提示。

### 主要工作

- 在 `llama.cpp/trace/` 扩展 trace sink；`run_trace_pipeline.sh` 自动运行推理并收集 JSONL trace；`analyze_trace.py` 聚合关键指标。
- 实现默认关闭的 OS hint 实验路径：`madvise(MADV_WILLNEED / MADV_SEQUENTIAL)`、`posix_fadvise(POSIX_FADV_WILLNEED / POSIX_FADV_SEQUENTIAL)` 与 expert-aware prefetch，全部通过环境变量开启，不改变默认推理行为。
- trace-driven 离线模拟 `lru` / `lfu` / `window_lfu` / `least_stale` 等替换策略（128–1024 MiB 预算），评估有限 cache budget 下替代 route prefetch 的可行性（含修正 `least_stale` heap 排序方向错误）。
- 测试 route top-k 截断与 slice coalescing 以降低 syscall 数量。
- 将同步 hint 调用改为用户态异步队列（`EXPERT_ASYNC_SUMMARY` 记录 enqueue / issued / queue depth）。
- 尝试 deadline-aware priority（`score` / `deadline` / `deadline_score`）与 route hint TTL；引入 `summarize_repeat_runs.py` 与 N 次重复矩阵。

### 阶段性结论

- baseline 稳定产出 tensor / KV / expert / memory trace，为后续实验提供对照。
- 早期样本曾显示 expert prefetch 可能降低 major faults 和 decode latency，但受实验条件影响，未直接作为性能结论。
- 负结果：朴素 LRU/LFU 在 ≤1 GiB 预算下 miss/eviction 过高；top-k 显著破坏 prefetch coverage；coalescing 受限于 expert slice 地址连续性——简单截断不可行。
- 异步化能消除 syscall 对 decode 路径的直接影响，但 FIFO 调度不足；`decode_ttl1` 减少 hint calls 但延迟与 RSS 无稳定优势。

## 大阶段二：主动控制探索与回退

### 问题

静态排序不知道系统是否正在 refault，也不能判断任务出队时是否已错过使用期限；简单跨层预测可能增加误取页面和 RSS。

### 主要工作

- 读取 cgroup v2 memory current/high/max、swap、PSI 和 workingset refault 增量，将压力分为四级并动态缩放 expert 预算。
- 用每层执行时间和 hint 系统调用时间 EWMA 估计 slack，priority heap 按真实 deadline 排序；worker 出队时复查 deadline、压力和 value ratio，支持取消。
- 增加有 100 µs 等待上限的 micro-batch 与相邻区间合并；按 token 在线学习相邻层 expert 转移，生成带最小样本和置信度要求的 top-2 预测候选。

### 阶段性结论

- `feedback_slack_predict` 短 smoke 中预测 precision 60.13%、set hit rate 80.13%；控制器共执行 636 次 hint，其中 72 次来自跨层预测，按 slack 取消 24 项——控制链生效，但不构成性能改善证据。
- 该方向整体作为历史探索回退：连续 aging、reserved service、复杂 pressure/shadow、Stage priority 等支线不再进入主线，可执行 controller 收敛为 `off` 与 `expert_prefetch`。这次清理降低了文档和代码把实验性想法误写成稳定能力的风险。

## 大阶段三：实验方法论加固与语义内存对象

### 问题

单次运行是否可比较缺少强制证据；hint 之外还需要把“哪些语义对象正在被使用”显式建模。

### 主要工作

- 可信基准修订：以 `process_ubatch()` 为权威范围增加 `STEP_BEGIN/STEP_END`；每个 trace sink 增加 `enqueued/written/dropped` 计数（正式证据要求零丢失）；增加 `evidence` 与 `benchmark` profile；文件级 `POSIX_FADV_DONTNEED` 冷缓存准备（失败时拒绝运行）；GNU time 采集全进程指标；每次运行生成 Manifest 与输出哈希；正式矩阵四方案位置轮换，聚合器拒绝脏仓库、缺失产物与输出不一致的样本。
- Memory Object：以 `(layer, expert, tensor)` 为核心的状态追踪，含 Working Set、eviction/probation、slot、stale demand 等语义记录，并配定向单元测试。
- Calibration Shadow（默认关闭）：记录 prefetch、fault 和 COLD candidate 的环境尺度。
- Residency Attribution：`RESIDENCY_DEMAND` 事件，按 Routed Expert / Shared Expert / Attention / Embedding / Norm 等对象类别统计 resident/nonresident bytes；`TRACE_PROFILE=attribution` 支持 Unlimited 与 `MemoryMax=7G` 的对象级观察。

### 阶段性结论

- 旧 N=3 数据仅用于候选筛选；正式结论必须来自 clean commit、可验证冷缓存、零丢失 trace、固定 cgroup 条件和位置轮换重复矩阵。
- Memory Object 首先验证状态闭环和观测字段（demand / first-use / lifecycle），不直接宣称性能收益；Residency Attribution 用于解释“哪些语义对象在使用前更可能不驻留”，不是 Major Fault 的因果归因。

## 大阶段四：三条主线机制成型

在上述观测链路、实验方法论与语义对象的基础上，系统沿“文件预读原语 → 阶段感知切换”和“cgroup/冷缓存方法论 → 资源感知准入”两条演化线，形成最终三条主线机制（正式口径与数字见根 README 第 3 节）。

### 1. 资源约束感知的模型映射准入（`LLAMA_MMAP_POPULATE_POLICY`）

- 动机：`MAP_POPULATE` 预填充是全有或全无；模型接近或超过 cgroup 限额时，强制预填充会触发大量 reclaim/refault 循环。
- 机制：根据模型 mmap 大小、Sparse MoE 结构（活跃 Expert 占比）、cgroup 限额与 `MemAvailable` 计算的 fit ratio / headroom，在 `DEFAULT / POPULATE / SKIP / AUTO` 中决策 mmap 策略。
- 实验：M2 压力实验（3 MemoryMax × 3 policy × N=3 = 27 runs，Latin square 交错，冷缓存准备，输出 SHA 全部一致、无 OOM）。12 GiB 下 `AUTO` 相比强制 `POPULATE`：wall −34.1%、major faults −21.7%、decode p95 −11.1%、TPS +13.95%。边界：不声称避免 OOM、不声称全局最优、不声称跨模型泛化。正式证据见 `experiments/report_result/07_m2_mmap_pressure_result.md`。

### 2. 推理阶段感知的文件预读控制（`LLAMA_MMAP_DECODE_NORMAL`）

- 动机：`POSIX_FADV_SEQUENTIAL` 对顺序加载友好，但 Decode 阶段 MoE 的 expert 访问由 Router 决定，预读窗口会造成跨 Expert 邻页过取。
- 机制：mmap 时施加 `POSIX_FADV_SEQUENTIAL`，在首次 Decode 前切换回 `POSIX_FADV_NORMAL`。
- 实验：严格 A/B（20 GiB cgroup、skip-populate cold-page、N=3 interleaved、单一变量）。未路由相邻 Expert 新增驻留页 2246.7 → 965.0（−57.0%）；Router 选中 Expert 5265 → 5265 保持一致；端到端性能基本持平。准确表述是**减少跨 Expert 邻页过取**，不是端到端性能提升。正式证据见 `experiments/report_result/09_decode_normal_ab_result.md`。

### 3. MoE Expert 语义内存对象管理

- 映射链：`Router` → `(layer, expert)` → `ExpertTensorRegistry` → `(layer, expert, tensor)` → mmap 虚拟地址区间 → Expert Memory Object。
- 状态闭环证据：16 GiB 档 semantic demands 102,186 / unmatched 0 / invariant violation 0；demand / first-use / lifecycle 全部闭合。

### Router Prefetch 压力窗口（行为边界，非性能贡献）

- 16–19 GiB transient sweep：19G Moderate 发出 5,067 hint、18G/17G High 分别发出 102,172 / 102,177、16G Critical 全部抑制（0 hint），四档均无 reclaim；16G 为 value-gate 全抑制档位。
- 12 GiB server tight-memory anchor：真实 reclaim（pgscan 7,307,338 / pgsteal 2,703,337 / events.max 81,142）下 created 20,184、rejected_pressure 2,181、rejected_value 0、issued 18,003，无 OOM，输出一致。
- 两组运行模式、token 数、trace profile、provenance 不同，不可直接横向比较，均不作为 Prefetch 性能收益证据。正式口径：Router 提供 Expert 需求，系统结合内存压力、任务价值和需求时效决定是否执行 OS Hint。证据见 `experiments/report_result/08_prefetch_pressure_window_result.md` 与 `10_prefetch_12g_reclaim_anchor.md`。

### Working Set / DONTNEED 边界结果

Working Set wall 约 +92.9%、DONTNEED 约 +41.6%；结论为“逻辑 Working Set budget 不等同于物理 Page Cache residency”，作为负结果/边界保留。

## 大阶段五：文档与证据整理（收尾）

- 把“可运行机制”“历史结果”“负结果”和“尚未证明的性能收益”分开，将 README 重组为三项核心贡献，避免用早期 N=3 数字代表当前系统。
- 统一测试口径（8/8 CTest、14/14 Python Trace Tests），修正 M2 RESULT.md 中与冻结 CSV 不符的陈旧数字，补全 Related Work 出处。
- 将四份正式证据（07–10）入库 `experiments/report_result/`，原始大体积实验载荷按仓库约定不入库。

## 当前状态

- 主线：①资源约束感知的模型映射准入；②推理阶段感知的文件预读控制；③MoE 语义内存对象管理（正式口径见根 README 第 3 节）。
- Controller 现状：可执行 controller 收敛为 `off` 与 `expert_prefetch`；Router Prefetch 不作性能收益声明。
- 研究机制（默认关闭或单独运行）：Memory Object、Working Set、Calibration Shadow、`MADV_COLD`、Residency Attribution。
- 历史归档：通用 expert cache、Stage priority、slack/pressure 主动控制、跨层预测、continuous aging、reserved service。
